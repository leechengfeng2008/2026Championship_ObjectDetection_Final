# pipeline/dedupe_processing.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from nt_utils.pose2d_reader import Pose2d
from pipeline.camera_processing import BallObservation

# 插入遮蔽球位置修正
# from pipeline.occlusion_correction import correct_ball_positions


Point2 = Tuple[float, float]
KeepMode = Literal["average", "cam1", "cam2"]


# -----------------------------
# Public data classes
# -----------------------------
@dataclass
class FovMatchPair:
    """
    一組被視為同一顆球的跨相機配對結果。
    """
    cam1_index: int
    cam2_index: int
    cam1_xy: Point2
    cam2_xy: Point2
    error_m: float
    merged_xy: Point2
    cam1_obs: BallObservation
    cam2_obs: BallObservation


@dataclass
class FovDedupeResult:
    """
    去重後的完整結果。
    """
    unique_points: List[Point2]
    matched_pairs: List[FovMatchPair]
    unmatched_cam1: List[Point2]
    unmatched_cam2: List[Point2]
    angle_rejected_count: int


# -----------------------------
# Internal helpers
# -----------------------------
def _dist(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_angle_deg(deg: float) -> float:
    """
    把角度包回 (-180, +180]。
    """
    deg = deg % 360.0
    if deg > 180.0:
        deg -= 360.0
    return deg


def _obs_xy(obs: BallObservation) -> Point2:
    """
    取得 observation 的球座標。

    目前先用 camera_processing.py 算出的原始球座標。
    之後若你在 BallObservation 加入 corrected_ball_x / corrected_ball_y，
    或要先跑遮蔽球修正，這裡就是最方便切換的地方。
    """
    return float(obs.ball_x), float(obs.ball_y)


def _split_observations_by_camera(
    all_observations: List[BallObservation],
    camera1_name: str,
    camera2_name: str,
) -> Tuple[List[BallObservation], List[BallObservation]]:
    """
    把合併後的 observation list，依相機名稱拆成兩組。
    """
    cam1_obs: List[BallObservation] = []
    cam2_obs: List[BallObservation] = []

    for obs in all_observations:
        if obs.camera_name == camera1_name:
            cam1_obs.append(obs)
        elif obs.camera_name == camera2_name:
            cam2_obs.append(obs)

    return cam1_obs, cam2_obs


def _merge_point(cam1_xy: Point2, cam2_xy: Point2, keep: KeepMode) -> Point2:
    """
    決定配對成功後，最後保留哪個座標。
    """
    if keep == "cam1":
        return cam1_xy
    if keep == "cam2":
        return cam2_xy
    return ((cam1_xy[0] + cam2_xy[0]) / 2.0, (cam1_xy[1] + cam2_xy[1]) / 2.0)


def _bearing_relative_deg(
    camera_pose2d: Pose2d,
    ball_xy: Point2,
) -> float:
    """
    計算球相對於「相機光軸」的水平夾角（度）。

    注意：
    在你現在的新架構裡，camera_pose2d.heading_rad 已經是相機自己的朝向，
    也就是已經包含 camera_yaw_offset_deg，
    因此這裡不能再額外加一次 yaw offset。
    """
    dx = ball_xy[0] - camera_pose2d.x
    dy = ball_xy[1] - camera_pose2d.y

    bearing_world_rad = math.atan2(dy, dx)
    relative_rad = bearing_world_rad - camera_pose2d.heading_rad

    return _normalize_angle_deg(math.degrees(relative_rad))


def _angle_feasible(
    camera_pose2d: Optional[Pose2d],
    ball_xy: Point2,
    max_angle_deg: float,
) -> bool:
    """
    判斷 ball_xy 是否位於某台相機的可視 yaw 視窗內。

    若 camera_pose2d 為 None，則略過角度檢查，直接視為可行。
    """
    if camera_pose2d is None:
        return True

    rel = _bearing_relative_deg(camera_pose2d, ball_xy)
    return abs(rel) <= max_angle_deg


# -----------------------------
# Main function
# -----------------------------
def dedupe_two_cameras_fov(
    all_observations: List[BallObservation],
    camera1_name: str,
    camera2_name: str,
    camera1_pose2d: Optional[Pose2d],
    camera2_pose2d: Optional[Pose2d],
    same_ball_error_m: float = 0.10,
    max_angle_deg: float = 35.0,
    keep: KeepMode = "average",
) -> FovDedupeResult:
    """
    用兩台相機 observation 做 FOV-aware 去重。

    流程：
    1. 依 camera_name 把 observation 拆成 cam1 / cam2
    2. （之後可插入球位置修正）
    3. 先做距離候選篩選
    4. 再做角度可見性檢查
    5. 最後做 greedy nearest-neighbour 配對

    參數：
        all_observations:
            來自 process_all_cameras() 的合併 observation list

        camera1_name, camera2_name:
            兩台相機在 observation 裡使用的名稱，
            例如 "Camera1", "Camera2"

        camera1_pose2d, camera2_pose2d:
            兩台相機在場地座標系下的 Pose2d
            注意：heading_rad 必須已包含相機安裝偏航角

        same_ball_error_m:
            兩顆 observation 被視為同一顆球的距離門檻

        max_angle_deg:
            相機可視 yaw 半窗。
            若球相對於相機光軸的偏角超過這個值，
            就視為幾何上不可能是這台相機看到的球。

        keep:
            配對成功後保留座標的方式：
            - "average" : 兩點平均
            - "cam1"    : 保留 cam1
            - "cam2"    : 保留 cam2
    """

    # ---------------------------------
    # 遮蔽球修正
    # ---------------------------------
    # corrected_observations = correct_ball_positions(all_observations)
    # working_observations = corrected_observations

    # 目前:原 observation
    working_observations = all_observations

    cam1_obs, cam2_obs = _split_observations_by_camera(
        working_observations,
        camera1_name=camera1_name,
        camera2_name=camera2_name,
    )

    cam1_pts = [_obs_xy(obs) for obs in cam1_obs]
    cam2_pts = [_obs_xy(obs) for obs in cam2_obs]

    used_cam2 = [False] * len(cam2_pts)
    matched_pairs: List[FovMatchPair] = []
    unique_points: List[Point2] = []
    unmatched_cam1: List[Point2] = []
    angle_rejected_count = 0

    for i, p1 in enumerate(cam1_pts):
        best_j = -1
        best_err = float("inf")

        for j, p2 in enumerate(cam2_pts):
            if used_cam2[j]:
                continue

            # Step 1: 距離候選篩選
            err = _dist(p1, p2)
            if err > same_ball_error_m:
                continue

            # Step 2: 角度可行性檢查
            cam2_sees_p1 = _angle_feasible(
                camera_pose2d=camera2_pose2d,
                ball_xy=p1,
                max_angle_deg=max_angle_deg,
            )
            cam1_sees_p2 = _angle_feasible(
                camera_pose2d=camera1_pose2d,
                ball_xy=p2,
                max_angle_deg=max_angle_deg,
            )

            if not (cam2_sees_p1 and cam1_sees_p2):
                angle_rejected_count += 1
                continue

            # Step 3: greedy 最近配對
            if err < best_err:
                best_err = err
                best_j = j

        if best_j >= 0:
            used_cam2[best_j] = True
            p2 = cam2_pts[best_j]
            merged = _merge_point(p1, p2, keep=keep)

            matched_pairs.append(
                FovMatchPair(
                    cam1_index=i,
                    cam2_index=best_j,
                    cam1_xy=p1,
                    cam2_xy=p2,
                    error_m=best_err,
                    merged_xy=merged,
                    cam1_obs=cam1_obs[i],
                    cam2_obs=cam2_obs[best_j],
                )
            )
            unique_points.append(merged)
        else:
            unmatched_cam1.append(p1)
            unique_points.append(p1)

    unmatched_cam2: List[Point2] = []
    for j, p2 in enumerate(cam2_pts):
        if not used_cam2[j]:
            unmatched_cam2.append(p2)
            unique_points.append(p2)

    return FovDedupeResult(
        unique_points=unique_points,
        matched_pairs=matched_pairs,
        unmatched_cam1=unmatched_cam1,
        unmatched_cam2=unmatched_cam2,
        angle_rejected_count=angle_rejected_count,
    )