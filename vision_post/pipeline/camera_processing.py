# pipeline/camera_processing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.camera_config import CameraConfig
from config.app_config import AppConfig

from nt_utils.pose2d_reader import Pose2d
from nt_utils.photon_nt_multicam import PhotonMultiCamClient

from geometry_utils.distance_utils import distance_from_pitch
from geometry_utils.pose_utils import camera_pose2d_calculate
from geometry_utils.ballpose_utils import ball_xy_from_camera


@dataclass
class BallObservation:
    """
    單一顆球的標準化結果。

    這層先把各種 target dict 整理成明確欄位，
    後面做去重、分群、選最佳球堆時會更好用。
    """
    camera_name: str
    target_index: int

    # 原始偵測值
    yaw_deg: float
    pitch_deg: float
    area: float
    confidence: Optional[float]

    # 幾何換算結果
    distance_m: float
    ball_x: float
    ball_y: float

    # 角點資料（之後若要用 corner 做優化，這裡直接保留）
    min_rect_corners: List[Tuple[float, float]]
    detected_corners: List[Tuple[float, float]]

    # 原始 target，方便 debug / 後續擴充
    raw_target: Dict[str, Any]


def process_camera(
    pv: PhotonMultiCamClient,
    robot_pose: Optional[Pose2d],
    cam_cfg: CameraConfig,
    pipe_cfg: AppConfig,
) -> List[BallObservation]:
    """
    處理單台相機的完整流程：
    1. 取該相機最新 targets
    2. 算相機在場地上的 Pose2d
    3. 逐顆 target 用 pitch 算距離
    4. 用 yaw + distance 投影成球座標
    5. 回傳有效球列表
    """

    observations: List[BallObservation] = []

    # 相機未啟用 -> 直接不處理
    if not cam_cfg.enabled:
        return observations

    # 沒有 robot pose -> 無法轉成場地絕對座標
    if robot_pose is None:
        return observations

    state = pv.get_state(cam_cfg.name)

    # 若 NT / decode 有錯，可視需求在這裡決定要不要印 log
    if state.last_error is not None:
        return observations

    targets = state.targets
    if not targets:
        return observations

    # 同一輪中，這台相機的場地絕對 Pose 只要算一次
    camera_pose = camera_pose2d_calculate(
        robot_pose=robot_pose,
        forward_m=cam_cfg.forward_m,
        left_m=cam_cfg.left_m,
        camera_yaw_offset_deg=cam_cfg.camera_yaw_offset_deg,
    )

    for i, target in enumerate(targets):
        yaw = target.get("yaw")
        pitch = target.get("pitch")

        # 缺核心欄位就跳過
        if yaw is None or pitch is None:
            continue

        yaw_deg = float(yaw)
        pitch_deg = float(pitch)
        area = float(target.get("area", 0.0))

        conf_raw = target.get("objDetectConf")
        confidence = float(conf_raw) if conf_raw is not None else None

        min_rect_corners = list(target.get("minAreaRectCorners", []))
        detected_corners = list(target.get("detectedCorners", []))

        # 1) pitch -> 地面水平距離
        distance_m = distance_from_pitch(
            pitch_deg=pitch_deg,
            camera_height_m=cam_cfg.height_m,
            camera_pitch_deg=cam_cfg.pitch_deg,
            target_height_m=pipe_cfg.target_height_m,
            eps=pipe_cfg.distance_eps,
        )

        if distance_m is None:
            continue

        # 2) 工程上的距離篩選
        if distance_m < cam_cfg.min_distance_m:
            continue
        if distance_m > cam_cfg.max_distance_m:
            continue

        # 3) 相機 pose + yaw + distance -> 球的場地座標
        ball_x, ball_y = ball_xy_from_camera(
            camera_pose=camera_pose,
            yaw_deg=yaw_deg,
            distance_m=distance_m,
            yaw_sign=cam_cfg.yaw_sign,
        )

        observations.append(
            BallObservation(
                camera_name=cam_cfg.name,
                target_index=i,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                area=area,
                confidence=confidence,
                distance_m=distance_m,
                ball_x=ball_x,
                ball_y=ball_y,
                min_rect_corners=min_rect_corners,
                detected_corners=detected_corners,
                raw_target=target,
            )
        )

    return observations


def process_all_cameras(
    pv: PhotonMultiCamClient,
    robot_pose: Optional[Pose2d],
    camera_cfgs: List[CameraConfig],
    pipe_cfg: AppConfig,
) -> List[BallObservation]:
    """
    依序處理多台相機，並把結果合併成單一列表。
    """
    all_observations: List[BallObservation] = []

    for cam_cfg in camera_cfgs:
        all_observations.extend(
            process_camera(
                pv=pv,
                robot_pose=robot_pose,
                cam_cfg=cam_cfg,
                pipe_cfg=pipe_cfg,
            )
        )

    return all_observations