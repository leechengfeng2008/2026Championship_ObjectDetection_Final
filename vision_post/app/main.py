from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

# 讓你可以從 project root 或 vision_post 目錄執行都比較穩
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP, CAMERAS, NETWORK
from dataclasses import dataclass

from nt_utils.photon_nt_multicam import PhotonMultiCamClient
from nt_utils.nt_publish_utils import (
    create_best_relative_pose2d_publisher,
    publish_best_relative_pile,
    close_publisher_instance,
)

from geometry_utils.pose_utils import camera_pose2d_calculate
from pipeline.camera_processing import process_all_cameras
from pipeline.dedupe_processing import dedupe_two_cameras_fov
from pipeline.pile_processing import process_piles


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    heading_rad: float

# 目前 dedupe 只實作到雙相機，啟動時就先確認，避免執行期才炸
_MAX_SUPPORTED_CAMERAS = 2


def _get_enabled_camera_cfgs() -> List:
    """
    根據 NETWORK.cameras 與 camera_config.py，挑出目前要啟用的相機設定。

    【修正】在此提前驗證啟用相機數量，超過上限直接中止，
    而非等到執行期 dedupe 邏輯才拋出 NotImplementedError。
    """
    cfgs = []
    for name in NETWORK.cameras:
        cfg = CAMERAS.get(name)
        if cfg is None:
            continue
        if not cfg.enabled:
            continue
        cfgs.append(cfg)

    if len(cfgs) > _MAX_SUPPORTED_CAMERAS:
        raise ValueError(
            f"[main] 啟用的相機數量（{len(cfgs)}）超過目前支援的上限"
            f"（{_MAX_SUPPORTED_CAMERAS}）。"
            f" 請在 network_config.py 或 camera_config.py 停用多餘的相機。"
            f" 啟用清單：{[c.name for c in cfgs]}"
        )

    return cfgs


def _filter_fresh_camera_cfgs(pv: PhotonMultiCamClient, camera_cfgs: List) -> List:
    """
    根據 APP.stale_timeout_s 過濾相機。
    若某台相機太久沒更新，這輪就先跳過。
    """
    now_mono = time.monotonic()
    fresh_cfgs = []

    for cfg in camera_cfgs:
        state = pv.get_state(cfg.name)

        # 尚未收過資料
        if state.last_update_monotonic <= 0.0:
            continue

        age_s = now_mono - state.last_update_monotonic
        if age_s > APP.stale_timeout_s:
            continue

        if state.last_error is not None:
            continue

        fresh_cfgs.append(cfg)

    return fresh_cfgs


def main() -> None:
    # ------------------------------------------------------------
    # 1. 啟動前驗證相機數量（提早發現設定錯誤）
    # ------------------------------------------------------------
    enabled_camera_cfgs = _get_enabled_camera_cfgs()

    # ------------------------------------------------------------
    # 2. 初始化 NT readers / publishers
    # ------------------------------------------------------------
    pv = PhotonMultiCamClient(
        server=NETWORK.nt_server,
        cameras=NETWORK.cameras,
        client_name=NETWORK.client_name,
        table_name=NETWORK.table_name,
        poll_dt=APP.nt_poll_dt,
        sort_targets_by_area_desc=APP.sort_targets_by_area_desc,
    )
    pv.start()

    publish_inst, best_pose_pub = create_best_relative_pose2d_publisher(
        server=NETWORK.nt_server,
        table="SmartDashboard",
        key="BestPileRelativePose2d",
        client_name="best-pile-relative-publisher",
    )

    robot_pose = RobotPose(x=0.0, y=0.0, heading_rad=0.0)

    if APP.debug:
        print("[main] enabled cameras:", [c.name for c in enabled_camera_cfgs])

    loop_count = 0

    try:
        while True:
            loop_count += 1

            # ----------------------------------------------------
            # 3. 過濾 stale / error cameras
            # ----------------------------------------------------
            fresh_camera_cfgs = _filter_fresh_camera_cfgs(pv, enabled_camera_cfgs)

            if not fresh_camera_cfgs:
                publish_best_relative_pile(best_pose_pub, None, robot_pose)

                if APP.debug and loop_count % APP.print_every_n_loops == 0:
                    print("[main] no fresh cameras available")
                time.sleep(APP.loop_sleep_s)
                continue

            # ----------------------------------------------------
            # 5. 第一條 pipeline：targets -> BallObservation
            # ----------------------------------------------------
            all_observations = process_all_cameras(
                pv=pv,
                robot_pose=robot_pose,
                camera_cfgs=fresh_camera_cfgs,
                app_cfg=APP,
            )

            # 若完全沒有球，清空 publish
            if not all_observations:
                publish_best_relative_pile(best_pose_pub, None, robot_pose)

                if APP.debug and loop_count % APP.print_every_n_loops == 0:
                    print("[main] no observations")
                time.sleep(APP.loop_sleep_s)
                continue

            # ----------------------------------------------------
            # 6. 第二條 pipeline 前段：dedupe
            #    - 1 台相機：直接使用 observation 座標
            #    - 2 台相機：做 FOV-aware dedupe
            # ----------------------------------------------------
            if len(fresh_camera_cfgs) == 1:
                unique_ball_points = [
                    (obs.ball_x, obs.ball_y)
                    for obs in all_observations
                ]

            elif len(fresh_camera_cfgs) == 2:
                cam1_cfg = fresh_camera_cfgs[0]
                cam2_cfg = fresh_camera_cfgs[1]

                camera1_pose = camera_pose2d_calculate(
                    robot_pose=robot_pose,
                    forward_m=cam1_cfg.forward_m,
                    left_m=cam1_cfg.left_m,
                    camera_yaw_offset_deg=cam1_cfg.camera_yaw_offset_deg,
                )
                camera2_pose = camera_pose2d_calculate(
                    robot_pose=robot_pose,
                    forward_m=cam2_cfg.forward_m,
                    left_m=cam2_cfg.left_m,
                    camera_yaw_offset_deg=cam2_cfg.camera_yaw_offset_deg,
                )

                dedupe_result = dedupe_two_cameras_fov(
                    all_observations=all_observations,
                    camera1_name=cam1_cfg.name,
                    camera2_name=cam2_cfg.name,
                    camera1_pose2d=camera1_pose,
                    camera2_pose2d=camera2_pose,
                    same_ball_error_m=APP.dedupe_same_ball_error_m,
                    max_angle_deg=APP.dedupe_max_angle_deg,
                    keep="average",
                )

                unique_ball_points = dedupe_result.unique_points

            else:
                # _get_enabled_camera_cfgs() 已在啟動時擋住這條路，
                # 這裡理論上不會到達，但保留作為安全網。
                raise NotImplementedError(
                    f"Current main.py supports up to {_MAX_SUPPORTED_CAMERAS} cameras for dedupe."
                )

            # ----------------------------------------------------
            # 7. 第二條 pipeline 後段：分堆 -> 選最佳堆
            # ----------------------------------------------------
            pile_result = process_piles(
                ball_points=unique_ball_points,
                robot_pose=robot_pose,
                app_cfg=APP,
                method="rect",   # 你之後若要切第二版可改成 "center"
            )

            best_center_xy = pile_result.best_center_xy
            best_candidate = pile_result.selection_result.best_candidate

            # ----------------------------------------------------
            # 8. 發布最佳球堆
            # ----------------------------------------------------
            publish_best_relative_pile(
                best_pose_pub,
                best_candidate,
                robot_pose,
            )

            # ----------------------------------------------------
            # 9. Debug print
            # ----------------------------------------------------
            if APP.debug and loop_count % APP.print_every_n_loops == 0:
                print("=" * 70)
                print(f"[main] robot_pose = ({robot_pose.x:.3f}, {robot_pose.y:.3f}, "
                      f"{robot_pose.heading_rad:.3f} rad)")
                print(f"[main] fresh cameras = {[c.name for c in fresh_camera_cfgs]}")
                print(f"[main] observations = {len(all_observations)}")
                print(f"[main] unique_ball_points = {len(unique_ball_points)}")
                print(f"[main] pile_method = {pile_result.method}")

                if best_center_xy is None:
                    print("[main] best pile = None")
                else:
                    print(f"[main] best pile center = ({best_center_xy[0]:.3f}, "
                          f"{best_center_xy[1]:.3f})")

                # 額外列出分堆評分前幾名
                score_infos = pile_result.selection_result.score_infos
                for i, s in enumerate(score_infos[:5]):
                    print(
                        f"[score {i}] pile_id={s.pile_id} "
                        f"count={s.count} "
                        f"dist={s.distance_from_robot_m:.3f} "
                        f"near={s.near_score:.3f} "
                        f"count_score={s.count_score:.3f} "
                        f"final={s.final_score:.3f}"
                    )

            time.sleep(APP.loop_sleep_s)

    except KeyboardInterrupt:
        print("\n[main] stopped by user")

    finally:
        try:
            pv.stop()
        except Exception:
            pass

        try:
            close_publisher_instance(publish_inst)
        except Exception:
            pass


if __name__ == "__main__":
    main()