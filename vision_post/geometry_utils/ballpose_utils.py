# ballpose_utils.py
from __future__ import annotations

import math
from typing import Tuple


from nt_utils.pose2d_reader import Pose2d


def project_point_from_pose(
    pose: Pose2d,
    bearing_deg: float,
    distance_m: float,
) -> Tuple[float, float]:
    """
    根據一個已知 Pose2d,沿著指定角度前進 distance_m,
    計算目標點在場地座標系下的 (x, y)。

    參數:
        pose:
            起點 Pose2d,通常這裡會是相機的場地絕對 Pose2d

        bearing_deg:
            相對於 pose.heading_rad 的額外夾角（度）
            例如:
            - 0   表示正前方
            - +10 表示往左偏 10 度
            - -10 表示往右偏 10 度

        distance_m:
            從 pose 出發到目標的地面水平距離（公尺）

    回傳:
        (x, y):
            目標點在場地座標系下的絕對座標
    """

    # 目標在場地座標系下的絕對方向角
    global_heading_rad = pose.heading_rad + math.radians(bearing_deg)

    # 用極座標轉平面座標
    x = pose.x + distance_m * math.cos(global_heading_rad)
    y = pose.y + distance_m * math.sin(global_heading_rad)

    return x, y


def ball_xy_from_camera(
    camera_pose: Pose2d,
    yaw_deg: float,
    distance_m: float,
    yaw_sign: float = 1.0,
) -> Tuple[float, float]:
    """
    根據相機 Pose2d、PhotonVision 量到的 yaw、以及球到相機的水平距離,
    換算出球在場地座標系下的 (x, y)。

    參數:
        camera_pose:
            相機在場地上的絕對 Pose2d
            camera_pose.heading_rad 已經包含相機安裝角

        yaw_deg:
            PhotonVision 回傳的 target yaw(度)

        distance_m:
            相機到球的地面水平距離（公尺）

        yaw_sign:
            修正 PhotonVision yaw 正負方向的係數
            - +1 : 保持原本 yaw 定義
            - -1 : 反轉 yaw 正負方向

    回傳:
        (ball_x, ball_y):
            球在場地上的絕對座標
    """

    corrected_yaw_deg = yaw_sign * yaw_deg

    # 由相機 Pose2d + 修正後 yaw + 距離，投影到場地上
    return project_point_from_pose(
        pose=camera_pose,
        bearing_deg=corrected_yaw_deg,
        distance_m=distance_m,
    )