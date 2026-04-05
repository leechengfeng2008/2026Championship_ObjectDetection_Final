# pose_utils.py
from __future__ import annotations

import math

from nt_utils.pose2d_reader import Pose2d


def camera_pose2d_calculate(
    robot_pose: Pose2d,
    forward_m: float,
    left_m: float,
    camera_yaw_offset_deg: float = 0.0,
) -> Pose2d:
    """
    根據機器人的 Pose2d 與相機相對於機器人的安裝位置/角度，
    計算相機在場地座標系下的 Pose2d。

    座標：
    - robot frame:
        forward_m > 0 代表相機在機器人前方
        left_m    > 0 代表相機在機器人左方
    - field frame:
        回傳的是場地上的絕對 x, y, heading

    參數：
        robot_pose:
            機器人在場地上的 Pose2d
            - x
            - y
            - heading_rad

        forward_m:
            相機相對機器人中心，在前後方向的位移（前正後負）

        left_m:
            相機相對機器人中心，在左右方向的位移（左正右負）

        camera_yaw_offset_deg:
            相機相對機器人朝向的偏航角（度）
            - 0   : 相機正朝前
            - +30 : 相機朝機器人左前方 30 度
            - -30 : 相機朝機器人右前方 30 度

    回傳：
        Pose2d:
            相機在場地座標系下的絕對位置與朝向
    """

    theta = robot_pose.heading_rad
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 把「機器人座標系下的相機位移」旋轉到「場地座標系」
    dx = forward_m * cos_t - left_m * sin_t
    dy = forward_m * sin_t + left_m * cos_t

    camera_x = robot_pose.x + dx
    camera_y = robot_pose.y + dy

    # 相機自己的朝向 = 機器人朝向 + 相機安裝偏航角
    camera_heading_rad = robot_pose.heading_rad + math.radians(camera_yaw_offset_deg)

    return Pose2d(
        x=camera_x,
        y=camera_y,
        heading_rad=camera_heading_rad,
    )