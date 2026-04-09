from __future__ import annotations

"""
nt_publish_utils.py
===================
NetworkTables 發布工具，供 main.py共用。

發布格式（與舊版相容）：
    SmartDashboard/BestPilePose2d  -> DoubleArray [x, y, heading_deg]
    SmartDashboard/BestPileRelativePose2d  -> DoubleArray [timestamp_s, x, y, heading_deg]

規則：
    - 無最佳堆 或 無機器人 pose -> 發布 []
    - heading_deg = 從機器人位置指向球堆中心的場地角度（不是機器人朝向）
"""

import math
import time
from typing import Optional

import ntcore


# ─────────────────────────────────────────────────────────────────────────────
# Publisher 建立
# ─────────────────────────────────────────────────────────────────────────────

def create_best_pose2d_publisher(
    server: str,
    table: str = "SmartDashboard",
    key: str = "BestPilePose2d",
    client_name: str = "best-pile-publisher",
):
    """
    建立並回傳 (inst, publisher)。

    注意：
    - inst 必須由呼叫端保留，否則連線可能中斷
    - pub 是 DoubleArray publisher
    """
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)

    pub = inst.getTable(table).getDoubleArrayTopic(key).publish()
    return inst, pub


def close_publisher_instance(inst) -> None:
    """
    若你之後想在程式結束時明確收尾，可以呼叫此函式。
    """
    if inst is None:
        return
    try:
        inst.stopClient()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 發布
# ─────────────────────────────────────────────────────────────────────────────

def publish_best_pile(
    pub,
    best_pile,       # RectPileInfo | PileCenterInfo | 任何有 .center_xy 的物件 | None
    robot_pose2d,    # Pose2d | 任何有 .x .y .heading_rad 的物件 | None
) -> None:
    """
    把最佳球堆位置發布到 NT。

    格式：[x, y, heading_deg]
      - x, y        : 球堆中心的場地座標（公尺）
      - heading_deg : 從機器人位置指向球堆中心的角度（場地座標系，度）

    無效情況（發布 []）：
      - best_pile 是 None
      - robot_pose2d 是 None（無法算方向角）
      - best_pile 沒有有效 center_xy
    """
    if best_pile is None or robot_pose2d is None:
        pub.set([])
        return

    try:
        x = float(best_pile.center_xy[0])
        y = float(best_pile.center_xy[1])
    except Exception:
        pub.set([])
        return

    dx = x - float(robot_pose2d.x)
    dy = y - float(robot_pose2d.y)

    # 若球堆中心與機器人位置重合，退回機器人目前 heading
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        heading_deg = math.degrees(float(robot_pose2d.heading_rad))
    else:
        heading_deg = math.degrees(math.atan2(dy, dx))

    pub.set([x, y, heading_deg])


def create_best_relative_pose2d_publisher(
    server: str,
    table: str = "SmartDashboard",
    key: str = "BestPileRelativePose2d",
    client_name: str = "best-pile-relative-publisher",
):
    inst = ntcore.NetworkTableInstance.create()
    inst.startClient4(client_name)
    inst.setServer(server)

    pub = inst.getTable(table).getDoubleArrayTopic(key).publish()
    return inst, pub


def publish_best_relative_pile(
    pub,
    best_pile,
    robot_pose2d,
) -> None:
    """
    Publish the best pile in robot-relative coordinates with a timestamp.

    Format: [timestamp_s, x, y, heading_deg]
      - x, y        : pile center in robot frame (m)
      - heading_deg : angle from robot to pile center in robot frame (deg)
    """
    if best_pile is None or robot_pose2d is None:
        pub.set([])
        return

    try:
        x = float(best_pile.center_xy[0])
        y = float(best_pile.center_xy[1])
    except Exception:
        pub.set([])
        return

    dx = x - float(robot_pose2d.x)
    dy = y - float(robot_pose2d.y)

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        heading_deg = math.degrees(float(robot_pose2d.heading_rad))
    else:
        heading_deg = math.degrees(math.atan2(dy, dx))

    timestamp_s = time.time()
    pub.set([timestamp_s, x, y, heading_deg])


def publish_best_center_xy_only(
    pub,
    best_pile,
) -> None:
    """
    若你之後某些前端 / debug 工具只想先看最佳堆中心 (x, y)，
    可以使用這個簡化版。

    格式：[x, y]
    無效時發布 []
    """
    if best_pile is None:
        pub.set([])
        return

    try:
        x = float(best_pile.center_xy[0])
        y = float(best_pile.center_xy[1])
    except Exception:
        pub.set([])
        return

    pub.set([x, y])