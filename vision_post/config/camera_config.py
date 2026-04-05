# config/camera_config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:

    name: str
    enabled: bool

    # 相機安裝幾何
    height_m: float
    pitch_deg: float
    forward_m: float
    left_m: float
    camera_yaw_offset_deg: float

    # PhotonVision yaw 符號修正
    yaw_sign: int

    # 有效距離範圍
    min_distance_m: float
    max_distance_m: float

CAMERA1 = CameraConfig(
    name="Camera1",
    enabled=True,
    height_m=0.527,
    pitch_deg=21.0,
    forward_m=-0.641,
    left_m=0.246,
    camera_yaw_offset_deg=30.0,
    yaw_sign=1,
    min_distance_m=0.15,
    max_distance_m=5.0,
)

CAMERA2 = CameraConfig(
    name="Camera2",
    enabled=True,
    height_m=0.527,
    pitch_deg=21.0,
    forward_m=-0.641,  
    left_m=-0.246,
    camera_yaw_offset_deg=30.0,
    yaw_sign=-1,
    min_distance_m=0.15,
    max_distance_m=5.0,
)

CAMERAS = {
    CAMERA1.name: CAMERA1,
    CAMERA2.name: CAMERA2,
}