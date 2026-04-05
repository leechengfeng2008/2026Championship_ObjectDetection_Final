# distance_utils.py
from __future__ import annotations

import math
from typing import Iterable, List, Optional


def distance_from_pitch(
    pitch_deg: float,
    camera_height_m: float,
    camera_pitch_deg: float,
    target_height_m: float,
    eps: float = 1e-6,
) -> Optional[float]:
    """
    根據目標的 pitch,計算目標到相機的「地面水平距離」。

    參數:
        pitch_deg:
            PhotonVision 量到的單一 target pitch 角度（單位：度）

        camera_height_m:
            相機離地高度（公尺）

        camera_pitch_deg:
            相機安裝俯角（單位：度）
            total_deg = camera_pitch_deg - pitch_deg

        target_height_m:
            目標點離地高度（公尺）
            例如球心高度

        eps:
            避免 tan(total_angle) 太接近 0 時發生數值問題的保護閾值

    回傳:
        - 正常時回傳距離（float, 單位：公尺）
        - 若無法計算或幾何上不合理，回傳 None

    幾何概念:
        delta_height = camera_height_m - target_height_m
        total_angle = camera_pitch_deg - pitch_deg
        distance = delta_height / tan(total_angle)

        計算「地面上的水平距離」，
        非相機到目標的 3D 直線距離。
    """

    # 相機與目標的高度差
    delta_height = camera_height_m - target_height_m

    # 將相機俯角與 target pitch 合成總角度
    total_deg = camera_pitch_deg - float(pitch_deg)
    total_rad = math.radians(total_deg)

    # 計算 tan(total_angle)
    tan_value = math.tan(total_rad)

    # 若 tan 太接近 0，代表角度太接近平行於地面，
    # 此時距離會趨近無限大或數值不穩定，直接視為無效
    if abs(tan_value) < eps:
        return None

    # 幾何公式：水平距離 = 高度差 / tan(總角度)
    distance_m = delta_height / tan_value

    # 若距離 <= 0，通常代表幾何上不符合「前方地面目標」的假設
    # 因此視為無效
    if distance_m <= 0:
        return None

    return distance_m


def distance_calculate(
    pitch_deg_list: Iterable[float],
    camera_height_m: float,
    camera_pitch_deg: float,
    target_height_m: float,
    eps: float = 1e-6,
) -> List[Optional[float]]:
    """
    陣列：把一串 pitch 角度逐一換算成距離。

    而是逐一呼叫底層核心函式 distance_from_pitch()。

    參數:
        pitch_deg_list:
            多個 target 的 pitch 角度

        camera_height_m:
            相機離地高度（公尺）

        camera_pitch_deg:
            相機安裝俯角（度）

        target_height_m:
            目標高度（公尺）

        eps:
            tan 保護閾值

    回傳:
        與輸入 pitch list 對應的距離 list
        每一項可能是 float 或 None
    """

    out: List[Optional[float]] = []

    for pitch_deg in pitch_deg_list:
        distance_m = distance_from_pitch(
            pitch_deg=pitch_deg,
            camera_height_m=camera_height_m,
            camera_pitch_deg=camera_pitch_deg,
            target_height_m=target_height_m,
            eps=eps,
        )
        out.append(distance_m)

    return out