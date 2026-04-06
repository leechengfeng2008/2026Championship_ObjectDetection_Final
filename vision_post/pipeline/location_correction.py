# pipeline/occlusion_correction.py
from __future__ import annotations

from typing import List

from pipeline.camera_processing import BallObservation


def correct_ball_positions(
    observations: List[BallObservation],
) -> List[BallObservation]:
    """
    遮蔽球 / 視覺幾何修正骨架。

    - 根據 detectedCorners 修正球中心
    - 根據 minAreaRectCorners 做遮蔽補償
    - 根據 area / 比例 / 形狀做位置微調

    參數:
        observations:
            camera_processing.py 輸出的 BallObservation list

    回傳:
        目前先直接回傳原 observations
        之後新增 corrected_ball_x / corrected_ball_y
    """
    # TODO:
    # 加入遮蔽球修正邏輯
    # 現階段先原樣回傳
    return observations