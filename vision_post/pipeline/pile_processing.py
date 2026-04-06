# pipeline/pile_processing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple

from config.app_config import AppConfig
from nt_utils.pose2d_reader import Pose2d

from pile_utils.ballpiles_rect_centers import plan_ballpile_rect_centers
from pile_utils.ballpiles_average_centers import plan_ballpile_centers
from pile_utils.select_best_pile import (
    PileCandidate,
    PileScoreInfo,
    PileSelectionResult,
    build_candidates_from_rect_piles,
    build_candidates_from_center_plans,
    select_best_pile,
)

Point2 = Tuple[float, float]
PileMethod = Literal["rect", "center"]


@dataclass
class PileProcessingResult:
    """
    整包 pile pipeline 的輸出。

    plan_result:
        分堆函式原始輸出（第一版或第二版）
        裡面保留前端畫圖需要的各種 lists

    selection_result:
        選堆輸出，包含最佳堆與所有分數資訊

    best_center_xy:
        方便主程式直接拿最佳堆中心
    """
    method: PileMethod
    plan_result: Any
    selection_result: PileSelectionResult
    best_center_xy: Optional[Point2]


def process_piles(
    ball_points: Sequence[Point2],
    robot_pose: Optional[Pose2d],
    app_cfg: AppConfig,
    method: PileMethod = "rect",

    # 第一版 rect/grid 參數
    rect_cell_size_m: float = 0.50,
    rect_diagonal_connect: bool = True,
    rect_center_mode: str = "density_weighted",
    rect_density_radius_m: float = 0.50,

    # 第二版 single-linkage 參數
    center_cluster_link_m: float = 0.30,
    center_mode: str = "density_vb",
    center_density_radius_m: float = 0.30,
    center_density_spread_limit_m: float = 0.30,
) -> PileProcessingResult:
    """
    統一入口：
    deduped ball points -> 分堆 -> 建 candidate -> 選最佳堆

    參數
    ----
    ball_points:
        去重後球點 [(x, y), ...]

    robot_pose:
        機器人在場地上的 Pose2d，提供 selector 做距離加權
        若為 None，則只依球數選堆

    app_cfg:
        用來提供 pile_ball_priority_0to10

    method:
        "rect"   -> 第一版 grid/rect 分堆
        "center" -> 第二版 single-linkage 分堆
    """
    # 先處理空輸入
    if not ball_points:
        empty_selection = PileSelectionResult(
            best_candidate=None,
            score_infos=[],
        )
        return PileProcessingResult(
            method=method,
            plan_result=None,
            selection_result=empty_selection,
            best_center_xy=None,
        )

    # -----------------------------
    # 第一版：grid / rect
    # -----------------------------
    if method == "rect":
        plan_result = plan_ballpile_rect_centers(
            ball_xys=ball_points,
            cell_size_m=rect_cell_size_m,
            diagonal_connect=rect_diagonal_connect,
            center_mode=rect_center_mode,
            density_radius_m=rect_density_radius_m,
        )

        candidates = build_candidates_from_rect_piles(plan_result.pile_plans)

    # -----------------------------
    # 第二版：single-linkage + density_vb
    # -----------------------------
    elif method == "center":
        plan_result = plan_ballpile_centers(
            ball_xys=ball_points,
            cluster_link_m=center_cluster_link_m,
            center_mode=center_mode,
            density_radius_m=center_density_radius_m,
            density_spread_limit_m=center_density_spread_limit_m,
        )

        candidates = build_candidates_from_center_plans(plan_result.pile_plans)

    else:
        raise ValueError(f"Unknown pile method: {method}")

    # -----------------------------
    # 選最佳堆
    # -----------------------------
    selection_result = select_best_pile(
        robot_pose=robot_pose,
        pile_candidates=candidates,
        ball_priority_0to10=app_cfg.pile_ball_priority_0to10,
    )

    if selection_result.best_candidate is None:
        best_center_xy = None
    else:
        best_center_xy = selection_result.best_candidate.center_xy

    return PileProcessingResult(
        method=method,
        plan_result=plan_result,
        selection_result=selection_result,
        best_center_xy=best_center_xy,
    )