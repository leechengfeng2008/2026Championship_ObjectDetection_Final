from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from nt_utils.pose2d_reader import Pose2d

Point2 = Tuple[float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PileCandidate:
    """
    給 selector 使用的標準候選格式。
    不管你前面是 RectPileInfo 還是 PileCenterInfo，
    最後都可以轉成這種格式再做選堆。
    """
    pile_id: int
    center_xy: Point2
    count: int


@dataclass
class PileScoreInfo:
    """
    每一堆的評分資訊，方便 debug / 前端顯示。
    """
    pile_id: int
    center_xy: Point2
    count: int
    distance_from_robot_m: float
    near_score: float
    count_score: float
    final_score: float


@dataclass
class PileSelectionResult:
    """
    選堆結果整包輸出。
    """
    best_candidate: Optional[PileCandidate]
    score_infos: List[PileScoreInfo]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_0to1(value: float, vmin: float, vmax: float) -> float:
    """
    把數值正規化到 [0, 1]。

    若所有值都相同，回傳 1.0，
    表示這個維度對所有候選都同樣好，不拉開差距。
    """
    if abs(vmax - vmin) < 1e-9:
        return 1.0
    return (value - vmin) / (vmax - vmin)


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_candidates(
    center_xys: Sequence[Point2],
    counts: Sequence[int],
    pile_ids: Optional[Sequence[int]] = None,
) -> List[PileCandidate]:
    """
    把外部算好的中心座標與球數，整理成 selector 可用的候選清單。
    """
    if len(center_xys) != len(counts):
        raise ValueError("center_xys and counts must have the same length")

    if pile_ids is None:
        pile_ids = list(range(len(center_xys)))

    if len(pile_ids) != len(center_xys):
        raise ValueError("pile_ids and center_xys must have the same length")

    out: List[PileCandidate] = []
    for pid, center, count in zip(pile_ids, center_xys, counts):
        out.append(
            PileCandidate(
                pile_id=int(pid),
                center_xy=(float(center[0]), float(center[1])),
                count=int(count),
            )
        )
    return out


def build_candidates_from_rect_piles(
    pile_plans,
) -> List[PileCandidate]:
    """
    從第一版 RectPileInfo 列表建立候選。
    只要求每個元素有：
    - pile_id
    - center_xy
    - count
    """
    out: List[PileCandidate] = []
    for p in pile_plans:
        out.append(
            PileCandidate(
                pile_id=int(p.pile_id),
                center_xy=(float(p.center_xy[0]), float(p.center_xy[1])),
                count=int(p.count),
            )
        )
    return out


def build_candidates_from_center_plans(
    pile_plans,
) -> List[PileCandidate]:
    """
    從第二版 PileCenterInfo 列表建立候選。
    同樣只要求：
    - pile_id
    - center_xy
    - count
    """
    out: List[PileCandidate] = []
    for p in pile_plans:
        out.append(
            PileCandidate(
                pile_id=int(p.pile_id),
                center_xy=(float(p.center_xy[0]), float(p.center_xy[1])),
                count=int(p.count),
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def select_best_pile(
    robot_pose: Optional[Pose2d],
    pile_candidates: Sequence[PileCandidate],
    ball_priority_0to10: float = 5.0,
) -> PileSelectionResult:
    """
    根據距離與球數加權，選出最佳球堆。

    ball_priority_0to10:
        0  -> 完全偏最近
        10 -> 完全偏球多
    """
    if not pile_candidates:
        return PileSelectionResult(best_candidate=None, score_infos=[])

    alpha = _clamp(float(ball_priority_0to10), 0.0, 10.0) / 10.0

    counts = [c.count for c in pile_candidates]
    count_min = min(counts)
    count_max = max(counts)

    if robot_pose is not None:
        distances = [
            math.hypot(c.center_xy[0] - robot_pose.x, c.center_xy[1] - robot_pose.y)
            for c in pile_candidates
        ]
        dist_min = min(distances)
        dist_max = max(distances)
    else:
        distances = [0.0 for _ in pile_candidates]
        dist_min = 0.0
        dist_max = 0.0

    score_infos: List[PileScoreInfo] = []
    candidate_map: Dict[int, PileCandidate] = {c.pile_id: c for c in pile_candidates}

    for c, dist_m in zip(pile_candidates, distances):
        count_score = _normalize_0to1(
            float(c.count),
            float(count_min),
            float(count_max),
        )

        if robot_pose is None:
            near_score = 0.0
            final_score = count_score
        else:
            dist_norm = _normalize_0to1(dist_m, dist_min, dist_max)
            near_score = 1.0 - dist_norm   # 越近越高
            final_score = (1.0 - alpha) * near_score + alpha * count_score

        score_infos.append(
            PileScoreInfo(
                pile_id=c.pile_id,
                center_xy=c.center_xy,
                count=c.count,
                distance_from_robot_m=dist_m,
                near_score=near_score,
                count_score=count_score,
                final_score=final_score,
            )
        )

    # 依照：
    # 1. final_score 高者優先
    # 2. count 多者優先
    # 3. 距離近者優先
    # 4. pile_id 小者優先（穩定 tie-break）
    score_infos.sort(
        key=lambda s: (
            -s.final_score,
            -s.count,
            s.distance_from_robot_m,
            s.pile_id,
        )
    )

    best_id = score_infos[0].pile_id
    best_candidate = candidate_map.get(best_id)

    return PileSelectionResult(
        best_candidate=best_candidate,
        score_infos=score_infos,
    )