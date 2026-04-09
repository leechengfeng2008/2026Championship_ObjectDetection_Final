from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Point2 = Tuple[float, float]
CenterMode = Literal["centroid", "density_vb"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PileCenterInfo:
    pile_id: int
    points: List[Point2]                # 這一堆中的球座標
    center_xy: Point2                   # 這一堆的中心（也可作為前往中心）
    count: int                          # 這一堆的球數

    # debug / center mode 資訊
    center_mode: CenterMode = "centroid"
    used_insurance_fallback: bool = False
    densest_neighbor_count: Optional[int] = None
    density_peak_count: Optional[int] = None


@dataclass
class PileCenterPlanResult:
    """
    給主程式 / 前端直接使用的整包輸出。

    pile_plans:
        完整的 pile 資訊，適合後續演算法使用

    go_to_centers:
        每堆前往中心（通常就是 center_xy）

    pile_points_list:
        每堆各自的球點列表，格式：
        [
            [(x1, y1), (x2, y2), ...],   # pile 0
            [(x1, y1), (x2, y2), ...],   # pile 1
        ]

    pile_x_lists / pile_y_lists:
        前端若喜歡 x/y 分開畫 scatter，可直接使用
    """
    pile_count: int
    pile_plans: List[PileCenterInfo]

    go_to_centers: List[Point2]

    pile_points_list: List[List[Point2]]
    pile_x_lists: List[List[float]]
    pile_y_lists: List[List[float]]

    center_x_list: List[float]
    center_y_list: List[float]

    pile_id_list: List[int]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _centroid(points: Sequence[Point2]) -> Point2:
    if not points:
        raise ValueError("points must not be empty")
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
    )


def _to_points(points: Iterable[Optional[Point2]]) -> List[Point2]:
    out: List[Point2] = []
    for p in points:
        if p is None:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _build_grid(points: Sequence[Point2], cell: float) -> Dict[Tuple[int, int], List[int]]:
    """
    Grid 加速：把點分進正方形格子，查詢鄰居時只需搜 3×3 共 9 個格子。
    cell 建議設成 density_radius_m，使每個鄰居一定落在相鄰格內。
    """
    cell = max(cell, 1e-9)
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(points):
        cx = int(math.floor(x / cell))
        cy = int(math.floor(y / cell))
        grid.setdefault((cx, cy), []).append(idx)
    return grid


def _count_neighbors_grid(
    points: Sequence[Point2],
    grid: Dict[Tuple[int, int], List[int]],
    radius: float,
) -> List[int]:
    """
    利用 grid 計算每個點在 radius 內的鄰居數（含自身）。
    """
    r2 = radius * radius
    counts = [0] * len(points)
    for i, (x, y) in enumerate(points):
        cx = int(math.floor(x / radius))
        cy = int(math.floor(y / radius))
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for j in grid.get((gx, gy), []):
                    dx = points[j][0] - x
                    dy = points[j][1] - y
                    if dx * dx + dy * dy <= r2:
                        counts[i] += 1
    return counts


def _smartest_fallback(
    candidates: Sequence[Point2],
    all_points: Sequence[Point2],
) -> Point2:
    """
    保險回退：從 candidates 中選「對 all_points 中所有點的總距離最小」的那一個。
    """
    best_pt = candidates[0]
    best_sum = float("inf")
    for cand in candidates:
        total = sum(_dist(cand, q) for q in all_points)
        if total < best_sum:
            best_sum = total
            best_pt = cand
    return best_pt


def _density_center(
    pile_points: Sequence[Point2],
    density_radius_m: float,
    spread_limit_m: float,
) -> Tuple[Point2, bool, int, int]:
    """
    Version B 密度中心：
    1. 先數每顆球在 radius 內的鄰居數
    2. 找最高密度球群
    3. 若高密度球群過度分散，啟用保險 fallback
    4. 否則取高密度球群的 centroid
    """
    n = len(pile_points)

    if n == 1:
        return pile_points[0], False, 1, 1

    grid = _build_grid(pile_points, cell=density_radius_m)
    neighbor_counts = _count_neighbors_grid(pile_points, grid, radius=density_radius_m)

    max_k = max(neighbor_counts)
    high_density_pts: List[Point2] = [
        pile_points[i] for i, k in enumerate(neighbor_counts) if k == max_k
    ]

    used_fallback = False
    if len(high_density_pts) > 1:
        max_spread = max(
            _dist(high_density_pts[a], high_density_pts[b])
            for a in range(len(high_density_pts))
            for b in range(a + 1, len(high_density_pts))
        )
        if max_spread > spread_limit_m:
            best = _smartest_fallback(high_density_pts, pile_points)
            return best, True, max_k, len(high_density_pts)

    center = _centroid(high_density_pts)
    return center, used_fallback, max_k, len(high_density_pts)


def _count_neighbors_simple(
    points: Sequence[Point2],
    radius: float,
) -> List[int]:
    """
    簡易的半徑內鄰居計算，O(N^2)。
    因為球的數量通常很少，這個簡單版本就夠好用了。
    """
    r2 = radius * radius
    counts: List[int] = []
    for i, (x, y) in enumerate(points):
        count = 0
        for j, (x2, y2) in enumerate(points):
            dx = x2 - x
            dy = y2 - y
            if dx * dx + dy * dy <= r2:
                count += 1
        counts.append(count)
    return counts


def _density_center_simple(
    pile_points: Sequence[Point2],
    density_radius_m: float,
    spread_limit_m: float,
) -> Tuple[Point2, bool, int, int]:
    """
    直接的 density center 演算法，先數每顆球在 radius 內的鄰居，
    再用最高密度點群計算中心。
    """
    n = len(pile_points)
    if n == 1:
        return pile_points[0], False, 1, 1

    neighbor_counts = _count_neighbors_simple(pile_points, radius=density_radius_m)
    max_k = max(neighbor_counts)
    high_density_pts = [
        pile_points[i] for i, k in enumerate(neighbor_counts) if k == max_k
    ]

    used_fallback = False
    if len(high_density_pts) > 1:
        max_spread = max(
            _dist(high_density_pts[a], high_density_pts[b])
            for a in range(len(high_density_pts))
            for b in range(a + 1, len(high_density_pts))
        )
        if max_spread > spread_limit_m:
            best = _smartest_fallback(high_density_pts, pile_points)
            return best, True, max_k, len(high_density_pts)

    center = _centroid(high_density_pts)
    return center, used_fallback, max_k, len(high_density_pts)


def find_best_cluster(
    ball_xys: Iterable[Optional[Point2]],
    cluster_link_m: float = 0.30,
    density_radius_m: float = 0.30,
    density_spread_limit_m: float = 0.30,
) -> Optional[PileCenterInfo]:
    """
    找出最好的球堆群組，使用 single-linkage clustering + density center。

    這個函式會：
    1. 先把球點聚成 cluster
    2. 針對每個 cluster 計算 density center
    3. 以最大球數優先、最高密度優先選出最佳 cluster
    """
    pts = _to_points(ball_xys)
    if not pts:
        return None

    piles = cluster_ball_piles(pts, link_distance_m=cluster_link_m)
    best_cluster: Optional[PileCenterInfo] = None
    best_key = (-1, -1)

    for pile_id, pile_points in enumerate(piles):
        center_xy, used_fb, max_k, peak_n = _density_center_simple(
            pile_points,
            density_radius_m=density_radius_m,
            spread_limit_m=density_spread_limit_m,
        )
        info = PileCenterInfo(
            pile_id=pile_id,
            points=list(pile_points),
            center_xy=center_xy,
            count=len(pile_points),
            center_mode="density_vb",
            used_insurance_fallback=used_fb,
            densest_neighbor_count=max_k,
            density_peak_count=peak_n,
        )

        key = (info.count, info.densest_neighbor_count or 0)
        if key > best_key:
            best_key = key
            best_cluster = info

    return best_cluster


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ball_piles(
    ball_xys: Sequence[Point2],
    link_distance_m: float = 0.30,
) -> List[List[Point2]]:
    """
    Version A: Single-linkage clustering（連通分量）。
    若任兩顆球距離 <= link_distance_m，視為相連，屬於同一堆。
    """
    if not ball_xys:
        return []

    n = len(ball_xys)
    visited = [False] * n
    piles: List[List[Point2]] = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: List[Point2] = []
        while stack:
            u = stack.pop()
            comp.append(ball_xys[u])
            for v in range(n):
                if visited[v]:
                    continue
                if _dist(ball_xys[u], ball_xys[v]) <= link_distance_m:
                    visited[v] = True
                    stack.append(v)
        piles.append(comp)

    return piles


def plan_ballpile_centers(
    ball_xys: Iterable[Optional[Point2]],
    cluster_link_m: float = 0.30,
    center_mode: CenterMode = "density_vb",
    density_radius_m: float = 0.30,
    density_spread_limit_m: float = 0.30,
) -> PileCenterPlanResult:
    """
    Pipeline:
    1. 清理輸入球座標
    2. single-linkage 分堆
    3. 依 center_mode 計算每堆中心
    4. 回傳完整結果，方便主程式與前端使用
    """
    pts = _to_points(ball_xys)
    if not pts:
        return PileCenterPlanResult(
            pile_count=0,
            pile_plans=[],
            go_to_centers=[],
            pile_points_list=[],
            pile_x_lists=[],
            pile_y_lists=[],
            center_x_list=[],
            center_y_list=[],
            pile_id_list=[],
        )

    piles = cluster_ball_piles(pts, link_distance_m=cluster_link_m)

    plans: List[PileCenterInfo] = []

    for pile_id, pile_points in enumerate(piles):
        if center_mode == "density_vb":
            center_xy, used_fb, max_k, peak_n = _density_center(
                pile_points,
                density_radius_m=density_radius_m,
                spread_limit_m=density_spread_limit_m,
            )
            info = PileCenterInfo(
                pile_id=pile_id,
                points=list(pile_points),
                center_xy=center_xy,
                count=len(pile_points),
                center_mode="density_vb",
                used_insurance_fallback=used_fb,
                densest_neighbor_count=max_k,
                density_peak_count=peak_n,
            )
        else:
            center_xy = _centroid(pile_points)
            info = PileCenterInfo(
                pile_id=pile_id,
                points=list(pile_points),
                center_xy=center_xy,
                count=len(pile_points),
                center_mode="centroid",
            )

        plans.append(info)

    # ── 整理成前端友善輸出 ────────────────────────────────────────────────
    go_to_centers = [p.center_xy for p in plans]
    pile_points_list = [list(p.points) for p in plans]
    pile_x_lists = [[pt[0] for pt in p.points] for p in plans]
    pile_y_lists = [[pt[1] for pt in p.points] for p in plans]
    center_x_list = [p.center_xy[0] for p in plans]
    center_y_list = [p.center_xy[1] for p in plans]
    pile_id_list = [p.pile_id for p in plans]

    return PileCenterPlanResult(
        pile_count=len(plans),
        pile_plans=plans,
        go_to_centers=go_to_centers,
        pile_points_list=pile_points_list,
        pile_x_lists=pile_x_lists,
        pile_y_lists=pile_y_lists,
        center_x_list=center_x_list,
        center_y_list=center_y_list,
        pile_id_list=pile_id_list,
    )