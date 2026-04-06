from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Point2 = Tuple[float, float]
Cell2 = Tuple[int, int]
CenterMode = Literal["density_weighted", "rect_center"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RectPileInfo:
    pile_id: int
    points: List[Point2]                # 堆內所有球的場地座標
    cells: List[Cell2]                  # 被占用的格子索引
    count: int                          # 球數
    center_xy: Point2                   # 這一堆的中心（也可視為前往中心）
    rect_min_xy: Point2                 # 外接長方形左下角
    rect_max_xy: Point2                 # 外接長方形右上角
    rect_size_xy: Point2                # 長方形寬、高
    occupied_cell_count: int            # 這堆占了幾個格子
    cell_size_m: float                  # 格子大小
    center_mode: CenterMode = "density_weighted"

    # density_weighted 模式的 debug 欄位
    max_neighbor_count: Optional[int] = None
    used_rect_fallback: bool = False


@dataclass
class RectPilePlanResult:
    """
    給主程式 / 前端使用的整包輸出結果。
    """
    pile_count: int
    pile_plans: List[RectPileInfo]

    # 各堆前往中心
    go_to_centers: List[Point2]

    # 各堆各自的球點
    pile_points_list: List[List[Point2]]

    # 前端若喜歡 x / y 分開可直接使用
    pile_x_lists: List[List[float]]
    pile_y_lists: List[List[float]]

    # 各堆中心拆開的 x / y
    center_x_list: List[float]
    center_y_list: List[float]

    # 各堆 id
    pile_id_list: List[int]

    # 外接矩形資訊（第一版特有，前端畫框很方便）
    rect_corners_list: List[List[Point2]]
    rect_min_list: List[Point2]
    rect_max_list: List[Point2]

    # 各堆占用格子（debug / 前端可視化用）
    cell_list: List[List[Cell2]]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_points(points: Iterable[Optional[Point2]]) -> List[Point2]:
    out: List[Point2] = []
    for p in points:
        if p is None:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _point_to_cell(p: Point2, cell_size_m: float) -> Cell2:
    return (
        int(math.floor(p[0] / cell_size_m)),
        int(math.floor(p[1] / cell_size_m)),
    )


def _build_cell_map(
    points: Sequence[Point2],
    cell_size_m: float,
) -> Dict[Cell2, List[Point2]]:
    cell_map: Dict[Cell2, List[Point2]] = {}
    for p in points:
        cell = _point_to_cell(p, cell_size_m)
        cell_map.setdefault(cell, []).append(p)
    return cell_map


def _get_neighbors(cell: Cell2, diagonal: bool) -> List[Cell2]:
    cx, cy = cell
    if diagonal:
        return [
            (cx - 1, cy - 1), (cx, cy - 1), (cx + 1, cy - 1),
            (cx - 1, cy    ),               (cx + 1, cy    ),
            (cx - 1, cy + 1), (cx, cy + 1), (cx + 1, cy + 1),
        ]
    return [
        (cx - 1, cy),
        (cx + 1, cy),
        (cx, cy - 1),
        (cx, cy + 1),
    ]


def _cluster_cells(
    occupied_cells: Sequence[Cell2],
    diagonal_connect: bool,
) -> List[List[Cell2]]:
    occupied_set = set(occupied_cells)
    visited: set[Cell2] = set()
    components: List[List[Cell2]] = []

    for start in occupied_cells:
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        comp: List[Cell2] = []

        while stack:
            cur = stack.pop()
            comp.append(cur)

            for nb in _get_neighbors(cur, diagonal_connect):
                if nb in occupied_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        components.append(comp)

    return components


def _cells_to_rect(
    cells: Sequence[Cell2],
    cell_size_m: float,
) -> Tuple[Point2, Point2]:
    min_cx = min(c[0] for c in cells)
    max_cx = max(c[0] for c in cells)
    min_cy = min(c[1] for c in cells)
    max_cy = max(c[1] for c in cells)

    rect_min = (
        min_cx * cell_size_m,
        min_cy * cell_size_m,
    )
    rect_max = (
        (max_cx + 1) * cell_size_m,
        (max_cy + 1) * cell_size_m,
    )
    return rect_min, rect_max


def _rect_center(rect_min: Point2, rect_max: Point2) -> Point2:
    return (
        (rect_min[0] + rect_max[0]) / 2.0,
        (rect_min[1] + rect_max[1]) / 2.0,
    )


def _rect_corners(rect_min: Point2, rect_max: Point2) -> List[Point2]:
    """
    回傳外接矩形四角，方便前端直接畫框。
    順序：
      左下 -> 右下 -> 右上 -> 左上
    """
    x0, y0 = rect_min
    x1, y1 = rect_max
    return [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
    ]


# ── Grid-accelerated density-weighted centroid ───────────────────────────────

def _build_grid(
    points: Sequence[Point2],
    cell: float,
) -> Dict[Cell2, List[int]]:
    """
    把點分進 grid，查鄰居時只需搜 3×3 格，平均 O(N) 取代 O(N²)。
    """
    cell = max(cell, 1e-9)
    g: Dict[Cell2, List[int]] = {}
    for i, (x, y) in enumerate(points):
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        g.setdefault(key, []).append(i)
    return g


def _count_neighbors(
    points: Sequence[Point2],
    grid: Dict[Cell2, List[int]],
    radius: float,
) -> List[int]:
    """
    每個點在 radius 內的鄰居數（含自身）。
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


def _density_weighted_centroid(
    pile_points: Sequence[Point2],
    density_radius_m: float,
    rect_min: Point2,
    rect_max: Point2,
) -> Tuple[Point2, bool, int]:
    """
    在矩形框內對所有球做密度加權平均。

    每顆球的權重 = 它在 density_radius_m 內的鄰居數（含自身）。
    若所有球密度相同，退回矩形幾何中心。
    """
    n = len(pile_points)

    if n == 0:
        return _rect_center(rect_min, rect_max), True, 0

    if n == 1:
        return pile_points[0], False, 1

    grid = _build_grid(pile_points, cell=density_radius_m)
    counts = _count_neighbors(pile_points, grid, radius=density_radius_m)
    max_k = max(counts)

    # 若所有球密度完全相同，退回矩形中心
    if all(c == counts[0] for c in counts):
        return _rect_center(rect_min, rect_max), True, max_k

    total_w = sum(counts)
    wx = sum(counts[i] * pile_points[i][0] for i in range(n)) / total_w
    wy = sum(counts[i] * pile_points[i][1] for i in range(n)) / total_w

    return (wx, wy), False, max_k


# ─────────────────────────────────────────────────────────────────────────────
# Public: plan
# ─────────────────────────────────────────────────────────────────────────────

def plan_ballpile_rect_centers(
    ball_xys: Iterable[Optional[Point2]],
    cell_size_m: float = 0.50,
    diagonal_connect: bool = True,
    center_mode: CenterMode = "density_weighted",
    density_radius_m: float = 0.50,
) -> RectPilePlanResult:
    """
    流程：
    1. 球的場地座標切到 cell_size_m × cell_size_m 格子
    2. 相鄰有球的格子連通成同一堆（diagonal_connect 控制斜角）
    3. 每堆建立外接長方形
    4. 依 center_mode 計算中心：
       - density_weighted : 矩形框內所有球的密度加權平均
       - rect_center      : 外接矩形幾何中心

    輸出：
    - 各堆詳細資訊 pile_plans
    - 各堆前往中心 go_to_centers
    - 各堆球點列表 pile_points_list
    - 各堆矩形四角 rect_corners_list
    - 各堆矩形 min/max
    - 各堆占用格子 cell_list
    - 前端友善 x/y lists

    選堆邏輯已統一移至 select_best_pile.py 的 select_best_pile()，
    本檔案不再提供 select_best_rect_pile()，避免雙軌維護造成不一致。
    """
    pts = _to_points(ball_xys)
    if not pts:
        return RectPilePlanResult(
            pile_count=0,
            pile_plans=[],
            go_to_centers=[],
            pile_points_list=[],
            pile_x_lists=[],
            pile_y_lists=[],
            center_x_list=[],
            center_y_list=[],
            pile_id_list=[],
            rect_corners_list=[],
            rect_min_list=[],
            rect_max_list=[],
            cell_list=[],
        )

    cell_map = _build_cell_map(pts, cell_size_m)
    occupied_cells = list(cell_map.keys())
    components = _cluster_cells(occupied_cells, diagonal_connect)

    plans: List[RectPileInfo] = []

    for pile_id, cells in enumerate(components):
        pile_points: List[Point2] = []
        for cell in cells:
            pile_points.extend(cell_map[cell])

        rect_min, rect_max = _cells_to_rect(cells, cell_size_m)
        rect_size = (
            rect_max[0] - rect_min[0],
            rect_max[1] - rect_min[1],
        )

        if center_mode == "density_weighted":
            center_xy, used_fb, max_k = _density_weighted_centroid(
                pile_points,
                density_radius_m=density_radius_m,
                rect_min=rect_min,
                rect_max=rect_max,
            )
            info = RectPileInfo(
                pile_id=pile_id,
                points=list(pile_points),
                cells=list(cells),
                count=len(pile_points),
                center_xy=center_xy,
                rect_min_xy=rect_min,
                rect_max_xy=rect_max,
                rect_size_xy=rect_size,
                occupied_cell_count=len(cells),
                cell_size_m=cell_size_m,
                center_mode="density_weighted",
                max_neighbor_count=max_k,
                used_rect_fallback=used_fb,
            )
        else:
            center_xy = _rect_center(rect_min, rect_max)
            info = RectPileInfo(
                pile_id=pile_id,
                points=list(pile_points),
                cells=list(cells),
                count=len(pile_points),
                center_xy=center_xy,
                rect_min_xy=rect_min,
                rect_max_xy=rect_max,
                rect_size_xy=rect_size,
                occupied_cell_count=len(cells),
                cell_size_m=cell_size_m,
                center_mode="rect_center",
            )

        plans.append(info)

    # ── 給主程式 / 前端直接使用的整理輸出 ────────────────────────────────
    go_to_centers = [p.center_xy for p in plans]
    pile_points_list = [list(p.points) for p in plans]
    pile_x_lists = [[pt[0] for pt in p.points] for p in plans]
    pile_y_lists = [[pt[1] for pt in p.points] for p in plans]
    center_x_list = [p.center_xy[0] for p in plans]
    center_y_list = [p.center_xy[1] for p in plans]
    pile_id_list = [p.pile_id for p in plans]
    rect_corners_list = [_rect_corners(p.rect_min_xy, p.rect_max_xy) for p in plans]
    rect_min_list = [p.rect_min_xy for p in plans]
    rect_max_list = [p.rect_max_xy for p in plans]
    cell_list = [list(p.cells) for p in plans]

    return RectPilePlanResult(
        pile_count=len(plans),
        pile_plans=plans,
        go_to_centers=go_to_centers,
        pile_points_list=pile_points_list,
        pile_x_lists=pile_x_lists,
        pile_y_lists=pile_y_lists,
        center_x_list=center_x_list,
        center_y_list=center_y_list,
        pile_id_list=pile_id_list,
        rect_corners_list=rect_corners_list,
        rect_min_list=rect_min_list,
        rect_max_list=rect_max_list,
        cell_list=cell_list,
    )