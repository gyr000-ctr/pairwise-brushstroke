"""brushstroke 核心 API。

沿笔迹按弧长距离重采样，压感随弧长线性插值（端点精确），
并沿法线生成变宽三角网格；输入非有限、几何退化或网格自交时
抛出 :class:`BrushError`。实例构造后不可变、方法无副作用，
可安全地并发调用。
"""

from __future__ import annotations

import math

__all__ = ["Brush", "BrushError"]


class BrushError(ValueError):
    """参数非法、输入非有限、笔迹退化或网格自交。"""


def _finite(value, what):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise BrushError(f"{what} 必须是实数，得到 {value!r}") from None
    if not math.isfinite(v):
        raise BrushError(f"{what} 必须是有限值，得到 {value!r}")
    return v


def _coerce_points(points):
    try:
        items = list(points)
    except TypeError:
        raise BrushError("points 必须是 ((x, y), pressure) 的可迭代序列") from None
    if len(items) < 2:
        raise BrushError(f"至少需要 2 个笔迹点，得到 {len(items)} 个")
    coerced = []
    for i, item in enumerate(items):
        try:
            xy, pressure = item
            x, y = xy
        except (TypeError, ValueError):
            raise BrushError(f"第 {i} 个笔迹点必须是 ((x, y), pressure) 形式") from None
        coerced.append((
            _finite(x, f"第 {i} 个点 x"),
            _finite(y, f"第 {i} 个点 y"),
            _finite(pressure, f"第 {i} 个点 pressure"),
        ))
    return coerced


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _segments_intersect(a, b, c, d):
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if o1 == 0.0 and _on_segment(a, b, c):
        return True
    if o2 == 0.0 and _on_segment(a, b, d):
        return True
    if o3 == 0.0 and _on_segment(c, d, a):
        return True
    if o4 == 0.0 and _on_segment(c, d, b):
        return True
    return False


def _boundary_self_intersects(boundary):
    """检查闭合边界多边形是否存在非相邻边相交（网格均匀格加速）。"""
    m = len(boundary)
    if m < 4:
        return False
    edges = [(boundary[i], boundary[(i + 1) % m]) for i in range(m)]
    boxes = []
    maxlen = 0.0
    for a, b in edges:
        boxes.append((min(a[0], b[0]), min(a[1], b[1]),
                      max(a[0], b[0]), max(a[1], b[1])))
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L > maxlen:
            maxlen = L
    if maxlen <= 0.0:
        return True  # 所有边界点重合：退化
    inv = 1.0 / maxlen
    grid = {}
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        for gx in range(int(math.floor(x0 * inv)), int(math.floor(x1 * inv)) + 1):
            for gy in range(int(math.floor(y0 * inv)), int(math.floor(y1 * inv)) + 1):
                grid.setdefault((gx, gy), []).append(idx)
    for i in range(m):
        x0, y0, x1, y1 = boxes[i]
        a, b = edges[i]
        seen = set()
        for gx in range(int(math.floor(x0 * inv)), int(math.floor(x1 * inv)) + 1):
            for gy in range(int(math.floor(y0 * inv)), int(math.floor(y1 * inv)) + 1):
                for j in grid.get((gx, gy), ()):
                    if j in seen:
                        continue
                    seen.add(j)
                    if j <= i + 1:  # 每对只测一次，且跳过共享端点的相邻边
                        continue
                    if i == 0 and j == m - 1:
                        continue  # 首尾边相邻
                    if _segments_intersect(a, b, edges[j][0], edges[j][1]):
                        return True
    return False


class Brush:
    """按固定间距重采样笔迹并生成变宽网格的笔刷。

    实例在构造后不可变，所有方法均为纯计算，线程安全。
    """

    __slots__ = ("_spacing", "_min_width", "_max_width")

    def __init__(self, spacing, min_width=0.1, max_width=1.0):
        spacing = _finite(spacing, "spacing")
        if spacing <= 0.0:
            raise BrushError(f"spacing 必须为正数，得到 {spacing!r}")
        min_width = _finite(min_width, "min_width")
        max_width = _finite(max_width, "max_width")
        if min_width < 0.0:
            raise BrushError(f"min_width 不能为负，得到 {min_width!r}")
        if max_width <= 0.0 or min_width > max_width:
            raise BrushError("需要 0 <= min_width <= max_width 且 max_width > 0")
        self._spacing = spacing
        self._min_width = min_width
        self._max_width = max_width

    @property
    def spacing(self):
        return self._spacing

    @property
    def min_width(self):
        return self._min_width

    @property
    def max_width(self):
        return self._max_width

    def width_at(self, pressure):
        """压感（截断到 [0, 1]）对应的线宽，在 min/max_width 间线性变化。"""
        p = _finite(pressure, "pressure")
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        return self._min_width + (self._max_width - self._min_width) * p

    def stroke(self, points):
        """沿笔迹按弧长重采样。

        返回 ``[((x, y), pressure), ...]``；采样间距为 ``self.spacing``，
        压感随弧长线性插值，首尾点与输入端点精确一致。不修改输入。
        """
        pts = _coerce_points(points)
        n = len(pts)
        cum = [0.0] * n
        for i in range(1, n):
            cum[i] = cum[i - 1] + math.hypot(pts[i][0] - pts[i - 1][0],
                                             pts[i][1] - pts[i - 1][1])
        total = cum[-1]
        if total <= 0.0:
            raise BrushError("笔迹退化：总长度为零")
        spacing = self._spacing
        eps = 1e-12 * (total if total > 1.0 else 1.0)
        dists = []
        i = 1
        k = int(total / spacing)
        while i <= k:
            d = i * spacing
            if d < total - eps:
                dists.append(d)
            i += 1
        out = [((pts[0][0], pts[0][1]), pts[0][2])]
        j = 0
        for d in dists:
            while cum[j + 1] < d:
                j += 1
            seg = cum[j + 1] - cum[j]
            t = 0.0 if seg <= 0.0 else (d - cum[j]) / seg
            out.append((
                (pts[j][0] + t * (pts[j + 1][0] - pts[j][0]),
                 pts[j][1] + t * (pts[j + 1][1] - pts[j][1])),
                pts[j][2] + t * (pts[j + 1][2] - pts[j][2]),
            ))
        out.append(((pts[-1][0], pts[-1][1]), pts[-1][2]))
        return out

    def mesh(self, points):
        """沿重采样笔迹的法线生成变宽三角网格。

        返回 ``(vertices, triangles)``：每个采样点贡献左、右两个顶点，
        triangles 为顶点索引三元组。笔迹退化、折返或边界自交时
        抛出 :class:`BrushError`。
        """
        samples = self.stroke(points)
        n = len(samples)
        dirs = []
        for i in range(n - 1):
            dx = samples[i + 1][0][0] - samples[i][0][0]
            dy = samples[i + 1][0][1] - samples[i][0][1]
            L = math.hypot(dx, dy)
            if L <= 0.0:
                raise BrushError("笔迹退化：相邻采样点重合（可能存在折返）")
            dirs.append((dx / L, dy / L))
        left = []
        right = []
        for i in range(n):
            if i == 0:
                tx, ty = dirs[0]
            elif i == n - 1:
                tx, ty = dirs[n - 2]
            else:
                tx = dirs[i - 1][0] + dirs[i][0]
                ty = dirs[i - 1][1] + dirs[i][1]
                L = math.hypot(tx, ty)
                if L <= 1e-12:
                    tx, ty = dirs[i - 1]  # 急折返，退化为段方向
                else:
                    tx, ty = tx / L, ty / L
            nx, ny = -ty, tx
            half = 0.5 * self.width_at(samples[i][1])
            cx, cy = samples[i][0]
            left.append((cx + nx * half, cy + ny * half))
            right.append((cx - nx * half, cy - ny * half))
        # 局部折回检查：偏移点沿前进方向不得倒退（线宽相对转弯半径过大）
        tol = 1e-9 * self._spacing
        for i in range(n - 1):
            dx, dy = dirs[i]
            if ((left[i + 1][0] - left[i][0]) * dx
                    + (left[i + 1][1] - left[i][1]) * dy) < -tol:
                raise BrushError("网格自交：线宽相对转弯半径过大（左侧折回）")
            if ((right[i + 1][0] - right[i][0]) * dx
                    + (right[i + 1][1] - right[i][1]) * dy) < -tol:
                raise BrushError("网格自交：线宽相对转弯半径过大（右侧折回）")
        boundary = left + right[::-1]
        if _boundary_self_intersects(boundary):
            raise BrushError("网格自交：边界多边形存在交叉")
        vertices = []
        for i in range(n):
            vertices.append(left[i])
            vertices.append(right[i])
        triangles = []
        for i in range(n - 1):
            a = 2 * i
            triangles.append((a, a + 1, a + 2))
            triangles.append((a + 1, a + 3, a + 2))
        return vertices, triangles

    build_mesh = mesh
