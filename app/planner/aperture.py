"""视轴投影：把望远镜视轴（含方位/高度误差矩形四角）投射到偏心球形穹顶内表面。

约定
----
- ENU 坐标：x = 东（East）、y = 北（North）、z = 天顶（Up）；
- 方位角以北为 0°、向东递增，高度角为相对地平面的仰角；
  视轴方向的单位向量为
  ``d = (sin A·cos h, cos A·cos h, sin h)``；
- 穹顶是半径 R、**球心位于 ENU 原点**的上半球，地平面为 z = 0；
  视轴起点 P（望远镜转轴）可相对球心**偏心**，但必须严格位于球内。

对每条前向射线 X(t) = P + t·d（t > 0）解 |X|² = R²：

    t² + 2(P·d)t + (|P|² − R²) = 0

起点在球内时常数项为负，判别式恒正，最近正根 t = −(P·d) + √disc
即射线进入球面的前向交点。交点的球面方位/高度自**球心**（ENU 原点）量取。

五个交点（标称 1 个 + 误差矩形四角）的开口包络：

- 方位包络取覆盖全部方位点的**最短圆弧**：在方位圆上删除最大间隙
  （并列时取间隙中心较小者，中心统一归一化到 [0, 360)），余下弧段即包络；
- 高度包络取五点球面高度的最小/最大值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import normalize_azimuth

FULL_TURN = 360.0

# 相对容差：球面/地平面边界判定与最近正根选择
REL_EPS = 1e-12
# 间隙宽度并列判定容差（度）
GAP_TIE_TOL = 1e-9

REASON_ORIGIN_OUTSIDE = "origin_outside_sphere"
REASON_ALTITUDE_RANGE = "altitude_error_out_of_range"
REASON_BELOW_GROUND = "intersection_below_ground"
REASON_RAY_MISS = "ray_misses_sphere"


class ApertureValidationError(ValueError):
    """视轴投影语义非法，应映射为 HTTP 422。

    ``sample_indices`` 为涉及的样本（0 基输入下标）升序列表；
    与单一样本无关的错误（如起点不在球内）为空列表。
    """

    def __init__(
        self,
        message: str,
        reason: str,
        sample_indices: list[int] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.sample_indices = sorted(set(sample_indices or []))


@dataclass(frozen=True)
class Intersection:
    """前向射线与球面的交点。"""

    distance: float
    east: float
    north: float
    up: float
    azimuth_deg: float
    altitude_deg: float


@dataclass(frozen=True)
class Arc:
    """覆盖方位包络的最短圆弧。

    ``crosses_zero=False`` 时以 ``start_deg`` / ``end_deg`` 表示
    （均在 [0, 360)，start <= end）；``crosses_zero=True`` 时弧段跨过
    0°/360° 边界，以 ``center_deg``（归一化到 [0, 360)）与
    ``width_deg`` 表示，``start_deg`` / ``end_deg`` 为 None。
    """

    crosses_zero: bool
    center_deg: float
    width_deg: float
    start_deg: float | None
    end_deg: float | None


@dataclass(frozen=True)
class SampleProjection:
    index: int
    nominal: Intersection
    envelope_arc: Arc
    min_altitude_deg: float
    max_altitude_deg: float


def look_vector(azimuth_deg: float, altitude_deg: float) -> tuple[float, float, float]:
    """ENU 视轴单位向量（方位北 0° 向东递增）。"""
    az = math.radians(azimuth_deg)
    el = math.radians(altitude_deg)
    horiz = math.cos(el)
    return (
        math.sin(az) * horiz,
        math.cos(az) * horiz,
        math.sin(el),
    )


def nearest_positive_sphere_root(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    radius: float,
) -> float | None:
    """求前向射线与球面的最近正根 t；无正根（外切点等）时返回 None。

    起点严格位于球内时最近正根恒存在；保留一般情形处理以兼作几何工具。
    """
    px, py, pz = origin
    dx, dy, dz = direction
    b = px * dx + py * dy + pz * dz
    c = px * px + py * py + pz * pz - radius * radius
    disc = b * b - c
    # 近切线数值保护：判别式轻微为负时视为相切
    scale = radius * radius + b * b + 1.0
    if disc < 0.0:
        if disc >= -1e-10 * scale:
            disc = 0.0
        else:
            return None
    s = math.sqrt(disc)
    t_near = -b - s
    t_far = -b + s
    tol = REL_EPS * max(radius, 1.0)
    if t_near > tol:
        return t_near
    if t_far > tol:
        return t_far
    return None


def point_azimuth_altitude(
    point: tuple[float, float, float], radius: float
) -> tuple[float, float]:
    """自球心（ENU 原点）量取的球面方位/高度（度）。"""
    x, y, z = point
    z_over_r = min(1.0, max(-1.0, z / radius))
    altitude = math.degrees(math.asin(z_over_r))
    if abs(altitude) < 1e-12:
        altitude = 0.0
    azimuth = normalize_azimuth(math.degrees(math.atan2(x, y)))
    return azimuth, altitude


def intersect_sphere(
    origin: tuple[float, float, float],
    azimuth_deg: float,
    altitude_deg: float,
    radius: float,
) -> Intersection:
    """投射单条前向射线并返回球面交点。"""
    direction = look_vector(azimuth_deg, altitude_deg)
    t = nearest_positive_sphere_root(origin, direction, radius)
    if t is None:
        raise ApertureValidationError(
            "前向射线与球面没有正根交点",
            REASON_RAY_MISS,
        )
    point = (
        origin[0] + t * direction[0],
        origin[1] + t * direction[1],
        origin[2] + t * direction[2],
    )
    az, el = point_azimuth_altitude(point, radius)
    return Intersection(
        distance=t,
        east=point[0],
        north=point[1],
        up=point[2],
        azimuth_deg=az,
        altitude_deg=el,
    )


def shortest_covering_arc(azimuths: list[float]) -> Arc:
    """求覆盖全部方位点的最短圆弧（删除方位圆上的最大间隙）。

    间隙中心统一归一化到 [0, 360)；最大间隙并列时取中心较小者。
    删除的是跨 0° 间隙时弧段不跨 0°（start/end 表示），否则跨 0°
    （center/width 表示）。
    """
    values = sorted(normalize_azimuth(v) for v in azimuths)
    n = len(values)

    # (间隙宽, 间隙中心, 相邻下标 k：间隙位于 values[k]→values[k+1]；
    #  跨 0° 间隙以 k = -1 标记)
    gaps: list[tuple[float, float, int]] = []
    for k in range(n - 1):
        gaps.append(
            (
                values[k + 1] - values[k],
                0.5 * (values[k] + values[k + 1]),
                k,
            )
        )
    wrap_width = values[0] + FULL_TURN - values[-1]
    wrap_center = 0.5 * (values[-1] + values[0] + FULL_TURN)
    if wrap_center >= FULL_TURN:
        wrap_center -= FULL_TURN
    gaps.append((wrap_width, wrap_center, -1))

    max_width = max(g[0] for g in gaps)
    tied = [g for g in gaps if g[0] >= max_width - GAP_TIE_TOL]
    _width, _center, dropped = min(tied, key=lambda g: g[1])

    if dropped < 0:
        # 删除跨 0° 间隙：弧段即 [values[0], values[-1]]，不跨 0°
        start = values[0]
        end = values[-1]
        width = end - start
        center = 0.5 * (start + end)
        return Arc(False, center, width, start, end)

    # 删除内部间隙：弧段自 values[k+1] 沿增向经 360°/0° 到 values[k]+360
    unwrapped_start = values[dropped + 1]
    unwrapped_end = values[dropped] + FULL_TURN
    width = unwrapped_end - unwrapped_start
    center = normalize_azimuth(0.5 * (unwrapped_start + unwrapped_end))
    return Arc(True, center, width, None, None)


def _altitude_in_range(altitude: float, tolerance: float = 1e-9) -> bool:
    return -90.0 - tolerance <= altitude <= 90.0 + tolerance


def project_aperture(
    radius: float,
    origin: tuple[float, float, float],
    samples: list[tuple[float, float, float, float]],
) -> list[SampleProjection]:
    """批量投影：samples 每项为 (方位角, 高度角, 方位对称误差, 高度对称误差)。

    依次校验：起点在球内 → 误差矩形高度角不越出 [-90°, 90°] →
    五个交点均不低于穹顶地平面；任一不过即抛
    :class:`ApertureValidationError`（HTTP 422）并指出样本索引。
    """
    ox, oy, oz = origin
    origin_norm2 = ox * ox + oy * oy + oz * oz
    # 严格“在球内”判定：留出 1e-12 相对数值余量，恰在球面或球外一律拒绝
    if origin_norm2 >= radius * radius * (1.0 - REL_EPS):
        raise ApertureValidationError(
            f"视轴起点不在穹顶球内：|P|²={origin_norm2:.9g} >= R²={radius ** 2:.9g}",
            REASON_ORIGIN_OUTSIDE,
        )

    out_of_range = [
        i
        for i, (_az, el, _da, de) in enumerate(samples)
        if not (
            _altitude_in_range(el - de) and _altitude_in_range(el + de)
        )
    ]
    if out_of_range:
        raise ApertureValidationError(
            "误差矩形的高度角越出 [-90°, 90°] 范围",
            REASON_ALTITUDE_RANGE,
            out_of_range,
        )

    results: list[SampleProjection] = []
    below_ground: list[int] = []
    ground_tol = REL_EPS * radius

    for i, (az, el, da, de) in enumerate(samples):
        nominal = intersect_sphere(origin, az, el, radius)
        corners = [
            intersect_sphere(origin, az - da, el - de, radius),
            intersect_sphere(origin, az - da, el + de, radius),
            intersect_sphere(origin, az + da, el - de, radius),
            intersect_sphere(origin, az + da, el + de, radius),
        ]
        points = [nominal, *corners]

        if any(q.up < -ground_tol for q in points):
            below_ground.append(i)
            continue

        arc = shortest_covering_arc([q.azimuth_deg for q in points])
        altitudes = [q.altitude_deg for q in points]
        results.append(
            SampleProjection(
                index=i,
                nominal=nominal,
                envelope_arc=arc,
                min_altitude_deg=min(altitudes),
                max_altitude_deg=max(altitudes),
            )
        )

    if below_ground:
        raise ApertureValidationError(
            "存在交点低于穹顶地平面（z < 0）",
            REASON_BELOW_GROUND,
            below_ground,
        )

    return results
