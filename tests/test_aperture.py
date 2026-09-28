"""视轴投影几何的单元测试：偏心、跨 0°、近切线、并列回归。"""

from __future__ import annotations

import math

import pytest

from app.planner.aperture import (
    REASON_ALTITUDE_RANGE,
    REASON_BELOW_GROUND,
    REASON_ORIGIN_OUTSIDE,
    ApertureValidationError,
    intersect_sphere,
    look_vector,
    nearest_positive_sphere_root,
    project_aperture,
    shortest_covering_arc,
)


def _on_sphere(p, radius: float, tol: float = 1e-8) -> bool:
    return abs(p.east**2 + p.north**2 + p.up**2 - radius**2) <= tol


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------


def test_look_vector_cardinal_directions():
    east, north, up = look_vector(0.0, 0.0)
    assert (east, north, up) == pytest.approx((0.0, 1.0, 0.0))

    east, north, up = look_vector(90.0, 0.0)
    assert (east, north, up) == pytest.approx((1.0, 0.0, 0.0))

    east, north, up = look_vector(180.0, 0.0)
    assert (east, north, up) == pytest.approx((0.0, -1.0, 0.0))

    east, north, up = look_vector(0.0, 90.0)
    assert (east, north, up) == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)

    # 单位向量
    d = look_vector(37.3, 12.8)
    assert sum(c * c for c in d) == pytest.approx(1.0)


def test_centered_horizontal_ray_hits_wall_at_horizon():
    p = intersect_sphere((0.0, 0.0, 0.0), 0.0, 0.0, 10.0)
    assert p.distance == pytest.approx(10.0)
    assert (p.east, p.north, p.up) == pytest.approx((0.0, 10.0, 0.0))
    assert p.azimuth_deg == pytest.approx(0.0)
    assert p.altitude_deg == pytest.approx(0.0)


def test_centered_ray_point_on_sphere_and_direction():
    radius = 10.0
    p = intersect_sphere((0.0, 0.0, 0.0), 30.0, 20.0, radius)
    assert p.distance == pytest.approx(radius)
    assert _on_sphere(p, radius)
    assert p.azimuth_deg == pytest.approx(30.0, abs=1e-9)
    assert p.altitude_deg == pytest.approx(20.0, abs=1e-9)


def test_centered_sample_envelope_is_error_rectangle():
    # 球心起点时交点方位/高度与射线一致：包络即误差矩形
    results = project_aperture(
        10.0, (0.0, 0.0, 0.0), [(30.0, 20.0, 2.0, 1.0)]
    )
    r = results[0]
    assert r.envelope_arc.crosses_zero is False
    assert r.envelope_arc.start_deg == pytest.approx(28.0)
    assert r.envelope_arc.end_deg == pytest.approx(32.0)
    assert r.envelope_arc.center_deg == pytest.approx(30.0)
    assert r.envelope_arc.width_deg == pytest.approx(4.0)
    assert r.min_altitude_deg == pytest.approx(19.0)
    assert r.max_altitude_deg == pytest.approx(21.0)


def test_eccentric_origin_all_five_intersections_valid():
    radius = 10.0
    origin = (1.5, -2.0, 0.5)
    results = project_aperture(radius, origin, [(90.0, 0.0, 1.0, 1.0)])
    r = results[0]

    # 标称：射线朝正东（d=(1,0,0)），t = -b + sqrt(b²-c) = -1.5 + sqrt(95.75)
    expected_t = -1.5 + math.sqrt(100.0 - 4.0 - 0.25)
    assert r.nominal.distance == pytest.approx(expected_t, abs=1e-9)
    assert (r.nominal.east, r.nominal.north, r.nominal.up) == pytest.approx(
        (1.5 + expected_t, -2.0, 0.5), abs=1e-9
    )
    assert _on_sphere(r.nominal, radius)
    # 偏心导致交点方位偏离射线方位
    assert r.nominal.azimuth_deg > 90.0
    assert r.nominal.altitude_deg == pytest.approx(
        math.degrees(math.atan2(0.5, math.hypot(1.5 + expected_t, -2.0))),
        abs=1e-9,
    )
    assert r.min_altitude_deg <= r.nominal.altitude_deg <= r.max_altitude_deg


def test_eccentric_origin_multiple_samples_order_preserved():
    results = project_aperture(
        10.0,
        (1.0, 1.0, 1.0),
        [(10.0, 30.0, 0.5, 0.2), (200.0, 45.0, 1.0, 0.5)],
    )
    assert [r.index for r in results] == [0, 1]
    for r in results:
        assert _on_sphere(r.nominal, 10.0)


def test_origin_on_sphere_surface_rejected():
    with pytest.raises(ApertureValidationError) as exc:
        project_aperture(10.0, (0.0, 0.0, 10.0), [(0.0, 10.0, 0.0, 0.0)])
    assert exc.value.reason == REASON_ORIGIN_OUTSIDE
    assert exc.value.sample_indices == []


def test_origin_outside_sphere_rejected():
    with pytest.raises(ApertureValidationError) as exc:
        project_aperture(10.0, (6.0, 8.0, 1.0), [(0.0, 10.0, 0.0, 0.0)])
    assert exc.value.reason == REASON_ORIGIN_OUTSIDE
    assert exc.value.sample_indices == []


def test_altitude_error_out_of_range_reports_indices():
    with pytest.raises(ApertureValidationError) as exc:
        project_aperture(
            10.0,
            (0.0, 0.0, 0.0),
            [
                (0.0, 45.0, 0.5, 0.5),      # 合法
                (0.0, 89.0, 0.0, 2.0),      # 上沿 91°
                (0.0, -89.0, 0.0, 2.0),     # 下沿 -91°
            ],
        )
    assert exc.value.reason == REASON_ALTITUDE_RANGE
    assert exc.value.sample_indices == [1, 2]


def test_altitude_error_exactly_ninety_accepted():
    results = project_aperture(
        10.0, (0.0, 0.0, 0.0), [(0.0, 89.0, 0.0, 1.0)]
    )
    assert results[0].max_altitude_deg == pytest.approx(90.0, abs=1e-9)
    assert results[0].min_altitude_deg == pytest.approx(88.0)


def test_below_ground_nominal_reports_index():
    with pytest.raises(ApertureValidationError) as exc:
        project_aperture(10.0, (0.0, 0.0, 0.0), [(180.0, -5.0, 0.0, 0.0)])
    assert exc.value.reason == REASON_BELOW_GROUND
    assert exc.value.sample_indices == [0]


def test_below_ground_corner_reports_index_and_keeps_others():
    with pytest.raises(ApertureValidationError) as exc:
        project_aperture(
            10.0,
            (0.0, 0.0, 0.0),
            [
                (0.0, 45.0, 0.5, 0.5),      # 合法
                (180.0, 0.0, 0.0, 0.5),     # 下沿角点在地下
                (90.0, 20.0, 0.1, 0.1),     # 合法
            ],
        )
    assert exc.value.reason == REASON_BELOW_GROUND
    assert exc.value.sample_indices == [1]


def test_exact_horizon_grazing_intersection_accepted():
    # 近切线：起点贴近球壁（y=9.9），射线沿直径方向到对侧地平点 z=0
    p = intersect_sphere((0.0, 9.9, 0.0), 180.0, 0.0, 10.0)
    assert p.distance == pytest.approx(19.9, abs=1e-9)
    assert (p.east, p.north, p.up) == pytest.approx((0.0, -10.0, 0.0), abs=1e-9)
    assert p.altitude_deg == pytest.approx(0.0, abs=1e-9)
    assert _on_sphere(p, 10.0)


def test_near_horizon_symmetric_error_one_corner_below_ground():
    # 近切线回归：标称恰好切地平，对称误差把一个角点带到地下
    with pytest.raises(ApertureValidationError) as exc:
        project_aperture(10.0, (0.0, 0.0, 0.0), [(180.0, 0.0, 0.0, 0.5)])
    assert exc.value.reason == REASON_BELOW_GROUND


def test_nearest_root_exterior_tangent():
    # 几何工具：球外直线到球心垂直距离大于半径 → 不相交
    t = nearest_positive_sphere_root(
        (0.0, 10.0, 10.0), (1.0, 0.0, 0.0), 10.0
    )
    assert t is None
    # 外点发出的割线有两个正根，取近者
    t = nearest_positive_sphere_root(
        (0.0, -20.0, 0.0), (0.0, 1.0, 0.0), 10.0
    )
    assert t == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 最短圆弧包络
# ---------------------------------------------------------------------------


def test_arc_internal_span_no_cross():
    arc = shortest_covering_arc([10.0, 50.0, 30.0, 20.0, 40.0])
    assert arc.crosses_zero is False
    assert arc.start_deg == pytest.approx(10.0)
    assert arc.end_deg == pytest.approx(50.0)
    assert arc.center_deg == pytest.approx(30.0)
    assert arc.width_deg == pytest.approx(40.0)


def test_arc_crosses_zero():
    arc = shortest_covering_arc([350.0, 355.0, 0.0, 5.0, 10.0])
    assert arc.crosses_zero is True
    assert arc.start_deg is None and arc.end_deg is None
    assert arc.width_deg == pytest.approx(20.0)
    # 弧段自 350° 沿增向到 10°（未展开 370°），中心 360° → 归一化为 0°
    assert arc.center_deg == pytest.approx(0.0, abs=1e-9)


def test_arc_crosses_zero_small_span():
    arc = shortest_covering_arc([2.0, 6.0, 6.0, 358.0, 358.0])
    assert arc.crosses_zero is True
    assert arc.width_deg == pytest.approx(8.0)
    assert arc.center_deg == pytest.approx(2.0)


def test_arc_all_identical_azimuths_is_point():
    arc = shortest_covering_arc([42.0] * 5)
    assert arc.width_deg == pytest.approx(0.0)
    assert arc.crosses_zero is False
    assert arc.start_deg == pytest.approx(42.0)
    assert arc.end_deg == pytest.approx(42.0)
    assert arc.center_deg == pytest.approx(42.0)


def test_arc_max_gap_tie_picks_smaller_center():
    # 五点等分方位圆：5 个 72° 间隙并列，各间隙中心
    # 36/108/180/252/324，取中心最小者（36，即 0°→72° 间隙）
    arc = shortest_covering_arc([0.0, 72.0, 144.0, 216.0, 288.0])
    assert arc.crosses_zero is True
    assert arc.width_deg == pytest.approx(288.0)
    # 弧段自 72° 沿增向到 360°，中心 216°
    assert arc.center_deg == pytest.approx(216.0)


def test_arc_max_gap_tie_with_duplicate_picks_smaller_center():
    # 重复 0° 点：0→90、90→180、180→270、跨零四个 90° 间隙并列，
    # 中心分别 45/135/225/324，取 45
    arc = shortest_covering_arc([0.0, 0.0, 90.0, 180.0, 270.0])
    assert arc.crosses_zero is True
    assert arc.width_deg == pytest.approx(270.0)
    assert arc.center_deg == pytest.approx(225.0)
