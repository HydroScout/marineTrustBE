"""Geometry / kinematic helpers for the sea-route planner."""

from __future__ import annotations

import math
from typing import Iterable

from shapely.geometry import LineString, Point


KM_PER_KNOT_HOUR = 1.852  # 1 knot = 1.852 km/h
EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lon, lat) points, in km."""
    lon1, lat1 = a
    lon2, lat2 = b
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def interpolate_route(
    start: tuple[float, float],
    end: tuple[float, float],
    n_segments: int = 64,
) -> list[list[float]]:
    """Simple linear interpolation between start and end (lon, lat).

    Good enough for hackathon mock — not great-circle, but visually fine
    for short/medium routes.
    """
    n_segments = max(2, n_segments)
    coords: list[list[float]] = []
    for i in range(n_segments + 1):
        t = i / n_segments
        lon = start[0] + (end[0] - start[0]) * t
        lat = start[1] + (end[1] - start[1]) * t
        coords.append([lon, lat])
    return coords


def vessel_position_at(
    route: list[list[float]],
    t_hours: float,
    speed_knots: float,
) -> tuple[float, float] | None:
    """Position along the route at time t_hours, given speed in knots.

    Returns None if the vessel has already arrived at the destination.
    """
    if not route or len(route) < 2 or speed_knots <= 0:
        return None
    speed_kmh = speed_knots * KM_PER_KNOT_HOUR
    target_km = t_hours * speed_kmh

    travelled = 0.0
    for i in range(len(route) - 1):
        a = (route[i][0], route[i][1])
        b = (route[i + 1][0], route[i + 1][1])
        seg = haversine_km(a, b)
        if seg <= 0:
            continue
        if travelled + seg >= target_km:
            frac = (target_km - travelled) / seg
            lon = a[0] + (b[0] - a[0]) * frac
            lat = a[1] + (b[1] - a[1]) * frac
            return (lon, lat)
        travelled += seg
    return None


def total_route_km(route: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        total += haversine_km(
            (route[i][0], route[i][1]),
            (route[i + 1][0], route[i + 1][1]),
        )
    return total


def km_per_degree_at(lat: float) -> float:
    """Approximate km per degree of longitude/latitude near a given latitude.

    Used to convert a 'distance' threshold into a degree-space buffer for
    cheap proximity checks.
    """
    # 1 deg lat ≈ 111 km; 1 deg lon ≈ 111 * cos(lat) km. Use the average so
    # we get a single scalar usable for both axes (good enough for warnings).
    lat_km = 111.0
    lon_km = 111.0 * max(0.1, math.cos(math.radians(lat)))
    return (lat_km + lon_km) / 2.0


def line_through_points(coords: Iterable[Iterable[float]]) -> LineString:
    return LineString([(c[0], c[1]) for c in coords])


def to_point(p: tuple[float, float]) -> Point:
    return Point(p[0], p[1])


# ---------------------------------------------------------------------------
# Land-avoiding "shipping corridor" route.
# Picks an open-water waypoint south of the great-circle line if the straight
# line crosses land, then smooths the polyline with a Catmull-Rom spline so
# the result looks like a real maritime trajectory rather than a polygon.
# ---------------------------------------------------------------------------

# Cheap import-once: pull the land STRtree from coastline.py (already loaded
# at backend startup). We only need to test points, so we reuse is_land().
def _is_land(lon: float, lat: float) -> bool:
    try:
        from coastline import is_land as _isl
        return _isl(lon, lat)
    except Exception:
        return False


def polyline_crosses_land(
    coords: list[list[float]],
    samples_per_segment: int = 12,
    coast_skip_deg: float = 0.40,
) -> bool:
    """True if any *interior* sample of the polyline falls on land.

    Most ports sit on the coast and the surrounding ~15-25 km of approach
    corridor is itself classified as land in Natural Earth 50m polygons
    (e.g. Sevastopol's Cape Khersones extends ~8 km SW of the city centre,
    Novorossiysk sits in a deep bay), which would otherwise mark every
    realistic route as land-crossing. We skip samples whose L1 distance
    in degrees to either endpoint is below `coast_skip_deg` (~33 km
    radius at mid-latitudes — the typical port-approach corridor).
    """
    if len(coords) < 2:
        return False
    start = coords[0]
    end = coords[-1]
    for i in range(len(coords) - 1):
        a = coords[i]
        b = coords[i + 1]
        for j in range(1, samples_per_segment):
            t = j / samples_per_segment
            lon = a[0] + (b[0] - a[0]) * t
            lat = a[1] + (b[1] - a[1]) * t
            if (abs(lon - start[0]) + abs(lat - start[1])) < coast_skip_deg:
                continue
            if (abs(lon - end[0]) + abs(lat - end[1])) < coast_skip_deg:
                continue
            if _is_land(lon, lat):
                return True
    return False


def catmull_rom_curve(
    control_points: list[list[float]],
    samples_per_segment: int = 24,
) -> list[list[float]]:
    """Centripetal-style Catmull-Rom curve through every control point.

    Endpoints are duplicated so the curve actually starts at p0 and ends at
    p[-1] (otherwise CR only draws between p1 and p[-2]).
    """
    if len(control_points) < 2:
        return [list(p) for p in control_points]
    pts = [control_points[0]] + control_points + [control_points[-1]]
    out: list[list[float]] = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            out.append([x, y])
    out.append([control_points[-1][0], control_points[-1][1]])
    return out


def candidate_waypoint_sets(
    start: tuple[float, float], end: tuple[float, float]
) -> list[list[list[float]]]:
    """Generate progressively more aggressive detour waypoints.

    Strategy: most coastal traffic in the Black Sea / Mediterranean / Gulf
    detours south of intervening land. We prefer a **2-waypoint coastal
    hugger** (curve heads south, runs east just below the coast, climbs
    back up) over a single deep waypoint — it produces routes that look
    like real shipping lanes and keeps the trip close to the great-circle
    latitude. We start with very small offsets (0.20° ≈ 22 km south of the
    endpoint latitude — enough to clear capes like Khersones) and only
    fall back to deeper detours when the cape is part of a wider peninsula.
    """
    sets: list[list[list[float]]] = []
    q1_lon = start[0] + 0.20 * (end[0] - start[0])
    q3_lon = start[0] + 0.80 * (end[0] - start[0])
    # Coastal hugger — preferred. Tries shallow offsets first.
    for offset in (0.20, 0.30, 0.40, 0.55, 0.75, 1.00, 1.30, 1.70, 2.20):
        sets.append([
            [q1_lon, start[1] - offset],
            [q3_lon, end[1] - offset],
        ])
    # Single mid-waypoint detour — used when the hugger family fails (e.g.
    # the obstruction is centred between the ports rather than near the coast).
    mid_lon = (start[0] + end[0]) / 2
    base_lat = min(start[1], end[1])
    for offset in (0.40, 0.70, 1.00, 1.50, 2.20):
        sets.append([[mid_lon, base_lat - offset]])
    return sets


def build_route(
    start: tuple[float, float],
    end: tuple[float, float],
    n_segments: int = 64,
) -> list[list[float]]:
    """Return a polyline from start to end that avoids land where possible
    and looks like a smooth maritime corridor (Catmull-Rom).

    Strategy:
      1. If the direct line is in open water → straight interpolation.
      2. Otherwise try increasingly aggressive south-offset waypoint sets.
         The first candidate whose smoothed curve also stays off land wins.
      3. If everything still crosses land (extreme case), fall back to the
         direct line so the demo never errors out.
    """
    # Resampling resolution per Catmull-Rom segment so the total point
    # count is in the same ballpark as the straight-line interpolation.
    direct = interpolate_route(start, end, n_segments=n_segments)
    if not polyline_crosses_land([list(start), list(end)]):
        return direct

    s = [start[0], start[1]]
    e = [end[0], end[1]]
    for waypoints in candidate_waypoint_sets(start, end):
        control = [s, *waypoints, e]
        per_seg = max(8, n_segments // (len(control) - 1))
        curve = catmull_rom_curve(control, samples_per_segment=per_seg)
        if not polyline_crosses_land(curve, samples_per_segment=4):
            return curve

    return direct
