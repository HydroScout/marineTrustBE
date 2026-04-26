"""POST /api/sea/routes/analyze — Waze-for-Sea pre-trip risk endpoint.

Spatio-temporal collision detection between a vessel route (LineString) and
drifting hazard polygons (mocked OpenDrift).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from shapely.geometry import Polygon

from .mock_drift import (
    GROWTH_PER_HOUR,
    WIND_DX_DEG_PER_H,
    WIND_DY_DEG_PER_H,
    patch_to_geojson_coords,
    predict_patch,
)
from .utils import (
    KM_PER_KNOT_HOUR,
    build_route,
    candidate_waypoint_sets,
    catmull_rom_curve,
    haversine_km,
    interpolate_route,
    km_per_degree_at,
    polyline_crosses_land,
    total_route_km,
    vessel_position_at,
)


router = APIRouter()


SPILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "spills" / "spills.json"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    start: list[float] = Field(..., description="[lon, lat] of departure")
    end: list[float] = Field(..., description="[lon, lat] of destination")
    speed_knots: float = Field(12.0, gt=0, le=60)
    departure_time: datetime = Field(..., description="UTC departure time")


class GeoLineString(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[list[float]]


class Intersection(BaseModel):
    hazard_type: str
    intersection_point: list[float]
    eta_hours: float
    confidence: float
    hazard_index: int


class PredictedPatch(BaseModel):
    hazard_index: int
    hazard_type: str
    t_hours: float
    polygon: list[list[list[float]]]


class DriftParams(BaseModel):
    """Constants used by the mock drift model. Returned to the client so the
    UI can recompute polygon shape at any slider time without round-tripping
    through the backend."""
    wind_dx_deg_per_h: float
    wind_dy_deg_per_h: float
    growth_per_hour: float


class AnalyzeResponse(BaseModel):
    route: GeoLineString
    risk_level: Literal["safe", "warning", "critical"]
    intersections: list[Intersection]
    hazards: dict
    predicted_patches: list[PredictedPatch]
    eta_total_hours: float
    distance_km: float
    departure_time: datetime
    recommendation: str
    drift_params: DriftParams


# ---------------------------------------------------------------------------
# Hazard loading (uses existing spills.json — read-only)
# ---------------------------------------------------------------------------

def _load_hazards() -> list[dict]:
    if not SPILLS_PATH.exists():
        return []
    with open(SPILLS_PATH) as f:
        spills = json.load(f)

    hazards: list[dict] = []
    for spill in spills:
        coords = spill.get("coordinates")
        if not coords:
            continue
        outer = coords[0]
        if len(outer) < 4:
            continue
        try:
            poly = Polygon([(c[0], c[1]) for c in outer])
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid or poly.is_empty:
                continue
        hazards.append({
            "polygon": poly,
            "hazard_type": (spill.get("pollutionType") or "oil_spill").lower().replace(" ", "_"),
            "raw_coordinates": coords,
        })
    return hazards


def _hazards_geojson(hazards: list[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "hazard_index": idx,
                    "type": h["hazard_type"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": h["raw_coordinates"],
                },
            }
            for idx, h in enumerate(hazards)
        ],
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

# Tunables — kept module-local so they don't leak into the existing code.
TIME_STEP_HOURS = 0.5
WARNING_DISTANCE_KM = 25.0
ROUTE_SEGMENTS = 96


def _collision_check(
    route: list[list[float]],
    hazards: list[dict],
    eta_total_hours: float,
    speed_knots: float,
) -> tuple[list[Intersection], list[PredictedPatch], dict[int, float]]:
    """Run the spatio-temporal collision loop for a pre-built route.

    Returns (intersections, predicted_patches_at_collision, min_proximity_km).
    Pure function — used by both /analyze and /alternative.
    """
    n_steps = min(240, max(2, int(eta_total_hours / TIME_STEP_HOURS) + 1))
    t_values = [i * TIME_STEP_HOURS for i in range(n_steps)]
    if t_values and t_values[-1] < eta_total_hours:
        t_values.append(eta_total_hours)

    intersections: list[Intersection] = []
    predicted_patches: list[PredictedPatch] = []
    seen_intersections: set[int] = set()
    min_proximity_km: dict[int, float] = {}

    from shapely.geometry import Point as _Pt

    for t in t_values:
        vessel = vessel_position_at(route, t, speed_knots)
        if vessel is None:
            continue

        for idx, hazard in enumerate(hazards):
            patch = predict_patch(hazard["polygon"], t)
            if patch.is_empty:
                continue

            centroid = patch.centroid
            prox_km = haversine_km(vessel, (centroid.x, centroid.y))
            if idx not in min_proximity_km or prox_km < min_proximity_km[idx]:
                min_proximity_km[idx] = prox_km

            vp = _Pt(vessel[0], vessel[1])
            hits = patch.contains(vp) or patch.touches(vp)
            if not hits:
                near_deg = (WARNING_DISTANCE_KM / 4) / km_per_degree_at(vessel[1])
                if patch.distance(vp) <= near_deg:
                    hits = True

            if hits and idx not in seen_intersections:
                seen_intersections.add(idx)
                confidence = max(0.4, 1.0 - 0.04 * t)
                intersections.append(
                    Intersection(
                        hazard_type=hazard["hazard_type"],
                        intersection_point=[vessel[0], vessel[1]],
                        eta_hours=round(t, 2),
                        confidence=round(confidence, 2),
                        hazard_index=idx,
                    )
                )
                predicted_patches.append(
                    PredictedPatch(
                        hazard_index=idx,
                        hazard_type=hazard["hazard_type"],
                        t_hours=round(t, 2),
                        polygon=patch_to_geojson_coords(patch),
                    )
                )

    return intersections, predicted_patches, min_proximity_km


def _build_response(
    route: list[list[float]],
    hazards: list[dict],
    intersections: list[Intersection],
    predicted_patches: list[PredictedPatch],
    min_proximity_km: dict[int, float],
    eta_total_hours: float,
    distance_km: float,
    departure_time: datetime,
    recommendation_override: str | None = None,
) -> AnalyzeResponse:
    # Always include at least one snapshot per hazard so the UI can animate.
    if not predicted_patches:
        for idx, hazard in enumerate(hazards):
            mid_t = eta_total_hours / 2 if eta_total_hours > 0 else 0
            patch = predict_patch(hazard["polygon"], mid_t)
            predicted_patches.append(
                PredictedPatch(
                    hazard_index=idx,
                    hazard_type=hazard["hazard_type"],
                    t_hours=round(mid_t, 2),
                    polygon=patch_to_geojson_coords(patch),
                )
            )

    if intersections:
        risk_level: Literal["safe", "warning", "critical"] = "critical"
        recommendation = recommendation_override or "Reroute around hazard or delay departure."
    elif any(d <= WARNING_DISTANCE_KM for d in min_proximity_km.values()):
        risk_level = "warning"
        recommendation = recommendation_override or "Proceed with caution — hazard predicted near route."
    else:
        risk_level = "safe"
        recommendation = recommendation_override or "Route appears clear — safe to depart."

    return AnalyzeResponse(
        route=GeoLineString(coordinates=route),
        risk_level=risk_level,
        intersections=intersections,
        hazards=_hazards_geojson(hazards),
        predicted_patches=predicted_patches,
        eta_total_hours=round(eta_total_hours, 2),
        distance_km=round(distance_km, 2),
        departure_time=departure_time,
        recommendation=recommendation,
        drift_params=DriftParams(
            wind_dx_deg_per_h=WIND_DX_DEG_PER_H,
            wind_dy_deg_per_h=WIND_DY_DEG_PER_H,
            growth_per_hour=GROWTH_PER_HOUR,
        ),
    )


def _analyse(req: AnalyzeRequest) -> AnalyzeResponse:
    start = (req.start[0], req.start[1])
    end = (req.end[0], req.end[1])

    # build_route returns a smooth Catmull-Rom curve that detours around land
    # (Crimea, Taman peninsula, etc.). For port pairs in fully open water it
    # collapses to a plain straight-line interpolation.
    route = build_route(start, end, n_segments=ROUTE_SEGMENTS)
    distance_km = total_route_km(route)
    speed_kmh = req.speed_knots * KM_PER_KNOT_HOUR
    eta_total_hours = distance_km / speed_kmh if speed_kmh > 0 else 0.0

    hazards = _load_hazards()
    intersections, predicted_patches, min_proximity_km = _collision_check(
        route, hazards, eta_total_hours, req.speed_knots
    )

    return _build_response(
        route=route,
        hazards=hazards,
        intersections=intersections,
        predicted_patches=predicted_patches,
        min_proximity_km=min_proximity_km,
        eta_total_hours=eta_total_hours,
        distance_km=distance_km,
        departure_time=req.departure_time,
    )


def _alternative_waypoint_sets(
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[list[list[float]]]:
    """Detour candidates for the alternative-route finder.

    Different from build_route's land-only candidates: here we explicitly
    try BOTH north-of-line and south-of-line offsets, since the safer side
    depends on which way the hazard is drifting. Larger offsets first, so
    the alternative is visibly distinct from the main route.
    """
    sets: list[list[list[float]]] = []
    mid_lon = (start[0] + end[0]) / 2
    avg_lat = (start[1] + end[1]) / 2
    for offset in (0.6, 0.9, 1.2, 1.6, 2.0, 2.6):
        sets.append([[mid_lon, avg_lat + offset]])  # north
        sets.append([[mid_lon, avg_lat - offset]])  # south
    # S-curves with 2 waypoints — for very long routes where a single waypoint
    # creates a sharp dog-leg.
    for offset in (1.0, 1.5, 2.0):
        q1_lon = start[0] + 0.30 * (end[0] - start[0])
        q3_lon = start[0] + 0.70 * (end[0] - start[0])
        sets.append([[q1_lon, avg_lat + offset], [q3_lon, avg_lat + offset]])
        sets.append([[q1_lon, avg_lat - offset], [q3_lon, avg_lat - offset]])
    return sets


def _find_alternative(req: AnalyzeRequest) -> AnalyzeResponse:
    """Find a curved route from req.start to req.end that:
       1. Starts and ends at the EXACT same coordinates as the main route.
       2. Stays clear of land (Natural Earth land polygons).
       3. Has no spatio-temporal collision with any (drifted) hazard.

    Strategy: try waypoint candidates in order of increasing offset; for each,
    build a Catmull-Rom curve, verify it clears land, then run the same
    collision loop. First clean candidate wins. If none qualifies we still
    return the safest candidate (lowest min_proximity), so the user sees an
    *attempted* alternative rather than nothing.
    """
    start = (req.start[0], req.start[1])
    end = (req.end[0], req.end[1])

    hazards = _load_hazards()
    speed_kmh = req.speed_knots * KM_PER_KNOT_HOUR

    best_safe: tuple[list[list[float]], float, list[Intersection], list[PredictedPatch], dict[int, float]] | None = None
    best_fallback: tuple[float, list[list[float]], float, list[Intersection], list[PredictedPatch], dict[int, float]] | None = None

    for waypoints in _alternative_waypoint_sets(start, end):
        control = [[start[0], start[1]], *waypoints, [end[0], end[1]]]
        per_seg = max(8, ROUTE_SEGMENTS // (len(control) - 1))
        curve = catmull_rom_curve(control, samples_per_segment=per_seg)

        # Endpoints are the EXACT requested ports — Catmull-Rom guarantees
        # the curve passes through the first and last control points.
        if polyline_crosses_land(curve, samples_per_segment=4):
            continue

        dist_km = total_route_km(curve)
        eta_h = dist_km / speed_kmh if speed_kmh > 0 else 0.0
        intersections, patches, min_prox = _collision_check(
            curve, hazards, eta_h, req.speed_knots
        )

        if not intersections:
            best_safe = (curve, eta_h, intersections, patches, min_prox)
            break

        # Track the candidate with the largest min-proximity as a graceful
        # fallback in pathological cases where every detour still clips a
        # drifted patch.
        worst_prox = min(min_prox.values()) if min_prox else float("inf")
        if best_fallback is None or worst_prox > best_fallback[0]:
            best_fallback = (worst_prox, curve, eta_h, intersections, patches, min_prox)

    if best_safe is not None:
        curve, eta_h, intersections, patches, min_prox = best_safe
        recommendation = (
            f"Alternative routing clear — adds {eta_h - total_route_km(build_route(start, end, ROUTE_SEGMENTS)) / speed_kmh:+.1f} h "
            f"vs. the direct course."
        )
    elif best_fallback is not None:
        _, curve, eta_h, intersections, patches, min_prox = best_fallback
        recommendation = "Best detour we could find — still close to a hazard. Consider delaying departure."
    else:
        # Nothing tried — fall back to the direct (build_route) result.
        curve = build_route(start, end, n_segments=ROUTE_SEGMENTS)
        eta_h = total_route_km(curve) / speed_kmh if speed_kmh > 0 else 0.0
        intersections, patches, min_prox = _collision_check(
            curve, hazards, eta_h, req.speed_knots
        )
        recommendation = "No detour needed — direct route is acceptable."

    return _build_response(
        route=curve,
        hazards=hazards,
        intersections=intersections,
        predicted_patches=patches,
        min_proximity_km=min_prox,
        eta_total_hours=eta_h,
        distance_km=total_route_km(curve),
        departure_time=req.departure_time,
        recommendation_override=recommendation,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/routes/analyze", response_model=AnalyzeResponse)
def analyze_route(req: AnalyzeRequest) -> AnalyzeResponse:
    return _analyse(req)


@router.post("/routes/alternative", response_model=AnalyzeResponse)
def alternative_route(req: AnalyzeRequest) -> AnalyzeResponse:
    """Same start/end as /analyze but returns a curved route that detours
    around land AND any (drifted) hazard polygons. Used by the UI's
    'Suggest alternative route' button."""
    return _find_alternative(req)


@router.get("/hazards")
def list_hazards() -> dict:
    """Convenience GET so the frontend can render hazards before analysis."""
    return _hazards_geojson(_load_hazards())


@router.get("/ports")
def list_ports() -> list[dict]:
    """Predefined ports to populate dropdowns. Coordinates are nudged a few
    km offshore from the actual city centre so they fall in open water rather
    than inside the Natural Earth 50m land polygons (which has ~5 km
    resolution). This matters for the route start/end pins on the map: with
    on-land coordinates the pin sits visually on top of the city, with the
    nudged ones it sits at the harbour mouth — and the route polyline can
    start/end without immediately tripping the land-avoidance check."""
    return [
        # Gulf of Mexico
        {"id": "new_orleans", "name": "New Orleans, USA", "lon": -89.95, "lat": 29.30},
        {"id": "houston", "name": "Houston, USA", "lon": -94.50, "lat": 28.90},
        {"id": "tampa", "name": "Tampa, USA", "lon": -82.75, "lat": 27.70},
        {"id": "miami", "name": "Miami, USA", "lon": -80.10, "lat": 25.77},
        # Black Sea — Crimean & Russian coast
        {"id": "sevastopol", "name": "Sevastopol, Crimea", "lon": 33.45, "lat": 44.52},
        {"id": "yalta", "name": "Yalta, Crimea", "lon": 34.16, "lat": 44.45},
        {"id": "kerch", "name": "Kerch, Crimea", "lon": 36.55, "lat": 45.28},
        {"id": "taman", "name": "Taman, Russia", "lon": 36.65, "lat": 45.10},
        {"id": "anapa", "name": "Anapa, Russia", "lon": 37.25, "lat": 44.83},
        {"id": "novorossiysk", "name": "Novorossiysk, Russia", "lon": 37.80, "lat": 44.65},
        {"id": "tuapse", "name": "Tuapse, Russia", "lon": 39.00, "lat": 44.08},
        {"id": "sochi", "name": "Sochi, Russia", "lon": 39.72, "lat": 43.58},
        {"id": "istanbul", "name": "Istanbul, Türkiye", "lon": 28.97, "lat": 41.01},
    ]
