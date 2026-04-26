"""HTTP API."""

import json
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import MultiPoint, Point

from csv_to_json import build_ship_route
from schemas import RunConfig, Seed, SimulationRequest, SimulationResult, TrajectoryFrame
from simulator import mock_simulate


app = FastAPI(title="MarineTrust API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


SPILLS_PATH = Path(__file__).parent / "data" / "spills" / "spills.json"
ROUTES_DIR = Path(__file__).parent / "data" / "routes"
SHIPS_PATH = Path(__file__).parent / "data" / "ships" / "ships.json"
FLAGGED_SHIPS_PATH = Path(__file__).parent / "data" / "flagged_ships" / "flagged_ships.json"
SPILL_WINDOW_HOURS = 60
SPILL_OUTPUT_STEP_S = 1800  # 30 min frames

# Map spill `type` from spills.json to (OpenDrift model, ADIOS oil_type).
SPILL_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "Oil": ("OpenOil", "GENERIC MEDIUM CRUDE"),
}


def _simulate_spill(spill: dict) -> SimulationResult:
    """Run the mock simulator for a spill record using the same parameters as
    /spills/{date} so the cloud shape matches what the frontend animates."""
    spill_time = datetime.fromisoformat(spill["dateTime"])
    model, oil_type = SPILL_TYPE_MAP.get(spill.get("type", ""), ("OceanDrift", None))
    req = SimulationRequest(
        model=model,
        seed=Seed(
            polygon=spill["coordinates"],
            time=spill_time,
            oil_type=oil_type,
        ),
        run=RunConfig(
            duration_back_hours=SPILL_WINDOW_HOURS,
            duration_forward_hours=SPILL_WINDOW_HOURS,
            output_step_s=SPILL_OUTPUT_STEP_S,
        ),
    )
    return mock_simulate(req)


def _cloud_hull(frame: TrajectoryFrame):
    """Convex hull of active particles in a frame, or None if too few points."""
    pts = [(p.lon, p.lat) for p in frame.points if p.status == "active"]
    if len(pts) < 3:
        return None
    return MultiPoint(pts).convex_hull


def _closest_frame_index(frame_times: list[datetime], ts: datetime) -> int:
    idx = bisect_left(frame_times, ts)
    if idx == 0:
        return 0
    if idx == len(frame_times):
        return len(frame_times) - 1
    before, after = frame_times[idx - 1], frame_times[idx]
    return idx if (after - ts) < (ts - before) else idx - 1


@app.get("/ships/{ship_id}/{date}")
def get_ship_on_date(ship_id: str, date: str):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")

    route = build_ship_route(ROUTES_DIR, ship_id, date)
    if route is None:
        raise HTTPException(status_code=404, detail="No route data for ship/date")

    ship_times = [
        datetime.strptime(ts.rstrip("Z"), "%Y-%m-%dT%H:%M:%S")
        for ts in route["timestamps"]
    ]
    ship_min, ship_max = ship_times[0], ship_times[-1]

    with open(SPILLS_PATH) as f:
        spills = json.load(f)

    # Run simulations only for spills whose animation window overlaps the
    # ship's timestamps. Cache hulls lazily — most ship coords map to the same
    # frame index, so we don't want to rebuild a hull per query.
    sims: list[tuple[list[datetime], list, list]] = []
    for spill in spills:
        spill_time = datetime.fromisoformat(spill["dateTime"])
        anim_start = spill_time - timedelta(hours=SPILL_WINDOW_HOURS)
        anim_end = spill_time + timedelta(hours=SPILL_WINDOW_HOURS)
        if anim_end < ship_min or anim_start > ship_max:
            continue
        result = _simulate_spill(spill)
        frame_times = [f.time for f in result.frames]
        hulls: list = [None] * len(result.frames)
        sims.append((frame_times, result.frames, hulls))

    collisions: list[bool] = []
    for ts, (lon, lat) in zip(ship_times, route["coordinates"]):
        point = Point(lon, lat)
        hit = False
        for frame_times, frames, hulls in sims:
            if ts < frame_times[0] or ts > frame_times[-1]:
                continue
            idx = _closest_frame_index(frame_times, ts)
            if hulls[idx] is None:
                hulls[idx] = _cloud_hull(frames[idx])
            hull = hulls[idx]
            if hull is not None and hull.contains(point):
                hit = True
                break
        collisions.append(hit)

    # shifted = [False] * len(collisions)
    # for i, hit in enumerate(collisions):
    #     if hit and i + 1 < len(collisions):
    #         shifted[i + 1] = True
    # collisions = shifted

    route["collisions"] = collisions
    return route


@app.get("/ships/{ship_id}")
def get_ship(ship_id: str):
    return {}


@app.get("/flaggedShips")
def get_flagged_ships():
    with open(FLAGGED_SHIPS_PATH) as f:
        flagged = {entry["id"]: entry["flaggedDate"] for entry in json.load(f)}
    with open(SHIPS_PATH) as f:
        ships = json.load(f)
    return [
        {**ship, "flaggedDate": flagged[ship["id"]]}
        for ship in ships
        if ship["id"] in flagged
    ]

@app.get("/spills/{date}", response_model=list[SimulationResult])
def get_spills_on_date(date: str):
    try:
        target = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")

    day_start = datetime.combine(target, datetime.min.time())
    day_end = day_start + timedelta(days=1)

    with open(SPILLS_PATH) as f:
        spills = json.load(f)

    animations: list[SimulationResult] = []
    for spill in spills:
        spill_time = datetime.fromisoformat(spill["dateTime"])
        anim_start = spill_time - timedelta(hours=SPILL_WINDOW_HOURS)
        anim_end = spill_time + timedelta(hours=SPILL_WINDOW_HOURS)
        if anim_end < day_start or anim_start >= day_end:
            continue

        result = _simulate_spill(spill)

        kept_frames = []
        new_seed_index = -1
        for i, frame in enumerate(result.frames):
            if frame.time.date() != target:
                continue
            if i == result.seed_frame_index:
                new_seed_index = len(kept_frames)
            kept_frames.append(frame)

        if not kept_frames:
            continue

        result.frames = kept_frames
        result.seed_frame_index = new_seed_index
        animations.append(result)

    return animations
