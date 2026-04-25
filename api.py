"""HTTP API."""

import json as _json
from datetime import datetime
import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from csv_to_json import csv_to_route
from schemas import FlaggedShipsResponse, ShipMeta, ShipRouteResponse


# ---------------------------------------------------------------------------
# Filesystem layout
#
#   data/ships/ships.json                   registry [{"id": int, "name": str, "type": str}, ...]
#   data/flagged_ships/flagged_ships.json   {"ids": [int, ...]}
#   data/routes/<filename>.csv              MarineTraffic CSV with leading
#                                           `id` column; one file may hold
#                                           rows for many ships (filtered by id).
#   data/spills/*.json                      GeoJSON Polygon (one file per spill)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
SHIPS_REGISTRY = DATA_DIR / "ships" / "ships.json"
FLAGGED_INDEX = DATA_DIR / "flagged_ships" / "flagged_ships.json"
SPILLS_DIR = DATA_DIR / "spills"
# Routes live under `data/routes/` and are read via `csv_to_json.csv_to_route`.

from schemas import RunConfig, Seed, SimulationRequest, SimulationResult
from simulator import mock_simulate


app = FastAPI(title="MarineTrust API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Loaders. Re-read on every request — files are tiny and edits should be live.
# ---------------------------------------------------------------------------

def _load_ships_registry() -> dict[int, dict]:
    """Index ships.json by integer id."""
    if not SHIPS_REGISTRY.exists():
        return {}
    with open(SHIPS_REGISTRY) as f:
        return {int(s["id"]): s for s in _json.load(f)}
SPILLS_PATH = Path(__file__).parent / "data" / "spills" / "spills.json"
SPILL_WINDOW_HOURS = 60
SPILL_OUTPUT_STEP_S = 1800  # 30 min frames

# Map spill `type` from spills.json to (OpenDrift model, ADIOS oil_type).
SPILL_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "Oil": ("OpenOil", "GENERIC MEDIUM CRUDE"),
}


def _load_spill_geom():
    """Union of all spill polygons in `data/spills/`. None when none exist."""
    if not SPILLS_DIR.exists():
        return None
        
    geoms = []
    for p in sorted(SPILLS_DIR.glob("*.json")):
        with open(p) as f:
            data = _json.load(f)
            
        # 1. Normalize everything into a list so we don't need messy nested if-statements
        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection":
                data = data.get("features", [])
            else:
                data = [data]
                
        # 2. Parse the list safely
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Case A: It's a standard GeoJSON Feature
                if item.get("type") == "Feature" and "geometry" in item:
                    geoms.append(shape(item["geometry"]))
                
                # Case B: It's a raw standard geometry (Polygon, MultiPolygon, Point, etc.)
                elif item.get("type") in ["Polygon", "MultiPolygon", "Point", "LineString"]:
                    geoms.append(shape(item))
                
                # Case C: It's your custom format (e.g. "type": "Oil", "coordinates": [...])
                # We ignore the word "Oil" and force it into a valid Shapely Polygon format
                elif "coordinates" in item:
                    geoms.append(shape({
                        "type": "Polygon", 
                        "coordinates": item["coordinates"]
                    }))

    if not geoms:
        return None
    return geoms[0] if len(geoms) == 1 else unary_union(geoms)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/flaggedShips", response_model=FlaggedShipsResponse)
def get_flagged_ships():
    """Read `data/flagged_ships/flagged_ships.json` and return its `ids` list."""
    if not FLAGGED_INDEX.exists():
        return FlaggedShipsResponse(ids=[])
    with open(FLAGGED_INDEX) as f:
        data = _json.load(f)
    raw = data.get("ids", [])
    # Tolerate a single int/str instead of a list (hand-edited demo data).
    if isinstance(raw, (str, int)):
        raw = [raw]
    return FlaggedShipsResponse(ids=[int(x) for x in raw])


@app.get("/ships/{ship_id}/{date}", response_model=ShipRouteResponse)
def get_ship_on_date(ship_id: int, date: str):
    """
    Route of the ship on a UTC calendar date with per-point collision flags.

    Path:
      ship_id  - integer id from `ships.json`
      date     - YYYY-MM-DD, interpreted as UTC

    Status codes:
      400 - invalid date format
      404 - unknown ship, no route file, or no fixes on that date
    """
    # Open the file in read mode
    with open('data/ship_tracks/1.json', 'r') as file:
        # Parse the JSON file into a Python dictionary
        data = json.load(file)
        
    return data

    try:
        target = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, f"Invalid date '{date}'; expected YYYY-MM-DD")

    registry = _load_ships_registry()
    if ship_id not in registry:
        raise HTTPException(404, f"Ship {ship_id} not found")

    # csv_to_json.csv_to_route does the CSV scanning, id-filtering, masked-row
    # dropping, sorting, and date filtering. We just add a parallel `collisions`
    # array so the response matches the ship_track.json shape on disk.
    route = csv_to_route(ship_id, target_date=target)
    coordinates = route["coordinates"]
    if not coordinates:
        raise HTTPException(404, f"No fixes for ship {ship_id} on {target.isoformat()}")

    # Per-point collision: is this fix inside any spill polygon?
    spill_geom = _load_spill_geom()
    if spill_geom is None:
        collisions = [False] * len(coordinates)
    else:
        collisions = [spill_geom.contains(Point(lon, lat)) for lon, lat in coordinates]

    return ShipRouteResponse(
        timestamps=route["timestamps"],
        coordinates=coordinates,
        speeds=route["speeds"],
        courses=route["courses"],
        collisions=collisions,
    )


@app.get("/ships/{ship_id}", response_model=ShipMeta)
def get_ship(ship_id: int):
    """Registry entry for a single ship."""
    registry = _load_ships_registry()
    if ship_id not in registry:
        raise HTTPException(404, f"Ship {ship_id} not found")
    return ShipMeta(**registry[ship_id])

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
        result = mock_simulate(req)

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
