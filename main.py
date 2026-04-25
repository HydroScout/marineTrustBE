"""
OpenDrift mock REST API.

Job-based async pattern:
  POST /simulations         -> 202 + job id, runs in background
  GET  /simulations/{id}    -> status / progress
  GET  /simulations/{id}/result -> trajectory frames as JSON (for the map)

To swap the mock for real OpenDrift, replace _mock_simulate() — see the comment
block above that function. The HTTP surface stays identical.
"""

import asyncio
import json as _json
import math
import random
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import Point, Polygon as ShapelyPolygon, shape
from shapely.strtree import STRtree


# ---------------------------------------------------------------------------
# Request schemas — these mirror OpenDrift's seed_elements() / run() inputs.
# When wiring real OpenDrift, fields map 1:1 to the library calls.
# ---------------------------------------------------------------------------

class Seed(BaseModel):
    # PROVIDE: where you're releasing oil.
    # Two modes (mutually exclusive):
    #   - point seed: lon + lat + radius_m (Gaussian cloud at the point)
    #   - polygon seed: polygon (uniform fill of the polygon's outer ring)
    # Real OpenDrift exposes both: seed_elements(lon, lat, radius=...) for
    # point releases and seed_within_polygon(lons, lats, ...) for area releases.
    lon: Optional[float] = Field(None, description="Release longitude (point mode)")
    lat: Optional[float] = Field(None, description="Release latitude (point mode)")
    radius_m: float = Field(1000, description="1-sigma release spread, meters (point mode)")
    polygon: Optional[list[list[list[float]]]] = Field(
        None,
        description="GeoJSON Polygon coordinates: [[[lon, lat], ...]]. "
                    "If set, particles uniformly fill the outer ring; lon/lat/radius_m are ignored."
    )
    number: int = Field(1000, ge=10, le=50000, description="Particle count")
    time: datetime = Field(..., description="Release time, UTC")
    z: float = Field(0, description="Depth in meters. 0 = surface.")
    # OpenOil-only:
    oil_type: Optional[str] = Field(None, description="ADIOS oil type")
    m3_per_hour: Optional[float] = Field(None, description="Continuous release rate")


class Forcing(BaseModel):
    # PROVIDE: URLs / paths to environmental data (currents, wind, waves...).
    # Real values look like:
    #   currents = "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be"
    #   wind     = "https://thredds.met.no/thredds/dodsC/meps25epsarchive/.../meps_det.nc"
    #   waves    = "https://thredds.met.no/thredds/dodsC/cmems/.../wave"
    # The mock ignores these — kept here so the request shape matches production.
    currents: Optional[str] = None
    wind: Optional[str] = None
    waves: Optional[str] = None
    temperature: Optional[str] = None


class RunConfig(BaseModel):
    # The simulation now runs in BOTH directions from the seed time.
    # Real OpenDrift supports backward runs via a negative `time_step` —
    # particles are traced upstream through the velocity field.
    duration_back_hours: float = Field(48, ge=0, le=720,
        description="Hours to trace backward from seed time")
    duration_forward_hours: float = Field(48, ge=0, le=720,
        description="Hours to forecast forward from seed time")
    time_step_s: int = Field(600, description="Internal calculation step")
    output_step_s: int = Field(3600, description="Position sampling step")


class SimulationRequest(BaseModel):
    model: Literal["OpenOil", "OceanDrift", "Leeway"] = "OceanDrift"
    seed: Seed
    forcing: Forcing = Field(default_factory=Forcing)
    run: RunConfig = Field(default_factory=RunConfig)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class SimulationJob(BaseModel):
    id: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    request: SimulationRequest


class TrajectoryPoint(BaseModel):
    particle_id: int
    lat: float
    lon: float
    status: Literal["active", "stranded", "evaporated"] = "active"
    mass_oil: Optional[float] = None  # OpenOil only


class TrajectoryFrame(BaseModel):
    time: datetime
    points: list[TrajectoryPoint]


class SimulationResult(BaseModel):
    job_id: str
    model: str
    # Centroid (point mode: the seed point itself; polygon mode: polygon centroid).
    seed_lon: float
    seed_lat: float
    # The polygon used for area-fill seeding (if any). Frontend draws this
    # on the map to show where particles were released.
    seed_polygon: Optional[list[list[list[float]]]] = None
    # Index into `frames` of the seed time (t=0). Frames before are backward
    # in time, frames after are forward. Frontend uses this to label the slider.
    seed_frame_index: int = 0
    frames: list[TrajectoryFrame]
    # In production: pre-signed S3 URL to the CF-compliant netCDF that
    # OpenDrift writes via run(outfile=...). Mock returns a placeholder path.
    netcdf_url: Optional[str] = None
    oil_budget: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# In-memory store. PROVIDE: replace with Postgres + S3 for production.
# ---------------------------------------------------------------------------

JOBS: dict[str, SimulationJob] = {}
RESULTS: dict[str, SimulationResult] = {}


# ---------------------------------------------------------------------------
# Mock simulator. REPLACE WITH REAL OPENDRIFT:
#
#   from opendrift.models.openoil import OpenOil
#   o = OpenOil(loglevel=20)
#   o.add_readers_from_list([req.forcing.currents, req.forcing.wind])
#   o.seed_elements(
#       lon=req.seed.lon, lat=req.seed.lat,
#       radius=req.seed.radius_m, number=req.seed.number,
#       time=req.seed.time, z=req.seed.z,
#       oil_type=req.seed.oil_type,
#   )
#   o.run(duration=timedelta(hours=req.run.duration_hours),
#         time_step=req.run.time_step_s,
#         time_step_output=req.run.output_step_s,
#         outfile=f"/data/{job_id}.nc")
#   # then read o.history (xarray.Dataset) into TrajectoryFrames
# ---------------------------------------------------------------------------

def _current(lon: float, lat: float, seed_lon: float, seed_lat: float) -> tuple[float, float]:
    """
    Spatially-varying mock current field. Returns (u, v) in m/s.

    PROVIDE: real OpenDrift reads u(x, y, t) and v(x, y, t) from netCDF
    forcing files (Copernicus Marine, ROMS, NEMO, HYCOM, ...). Real fields
    contain jets, eddies, fronts, and tides — all of which stretch and fold
    a particle cloud into the filament-and-streak shapes you see in
    satellite imagery of actual oil slicks.

    The mock combines three pieces so the cloud visibly deforms:
      1. uniform NE base flow,
      2. meridional shear: eastward speed grows with latitude (coastal-jet
         analogue) — stretches the cloud along its direction of travel,
      3. a Gaussian-blob eddy NE of the seed — particles passing through
         get curled around it.
    """
    # 1. base flow (~0.32 m/s NNE — comparable to a regional coastal jet
    #    such as the Black Sea Rim Current or Norwegian Coastal Current).
    u = 0.25
    v = 0.20
    # 2. meridional shear (~1.5 m/s per degree latitude — strong but plausible
    #    for a coastal jet)
    u += 1.5 * (lat - seed_lat)
    # 3. Gaussian eddy 0.3° east, 0.15° north of the seed
    dx = lon - (seed_lon + 0.3)
    dy = lat - (seed_lat + 0.15)
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2 + 1e-6)
    sigma = 0.15
    swirl = 0.18 * math.exp(-r2 / (2 * sigma * sigma))
    u += -swirl * dy / r
    v += swirl * dx / r
    return u, v


# ---- Coastline ------------------------------------------------------------
# Natural Earth 50m land polygons (~5 MB) — accurate to ~5 km, matches OSM
# tiles at zoom ~9. Downloaded once on first server startup, cached on disk.
#
# PROVIDE: real OpenDrift uses GSHHG (sub-km accuracy) via
# `reader_global_landmask`. Swap _COASTLINE_URL for the higher-resolution
# `ne_10m_land.geojson` (~30 MB, ~1 km accuracy) if your zoom level needs it.

_COASTLINE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
_COASTLINE_FILE = Path(__file__).parent / "data" / "ne_50m_land.geojson"
_LAND_TREE: Optional[STRtree] = None
_LAND_GEOMS: list = []


def _load_coastline() -> None:
    global _LAND_TREE, _LAND_GEOMS
    _COASTLINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Download atomically (to .tmp, then rename) so a truncated transfer
    # doesn't poison the cache.
    if not _COASTLINE_FILE.exists():
        tmp = _COASTLINE_FILE.with_suffix(".tmp")
        print(f"[opendrift-mock] downloading coastline: {_COASTLINE_URL}")
        with urllib.request.urlopen(_COASTLINE_URL, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.rename(_COASTLINE_FILE)
        print(f"[opendrift-mock] cached at {_COASTLINE_FILE}")
    try:
        with open(_COASTLINE_FILE) as f:
            data = _json.load(f)
    except Exception:
        # Corrupt cache — wipe and let the next startup retry.
        _COASTLINE_FILE.unlink(missing_ok=True)
        raise
    geoms = []
    for feat in data["features"]:
        g = shape(feat["geometry"])
        # Polygon -> single, MultiPolygon -> split so the STRtree gets tighter
        # bounding boxes per piece (faster queries near complex coasts).
        if g.geom_type == "MultiPolygon":
            geoms.extend(list(g.geoms))
        else:
            geoms.append(g)
    _LAND_GEOMS = geoms
    _LAND_TREE = STRtree(geoms)
    print(f"[opendrift-mock] loaded {len(geoms)} land polygons")


_load_coastline()


def _is_land(lon: float, lat: float) -> bool:
    """Real point-in-polygon check against Natural Earth coastlines."""
    p = Point(lon, lat)
    # STRtree.query returns indices of polygons whose bounding boxes contain p;
    # we then run actual geometry .contains() on the (usually 0-2) candidates.
    for idx in _LAND_TREE.query(p):
        if _LAND_GEOMS[idx].contains(p):
            return True
    return False


def _make_frame(t, lats, lons, statuses, masses, model) -> TrajectoryFrame:
    return TrajectoryFrame(
        time=t,
        points=[
            TrajectoryPoint(
                particle_id=i,
                lat=round(lats[i], 6),
                lon=round(lons[i], 6),
                status=statuses[i],
                mass_oil=round(masses[i], 4) if model == "OpenOil" else None,
            )
            for i in range(len(lats))
        ],
    )


def _mock_simulate(req: SimulationRequest) -> SimulationResult:
    n = req.seed.number

    # ---- Seed the initial cloud --------------------------------------------
    # Two modes: polygon (uniform fill) or point (Gaussian cloud).
    if req.seed.polygon:
        # Polygon mode — uniform fill via rejection sampling on the bounding box.
        # Real OpenDrift: o.seed_within_polygon(lons=poly_lons, lats=poly_lats, ...).
        ring = req.seed.polygon[0]                         # outer ring [[lon, lat], ...]
        holes = req.seed.polygon[1:] if len(req.seed.polygon) > 1 else None
        poly_geom = ShapelyPolygon(ring, holes=holes)
        if not poly_geom.is_valid or poly_geom.is_empty:
            raise ValueError("Invalid or empty seed polygon")
        centroid = poly_geom.centroid
        seed_lon, seed_lat = centroid.x, centroid.y
        rng = random.Random(int((seed_lon + 180) * 1000 + (seed_lat + 90) * 1000))
        minx, miny, maxx, maxy = poly_geom.bounds
        init_lons, init_lats = [], []
        # Cap attempts to avoid infinite loops on bad polygons.
        max_attempts = max(n * 50, 10_000)
        attempts = 0
        while len(init_lons) < n and attempts < max_attempts:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if poly_geom.contains(Point(x, y)):
                init_lons.append(x)
                init_lats.append(y)
            attempts += 1
        if len(init_lons) < n:
            raise ValueError(f"Could only seed {len(init_lons)}/{n} particles in polygon")
    elif req.seed.lon is not None and req.seed.lat is not None:
        # Point mode — Gaussian cloud.
        seed_lon, seed_lat = req.seed.lon, req.seed.lat
        rng = random.Random(int(seed_lon * 1000 + seed_lat * 1000))
        radius_deg = req.seed.radius_m / 111_000
        cos_lat0 = math.cos(math.radians(seed_lat))
        init_lats = [seed_lat + rng.gauss(0, radius_deg) for _ in range(n)]
        init_lons = [seed_lon + rng.gauss(0, radius_deg / cos_lat0) for _ in range(n)]
    else:
        raise ValueError("Seed must include either 'polygon' or both 'lon' and 'lat'")

    output_step_h = req.run.output_step_s / 3600
    n_back = int(req.run.duration_back_hours / output_step_h)
    n_fwd = int(req.run.duration_forward_hours / output_step_h)
    dt_s = req.run.output_step_s

    evap_rate = 0.02 if req.model == "OpenOil" else 0.0

    def advect(lon, lat, sign):
        """Move one particle by one output-step; sign=+1 forward, -1 backward."""
        u_mps, v_mps = _current(lon, lat, seed_lon, seed_lat)
        cos_lat_i = math.cos(math.radians(lat))
        d_lon = sign * u_mps * dt_s / (111_000 * cos_lat_i) + rng.gauss(0, 0.0008)
        d_lat = sign * v_mps * dt_s / 111_000 + rng.gauss(0, 0.0008)
        return lon + d_lon, lat + d_lat

    # ---- Backward leg: trace particles upstream from t=0 ----
    # Backward "stranding" is non-physical for an oil spill (oil doesn't come
    # from beaches), so we don't apply the land mask going backward.
    lats = list(init_lats)
    lons = list(init_lons)
    statuses = ["active"] * n
    masses = [1.0] * n
    back_frames: list[TrajectoryFrame] = []
    for f in range(1, n_back + 1):
        for i in range(n):
            lons[i], lats[i] = advect(lons[i], lats[i], sign=-1)
        t = req.seed.time + timedelta(hours=-f * output_step_h)
        back_frames.append(_make_frame(t, lats, lons, statuses, masses, req.model))
    back_frames.reverse()  # earliest time first

    # ---- t=0 frame ----
    lats = list(init_lats)
    lons = list(init_lons)
    statuses = ["active"] * n
    masses = [1.0] * n
    t0_frame = _make_frame(req.seed.time, lats, lons, statuses, masses, req.model)

    # ---- Forward leg: forecast downstream, applying land-mask stranding ----
    fwd_frames: list[TrajectoryFrame] = []
    for f in range(1, n_fwd + 1):
        for i in range(n):
            if statuses[i] != "active":
                continue  # stranded / evaporated — frozen in place
            new_lon, new_lat = advect(lons[i], lats[i], sign=+1)
            if _is_land(new_lon, new_lat):
                # Real OpenDrift would interpolate to the coastline; for the
                # mock we just freeze at the last sea position.
                statuses[i] = "stranded"
                continue
            lons[i] = new_lon
            lats[i] = new_lat
            if req.model == "OpenOil":
                masses[i] *= (1 - evap_rate)
                if masses[i] < 0.1 and rng.random() < 0.05:
                    statuses[i] = "evaporated"
        t = req.seed.time + timedelta(hours=f * output_step_h)
        fwd_frames.append(_make_frame(t, lats, lons, statuses, masses, req.model))

    frames = back_frames + [t0_frame] + fwd_frames
    seed_frame_index = len(back_frames)  # index of the t=0 frame

    oil_budget = None
    if req.model == "OpenOil":
        oil_budget = {
            "total_kg": round(sum(masses), 2),
            "evaporated_pct": round(100 * statuses.count("evaporated") / n, 2),
            "stranded_pct": round(100 * statuses.count("stranded") / n, 2),
            "surface_pct": round(100 * statuses.count("active") / n, 2),
        }

    return SimulationResult(
        job_id="",
        model=req.model,
        seed_lon=seed_lon,
        seed_lat=seed_lat,
        seed_polygon=req.seed.polygon,
        seed_frame_index=seed_frame_index,
        frames=frames,
        oil_budget=oil_budget,
    )


async def _run_job(job_id: str):
    job = JOBS[job_id]
    job.status = JobStatus.running
    job.started_at = datetime.now(timezone.utc)
    try:
        # Fake progress ticks. Real OpenDrift run() blocks for minutes-to-hours,
        # so in production you'd run it in a separate worker process (Celery,
        # RQ, Dramatiq) and have it write progress to Redis/DB.
        for p in (0.25, 0.5, 0.75):
            await asyncio.sleep(0.4)
            job.progress = p
        result = _mock_simulate(job.request)
        result.job_id = job_id
        # In production: upload .nc to S3 and set this to a pre-signed URL.
        result.netcdf_url = f"/simulations/{job_id}/netcdf"
        RESULTS[job_id] = result
        job.progress = 1.0
        job.status = JobStatus.done
    except Exception as e:  # noqa: BLE001
        job.status = JobStatus.failed
        job.error = str(e)
    finally:
        job.finished_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenDrift Mock API", version="0.1.0")

# CORS open for local React dev. PROVIDE: tighten to your real origins in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def list_models():
    # Real OpenDrift ships many more (LarvalFish, ChemicalDrift, PlastDrift, ...).
    return {"models": [
        {"id": "OceanDrift", "description": "Generic passive tracer"},
        {"id": "OpenOil",    "description": "Oil spill with weathering / evaporation"},
        {"id": "Leeway",     "description": "Search & rescue (drifting objects)"},
    ]}


@app.get("/oil-types")
def list_oil_types():
    # PROVIDE: the real list comes from the ADIOS oil database bundled with
    # OpenDrift. Expose `opendrift.models.openoil.adios.dirjs.oils()` here.
    return {"types": [
        "GENERIC LIGHT CRUDE",
        "GENERIC MEDIUM CRUDE",
        "GENERIC HEAVY CRUDE",
        "IFO-180LS 2014",
        "IFO-380LS 2014",
        "EKOFISK BLEND 2002",
        "TROLL B 2003",
    ]}


@app.post("/simulations", status_code=202)
async def create_simulation(req: SimulationRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = SimulationJob(
        id=job_id,
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        request=req,
    )
    bg.add_task(_run_job, job_id)
    return {
        "id": job_id,
        "status": "queued",
        "status_url": f"/simulations/{job_id}",
        "result_url": f"/simulations/{job_id}/result",
    }


@app.get("/simulations")
def list_simulations():
    return [
        {"id": j.id, "status": j.status, "progress": j.progress,
         "model": j.request.model, "created_at": j.created_at}
        for j in JOBS.values()
    ]


@app.get("/simulations/{job_id}")
def get_simulation(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Simulation not found")
    return JOBS[job_id]


@app.get("/simulations/{job_id}/result")
def get_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Simulation not found")
    job = JOBS[job_id]
    if job.status != JobStatus.done:
        raise HTTPException(409, f"Simulation not ready (status: {job.status})")
    return RESULTS[job_id]


@app.delete("/simulations/{job_id}")
def delete_simulation(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Simulation not found")
    JOBS.pop(job_id, None)
    RESULTS.pop(job_id, None)
    return {"deleted": job_id}
