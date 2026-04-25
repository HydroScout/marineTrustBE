"""
Mock simulator and async job runner.

REPLACE WITH REAL OPENDRIFT:

  from opendrift.models.openoil import OpenOil
  o = OpenOil(loglevel=20)
  o.add_readers_from_list([req.forcing.currents, req.forcing.wind])
  o.seed_elements(
      lon=req.seed.lon, lat=req.seed.lat,
      radius=req.seed.radius_m, number=req.seed.number,
      time=req.seed.time, z=req.seed.z,
      oil_type=req.seed.oil_type,
  )
  o.run(duration=timedelta(hours=req.run.duration_hours),
        time_step=req.run.time_step_s,
        time_step_output=req.run.output_step_s,
        outfile=f"/data/{job_id}.nc")
  # then read o.history (xarray.Dataset) into TrajectoryFrames
"""

import asyncio
import math
import random
from datetime import datetime, timedelta, timezone

from shapely.geometry import Point, Polygon as ShapelyPolygon

from coastline import is_land
from schemas import (
    JobStatus,
    SimulationRequest,
    SimulationResult,
    TrajectoryFrame,
    TrajectoryPoint,
)
from store import JOBS, RESULTS


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


def mock_simulate(req: SimulationRequest) -> SimulationResult:
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
            if is_land(new_lon, new_lat):
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


async def run_job(job_id: str):
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
        result = mock_simulate(job.request)
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
