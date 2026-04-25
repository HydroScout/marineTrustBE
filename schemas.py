"""
Request and response schemas for the OpenDrift mock REST API.

These mirror OpenDrift's seed_elements() / run() inputs. When wiring real
OpenDrift, fields map 1:1 to the library calls.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
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
# Ship route + collision response schemas (used by api.py).
# ---------------------------------------------------------------------------

class ShipMeta(BaseModel):
    """Returned by GET /ships/{ship_id} — passthrough of the registry entry."""
    id: int
    name: str
    type: str


class ShipRouteResponse(BaseModel):
    """
    Returned by GET /ships/{ship_id}/{date}.

    Same column-oriented layout as the standalone `ship_track.json` produced
    by `csv_to_json.main()`, with the extra `collisions` array. The i-th
    element of every list refers to the same fix, so the frontend can index
    by frame for animation:

        timestamps[i]   ISO 8601 UTC string
        coordinates[i]  [lon, lat] (GeoJSON convention)
        speeds[i]       knots; None when the source row was masked
        courses[i]      degrees true; None when masked
        collisions[i]   True iff that fix lies inside any spill polygon

    Rows whose lat/lon were "masked" are dropped — every output entry has
    a real position.
    """
    timestamps: list[str]
    coordinates: list[list[float]]
    speeds: list[Optional[float]]
    courses: list[Optional[float]]
    collisions: list[bool]


class FlaggedShipsResponse(BaseModel):
    ids: list[int]
