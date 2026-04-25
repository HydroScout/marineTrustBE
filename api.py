"""HTTP API."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import RunConfig, Seed, SimulationRequest, SimulationResult
from simulator import mock_simulate


app = FastAPI(title="MarineTrust API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


SPILLS_PATH = Path(__file__).parent / "data" / "spills" / "spills.json"
SPILL_WINDOW_HOURS = 60
SPILL_OUTPUT_STEP_S = 1800  # 30 min frames

# Map spill `type` from spills.json to (OpenDrift model, ADIOS oil_type).
SPILL_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "Oil": ("OpenOil", "GENERIC MEDIUM CRUDE"),
}


@app.get("/ships/{ship_id}/{date}")
def get_ship_on_date(ship_id: str, date: str):
    return {}


@app.get("/ships/{ship_id}")
def get_ship(ship_id: str):
    return {}


@app.get("/flaggedShips")
def get_flagged_ships():
    return {}

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
