"""
HTTP API for the OpenDrift mock service.

Job-based async pattern:
  POST /simulations             -> 202 + job id, runs in background
  GET  /simulations/{id}        -> status / progress
  GET  /simulations/{id}/result -> trajectory frames as JSON (for the map)
"""

import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import JobStatus, SimulationJob, SimulationRequest
from simulator import run_job
from store import JOBS, RESULTS


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
    bg.add_task(run_job, job_id)
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
