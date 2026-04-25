"""
OpenDrift mock REST API — entry point.

Run with:
    uvicorn main:app --reload

Modules:
    schemas.py    — request/response Pydantic models
    store.py      — in-memory JOBS / RESULTS
    coastline.py  — Natural Earth land polygons + is_land()
    simulator.py  — mock simulator + async job runner
    api.py        — FastAPI app and HTTP endpoints
"""

from api import app

__all__ = ["app"]
