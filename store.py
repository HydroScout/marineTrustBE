"""
In-memory job and result store.

PROVIDE: replace with Postgres + S3 for production.
"""

from schemas import SimulationJob, SimulationResult


JOBS: dict[str, SimulationJob] = {}
RESULTS: dict[str, SimulationResult] = {}
