"""HTTP API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="MarineTrust API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ships/{ship_id}/{date}")
def get_ship_on_date(ship_id: str, date: str):
    return {}


@app.get("/ships/{ship_id}")
def get_ship(ship_id: str):
    return {}


@app.get("/flaggedShips")
def get_flagged_ships():
    return {}


@app.get("/spills")
def get_spills():
    return {}
