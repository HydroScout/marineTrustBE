"""Regenerate data/routes/sample-ship.csv with a natural straight-line track
that crosses the simulated spill cloud during 2026-04-18.

Bearing/speed picked so the ship enters the cloud's convex hull around 11:00
and exits around 12:00. AIS-style 6-decimal coords with mild observation
jitter so the line doesn't look mathematically perfect."""

import csv
import math
import random
from pathlib import Path

random.seed(7)

START_LON = -88.80
START_LAT = 30.54
BASE_COURSE = 175.0       # slight east of south
BASE_SPEED_KNOTS = 5.0
N_STEPS = 48              # 30-min cadence over 24 h
STEP_HOURS = 0.5

# Mean displacement per 30-min step at base course/speed.
half_hour_km = BASE_SPEED_KNOTS * 1.852 * STEP_HOURS
cr = math.radians(BASE_COURSE)
mean_lat = 29.5  # rough midpoint, sets the lon scale factor
D_LAT = (half_hour_km * math.cos(cr)) / 111.0
D_LON = (half_hour_km * math.sin(cr)) / (111.0 * math.cos(math.radians(mean_lat)))

rows = []
for i in range(N_STEPS):
    lat = START_LAT + i * D_LAT + random.gauss(0, 0.00025)
    lon = START_LON + i * D_LON + random.gauss(0, 0.00025)
    speed = round(max(4.6, min(5.4, BASE_SPEED_KNOTS + random.gauss(0, 0.18))), 1)
    course = int(round(max(172, min(179, BASE_COURSE + random.gauss(0, 1.2)))))
    hour, minute = divmod(i * 30, 60)
    timestamp = f"2026-04-18 {hour:02d}:{minute:02d}:00"
    rows.append((2, timestamp, "Roaming", speed, course, round(lat, 6), round(lon, 6)))

# Match existing file: descending timestamp order
rows.reverse()

out = Path(__file__).resolve().parent.parent / "data" / "routes" / "sample-ship.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "Timestamp", "Source", "Speed", "Course", "Latitude", "Longitude"])
    w.writerows(rows)
print(f"Wrote {len(rows)} rows to {out}")
