"""
Convert MarineTraffic AIS CSV exports into clean route JSON.

Two ways to use this module:

1. As a library (called from api.py):
       from csv_to_json import csv_to_route
       route = csv_to_route(ship_id=1, target_date=date(2026, 4, 18))

2. As a CLI batch job — group every CSV in `data/routes/` by `id` and write
   one `data/ship_tracks/<id>.json` per ship:
       python csv_to_json.py

CSV format (MarineTraffic export with leading id column):
    id,Timestamp,Source,Speed,Course,Latitude,Longitude
    "masked" cells (Satellite-only fixes) become None and are dropped from
    the output (every output index has a real position).
"""

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


ROUTES_DIR = Path(__file__).parent / "data" / "routes"
SHIP_TRACKS_DIR = Path(__file__).parent / "data" / "ship_tracks"

def process_ais_data():
    # 1. Read the CSV
    file_path = 'MarineTraffic_Vessel_positions_Export_2026-04-25.csv'
    df = pd.read_csv(file_path)

    # 2. Filter out rows where Latitude or Longitude is 'masked' or NaN
    df = df[(df['Latitude'] != 'masked') & (df['Longitude'] != 'masked')].copy()
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # 3. Convert coordinates to standard floats
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    # 4. Parse timestamps and sort them strictly chronologically
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    # 5. Format the output data optimally for our React frontend
    route_data = {
        # Format as strict ISO strings for easy JS Date() parsing
        "timestamps": df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
        # MapLibre requires coordinates in [Longitude, Latitude] order
        "coordinates": df[['Longitude', 'Latitude']].values.tolist(), 
        "speeds": df['Speed'].tolist(),
        "courses": df['Course'].tolist()
    }

    # 6. Save to a clean JSON file
    output_path = 'ship_track.json'
    with open(output_path, 'w') as f:
        json.dump(route_data, f)

    print(f"Successfully processed {len(df)} valid tracking points.")
    print(f"Start time: {route_data['timestamps'][0]}")
    print(f"End time: {route_data['timestamps'][-1]}")
    print(f"Saved to {output_path}")

def _maybe(v: str) -> Optional[float]:
    """Numeric-or-None: 'masked' / empty -> None, else float."""
    v = (v or "").strip()
    return None if v.lower() == "masked" or v == "" else float(v)


def csv_to_route(
    ship_id: int,
    target_date: Optional[date] = None,
    routes_dir: Path = ROUTES_DIR,
) -> dict:
    """
    Read every CSV in `routes_dir`, keep rows whose `id` column equals
    `ship_id`, drop masked positions, sort by time, and return a column-
    oriented dict ready for JSON serialization:

        {
          "id": 1,
          "timestamps":  ["2026-04-18T12:03:38Z", ...],   # UTC ISO 8601
          "coordinates": [[lon, lat], ...],               # GeoJSON order
          "speeds":      [10.0, ...],                     # knots, None = masked
          "courses":     [357.0, ...],                    # degrees, None = masked
        }

    `target_date`, if given, filters to that single UTC calendar date.
    Empty arrays when the ship has no matching fixes.
    """
    rows: list[dict] = []
    if routes_dir.exists():
        for csv_path in sorted(routes_dir.glob("*.csv")):
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if "id" not in (reader.fieldnames or []):
                    continue  # legacy single-ship CSV with no id column
                for r in reader:
                    try:
                        if int(r["id"]) != ship_id:
                            continue
                    except (TypeError, ValueError):
                        continue
                    lat = _maybe(r.get("Latitude", ""))
                    lon = _maybe(r.get("Longitude", ""))
                    if lat is None or lon is None:
                        continue  # masked position — skip
                    try:
                        t = (datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S")
                                     .replace(tzinfo=timezone.utc))
                    except (KeyError, ValueError):
                        continue  # malformed timestamp — skip
                    rows.append({
                        "time": t,
                        "lon": lon,
                        "lat": lat,
                        "speed": _maybe(r.get("Speed", "")),
                        "course": _maybe(r.get("Course", "")),
                    })

    rows.sort(key=lambda r: r["time"])
    if target_date is not None:
        rows = [r for r in rows if r["time"].astimezone(timezone.utc).date() == target_date]

    return {
        "id": ship_id,
        "timestamps":  [r["time"].strftime("%Y-%m-%dT%H:%M:%SZ") for r in rows],
        "coordinates": [[r["lon"], r["lat"]] for r in rows],
        "speeds":      [r["speed"] for r in rows],
        "courses":     [r["course"] for r in rows],
    }


def _discover_ids(routes_dir: Path = ROUTES_DIR) -> set[int]:
    """Every distinct ship id seen in any CSV under `routes_dir`."""
    ids: set[int] = set()
    if not routes_dir.exists():
        return ids
    for csv_path in sorted(routes_dir.glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "id" not in (reader.fieldnames or []):
                continue
            for r in reader:
                try:
                    ids.add(int(r["id"]))
                except (TypeError, ValueError):
                    pass
    return ids


def main() -> None:
    """CLI batch mode: write `data/ship_tracks/<id>.json` for each ship."""
    SHIP_TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    ids = _discover_ids()
    if not ids:
        print(f"No ids found in {ROUTES_DIR}")
        return
    for ship_id in sorted(ids):
        route = csv_to_route(ship_id)
        out_path = SHIP_TRACKS_DIR / f"{ship_id}.json"
        with open(out_path, "w") as f:
            json.dump(route, f)
        print(f"  wrote {out_path}  ({len(route['timestamps'])} fixes)")


if __name__ == "__main__":
    main()
