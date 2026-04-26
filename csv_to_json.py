import pandas as pd
import json
from pathlib import Path


def _clean_ais_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['Latitude'] != 'masked') & (df['Longitude'] != 'masked')].copy()
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.sort_values('Timestamp')


def _to_route_dict(df: pd.DataFrame) -> dict:
    return {
        # Strict ISO strings for easy JS Date() parsing
        "timestamps": df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
        # MapLibre requires coordinates in [Longitude, Latitude] order
        "coordinates": df[['Longitude', 'Latitude']].values.tolist(),
        "speeds": df['Speed'].tolist(),
        "courses": df['Course'].tolist(),
    }


def build_ship_route(routes_dir: Path, ship_id: str, date_str: str) -> dict | None:
    """Read every CSV in `routes_dir`, filter to `ship_id` on `date_str` (YYYY-MM-DD),
    and return the route dict in ship_track.json shape (without `collisions`).
    Returns None when no rows match."""
    target = pd.to_datetime(date_str).date()
    csv_paths = sorted(Path(routes_dir).glob('*.csv'))
    if not csv_paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df = df[df['id'].astype(str) == str(ship_id)]
    if df.empty:
        return None
    df = _clean_ais_dataframe(df)
    df = df[df['Timestamp'].dt.date == target]
    if df.empty:
        return None
    return _to_route_dict(df)


def process_ais_data():
    # 1. Read the CSV
    file_path = 'MarineTraffic_Vessel_positions_Export_2026-04-25.csv'
    df = pd.read_csv(file_path)

    # 2-4. Clean, parse timestamps, sort
    df = _clean_ais_dataframe(df)

    # 5. Format the output data optimally for our React frontend
    route_data = _to_route_dict(df)

    # 6. Save to a clean JSON file
    output_path = 'ship_track.json'
    with open(output_path, 'w') as f:
        json.dump(route_data, f)

    print(f"Successfully processed {len(df)} valid tracking points.")
    print(f"Start time: {route_data['timestamps'][0]}")
    print(f"End time: {route_data['timestamps'][-1]}")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    process_ais_data()
