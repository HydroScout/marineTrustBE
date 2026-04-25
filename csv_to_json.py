import pandas as pd
import json

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

if __name__ == '__main__':
    process_ais_data()