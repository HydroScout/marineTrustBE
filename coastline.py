"""
Coastline data loader and point-in-land test.

Natural Earth 50m land polygons (~5 MB) — accurate to ~5 km, matches OSM
tiles at zoom ~9. Downloaded once on first server startup, cached on disk.

PROVIDE: real OpenDrift uses GSHHG (sub-km accuracy) via
`reader_global_landmask`. Swap _COASTLINE_URL for the higher-resolution
`ne_10m_land.geojson` (~30 MB, ~1 km accuracy) if your zoom level needs it.
"""

import json as _json
import urllib.request
from pathlib import Path
from typing import Optional

from shapely.geometry import Point, shape
from shapely.strtree import STRtree


_COASTLINE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
_COASTLINE_FILE = Path(__file__).parent / "data" / "ne_50m_land.geojson"
_LAND_TREE: Optional[STRtree] = None
_LAND_GEOMS: list = []


def _load_coastline() -> None:
    global _LAND_TREE, _LAND_GEOMS
    _COASTLINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Download atomically (to .tmp, then rename) so a truncated transfer
    # doesn't poison the cache.
    if not _COASTLINE_FILE.exists():
        tmp = _COASTLINE_FILE.with_suffix(".tmp")
        print(f"[opendrift-mock] downloading coastline: {_COASTLINE_URL}")
        with urllib.request.urlopen(_COASTLINE_URL, timeout=60) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.rename(_COASTLINE_FILE)
        print(f"[opendrift-mock] cached at {_COASTLINE_FILE}")
    try:
        with open(_COASTLINE_FILE) as f:
            data = _json.load(f)
    except Exception:
        # Corrupt cache — wipe and let the next startup retry.
        _COASTLINE_FILE.unlink(missing_ok=True)
        raise
    geoms = []
    for feat in data["features"]:
        g = shape(feat["geometry"])
        # Polygon -> single, MultiPolygon -> split so the STRtree gets tighter
        # bounding boxes per piece (faster queries near complex coasts).
        if g.geom_type == "MultiPolygon":
            geoms.extend(list(g.geoms))
        else:
            geoms.append(g)
    _LAND_GEOMS = geoms
    _LAND_TREE = STRtree(geoms)
    print(f"[opendrift-mock] loaded {len(geoms)} land polygons")


_load_coastline()


def is_land(lon: float, lat: float) -> bool:
    """Real point-in-polygon check against Natural Earth coastlines."""
    p = Point(lon, lat)
    # STRtree.query returns indices of polygons whose bounding boxes contain p;
    # we then run actual geometry .contains() on the (usually 0-2) candidates.
    for idx in _LAND_TREE.query(p):
        if _LAND_GEOMS[idx].contains(p):
            return True
    return False
