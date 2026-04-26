"""Mock OpenDrift — deterministic drift of a polygon over time.

Real OpenDrift uses currents/wind/wave fields. Here we approximate with a
constant wind vector + slow polygon expansion as the patch ages.
"""

from __future__ import annotations

from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate
from shapely.geometry import Polygon


# Drift constants (degrees per hour). South-southwest drift — represents oil
# being pushed south by a steady north wind, with a small westerly component
# from the prevailing surface current. Tuned so that on a 12-15 h trip the
# patch visibly migrates into typical Black-Sea coastal routes (Sevastopol →
# Novorossiysk passes ~33 km south of the original spill; the drifted +
# expanded patch reaches the route around hour 12).
WIND_DX_DEG_PER_H = -0.005
WIND_DY_DEG_PER_H = -0.025

# Patch grows ~5% per hour (wind/turbulence spreading).
GROWTH_PER_HOUR = 0.05


def predict_patch(polygon: Polygon, t_hours: float) -> Polygon:
    """Return predicted polygon at t_hours from now.

    Steps:
      1. translate by (wind_dx * t, wind_dy * t)
      2. scale around centroid by (1 + growth * t)
    """
    if t_hours <= 0:
        return polygon

    shifted = shp_translate(
        polygon,
        xoff=WIND_DX_DEG_PER_H * t_hours,
        yoff=WIND_DY_DEG_PER_H * t_hours,
    )
    factor = 1.0 + GROWTH_PER_HOUR * t_hours
    return shp_scale(shifted, xfact=factor, yfact=factor, origin="centroid")


def patch_to_geojson_coords(polygon: Polygon) -> list[list[list[float]]]:
    """Convert a shapely Polygon back to GeoJSON-style coords [[[lon, lat], ...]]."""
    if polygon.is_empty:
        return []
    exterior = list(polygon.exterior.coords)
    return [[[float(x), float(y)] for x, y in exterior]]
