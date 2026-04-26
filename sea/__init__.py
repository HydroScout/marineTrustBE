"""Waze for Sea — pre-trip risk prediction (additive module).

Provides POST /api/sea/routes/analyze. Existing endpoints are untouched.
"""

from .routes import router

__all__ = ["router"]
