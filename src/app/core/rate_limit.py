"""Shared rate limiter instance.

Split into its own module (rather than living in main.py) so route modules
(e.g. `app/api/routes/chat.py`) can import `limiter` to decorate their
endpoints without importing `main` itself — `main.py` imports the routers,
so a route importing back from `main` would be a circular import.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
