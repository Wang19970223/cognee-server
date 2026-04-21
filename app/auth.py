"""
app/auth.py
─────────────────────────────────────────────────────────────────────────────
FastAPI dependency for API Key authentication.

Keys and permission levels are read from settings.cognee_api_keys:
    COGNEE_API_KEYS=key1:read,key2:write,key3:admin

Permission hierarchy:  admin > write > read
Usage:
    @router.post("/index")
    async def index(perm: str = Depends(require_write)):
        ...
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_PERM_RANK = {"read": 0, "write": 1, "admin": 2}


def _get_permission(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required",
        )
    keys = get_settings().parsed_api_keys()
    level = keys.get(api_key)
    if level is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return level


def _require_level(min_level: str):
    """Return a dependency that ensures the caller has at least `min_level`."""

    def _dep(permission: str = Depends(_get_permission)) -> str:
        if _PERM_RANK.get(permission, -1) < _PERM_RANK[min_level]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This operation requires '{min_level}' permission (current: '{permission}')",
            )
        return permission

    return _dep


require_read = _require_level("read")
require_write = _require_level("write")
require_admin = _require_level("admin")
