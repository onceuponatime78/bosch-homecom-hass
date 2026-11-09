"""Observe OAuth responses before the library discards authentication errors."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import logging
import time
from typing import Any

from aiohttp import ClientError, ClientResponse, ClientSession
from homecom_alt.const import OAUTH_DOMAIN, OAUTH_ENDPOINT

_LOGGER = logging.getLogger(__name__)
_ERROR_CODES = {
    "invalid_request",
    "invalid_client",
    "invalid_grant",
    "unauthorized_client",
    "unsupported_grant_type",
    "invalid_scope",
    "access_denied",
    "server_error",
    "temporarily_unavailable",
}
_REPEAT_INTERVAL = 900


def _fingerprint(value: Any) -> str | None:
    """Compare opaque credentials without retaining them in diagnostics."""
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


class AuthDiagnosticSession:
    """Delegate to HA's session, preserving its ownership and request semantics.

    aiohttp's per-request status callback observes a response before the
    library's raise_for_status=True can discard it. Returning the original
    request context manager preserves both await and async-with usage.
    """

    def __init__(self, session: ClientSession, entry_id: str) -> None:
        """Initialize per-entry diagnostics, never a new HTTP session."""
        self._session = session
        self._entry_id = entry_id
        self._last_success: str | None = None
        self._last_issued: str | None = None
        self._last_failure: tuple | None = None
        self._last_warning = 0.0

    def __getattr__(self, name: str) -> Any:
        """Leave all other client-session features untouched."""
        return getattr(self._session, name)

    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        """Attach diagnostics only to the known OAuth token endpoint."""
        if (
            str(url) == OAUTH_DOMAIN + OAUTH_ENDPOINT
            and kwargs.get("raise_for_status") is True
        ):
            data = kwargs.get("data")
            used = (
                _fingerprint(data.get("refresh_token"))
                if isinstance(data, dict)
                else None
            )

            async def check(response: ClientResponse) -> None:
                await self._observe(response, used)
                response.raise_for_status()

            kwargs["raise_for_status"] = check
        return self._session.request(method, url, **kwargs)

    async def _observe(self, response: ClientResponse, used: str | None) -> None:
        """Log fixed classifications, never raw bodies, descriptions or headers."""
        payload = None
        try:
            payload = await response.json()
        except (ValueError, ClientError, TimeoutError):
            # Diagnostics must not change how the library handles the response.
            pass
        if response.status == 200:
            self._last_success = datetime.now(UTC).isoformat()
            self._last_issued = (
                _fingerprint(payload.get("refresh_token"))
                if isinstance(payload, dict)
                else None
            )
            self._last_failure = None
            _LOGGER.info(
                "OAuth token exchange succeeded for entry %s; refresh_token_rotated=%s",
                self._entry_id,
                self._last_issued is not None and self._last_issued != used,
            )
            return
        if response.status < 400:
            return
        code = payload.get("error") if isinstance(payload, dict) else None
        code = (
            code
            if isinstance(code, str) and code in _ERROR_CODES
            else "unreported_or_unrecognized"
        )
        description = (
            payload.get("error_description") if isinstance(payload, dict) else None
        )
        hint = "unspecified"
        if isinstance(description, str):
            for word in ("expired", "revoked", "reused"):
                if word in description.lower():
                    hint = word
                    break
        matches = used == self._last_issued if used and self._last_issued else None
        signature = (response.status, code, used)
        now = time.monotonic()
        if (
            signature != self._last_failure
            or now - self._last_warning >= _REPEAT_INTERVAL
        ):
            _LOGGER.warning(
                "OAuth token exchange failed for entry %s: HTTP %s, oauth_error=%s, hint=%s, "
                "refresh_matches_last_issued=%s, last_success=%s. "
                "Response bodies and credentials are omitted",
                self._entry_id,
                response.status,
                code,
                hint,
                matches,
                self._last_success or "not observed since setup",
            )
            self._last_warning = now
            self._last_failure = signature
