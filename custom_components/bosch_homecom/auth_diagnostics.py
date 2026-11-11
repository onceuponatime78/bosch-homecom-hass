"""Observe OAuth responses before the library discards authentication errors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import logging
import time
from typing import Any
from uuid import uuid4

from aiohttp import (
    ClientConnectorDNSError,
    ClientConnectorError,
    ClientError,
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    ConnectionTimeoutError,
    ContentTypeError,
    SocketTimeoutError,
)
from homecom_alt import decode_jwt_exp
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


@dataclass
class _Attempt:
    """Non-secret context for one exchange; token digests never leave memory."""

    number: int
    used: str | None
    expires_at: datetime | None
    timeout: tuple
    started: float | None = None
    expires_in: float | None = None
    status: int | None = None
    overlap: bool = False


class _ObservedRequest:
    """Preserve await/context-manager behavior while observing transport errors."""

    def __init__(self, owner, attempt: _Attempt, request) -> None:
        self._owner = owner
        self._attempt = attempt
        self._request = request

    def __await__(self):
        return self._owner._run(self._attempt, self._request).__await__()

    async def __aenter__(self):
        return await self._owner._run(self._attempt, self._request.__aenter__())

    async def __aexit__(self, *args):
        return await self._request.__aexit__(*args)


class AuthDiagnosticSession:
    """Delegate to HA's session, preserving its ownership and request semantics.

    aiohttp's per-request status callback observes a response before the
    library's raise_for_status=True can discard it. The OAuth-only wrapper
    also observes transport failures and preserves await and async-with usage.
    """

    def __init__(self, session: ClientSession, entry_id: str) -> None:
        """Initialize per-entry diagnostics, never a new HTTP session."""
        self._session = session
        self._entry_id = entry_id
        self._last_success: str | None = None
        self._last_issued: str | None = None
        self._last_failure: tuple | None = None
        self._last_warning = 0.0
        self._session_id = uuid4().hex[:8]
        self._sequence = 0
        self._active: dict[int, _Attempt] = {}
        self._uncertain: tuple[str | None, int, float] | None = None

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
            headers = kwargs.get("headers") or {}
            authorization = headers.get("Authorization")
            expires_at = (
                decode_jwt_exp(authorization.removeprefix("Bearer "))
                if isinstance(authorization, str)
                else None
            )
            timeout = kwargs.get("timeout", self._session.timeout)
            limits = (
                (
                    timeout.total,
                    timeout.connect,
                    timeout.sock_connect,
                    timeout.sock_read,
                )
                if isinstance(timeout, ClientTimeout)
                else (
                    timeout if isinstance(timeout, (int, float)) else None,
                    None,
                    None,
                    None,
                )
            )
            self._sequence += 1
            attempt = _Attempt(self._sequence, used, expires_at, limits)

            async def check(response: ClientResponse) -> None:
                attempt.status = response.status
                await self._observe(response, attempt)
                response.raise_for_status()

            kwargs["raise_for_status"] = check
            return _ObservedRequest(
                self, attempt, self._session.request(method, url, **kwargs)
            )
        return self._session.request(method, url, **kwargs)

    async def _run(self, attempt: _Attempt, request) -> Any:
        """Measure the actual execution rather than context-manager creation."""
        attempt.started = time.monotonic()
        if attempt.expires_at is not None:
            attempt.expires_in = round(
                (attempt.expires_at - datetime.now(UTC)).total_seconds(), 3
            )
        if self._active:
            attempt.overlap = True
            same_refresh = any(
                attempt.used is not None and other.used == attempt.used
                for other in self._active.values()
            )
            for other in self._active.values():
                other.overlap = True
            _LOGGER.warning(
                "Overlapping OAuth exchanges: entry=%s session=%s request=%s "
                "active_requests=%s same_refresh_token=%s",
                self._entry_id,
                self._session_id,
                attempt.number,
                tuple(self._active),
                same_refresh,
            )
        self._active[attempt.number] = attempt
        try:
            return await request
        except ClientResponseError:
            # Already logged with the OAuth error code by the status callback.
            raise
        except (ClientError, TimeoutError, asyncio.CancelledError) as error:
            self._transport_failure(attempt, error)
            raise
        finally:
            self._active.pop(attempt.number, None)

    def _timing(self, attempt: _Attempt) -> str:
        """Format only typed metadata, never an exception or request repr."""
        elapsed = (
            round(time.monotonic() - attempt.started, 3)
            if attempt.started is not None
            else None
        )
        return (
            f"session={self._session_id} request={attempt.number} elapsed_s={elapsed} "
            f"http_status={attempt.status} access_expires_in_s_at_start={attempt.expires_in} "
            f"overlap_observed={attempt.overlap} "
            f"timeout_total_connect_sock_connect_sock_read_s={attempt.timeout}"
        )

    def _transport_failure(self, attempt: _Attempt, error: BaseException) -> None:
        """Distinguish failure phases; a missing response is not proof of no rotation."""
        if isinstance(error, ClientConnectorDNSError):
            phase = "dns"
        elif isinstance(error, (ClientConnectorError, ConnectionTimeoutError)):
            phase = "connect"
        elif attempt.status is not None:
            phase = "response_body"
        elif isinstance(error, SocketTimeoutError):
            phase = "read"
        elif isinstance(error, asyncio.CancelledError):
            phase = "cancelled"
        else:
            phase = "unknown"
        uncertain = phase not in ("dns", "connect") and (
            attempt.status is None or attempt.status < 400
        )
        if uncertain:
            self._uncertain = (attempt.used, attempt.number, time.monotonic())
        os_error = getattr(error, "os_error", None)
        errno = os_error.errno if isinstance(os_error, OSError) else None
        cause = error.__cause__
        _LOGGER.warning(
            "OAuth transport failure for entry %s: exception=%s cause=%s phase=%s "
            "os_errno=%s server_processing_uncertain=%s; %s. "
            "Exception messages, URLs and credentials are omitted",
            self._entry_id,
            type(error).__name__,
            type(cause).__name__ if cause else None,
            phase,
            errno,
            uncertain,
            self._timing(attempt),
        )

    async def _observe(self, response: ClientResponse, attempt: _Attempt) -> None:
        """Log fixed classifications, never raw bodies, descriptions or headers."""
        payload = None
        try:
            payload = await response.json()
        except (ValueError, ContentTypeError):
            pass
        except (ClientError, TimeoutError) as error:
            if response.status < 400:
                # Never record success if a rotated token could not be read.
                raise
            self._transport_failure(attempt, error)
        if response.status == 200:
            if not isinstance(payload, dict) or not all(
                isinstance(payload.get(key), str) and payload[key]
                for key in ("access_token", "refresh_token")
            ):
                _LOGGER.warning(
                    "OAuth token response for entry %s was HTTP 200 but did not "
                    "contain a readable token pair; %s. Response body omitted",
                    self._entry_id,
                    self._timing(attempt),
                )
                return
            self._last_success = datetime.now(UTC).isoformat()
            self._last_issued = (
                _fingerprint(payload.get("refresh_token"))
                if isinstance(payload, dict)
                else None
            )
            self._last_failure = None
            self._uncertain = None
            _LOGGER.info(
                "OAuth token exchange succeeded for entry %s; refresh_token_rotated=%s; %s",
                self._entry_id,
                self._last_issued is not None and self._last_issued != attempt.used,
                self._timing(attempt),
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
        matches = (
            attempt.used == self._last_issued
            if attempt.used and self._last_issued
            else None
        )
        signature = (response.status, code, attempt.used)
        now = time.monotonic()
        if (
            signature != self._last_failure
            or now - self._last_warning >= _REPEAT_INTERVAL
        ):
            _LOGGER.warning(
                "OAuth token exchange failed for entry %s: HTTP %s, oauth_error=%s, hint=%s, "
                "refresh_matches_last_issued=%s, last_success=%s; %s; "
                "same_refresh_after_uncertain_request=%s uncertain_request=%s "
                "uncertain_age_s=%s. "
                "Response bodies and credentials are omitted",
                self._entry_id,
                response.status,
                code,
                hint,
                matches,
                self._last_success or "not observed since setup",
                self._timing(attempt),
                self._uncertain is not None
                and attempt.used is not None
                and self._uncertain[0] == attempt.used,
                self._uncertain[1] if self._uncertain else None,
                round(now - self._uncertain[2], 3) if self._uncertain else None,
            )
            self._last_warning = now
            self._last_failure = signature
