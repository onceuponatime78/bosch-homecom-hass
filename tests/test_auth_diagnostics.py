"""OAuth diagnostics must survive library error handling without exposing secrets."""

import asyncio
from datetime import UTC, datetime, timedelta
import logging
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import (
    ClientConnectorDNSError,
    ClientResponseError,
    ClientTimeout,
    ConnectionTimeoutError,
    SocketTimeoutError,
)
from homecom_alt import AuthFailedError, ConnectionOptions, HomeComAlt
from homecom_alt.const import OAUTH_DOMAIN, OAUTH_ENDPOINT
import jwt
import pytest

from custom_components.bosch_homecom.auth_diagnostics import AuthDiagnosticSession

TOKEN_URL = OAUTH_DOMAIN + OAUTH_ENDPOINT


def _response(status, payload):
    response = Mock(status=status)
    response.json = AsyncMock(return_value=payload)
    if status >= 400:
        response.raise_for_status.side_effect = ClientResponseError(
            Mock(), (), status=status
        )
    return response


@pytest.mark.parametrize("status", [400, 401])
async def test_oauth_rejections_logged_before_library_discards_response(caplog, status):
    """The real library still raises its normal auth error, with safe diagnostics."""
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(
        status,
        {
            "error": "invalid_grant",
            "error_description": "refresh-secret was revoked; access-secret",
            "refresh_token": "response-secret",
        },
    )

    async def request(method, url, **kwargs):
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)
    client = HomeComAlt(
        wrapped,
        ConnectionOptions(token="access-secret", refresh_token="refresh-secret"),
        True,
    )
    with pytest.raises(AuthFailedError):
        await client.get_token(force=True)

    assert f"HTTP {status}" in caplog.text
    assert "oauth_error=invalid_grant" in caplog.text
    assert "hint=revoked" in caplog.text
    for secret in ("refresh-secret", "access-secret", "response-secret"):
        assert secret not in caplog.text


@pytest.mark.parametrize("use_stale_refresh", [False, True])
async def test_successful_rotation_is_correlated_and_response_remains_readable(
    caplog, use_stale_refresh
):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(
        200,
        {"access_token": "new-access-secret", "refresh_token": "new-refresh-secret"},
    )

    async def request(method, url, **kwargs):
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)
    client = HomeComAlt(
        wrapped,
        ConnectionOptions(
            token="old-access-secret", refresh_token="old-refresh-secret"
        ),
        True,
    )
    assert await client.get_token(force=True)
    assert client.refresh_token == "new-refresh-secret"

    if use_stale_refresh:
        client.refresh_token = "old-refresh-secret"
    response = _response(400, {"error": "invalid_grant"})
    with pytest.raises(AuthFailedError):
        await client.get_token(force=True)
    assert f"refresh_matches_last_issued={not use_stale_refresh}" in caplog.text
    assert "last_success=not observed" not in caplog.text
    assert "new-refresh-secret" not in caplog.text


async def test_warning_deduplication_and_unknown_error_never_leak_values(caplog):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(
        400,
        {"error": "secret-in-error-code", "error_description": "secret-description"},
    )
    wrapped.request(
        "post", TOKEN_URL, raise_for_status=True, data={"refresh_token": "secret-token"}
    )
    callback = raw.request.call_args.kwargs["raise_for_status"]
    for _ in range(2):
        with pytest.raises(ClientResponseError):
            await callback(response)
    assert caplog.text.count("OAuth token exchange failed") == 1
    assert "oauth_error=unreported_or_unrecognized" in caplog.text
    assert "secret" not in caplog.text


async def test_non_json_error_still_raises_original_http_error(caplog):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(500, None)
    response.json.side_effect = ValueError("secret HTML response")
    wrapped.request("post", TOKEN_URL, raise_for_status=True)
    with pytest.raises(ClientResponseError) as raised:
        await raw.request.call_args.kwargs["raise_for_status"](response)
    assert raised.value.status == 500
    assert "secret HTML response" not in caplog.text


def test_other_requests_and_session_features_are_delegated_unchanged():
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    original_context_manager = raw.request.return_value
    assert (
        wrapped.request("get", "https://example.com", raise_for_status=True)
        is original_context_manager
    )
    raw.request.assert_called_once_with(
        "get", "https://example.com", raise_for_status=True
    )
    assert wrapped.get is raw.get
    assert wrapped.closed is raw.closed


@pytest.mark.parametrize(
    ("error", "phase", "uncertain"),
    [
        (TimeoutError("secret-timeout"), "unknown", True),
        (SocketTimeoutError("secret-read"), "read", True),
        (ConnectionTimeoutError("secret-connect"), "connect", False),
        (ClientConnectorDNSError(Mock(), OSError(8, "secret-dns")), "dns", False),
    ],
)
async def test_transport_failure_records_original_type_duration_and_deadline(
    caplog, error, phase, uncertain
):
    """Observe errors before homecom_alt wraps them in NotRespondingError."""
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    raw.request = Mock(side_effect=AsyncMock(side_effect=error))
    with patch(
        "custom_components.bosch_homecom.auth_diagnostics.time",
        Mock(monotonic=Mock(side_effect=[100.0, 110.0, 110.0])),
    ):
        with pytest.raises(type(error)) as raised:
            await wrapped.request(
                "post",
                TOKEN_URL,
                raise_for_status=True,
                data={"refresh_token": "refresh-secret"},
                timeout=ClientTimeout(total=10, connect=3, sock_read=7),
            )
    assert raised.value is error
    assert f"exception={type(error).__name__}" in caplog.text
    assert f"phase={phase}" in caplog.text
    assert f"server_processing_uncertain={uncertain}" in caplog.text
    assert "elapsed_s=10.0" in caplog.text
    assert "(10, 3, None, 7)" in caplog.text
    assert "secret" not in caplog.text


async def test_invalid_grant_correlates_with_previous_unknown_refresh_outcome(caplog):
    """An invalid_grant after a timeout identifies the specific prior request."""
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")

    async def request(method, url, **kwargs):
        if raw.request.call_count == 1:
            raise TimeoutError("secret")
        response = _response(400, {"error": "invalid_grant"})
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)
    for error in (TimeoutError, ClientResponseError):
        with pytest.raises(error):
            await wrapped.request(
                "post",
                TOKEN_URL,
                raise_for_status=True,
                data={"refresh_token": "secret"},
            )
    assert (
        "same_refresh_after_uncertain_request=True uncertain_request=1" in caplog.text
    )
    assert "request=2" in caplog.text
    assert "secret" not in caplog.text


async def test_overlapping_refreshes_are_reported_and_cancelled_request_is_cleaned_up(
    caplog,
):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    started = asyncio.Event()
    finish = asyncio.Event()

    async def request(method, url, **kwargs):
        started.set()
        await finish.wait()
        response = _response(
            200, {"access_token": "access-secret", "refresh_token": "new-secret"}
        )
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)

    async def run():
        return await wrapped.request(
            "post", TOKEN_URL, raise_for_status=True, data={"refresh_token": "secret"}
        )

    first = asyncio.create_task(run())
    await started.wait()
    # Start the second exchange while the first is still blocked.
    second = asyncio.create_task(run())
    await asyncio.sleep(0)
    first.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await first
    finally:
        finish.set()
        await second
    assert "Overlapping OAuth exchanges" in caplog.text
    assert "same_refresh_token=True" in caplog.text
    assert "exception=CancelledError" in caplog.text
    assert "phase=cancelled" in caplog.text
    assert not wrapped._active
    assert "secret" not in caplog.text


async def test_response_body_timeout_is_not_recorded_as_success(caplog):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(200, {})
    response.json.side_effect = SocketTimeoutError("secret-body")

    async def request(method, url, **kwargs):
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)
    with pytest.raises(SocketTimeoutError):
        await wrapped.request("post", TOKEN_URL, raise_for_status=True)
    assert "phase=response_body" in caplog.text
    assert "http_status=200" in caplog.text
    assert "server_processing_uncertain=True" in caplog.text
    assert wrapped._last_success is None
    assert "secret-body" not in caplog.text


async def test_error_body_timeout_keeps_original_http_rejection(caplog):
    """Reading diagnostic details must not turn a known HTTP 400 into a timeout."""
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(400, {})
    response.json.side_effect = SocketTimeoutError("secret-body")

    async def request(method, url, **kwargs):
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)
    with pytest.raises(ClientResponseError) as raised:
        await wrapped.request("post", TOKEN_URL, raise_for_status=True)
    assert raised.value.status == 400
    assert "phase=response_body" in caplog.text
    assert "server_processing_uncertain=False" in caplog.text
    assert "secret-body" not in caplog.text


async def test_incomplete_http_200_is_not_recorded_as_token_success(caplog):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    response = _response(200, {"access_token": "secret"})

    async def request(method, url, **kwargs):
        await kwargs["raise_for_status"](response)
        return response

    raw.request = Mock(side_effect=request)
    assert await wrapped.request("post", TOKEN_URL, raise_for_status=True) is response
    assert wrapped._last_success is None
    assert "did not contain a readable token pair" in caplog.text
    assert "secret" not in caplog.text


async def test_oauth_async_context_manager_preserves_cleanup_and_expiry_metadata(
    caplog,
):
    raw = Mock()
    wrapped = AuthDiagnosticSession(raw, "test-entry")
    expires = datetime.now(UTC) + timedelta(minutes=5)
    token = jwt.encode({"exp": int(expires.timestamp())}, "x" * 32, algorithm="HS256")
    response = _response(
        200, {"access_token": "new-access", "refresh_token": "new-refresh"}
    )

    async def enter():
        await raw.request.call_args.kwargs["raise_for_status"](response)
        return response

    context = Mock(__aenter__=AsyncMock(side_effect=enter), __aexit__=AsyncMock())
    raw.request.return_value = context
    with caplog.at_level(logging.INFO):
        async with wrapped.request(
            "post",
            TOKEN_URL,
            raise_for_status=True,
            headers={"Authorization": f"Bearer {token}"},
        ) as result:
            assert result is response
    context.__aexit__.assert_awaited_once_with(None, None, None)
    assert "access_expires_in_s_at_start=" in caplog.text
    assert "access_expires_in_s_at_start=None" not in caplog.text
    assert "overlap_observed=False" in caplog.text
    assert token not in caplog.text
