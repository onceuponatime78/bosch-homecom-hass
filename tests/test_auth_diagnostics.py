"""OAuth diagnostics must survive library error handling without exposing secrets."""

from unittest.mock import AsyncMock, Mock

from aiohttp import ClientResponseError
from homecom_alt import AuthFailedError, ConnectionOptions, HomeComAlt
from homecom_alt.const import OAUTH_DOMAIN, OAUTH_ENDPOINT
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
