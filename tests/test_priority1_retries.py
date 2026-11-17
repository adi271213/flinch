"""Exercise real SDK transport errors and bounded HTTP range retries without a network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.client import IncompleteRead
from importlib import import_module
from types import ModuleType
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from huggingface_hub import errors as hub_errors
from huggingface_hub.utils import _http as hub_http

import lerobot_flinch.hub_scanner as scanner


def _transport_cases() -> list[Any]:
    cases = []
    for module_name in ("httpx", "httpx2", "requests.exceptions"):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        for name in ("ConnectError", "ReadTimeout", "RemoteProtocolError", "ConnectTimeout"):
            error = getattr(module, name, None)
            if error is not None:
                cases.append(pytest.param(error, id=f"{module_name}.{name}"))
    return cases


@pytest.fixture
def delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    result: list[float] = []
    monkeypatch.setattr(scanner, "_sleep", result.append)
    return result


@pytest.mark.parametrize("error_type", _transport_cases())
def test_real_installed_transport_errors_retry(error_type: type[Exception], delays: list[float]) -> None:
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise error_type("connection interrupted")
        return "ok"

    assert scanner.with_retries(call) == "ok"
    assert calls == 3
    assert len(delays) == 2
    assert all(0 <= delay <= scanner.RETRY_MAX_SECONDS for delay in delays)


def _sdk_response(status: int) -> Any:
    transport: ModuleType = next(
        getattr(hub_http, name) for name in ("httpx", "httpx2", "requests") if hasattr(hub_http, name)
    )
    if transport.__name__ == "requests":
        response = transport.Response()
        response.status_code = status
        response.url = "https://huggingface.co/example/checkpoint"
        return response
    return transport.Response(status, request=transport.Request("GET", "https://huggingface.co/example"))


@pytest.mark.parametrize(
    "error",
    [
        hub_errors.GatedRepoError("gated", response=_sdk_response(403)),
        hub_errors.RepositoryNotFoundError("private", response=_sdk_response(401)),
        hub_errors.RevisionNotFoundError("missing revision", response=_sdk_response(404)),
        hub_errors.OfflineModeIsEnabled("offline"),
        ValueError("invalid byte range"),
        HTTPError("https://example.test", 401, "private", {}, None),
        HTTPError("https://example.test", 403, "gated", {}, None),
        HTTPError("https://example.test", 404, "missing", {}, None),
    ],
)
def test_real_permanent_errors_are_not_retried(error: Exception, delays: list[float]) -> None:
    calls = 0

    def call() -> None:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        scanner.with_retries(call)
    assert calls == 1
    assert not delays


@pytest.mark.parametrize("status", [429, 500, 503])
def test_real_sdk_http_status_errors_are_retried(status: int, delays: list[float]) -> None:
    error = hub_errors.HfHubHTTPError("temporarily unavailable", response=_sdk_response(status))
    calls = 0

    def call() -> None:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(hub_errors.HfHubHTTPError):
        scanner.with_retries(call)
    assert calls == scanner.RETRY_ATTEMPTS
    assert len(delays) == scanner.RETRY_ATTEMPTS - 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2", 2.0),
        ("-1", 0.0),
        ("0", 0.0),
        ("999999", scanner.RETRY_MAX_SECONDS),
        ("NaN", None),
        ("Infinity", None),
        ("-Infinity", None),
        ("not a date", None),
        ("Wed, 01 Jan 2000 00:00:00 GMT", 0.0),
        (
            format_datetime(datetime.now(timezone.utc) + timedelta(days=2), usegmt=True),
            scanner.RETRY_MAX_SECONDS,
        ),
    ],
)
def test_retry_after_values_are_safe_and_bounded(value: str, expected: float | None) -> None:
    error = HTTPError("https://example.test", 429, "busy", {"retry-after": value}, None)
    assert scanner._retry_after(error) == expected


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "broken"])
def test_malformed_retry_after_falls_back_to_jitter(
    value: str, delays: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scanner.random, "uniform", lambda low, high: high)
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError("https://example.test", 429, "busy", {"Retry-After": value}, None)
        return "ok"

    assert scanner.with_retries(call) == "ok"
    assert delays == [scanner.RETRY_BASE_SECONDS]


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5, "2", scanner.RETRY_ATTEMPTS + 1])
def test_invalid_attempt_counts_do_not_issue_requests(attempts: Any, delays: list[float]) -> None:
    calls = []
    with pytest.raises(ValueError, match="Retry attempts"):
        scanner.with_retries(lambda: calls.append(True), attempts=attempts)
    assert not calls and not delays


class _Response:
    def __init__(self, *, kind: str, failure: str | None = None):
        self.status = 200 if kind == "header" else 206
        self.headers = (
            {"Content-Length": "1000000000"}
            if kind == "header"
            else {"Content-Range": "bytes 1000000-1000007/1000008", "Content-Length": "8"}
        )
        self.failure = failure
        self.closed = False
        self.read_sizes: list[int] = []
        self.bytes_read = 0

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True
        if self.failure == "close":
            raise ConnectionResetError("connection failed while closing")

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if self.failure == "read" and len(self.read_sizes) == 2:
            raise TimeoutError("connection failed during body read")
        amount = min(3, size) if self.failure == "read" else size
        self.bytes_read += amount
        return b"x" * amount


@pytest.mark.parametrize("kind", ["header", "tensor"])
@pytest.mark.parametrize("failure", ["read", "close"])
def test_whole_range_transaction_retries_and_closes_failed_response(
    kind: str, failure: str, delays: list[float]
) -> None:
    failed = _Response(kind=kind, failure=failure)
    healthy = _Response(kind=kind)
    responses = iter([failed, healthy])
    requests = []

    def opener(request: Any, **kwargs: Any) -> _Response:
        requests.append(request)
        return next(responses)

    if kind == "header":
        payload, consumed, total = scanner._range_read("https://example.test", 4, 8, opener)
        assert consumed == 12 and total == 1000000000
        budget = 12
    else:
        payload = scanner._strict_tensor_range_read("https://example.test", 1000000, 8, 1000008, opener)
        budget = 8
    assert payload == b"x" * 8
    assert len(requests) == 2 and len(delays) == 1
    assert all(response.closed and response.bytes_read <= budget for response in (failed, healthy))
    assert requests[0].get_header("Range") == requests[1].get_header("Range")
    assert all(
        size <= scanner.HEADER_PREFIX_BYTES for response in (failed, healthy) for size in response.read_sizes
    )


@pytest.mark.parametrize("kind", ["header", "tensor"])
def test_failed_stream_attempts_keep_byte_and_retry_bounds(kind: str, delays: list[float]) -> None:
    responses = []

    def opener(*args: Any, **kwargs: Any) -> _Response:
        response = _Response(kind=kind, failure="read")
        responses.append(response)
        return response

    with pytest.raises(TimeoutError):
        if kind == "header":
            scanner._range_read("https://example.test", 4, 8, opener)
        else:
            scanner._strict_tensor_range_read("https://example.test", 1000000, 8, 1000008, opener)
    assert len(responses) == scanner.RETRY_ATTEMPTS
    assert len(delays) == scanner.RETRY_ATTEMPTS - 1
    assert all(response.closed and response.bytes_read <= 3 for response in responses)


def test_invalid_range_response_is_closed_without_read_or_retry(delays: list[float]) -> None:
    response = _Response(kind="tensor")
    response.headers["Content-Range"] = "bytes 0-7/1000008"
    calls = []

    def opener(*args: Any, **kwargs: Any) -> _Response:
        calls.append(True)
        return response

    with pytest.raises(ValueError, match="different tensor byte range"):
        scanner._strict_tensor_range_read("https://example.test", 1000000, 8, 1000008, opener)
    assert response.closed and not response.read_sizes
    assert len(calls) == 1 and not delays


def test_incomplete_urllib_body_is_retryable(delays: list[float]) -> None:
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IncompleteRead(b"partial", 8)
        return "ok"

    assert scanner.with_retries(call) == "ok"
    assert len(delays) == 1


def test_urllib_connection_failure_is_retryable(delays: list[float]) -> None:
    calls = 0

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError("connection reset")
        return "ok"

    assert scanner.with_retries(call) == "ok"
    assert len(delays) == 1
