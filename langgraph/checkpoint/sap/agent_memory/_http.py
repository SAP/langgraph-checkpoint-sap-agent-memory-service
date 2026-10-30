"""HTTP clients with cached XSUAA/IAS authentication and bounded HTTP retries."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_IASTokenSource = str | Callable[[], str] | Callable[[], Awaitable[str]]

_RETRY_STATUSES = {429, 502, 503, 504}
_NO_RETRY_STATUSES = {400, 401, 403, 404, 413}
_TOKEN_EXPIRY_BUFFER = 60  # seconds

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 0.1
DEFAULT_MAX_BACKOFF = 2.0


def _backoff(attempt: int, initial: float, maximum: float) -> float:
    return min(initial * (2**attempt), maximum)


def validate_token(value: object) -> str:
    """Normalize a bearer token without exposing invalid credentials in errors."""
    if not isinstance(value, str):
        raise TypeError("Token source must return a string.")
    token = value.strip(" ")
    if not token:
        raise ValueError("Token source returned an empty token.")
    if any(ord(char) <= 32 or ord(char) >= 127 for char in token):
        raise ValueError("Token contains whitespace, control, or non-ASCII characters.")
    return token


def optional_ias_token(value: object) -> str | None:
    """A missing or space-only request override selects the default source."""
    if value is None or (isinstance(value, str) and not value.strip(" ")):
        return None
    return validate_token(value)


@dataclass(frozen=True)
class _TokenSnapshot:
    token: str = field(repr=False)
    generation: int


class CachedToken:
    """Per-tenant token state with one refresh shared across threads/event loops."""

    def __init__(self, x_zid: str | None = None) -> None:
        self._x_zid = x_zid
        self._token: str | None = None
        self._expires_at = 0.0
        self._no_expiry = False  # IAS tokens have no timer; only a 401 expires them
        self._generation = 0
        self._sync_lock = threading.Lock()
        self._pending: Future[_TokenSnapshot] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def is_valid(self) -> bool:
        if self._token is None:
            return False
        return self._no_expiry or time.monotonic() < self._expires_at

    def store(self, token_data: dict[str, Any]) -> None:
        token = validate_token(token_data["access_token"])
        expires_at = (
            time.monotonic()
            + int(token_data.get("expires_in", 3600))
            - _TOKEN_EXPIRY_BUFFER
        )
        with self._sync_lock:
            self._token, self._expires_at, self._no_expiry = token, expires_at, False
            self._generation += 1

    def store_ias(self, token: str) -> None:
        token = validate_token(token)
        with self._sync_lock:
            self._token, self._no_expiry = token, True
            self._generation += 1

    def invalidate(self) -> None:
        with self._sync_lock:
            self._token, self._expires_at, self._no_expiry = None, 0.0, False

    @property
    def token(self) -> str | None:
        with self._sync_lock:
            return self._token

    def _begin(
        self, rejected: _TokenSnapshot | None
    ) -> tuple[Future[_TokenSnapshot], bool]:
        with self._sync_lock:
            if self._pending is not None:
                return self._pending, False
            if rejected is not None and rejected.generation == self._generation:
                self._token, self._expires_at, self._no_expiry = None, 0.0, False
            pending: Future[_TokenSnapshot] = Future()
            if self.is_valid():
                pending.set_result(
                    _TokenSnapshot(validate_token(self._token), self._generation)
                )
                return pending, False
            self._pending = pending
            return pending, True

    def _finish(
        self, pending: Future[_TokenSnapshot], error: BaseException | None = None
    ) -> None:
        with self._sync_lock:
            if self._pending is not pending:
                return
            self._pending = None
            if error is None:
                try:
                    snapshot = _TokenSnapshot(
                        validate_token(self._token), self._generation
                    )
                except (ValueError, TypeError) as exc:
                    error = exc
        # Future callbacks may acquire the state lock; notify outside it.
        if error is not None:
            pending.set_exception(error)
        else:
            pending.set_result(snapshot)

    def get_sync(
        self, fetch: Callable[[], None], rejected: _TokenSnapshot | None = None
    ) -> _TokenSnapshot:
        pending, leader = self._begin(rejected)
        # A sync fetch/wait blocks the calling thread; on a live event loop that
        # stalls the loop (leader) or risks deadlock (follower). Refuse either way
        # unless the token was already cached and resolved synchronously.
        if not pending.done():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                if leader:
                    self._finish(pending)
                raise RuntimeError(
                    "Sync token refresh on a running event loop would block it; "
                    "use async checkpointer methods instead."
                )
        if leader:
            try:
                fetch()
            except BaseException as exc:
                self._finish(pending, exc)
            else:
                self._finish(pending)
        return pending.result()

    async def get_async(
        self,
        fetch: Callable[[], Awaitable[None]],
        rejected: _TokenSnapshot | None = None,
    ) -> _TokenSnapshot:
        pending, leader = self._begin(rejected)
        if leader:

            async def refresh() -> None:
                try:
                    await fetch()
                except BaseException as exc:
                    self._finish(pending, exc)
                else:
                    self._finish(pending)

            def finished(task: asyncio.Task[None]) -> None:
                with self._sync_lock:
                    self._tasks.discard(task)
                if task.cancelled():
                    self._finish(
                        pending, RuntimeError("Token refresh event loop was closed.")
                    )

            task = asyncio.create_task(refresh())
            with self._sync_lock:
                self._tasks.add(task)
            task.add_done_callback(finished)
        waiter = asyncio.wrap_future(pending)
        # Retrieve late errors even when all callers cancel their waits.
        waiter.add_done_callback(
            lambda future: None if future.cancelled() else future.exception()
        )
        return await asyncio.shield(waiter)


class XSUAATokenFetcher:
    """XSUAA ``client_credentials`` fetch logic.

    Owns the service credentials and knows how to obtain a token from the
    token endpoint.  Works with :class:`CachedToken` instances — it stores
    the fetched token there but holds no token state itself.
    """

    def __init__(
        self,
        token_url: str | None,
        client_id: str | None,
        client_secret: str | None,
        timeout: float,
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout

    def _token_headers(self, x_zid: str | None) -> dict[str, str]:
        h = {"Content-Type": "application/x-www-form-urlencoded"}
        if x_zid:
            h["x-zid"] = x_zid
        return h

    def _fetch_sync(self, cached: CachedToken) -> None:
        if not (self._token_url and self._client_id and self._client_secret):
            raise RuntimeError(
                "XSUAATokenFetcher called without credentials — "
                "this indicates a construction-time bug."
            )
        logger.debug(
            "Fetching OAuth2 token from %s (x-zid=%s)",
            self._token_url,
            cached._x_zid or "(none — single-tenant)",
        )
        start = time.time()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                r = client.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    headers=self._token_headers(cached._x_zid),
                )
                r.raise_for_status()
                token_data = r.json()
                cached.store(token_data)
                elapsed = time.time() - start
                logger.debug(
                    "OAuth2 token fetched: expires_in=%ds (%.3fs)",
                    token_data.get("expires_in", 3600),
                    elapsed,
                )
        except httpx.HTTPStatusError as e:
            logger.error(
                "OAuth2 token fetch failed with status %d",
                e.response.status_code,
            )
            raise ValueError("Failed to obtain OAuth2 token") from e
        except (httpx.RequestError, ValueError, KeyError) as e:
            logger.error("OAuth2 token fetch failed: %s", type(e).__name__)
            raise ValueError("OAuth2 token service unavailable") from e

    async def _fetch_async(self, cached: CachedToken) -> None:
        if not (self._token_url and self._client_id and self._client_secret):
            raise RuntimeError(
                "XSUAATokenFetcher called without credentials — "
                "this indicates a construction-time bug."
            )
        logger.debug(
            "Fetching OAuth2 token (async) from %s (x-zid=%s)",
            self._token_url,
            cached._x_zid or "(none — single-tenant)",
        )
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                    },
                    headers=self._token_headers(cached._x_zid),
                )
                r.raise_for_status()
                token_data = r.json()
                cached.store(token_data)
                elapsed = time.time() - start
                logger.debug(
                    "OAuth2 token fetched (async): expires_in=%ds (%.3fs)",
                    token_data.get("expires_in", 3600),
                    elapsed,
                )
        except httpx.HTTPStatusError as e:
            logger.error(
                "OAuth2 token fetch failed with status %d",
                e.response.status_code,
            )
            raise ValueError("Failed to obtain OAuth2 token") from e
        except (httpx.RequestError, ValueError, KeyError) as e:
            logger.error("OAuth2 token fetch failed: %s", type(e).__name__)
            raise ValueError("OAuth2 token service unavailable") from e

    def get_sync(self, cached: CachedToken) -> str | None:
        if not self._token_url:
            return None
        return self.get_sync_snapshot(cached).token

    async def get_async(self, cached: CachedToken) -> str | None:
        if not self._token_url:
            return None
        return (await self.get_async_snapshot(cached)).token

    def get_sync_snapshot(
        self, cached: CachedToken, rejected: _TokenSnapshot | None = None
    ) -> _TokenSnapshot:
        return cached.get_sync(lambda: self._fetch_sync(cached), rejected)

    async def get_async_snapshot(
        self, cached: CachedToken, rejected: _TokenSnapshot | None = None
    ) -> _TokenSnapshot:
        return await cached.get_async(lambda: self._fetch_async(cached), rejected)

    def headers_sync(self, cached: CachedToken) -> dict[str, str]:
        """Return auth headers for a sync request."""
        return _build_headers(self.get_sync(cached))

    async def headers_async(self, cached: CachedToken) -> dict[str, str]:
        """Return auth headers for an async request."""
        return _build_headers(await self.get_async(cached))


class IASTokenFetcher:
    """One stable IAS principal, supplied as a string or token callback."""

    def __init__(self, source: _IASTokenSource, timeout: float) -> None:
        self._timeout = timeout
        self._source = source if callable(source) else validate_token(source)

    @property
    def is_refreshable(self) -> bool:
        return callable(self._source)

    def _is_async(self) -> bool:
        return callable(self._source) and (
            inspect.iscoroutinefunction(self._source)
            or inspect.iscoroutinefunction(self._source.__call__)
        )

    def ensure_sync(self) -> None:
        if self._is_async():
            raise TypeError(
                "IAS token source is an async callable; use async checkpointer methods."
            )

    def _fetch_sync(self, cached: CachedToken) -> None:
        self.ensure_sync()
        value = self._source() if callable(self._source) else self._source
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            raise TypeError(
                "IAS token source returned an awaitable; use async checkpointer methods."
            )
        cached.store_ias(validate_token(value))

    async def _fetch_async(self, cached: CachedToken) -> None:
        if not callable(self._source):
            value = self._source
        elif self._is_async():
            value = self._source()
        else:
            value = await asyncio.to_thread(self._source)
        if inspect.isawaitable(value):
            value = await value
        cached.store_ias(validate_token(value))

    def get_sync(self, cached: CachedToken) -> str:
        return self.get_sync_snapshot(cached).token

    async def get_async(self, cached: CachedToken) -> str:
        return (await self.get_async_snapshot(cached)).token

    def get_sync_snapshot(
        self, cached: CachedToken, rejected: _TokenSnapshot | None = None
    ) -> _TokenSnapshot:
        self.ensure_sync()
        return cached.get_sync(lambda: self._fetch_sync(cached), rejected)

    async def get_async_snapshot(
        self, cached: CachedToken, rejected: _TokenSnapshot | None = None
    ) -> _TokenSnapshot:
        return await cached.get_async(lambda: self._fetch_async(cached), rejected)


class TokenRegistry:
    """Per-tenant token-state registry.

    Maps ``x_zid`` → :class:`CachedToken`. Has no knowledge of credentials or
    fetch logic — it only creates and looks up :class:`CachedToken` entries.
    Thread-safe: the internal dict is protected by a :class:`threading.Lock`
    during entry creation only.
    """

    def __init__(self, *, default_x_zid: str | None = None) -> None:
        self._default_x_zid = default_x_zid
        self._entries: dict[str | None, CachedToken] = {}
        self._lock = threading.Lock()

    def get(self, x_zid: str | None = None) -> CachedToken:
        """Return the :class:`CachedToken` for *x_zid*, creating it on first access.

        When *x_zid* is ``None`` the instance's ``default_x_zid`` is used.
        """
        key = x_zid if x_zid is not None else self._default_x_zid
        with self._lock:
            if key not in self._entries:
                self._entries[key] = CachedToken(x_zid=key)
            return self._entries[key]


def _build_headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _classify_response(response: httpx.Response) -> tuple[bool, Exception | None]:
    """Classify HTTP response and return (should_retry, exception).

    Args:
        response: HTTP response object

    Returns:
        Tuple of (should_retry, exception). If should_retry is False, exception is None.
        If should_retry is True, exception contains the HTTPStatusError to raise later.
    """
    if response.status_code in _NO_RETRY_STATUSES:
        return False, None
    if response.status_code not in _RETRY_STATUSES:
        return False, None
    exc = httpx.HTTPStatusError(
        f"HTTP {response.status_code}",
        request=response.request,
        response=response,
    )
    return True, exc


def _log_request(
    method: str, url: str, headers: dict[str, str], request_kwargs: dict[str, Any]
) -> None:
    """Log HTTP request details at DEBUG level.

    Args:
        method: HTTP method
        url: Request URL
        headers: Request headers (authorization masked)
        request_kwargs: Additional request arguments
    """
    import json as _json

    logger.debug(
        "HTTP %s %s headers=%s",
        method,
        url,
        {k: v for k, v in headers.items() if k.lower() != "authorization"},
    )
    if "json" in request_kwargs:
        try:
            body_size = len(_json.dumps(request_kwargs["json"]))
            logger.debug("HTTP request body: %d bytes", body_size)
        except Exception:
            pass


def _log_response(
    method: str, url: str, status_code: int, elapsed: float, content_size: int
) -> None:
    logger.debug(
        "HTTP %s %s → %d (%.3fs, %d bytes)",
        method,
        url,
        status_code,
        elapsed,
        content_size,
    )


def _log_failure(method: str, exc: Exception, elapsed: float) -> None:
    logger.debug(
        "HTTP %s → FAILED: %s (%.3fs)",
        method,
        type(exc).__name__,
        elapsed,
    )


def _log_retry(
    attempt: int, max_retries: int, sleep: float, exc: Exception | None
) -> None:
    """Log retry attempt at WARNING level.

    Args:
        attempt: Current attempt number (0-based)
        max_retries: Maximum number of retries
        sleep: Sleep duration before next retry
        exc: Exception that triggered the retry
    """
    logger.warning(
        "Retryable failure on attempt %d/%d, sleeping %.2fs: %s",
        attempt + 1,
        max_retries + 1,
        sleep,
        type(exc).__name__,
    )


def _retry_loop_sync(
    method: str,
    url: str,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    **request_kwargs: Any,
) -> httpx.Response:
    _log_request(method, url, headers, request_kwargs)

    last_exc: Exception | None = None
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                response = client.request(
                    method, url, headers=headers, **request_kwargs
                )
                elapsed = time.time() - start_time
                _log_response(
                    method,
                    url,
                    response.status_code,
                    elapsed,
                    len(response.content),
                )

                should_retry, exc = _classify_response(response)
                if not should_retry:
                    if response.status_code >= 400:
                        logger.warning(
                            "HTTP %s %s → %d body: %s",
                            method,
                            url,
                            response.status_code,
                            response.text[:1000],
                        )
                    response.raise_for_status()
                    return response
                last_exc = exc
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as exc:
                elapsed = time.time() - start_time
                _log_failure(method, exc, elapsed)
                last_exc = exc
            except httpx.HTTPStatusError:
                raise

            if attempt < max_retries:
                sleep = _backoff(attempt, initial_backoff, max_backoff)
                _log_retry(attempt, max_retries, sleep, last_exc)
                time.sleep(sleep)

    assert last_exc is not None
    raise last_exc


async def _retry_loop_async(
    method: str,
    url: str,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    initial_backoff: float,
    max_backoff: float,
    **request_kwargs: Any,
) -> httpx.Response:
    _log_request(method, url, headers, request_kwargs)

    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                response = await client.request(
                    method, url, headers=headers, **request_kwargs
                )
                elapsed = time.time() - start_time
                _log_response(
                    method,
                    url,
                    response.status_code,
                    elapsed,
                    len(response.content),
                )

                should_retry, exc = _classify_response(response)
                if not should_retry:
                    if response.status_code >= 400:
                        logger.warning(
                            "HTTP %s %s → %d body: %s",
                            method,
                            url,
                            response.status_code,
                            response.text[:1000],
                        )
                    response.raise_for_status()
                    return response
                last_exc = exc
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as exc:
                elapsed = time.time() - start_time
                _log_failure(method, exc, elapsed)
                last_exc = exc
            except httpx.HTTPStatusError:
                raise

            if attempt < max_retries:
                sleep = _backoff(attempt, initial_backoff, max_backoff)
                _log_retry(attempt, max_retries, sleep, last_exc)
                await asyncio.sleep(sleep)

    assert last_exc is not None
    raise last_exc


class RetryClient:
    """Async HTTP client with exponential backoff and OAuth2 token management.

    Retries on HTTP 429/502/503/504 and connection/timeout/read errors.
    A 401 allows one refresh retry for refreshable tokens; 400/403/404/413 fail immediately.
    """

    def __init__(
        self,
        fetcher: XSUAATokenFetcher | IASTokenFetcher,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
    ) -> None:
        self._fetcher = fetcher
        self._tokens = CachedToken()
        self._timeout = fetcher._timeout
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    def _retry_kwargs(self) -> dict[str, Any]:
        return {
            "timeout": self._timeout,
            "max_retries": self._max_retries,
            "initial_backoff": self._initial_backoff,
            "max_backoff": self._max_backoff,
        }

    async def get(
        self,
        url: str,
        *,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = await f.get_async_snapshot(cache)
        try:
            return await _retry_loop_async(
                "GET", url, _build_headers(token.token), **self._retry_kwargs()
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = await f.get_async_snapshot(cache, rejected=token)
                return await _retry_loop_async(
                    "GET", url, _build_headers(fresh.token), **self._retry_kwargs()
                )
            raise

    async def post(
        self,
        url: str,
        *,
        json: Any,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = await f.get_async_snapshot(cache)
        try:
            return await _retry_loop_async(
                "POST",
                url,
                _build_headers(token.token),
                **self._retry_kwargs(),
                json=json,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = await f.get_async_snapshot(cache, rejected=token)
                return await _retry_loop_async(
                    "POST",
                    url,
                    _build_headers(fresh.token),
                    **self._retry_kwargs(),
                    json=json,
                )
            raise

    async def patch(
        self,
        url: str,
        *,
        json: Any,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = await f.get_async_snapshot(cache)
        try:
            return await _retry_loop_async(
                "PATCH",
                url,
                _build_headers(token.token),
                **self._retry_kwargs(),
                json=json,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = await f.get_async_snapshot(cache, rejected=token)
                return await _retry_loop_async(
                    "PATCH",
                    url,
                    _build_headers(fresh.token),
                    **self._retry_kwargs(),
                    json=json,
                )
            raise

    async def delete(
        self,
        url: str,
        *,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = await f.get_async_snapshot(cache)
        try:
            return await _retry_loop_async(
                "DELETE", url, _build_headers(token.token), **self._retry_kwargs()
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = await f.get_async_snapshot(cache, rejected=token)
                return await _retry_loop_async(
                    "DELETE", url, _build_headers(fresh.token), **self._retry_kwargs()
                )
            raise


class SyncRetryClient:
    """Sync HTTP client with exponential backoff and OAuth2 token management.

    Same retry policy as :class:`RetryClient`; used by the sync paths.
    """

    def __init__(
        self,
        fetcher: XSUAATokenFetcher | IASTokenFetcher,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
    ) -> None:
        self._fetcher = fetcher
        self._tokens = CachedToken()
        self._timeout = fetcher._timeout
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    def _retry_kwargs(self) -> dict[str, Any]:
        return {
            "timeout": self._timeout,
            "max_retries": self._max_retries,
            "initial_backoff": self._initial_backoff,
            "max_backoff": self._max_backoff,
        }

    def get(
        self,
        url: str,
        *,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = f.get_sync_snapshot(cache)
        try:
            return _retry_loop_sync(
                "GET", url, _build_headers(token.token), **self._retry_kwargs()
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = f.get_sync_snapshot(cache, rejected=token)
                return _retry_loop_sync(
                    "GET", url, _build_headers(fresh.token), **self._retry_kwargs()
                )
            raise

    def post(
        self,
        url: str,
        *,
        json: Any,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = f.get_sync_snapshot(cache)
        try:
            return _retry_loop_sync(
                "POST",
                url,
                _build_headers(token.token),
                **self._retry_kwargs(),
                json=json,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = f.get_sync_snapshot(cache, rejected=token)
                return _retry_loop_sync(
                    "POST",
                    url,
                    _build_headers(fresh.token),
                    **self._retry_kwargs(),
                    json=json,
                )
            raise

    def patch(
        self,
        url: str,
        *,
        json: Any,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = f.get_sync_snapshot(cache)
        try:
            return _retry_loop_sync(
                "PATCH",
                url,
                _build_headers(token.token),
                **self._retry_kwargs(),
                json=json,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = f.get_sync_snapshot(cache, rejected=token)
                return _retry_loop_sync(
                    "PATCH",
                    url,
                    _build_headers(fresh.token),
                    **self._retry_kwargs(),
                    json=json,
                )
            raise

    def delete(
        self,
        url: str,
        *,
        token_cache: CachedToken | None = None,
        fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
    ) -> httpx.Response:
        f = fetcher or self._fetcher
        cache = token_cache or self._tokens
        token = f.get_sync_snapshot(cache)
        try:
            return _retry_loop_sync(
                "DELETE", url, _build_headers(token.token), **self._retry_kwargs()
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and (
                not isinstance(f, IASTokenFetcher) or f.is_refreshable
            ):
                fresh = f.get_sync_snapshot(cache, rejected=token)
                return _retry_loop_sync(
                    "DELETE", url, _build_headers(fresh.token), **self._retry_kwargs()
                )
            raise


def collect_all_pages_sync(
    client: SyncRetryClient,
    first_url: str,
    *,
    token_cache: CachedToken | None = None,
    fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
) -> list[Any]:
    """Fetch all OData pages by following @odata.nextLink, return combined value list."""
    items: list[Any] = []
    url: str | None = first_url
    while url:
        r = client.get(url, token_cache=token_cache, fetcher=fetcher)
        r.raise_for_status()
        body = r.json()
        items.extend(body.get("value", []))
        url = body.get("@odata.nextLink")
    return items


async def collect_all_pages_async(
    client: RetryClient,
    first_url: str,
    *,
    token_cache: CachedToken | None = None,
    fetcher: XSUAATokenFetcher | IASTokenFetcher | None = None,
) -> list[Any]:
    """Async variant of :func:`collect_all_pages_sync`."""
    items: list[Any] = []
    url: str | None = first_url
    while url:
        r = await client.get(url, token_cache=token_cache, fetcher=fetcher)
        r.raise_for_status()
        body = r.json()
        items.extend(body.get("value", []))
        url = body.get("@odata.nextLink")
    return items
