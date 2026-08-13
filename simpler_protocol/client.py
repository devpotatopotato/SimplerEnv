"""Standard-library HTTP client for remote policies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from .schema import validate_metadata


class PolicyClientError(RuntimeError):
    pass


class PolicyClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None):
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PolicyClientError(f"policy server returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PolicyClientError(f"policy server request failed: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/healthz")

    def metadata(self) -> dict[str, Any]:
        return validate_metadata(self._request("GET", "/v1/metadata"))

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/reset", payload)

    def actions(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/actions", payload)
