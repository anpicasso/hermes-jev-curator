"""Small secure transport for Jev's typed decision endpoint."""

from __future__ import annotations

import json
import http.client
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from .models import Settings


_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
_MAX_ATTEMPTS = 3
_MAX_REQUEST_BYTES = 2_000_000
_MAX_RESPONSE_BYTES = 2_000_000
_RETRY_STATUS = frozenset({429, 529})
_PROB_TOLERANCE = 0.02

# Core handles broad cases; these cover credential-bearing shell and HTTP
# shapes commonly embedded in operational skills before third-party egress.
_FLAG_RE = re.compile(
    r"(?i)(?<![\w-])(--?(?:password|passwd|pass|token|api[-_]?key|secret|"
    r"access[-_]?key|auth[-_]?token|client[-_]?secret)[=\s]+)"
    r"(\"[^\"\n]*\"|'[^'\n]*'|\S+)")
_COOKIE_RE = re.compile(r"(?i)(?<![\w-])((?:set-)?cookie\s*:\s*)[^\"'\n]+")
_COOKIE_FLAG_RE = re.compile(
    r"(?i)(?<![\w-])((?:--cookie|-b)[=\s]{1,2}['\"]?)(?=[^\"'\s]*=)([^\"'\s]+)")
_CURL_USER_RE = re.compile(
    r"(?i)(?<![\w-])((?:--user|-u)[=\s]{0,2}['\"]?)(?![0-9]+:)"
    r"([^\s:@'\"]{0,64}):([^\s@'\"]{1,256})")
_URL_USERINFO_RE = re.compile(r"(?i)(://[^/\s:@]+:)([^@/\s]+)(@)")
_BASIC_RE = re.compile(r"(?i)(authorization\s*:\s*basic\s+)\S+")
_ASSIGN_RE = re.compile(
    r"(?i)([\w.-]*(?:password|passwd|token|secret|api[_-]?key|access[_-]?key)"
    r"\s*[\"']?\s*[:=]\s*)(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s\"']+)")
_KEY_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY(?: BLOCK)?-----.*?"
    r"-----END [^-\r\n]*PRIVATE KEY(?: BLOCK)?-----", re.S)


@dataclass(frozen=True)
class Route:
    url: str
    model: str
    credential_provider: str = ""
    credential_env: str = ""
    custom: bool = False


@dataclass(frozen=True)
class JevResponse:
    answers: Mapping[str, Mapping[str, Any]]
    model: str
    provider: str
    usage: Mapping[str, Any]
    attempts: int
    http_status: int
    request_id: str = ""


class _PayloadError(RuntimeError):
    """A deterministic response-contract failure that retries cannot repair."""


def resolve_route(settings: Settings) -> Route:
    provider = settings.provider.lower()
    if provider in {"typesafe", "typesafe-jev", "jev"}:
        if settings.base_url:
            raise ValueError("base_url requires provider: custom")
        if settings.key_env:
            raise ValueError("key_env is only valid with provider: custom")
        return Route(_validated_url(_TYPESAFE_URL), settings.model or "jev-latest",
                     "typesafe-jev", "TYPESAFE_API_KEY")
    if provider in {"openrouter", "open-router"}:
        if settings.base_url:
            raise ValueError("base_url requires provider: custom")
        if settings.key_env:
            raise ValueError("key_env is only valid with provider: custom")
        return Route(_validated_url(_OPENROUTER_URL), settings.model or "~typesafe/jev-latest",
                     "openrouter", "OPENROUTER_API_KEY")
    if provider == "custom":
        if not settings.base_url:
            raise ValueError("provider: custom requires a full Jev decision base_url")
        return Route(_validated_url(settings.base_url), settings.model or "jev-latest",
                     "", settings.key_env, custom=True)
    raise ValueError("provider must be typesafe, openrouter, or custom")


def request(
    state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]], settings: Settings,
) -> JevResponse:
    """Redact, POST under one deadline, and validate every requested typed answer."""
    route = resolve_route(settings)
    try:
        safe_state = _redact_mapping(state)
    except RuntimeError:
        raise
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeError("Jev request state could not be safely redacted") from exc
    body = {"state": safe_state, "model": route.model, "questions": dict(questions)}
    try:
        data = json.dumps(
            body, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeError("Jev request state is not safely serializable") from exc
    if len(data) > _MAX_REQUEST_BYTES:
        raise RuntimeError("Jev request exceeds the 2 MB safety limit")
    key = _resolve_key(route)
    if route.credential_provider and not key:
        raise RuntimeError(
            f"No credential is configured for Jev provider {route.credential_provider!r}; "
            f"configure {route.credential_env} in the active profile")
    if route.custom and route.credential_env and not key:
        raise RuntimeError(
            f"Configured Jev credential environment variable {route.credential_env!r} is empty")
    payload, attempts, status, request_id = _post(
        route.url, data, key=key, timeout=settings.timeout_seconds)
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise RuntimeError("Jev response carries no answers object")
    normalized = validate_answers(answers, questions)
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return JevResponse(
        answers=normalized,
        model=str(payload.get("model") or route.model),
        provider=str(payload.get("provider") or settings.provider),
        usage=usage,
        attempts=attempts,
        http_status=status,
        request_id=request_id,
    )


def validate_answers(
    answers: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, question in questions.items():
        answer = answers.get(key)
        if not isinstance(answer, dict):
            raise RuntimeError(f"Jev question {key!r} was asked but not answered")
        kind = question.get("type")
        if kind == "noul":
            value = _probability(answer.get("noul"), f"{key}.noul")
            item = dict(answer)
            item["noul"] = value
            if "confidence" in item:
                item["confidence"] = _probability(item["confidence"], f"{key}.confidence")
            normalized[key] = item
        elif kind == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise RuntimeError(f"Jev choice question {key!r} has no criteria")
            options = tuple(str(option) for option in criteria)
            choice = answer.get("choice")
            if not isinstance(choice, str) or choice not in options:
                raise RuntimeError(f"Jev {key}.choice must be one of {options}")
            item = dict(answer)
            item["choice"] = choice
            item["confidence"] = _probability(answer.get("confidence"), f"{key}.confidence")
            _validate_probabilities(answer.get("probabilities"), options, choice, key)
            normalized[key] = item
        elif kind == "score":
            criteria = question.get("criteria")
            top = len(criteria) - 1 if isinstance(criteria, list) else -1
            score = answer.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RuntimeError(f"Jev {key}.score is not numeric")
            try:
                score = float(score)
            except (OverflowError, ValueError) as exc:
                raise RuntimeError(f"Jev {key}.score is not numeric") from exc
            if not math.isfinite(score) or not 0 <= score <= top:
                raise RuntimeError(f"Jev {key}.score is outside 0..{top}")
            item = dict(answer)
            item["score"] = score
            normalized[key] = item
        else:
            raise RuntimeError(f"Unsupported Jev question type {kind!r}")
    return normalized


def _resolve_key(route: Route) -> str:
    if route.credential_provider:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(requested=route.credential_provider)
            key = str(runtime.get("api_key") or "").strip()
            runtime_url = str(runtime.get("base_url") or "").strip()
            # A missing/unknown provider can fall through to another configured runtime.
            # Never forward that runtime's credential to a different decision host.
            if key and runtime_url and _origin(runtime_url) == _origin(route.url):
                return key
        except Exception:
            pass
    if route.credential_env:
        try:
            from hermes_cli.config import get_env_value_prefer_dotenv
            return str(get_env_value_prefer_dotenv(route.credential_env) or "").strip()
        except Exception:
            return ""
    return ""


def _redact_mapping(value: Any) -> Any:
    if isinstance(value, str):
        try:
            from agent.redact import redact_sensitive_text
            try:
                redacted = redact_sensitive_text(
                    value, force=True, redact_url_credentials=True)
            except TypeError:
                redacted = redact_sensitive_text(value, force=True)
            return _local_redact(redacted)
        except Exception as exc:
            # The host owns broad credential detection.  A narrower fallback can
            # never prove arbitrary skill text safe, so refuse third-party egress.
            raise RuntimeError(
                "Jev egress redaction is unavailable; refusing request") from exc
    if isinstance(value, Mapping):
        return {_redact_mapping(str(key)): _redact_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_mapping(item) for item in value]
    return value


def _local_redact(text: str) -> str:
    out = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-+/=]{8,}", r"\1[REDACTED]", text)
    out = _ASSIGN_RE.sub(r"\1[REDACTED]", out)
    out = _KEY_RE.sub("[REDACTED PRIVATE KEY]", out)
    out = _COOKIE_RE.sub(r"\1***", out)
    out = _COOKIE_FLAG_RE.sub(r"\1***", out)
    out = _CURL_USER_RE.sub(r"\1\2:***", out)
    out = _URL_USERINFO_RE.sub(r"\1***\3", out)
    out = _BASIC_RE.sub(r"\1[REDACTED]", out)
    return _FLAG_RE.sub(r"\1[REDACTED]", out)


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(req.full_url) != _origin(newurl):
            raise urllib.error.HTTPError(req.full_url, code, "cross-origin redirect refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameOriginRedirects())


def _post(url: str, data: bytes, *, key: str, timeout: float) -> tuple[dict[str, Any], int, int, str]:
    deadline = time.monotonic() + max(1.0, min(float(timeout), 120.0))
    last: Exception = RuntimeError("Jev request made no attempt")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "hermes-jev-curator/0.1",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with _OPENER.open(req, timeout=remaining) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise _PayloadError("Jev response exceeds the 2 MB safety limit")
                try:
                    payload = json.loads(raw)
                except UnicodeDecodeError as exc:
                    raise _PayloadError("Jev response is not valid UTF-8 JSON") from exc
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Jev response is not valid UTF-8 JSON") from exc
                if not isinstance(payload, dict):
                    raise _PayloadError("Jev response is not a JSON object")
                status = int(getattr(response, "status", 200) or 200)
                request_id = _request_id(getattr(response, "headers", None), payload)
                return payload, attempt, status, request_id
        except urllib.error.HTTPError as exc:
            last = RuntimeError(f"Jev HTTP {exc.code}")
            if exc.code not in _RETRY_STATUS and exc.code < 500:
                raise last from exc
        except _PayloadError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                http.client.InvalidURL, RuntimeError) as exc:
            last = exc
        if attempt < _MAX_ATTEMPTS:
            delay = min(0.5 * 2 ** (attempt - 1), 4.0) + random.random() * 0.25
            if time.monotonic() + delay >= deadline:
                break
            time.sleep(delay)
    detail = str(last).replace("\n", " ")[:200]
    raise RuntimeError(
        f"Jev request failed after bounded retries: {type(last).__name__}: {detail}") from last


def _validated_url(raw: str) -> str:
    value = str(raw or "").strip()
    if re.search(r"[\x00-\x20\x7f]", value):
        raise ValueError("base_url must not contain whitespace or control characters")
    authority = value.split("://", 1)[-1].split("/", 1)[0]
    if "%" in authority:
        raise ValueError("base_url host must not contain percent escapes")
    parsed = urllib.parse.urlparse(value)
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    if parsed.scheme != "https" or not host:
        raise ValueError("base_url must name an https host")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not contain URL credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    if port not in (None, 443):
        raise ValueError("base_url must use the default HTTPS port")
    return value


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlparse(url)
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:
        port = -1
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), port


def _request_id(headers: Any, payload: Mapping[str, Any]) -> str:
    for name in ("x-typesafe-request-id", "x-generation-id", "request-id", "x-request-id"):
        try:
            value = (headers or {}).get(name)
        except Exception:
            value = None
        if value:
            return str(value)[:128]
    return str(payload.get("id") or "")[:128]


def _probability(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Jev {where} is not numeric")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise RuntimeError(f"Jev {where} is not numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeError(f"Jev {where} is outside [0,1]")
    return number


def _validate_probabilities(raw: Any, options: tuple[str, ...], picked: str, where: str) -> None:
    if not isinstance(raw, dict) or set(raw) != set(options):
        raise RuntimeError(f"Jev {where}.probabilities do not match the requested options")
    values = {option: _probability(raw[option], f"{where}.probabilities[{option}]") for option in options}
    total = sum(values.values())
    if abs(total - 1.0) > _PROB_TOLERANCE:
        raise RuntimeError(f"Jev {where}.probabilities do not sum to one")
    if values[picked] < max(values.values()) - _PROB_TOLERANCE:
        raise RuntimeError(f"Jev {where}.choice is not an argmax")
