"""Profile-scoped lifecycle debounce for automatic dry curator scans."""

from __future__ import annotations

import importlib
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 60.0
MUTATION_ACTIONS = frozenset({
    "created", "installed", "patched", "edited", "archived", "restored", "stale",
})
_BUSY_ERROR = "another jev-curator run holds the profile claim"


class LifecycleDebouncer:
    """Coalesce skill mutations into one dry run per active Hermes profile."""

    def __init__(
        self,
        service_factory: Callable[[], Any],
        *,
        delay: float = DEBOUNCE_SECONDS,
        timer_factory: Callable[..., Any] = threading.Timer,
    ) -> None:
        self._service_factory = service_factory
        self._delay = float(delay)
        self._timer_factory = timer_factory
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[int, Any]] = {}
        self._generation = 0
        self._closed = False

    def notify(self, action: str, *, service: Any = None) -> bool:
        """Reset this profile's timer for a successful mutation event."""
        if str(action or "").strip().lower() not in MUTATION_ACTIONS:
            return False
        current = service or self._service_factory()
        if not _automatic_run_enabled(current):
            return False
        try:
            home, secrets = _capture_scope()
        except Exception:
            logger.warning("jev-curator auto-scan scope capture failed", exc_info=True)
            return False
        return self._schedule(home, secrets)

    def close(self) -> None:
        """Cancel pending timers when Hermes unloads the plugin."""
        with self._lock:
            self._closed = True
            timers = [timer for _, timer in self._pending.values()]
            self._pending.clear()
        for timer in timers:
            timer.cancel()

    def _schedule(self, home: str, secrets: Mapping[str, str] | None) -> bool:
        with self._lock:
            if self._closed:
                return False
            previous = self._pending.pop(home, None)
            if previous is not None:
                previous[1].cancel()
            self._generation += 1
            generation = self._generation
            timer = self._timer_factory(
                self._delay,
                self._fire,
                args=(home, generation, secrets),
            )
            timer.daemon = True
            self._pending[home] = (generation, timer)
            timer.start()
            return True

    def _fire(self, home: str, generation: int, secrets: Mapping[str, str] | None) -> None:
        with self._lock:
            pending = self._pending.get(home)
            if self._closed or pending is None or pending[0] != generation:
                return
            self._pending.pop(home, None)

        try:
            with _profile_scope(home, secrets):
                service = self._service_factory()
                if not _automatic_run_enabled(service):
                    return
                result = service.run(apply=False)
        except Exception:
            logger.warning("jev-curator automatic dry run failed", exc_info=True)
            return

        if isinstance(result, Mapping) and result.get("error") == _BUSY_ERROR:
            # ponytail: retry later rather than lose a mutation that raced another profile-local run.
            self._schedule(home, secrets)


def _automatic_run_enabled(service: Any) -> bool:
    settings = getattr(service, "settings", None)
    return bool(
        settings is not None
        and getattr(settings, "mode", "off") != "off"
        and getattr(settings, "allow_content_egress", False) is True
    )


def _capture_scope() -> tuple[str, Mapping[str, str] | None]:
    secret_scope = importlib.import_module("agent.secret_scope")
    constants = importlib.import_module("hermes_constants")

    home = str(Path(constants.get_hermes_home()).resolve(strict=False))
    secrets = secret_scope.current_secret_scope()
    return home, dict(secrets) if secrets is not None else None


@contextmanager
def _profile_scope(home: str, secrets: Mapping[str, str] | None):
    secret_scope = importlib.import_module("agent.secret_scope")
    constants = importlib.import_module("hermes_constants")

    home_token = constants.set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope(secrets)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        constants.reset_hermes_home_override(home_token)
