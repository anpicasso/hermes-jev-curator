"""Lifecycle auto-run debounce: one dry run per profile after a quiet minute."""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from plugin import debounce
from plugin.models import Settings


class FakeTimer:
    instances = []

    def __init__(self, interval, callback, args=(), kwargs=None):
        self.interval = interval
        self.callback = callback
        self.args = args
        self.kwargs = kwargs or {}
        self.cancelled = False
        self.started = False
        self.daemon = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        # Invoke even after cancel to exercise the generation check against timer races.
        self.callback(*self.args, **self.kwargs)


class FakeService:
    def __init__(self, *, mode="observe", egress=True, results=None):
        self.settings = Settings(mode=mode, allow_content_egress=egress)
        self.results = list(results or ({"ok": True},))
        self.apply_calls = []

    def run(self, *, apply=False):
        self.apply_calls.append(apply)
        return self.results.pop(0) if self.results else {"ok": True}


class DebounceTests(unittest.TestCase):
    def setUp(self):
        FakeTimer.instances = []
        self.bindings = []

    def _scope(self, home, secrets):
        @contextlib.contextmanager
        def bound():
            self.bindings.append((home, secrets))
            yield
        return bound()

    def test_burst_coalesces_after_one_minute_and_never_applies(self):
        service = FakeService(mode="apply")
        worker = debounce.LifecycleDebouncer(lambda: service, timer_factory=FakeTimer)
        with mock.patch.object(debounce, "_capture_scope", return_value=("/profiles/a", {"KEY": "a"})), \
             mock.patch.object(debounce, "_profile_scope", side_effect=self._scope):
            self.assertTrue(worker.notify("created", service=service))
            self.assertTrue(worker.notify("patched", service=service))
            first, second = FakeTimer.instances
            self.assertEqual((first.interval, second.interval), (60.0, 60.0))
            self.assertTrue(first.cancelled)
            first.fire()
            second.fire()

        self.assertEqual(service.apply_calls, [False])
        self.assertEqual(self.bindings, [("/profiles/a", {"KEY": "a"})])

    def test_loaded_off_and_no_consent_do_not_schedule_and_unload_cancels(self):
        enabled = FakeService()
        worker = debounce.LifecycleDebouncer(lambda: enabled, timer_factory=FakeTimer)
        with mock.patch.object(debounce, "_capture_scope", return_value=("/profiles/a", None)):
            self.assertFalse(worker.notify("loaded", service=enabled))
            self.assertFalse(worker.notify("patched", service=FakeService(mode="off")))
            self.assertFalse(worker.notify("patched", service=FakeService(egress=False)))
            self.assertTrue(worker.notify("patched", service=enabled))
        timer = FakeTimer.instances[0]
        worker.close()
        self.assertTrue(timer.cancelled)
        timer.fire()
        self.assertEqual(enabled.apply_calls, [])

    def test_profiles_are_independent_and_busy_claim_retries(self):
        busy = "another jev-curator run holds the profile claim"
        service = FakeService(results=[{"ok": False, "error": busy}, {"ok": True}, {"ok": True}])
        worker = debounce.LifecycleDebouncer(lambda: service, timer_factory=FakeTimer)
        with mock.patch.object(
            debounce, "_capture_scope",
            side_effect=[("/profiles/a", {"KEY": "a"}), ("/profiles/b", {"KEY": "b"})],
        ), mock.patch.object(debounce, "_profile_scope", side_effect=self._scope):
            worker.notify("patched", service=service)
            worker.notify("installed", service=service)
            first_a, first_b = FakeTimer.instances
            self.assertFalse(first_a.cancelled)
            self.assertFalse(first_b.cancelled)
            first_a.fire()  # busy claim: schedules a delayed retry for profile A
            retry_a = FakeTimer.instances[2]
            self.assertEqual(retry_a.interval, 60.0)
            first_b.fire()
            retry_a.fire()

        self.assertEqual(service.apply_calls, [False, False, False])
        self.assertEqual(
            self.bindings,
            [("/profiles/a", {"KEY": "a"}), ("/profiles/b", {"KEY": "b"}),
             ("/profiles/a", {"KEY": "a"})],
        )


if __name__ == "__main__":
    unittest.main()
