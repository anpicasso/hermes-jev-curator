"""Profile-scoped service facade for commands and plugin registration."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Mapping

from .engine import CuratorEngine
from .models import Settings
from .transport import resolve_route


_SETTING_KEYS = (
    "mode", "provider", "base_url", "jev_model", "key_env", "allow_content_egress", "timeout_seconds",
    "max_requests", "max_pairs", "top_k",
)


class CuratorService:
    def __init__(self, settings: Settings, ctx: Any = None):
        self.settings = settings
        self.ctx = ctx
        self.engine = CuratorEngine(settings, ctx)

    def status(self) -> dict[str, Any]:
        from .state import load_state

        result = self.engine.status()
        last = load_state().get("last_run")
        result["last_run"] = last if isinstance(last, Mapping) else None
        return result

    def scan(self) -> dict[str, Any]:
        result = self.engine.scan(use_jev=self.settings.mode != "off")
        self._audit("scan", result)
        return result

    def review(self, name: str) -> dict[str, Any]:
        result = self.engine.review(str(name).strip())
        self._audit("review", result, skill=str(name).strip())
        return result

    def graph(self) -> dict[str, Any]:
        from .graph import build_graph
        from .inventory import collect_inventory
        from .engine import _judgment_from_mapping

        scan = self.engine.scan(use_jev=self.settings.mode != "off")
        judgments = [_judgment_from_mapping(row) for row in scan.get("judgments", [])]
        graph = build_graph(collect_inventory(), judgments)
        result = {
            "ok": bool(scan.get("ok")),
            "version": graph.version,
            "nodes": list(graph.nodes),
            "edges": [asdict(edge) for edge in graph.edges],
            "authorized_edges": len(graph.authorized),
            "refusals": graph.refusal_counts(),
            "skipped": scan.get("skipped", []),
            "errors": scan.get("errors", []),
        }
        self._audit("graph", result)
        return result

    def plan(self) -> dict[str, Any]:
        scan = self.engine.scan(use_jev=self.settings.mode != "off")
        plans = self.engine.build_plans(scan)
        result = {
            "ok": bool(scan.get("ok")),
            "mode": self.settings.mode,
            "plans": [asdict(plan) for plan in plans],
            "applicable": sum(plan.applicable for plan in plans),
            "skipped": scan.get("skipped", []),
            "errors": scan.get("errors", []),
        }
        self._audit("plan", result)
        return result

    def run(self, *, apply: bool = False) -> dict[str, Any]:
        """Run once under the plugin claim; dry by default, explicit apply only."""
        from .state import claim_lock, save_state, write_report

        if apply and self.settings.mode != "apply":
            return {"ok": False, "error": "apply requires mode=apply and an explicit --apply flag"}
        with claim_lock(stale_seconds=max(300.0, self.settings.timeout_seconds * self.settings.max_requests + 60.0)) as held:
            if not held:
                return {"ok": False, "error": "another jev-curator run holds the profile claim"}
            started = time.time()
            scan = self.engine.scan(use_jev=self.settings.mode != "off")
            plans = self.engine.build_plans(scan)
            authorization: dict[str, Any] = {"installed": [], "dropped": [], "ok": True}
            if self.settings.mode in {"guard", "apply"}:
                from .guard import load_authorized_plans, replace_authorized_plans
                previous = {str(row.get("plan_id")) for row in load_authorized_plans()
                            if isinstance(row, Mapping) and row.get("plan_id")}
                installed = replace_authorized_plans(plans)
                authorization = {
                    "installed": installed or [],
                    "dropped": sorted(previous - set(installed or ())) if installed is not None else [],
                    "ok": installed is not None,
                }
            execution = self.engine.apply(plans) if apply else {
                "ok": True, "applied": [], "message": "observe-only; no skill mutations"
            }
            result = {
                "ok": bool(scan.get("ok")) and bool(execution.get("ok")),
                "mode": self.settings.mode,
                "apply_requested": bool(apply),
                "started_at": started,
                "finished_at": time.time(),
                "inventory_count": len(scan.get("inventory", [])),
                "candidate_count": len(scan.get("candidates", [])),
                "judgment_count": len(scan.get("judgments", [])),
                "skipped_count": len(scan.get("skipped", [])),
                "skipped": scan.get("skipped", []),
                "plans": [asdict(plan) for plan in plans],
                "authorization": authorization,
                "errors": scan.get("errors", []),
                "execution": execution,
            }
            run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started)) + f"-{time.time_ns() % 1_000_000:06d}"
            try:
                report_path = write_report(run_id, result)
                result["report"] = str(report_path)
            except Exception as exc:
                result.setdefault("warnings", []).append(f"report write failed: {type(exc).__name__}")
            try:
                save_state({"last_run": _last_run(result, run_id)})
            except Exception as exc:
                result.setdefault("warnings", []).append(f"state write failed: {type(exc).__name__}")
            self._audit("run", result, run_id=run_id)
            return result

    def doctor(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        try:
            route = resolve_route(self.settings)
            checks.append({"name": "route", "ok": True, "provider": self.settings.provider,
                           "host": _safe_host(route.url), "model": route.model})
        except Exception as exc:
            # ponytail: route errors can contain credential-bearing URLs or config keys.
            checks.append({"name": "route", "ok": False,
                           "error": f"{type(exc).__name__}: invalid route configuration"})
        try:
            status = self.engine.status()
            checks.append({"name": "inventory", "ok": True,
                           "managed_skills": status["managed_skills"],
                           "protected_skills": status["protected_skills"]})
        except Exception as exc:
            # ponytail: dependency errors may contain paths, URLs, or credentials.
            checks.append({"name": "inventory", "ok": False,
                           "error": f"{type(exc).__name__}: inventory check failed"})
        checks.append({"name": "mutation-default", "ok": self.settings.mode != "apply",
                       "mode": self.settings.mode,
                       "note": "apply mode still requires an explicit terminal --apply"})
        result = {"ok": all(bool(row["ok"]) for row in checks), "checks": checks}
        self._audit("doctor", result)
        return result

    def lifecycle(self, **event: Any) -> None:
        """Observer hook: append facts only; never mutates skills and never raises."""
        try:
            from .state import audit
            audit("skill_lifecycle", **event)
        except Exception:
            return

    def _audit(self, event: str, result: Mapping[str, Any], **fields: Any) -> None:
        try:
            from .state import audit
            audit(event, ok=bool(result.get("ok")), mode=self.settings.mode,
                  errors=len(result.get("errors", [])) if isinstance(result.get("errors"), list) else 0,
                  **fields)
        except Exception:
            return


def build_service(ctx: Any = None) -> CuratorService:
    return CuratorService(_load_settings(ctx), ctx)


def _load_settings(ctx: Any = None) -> Settings:
    if ctx is not None and callable(getattr(ctx, "get_config", None)):
        return Settings.from_mapping({key: ctx.get_config(key, None) for key in _SETTING_KEYS})
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        entry = (((config.get("plugins") or {}).get("entries") or {}).get("jev-curator") or {})
        raw = entry.get("settings") if isinstance(entry, Mapping) else {}
        return Settings.from_mapping(raw if isinstance(raw, Mapping) else {})
    except Exception:
        return Settings()


def _last_run(result: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ok": bool(result.get("ok")),
        "mode": str(result.get("mode") or ""),
        "finished_at": result.get("finished_at"),
        "inventory_count": int(result.get("inventory_count") or 0),
        "candidate_count": int(result.get("candidate_count") or 0),
        "judgment_count": int(result.get("judgment_count") or 0),
        "skipped_count": int(result.get("skipped_count") or 0),
        "applied_count": len(((result.get("execution") or {}).get("applied") or []))
        if isinstance(result.get("execution"), Mapping) else 0,
    }


def _safe_host(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return str(urlparse(url).hostname or "")
    except Exception:
        return ""
