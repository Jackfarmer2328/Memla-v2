from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from .memory.ontology import summarize_memory_ontology
from .natural_terminal import (
    BrowserSessionState,
    TERMINAL_MEMORY_ONTOLOGY_FILENAME,
    TerminalExecutionResult,
    TerminalPlan,
    build_llm_client,
    build_raw_terminal_plan,
    build_terminal_plan,
    execute_terminal_plan,
    load_browser_session_state,
    run_terminal_scout,
    terminal_execution_to_dict,
    terminal_memory_ontology_path,
    terminal_model_default,
    terminal_plan_to_dict,
    terminal_scout_to_dict,
)


class MemlaRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = ""
    provider: str = "ollama"
    base_url: str = ""
    temperature: float = 0.1
    heuristic_only: bool = False
    without_memla: bool = False


class MemlaScoutRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    save_state: bool = True


class MemlaFollowupRequest(MemlaRunRequest):
    pass


def _resolve_terminal_defaults(
    *,
    model: str,
    provider: str,
    base_url: str,
    temperature: float,
    heuristic_only: bool,
) -> dict[str, Any]:
    return {
        "model": str(model or terminal_model_default()).strip() or terminal_model_default(),
        "provider": str(provider or "ollama").strip() or "ollama",
        "base_url": str(base_url or "").strip(),
        "temperature": float(temperature),
        "heuristic_only": bool(heuristic_only),
    }


def _resolve_request_value(raw: str, default: str) -> str:
    return str(raw or "").strip() or default


def _resolve_memory_ontology_path(state_path: str | Path | None) -> Path:
    if state_path is None:
        return terminal_memory_ontology_path()
    return (Path(state_path).expanduser().resolve().parent / TERMINAL_MEMORY_ONTOLOGY_FILENAME).resolve()


def _build_run_payload(
    *,
    request: MemlaRunRequest,
    browser_state: BrowserSessionState,
    state_path: str | Path | None,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    model = _resolve_request_value(request.model, str(defaults["model"]))
    provider = _resolve_request_value(request.provider, str(defaults["provider"]))
    base_url = _resolve_request_value(request.base_url, str(defaults["base_url"]))
    temperature = float(request.temperature if request.temperature is not None else defaults["temperature"])
    heuristic_only = bool(request.heuristic_only or defaults["heuristic_only"])

    client = None if heuristic_only and not request.without_memla else build_llm_client(provider=provider, base_url=base_url)
    started = time.perf_counter()
    if request.without_memla:
        plan = build_raw_terminal_plan(
            prompt=request.prompt,
            model=model,
            client=client,
            temperature=temperature,
            browser_state=browser_state,
        )
    else:
        plan = build_terminal_plan(
            prompt=request.prompt,
            model=model,
            client=client,
            heuristic_only=heuristic_only,
            temperature=temperature,
            browser_state=browser_state,
        )
    planning_seconds = round(time.perf_counter() - started, 4)
    execution: TerminalExecutionResult | None = None
    execution_seconds = 0.0
    if plan.actions:
        run_started = time.perf_counter()
        execution = execute_terminal_plan(plan, browser_state=browser_state, state_path=state_path)
        execution_seconds = round(time.perf_counter() - run_started, 4)
    total_seconds = round(planning_seconds + execution_seconds, 4)
    return {
        "ok": bool(execution.ok if execution is not None else False),
        "prompt": request.prompt,
        "mode": "execution" if execution is not None else "plan",
        "plan": terminal_plan_to_dict(plan),
        "execution": terminal_execution_to_dict(execution) if execution is not None else None,
        "planning_duration_seconds": planning_seconds,
        "execution_duration_seconds": execution_seconds,
        "total_duration_seconds": total_seconds,
        "runtime_defaults": {
            "model": model,
            "provider": provider,
            "heuristic_only": heuristic_only,
        },
    }


def create_memla_app(
    *,
    state_path: str | Path | None = None,
    default_model: str = "",
    default_provider: str = "ollama",
    default_base_url: str = "",
    default_temperature: float = 0.1,
    default_heuristic_only: bool = False,
) -> FastAPI:
    defaults = _resolve_terminal_defaults(
        model=default_model,
        provider=default_provider,
        base_url=default_base_url,
        temperature=default_temperature,
        heuristic_only=default_heuristic_only,
    )
    app = FastAPI(
        title="Memla API",
        version="0.1.1",
        description="Thin HTTP wrapper around Memla's bounded terminal/browser runtime.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "memla-api",
            "runtime_defaults": {
                "model": defaults["model"],
                "provider": defaults["provider"],
                "heuristic_only": defaults["heuristic_only"],
            },
        }

    @app.get("/state")
    def state() -> dict[str, Any]:
        browser_state = load_browser_session_state(state_path)
        return {
            "ok": True,
            "state": asdict(browser_state),
        }

    @app.get("/memory")
    def memory() -> dict[str, Any]:
        ontology_path = _resolve_memory_ontology_path(state_path)
        return {
            "ok": True,
            "path": str(ontology_path),
            "summary": summarize_memory_ontology(ontology_path),
        }

    @app.post("/scout")
    def scout(request: MemlaScoutRequest) -> dict[str, Any]:
        browser_state = load_browser_session_state(state_path)
        started = time.perf_counter()
        result = run_terminal_scout(
            request.prompt,
            browser_state=browser_state,
            state_path=state_path,
            save_state=request.save_state,
        )
        total_seconds = round(time.perf_counter() - started, 4)
        return {
            "ok": result.ok,
            "mode": "scout",
            "result": terminal_scout_to_dict(result),
            "total_duration_seconds": total_seconds,
        }

    @app.post("/run")
    def run(request: MemlaRunRequest) -> dict[str, Any]:
        browser_state = load_browser_session_state(state_path)
        return _build_run_payload(
            request=request,
            browser_state=browser_state,
            state_path=state_path,
            defaults=defaults,
        )

    @app.post("/followup")
    def followup(request: MemlaFollowupRequest) -> dict[str, Any]:
        browser_state = load_browser_session_state(state_path)
        return _build_run_payload(
            request=request,
            browser_state=browser_state,
            state_path=state_path,
            defaults=defaults,
        )

    return app


def serve_memla_api(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    state_path: str | Path | None = None,
    default_model: str = "",
    default_provider: str = "ollama",
    default_base_url: str = "",
    default_temperature: float = 0.1,
    default_heuristic_only: bool = False,
) -> None:
    app = create_memla_app(
        state_path=state_path,
        default_model=default_model,
        default_provider=default_provider,
        default_base_url=default_base_url,
        default_temperature=default_temperature,
        default_heuristic_only=default_heuristic_only,
    )
    uvicorn.run(app, host=str(host or "127.0.0.1").strip() or "127.0.0.1", port=int(port))


__all__ = [
    "MemlaFollowupRequest",
    "MemlaRunRequest",
    "MemlaScoutRequest",
    "create_memla_app",
    "serve_memla_api",
]
