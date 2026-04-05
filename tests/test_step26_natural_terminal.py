from __future__ import annotations

import json
from pathlib import Path

from memory_system.cli import main
from memory_system.natural_terminal import (
    TerminalExecutionRecord,
    TerminalExecutionResult,
    TerminalPlan,
    TerminalAction,
    build_terminal_plan,
    execute_terminal_plan,
)


def test_terminal_heuristic_plan_launches_multiple_apps():
    plan = build_terminal_plan(prompt="open chrome and spotify", heuristic_only=True)

    assert plan.source == "heuristic"
    assert [action.kind for action in plan.actions] == ["launch_app", "launch_app"]
    assert [action.resolved_target for action in plan.actions] == ["chrome", "spotify"]


def test_terminal_heuristic_plan_maps_downloads_folder():
    plan = build_terminal_plan(prompt="open downloads folder", heuristic_only=True)

    assert [action.kind for action in plan.actions] == ["open_path"]
    assert plan.actions[0].resolved_target.endswith("Downloads")


def test_terminal_execute_plan_launches_linux_apps(monkeypatch):
    launched: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        if name == "google-chrome-stable":
            return "/usr/bin/google-chrome-stable"
        if name == "spotify":
            return "/usr/bin/spotify"
        return None

    class DummyProcess:
        def __init__(self, command, **kwargs):
            launched.append(list(command))

    monkeypatch.setattr("memory_system.natural_terminal.shutil.which", fake_which)
    monkeypatch.setattr("memory_system.natural_terminal.subprocess.Popen", DummyProcess)

    plan = build_terminal_plan(prompt="open chrome and spotify", heuristic_only=True)
    result = execute_terminal_plan(plan, platform_name="linux")

    assert result.ok is True
    assert launched == [
        ["/usr/bin/google-chrome-stable"],
        ["/usr/bin/spotify"],
    ]


def test_memla_terminal_plan_json_outputs_structured_plan(capsys):
    rc = main(
        [
            "terminal",
            "plan",
            "--prompt",
            "open chrome and spotify",
            "--heuristic-only",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "heuristic"
    assert [action["resolved_target"] for action in payload["actions"]] == ["chrome", "spotify"]


def test_memla_terminal_run_json_outputs_execution(monkeypatch, capsys):
    plan = TerminalPlan(prompt="open chrome", source="heuristic", actions=build_terminal_plan(prompt="open chrome", heuristic_only=True).actions)

    monkeypatch.setattr("memory_system.cli.build_terminal_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        "memory_system.cli.execute_terminal_plan",
        lambda current_plan: TerminalExecutionResult(
            prompt=current_plan.prompt,
            plan_source=current_plan.source,
            ok=True,
            records=[
                TerminalExecutionRecord(
                    kind="launch_app",
                    target="chrome",
                    status="ok",
                    message="Launched chrome.",
                    command=["/usr/bin/google-chrome"],
                )
            ],
        ),
    )

    rc = main(
        [
            "terminal",
            "run",
            "--prompt",
            "open chrome",
            "--heuristic-only",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["records"][0]["command"] == ["/usr/bin/google-chrome"]


def test_memla_terminal_compare_json_outputs_raw_and_memla(monkeypatch, capsys):
    monkeypatch.setattr(
        "memory_system.cli.build_raw_terminal_plan",
        lambda **kwargs: TerminalPlan(
            prompt=kwargs["prompt"],
            source="raw_model",
            actions=[TerminalAction(kind="launch_app", target="chrome", resolved_target="chrome")],
        ),
    )
    monkeypatch.setattr(
        "memory_system.cli.build_terminal_plan",
        lambda **kwargs: TerminalPlan(
            prompt=kwargs["prompt"],
            source="heuristic",
            actions=[
                TerminalAction(kind="launch_app", target="chrome", resolved_target="chrome"),
                TerminalAction(kind="launch_app", target="spotify", resolved_target="spotify"),
            ],
        ),
    )

    rc = main(
        [
            "terminal",
            "compare",
            "--prompt",
            "open chrome and spotify",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["raw"]["source"] == "raw_model"
    assert [action["resolved_target"] for action in payload["memla"]["actions"]] == ["chrome", "spotify"]
