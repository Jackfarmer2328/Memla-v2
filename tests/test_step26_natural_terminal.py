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
    load_terminal_benchmark_cases,
    run_terminal_benchmark,
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


def test_memla_terminal_plan_without_memla_uses_raw_model(monkeypatch, capsys):
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
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("memla plan path should not run")),
    )

    rc = main(
        [
            "terminal",
            "plan",
            "open",
            "chrome",
            "--without-memla",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "raw_model"
    assert payload["actions"][0]["resolved_target"] == "chrome"


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


def test_memla_terminal_run_without_memla_executes_raw_plan(monkeypatch, capsys):
    plan = TerminalPlan(
        prompt="open chrome",
        source="raw_model",
        actions=[TerminalAction(kind="launch_app", target="chrome", resolved_target="chrome")],
    )

    monkeypatch.setattr("memory_system.cli.build_raw_terminal_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        "memory_system.cli.build_terminal_plan",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("memla plan path should not run")),
    )
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
            "open",
            "chrome",
            "--without-memla",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan_source"] == "raw_model"
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


def test_memla_terminal_compare_accepts_positional_prompt(monkeypatch, capsys):
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
            actions=[TerminalAction(kind="launch_app", target="spotify", resolved_target="spotify")],
        ),
    )

    rc = main(
        [
            "terminal",
            "compare",
            "open",
            "chrome",
            "and",
            "spotify",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt"] == "open chrome and spotify"


def test_terminal_benchmark_pack_loads_expected_cases():
    cases = load_terminal_benchmark_cases("cases/terminal_eval_cases.jsonl")

    assert {case.case_id for case in cases} == {
        "open_chrome",
        "open_chrome_and_spotify",
        "open_downloads",
        "open_github",
        "list_documents",
        "check_disk_usage",
        "show_battery",
        "open_vscode",
    }


def test_terminal_benchmark_reports_latency_and_heuristic_hits(monkeypatch, tmp_path):
    cases_path = tmp_path / "terminal_cases.jsonl"
    cases_path.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "a", "prompt": "open chrome", "expected_actions": ["launch_app:chrome"]}),
                json.dumps({"case_id": "b", "prompt": "open downloads folder", "expected_actions": ["open_path:downloads"]}),
            ]
        ),
        encoding="utf-8",
    )

    class DummyClient:
        provider = "ollama"

    monkeypatch.setattr("memory_system.natural_terminal.build_llm_client", lambda **kwargs: DummyClient())
    monkeypatch.setattr(
        "memory_system.natural_terminal.build_raw_terminal_plan",
        lambda **kwargs: TerminalPlan(
            prompt=kwargs["prompt"],
            source="raw_model",
            actions=[TerminalAction(kind="launch_app", target="chrome", resolved_target="chrome")]
            if "chrome" in kwargs["prompt"]
            else [TerminalAction(kind="open_path", target="downloads", resolved_target="/tmp/Downloads")],
        ),
    )
    monkeypatch.setattr(
        "memory_system.natural_terminal.build_terminal_plan",
        lambda **kwargs: TerminalPlan(
            prompt=kwargs["prompt"],
            source="heuristic",
            actions=[TerminalAction(kind="launch_app", target="chrome", resolved_target="chrome")]
            if "chrome" in kwargs["prompt"]
            else [TerminalAction(kind="open_path", target="downloads", resolved_target="/tmp/Downloads")],
        ),
    )
    timeline = iter([0.0, 1.0, 1.0, 1.1, 2.0, 2.8, 2.8, 2.9])
    monkeypatch.setattr("memory_system.natural_terminal.time.perf_counter", lambda: next(timeline))

    report = run_terminal_benchmark(
        cases_path=str(cases_path),
        raw_model="phi3",
        memla_model="phi3",
    )

    assert report["cases"] == 2
    assert report["avg_raw_latency_ms"] == 900.0
    assert report["avg_memla_latency_ms"] == 100.0
    assert report["memla_vs_raw_speedup"] == 9.0
    assert report["memla_heuristic_hit_count"] == 2
    assert report["avg_memla_terminal_utility"] == 1.0


def test_memla_terminal_benchmark_writes_report_bundle(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "memory_system.cli.run_terminal_benchmark",
        lambda **kwargs: {
            "avg_raw_latency_ms": 950.0,
            "avg_memla_latency_ms": 25.0,
            "memla_vs_raw_speedup": 38.0,
            "rows": [],
        },
    )
    monkeypatch.setattr(
        "memory_system.cli.render_terminal_benchmark_markdown",
        lambda report: "# Terminal Benchmark\n",
    )

    out_dir = tmp_path / "terminal_benchmark"
    rc = main(
        [
            "terminal",
            "benchmark",
            "--model",
            "phi3",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "terminal_benchmark_report.json").exists()
    assert (out_dir / "terminal_benchmark_report.md").exists()
    out = capsys.readouterr().out
    assert "Wrote terminal benchmark JSON" in out
    assert "speedup 38.0x" in out
