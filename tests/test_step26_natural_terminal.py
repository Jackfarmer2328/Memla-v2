from __future__ import annotations

import json
from pathlib import Path

from memory_system.cli import main
from memory_system.natural_terminal import (
    BrowserSessionState,
    TerminalExecutionRecord,
    TerminalExecutionResult,
    TerminalPlan,
    TerminalAction,
    _fetch_search_result_urls,
    build_raw_terminal_plan,
    build_terminal_step_report,
    build_terminal_plan,
    execute_terminal_step,
    execute_terminal_plan,
    load_terminal_benchmark_cases,
    save_browser_session_state,
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


def test_terminal_heuristic_plan_builds_search_urls():
    plan = build_terminal_plan(prompt="open youtube and search lo fi hip hop", heuristic_only=True)

    assert plan.source == "heuristic"
    assert [action.kind for action in plan.actions] == ["open_url"]
    assert plan.actions[0].resolved_target == "https://www.youtube.com/results?search_query=lo+fi+hip+hop"


def test_terminal_heuristic_plan_builds_search_urls_for_casual_phrase():
    plan = build_terminal_plan(
        prompt="open youtube bro then i want you to search lo fi hip hop",
        heuristic_only=True,
    )

    assert plan.source == "heuristic"
    assert [action.kind for action in plan.actions] == ["open_url"]
    assert plan.actions[0].resolved_target == "https://www.youtube.com/results?search_query=lo+fi+hip+hop"


def test_terminal_heuristic_plan_uses_browser_state_for_follow_up():
    browser_state = BrowserSessionState(
        current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
        page_kind="search_results",
        search_engine="youtube",
        search_query="lo fi hip hop",
    )

    plan = build_terminal_plan(
        prompt="now press on the video so i can listen",
        heuristic_only=True,
        browser_state=browser_state,
    )

    assert plan.source == "heuristic"
    assert [action.kind for action in plan.actions] == ["open_search_result"]
    assert plan.actions[0].resolved_target == "1"


def test_terminal_heuristic_plan_handles_click_first_vid_follow_up():
    browser_state = BrowserSessionState(
        current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
        page_kind="search_results",
        search_engine="youtube",
        search_query="lo fi hip hop",
    )

    plan = build_terminal_plan(
        prompt="now click the first vid",
        heuristic_only=True,
        browser_state=browser_state,
    )

    assert plan.source == "heuristic"
    assert [action.kind for action in plan.actions] == ["open_search_result"]
    assert plan.actions[0].resolved_target == "1"


def test_terminal_heuristic_plan_reads_current_repo_page():
    browser_state = BrowserSessionState(
        current_url="https://github.com/ggml-org/llama.cpp",
        page_kind="repo_page",
    )

    plan = build_terminal_plan(
        prompt="what is this repo",
        heuristic_only=True,
        browser_state=browser_state,
    )

    assert plan.source == "heuristic"
    assert [action.kind for action in plan.actions] == ["browser_read_page"]


def test_terminal_step_report_includes_prompt_and_state_candidates():
    browser_state = BrowserSessionState(
        current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
        page_kind="search_results",
        search_engine="youtube",
        search_query="lo fi hip hop",
    )

    report = build_terminal_step_report(
        prompt="now click the first vid",
        heuristic_only=True,
        browser_state=browser_state,
    )

    assert report.constraints["page_kind"] == "search_results"
    assert report.constraints["search_engine"] == "youtube"
    assert report.candidates[0].recommended is True
    assert report.candidates[0].plan.actions[0].kind == "open_search_result"
    labels = [candidate.label for candidate in report.candidates]
    assert "Open search result #2" in labels


def test_raw_terminal_plan_receives_browser_context():
    captured_messages = {}

    class DummyClient:
        def chat(self, *, model, messages, temperature):
            captured_messages["messages"] = messages
            return json.dumps({"actions": [{"kind": "open_search_result", "target": "1"}]})

    browser_state = BrowserSessionState(
        current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
        page_kind="search_results",
        search_engine="youtube",
        search_query="lo fi hip hop",
        result_urls=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )

    plan = build_raw_terminal_plan(
        prompt="click the first video",
        model="phi3:mini",
        client=DummyClient(),
        browser_state=browser_state,
    )

    assert plan.source == "raw_model"
    assert plan.actions[0].kind == "open_search_result"
    system_message = captured_messages["messages"][0].content
    assert "Current browser URL" in system_message
    assert "youtube" in system_message
    assert "Cached browser results: 1" in system_message


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


def test_terminal_execute_plan_opens_first_search_result(monkeypatch, tmp_path):
    launched: list[list[str]] = []

    class DummyProcess:
        def __init__(self, command, **kwargs):
            launched.append(list(command))

    state_path = tmp_path / "browser_state.json"
    save_browser_session_state(
        BrowserSessionState(
            current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
            page_kind="search_results",
            search_engine="youtube",
            search_query="lo fi hip hop",
        ),
        state_path,
    )
    monkeypatch.setattr("memory_system.natural_terminal.subprocess.Popen", DummyProcess)
    monkeypatch.setattr(
        "memory_system.natural_terminal._fetch_search_result_urls",
        lambda engine, query, limit=5: ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )

    plan = TerminalPlan(
        prompt="click the first video",
        source="heuristic",
        actions=[TerminalAction(kind="open_search_result", target="1", resolved_target="1")],
    )
    result = execute_terminal_plan(plan, platform_name="linux", state_path=state_path)

    assert result.ok is True
    assert launched == [["xdg-open", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]]
    assert result.browser_state["page_kind"] == "video_page"
    assert result.browser_state["current_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_fetch_search_result_urls_uses_github_api_first(monkeypatch):
    def fake_fetch(url: str, *, accept: str = "text/html") -> str:
        assert "api.github.com/search/repositories" in url
        assert accept == "application/vnd.github+json"
        return json.dumps(
            {
                "items": [
                    {"html_url": "https://github.com/ggml-org/llama.cpp"},
                    {"html_url": "https://github.com/oobabooga/text-generation-webui"},
                ]
            }
        )

    monkeypatch.setattr("memory_system.natural_terminal._fetch_url_text", fake_fetch)

    results = _fetch_search_result_urls("github", "llama.cpp", limit=2)

    assert results == [
        "https://github.com/ggml-org/llama.cpp",
        "https://github.com/oobabooga/text-generation-webui",
    ]


def test_terminal_execute_plan_reads_current_repo_page(monkeypatch, tmp_path):
    state_path = tmp_path / "browser_state.json"
    save_browser_session_state(
        BrowserSessionState(
            current_url="https://github.com/ggml-org/llama.cpp",
            page_kind="repo_page",
        ),
        state_path,
    )
    html = """
    <html>
      <head>
        <title>ggml-org/llama.cpp: LLM inference in C/C++</title>
        <meta property="og:description" content="Inference of LLaMA models in pure C/C++." />
      </head>
      <body>
        <a href="/ggml-org/llama.cpp/stargazers"><span>77.7k</span></a>
        <a href="/ggml-org/llama.cpp/forks"><span>11.2k</span></a>
      </body>
    </html>
    """
    monkeypatch.setattr("memory_system.natural_terminal._fetch_page_html", lambda url: html)
    monkeypatch.setattr(
        "memory_system.natural_terminal._fetch_github_repo_snapshot",
        lambda owner, repo: {
            "repo": "ggml-org/llama.cpp",
            "description": "Inference of LLaMA models in pure C/C++.",
            "summary": "ggml-org/llama.cpp: Inference of LLaMA models in pure C/C++.",
            "stars": "77.7k",
            "forks": "11.2k",
            "language": "C++",
            "topics": "llm, inference, cpp",
        },
    )

    plan = TerminalPlan(
        prompt="what is this repo",
        source="heuristic",
        actions=[TerminalAction(kind="browser_read_page", target="current_page", resolved_target="current_page")],
    )
    result = execute_terminal_plan(plan, platform_name="linux", state_path=state_path)

    assert result.ok is True
    assert result.records[0].details["repo"] == "ggml-org/llama.cpp"
    assert result.records[0].details["stars"] == "77.7k"
    assert result.records[0].details["forks"] == "11.2k"
    assert "Repo summary:" in result.records[0].message
    assert "language C++" in result.records[0].message


def test_execute_terminal_step_logs_trace(monkeypatch, tmp_path):
    state_path = tmp_path / "browser_state.json"
    trace_path = tmp_path / "trace.jsonl"
    save_browser_session_state(
        BrowserSessionState(
            current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
            page_kind="search_results",
            search_engine="youtube",
            search_query="lo fi hip hop",
        ),
        state_path,
    )

    report = build_terminal_step_report(
        prompt="now click the first vid",
        heuristic_only=True,
        browser_state=BrowserSessionState(
            current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
            page_kind="search_results",
            search_engine="youtube",
            search_query="lo fi hip hop",
        ),
    )
    monkeypatch.setattr(
        "memory_system.natural_terminal.execute_terminal_plan",
        lambda plan, **kwargs: TerminalExecutionResult(
            prompt=plan.prompt,
            plan_source=plan.source,
            ok=True,
            records=[TerminalExecutionRecord(kind="open_search_result", target="1", status="ok", message="Opened result.")],
        ),
    )

    execution = execute_terminal_step(
        report,
        choice="1",
        state_path=state_path,
        trace_path=trace_path,
    )

    assert execution.result.ok is True
    assert execution.chosen_candidate.candidate_id == "prompt_plan"
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["chosen_candidate_id"] == "prompt_plan"
    assert payload["execution"]["ok"] is True


def test_memla_terminal_plan_json_outputs_structured_plan(monkeypatch, capsys):
    timeline = iter([0.0, 0.25])
    monkeypatch.setattr("memory_system.cli.time.perf_counter", lambda: next(timeline))
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
    assert payload["planning_duration_seconds"] == 0.25


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
    timeline = iter([1.0, 1.2, 1.2, 1.35])

    monkeypatch.setattr("memory_system.cli.build_terminal_plan", lambda **kwargs: plan)
    monkeypatch.setattr("memory_system.cli.time.perf_counter", lambda: next(timeline))
    monkeypatch.setattr(
        "memory_system.cli.execute_terminal_plan",
        lambda current_plan, **kwargs: TerminalExecutionResult(
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
    assert payload["planning_duration_seconds"] == 0.2
    assert payload["execution_duration_seconds"] == 0.15
    assert payload["total_duration_seconds"] == 0.35


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
        lambda current_plan, **kwargs: TerminalExecutionResult(
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


def test_memla_terminal_step_json_outputs_candidates(monkeypatch, capsys):
    timeline = iter([3.0, 3.25])
    monkeypatch.setattr("memory_system.cli.time.perf_counter", lambda: next(timeline))
    monkeypatch.setattr(
        "memory_system.cli.build_terminal_step_report",
        lambda **kwargs: build_terminal_step_report(
            prompt=kwargs["prompt"],
            heuristic_only=True,
            browser_state=BrowserSessionState(
                current_url="https://www.youtube.com/results?search_query=lo+fi+hip+hop",
                page_kind="search_results",
                search_engine="youtube",
                search_query="lo fi hip hop",
            ),
        ),
    )

    rc = main(
        [
            "terminal",
            "step",
            "now",
            "click",
            "the",
            "first",
            "vid",
            "--heuristic-only",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["constraints"]["page_kind"] == "search_results"
    assert payload["candidates"][0]["recommended"] is True
    assert payload["planning_duration_seconds"] == 0.25


def test_memla_terminal_step_choice_executes_candidate(monkeypatch, capsys, tmp_path):
    timeline = iter([5.0, 5.2, 5.2, 5.35])
    report = build_terminal_step_report(
        prompt="what is this repo",
        heuristic_only=True,
        browser_state=BrowserSessionState(
            current_url="https://github.com/ggml-org/llama.cpp",
            page_kind="repo_page",
        ),
    )
    monkeypatch.setattr("memory_system.cli.time.perf_counter", lambda: next(timeline))
    monkeypatch.setattr("memory_system.cli.build_terminal_step_report", lambda **kwargs: report)
    monkeypatch.setattr(
        "memory_system.cli.execute_terminal_step",
        lambda current_report, **kwargs: type(
            "DummyExecution",
            (),
            {
                "report": current_report,
                "chosen_candidate": current_report.candidates[0],
                "result": TerminalExecutionResult(
                    prompt=current_report.prompt,
                    plan_source=current_report.candidates[0].plan.source,
                    ok=True,
                    records=[TerminalExecutionRecord(kind="browser_read_page", target="current_page", status="ok", message="Read page.")],
                ),
                "trace_path": str(tmp_path / "trace.jsonl"),
            },
        )(),
    )

    rc = main(
        [
            "terminal",
            "step",
            "what",
            "is",
            "this",
            "repo",
            "--heuristic-only",
            "--choice",
            "1",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chosen_candidate"]["candidate_id"] == "prompt_plan"
    assert payload["result"]["ok"] is True
    assert payload["planning_duration_seconds"] == 0.2
    assert payload["execution_duration_seconds"] == 0.15
    assert payload["total_duration_seconds"] == 0.35


def test_memla_terminal_compare_json_outputs_raw_and_memla(monkeypatch, capsys):
    timeline = iter([2.0, 4.0, 4.0, 4.5])

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
    monkeypatch.setattr("memory_system.cli.time.perf_counter", lambda: next(timeline))

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
    assert payload["raw_duration_seconds"] == 2.0
    assert payload["memla_duration_seconds"] == 0.5
    assert payload["memla_speedup"] == 4.0


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
        "search_youtube_lofi",
        "search_github_llamacpp",
        "search_reddit_local_llm",
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
