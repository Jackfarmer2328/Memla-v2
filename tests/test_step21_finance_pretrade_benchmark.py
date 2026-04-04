from __future__ import annotations

import json

from memory_system.cli import main
from memory_system.distillation.finance_pretrade_benchmark import (
    render_finance_pretrade_markdown,
    run_finance_pretrade_benchmark,
)


def test_run_finance_pretrade_benchmark_repairs_notional_breach(monkeypatch, tmp_path):
    cases_path = tmp_path / "finance_cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "notional_reduce_to_limit",
                "prompt": "Reduce the order to the configured hard limit.",
                "order": {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 7000,
                    "price": 100.0,
                    "route": "NYSE",
                    "ts": 1712251200,
                },
                "account": {"current_position": 12000},
                "market": {"reference_price": 100.0},
                "controls": {
                    "max_order_notional": 500000.0,
                    "approval_notional": 500000.0,
                    "max_long_position": 25000,
                    "max_short_position": 10000,
                    "max_price_deviation_pct": 0.05,
                    "duplicate_window_seconds": 5,
                    "allowed_routes": ["NYSE", "ARCA"],
                },
                "recent_orders": [],
                "expected_outcome": "modify",
                "expected_rule_hits": ["max_order_notional"],
                "expected_actions": ["reduce_quantity"],
                "expected_rewrite": {"quantity": 5000},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyClient:
        def __init__(self, provider: str):
            self.provider = provider
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            if self.provider == "github_models":
                return json.dumps(
                    {
                        "decision": "allow",
                        "predicted_rule_hits": ["max_order_notional"],
                        "next_actions": [],
                        "rewrite": {},
                        "rationale": "Looks safe enough.",
                    }
                )
            if self.calls == 1:
                return json.dumps(
                    {
                        "decision": "allow",
                        "predicted_rule_hits": ["max_order_notional"],
                        "next_actions": [],
                        "rewrite": {},
                        "rationale": "Looks safe enough.",
                    }
                )
            return json.dumps(
                {
                    "decision": "modify",
                    "predicted_rule_hits": ["max_order_notional", "approval_required_notional"],
                    "next_actions": ["reduce_quantity"],
                    "rewrite": {"quantity": 5000},
                    "rationale": "Reduce quantity to the hard notional threshold.",
                }
            )

    monkeypatch.setattr(
        "memory_system.distillation.finance_pretrade_benchmark._build_llm_client",
        lambda provider=None, base_url=None, api_key=None: DummyClient(provider or "ollama"),
    )

    report = run_finance_pretrade_benchmark(
        cases_path=str(cases_path),
        raw_model="grok-3",
        memla_model="qwen3.5:9b",
        raw_provider="github_models",
        memla_provider="ollama",
    )

    assert report["cases"] == 1
    assert report["raw_provider"] == "github_models"
    assert report["memla_provider"] == "ollama"
    assert report["avg_memla_backtest_passed"] > report["avg_raw_backtest_passed"]
    assert report["avg_memla_finance_utility"] > report["avg_raw_finance_utility"]
    assert report["memla_vs_raw_finance_utility_index"] is not None
    assert report["memla_vs_raw_finance_utility_index"] > 1.0
    row = report["rows"][0]
    assert row["raw_decision"] == "allow"
    assert row["memla_decision"] == "modify"
    assert len(row["memla_iteration_trace"]) == 2

    md = render_finance_pretrade_markdown(report)
    assert "# Finance Pre-Trade Benchmark" in md
    assert "Finance utility" in md
    assert "Utility delta" in md


def test_memla_finance_pretrade_benchmark_writes_report_bundle(monkeypatch, capsys, tmp_path):
    captured: dict[str, object] = {}

    def _fake_finance_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "avg_raw_finance_utility": 0.3,
            "avg_memla_finance_utility": 0.8,
            "memla_vs_raw_finance_utility_index": 2.6667,
            "rows": [],
        }

    monkeypatch.setattr(
        "memory_system.cli.run_finance_pretrade_benchmark",
        _fake_finance_benchmark,
    )
    monkeypatch.setattr(
        "memory_system.cli.render_finance_pretrade_markdown",
        lambda report: "# Finance Pre-Trade Benchmark\n",
    )

    out_dir = tmp_path / "finance_report"
    rc = main(
        [
            "finance",
            "benchmark-pretrade",
            "--cases",
            "cases.jsonl",
            "--raw-model",
            "Meta-Llama-3.1-405B-Instruct",
            "--memla-model",
            "qwen3.5:9b",
            "--raw-provider",
            "github_models",
            "--raw-base-url",
            "https://models.github.ai/inference",
            "--memla-provider",
            "ollama",
            "--memla-base-url",
            "http://127.0.0.1:11435",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "finance_pretrade_benchmark_report.json").exists()
    assert (out_dir / "finance_pretrade_benchmark_report.md").exists()
    out = capsys.readouterr().out
    assert "Wrote finance benchmark JSON" in out
    assert "memla utility 0.8" in out
    assert captured["raw_provider"] == "github_models"
    assert captured["memla_provider"] == "ollama"
