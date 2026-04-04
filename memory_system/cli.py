from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .distillation.coding_proxy import CodingSession
from .distillation.compile_loop_benchmark import (
    render_compile_loop_benchmark_markdown,
    run_compile_loop_benchmark,
)
from .distillation.coding_c2a_benchmark import (
    render_coding_c2a_markdown,
    run_coding_c2a_benchmark,
)
from .distillation.finance_pretrade_benchmark import (
    render_finance_pretrade_markdown,
    run_finance_pretrade_benchmark,
)
from .distillation.healthcare_denial_benchmark import (
    render_healthcare_denial_markdown,
    run_healthcare_denial_benchmark,
)
from .distillation.finance_policy_bank import (
    distill_finance_policy_bank,
    render_finance_policy_bank_markdown,
)
from .distillation.finance_trace_bank import (
    extract_finance_trace_bank,
    render_finance_trace_bank_markdown,
)
from .distillation.c2a_trace_bank import (
    extract_c2a_trace_bank,
    render_c2a_trace_bank_markdown,
)
from .distillation.c2a_policy_bank import (
    distill_c2a_policy_bank,
    render_c2a_policy_bank_markdown,
)
from .distillation.math_c2a_benchmark import (
    render_math_c2a_teacher_student_markdown,
    run_math_c2a_teacher_student_benchmark,
)
from .distillation.patch_execution_benchmark import (
    render_patch_execution_markdown,
    run_patch_execution_benchmark,
)
from .distillation.thesis_pack_builder import build_thesis_pack
from .distillation.workflow_planner import render_workflow_plan_block


def _coding_model_default() -> str:
    return os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")


def _user_id_default() -> str:
    return os.environ.get("USER_ID", "default")


def _timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=_json_default))


def _resolve_repo_root(raw: str) -> Path:
    return Path(raw or ".").resolve()


def _resolve_db_path(raw: str, repo_root: Path) -> Path:
    if raw.strip():
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
    else:
        path = repo_root / ".memla" / "memory.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _default_report_dir(kind: str) -> Path:
    out_dir = Path.cwd() / "memla_reports" / f"{kind}_{_timestamp_slug()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


_PUBLIC_SITE_VERCEL_CONFIG = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "cleanUrls": True,
    "trailingSlash": False,
    "builds": [
        {"src": "index.html", "use": "@vercel/static"},
        {"src": "og-card.svg", "use": "@vercel/static"},
        {"src": "90_second_demo.md", "use": "@vercel/static"},
        {"src": "one_sentence_pitch.txt", "use": "@vercel/static"},
        {"src": "strategic_memo.md", "use": "@vercel/static"},
        {"src": "frozen/**", "use": "@vercel/static"},
    ],
}


def _candidate_ollama_urls(preferred: str) -> list[str]:
    urls: list[str] = []
    for raw in (preferred, os.environ.get("OLLAMA_URL", ""), "http://127.0.0.1:11434", "http://127.0.0.1:11435"):
        clean = str(raw or "").strip().rstrip("/")
        if not clean or clean in urls:
            continue
        urls.append(clean)
    return urls


def _probe_ollama(url: str, *, timeout: float = 2.0) -> dict[str, Any]:
    target = f"{url.rstrip('/')}/api/tags"
    req = urllib_request.Request(target, headers={"Accept": "application/json"})
    with urllib_request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models") or []
    names = [str(model.get("name") or "").strip() for model in models if str(model.get("name") or "").strip()]
    return {
        "reachable": True,
        "url": url,
        "model_count": len(names),
        "models": names,
    }


def _sync_public_site(*, source_dir: Path, out_dir: Path) -> dict[str, Any]:
    source = source_dir.resolve()
    target = out_dir.resolve()
    files = [
        "index.html",
        "og-card.svg",
        "one_sentence_pitch.txt",
        "90_second_demo.md",
        "strategic_memo.md",
    ]
    copied: list[str] = []
    target.mkdir(parents=True, exist_ok=True)
    for name in files:
        src_file = source / name
        if not src_file.exists():
            continue
        dst_file = target / name
        shutil.copy2(src_file, dst_file)
        copied.append(name)
    frozen_src = source / "frozen"
    frozen_dst = target / "frozen"
    if frozen_dst.exists():
        shutil.rmtree(frozen_dst)
    if frozen_src.exists():
        shutil.copytree(frozen_src, frozen_dst)
        copied.append("frozen/")
    (target / "vercel.json").write_text(json.dumps(_PUBLIC_SITE_VERCEL_CONFIG, indent=2), encoding="utf-8")
    copied.append("vercel.json")
    return {
        "source_dir": str(source),
        "out_dir": str(target),
        "copied": copied,
        "site_ready": (target / "index.html").exists() and (target / "vercel.json").exists(),
    }


def _render_proxy_text(result: Any) -> str:
    lines = [str(result.answer or "").strip()]
    if result.suggested_files:
        lines.append("")
        lines.append(f"Likely files: {', '.join(result.suggested_files[:8])}")
    if result.suggested_commands:
        lines.append(f"Likely commands: {', '.join(result.suggested_commands[:6])}")
    if result.likely_tests:
        lines.append(f"Likely tests: {', '.join(result.likely_tests[:6])}")
    if result.predicted_constraints:
        lines.append(f"Predicted constraints: {', '.join(result.predicted_constraints[:6])}")
    if result.transmutations:
        lines.append(f"Transmutations: {', '.join(result.transmutations[:6])}")
    if result.residual_constraints:
        lines.append(f"Residual constraints: {', '.join(result.residual_constraints[:6])}")
    validated_trade_path = dict(result.validated_trade_path or {})
    validated_files = [str(item).strip() for item in validated_trade_path.get("supporting_files") or [] if str(item).strip()]
    validated_commands = [str(item).strip() for item in validated_trade_path.get("supporting_commands") or [] if str(item).strip()]
    if validated_files:
        lines.append(f"Validated files: {', '.join(validated_files[:6])}")
    if validated_commands:
        lines.append(f"Validated commands: {', '.join(validated_commands[:6])}")
    test_result = dict(result.test_result or {})
    if test_result:
        command = str(test_result.get("command") or "").strip()
        status = str(test_result.get("status") or "").strip()
        if command or status:
            lines.append(f"Test result: {(status or 'unknown').upper()} {command}".rstrip())
    return "\n".join(lines).strip()


def _write_report_bundle(*, report: dict[str, Any], markdown: str, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _handle_coding_run(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    db_path = _resolve_db_path(args.db, repo_root)
    session = CodingSession(
        model=args.model,
        db_path=str(db_path),
        user_id=args.user_id,
        repo_root=str(repo_root),
        temperature=args.temperature,
        top_k=args.top_k,
        num_ctx=args.num_ctx,
        enable_compile_loop=not args.disable_compile_loop,
        c2a_policy_path=args.c2a_policy_path,
        disable_c2a_policy=args.disable_c2a_policy,
    )
    try:
        result = session.ask(args.prompt, test_command=args.test_command)
    finally:
        session.close()
    if args.json:
        _print_json(result)
    else:
        print(_render_proxy_text(result))
    return 0


def _handle_coding_plan(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    db_path = _resolve_db_path(args.db, repo_root)
    session = CodingSession(
        model=args.model,
        db_path=str(db_path),
        user_id=args.user_id,
        repo_root=str(repo_root),
        temperature=args.temperature,
        top_k=args.top_k,
        num_ctx=args.num_ctx,
        enable_compile_loop=not args.disable_compile_loop,
        c2a_policy_path=args.c2a_policy_path,
        disable_c2a_policy=args.disable_c2a_policy,
    )
    try:
        plan = session.build_plan(args.prompt)
    finally:
        session.close()
    if args.json:
        _print_json(plan)
    else:
        print(render_workflow_plan_block(plan).strip() or "No workflow plan generated.")
    return 0


def _handle_patch_benchmark(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(".")
    db_path = _resolve_db_path(args.db, repo_root)
    report = run_patch_execution_benchmark(
        pack_path=args.pack,
        split=args.split,
        raw_model=args.raw_model,
        memla_model=args.memla_model,
        db_path=str(db_path),
        user_id=args.user_id,
        limit=args.limit,
        top_k=args.top_k,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        raw_iterations=args.raw_iterations,
        memla_iterations=args.memla_iterations,
        raw_provider=args.raw_provider,
        raw_base_url=args.raw_base_url,
        memla_provider=args.memla_provider,
        memla_base_url=args.memla_base_url,
        memla_c2a_policy_path=args.memla_c2a_policy_path,
        disable_memla_c2a_policy=args.disable_memla_c2a_policy,
    )
    markdown = render_patch_execution_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("patch_benchmark")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="patch_execution_report",
    )
    print(f"Wrote patch benchmark JSON: {json_path}")
    print(f"Wrote patch benchmark Markdown: {md_path}")
    print(
        "Summary: "
        f"raw apply {report.get('raw_apply_rate', 0.0)} | "
        f"memla apply {report.get('memla_apply_rate', 0.0)} | "
        f"raw semantic {report.get('avg_raw_semantic_command_success_rate', 0.0)} | "
        f"memla semantic {report.get('avg_memla_semantic_command_success_rate', 0.0)}"
    )
    return 0


def _handle_compile_benchmark(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    db_path = _resolve_db_path(args.db, repo_root)
    report = run_compile_loop_benchmark(
        db_path=str(db_path),
        repo_root=str(repo_root),
        user_id=args.user_id,
        cases_path=args.cases,
        model=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        num_ctx=args.num_ctx,
        memla_c2a_policy_path=args.memla_c2a_policy_path,
        disable_memla_c2a_policy=args.disable_memla_c2a_policy,
    )
    markdown = render_compile_loop_benchmark_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("compile_benchmark")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="compile_loop_benchmark_report",
    )
    print(f"Wrote compile benchmark JSON: {json_path}")
    print(f"Wrote compile benchmark Markdown: {md_path}")
    print(
        "Summary: "
        f"raw command recall {report.get('avg_raw_command_recall', 0.0)} | "
        f"compile combined command recall {report.get('avg_compile_combined_command_recall', 0.0)} | "
        f"compile validated command recall {report.get('avg_compile_validated_command_recall', 0.0)}"
    )
    return 0


def _handle_c2a_benchmark(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    db_path = _resolve_db_path(args.db, repo_root)
    report = run_coding_c2a_benchmark(
        db_path=str(db_path),
        repo_root=str(repo_root),
        user_id=args.user_id,
        cases_path=args.cases,
        raw_model=args.raw_model,
        memla_model=args.memla_model,
        temperature=args.temperature,
        top_k=args.top_k,
        num_ctx=args.num_ctx,
        raw_provider=args.raw_provider,
        raw_base_url=args.raw_base_url,
        memla_provider=args.memla_provider,
        memla_base_url=args.memla_base_url,
        memla_c2a_policy_path=args.memla_c2a_policy_path,
        disable_memla_c2a_policy=args.disable_memla_c2a_policy,
    )
    markdown = render_coding_c2a_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("coding_c2a_benchmark")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="coding_c2a_benchmark_report",
    )
    print(f"Wrote coding C2A benchmark JSON: {json_path}")
    print(f"Wrote coding C2A benchmark Markdown: {md_path}")
    utility_index = report.get("memla_vs_raw_c2a_utility_index")
    utility_text = utility_index if utility_index is not None else "n/a"
    print(
        "Summary: "
        f"raw utility {report.get('avg_raw_c2a_utility', 0.0)} | "
        f"memla utility {report.get('avg_memla_c2a_utility', 0.0)} | "
        f"utility index {utility_text}"
    )
    return 0


def _handle_extract_c2a(args: argparse.Namespace) -> int:
    report = extract_c2a_trace_bank(
        report_paths=list(args.report or []),
        min_utility_delta=args.min_delta,
    )
    markdown = render_c2a_trace_bank_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("coding_c2a_extract")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="c2a_trace_bank_summary",
    )
    jsonl_path = out_dir / "c2a_trace_bank.jsonl"
    jsonl_lines = [json.dumps(row, ensure_ascii=True) for row in report.get("rows", [])]
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
    if args.json:
        _print_json(
            {
                "json_summary": str(json_path),
                "markdown_summary": str(md_path),
                "jsonl_bank": str(jsonl_path),
                "rows_extracted": report.get("rows_extracted", 0),
                "winner_counts": report.get("winner_counts", {}),
                "teacher_signal_class_counts": report.get("teacher_signal_class_counts", {}),
            }
        )
    else:
        print(f"Wrote C2A trace bank JSON summary: {json_path}")
        print(f"Wrote C2A trace bank Markdown summary: {md_path}")
        print(f"Wrote C2A trace bank JSONL: {jsonl_path}")
        print(
            "Summary: "
            f"rows {report.get('rows_extracted', 0)} | "
            f"winner counts {report.get('winner_counts', {})} | "
            f"teacher signals {report.get('teacher_signal_class_counts', {})}"
        )
    return 0


def _handle_distill_c2a(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    out_path = Path(args.out).resolve() if args.out else (repo_root / ".memla" / "c2a_policy_bank.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = distill_c2a_policy_bank(
        trace_bank_path=args.trace_bank,
        min_priority=args.min_priority,
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_c2a_policy_bank_markdown(report), encoding="utf-8")
    if args.json:
        _print_json(
            {
                "policy_bank": str(out_path),
                "markdown_summary": str(md_path),
                "rows_used": report.get("rows_used", 0),
                "source_models": report.get("source_models", {}),
            }
        )
    else:
        print(f"Wrote C2A policy bank JSON: {out_path}")
        print(f"Wrote C2A policy bank Markdown: {md_path}")
        print(
            "Summary: "
            f"rows used {report.get('rows_used', 0)} | "
            f"source models {report.get('source_models', {})}"
        )
    return 0


def _handle_math_benchmark(args: argparse.Namespace) -> int:
    report = run_math_c2a_teacher_student_benchmark(
        cases_path=args.cases,
        teacher_model=args.teacher_model,
        student_models=args.student_models,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        max_iterations=args.max_iterations,
        top_k=args.top_k,
        executor_mode=args.executor_mode,
        teacher_trace_source=args.teacher_trace_source,
    )
    markdown = render_math_c2a_teacher_student_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("math_benchmark")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="math_c2a_teacher_student_report",
    )
    print(f"Wrote math benchmark JSON: {json_path}")
    print(f"Wrote math benchmark Markdown: {md_path}")
    if args.executor_mode == "stepwise_rerank":
        lane_lines = [
            f"{lane.get('lane_id')}: choice={lane.get('avg_choice_accuracy', 0.0)} ambiguous={lane.get('avg_ambiguous_choice_accuracy', 0.0)}"
            for lane in report.get("lanes", [])
        ]
    else:
        lane_lines = [
            f"{lane.get('lane_id')}: answer={lane.get('avg_answer_accuracy', 0.0)} transmutation={lane.get('avg_transmutation_recall', 0.0)}"
            for lane in report.get("lanes", [])
        ]
    print("Summary:")
    for line in lane_lines:
        print(f"  {line}")
    return 0


def _handle_finance_pretrade_benchmark(args: argparse.Namespace) -> int:
    report = run_finance_pretrade_benchmark(
        cases_path=args.cases,
        repo_root=str(_resolve_repo_root(args.repo_root)),
        case_ids=list(args.case_id or []),
        limit=args.limit,
        raw_model=args.raw_model,
        memla_model=args.memla_model,
        raw_iterations=args.raw_iterations,
        memla_iterations=args.memla_iterations,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        raw_provider=args.raw_provider,
        raw_base_url=args.raw_base_url,
        memla_provider=args.memla_provider,
        memla_base_url=args.memla_base_url,
        memla_finance_policy_path=args.memla_finance_policy_path,
        disable_memla_finance_policy=args.disable_memla_finance_policy,
    )
    markdown = render_finance_pretrade_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("finance_pretrade_benchmark")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="finance_pretrade_benchmark_report",
    )
    print(f"Wrote finance benchmark JSON: {json_path}")
    print(f"Wrote finance benchmark Markdown: {md_path}")
    utility_index = report.get("memla_vs_raw_finance_utility_index")
    utility_text = utility_index if utility_index is not None else "n/a"
    print(
        "Summary: "
        f"raw utility {report.get('avg_raw_finance_utility', 0.0)} | "
        f"memla utility {report.get('avg_memla_finance_utility', 0.0)} | "
        f"utility index {utility_text}"
    )
    return 0


def _handle_extract_finance_pretrade(args: argparse.Namespace) -> int:
    report = extract_finance_trace_bank(
        report_paths=list(args.report or []),
        min_utility_delta=args.min_delta,
    )
    markdown = render_finance_trace_bank_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("finance_pretrade_extract")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="finance_trace_bank_summary",
    )
    jsonl_path = out_dir / "finance_trace_bank.jsonl"
    jsonl_lines = [json.dumps(row, ensure_ascii=True) for row in report.get("rows", [])]
    jsonl_path.write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
    if args.json:
        _print_json(
            {
                "json_summary": str(json_path),
                "markdown_summary": str(md_path),
                "jsonl_bank": str(jsonl_path),
                "rows_extracted": report.get("rows_extracted", 0),
                "winner_counts": report.get("winner_counts", {}),
                "teacher_signal_class_counts": report.get("teacher_signal_class_counts", {}),
            }
        )
    else:
        print(f"Wrote finance trace bank JSON summary: {json_path}")
        print(f"Wrote finance trace bank Markdown summary: {md_path}")
        print(f"Wrote finance trace bank JSONL: {jsonl_path}")
        print(
            "Summary: "
            f"rows {report.get('rows_extracted', 0)} | "
            f"winner counts {report.get('winner_counts', {})} | "
            f"teacher signals {report.get('teacher_signal_class_counts', {})}"
        )
    return 0


def _handle_distill_finance_pretrade(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    out_path = Path(args.out).resolve() if args.out else (repo_root / ".memla" / "finance_policy_bank.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = distill_finance_policy_bank(
        trace_bank_path=args.trace_bank,
        min_priority=args.min_priority,
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_finance_policy_bank_markdown(report), encoding="utf-8")
    if args.json:
        _print_json(
            {
                "policy_bank": str(out_path),
                "markdown_summary": str(md_path),
                "rows_used": report.get("rows_used", 0),
                "source_models": report.get("source_models", {}),
            }
        )
    else:
        print(f"Wrote finance policy bank JSON: {out_path}")
        print(f"Wrote finance policy bank Markdown: {md_path}")
        print(
            "Summary: "
            f"rows used {report.get('rows_used', 0)} | "
            f"source models {report.get('source_models', {})}"
        )
    return 0


def _handle_healthcare_denial_benchmark(args: argparse.Namespace) -> int:
    report = run_healthcare_denial_benchmark(
        cases_path=args.cases,
        case_ids=list(args.case_id or []),
        limit=args.limit,
        raw_model=args.raw_model,
        memla_model=args.memla_model,
        raw_iterations=args.raw_iterations,
        memla_iterations=args.memla_iterations,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        raw_provider=args.raw_provider,
        raw_base_url=args.raw_base_url,
        memla_provider=args.memla_provider,
        memla_base_url=args.memla_base_url,
    )
    markdown = render_healthcare_denial_markdown(report)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("healthcare_denial_benchmark")
    json_path, md_path = _write_report_bundle(
        report=report,
        markdown=markdown,
        out_dir=out_dir,
        stem="healthcare_denial_benchmark_report",
    )
    print(f"Wrote healthcare benchmark JSON: {json_path}")
    print(f"Wrote healthcare benchmark Markdown: {md_path}")
    utility_index = report.get("memla_vs_raw_healthcare_utility_index")
    utility_text = utility_index if utility_index is not None else "n/a"
    print(
        "Summary: "
        f"raw utility {report.get('avg_raw_healthcare_utility', 0.0)} | "
        f"memla utility {report.get('avg_memla_healthcare_utility', 0.0)} | "
        f"utility index {utility_text}"
    )
    return 0


def _handle_thesis_pack(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_report_dir("thesis_pack")
    result = build_thesis_pack(
        coding_path=args.coding,
        math_rerank_path=args.math_rerank,
        math_progress_path=args.math_progress,
        out_dir=str(out_dir),
        site_url=args.site_url,
        coding_secondary_path=args.coding_secondary,
        compile_support_path=args.compile_support,
    )
    _print_json(result)
    return 0


def _handle_publish_site(args: argparse.Namespace) -> int:
    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    result = _sync_public_site(source_dir=source_dir, out_dir=out_dir)
    if args.json:
        _print_json(result)
    else:
        print(f"Published site from {result['source_dir']} to {result['out_dir']}")
        print(f"Copied: {', '.join(result['copied'])}")
        print(f"Site ready: {'yes' if result['site_ready'] else 'no'}")
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    db_path = _resolve_db_path(args.db, repo_root)
    db_parent = db_path.parent
    report: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "ok": sys.version_info >= (3, 11),
        },
        "repo_root": {
            "path": str(repo_root),
            "exists": repo_root.exists(),
            "git_repo": (repo_root / ".git").exists(),
        },
        "db_path": {
            "path": str(db_path),
            "parent_exists": db_parent.exists(),
            "parent_writable": os.access(db_parent, os.W_OK),
        },
        "site": {
            "root_index": str((repo_root / "index.html").resolve()),
            "root_vercel_json": str((repo_root / "vercel.json").resolve()),
            "ready": (repo_root / "index.html").exists() and (repo_root / "vercel.json").exists(),
        },
        "ollama": {
            "requested_model": args.model,
            "reachable": False,
            "url": "",
            "model_count": 0,
            "model_present": False,
            "models": [],
            "error": "",
        },
    }

    last_error = ""
    for url in _candidate_ollama_urls(args.ollama_url):
        try:
            ollama = _probe_ollama(url)
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            continue
        report["ollama"].update(ollama)
        report["ollama"]["reachable"] = True
        report["ollama"]["model_present"] = args.model in set(ollama.get("models") or [])
        break
    else:
        report["ollama"]["error"] = last_error or "Could not reach Ollama."

    report["overall_ok"] = all(
        [
            report["python"]["ok"],
            report["repo_root"]["exists"],
            report["db_path"]["parent_writable"],
        ]
    )

    if args.json:
        _print_json(report)
        return 0

    def _status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"[{_status(report['python']['ok'])}] Python {report['python']['version']} (need 3.11+)")
    print(
        f"[{_status(report['repo_root']['exists'])}] Repo root {report['repo_root']['path']}"
        + (" (git repo)" if report["repo_root"]["git_repo"] else " (no .git found)")
    )
    print(
        f"[{_status(report['db_path']['parent_writable'])}] DB parent {db_parent}"
        + ("" if report["db_path"]["parent_writable"] else " is not writable")
    )
    print(
        f"[{_status(report['site']['ready'])}] Root site "
        + ("is Vercel-ready" if report["site"]["ready"] else "is not published at repo root yet")
    )
    if report["ollama"]["reachable"]:
        model_note = "present" if report["ollama"]["model_present"] else "missing"
        print(
            f"[PASS] Ollama reachable at {report['ollama']['url']} "
            f"({report['ollama']['model_count']} models, requested model {model_note})"
        )
    else:
        print(f"[FAIL] Ollama unreachable: {report['ollama']['error']}")
    return 0 if report["overall_ok"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memla",
        description="Memla CLI for bounded coding and math runtimes.",
    )
    subparsers = parser.add_subparsers(dest="command")

    coding_parser = subparsers.add_parser("coding", help="Run coding plans, repairs, and coding benchmarks.")
    coding_sub = coding_parser.add_subparsers(dest="coding_command")

    run_parser = coding_sub.add_parser("run", help="Run a Memla-assisted coding turn against a repository.")
    run_parser.add_argument("--prompt", required=True, help="Task prompt for the coding assistant.")
    run_parser.add_argument("--repo-root", default=".", help="Repository root to operate in. Defaults to the current directory.")
    run_parser.add_argument("--db", default="", help="SQLite path for Memla memory. Defaults to <repo>/.memla/memory.sqlite.")
    run_parser.add_argument("--user-id", default=_user_id_default(), help="User or tenant identifier for memory retrieval.")
    run_parser.add_argument("--model", default=_coding_model_default(), help="Technician model to run.")
    run_parser.add_argument("--test-command", default="", help="Optional command to run after the answer is produced.")
    run_parser.add_argument("--temperature", type=float, default=0.1)
    run_parser.add_argument("--top-k", type=int, default=12)
    run_parser.add_argument("--num-ctx", type=int, default=None)
    run_parser.add_argument("--disable-compile-loop", action="store_true", help="Turn off compile-loop priors for this run.")
    run_parser.add_argument("--c2a-policy-path", default="", help="Optional explicit C2A policy bank JSON path.")
    run_parser.add_argument("--disable-c2a-policy", action="store_true", help="Disable self-transmutation priors for this run.")
    run_parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of readable text.")
    run_parser.set_defaults(func=_handle_coding_run)

    plan_parser = coding_sub.add_parser("plan", help="Build a Memla workflow plan without asking the model for a final answer.")
    plan_parser.add_argument("--prompt", required=True, help="Task prompt to plan.")
    plan_parser.add_argument("--repo-root", default=".", help="Repository root to inspect. Defaults to the current directory.")
    plan_parser.add_argument("--db", default="", help="SQLite path for Memla memory. Defaults to <repo>/.memla/memory.sqlite.")
    plan_parser.add_argument("--user-id", default=_user_id_default(), help="User or tenant identifier for memory retrieval.")
    plan_parser.add_argument("--model", default=_coding_model_default(), help="Model used for retrieval-backed planning.")
    plan_parser.add_argument("--temperature", type=float, default=0.1)
    plan_parser.add_argument("--top-k", type=int, default=12)
    plan_parser.add_argument("--num-ctx", type=int, default=None)
    plan_parser.add_argument("--disable-compile-loop", action="store_true", help="Turn off compile-loop priors while planning.")
    plan_parser.add_argument("--c2a-policy-path", default="", help="Optional explicit C2A policy bank JSON path.")
    plan_parser.add_argument("--disable-c2a-policy", action="store_true", help="Disable self-transmutation priors while planning.")
    plan_parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of readable text.")
    plan_parser.set_defaults(func=_handle_coding_plan)

    patch_parser = coding_sub.add_parser("benchmark-patch", help="Run raw-vs-Memla patch execution on git-history patch cases.")
    patch_parser.add_argument("--pack", required=True, help="Patch case pack JSON path.")
    patch_parser.add_argument("--raw-model", required=True, help="Baseline raw model.")
    patch_parser.add_argument("--memla-model", required=True, help="Memla-assisted model.")
    patch_parser.add_argument("--db", default="", help="SQLite path for Memla memory. Defaults to ./.memla/memory.sqlite.")
    patch_parser.add_argument("--user-id", default=_user_id_default(), help="User or tenant identifier for memory retrieval.")
    patch_parser.add_argument("--split", default="unseen", help="Case split to evaluate.")
    patch_parser.add_argument("--limit", type=int, default=0, help="Optional case limit.")
    patch_parser.add_argument("--top-k", type=int, default=12)
    patch_parser.add_argument("--temperature", type=float, default=0.1)
    patch_parser.add_argument("--num-ctx", type=int, default=None)
    patch_parser.add_argument("--raw-iterations", type=int, default=1)
    patch_parser.add_argument("--memla-iterations", type=int, default=3)
    patch_parser.add_argument("--raw-provider", default="", help="Optional provider override for the raw lane.")
    patch_parser.add_argument("--raw-base-url", default="", help="Optional base URL override for the raw lane.")
    patch_parser.add_argument("--memla-provider", default="", help="Optional provider override for the Memla lane.")
    patch_parser.add_argument("--memla-base-url", default="", help="Optional base URL override for the Memla lane.")
    patch_parser.add_argument("--memla-c2a-policy-path", default="", help="Optional explicit C2A policy bank JSON path for the Memla lane.")
    patch_parser.add_argument("--disable-memla-c2a-policy", action="store_true", help="Disable self-transmutation priors for the Memla lane.")
    patch_parser.add_argument("--out-dir", default="", help="Directory for report artifacts. Defaults to ./memla_reports/<timestamp>.")
    patch_parser.set_defaults(func=_handle_patch_benchmark)

    compile_parser = coding_sub.add_parser("benchmark-compile", help="Run the compile-loop coding benchmark.")
    compile_parser.add_argument("--cases", required=True, help="Plan eval case JSONL path.")
    compile_parser.add_argument("--repo-root", required=True, help="Repository root under test.")
    compile_parser.add_argument("--model", required=True, help="Model to benchmark.")
    compile_parser.add_argument("--db", default="", help="SQLite path for Memla memory. Defaults to <repo>/.memla/memory.sqlite.")
    compile_parser.add_argument("--user-id", default=_user_id_default(), help="User or tenant identifier for memory retrieval.")
    compile_parser.add_argument("--temperature", type=float, default=0.1)
    compile_parser.add_argument("--top-k", type=int, default=12)
    compile_parser.add_argument("--num-ctx", type=int, default=None)
    compile_parser.add_argument("--memla-c2a-policy-path", default="", help="Optional explicit C2A policy bank JSON path for the planning and compile lanes.")
    compile_parser.add_argument("--disable-memla-c2a-policy", action="store_true", help="Disable self-transmutation priors for the planning and compile lanes.")
    compile_parser.add_argument("--out-dir", default="", help="Directory for report artifacts. Defaults to ./memla_reports/<timestamp>.")
    compile_parser.set_defaults(func=_handle_compile_benchmark)

    c2a_parser = coding_sub.add_parser("benchmark-c2a", help="Run a pure next-move coding C2A benchmark.")
    c2a_parser.add_argument("--cases", required=True, help="Coding C2A case JSONL path.")
    c2a_parser.add_argument("--repo-root", required=True, help="Repository root under test.")
    c2a_parser.add_argument("--raw-model", required=True, help="Baseline raw model.")
    c2a_parser.add_argument("--memla-model", required=True, help="Memla-assisted planning model.")
    c2a_parser.add_argument("--db", default="", help="SQLite path for Memla memory. Defaults to <repo>/.memla/memory.sqlite.")
    c2a_parser.add_argument("--user-id", default=_user_id_default(), help="User or tenant identifier for memory retrieval.")
    c2a_parser.add_argument("--temperature", type=float, default=0.1)
    c2a_parser.add_argument("--top-k", type=int, default=12)
    c2a_parser.add_argument("--num-ctx", type=int, default=None)
    c2a_parser.add_argument("--raw-provider", default="", help="Optional provider override for the raw lane.")
    c2a_parser.add_argument("--raw-base-url", default="", help="Optional base URL override for the raw lane.")
    c2a_parser.add_argument("--memla-provider", default="", help="Optional provider override for the Memla lane.")
    c2a_parser.add_argument("--memla-base-url", default="", help="Optional base URL override for the Memla lane.")
    c2a_parser.add_argument("--memla-c2a-policy-path", default="", help="Optional explicit C2A policy bank JSON path for the Memla lane.")
    c2a_parser.add_argument("--disable-memla-c2a-policy", action="store_true", help="Disable self-transmutation priors for the Memla lane.")
    c2a_parser.add_argument("--out-dir", default="", help="Directory for report artifacts. Defaults to ./memla_reports/<timestamp>.")
    c2a_parser.set_defaults(func=_handle_c2a_benchmark)

    extract_parser = coding_sub.add_parser("extract-c2a", help="Extract normalized teacher-vs-Memla rows from coding C2A benchmark reports.")
    extract_parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Path to a coding_c2a_benchmark_report.json file. Repeat for multiple reports.",
    )
    extract_parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Optional minimum Memla-minus-raw utility delta required to keep a row.",
    )
    extract_parser.add_argument("--out-dir", default="", help="Directory for extracted trace-bank artifacts. Defaults to ./memla_reports/<timestamp>.")
    extract_parser.add_argument("--json", action="store_true", help="Print the extraction summary as JSON.")
    extract_parser.set_defaults(func=_handle_extract_c2a)

    distill_parser = coding_sub.add_parser("distill-c2a", help="Distill a self-transmutation policy bank from an extracted C2A trace bank.")
    distill_parser.add_argument("--trace-bank", required=True, help="Path to a c2a_trace_bank summary JSON or JSONL file.")
    distill_parser.add_argument("--repo-root", default=".", help="Repository root where the policy bank should live. Defaults to the current directory.")
    distill_parser.add_argument("--out", default="", help="Optional explicit output JSON path. Defaults to <repo>/.memla/c2a_policy_bank.json.")
    distill_parser.add_argument(
        "--min-priority",
        default="medium",
        choices=["low", "medium", "high"],
        help="Minimum teaching priority a row must have to influence the distilled bank.",
    )
    distill_parser.add_argument("--json", action="store_true", help="Print the distillation summary as JSON.")
    distill_parser.set_defaults(func=_handle_distill_c2a)

    math_parser = subparsers.add_parser("math", help="Run bounded math teacher-student benchmarks.")
    math_sub = math_parser.add_subparsers(dest="math_command")
    math_bench = math_sub.add_parser("benchmark", help="Run a math C2A benchmark.")
    math_bench.add_argument("--cases", required=True, help="Math case JSONL path.")
    math_bench.add_argument("--teacher-model", required=True, help="Teacher model for trace capture or labeling.")
    math_bench.add_argument(
        "--student-models",
        nargs="+",
        required=True,
        help="One or more student models to benchmark.",
    )
    math_bench.add_argument("--temperature", type=float, default=0.1)
    math_bench.add_argument("--num-ctx", type=int, default=None)
    math_bench.add_argument("--max-iterations", type=int, default=3)
    math_bench.add_argument("--top-k", type=int, default=3)
    math_bench.add_argument(
        "--executor-mode",
        default="oneshot",
        choices=["oneshot", "stepwise", "stepwise_select", "stepwise_rerank"],
        help="Math executor mode to evaluate.",
    )
    math_bench.add_argument(
        "--teacher-trace-source",
        default="llm",
        choices=["llm", "sympy", "hybrid"],
        help="Source for teacher traces.",
    )
    math_bench.add_argument("--out-dir", default="", help="Directory for report artifacts. Defaults to ./memla_reports/<timestamp>.")
    math_bench.set_defaults(func=_handle_math_benchmark)

    finance_parser = subparsers.add_parser("finance", help="Run finance compliance backtests and benchmarks.")
    finance_sub = finance_parser.add_subparsers(dest="finance_command")
    finance_bench = finance_sub.add_parser("benchmark-pretrade", help="Run a pre-trade compliance replay benchmark.")
    finance_bench.add_argument("--cases", required=True, help="Finance pre-trade case JSONL path.")
    finance_bench.add_argument("--repo-root", default=".", help="Repository root used for local finance policy banks.")
    finance_bench.add_argument("--case-id", action="append", default=[], help="Optional case id filter. Repeat to run only specific finance cases.")
    finance_bench.add_argument("--limit", type=int, default=None, help="Optional max number of finance cases to run after filtering.")
    finance_bench.add_argument("--raw-model", required=True, help="Baseline raw model.")
    finance_bench.add_argument("--memla-model", required=True, help="Memla repair-loop model.")
    finance_bench.add_argument("--raw-iterations", type=int, default=1, help="How many attempts the raw lane gets.")
    finance_bench.add_argument("--memla-iterations", type=int, default=3, help="How many verifier-backed repair attempts the Memla lane gets.")
    finance_bench.add_argument("--temperature", type=float, default=0.1)
    finance_bench.add_argument("--num-ctx", type=int, default=None)
    finance_bench.add_argument("--raw-provider", default="", help="Optional provider override for the raw lane.")
    finance_bench.add_argument("--raw-base-url", default="", help="Optional base URL override for the raw lane.")
    finance_bench.add_argument("--memla-provider", default="", help="Optional provider override for the Memla lane.")
    finance_bench.add_argument("--memla-base-url", default="", help="Optional base URL override for the Memla lane.")
    finance_bench.add_argument("--memla-finance-policy-path", default="", help="Optional explicit finance policy bank JSON path for the Memla lane.")
    finance_bench.add_argument("--disable-memla-finance-policy", action="store_true", help="Disable finance self-transmutation priors for the Memla lane.")
    finance_bench.add_argument("--out-dir", default="", help="Directory for report artifacts. Defaults to ./memla_reports/<timestamp>.")
    finance_bench.set_defaults(func=_handle_finance_pretrade_benchmark)

    finance_extract = finance_sub.add_parser("extract-pretrade", help="Extract normalized teacher-vs-Memla rows from finance pre-trade benchmark reports.")
    finance_extract.add_argument(
        "--report",
        action="append",
        required=True,
        help="Path to a finance_pretrade_benchmark_report.json file. Repeat for multiple reports.",
    )
    finance_extract.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Optional minimum Memla-minus-raw utility delta required to keep a row.",
    )
    finance_extract.add_argument("--out-dir", default="", help="Directory for extracted finance trace-bank artifacts. Defaults to ./memla_reports/<timestamp>.")
    finance_extract.add_argument("--json", action="store_true", help="Print the extraction summary as JSON.")
    finance_extract.set_defaults(func=_handle_extract_finance_pretrade)

    finance_distill = finance_sub.add_parser("distill-pretrade", help="Distill a finance self-transmutation policy bank from extracted finance traces.")
    finance_distill.add_argument("--trace-bank", required=True, help="Path to a finance trace-bank summary JSON or JSONL file.")
    finance_distill.add_argument("--repo-root", default=".", help="Repository root where the finance policy bank should live. Defaults to the current directory.")
    finance_distill.add_argument("--out", default="", help="Optional explicit output JSON path. Defaults to <repo>/.memla/finance_policy_bank.json.")
    finance_distill.add_argument(
        "--min-priority",
        default="medium",
        choices=["low", "medium", "high"],
        help="Minimum teaching priority a row must have to influence the distilled finance bank.",
    )
    finance_distill.add_argument("--json", action="store_true", help="Print the distillation summary as JSON.")
    finance_distill.set_defaults(func=_handle_distill_finance_pretrade)

    healthcare_parser = subparsers.add_parser("healthcare", help="Run healthcare denial replay benchmarks.")
    healthcare_sub = healthcare_parser.add_subparsers(dest="healthcare_command")
    healthcare_bench = healthcare_sub.add_parser("benchmark-denials", help="Run a denied-claim replay benchmark.")
    healthcare_bench.add_argument("--cases", required=True, help="Healthcare denied-claim case JSONL path.")
    healthcare_bench.add_argument("--case-id", action="append", default=[], help="Optional case id filter. Repeat to run only specific healthcare cases.")
    healthcare_bench.add_argument("--limit", type=int, default=None, help="Optional max number of healthcare cases to run after filtering.")
    healthcare_bench.add_argument("--raw-model", required=True, help="Baseline raw model.")
    healthcare_bench.add_argument("--memla-model", required=True, help="Memla repair-loop model.")
    healthcare_bench.add_argument("--raw-iterations", type=int, default=1, help="How many attempts the raw lane gets.")
    healthcare_bench.add_argument("--memla-iterations", type=int, default=3, help="How many verifier-backed repair attempts the Memla lane gets.")
    healthcare_bench.add_argument("--temperature", type=float, default=0.1)
    healthcare_bench.add_argument("--num-ctx", type=int, default=None)
    healthcare_bench.add_argument("--raw-provider", default="", help="Optional provider override for the raw lane.")
    healthcare_bench.add_argument("--raw-base-url", default="", help="Optional base URL override for the raw lane.")
    healthcare_bench.add_argument("--memla-provider", default="", help="Optional provider override for the Memla lane.")
    healthcare_bench.add_argument("--memla-base-url", default="", help="Optional base URL override for the Memla lane.")
    healthcare_bench.add_argument("--out-dir", default="", help="Directory for report artifacts. Defaults to ./memla_reports/<timestamp>.")
    healthcare_bench.set_defaults(func=_handle_healthcare_denial_benchmark)

    pack_parser = subparsers.add_parser("pack", help="Build Memla proof and buyer packs.")
    pack_sub = pack_parser.add_subparsers(dest="pack_command")
    thesis_parser = pack_sub.add_parser("thesis", help="Build the current thesis proof pack.")
    thesis_parser.add_argument("--coding", required=True, help="Primary coding report JSON.")
    thesis_parser.add_argument("--math-rerank", required=True, help="Math reranker report JSON.")
    thesis_parser.add_argument("--math-progress", required=True, help="Math end-to-end report JSON.")
    thesis_parser.add_argument("--coding-secondary", default="", help="Optional second coding repo-family report JSON.")
    thesis_parser.add_argument("--compile-support", default="", help="Optional compile-loop support report JSON.")
    thesis_parser.add_argument("--out-dir", default="", help="Output directory. Defaults to ./memla_reports/<timestamp>.")
    thesis_parser.add_argument("--site-url", default="https://memla.vercel.app", help="Site URL embedded in the pack.")
    thesis_parser.set_defaults(func=_handle_thesis_pack)

    publish_site_parser = pack_sub.add_parser("publish-site", help="Publish a proof pack as a Vercel-ready static site directory.")
    publish_site_parser.add_argument("--source", default="proof/current_pack", help="Source pack directory to publish.")
    publish_site_parser.add_argument("--out-dir", default=".", help="Output directory for the static site.")
    publish_site_parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of readable text.")
    publish_site_parser.set_defaults(func=_handle_publish_site)

    doctor_parser = subparsers.add_parser("doctor", help="Check Python, repo, Ollama, and site readiness.")
    doctor_parser.add_argument("--repo-root", default=".", help="Repository root to inspect.")
    doctor_parser.add_argument("--db", default="", help="SQLite path for Memla memory. Defaults to <repo>/.memla/memory.sqlite.")
    doctor_parser.add_argument("--model", default=_coding_model_default(), help="Model to look for in Ollama.")
    doctor_parser.add_argument("--ollama-url", default="", help="Optional Ollama base URL override.")
    doctor_parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of readable text.")
    doctor_parser.set_defaults(func=_handle_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    return int(func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
