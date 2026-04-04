# Memla CLI

Memla is a bounded runtime that helps smaller models make better technical decisions inside verifier-backed loops.

Install from PyPI:

```bash
pip install memla
```

PyPI:
- [memla on PyPI](https://pypi.org/project/memla/)

This repo is the public, CLI-first version of Memla. It is intentionally narrower than the internal research repo and keeps the public snapshot focused on the tool, not every generated artifact.

## What this repo contains

- `memla.py`
  - thin top-level entry point
- `memory_system/`
  - CLI runtime, coding loop, math benchmark, and pack builder
- `cases/`
  - bundled case files for quick local runs
- `proof/`
  - lightweight public proof summary and a small site template
- `tests/`
  - focused coverage for the CLI and benchmark surfaces

## Current bounded claim

Public proof summary:
- `proof/summary.md`

Current strongest public result:
- on coding, local `qwen3.5:9b + Memla` beat hosted `Meta-Llama-3.1-405B-Instruct` raw on execution outcome in the primary patch benchmark
- on coding, the same `qwen3.5:9b` base model moved from `0.0` apply / `0.0` semantic success raw to `1.0` apply / `0.6667` semantic success with Memla on the same OAuth slice
- on a second repo family, hosted `meta/Llama-3.3-70B-Instruct` raw again stayed at `0.0` apply while local `qwen3.5:9b + Memla` reached `0.3333` apply on the FastAPI slice
- on math, `qwen3.5:4b + Memla` matched `qwen2.5:32b` raw on the harder bounded pack
- on ambiguous math decision states, Memla lifted both `4b` and `9b` to perfect choice accuracy on the tested slice

This is a bounded-runtime claim, not a universal model-parity claim.

## Quick start

Prerequisites:
- Python 3.11+
- either Ollama running locally or a hosted chat model reachable through the shared LLM client
- one or more models already available

Install:

```bash
pip install memla
```

For local editable development instead:

```bash
py -3 -m pip install -e .
```

Smoke-check the CLI:

```bash
memla --help
```

Run a local environment check:

```bash
memla doctor --repo-root . --model qwen3.5:9b
```

Use a hosted GitHub Models endpoint instead of Ollama:

```powershell
$env:LLM_PROVIDER="github_models"
$env:GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
$env:LLM_BASE_URL="https://models.github.ai/inference"
memla coding run --prompt "Repair the failing auth tests" --repo-root . --model "meta/Llama-3.3-70B-Instruct"
```

If you prefer, `LLM_API_KEY` also works in place of `GITHUB_TOKEN`.

## Main commands

Build a workflow plan inside a repo:

```bash
memla coding plan --prompt "Fix the auth regression" --repo-root .
```

Run a bounded coding turn with optional verification:

```bash
memla coding run --prompt "Repair the failing auth tests" --repo-root . --test-command "pytest -q"
```

Run the patch execution benchmark:

```bash
memla coding benchmark-patch --pack path\\to\\git_history_case_pack.json --raw-model qwen2.5:32b --memla-model qwen3.5:9b
```

Mix a hosted raw lane with a local Memla lane:

```bash
memla coding benchmark-patch --pack path\\to\\git_history_case_pack.json --raw-model meta/Llama-3.3-70B-Instruct --memla-model qwen3.5:9b --raw-provider github_models --raw-base-url https://models.github.ai/inference --memla-provider ollama --memla-base-url http://127.0.0.1:11435
```

Run the compile-loop benchmark:

```bash
memla coding benchmark-compile --cases cases\\coding_eval_cases.jsonl --repo-root . --model qwen3.5:9b
```

Run the bounded math benchmark:

```bash
memla math benchmark --cases cases\\math_linear_c2a_v2_harder.jsonl --teacher-model qwen2.5:32b --student-models qwen3.5:4b qwen3.5:9b --executor-mode stepwise_rerank --teacher-trace-source hybrid
```

Build a proof pack from generated report JSONs:

```bash
memla pack thesis --coding path\\to\\coding_patch_execution_report.json --math-rerank path\\to\\math_step_rerank_report.json --math-progress path\\to\\math_progress_report.json
```

Benchmark commands write report bundles under `./memla_reports/` by default.

## Public proof note

This repo intentionally omits bulky raw benchmark dumps from version control so the public snapshot stays product-shaped.

If you want the underlying artifacts, generate them locally with:
- `memla coding benchmark-patch`
- `memla coding benchmark-compile`
- `memla math benchmark`
- `memla pack thesis`

## Tests

Focused verification:

```bash
py -3 -m pytest -q tests\\test_step13_coding_compile_loop.py tests\\test_step14_compile_loop_benchmark.py tests\\test_step15_patch_execution_benchmark.py tests\\test_step16_math_c2a_benchmark.py tests\\test_step17_memla_cli.py
```

## Product direction

Memla is being packaged as:
- a local/private coding runtime for smaller models
- a CLI first, not a chat app first
- a verifier-backed system, not a prompt wrapper

The wedge is:

**make local 9b/14b/32b coding models more execution-capable than their raw form.**
