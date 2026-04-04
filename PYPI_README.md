# Memla

Memla is a CLI runtime for local or hosted models.

It helps smaller models make better technical decisions inside verifier-backed loops instead of using raw prompting alone.

Install:

```bash
pip install memla
```

Quick check:

```bash
memla --help
memla doctor --repo-root . --model qwen3.5:9b
```

Hosted GitHub Models example:

```powershell
$env:LLM_PROVIDER="github_models"
$env:GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
$env:LLM_BASE_URL="https://models.github.ai/inference"
memla coding run --prompt "Repair the failing auth tests" --repo-root . --model "meta/Llama-3.3-70B-Instruct"
```

Core use cases:
- local coding workflow planning and repair
- coding patch-execution and compile-loop benchmarks
- bounded math benchmarks for decision-layer evaluation
- proof-pack generation for static report sites

Current bounded public claim:
- on coding, local `qwen3.5:9b + Memla` beat hosted `Meta-Llama-3.1-405B-Instruct` raw on execution outcome in the primary patch benchmark
- on coding, the same `qwen3.5:9b` base model moved from `0.0` apply / `0.0` semantic success raw to `1.0` apply / `0.6667` semantic success with Memla on the same OAuth slice
- on coding, hosted `Grok-3` raw also stayed at `0.0` apply / `0.0` semantic success on the OAuth slice while local `qwen3.5:9b + Memla` reached `0.6667` apply / `0.6667` semantic success
- on a second repo family, hosted `meta/Llama-3.3-70B-Instruct` raw again stayed at `0.0` apply while local `qwen3.5:9b + Memla` reached `0.3333` apply on the FastAPI slice
- on a second repo family against hosted `Grok-3` raw, local `qwen3.5:9b + Memla` reached `0.5` apply on `2` completed FastAPI cases while the raw lane stayed at `0.0` apply and one raw-lane case failed with `HTTPError`
- on math, `qwen3.5:4b + Memla` matched `qwen2.5:32b` raw on a harder bounded pack
- on ambiguous math decision states, Memla lifted both `4b` and `9b` to perfect choice accuracy on the tested slice

This is not a claim of universal model parity. It is a claim about bounded runtimes with verifiers.

Useful commands:

```bash
memla coding plan --prompt "Fix the auth regression" --repo-root .
memla coding run --prompt "Repair the failing auth tests" --repo-root . --test-command "pytest -q"
memla coding benchmark-compile --cases cases/coding_eval_cases.jsonl --repo-root . --model qwen3.5:9b
memla math benchmark --cases cases/math_linear_c2a_v2_harder.jsonl --teacher-model qwen2.5:32b --student-models qwen3.5:4b qwen3.5:9b --executor-mode stepwise_rerank --teacher-trace-source hybrid
```

Project links:
- GitHub: [Jackfarmer2328/Memla-v2](https://github.com/Jackfarmer2328/Memla-v2)
- Proof site: [memla.vercel.app](https://memla.vercel.app)
