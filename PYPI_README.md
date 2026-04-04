# Memla

Memla is a local CLI runtime for Ollama models.

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

Core use cases:
- local coding workflow planning and repair
- coding patch-execution and compile-loop benchmarks
- bounded math benchmarks for decision-layer evaluation
- proof-pack generation for static report sites

Current bounded public claim:
- on coding, `qwen3.5:9b + Memla` beat local `qwen2.5:32b` raw on execution outcome in the primary patch benchmark
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
