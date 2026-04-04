# Public Proof Summary

This public repo keeps the proof story lightweight on purpose. The raw benchmark blobs are generated artifacts, not product source, so they are omitted from version control here.

You can regenerate fresh artifacts with:
- `memla coding benchmark-patch`
- `memla coding benchmark-compile`
- `memla math benchmark`
- `memla pack thesis`

## Coding

Primary bounded patch benchmark:

| Metric | Hosted `meta/Llama-3.3-70B-Instruct` raw | Local `qwen3.5:9b + Memla` |
| --- | --- | --- |
| Apply rate | `0.0` | `1.0` |
| Semantic success | `0.0` | `1.0` |

Same-model control on the same OAuth slice:

| Metric | `qwen3.5:9b` raw | `qwen3.5:9b + Memla` |
| --- | --- | --- |
| Apply rate | `0.0` | `1.0` |
| Semantic success | `0.0` | `0.6667` |

Additional support:
- second repo family against hosted `70b` raw: apply `0.0 -> 0.3333`
- earlier larger-local baseline: `qwen2.5:32b` raw apply `0.0` -> `qwen3.5:9b + Memla` apply `0.6667`

## Math

Held-constant reranker benchmark:

| Metric | Raw | Memla |
| --- | --- | --- |
| `4b` ambiguous-step choice accuracy | `0.5455` | `1.0` |
| `9b` ambiguous-step choice accuracy | `0.4545` | `1.0` |

Harder end-to-end bounded math:

| Metric | Raw | Memla |
| --- | --- | --- |
| `4b` solve accuracy | `0.875` | `1.0` |
| `32b` raw solve accuracy | `1.0` | n/a |

## Honest claim

What this supports:
- Memla improves technical decision quality inside bounded executors.
- Smaller local models can behave like much larger raw models on verifier-backed slices.
- The same base model improves materially once the Memla runtime is added.
- The coding wedge is real enough to package as a CLI.

What this does not claim:
- universal model parity
- `9b` beats `70b` at everything
- tiny models become frontier models everywhere
