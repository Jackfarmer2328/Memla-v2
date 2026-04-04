# Memla Strategic Memo

## Strong current claim
- Coding executor proof: qwen2.5:32b raw apply `0` -> qwen3.5:9b + Memla apply `0.7` and semantic success `0` -> `0.6667`.
- Math decision proof: 4b ambiguous-step choice `0.5455` -> `1` and 9b `0.4545` -> `1` with the same candidate hands.
- End-to-end bounded math proof: 4b solve accuracy `0.875` -> `1`, matching teacher raw at `1`.

## Extra coding support
- Second repo-family patch execution: raw apply `0` -> Memla apply `0.3`.

## Second benchmark type
- Compile-loop command recall: raw `0` -> full compile-loop `0.3125`.
- Compile validated command recall: `0.25`.

## Honest limit
- This is bounded-runtime evidence, not universal proof complete.
- Coding still benefits from better retrieval, cleaner execution, and more repo-family repeats.

## Why this matters
- Memla is looking more like a real runtime layer, not just retrieval.
- It makes capable local models more useful under bounded executors.
- It is a natural fit for a CLI or harness with plug-in domain executors.