# 90-Second Demo Flow

1. Open the current coding patch report.
   Hosted Meta-Llama-3.1-405B-Instruct raw apply rate: `0`. qwen3.5:9b + Memla apply rate: `1`.
   Raw semantic success: `0`. Memla semantic success: `1`.
2. Open the same-model control.
   qwen3.5:9b raw apply rate: `0`. qwen3.5:9b + Memla apply rate: `1`.
   Raw semantic success: `0`. Memla semantic success: `0.6667`.
3. Show the second repo-family repeat.
   Hosted Llama-3.3-70B raw apply rate: `0`. qwen3.5:9b + Memla apply rate: `0.3333`.
   Same shape on FastAPI, even though semantic success stayed flat there.
4. Open the math reranker report with the executor held constant.
   4b ambiguous-step choice accuracy: `0.5455` -> `1`.
   9b ambiguous-step choice accuracy: `0.4545` -> `1`.
5. Open the harder end-to-end math report.
   4b raw solve accuracy: `0.875`. 4b + Memla: `1`. Teacher raw: `1`.
6. Close with the thesis: Memla improves the decisions that bounded executors turn into real work.
