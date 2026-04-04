# 90-Second Demo Flow

1. Open the current coding patch report.
   qwen2.5:32b apply rate: `0`. qwen3.5:9b + Memla apply rate: `0.7`.
   Raw semantic success: `0`. Memla semantic success: `0.6667`.
2. Show one coding case where raw never reached an applyable patch and Memla did.
3. Open the math reranker report with the executor held constant.
   4b ambiguous-step choice accuracy: `0.5455` -> `1`.
   9b ambiguous-step choice accuracy: `0.4545` -> `1`.
4. Open the harder end-to-end math report.
   4b raw solve accuracy: `0.875`. 4b + Memla: `1`. Teacher raw: `1`.
5. Close with the thesis: Memla improves the decisions that bounded executors turn into real work.