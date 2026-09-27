# Measurements and limitations

These are **historical research results**, not an official benchmark submission or a rerun of the repackaged public code. The model is Mapika's public Decider 4B v1; this project changes its runtime, not its training.

## Selected configuration

Apple M4, 16 GiB unified memory. MLX 0.32.2, MLX-LM 0.31.3, FP16 weights with audited FP32 normalization conversion, zero retained allocator cache. Full request timing includes prompt construction, tokenization, native branches, inference and readout; accelerator work is synchronized. Startup and warmup are excluded.

| Evaluation | Correct / requested | Median | p95 |
|---|---:|---:|---:|
| Clean single-policy screening | 823 / 951 | 697.8 ms | 3,399.9 ms |
| Additional synthetic holdout | 333 / 360 | 723.5 ms | 5,121.3 ms |

The 951 questions combine reused cross-domain synthetic actionability cases and public-derived cases. They are not 951 independent real-world scenarios. The holdout was authored in the same evaluation workflow, not independently adjudicated by humans. Repeated candidate selection makes the main suite screening, not an untouched test set.

**Privacy boundary:** the original experiment directories, raw inputs, tokenized evidence, logs, HTML reports and machine paths are not distributed. The aggregate JSON includes a source-metrics digest for provenance, but the historical totals cannot be independently regenerated from this repository alone. Run the included synthetic example and your own appropriately licensed fixtures. Do not describe the historical suite as a fully reproducible public benchmark.

## What improved—and what is unproven

Earlier 2B experiments measured lower per-request latency after moving from the MPS runtime to the optimized MLX path: routing median 389→266 ms, permission 313→216 ms and urgency 828→288 ms. These are combined configuration comparisons, not isolated attribution to each optimization. They do **not** establish a 4B-specific speedup.

The 4B runtime completed the full screen under a clean fixed memory policy. A complete matched stock-versus-ported 4B test has **not** been performed. Neither bitwise equivalence to upstream nor a model-quality improvement caused by the port is claimed.

## Rejected variants

- Conservative 8-bit MLP and wider projection quantization each retained 66/72 development decisions but slowed requests; neither proceeded to the complete screen. Lower MLX allocation did not justify promotion.
- Full-token 512-token chunked prefill passed 72-question development but scored 822/951 on the complete screen, introducing one right→wrong decision. It was rejected. It is not enabled in this distribution.
- A separate Open-Jev 9B Q4 experiment was not a Decider variant. It returned 60 correct on 79 completed questions versus 72 for Decider on those IDs, at a 3.65-second median; the next request timed out. It is not included.

## Operational limits

- Scores are expected-label agreement, not permission guarantees or real-world safety certification.
- Median latency is not a deadline guarantee. Long contexts have much higher tail latency.
- MLX allocator limits are advisory; GPU and CPU share physical memory. RSS and MLX allocation counters overlap.
- Keep send, purchase, deletion and booking authority in deterministic application controls.
- The packed kernel and the within-request cache path are numerical/backend adaptations. Changes in shapes, precision or dependency versions require new paired testing.
