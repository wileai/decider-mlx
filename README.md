# decider-mlx

Unofficial Apple Silicon runtime adaptation for [Mapika Decider](https://github.com/Mapika/decider), using MLX/Metal.

**Initial repository preparation:** attribution, privacy policy and sanitized historical measurements are available. The installable runtime is still being packaged and verified; this commit is not a runnable release.

## Model, not a new training run

The selected checkpoint is [Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b), version **v1**, pinned to `c23ab4d2483e7c6a93484463e9afe1688b29bedd`. The model was not trained or blended by this project. Runtime adaptations include streaming FP16 conversion, native typed-answer readout, within-request prefix sharing and a specialized recurrent kernel.

The trained decision head, fitted temperature and independent ordinal branches are preserved. Quantization and 512-token chunked prefill were rejected in evaluation and are not the selected configuration. Numerical identity to the original backend is **not** claimed.

## Historical local measurements

Apple M4, 16 GiB unified memory; full independent-question timing, not batch averages:

| Scope | Correct / requested | Median | p95 |
|---|---:|---:|---:|
| Clean fixed-policy screening | 823 / 951 | 697.8 ms | 3,399.9 ms |
| Additional synthetic holdout | 333 / 360 | 723.5 ms | 5,121.3 ms |

These are research measurements from the original harness, not an official benchmark score or a rerun of the packaged release. The historical raw fixtures are intentionally not included, so the totals cannot be independently reproduced from this repository alone. See [methodology and limitations](docs/BENCHMARKS.md) and [aggregate metrics](docs/benchmark-summary.json).

A full stock-versus-ported **4B v1** comparison remains unperformed. Earlier 2B runtime speedups must not be presented as measured 4B speedups.

## Safety and privacy

This is a local experimental runtime, not an authorization system. Keep permission to send, purchase, delete or book in deterministic application controls. Long requests can substantially exceed median latency.

No model weights, personal prompts, conversation exports, local experiment archives or credentials are bundled. See [SECURITY.md](SECURITY.md).

## Attribution and licensing

Mapika supplies the model and upstream decision semantics. MLX/MLX-LM supplies the Apple Silicon framework and recurrent implementation basis. This project is not affiliated with or endorsed by either organization.

See [NOTICE](NOTICE), [Apache-2.0 license](LICENSE), and the retained [MLX-LM MIT license](licenses/MLX-LM-MIT.txt). Downloaded model weights retain their upstream licenses.
