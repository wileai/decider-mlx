# decider-mlx

Unofficial Apple Silicon runtime adaptation for [Mapika Decider](https://github.com/Mapika/decider), using MLX/Metal.

**Experimental runnable package.** Verified with a clean Python 3.11 environment on Apple Silicon, 29 unit/integration tests (including tiny Metal tensors), and real-checkpoint Boolean, Choice and Score smoke calls. This is not a production service or a full packaged-release benchmark rerun.

## Install and run

Requires Apple Silicon macOS, Python 3.11 or 3.12, and enough available unified memory for the approximately 8 GiB model plus runtime and the OS. The tested host had 16 GiB; fit is not guaranteed under competing load. The default state limit is 2,048 tokens; longer inputs are rejected rather than silently truncated. The historical research screen explicitly used 32,768, which is not a memory-safety guarantee.

```bash
# With uv installed:
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install '.[test]'

# Fetch only the two SHA-256-verified upstream prompt/readout modules.
decider-mlx fetch-source ./decider-source

# Original public weights; download is explicit and may be several GiB.
hf download Mapika/decider-4b \
  --revision c23ab4d2483e7c6a93484463e9afe1688b29bedd \
  --include '*.json' '*.safetensors' --local-dir ./checkpoint

# One synthetic example, not an application action.
decider-mlx decide --checkpoint ./checkpoint \
  --upstream-source ./decider-source --example

# Tests use tiny tensors, not a downloaded checkpoint.
python -m pytest --upstream-source ./decider-source
```

Install from a clone of this repository; no PyPI release is implied. With pip, use a separate Python 3.11/3.12 virtual environment and `python -m pip install '.[test]'`. `python -m decider_mlx` is equivalent to the console command.

### Python API

```python
from decider_mlx import Decider

model = Decider('./checkpoint', './decider-source')
answer = model.decide(
    'The demo parcel has been delivered.',
    {'arrived': {'type': 'noul', 'instructions': 'Has the parcel arrived?'}},
)
print(answer)
```

`decider-mlx example` prints a synthetic request without loading a model. Pass a JSON object with exactly `state` and `questions` using `decider-mlx decide --checkpoint ./checkpoint --upstream-source ./decider-source --request .private/request.json` or `--stdin`. Choice criteria are ordered ID→description mappings; Score criteria are ordered descriptions. Typed output probabilities are model judgments, not permission to execute an action.

Store private requests and reports under the Git-ignored `.private/` directory. Successful CLI output may include supplied question IDs, option names and score descriptions; treat it as sensitive. CLI errors omit exception details and paths. Python API users must sanitize their own logs.

The runtime is intentionally restricted to compatible unquantized v1 configurations. Compatibility checks do not establish weight authenticity; use the pinned download revision. Upstream prompt/readout files are byte-verified against `b44b4c9880a67291206499b86aac89004850134a` rather than an unverified moving checkout.

### Process and memory scope

Use a dedicated sequential model process. MLX allocator settings are process-global, and the packed recurrent path temporarily replaces an MLX-LM module-level dispatch function under a lock. Do not run unrelated Qwen inference concurrently in that process. Cache retention is zero for the MLX allocator, and the upstream option-text/token cache is cleared after prompt construction, including on failure. This is not secure memory erasure. Do not share a long-lived process between mutually untrusted tenants; use separate processes. A 9 GiB MLX allocation setting is advisory, not a hard physical-memory cap. Apply an external timeout/memory watchdog in an application. The CLI itself does not provide that watchdog. Supply only trusted checkpoint files: the streaming loader is not a sandbox for malicious or malformed model files.

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
