"""FP16-only Decider 4B v1 inference; no checkpoint or source downloaded implicitly."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from transformers import AutoTokenizer

from .protocol import request_temperature
from .stream_weights import iter_weights
from .upstream import load_upstream

_MODEL_FIELDS = {
    'model_type': 'qwen3_5_text', 'tie_word_embeddings': True,
    'hidden_size': 2560, 'intermediate_size': 9216,
    'num_hidden_layers': 32, 'num_attention_heads': 16,
    'num_key_value_heads': 4, 'head_dim': 256, 'vocab_size': 248320,
    'linear_num_key_heads': 16, 'linear_num_value_heads': 32,
    'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
    'linear_conv_kernel_dim': 4, 'full_attention_interval': 4,
    'rms_norm_eps': 1e-6, 'partial_rotary_factor': 0.25,
    'rope_parameters': {'mrope_interleaved': True, 'mrope_section': [11, 11, 10],
                        'partial_rotary_factor': 0.25, 'rope_theta': 10000000,
                        'rope_type': 'default'},
}
_DECIDER_FIELDS = {
    'version': '4b-v1', 'base': 'Qwen/Qwen3.5-4B-Base',
    'isolated_levels': True, 'neutralize_none': False,
    'max_options': 255, 'schema_first': False, 'temperature': 1.05,
}
_KERNEL_LOCK = threading.RLock()


def validate_config(config, decider_config):
    """Fail closed on unsupported architecture, head, calibration or layout.

    These checks identify compatibility, not checkpoint authenticity. Obtain
    weights from the separately pinned public model revision.
    """
    for fields, actual in ((_MODEL_FIELDS, config), (_DECIDER_FIELDS, decider_config)):
        for key, value in fields.items():
            if actual.get(key) != value:
                raise ValueError(f'Unsupported Decider 4B v1 config field: {key}')
    if config.get('num_experts', 0) != 0 or config.get('attention_bias', False):
        raise ValueError('Only the dense v1 architecture without attention bias is supported')
    if any(key in config for key in ('quantization', 'quantization_config')):
        raise ValueError('Only unquantized Decider 4B v1 weights are supported')
    if (decider_config.get('layout', 'plain') != 'plain'
            or decider_config.get('chat_template', False)
            or decider_config.get('temperature_by_type')):
        raise ValueError('Only the v1 plain layout and calibration are supported')


@contextmanager
def _packed_update():
    # mlx-lm currently calls a module-global updater. Restore it even on error;
    # serialise this package's forwards. Do not run unrelated Qwen inference
    # concurrently in the same process while this context is active.
    import mlx_lm.models.qwen3_5 as qwen
    from .gated_delta_packed import gated_delta_update
    with _KERNEL_LOCK:
        original = qwen.gated_delta_update
        qwen.gated_delta_update = gated_delta_update
        try:
            yield
        finally:
            qwen.gated_delta_update = original


def pointer_probs(hs, head, nopts, temperature):
    logits = (hs @ head.T).astype(mx.float32)
    mask = mx.arange(head.shape[0])[None, :] < mx.array(nopts)[:, None]
    return mx.softmax(mx.where(mask, logits / temperature, -float('inf')), axis=-1)


def shared_hidden(model, items, pad_id):
    """Share only this request's common prefix; read before right padding."""
    batch = len(items)
    lcp = 0
    if batch > 1:
        for i in range(min(len(x['ids']) for x in items) - 1):
            if not all(x['ids'][i] == items[0]['ids'][i] for x in items):
                break
            lcp = i + 1
    cache = None
    if lcp:
        cache = model.make_cache()
        prefix = model.language_model.model(mx.array([items[0]['ids'][:lcp]]), cache=cache)
        mx.eval(prefix, [c.state for c in cache])
        for c in cache:
            c.state = [mx.repeat(a, batch, axis=0) for a in c.state]
    lengths = [len(x['ids']) - lcp for x in items]
    width = max(lengths)
    ids = mx.array([x['ids'][lcp:] + [pad_id] * (width - length)
                    for x, length in zip(items, lengths)])
    h = model.language_model.model(ids, cache=cache)
    return h[mx.arange(batch), mx.array([x['slots'][0] - lcp for x in items])]


class NoShuffle:
    def shuffle(self, values):
        pass

    def sample(self, values, count):
        return values[:count]


class StableNorm(nn.RMSNorm):
    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)


class Decider:
    """Local FP16 runtime with original head, calibration and Score branches.

    checkpoint: local unmodified Mapika/decider-4b v1 snapshot directory.
    upstream_source: local checkout/archive root at the pinned upstream revision.
    MLX allocator cache is disabled globally. The advisory memory limit is also
    process-global. No recurrent or KV state survives a request.
    """

    def __init__(self, checkpoint, upstream_source, *, max_state_tokens=2048,
                 memory_limit=9 * 1024**3):
        checkpoint = Path(checkpoint).expanduser().resolve()
        self.upstream = load_upstream(upstream_source)
        self.cfg = json.loads((checkpoint / 'decider_config.json').read_text())
        config = json.loads((checkpoint / 'config.json').read_text())
        validate_config(config, self.cfg)
        if not isinstance(max_state_tokens, int) or not 1 <= max_state_tokens <= 32768:
            raise ValueError('max_state_tokens must be between 1 and 32768')
        if not isinstance(memory_limit, int) or memory_limit <= 0:
            raise ValueError('memory_limit must be a positive byte count')
        if not mx.metal.is_available():
            raise RuntimeError('Decider requires Apple Silicon with Metal')
        weights_path = checkpoint / 'model.safetensors'
        if not weights_path.is_file():
            raise FileNotFoundError('checkpoint must contain model.safetensors')
        mx.set_cache_limit(0)
        mx.set_memory_limit(memory_limit)
        self.max_state_tokens = max_state_tokens
        self.tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                                trust_remote_code=False)
        if self.tok.pad_token_id is None:
            raise ValueError('Checkpoint tokenizer must define a pad token')
        self.model = Model(ModelArgs.from_dict(config))
        expected = {key: tuple(value.shape) for key, value in tree_flatten(self.model.parameters())}
        seen = set()
        for key, value in iter_weights(weights_path):
            if key not in expected or tuple(value.shape) != expected[key]:
                raise ValueError(f'Unexpected checkpoint tensor or shape: {key}')
            self.model.load_weights([(key, value)], strict=False)
            seen.add(key)
        if seen != set(expected):
            raise ValueError(f'Checkpoint is missing tensors: {sorted(set(expected) - seen)}')
        del expected
        backbone = self.model.language_model.model
        norms = [backbone.norm]
        for layer in backbone.layers:
            norms.extend([layer.input_layernorm, layer.post_attention_layernorm])
            if not layer.is_linear:
                norms.extend([layer.self_attn.q_norm, layer.self_attn.k_norm])
        for norm in norms:
            norm.__class__ = StableNorm
        self.model.eval()
        labels = self.upstream.prompt.letter_ids(self.tok)
        if len(labels) != 255 or len(set(labels)) != 255 or max(labels) >= config['vocab_size']:
            raise ValueError('Checkpoint tokenizer is incompatible with the native pointer head')
        self.head = backbone.embed_tokens.weight[mx.array(labels)]
        mx.eval(self.head, self.model.parameters())
        mx.synchronize()

    def items(self, state, questions):
        for spec in questions.values():
            if spec.get('type') == 'score' and spec.get('isolated', True) is not True:
                raise ValueError('Score questions require isolated=True')
        system = self.upstream.systemone
        rqs = {key: system.render_question(value) for key, value in questions.items()}
        rows, index = system.plan_rows(rqs, True)
        context = system.render_state(state)
        if len(self.tok.encode('Context:\n' + context, add_special_tokens=False)) > self.max_state_tokens:
            raise ValueError('State exceeds max_state_tokens; refusing truncation')
        items = [self.upstream.prompt.build(
            SimpleNamespace(context=context, qs=[SimpleNamespace(
                text=row['question'], options=row['options'], gold=0)]),
            self.tok, NoShuffle(), max_options=self.upstream.prompt.MAX_OPTIONS,
            max_ctx_tokens=self.max_state_tokens, chat=None,
        ) for row in rows]
        return rqs, index, items

    def score(self, items, temperature):
        with _packed_update():
            hs = shared_hidden(self.model, items, self.tok.pad_token_id)
            probs = pointer_probs(hs, self.head, [x['nopts'][0] for x in items], temperature)
            mx.eval(probs)
            mx.synchronize()
        return probs.tolist()

    def __call__(self, state, questions):
        """Return (response, row probabilities, timing metadata)."""
        start = time.perf_counter()
        temperature = request_temperature(self.cfg, questions)
        rqs, index, items = self.items(state, questions)
        built = time.perf_counter()
        probs = self.score(items, temperature)
        forward = time.perf_counter()
        answers = self.upstream.systemone.assemble(rqs, index, probs)
        return ({'model': 'decider-4b-v1-native-mlx-fp16', 'answers': answers}, probs,
                {'build_ms': (built - start) * 1000,
                 'forward_ms': (forward - built) * 1000,
                 'assembly_ms': (time.perf_counter() - forward) * 1000,
                 'lengths': [len(x['ids']) for x in items]})

    def decide(self, state, questions):
        """Return an answer response for one independent question (possibly many Score levels)."""
        return self(state, questions)[0]


Native = Decider
