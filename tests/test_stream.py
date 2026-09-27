"""Tiny synthetic safetensors; no checkpoint content is included."""
import json
import struct

import mlx.core as mx
import pytest
from decider_mlx.stream_weights import iter_weights


def tensor_file(path, header, payload):
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data + payload)
    return path


def test_streamed_bf16_conversion_and_flat_names(tmp_path):
    path = tensor_file(tmp_path / 'toy.safetensors', {
        'model.norm.weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
        'model.embed_tokens.weight': {'dtype': 'BF16', 'shape': [1, 2], 'data_offsets': [4, 8]},
    }, struct.pack('<4H', 0x0000, 0x3f80, 0x3f80, 0x4000))
    weights = dict(iter_weights(path))
    assert weights['language_model.model.norm.weight'].tolist() == [1, 2]
    assert weights['language_model.model.embed_tokens.weight'].tolist() == [[1, 2]]
    assert weights['language_model.model.norm.weight'].dtype == mx.float32
    assert weights['language_model.model.embed_tokens.weight'].dtype == mx.float16


def test_nested_names_conv_transpose_and_fp32_preservation(tmp_path):
    path = tensor_file(tmp_path / 'toy.safetensors', {
        'model.language_model.layers.0.linear_attn.conv1d.weight': {
            'dtype': 'F16', 'shape': [2, 1, 2], 'data_offsets': [0, 8]},
        'model.language_model.layers.0.linear_attn.A_log': {
            'dtype': 'F32', 'shape': [1], 'data_offsets': [8, 12]},
    }, struct.pack('<4ef', 1, 2, 3, 4, .5))
    weights = dict(iter_weights(path))
    conv = weights['language_model.model.layers.0.linear_attn.conv1d.weight']
    assert conv.shape == (2, 2, 1)
    assert conv.tolist() == [[[1], [2]], [[3], [4]]]
    assert weights['language_model.model.layers.0.linear_attn.A_log'].dtype == mx.float32


def test_untied_head_is_rejected(tmp_path):
    path = tensor_file(tmp_path / 'toy.safetensors', {
        'lm_head.weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]},
    }, struct.pack('<e', 1))
    with pytest.raises(ValueError, match='untied head'):
        dict(iter_weights(path))


def test_truncated_tensor_is_rejected_with_clear_error(tmp_path):
    path = tensor_file(tmp_path / 'toy.safetensors', {
        'model.norm.weight': {'dtype': 'F16', 'shape': [2], 'data_offsets': [0, 4]},
    }, struct.pack('<e', 1))
    with pytest.raises(ValueError, match='Truncated tensor'):
        dict(iter_weights(path))
