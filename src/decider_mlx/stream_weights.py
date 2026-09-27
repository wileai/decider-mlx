"""Stream BF16 to FP16 one tensor at a time; shift zero-centred norms in FP32."""
import json
import struct

import mlx.core as mx
import numpy as np

_NORM_SUFFIXES = (
    '.input_layernorm.weight', '.post_attention_layernorm.weight',
    '.q_norm.weight', '.k_norm.weight', 'model.norm.weight',
)


def iter_weights(path):
    """Yield converted HF tensors without holding a full duplicate checkpoint."""
    with open(path, 'rb') as source:
        header_size = struct.unpack('<Q', source.read(8))[0]
        header = json.loads(source.read(header_size))
        base = 8 + header_size
        for key, metadata in header.items():
            if key == '__metadata__' or 'mtp.' in key:
                continue
            if key == 'lm_head.weight':
                raise ValueError('Unexpected untied head: requires explicit audited support')
            low, high = metadata['data_offsets']
            source.seek(base + low)
            dtype = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32}[metadata['dtype']]
            payload = source.read(high - low)
            if len(payload) != high - low:
                raise ValueError(f'Truncated tensor: {key}')
            data = np.frombuffer(payload, dtype=dtype).copy()
            del payload
            value = mx.array(data).reshape(metadata['shape'])
            if metadata['dtype'] == 'BF16':
                value = value.view(mx.bfloat16)
            name = key.replace('model.language_model.', 'language_model.model.')
            if name.startswith('model.'):
                name = 'language_model.' + name
            if name.endswith('conv1d.weight'):
                value = value.transpose(0, 2, 1)
            if name.endswith(_NORM_SUFFIXES):
                value = value.astype(mx.float32) + 1
            elif value.dtype != mx.float32:
                value = value.astype(mx.float16)
            mx.eval(value)
            del data
            yield name, value
