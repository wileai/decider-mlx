"""Packaging and checkpoint contract tests; never load model weights."""
import inspect
import pytest


def v1_config():
    return {
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


def v1_decider_config():
    return {'version': '4b-v1', 'base': 'Qwen/Qwen3.5-4B-Base',
            'isolated_levels': True, 'neutralize_none': False,
            'max_options': 255, 'schema_first': False, 'temperature': 1.05}


def test_explicit_constructor_has_no_quantization_or_env_options():
    from decider_mlx import Decider, Native
    sig = inspect.signature(Decider)
    assert sig.parameters['checkpoint'].default is inspect.Parameter.empty
    assert sig.parameters['upstream_source'].default is inspect.Parameter.empty
    assert 'bits' not in sig.parameters
    assert Native is Decider
    assert callable(Decider.decide)


def test_supported_v1_contract_and_wrong_heads():
    from decider_mlx.native import validate_config
    validate_config(v1_config(), v1_decider_config())
    for key, bad in [('tie_word_embeddings', False), ('hidden_size', 42),
                     ('vocab_size', 100), ('num_experts', 4),
                     ('attention_bias', True), ('quantization', {'bits': 4})]:
        config = v1_config()
        config[key] = bad
        with pytest.raises(ValueError):
            validate_config(config, v1_decider_config())
    for key, bad in [('version', '4b-v2.1'), ('neutralize_none', True),
                     ('isolated_levels', False), ('layout', 'chat'),
                     ('temperature', 0), ('max_options', 10)]:
        config = v1_decider_config()
        config[key] = bad
        with pytest.raises(ValueError):
            validate_config(v1_config(), config)
