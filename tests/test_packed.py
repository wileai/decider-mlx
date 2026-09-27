"""Small synthetic packed/unpacked recurrent-kernel comparisons."""
import mlx.core as mx
import pytest

pytestmark = [pytest.mark.metal, pytest.mark.skipif(
    not mx.metal.is_available(), reason='Metal GPU required')]


@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
def test_packed_matches_explicit_tree_for_synthetic_prefill(dtype):
    from decider_mlx.gated_delta_packed import gated_delta_kernel, gated_delta_kernel_xtree
    mx.random.seed(41)
    q = (mx.random.normal((2, 7, 1, 128)) * .02).astype(dtype)
    k = (mx.random.normal(q.shape) * .02).astype(dtype)
    v = mx.random.normal((2, 7, 2, 16)).astype(dtype)
    g = mx.full((2, 7, 2), .97, dtype=mx.float32)
    beta = mx.full((2, 7, 2), .4, dtype=dtype)
    state = mx.zeros((2, 2, 16, 128), dtype=mx.float32)
    actual = gated_delta_kernel(q, k, v, g, beta, state)
    expected = gated_delta_kernel_xtree(q, k, v, g, beta, state)
    mx.eval(actual, expected)
    for a, b in zip(actual, expected):
        assert mx.array_equal(a, b).item()
