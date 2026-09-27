import pytest


def pytest_addoption(parser):
    parser.addoption('--upstream-source', help='Optional pinned source checkout for protocol integration tests')


@pytest.fixture
def upstream(request):
    source = request.config.getoption('--upstream-source')
    if source is None:
        pytest.skip('supply --upstream-source for pinned upstream integration tests')
    from decider_mlx.upstream import load_upstream
    return load_upstream(source)
