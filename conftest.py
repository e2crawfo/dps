import pytest
from dps.utils import Config


def pytest_configure(config):
    assert len(Config._stack) == 1
    dc = Config._stack[0]
    dc.start_tensorboard = False
    dc.mpl_backend = 'Agg'
    dc.display = False
    dc.save_display = False
    dc.update_latest = False


def pytest_addoption(parser):
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--display", action='store_true', help="Display any graphs that are created.")


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")


@pytest.fixture
def display(request):
    return request.config.getoption("--display")


@pytest.fixture
def verbose(request):
    return request.config.getoption("verbose")
