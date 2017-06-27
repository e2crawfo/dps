import pytest
from dps.utils import DpsConfig


def pytest_configure(config):
    dc = DpsConfig
    dc.start_tensorboard = False
    dc.mpl_backend = 'Agg'
    dc.display = False
    dc.save_display = False
    dc.update_latest = False
    dc.use_gpu = False


def pytest_addoption(parser):
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--display", action='store_true', help="Display any graphs that are created.")
    parser.addoption("--run-slow", action="store_true", help="run slow tests")


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")


@pytest.fixture
def display(request):
    return request.config.getoption("--display")


@pytest.fixture
def verbose(request):
    return request.config.getoption("verbose")
