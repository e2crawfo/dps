import pytest


def pytest_addoption(parser):
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--show-plots", action='store_true', help="Display any graphs that are created.")
    parser.addoption("--run-slow", action="store_true", help="run slow tests")


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")


@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show-plots")


@pytest.fixture
def verbose(request):
    return request.config.getoption("verbose")
