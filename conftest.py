import pytest


def pytest_addoption(parser):
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--show-plots", action='store_true', help="Display any graphs that are created.")
    parser.addoption("--save-plots", action='store_true', help="Save any graphs that are created.")
    parser.addoption("--run-slow", action="store_true", help="Run slow tests.")


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")


@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show-plots")


@pytest.fixture
def save_plots(request):
    return request.config.getoption("--save-plots")


@pytest.fixture
def verbose(request):
    return request.config.getoption("verbose")
