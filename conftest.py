import pytest


def pytest_addoption(parser):
    parser.addoption("--mode", choices=["debug", "basic", "real"], default="basic",
                     help="In debug mode, run small versions of test with lots of output. "
                          "In real mode, wait for each optimization to finish. Basic mode "
                          "is somewhere in between.")
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--display", action='store_true', help="Display any graphs that are created.")


@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")


@pytest.fixture
def display(request):
    return request.config.getoption("--display")


@pytest.fixture
def verbose(request):
    return request.config.getoption("verbose")
