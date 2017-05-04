import pytest


def pytest_addoption(parser):
    parser.addoption("--mode", choices=["debug", "basic", "real"], default="basic",
                     help="In debug mode, run small versions of test with lots of output. "
                          "In real mode, wait for each optimization to finish. Basic mode "
                          "is somewhere in between.")
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")


@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")
