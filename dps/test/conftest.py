import pytest


def pytest_addoption(parser):
    parser.addoption("--debug", action="store_true", help="Run small tests in debug mode.")


@pytest.fixture
def debug(request):
    return request.config.getoption("--debug")
