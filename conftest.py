import pytest


def pytest_addoption(parser):
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--show-plots", action='store_true', help="Display any graphs that are created.")
    parser.addoption("--save-plots", action='store_true', help="Save any graphs that are created.")
    parser.addoption("--run-slow", action="store_true", help="Run slow tests.")
    parser.addoption(
        "--tf-log-level", type=int, default=3,
        help="Quietness of tensorflow logging; 3 (default) is most quiet, 0 is least quiet.")


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


@pytest.fixture(scope="session", autouse=True)
def tf_log_level(request):
    import tensorflow as tf
    ll = request.config.getoption("--tf-log-level")
    tf.logging.set_verbosity(ll)


@pytest.fixture
def test_config(request):
    return dict(
        start_tensorboard=False,
        save_summaries=False,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        use_gpu=False,
        show_plots=request.config.getoption("--show-plots"),
        save_plots=request.config.getoption("--save-plots"),
    )
