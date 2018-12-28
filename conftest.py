import pytest
import time
import logging


logging.getLogger('PIL').setLevel(logging.WARNING)


def pytest_addoption(parser):
    parser.addoption("--max-steps", default=None, help="Maximum number of steps to run.")
    parser.addoption("--show-plots", action='store_true', help="Display any graphs that are created.")
    parser.addoption("--skip-slow", action="store_true", help="Skip slow tests.")
    parser.addoption("--skip-fast", action="store_true", help="Skip fast tests.")
    parser.addoption(
        "--tf-log-level", type=int, default=3,
        help="Quietness of tensorflow logging; 3 (default) is most quiet, 0 is least quiet.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="Test is marked slow and --skip-slow was supplied.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    elif config.getoption("--skip-fast"):
        skip_fast = pytest.mark.skip(reason="Test is not marked slow and --skip-fast was supplied.")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_fast)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    start = time.time()
    try:
        yield
    finally:
        print("- Took {:.3f}s.".format(time.time() - start))


@pytest.fixture
def max_steps(request):
    return request.config.getoption("--max-steps")


@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show-plots")


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
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        use_gpu=False,
        show_plots=request.config.getoption("--show-plots"),
        render_step=0,
        readme="testing",
    )
