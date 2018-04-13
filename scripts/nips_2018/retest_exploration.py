import clify

from dps.env.advanced import yolo_rl
from dps.datasets import EMNIST_ObjectDetection


def prepare_func():
    from dps import cfg
    import os

    def get_curriculum_for_idx(idx):
        path = "/scratch/e2crawfo/dps_data/run_experiments/run_search_exploration_seed=0_2018_04_06_23_16_46/experiments"
        experiments = os.listdir(path)
        my_paths = [e for e in experiments if "_idx={}_".format(idx) in e]
        assert len(my_paths) == 1
        my_path = os.path.join(path, my_paths[0])

        weight_files = os.listdir(os.path.join(my_path, "weights"))
        weight_files = list(set([w.split(".")[0] for w in weight_files if w.startswith("best_of_stage_")]))
        stage_indices = sorted([int(w.split("_")[-1]) for w in weight_files])
        load_paths = [os.path.join(my_path, "weights", "best_of_stage_{}".format(idx)) for idx in stage_indices]
        return [dict(load_path=lp) for lp in load_paths]

    cfg.curriculum = get_curriculum_for_idx(cfg.idx)


distributions = [dict(idx=idx) for idx in range(24)]


config = yolo_rl.small_test_config.copy(
    prepare_func=prepare_func,
    patience=2500,
    render_step=100000,
    n_val=16,
    eval_step=1000,
    do_train=False
)

# Create the datasets if necessary.
with config:
    train = EMNIST_ObjectDetection(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=distributions)
