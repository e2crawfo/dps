import clify

from dps import cfg
from dps.env.advanced import yolo_rl
from dps.datasets import EMNIST_ObjectDetection


distributions = dict(
    area_weight=list([1.5, 2.0, 2.5, 3.0]),
    nonzero_weight=list([150, 200, 250, 300]),
)

config = yolo_rl.good_sequential_config.copy(
    render_step=100000,
    eval_step=1000,

    max_experiences=175000,
    patience=10000000,
    max_steps=1000000,

    area_weight=None,
    nonzero_weight=None,
)

config.curriculum[-1]['max_experiences'] = 100000000

# Create the datasets if necessary.
print("Forcing creation of first dataset.")
with config:
    train = EMNIST_ObjectDetection(n_examples=int(config.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(config.n_val), shuffle=True, example_range=(0.9, 1.))

print("Forcing creation of second dataset.")
with config.copy(config.curriculum[-1]):
    train = EMNIST_ObjectDetection(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
    val = EMNIST_ObjectDetection(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

from dps.hyper import build_and_submit
clify.wrap_function(build_and_submit)(config=config, distributions=distributions)
