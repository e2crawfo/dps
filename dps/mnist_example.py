import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from dps import cfg
from dps.utils import Param, Config
from dps.utils.tf import MLP, build_gradient_train_op
from dps.updater import Updater as _Updater, DataManager, Evaluator, TensorRecorder
from dps.datasets.base import Environment, EmnistDataset


class Updater(_Updater):
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()
    l2_weight = Param()
    build_network = Param()

    feature_name = 'image'

    def __init__(self, env, scope=None, **kwargs):
        super(Updater, self).__init__(env, scope=scope, **kwargs)
        self.network = self.build_network(env, self, scope="network")

    def trainable_variables(self, for_opt):
        return self.network.trainable_variables(for_opt)

    def _update(self, batch_size):
        if cfg.get('no_gradient', False):
            return dict(train=dict())

        feed_dict = self.data_manager.do_train()

        sess = tf.get_default_session()
        _, record, train_record = sess.run(
            [self.train_op, self.recorded_tensors, self.train_records], feed_dict=feed_dict)
        record.update(train_record)

        return dict(train=record)

    def _evaluate(self, _batch_size, mode):
        result = self.evaluator.eval(self.recorded_tensors, self.data_manager, mode)
        return result

    def _build_graph(self):
        self.data_manager = DataManager(datasets=self.env.datasets)
        self.data_manager.build_graph()
        data = self.data_manager.iterator.get_next()

        network_inp = data[self.feature_name]

        if network_inp.dtype == tf.uint8:
            network_inp = tf.image.convert_image_dtype(network_inp, tf.float32)

        n_classes = self.env.datasets['train'].n_classes
        is_training = self.data_manager.is_training
        network_outputs = self.network(network_inp, data['label'], n_classes, is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        batch_size = tf.shape(network_inp)[0]
        float_is_training = tf.to_float(is_training)

        self.tensors = network_tensors
        self.tensors.update(data)
        self.tensors.update(
            network_inp=network_inp,
            is_training=is_training,
            float_is_training=float_is_training,
            batch_size=batch_size,
        )

        self.recorded_tensors = recorded_tensors = dict(
            global_step=tf.train.get_or_create_global_step(),
            batch_size=batch_size,
            is_training=float_is_training,
        )

        tvars = self.trainable_variables(for_opt=True)
        if self.l2_weight > 0.0:
            network_losses['l2'] = self.l2_weight * sum(tf.nn.l2_loss(v) for v in tvars if 'weights' in v.name)

        # --- loss ---

        self.loss = tf.constant(0., tf.float32)
        for name, tensor in network_losses.items():
            self.loss += tensor
            recorded_tensors['loss_' + name] = tensor
        recorded_tensors['loss'] = self.loss

        # --- train op ---

        if cfg.do_train and not cfg.get('no_gradient', False):
            tvars = self.trainable_variables(for_opt=True)

            self.train_op, self.train_records = build_gradient_train_op(
                self.loss, tvars, self.optimizer_spec, self.lr_schedule,
                self.max_grad_norm, self.noise_schedule)

        sess = tf.get_default_session()
        for k, v in getattr(sess, 'scheduled_values', {}).items():
            if k in recorded_tensors:
                recorded_tensors['scheduled_' + k] = v
            else:
                recorded_tensors[k] = v

        # --- recorded values ---

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        intersection = recorded_tensors.keys() & self.network.eval_funcs.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

        if self.network.eval_funcs:
            eval_funcs = self.network.eval_funcs
        else:
            eval_funcs = {}

        self.evaluator = Evaluator(eval_funcs, network_tensors, self)


class ClassificationNetwork(TensorRecorder):
    build_classifier = Param()
    loss_type = Param()

    def __init__(self, env, updater, scope=None, **kwargs):
        self.updater = updater
        self.eval_funcs = {}

        super(ClassificationNetwork, self).__init__(scope=scope, **kwargs)

    def _call(self, inp, label, n_classes, is_training):
        self._tensors = dict()
        self.maybe_build_subnet('classifier')

        logits = self.classifier(inp, n_classes, is_training)
        probs = tf.nn.softmax(logits, axis=1)
        predicted_labels = tf.cast(tf.argmax(logits, axis=1), label.dtype)
        one_hot_label = tf.one_hot(label, n_classes, dtype=probs.dtype)
        correct_class_prob = tf.reduce_sum(one_hot_label * probs, axis=1)

        if self.loss_type == 'xent':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)

        elif self.loss_type.startswith('focal'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
            gamma = float(self.loss_type.split(',')[1])
            focal_weight = (1 - correct_class_prob) ** gamma
            loss = focal_weight * loss

        else:
            raise Exception("Unknown loss type: {}".format(self.loss_type))

        self.losses = dict(
            classification=tf.reduce_mean(loss),
        )

        self.record_tensors(
            accuracy=tf.equal(predicted_labels, label),
            correct_class_prob=correct_class_prob,
        )

        self._tensors = dict(
            logits=logits,
            probs=probs,
            predicted_labels=predicted_labels,
        )

        return dict(
            tensors=self._tensors,
            recorded_tensors=self.recorded_tensors,
            losses=self.losses,
        )


class MnistEnvironment(Environment):
    def __init__(self):
        train_seed, val_seed, test_seed = 0, 1, 2
        train = EmnistDataset(
            n_examples=int(cfg.n_train),
            example_range=cfg.train_example_range, seed=train_seed)

        val = EmnistDataset(
            n_examples=int(cfg.n_val),
            example_range=cfg.val_example_range, seed=val_seed)

        test = EmnistDataset(
            n_examples=int(cfg.n_val),
            example_range=cfg.test_example_range, seed=test_seed)

        self.datasets = dict(train=train, val=val, test=test)


def mnist_config_func():
    env_name = 'mnist'
    eval_step = 1000
    display_step = -1
    checkpoint_step = -1
    weight_step = -1
    backup_step = -1

    shuffle_buffer_size = 10000

    n_train = 60000
    n_val = 10000

    train_example_range = (0, 0.8)
    val_example_range = (0.8, 0.9)
    test_example_range = (0.9, 1.0)

    build_env = MnistEnvironment

    classes = list('0123456789')
    image_shape = (28, 28)
    include_blank = False
    balance = False

    return locals()


def mlp_config_func():
    alg_name = 'mlp'
    build_classifier = lambda scope: MLP(n_units=[128, 128, 128, 128], scope=scope)

    batch_size = 32
    log_device_placement = False
    use_gpu = False
    loss_type = 'xent'

    get_updater = Updater

    l2_weight = 0.000
    stopping_criteria = 'loss,min'
    threshold = -np.inf
    optimizer_spec = 'adam'
    lr_schedule = 1e-4
    noise_schedule = 0.0
    max_grad_norm = 1.0
    build_network = ClassificationNetwork
    patience = 1000000

    curriculum = [
        dict(),
        dict(lr_schedule=.3*lr_schedule),
        dict(lr_schedule=.09*lr_schedule),
        dict(lr_schedule=0.027*lr_schedule)
    ]

    return locals()


mnist_config = Config(mnist_config_func())
mlp_config = Config(mlp_config_func())
