# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dps import cfg
from dps.utils import Param, Parameterized, Config, square_subplots
from dps.utils.tf import trainable_variables, build_scheduled_value
from dps.env.advanced import yolo_rl

tf_flatten = tf.layers.flatten


# ------ transformer.py -------


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f-1.001) / 2.0
            y = (y + 1.0)*(height_f-1.001) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output

# ------ vae.py -------


def vae(inputs, input_dim, rec_hidden_units, latent_dim,
        gen_hidden_units, likelihood_std=0.0, activation=tf.nn.softplus):

    batch_size = tf.shape(inputs)[0]

    next_layer = inputs
    for i in range(len(rec_hidden_units)):
        with tf.variable_scope("recognition_" + str(i+1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, rec_hidden_units[i], activation_fn=activation, scope=scope
            )

    with tf.variable_scope("rec_mean") as scope:
        recognition_mean = layers.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)
    with tf.variable_scope("rec_log_variance") as scope:
        recognition_log_variance = layers.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("rec_sample"):
        standard_normal_sample = tf.random_normal([batch_size, latent_dim])
        recognition_sample = recognition_mean + standard_normal_sample * tf.sqrt(tf.exp(recognition_log_variance))

    next_layer = recognition_sample
    for i in range(len(gen_hidden_units)):
        with tf.variable_scope("generative_" + str(i+1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, gen_hidden_units[i], activation_fn=activation, scope=scope
            )

    with tf.variable_scope("gen_mean") as scope:
        generative_mean = layers.fully_connected(next_layer, input_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("gen_sample"):
        standard_normal_sample2 = tf.random_normal([batch_size, input_dim])
        generative_sample = generative_mean + standard_normal_sample2 * likelihood_std
        reconstruction = tf.nn.sigmoid(generative_sample)

    return reconstruction, recognition_mean, recognition_log_variance, recognition_mean


# ------ concrete.py -------


def concrete_binary_sample(log_odds, temperature, hard=False, eps=10e-10):
    count = tf.shape(log_odds)[0]

    u = tf.random_uniform([count], minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)

    y = log_odds + noise
    sig_y = tf.nn.sigmoid(y / temperature)

    if hard:
        sig_y_hard = tf.round(sig_y)
        sig_y = tf.stop_gradient(sig_y_hard - sig_y) + sig_y

    return y, sig_y


def concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=10e-10):
    count = tf.shape(log_odds)[0]

    u = tf.random_uniform([count], minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)
    y = (log_odds + noise) / temperature

    return y


def concrete_binary_kl_mc_sample(y,
                                 prior_log_odds, prior_temperature,
                                 posterior_log_odds, posterior_temperature,
                                 eps=10e-10):

    y_times_prior_temp = y * prior_temperature
    log_prior = tf.log(prior_temperature + eps) - y_times_prior_temp + prior_log_odds - \
        2.0 * tf.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)

    y_times_posterior_temp = y * posterior_temperature
    log_posterior = tf.log(posterior_temperature + eps) - y_times_posterior_temp + posterior_log_odds - \
        2.0 * tf.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)

    return log_posterior - log_prior


def _sample_from_mvn(mean, diag_variance):
    """ sample from multivariate normal with given mean and diagonal covariance """
    standard_normal = tf.random_normal(tf.shape(mean))
    return mean + standard_normal * tf.sqrt(diag_variance)


class AIR_RenderHook(object):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        if updater.stage_idx == 0:
            path = updater.exp_dir.path_for('plots', 'frames.pdf')
            if not os.path.exists(path):
                fig, axes = square_subplots(16)
                for ax, frame in zip(axes.flatten(), updater.datasets['train'].x):
                    ax.imshow(frame)

                fig.savefig(path)
                plt.close(fig)

        fetched = self._fetch(self.N, updater)

        self._plot_reconstruction(updater, fetched)

    def _fetch(self, N, updater):
        network = updater.network

        feed_dict, other = updater.make_feed_dict(N, 'val', True)
        network.update_feed_dict(feed_dict, other)

        images = feed_dict[network.inp_ph]

        to_fetch = {}
        to_fetch["output"] = network.reconstruction
        to_fetch["transform"] = network.rec_st_back
        to_fetch["steps"] = network.rec_num_digits
        to_fetch["background"] = network.background
        to_fetch["windows"] = network.rec_windows

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)
        fetched["annotations"] = other["annotations"]

        return fetched

    def _plot_reconstruction(self, updater, fetched):
        images = fetched['images'].reshape(-1, *updater.network.obs_shape)
        _, image_height, image_width, _ = images.shape
        output = fetched['output'].reshape(-1, *updater.network.obs_shape)
        object_shape = updater.network.object_shape
        windows = fetched['windows'].reshape(
            -1, updater.network.max_air_steps, object_shape[0], object_shape[1], updater.network.image_depth)
        background = fetched['background']

        N = images.shape[0]
        transform = fetched['transform'].reshape(N, -1, 6)

        annotations = fetched["annotations"]
        if annotations is None:
            annotations = [[]] * N

        color_order = "red green blue".split()

        fig, axes = plt.subplots(N, updater.network.max_air_steps + 3, figsize=(20, 20))
        for i in range(N):
            ax_gt = axes[i, 0]
            ax_gt.imshow(images[i])
            ax_gt.set_title('ground truth')

            ax_rec = axes[i, 1]
            ax_rec.imshow(output[i])
            ax_rec.set_title('reconstruction')

            # Plot true bounding boxes
            for a in annotations[i]:
                _, t, b, l, r = a
                h = b - t
                w = r - l

                rect = patches.Rectangle(
                    (l, t), w, h, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax_rec.add_patch(rect)

            ax_bg = axes[i, 2]
            ax_bg.imshow(background[i])
            ax_bg.set_title('background')

            for t in range(updater.network.max_air_steps):
                inv_s, _, neg_x_over_s, _, _, neg_y_over_s = transform[i, t, :]

                s = 1 / inv_s
                x = -neg_x_over_s * s
                y = -neg_y_over_s * s

                transformed_x = 0.5 * (x + 1.)
                transformed_y = 0.5 * (y + 1.)

                h = s * image_height
                w = s * image_width

                top = image_height * transformed_y - h / 2
                left = image_width * transformed_x - w / 2

                rect = patches.Rectangle(
                    (left, top), w, h, linewidth=1, edgecolor=color_order[t], facecolor='none')
                ax_rec.add_patch(rect)

                ax = axes[i, t+3]
                ax.imshow(windows[i, t])

        fig.suptitle('Stage={}. After {} experiences ({} updates, {} experiences per batch).'.format(
            updater.stage_idx, updater.n_experiences, updater.n_updates, cfg.batch_size))

        path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), 'sampled_reconstruction.pdf')
        fig.savefig(path)

        plt.close(fig)


class AIR_Network(Parameterized):
    max_air_steps = Param()
    max_chars = Param()
    rnn_units = Param()
    object_shape = Param()

    vae_latent_dimensions = Param()
    vae_recognition_units = Param()
    vae_generative_units = Param()

    scale_prior_mean = Param()
    scale_prior_variance = Param()
    shift_prior_mean = Param()
    shift_prior_variance = Param()
    vae_prior_mean = Param()
    vae_prior_variance = Param()
    vae_likelihood_std = Param()

    scale_hidden_units = Param()
    shift_hidden_units = Param()
    z_pres_hidden_units = Param()

    z_pres_prior_log_odds = Param()
    z_pres_temperature = Param()
    stopping_threshold = Param()

    cnn = Param()
    cnn_filters = Param()

    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    def __init__(self, env, scope=None, **kwargs):
        self.rec_num_digits = None
        self.rec_scales = None
        self.rec_shifts = None
        self.reconstruction = None
        self.loss = None
        self.accuracy = None

        self.datasets = env.datasets
        for dset in self.datasets.values():
            dset.reset()

        self.obs_shape = self.datasets['train'].x.shape[1:]
        self.image_height, self.image_width, self.image_depth = self.obs_shape

    def trainable_variables(self, for_opt, rl_only=False):
        tvars = trainable_variables(self.scope, for_opt=for_opt)
        return tvars

    def update_feed_dict(self, feed_dict, other):
        batch_size = feed_dict[self.inp_ph].shape[0]
        annotations = other.get("annotations", [[]] * batch_size)
        feed_dict[self.target_n_digits] = [len(a) for a in annotations]

    def _summarize_by_digit_count(self, tensor, digits, name):
        float_tensor = tf.to_float(tensor)

        recorded_tensors = {}
        for i in range(self.max_chars+1):
            recorded_tensors["{}_{}_dig".format(name, i)] = \
                tf.reduce_mean(tf.boolean_mask(float_tensor, tf.equal(digits, i)))

        recorded_tensors[name + "_all_dig"] = tf.reduce_mean(float_tensor)

        return recorded_tensors

    def _summarize_by_step(self, tensor, steps, name, one_more_step=False, all_steps=False):
        # padding (if required) the number of rows in the tensor
        # up to self.max_air_steps to avoid possible "out of range" errors
        # in case if there were less than self.max_air_steps steps globally
        tensor = tf.pad(tensor, [[0, 0], [0, self.max_air_steps - tf.shape(tensor)[1]]])

        recorded_tensors = {}

        for i in range(self.max_air_steps):
            if all_steps:
                # summarizing the entire (i+1)-st step without
                # differentiating between actual step counts
                _recorded_tensors = self._summarize_by_digit_count(
                    tensor[:, i], self.target_n_digits,
                    name + "_" + str(i+1) + "_step"
                )
            else:
                # considering one more step if required by one_more_step=True
                # by subtracting 1 from loop variable i (e.g. 0 steps > -1)
                mask = tf.greater(steps, i - (1 if one_more_step else 0))

                # summarizing (i+1)-st step only for those
                # batch items that actually had (i+1)-st step
                _recorded_tensors = self._summarize_by_digit_count(
                    tf.boolean_mask(tensor[:, i], mask),
                    tf.boolean_mask(self.target_n_digits, mask),
                    name + "_" + str(i+1) + "_step"
                )
            recorded_tensors.update(_recorded_tensors)
        return recorded_tensors

    def build_graph(self, **kwargs):
        with tf.variable_scope('AIR'):
            self.scope = tf.get_variable_scope()
            return self._build_graph(**kwargs)

    def _build_graph(self, loss_key, inp, inp_ph, is_training, float_is_training, background):

        # --- process input ---

        self.inp_ph = inp_ph

        # This is the form expected by the AIR code. It reshapes it to a volume when necessary.
        self.input_images = tf_flatten(inp)

        self.batch_size = tf.shape(self.input_images)[0]

        self.is_training = is_training
        self.float_is_training = float_is_training
        self.background = background
        self.loss_key = loss_key

        self.target_n_digits = tf.placeholder(tf.int32, (None,), name="target_n_digits")

        # --- build graph ---

        self.scale_prior_log_variance = tf.log(self.scale_prior_variance, name="scale_prior_log_variance")
        self.shift_prior_log_variance = tf.log(self.shift_prior_variance, name="shift_prior_log_variance")
        self.vae_prior_log_variance = tf.log(self.vae_prior_variance, name="vae_prior_log_variance")

        self.z_pres_prior_log_odds = build_scheduled_value(self.z_pres_prior_log_odds, "z_pres_prior_log_odds")

        # --- condition of while-loop ---

        def cond(step, stopping_sum, *_):
            return tf.logical_and(
                tf.less(step, self.max_air_steps),
                tf.reduce_any(tf.less(stopping_sum, self.stopping_threshold))
            )

        # --- body of while-loop ---

        def body(step, stopping_sum, prev_state,
                 running_recon, running_loss, running_digits,
                 scales_ta, shifts_ta, z_pres_probs_ta,
                 z_pres_kls_ta, scale_kls_ta, shift_kls_ta, vae_kls_ta,
                 st_backward_ta, windows_ta, latents_ta):

            with tf.variable_scope("rnn") as scope:
                # RNN time step
                outputs, next_state = cell(self.rnn_input, prev_state, scope=scope)

            with tf.variable_scope("scale"):
                # sampling scale
                with tf.variable_scope("mean"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.scale_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        scale_mean = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
                with tf.variable_scope("log_variance"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.scale_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        scale_log_variance = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
                scale_variance = tf.exp(scale_log_variance)
                scale = tf.nn.sigmoid(_sample_from_mvn(scale_mean, scale_variance))
                scales_ta = scales_ta.write(scales_ta.size(), scale)
                s = scale[:, 0]

            with tf.variable_scope("shift"):
                # sampling shift
                with tf.variable_scope("mean"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.shift_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        shift_mean = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
                with tf.variable_scope("log_variance"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.shift_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        shift_log_variance = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
                shift_variance = tf.exp(shift_log_variance)
                shift = tf.nn.tanh(_sample_from_mvn(shift_mean, shift_variance))
                shifts_ta = shifts_ta.write(shifts_ta.size(), shift)
                x, y = shift[:, 0], shift[:, 1]

            # Everything is treated as vector (ie has shape (batch_size, vector_dim)) except when applying
            # the function `transformation`, at which point we shape the vectors into a volume of appropriate size.
            # We reshape back to flat vector immediately afterwards.

            with tf.variable_scope("st_forward"):
                # ST: theta of forward transformation
                theta = tf.stack([
                    tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(x, 1)], axis=1),
                    tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(y, 1)], axis=1),
                ], axis=1)

                # ST forward transformation: canvas -> window
                window = transformer(
                    tf.reshape(self.input_images, (-1,) + self.obs_shape),
                    theta,
                    self.object_shape
                )

                window = tf.reshape(window, [-1, self.object_shape[0] * self.object_shape[1] * self.image_depth])

            with tf.variable_scope("vae"):
                # reconstructing the window in VAE

                vae_recon, vae_mean, vae_log_variance, vae_latent = vae(
                    window,
                    self.object_shape[0] * self.object_shape[1] * self.image_depth,
                    self.vae_recognition_units,
                    self.vae_latent_dimensions,
                    self.vae_generative_units,
                    self.vae_likelihood_std
                )

                # collecting individual reconstruction windows
                # for each of the inferred digits on the canvas
                windows_ta = windows_ta.write(windows_ta.size(), vae_recon)

                # collecting individual latent variable values
                # for each of the inferred digits on the canvas
                latents_ta = latents_ta.write(latents_ta.size(), vae_latent)

            with tf.variable_scope("st_backward"):
                # ST: theta of backward transformation
                theta_recon = tf.stack([
                    tf.concat([tf.stack([1.0 / s, tf.zeros_like(s)], axis=1), tf.expand_dims(-x / s, 1)], axis=1),
                    tf.concat([tf.stack([tf.zeros_like(s), 1.0 / s], axis=1), tf.expand_dims(-y / s, 1)], axis=1),
                ], axis=1)

                # collecting backward transformation matrices of ST
                # to be used for visualizing the attention windows
                st_backward_ta = st_backward_ta.write(st_backward_ta.size(), theta_recon)

                # ST backward transformation: window -> canvas
                window_recon = transformer(
                    tf.reshape(vae_recon, (-1,) + self.object_shape + (self.image_depth,)),
                    theta_recon,
                    self.obs_shape
                )

                window_recon = tf.reshape(window_recon, [-1, self.image_height * self.image_width * self.image_depth])

            with tf.variable_scope("z_pres"):
                # sampling relaxed (continuous) value of z_pres flag
                # from Concrete distribution (closer to 1 - more digits,
                # closer to 0 - no more digits)
                with tf.variable_scope("log_odds"):
                    with tf.variable_scope("hidden") as scope:
                        hidden = layers.fully_connected(outputs, self.z_pres_hidden_units, scope=scope)
                    with tf.variable_scope("output") as scope:
                        z_pres_log_odds = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)[:, 0]
                with tf.variable_scope("gumbel"):
                    # sampling pre-sigmoid value from concrete distribution
                    # with given location (z_pres_log_odds) and temperature
                    z_pres_pre_sigmoid = concrete_binary_pre_sigmoid_sample(
                        z_pres_log_odds, self.z_pres_temperature
                    )

                    # applying sigmoid to render the Concrete sample
                    z_pres = tf.nn.sigmoid(z_pres_pre_sigmoid)

                    # during test time, rounding the Concrete sample
                    # to obtain the corresponding Bernoulli sample
                    z_pres = (
                        self.float_is_training * z_pres +
                        (1 - self.float_is_training) * tf.round(z_pres)
                    )

                    # computing and collecting underlying Bernoulli
                    # probability from inferred log-odds solely for
                    # analysis purposes (not used in the model)
                    z_pres_prob = tf.nn.sigmoid(z_pres_log_odds)
                    z_pres_probs_ta = z_pres_probs_ta.write(z_pres_probs_ta.size(), z_pres_prob)

            with tf.variable_scope("loss/z_pres_kl"):
                # z_pres KL-divergence:
                # previous value of stop_sum is used
                # to account for KL of first z_pres after
                # stop_sum becomes >= 1.0
                z_pres_kl = concrete_binary_kl_mc_sample(
                    z_pres_pre_sigmoid,
                    self.z_pres_prior_log_odds, self.z_pres_temperature,
                    z_pres_log_odds, self.z_pres_temperature
                )

                # adding z_pres KL scaled by z_pres to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    z_pres_kl,
                    tf.zeros_like(running_loss)
                )

                # populating z_pres KL's TensorArray with a new value
                z_pres_kls_ta = z_pres_kls_ta.write(z_pres_kls_ta.size(), z_pres_kl)

            # updating stop sum by adding (1 - z_pres) to it:
            # for small z_pres values stop_sum becomes greater
            # or equal to self.stopping_threshold and attention
            # counting of the corresponding batch item stops
            stopping_sum += (1.0 - z_pres)

            # updating inferred number of digits per batch item
            running_digits += tf.cast(tf.less(stopping_sum, self.stopping_threshold), tf.int32)

            with tf.variable_scope("canvas"):
                # continuous relaxation:
                # adding reconstructed window scaled
                # by z_pres to the running canvas
                running_recon += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    tf.expand_dims(z_pres, 1) * window_recon,
                    tf.zeros_like(running_recon)
                )

            with tf.variable_scope("loss/scale_kl"):
                # scale KL-divergence
                scale_kl = 0.5 * tf.reduce_sum(
                    self.scale_prior_log_variance - scale_log_variance -
                    1.0 + scale_variance / self.scale_prior_variance +
                    tf.square(scale_mean - self.scale_prior_mean) / self.scale_prior_variance, 1
                )

                # adding scale KL scaled by z_pres to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    scale_kl,
                    tf.zeros_like(running_loss)
                )

                # populating scale KL's TensorArray with a new value
                scale_kls_ta = scale_kls_ta.write(scale_kls_ta.size(), scale_kl)

            with tf.variable_scope("loss/shift_kl"):
                # shift KL-divergence
                shift_kl = 0.5 * tf.reduce_sum(
                    self.shift_prior_log_variance - shift_log_variance -
                    1.0 + shift_variance / self.shift_prior_variance +
                    tf.square(shift_mean - self.shift_prior_mean) / self.shift_prior_variance, 1
                )

                # adding shift KL scaled by z_pres to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    shift_kl,
                    tf.zeros_like(running_loss)
                )

                # populating shift KL's TensorArray with a new value
                shift_kls_ta = shift_kls_ta.write(shift_kls_ta.size(), shift_kl)

            with tf.variable_scope("loss/VAE_kl"):
                # VAE KL-divergence
                vae_kl = 0.5 * tf.reduce_sum(
                    self.vae_prior_log_variance - vae_log_variance -
                    1.0 + tf.exp(vae_log_variance) / self.vae_prior_variance +
                    tf.square(vae_mean - self.vae_prior_mean) / self.vae_prior_variance, 1
                )

                # adding VAE KL scaled by (1-z_pres) to the loss
                # for those batch items that are not yet finished
                running_loss += tf.where(
                    tf.less(stopping_sum, self.stopping_threshold),
                    vae_kl,
                    tf.zeros_like(running_loss)
                )

                # populating VAE KL's TensorArray with a new value
                vae_kls_ta = vae_kls_ta.write(vae_kls_ta.size(), vae_kl)

            # explicating the shape of "batch-sized"
            # tensors for TensorFlow graph compiler
            stopping_sum.set_shape([None])
            running_digits.set_shape([None])
            running_loss.set_shape([None])

            return step + 1, stopping_sum, next_state, \
                running_recon, running_loss, running_digits, \
                scales_ta, shifts_ta, z_pres_probs_ta, \
                z_pres_kls_ta, scale_kls_ta, shift_kls_ta, vae_kls_ta, \
                st_backward_ta, windows_ta, latents_ta

        # --- end of while-loop body ---

        if self.cnn:
            with tf.variable_scope("cnn") as cnn_scope:
                # The cnn is hard-coded here...

                cnn_input = tf.reshape(self.input_images, [-1, 50, 50, 3], name="cnn_input")

                conv1 = tf.layers.conv2d(
                    inputs=cnn_input, filters=self.cnn_filters, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, reuse=cnn_scope.reuse, name="conv1"
                )

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")

                conv2 = tf.layers.conv2d(
                    inputs=pool1, filters=self.cnn_filters, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, reuse=cnn_scope.reuse, name="conv2"
                )

                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")

                conv3 = tf.layers.conv2d(
                    inputs=pool2, filters=self.cnn_filters, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, reuse=cnn_scope.reuse, name="conv3"
                )

                self.rnn_input = tf.reshape(conv3, [-1, 12 * 12 * self.cnn_filters], name="cnn_output")
        else:
            self.rnn_input = self.input_images

        with tf.variable_scope("rnn") as rnn_scope:
            # creating RNN cells and initial state
            cell = rnn.BasicLSTMCell(self.rnn_units, reuse=rnn_scope.reuse)
            rnn_init_state = cell.zero_state(
                self.batch_size, self.input_images.dtype
            )

            # RNN while_loop with variable number of steps for each batch item
            _, _, _, reconstruction, running_loss, self.rec_num_digits, scales, shifts, \
                z_pres_probs, z_pres_kls, scale_kls, shift_kls, vae_kls, \
                st_backward, windows, latents = tf.while_loop(
                    cond, body, [
                        tf.constant(0),                                 # RNN time step, initially zero
                        tf.zeros([self.batch_size]),                    # running sum of z_pres samples
                        rnn_init_state,                                 # initial RNN state
                        tf.zeros_like(self.input_images),               # reconstruction canvas, initially empty
                        tf.zeros([self.batch_size]),                    # running value of the loss function
                        tf.zeros([self.batch_size], dtype=tf.int32),    # running inferred number of digits
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred scales
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # inferred shifts
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # z_pres probabilities
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # z_pres KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # scale KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # shift KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # VAE KL-divergence
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # backward ST matrices
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True),   # individual recon. windows
                        tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)    # latents of individual digits
                    ]
                )

        # transposing contents of TensorArray's fetched from while_loop iterations
        self.rec_scales = tf.transpose(scales.stack(), (1, 0, 2), name="rec_scales")
        self.rec_shifts = tf.transpose(shifts.stack(), (1, 0, 2), name="rec_shifts")
        self.rec_st_back = tf.transpose(st_backward.stack(), (1, 0, 2, 3), name="rec_st_back")
        self.rec_windows = tf.transpose(windows.stack(), (1, 0, 2), name="rec_windows")
        self.rec_latents = tf.transpose(latents.stack(), (1, 0, 2), name="rec_latents")
        self.z_pres_probs = tf.transpose(z_pres_probs.stack(), name="z_pres_probs")
        self.z_pres_kls = tf.transpose(z_pres_kls.stack(), name="z_pres_kls")
        self.scale_kls = tf.transpose(scale_kls.stack(), name="scale_kls")
        self.shift_kls = tf.transpose(shift_kls.stack(), name="shift_kls")
        self.vae_kls = tf.transpose(vae_kls.stack(), name="vae_kls")

        reconstruction = tf.clip_by_value(reconstruction, 1e-6, 1-1e-6)
        reconstruction = tf.reshape(reconstruction, (-1,) + self.obs_shape)

        self.reconstruction = reconstruction

        output_logits = tf.log(reconstruction / (1 - reconstruction))

        tensors = {"output_logits": output_logits}

        losses = {}
        reconstruction_loss = tf.reduce_sum(tf_flatten(yolo_rl.loss_builders[loss_key](output_logits, inp)), axis=1)
        losses['reconstruction'] = tf.reduce_mean(reconstruction_loss)
        losses['running'] = tf.reduce_mean(running_loss)  # the shape of running_loss is (batch_size,)

        recorded_tensors = {}

        # accuracy of inferred number of digits
        count_accuracy = 1 - tf.to_float(tf.equal(self.target_n_digits, self.rec_num_digits))

        # post while-loop numeric summaries grouped by digit count

        rt = self._summarize_by_digit_count(self.rec_num_digits, self.target_n_digits, "steps")
        recorded_tensors.update(rt)

        rt = self._summarize_by_digit_count(reconstruction_loss, self.target_n_digits, "rec_loss")
        recorded_tensors.update(rt)

        rt = self._summarize_by_digit_count(count_accuracy, self.target_n_digits, "digit_acc")
        recorded_tensors.update(rt)

        rt = self._summarize_by_digit_count(running_loss, self.target_n_digits, "running_loss")
        recorded_tensors.update(rt)

        # --- grouped by digit count of ground-truth image and step ---

        rt = self._summarize_by_step(self.rec_scales[:, :, 0], self.rec_num_digits, "scale")
        recorded_tensors.update(rt)

        rt = self._summarize_by_step(self.z_pres_probs, self.rec_num_digits, "z_pres_prob", all_steps=True)
        recorded_tensors.update(rt)

        rt = self._summarize_by_step(self.z_pres_kls, self.rec_num_digits, "z_pres_kl", one_more_step=True)
        recorded_tensors.update(rt)

        rt = self._summarize_by_step(self.scale_kls, self.rec_num_digits, "scale_kl")
        recorded_tensors.update(rt)

        rt = self._summarize_by_step(self.shift_kls, self.rec_num_digits, "shift_kl")
        recorded_tensors.update(rt)

        rt = self._summarize_by_step(self.vae_kls, self.rec_num_digits, "vae_kl")
        recorded_tensors.update(rt)

        return {
            "tensors": tensors,
            "recorded_tensors": recorded_tensors,
            "losses": losses
        }


def get_updater(env):
    network = AIR_Network(env)
    return yolo_rl.YoloRL_Updater(env, network)


config = Config(
    log_name="attend_infer_repeat",
    get_updater=get_updater,
    build_env=yolo_rl.Env,

    curriculum=[
        dict()
    ],

    # dataset params

    min_chars=1,
    max_chars=1,
    characters=[0, 1, 2],
    n_sub_image_examples=0,
    xent_loss=True,
    sub_image_shape=(28, 28),
    use_dataset_cache=True,
    image_shape=(50, 50),

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    background_cfg=dict(
        mode="none",

        static_cfg=dict(
            n_clusters=4,
            use_k_modes=True,  # If False, use k_means.
            n_clustering_examples=True,
            n_clustering_bins=None,
            min_cluster_distance=None,
        ),

        mode_threshold=0.99,
    ),

    # training loop params

    optimizer_spec="adam",
    lr_schedule=1e-4,
    max_grad_norm=1.0,
    batch_size=64,

    preserve_env=True,
    max_steps=1e7,
    patience=10000,
    use_gpu=True,
    gpu_allow_growth=True,
    seed=93434014,
    stopping_criteria="loss,min",
    threshold=-np.inf,
    max_experiments=None,
    render_hook=AIR_RenderHook(),

    render_step=100,
    eval_step=100,
    display_step=100,

    # Based on the values in tf-attend-infer-repeat/training.py
    max_air_steps=3,
    rnn_units=256,
    object_shape=(28, 28),
    vae_latent_dimensions=50,
    vae_recognition_units=(512, 256),
    vae_generative_units=(256, 512),
    scale_prior_mean=-1.0,
    scale_prior_variance=0.1,
    shift_prior_mean=0.0,
    shift_prior_variance=1.0,
    vae_prior_mean=0.0,
    vae_prior_variance=1.0,
    vae_likelihood_std=0.3,
    scale_hidden_units=64,
    shift_hidden_units=64,
    z_pres_hidden_units=64,
    z_pres_prior_log_odds="Exponential(start=10000.0, end=0.000000001, decay_rate=0.1, decay_steps=3000, log=True)",
    # z_pres_prior_log_odds=-0.01,
    z_pres_temperature=1.0,
    stopping_threshold=0.99,
    cnn=True,
    cnn_filters=8,
    annealing_schedules=None
)
