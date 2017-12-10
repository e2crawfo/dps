# Attention for feeding into a pre-trained model.
import tensorflow as tf

from dps.utils.tf import MLP
from dps.config import DEFAULT_CONFIG
from dps.datasets import VisualArithmeticDataset
from dps.utils import image_to_string


if __name__ == "__main__":
    s = 42
    d = 14
    f = s - d + 1
    sub_image_shape = (d, d)
    image_shape = (s, s)
    dataset = VisualArithmeticDataset(
        reductions="sum", min_digits=1, max_digits=1, sub_image_shape=sub_image_shape,
        image_shape=image_shape, n_examples=1000, one_hot=True)

    images = dataset.x
    print(image_to_string(images[0, ...]))

    image_shapes = images.shape[1:]
    images = images[..., None]

    activations = tf.get_variable("activations", f**2)
    weights = tf.nn.softmax(activations)
    filt = tf.reshape(weights, (f, f, 1, 1))

    attn_result = tf.nn.convolution(images, filt, padding="VALID")

    emnist_config = Config()

    classification = cfg.build_digit_classifier()

    classifier = LeNet(128, scope="digit_classifier")
    classifier.set_pretraining_params(
        digit_config, name_params='classes include_blank shape n_controller_units',
        directory=cfg.model_dir + '/emnist_pretrained'
    )

    classification = classifier(attn_result, 10)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=actions)
    loss = tf.reduce_mean(loss)

    loss = tf.reduce_mean(z**2)

    opt = tf.train.AdamOptimizer(0.001)
    train_op = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    y = sess.run(result)
    print(y)

    for i in range(10):
        _, _loss = sess.run([train_op, loss])
        print(_loss)
        print(sess.run(loss))
