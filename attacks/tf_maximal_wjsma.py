from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

import tensorflow as tf
import numpy as np


def attack(model, x, clip_min=0., clip_max=1., max_iter=60, cast=tf.float32, use_logits=True, non_stop=False):
    """
    Original tensorflow implementation of Maximal Weighted JSMA for non-targeted attacks.
    Paper link: https://arxiv.org/pdf/2007.06032

    Parameters
    ----------
    model: cleverhans.model.Model
        The cleverhans model.
    x: tf.Tensor
        The input tf placeholder.
    clip_min: float
        Minimum component value for clipping.
    clip_max: float
        Maximum component value for clipping.
    max_iter: int
        Maximum iteration before the attack stops.
    cast: tf.dtype
        The tensor data type used.
    use_logits: bool
        Uses the logits (Z variation) when set to True and the softmax (F variation) values otherwise.
    non_stop: bool
        When set to True, the attacks continue until max_iter is reached. (used to attack the substitute for black box)

    Returns
    -------
    x_adv: tf.Tensor
        The tensor of the adversarial samples.
    """

    batch_size, width, height, depth = x.shape

    original_preds = model(x)

    nb_classes = original_preds.shape[1]
    nb_features = width * height * depth

    y = tf.one_hot(tf.argmax(original_preds, axis=1), depth=nb_classes)

    search_domain = tf.reshape(tf.ones_like(x), shape=(-1, nb_features))
    previous_mods = tf.reshape(tf.zeros_like(x), shape=(-1, nb_features))

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = tf.constant(tmp, cast)

    def condition(x_in, y_in, domain_in, mods_in, i_in, cond_in):
        return tf.logical_and(tf.less(i_in, max_iter), cond_in)

    def body(x_in, y_in, domain_in, mods_in, i_in, cond_in):
        logits = model.get_logits(x_in)
        preds = tf.nn.softmax(logits)
        preds_one_hot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

        if use_logits:
            derivatives = tf.reshape(batch_jacobian(logits, x_in), shape=(-1, nb_classes, nb_features))
        else:
            derivatives = tf.reshape(batch_jacobian(preds, x_in), shape=(-1, nb_classes, nb_features))

        derivatives = tf.transpose(derivatives, perm=(1, 0, 2))

        preds_transpose = tf.transpose(tf.reshape(preds, shape=(-1, nb_classes, 1)), perm=(1, 0, 2))
        gradients_sum = tf.reduce_sum(derivatives * preds_transpose, axis=0)

        alphas = tf.reshape(derivatives, shape=(nb_classes, -1, nb_features, 1)) + \
            tf.reshape(derivatives, shape=(nb_classes, -1, 1, nb_features))
        betas = tf.reshape(gradients_sum, shape=(-1, nb_features, 1)) + \
            tf.reshape(gradients_sum, shape=(-1, 1, nb_features))
        betas -= alphas * tf.reshape(preds_transpose, shape=(nb_classes, -1, 1, 1))

        remove_mask = tf.reshape(1 - domain_in, shape=(-1, nb_features, 1)) + \
            tf.reshape(1 - domain_in, shape=(-1, 1, nb_features))
        remove_mask = tf.transpose(tf.stack([remove_mask] * nb_classes, axis=0), perm=(1, 0, 2, 3))

        product = tf.transpose(-alphas * betas, perm=(1, 0, 2, 3))
        product -= remove_mask * tf.reshape(tf.reduce_max(tf.abs(product), axis=(1, 2, 3)), shape=(-1, 1, 1, 1))

        max_classes = tf.reduce_max(tf.reshape(product, shape=(-1, nb_classes, nb_features * nb_features)), axis=2)
        max_class = tf.argmax(max_classes, axis=1)
        max_class_one_hot = tf.one_hot(max_class, depth=nb_classes)

        scores = tf.reduce_sum(tf.reshape(max_class_one_hot, shape=(-1, nb_classes, 1, 1)) * product, axis=1)
        best = tf.argmax(tf.reshape(scores * zero_diagonal, shape=(-1, nb_features * nb_features)), axis=1)

        p1 = tf.mod(best, nb_features)
        p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)

        target_gradient = tf.reduce_sum(
            tf.reshape(max_class_one_hot, shape=(-1, nb_classes, 1)) * tf.transpose(derivatives, perm=(1, 0, 2)), axis=1
        )

        thetas = tf.sign(tf.reduce_sum(target_gradient * (p1_one_hot + p2_one_hot), axis=1))
        thetas *= - 2 * tf.reduce_sum(y_in * max_class_one_hot, axis=1) + 1

        cond = (tf.reduce_sum(domain_in, axis=1) >= 2) & tf.not_equal(tf.reduce_max(max_classes, axis=1), 0)

        if not non_stop:
            cond &= tf.equal(tf.reduce_sum(y_in * preds_one_hot, axis=1), 1)

        cond_float = tf.reshape(tf.cast(cond, dtype=cast), shape=(-1, 1))

        cond_mod_p1 = tf.not_equal(tf.reduce_sum(mods_in * p1_one_hot, axis=1), - thetas)
        cond_mod_p2 = tf.not_equal(tf.reduce_sum(mods_in * p2_one_hot, axis=1), - thetas)
        cond_mod_p1_float = tf.reshape(tf.cast(cond_mod_p1, dtype=cast), shape=(-1, 1))
        cond_mod_p2_float = tf.reshape(tf.cast(cond_mod_p2, dtype=cast), shape=(-1, 1))

        to_mod = cond_float * (p1_one_hot * cond_mod_p1_float + p2_one_hot * cond_mod_p2_float)
        to_mod_theta = to_mod * tf.reshape(thetas, shape=(-1, 1))
        to_mod_reshape = tf.reshape(to_mod_theta, shape=(-1, width, height, depth))

        x_out = tf.cast(tf.maximum(clip_min, tf.minimum(clip_max, x_in + to_mod_reshape)), dtype=cast)

        domain_out = domain_in - tf.minimum(to_mod, 1)
        mods_out = mods_in + (p1_one_hot + p2_one_hot) * (tf.expand_dims(thetas, axis=1) - mods_in)

        i_out = tf.add(i_in, 1)
        cond_out = tf.reduce_any(cond)

        return x_out, y_in, domain_out, mods_out, i_out, cond_out

    x_adv, _, _, _, _, _ = tf.while_loop(
        condition, body, [x, y, search_domain, previous_mods, 0, True], parallel_iterations=1
    )

    return x_adv
