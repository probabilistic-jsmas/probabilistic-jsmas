from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

import tensorflow as tf
import numpy as np


def attack(model, x, theta=1., clip_min=0., clip_max=1., max_iter=60, cast=tf.float32, use_logits=True, non_stop=False):
    """
    Original tensorflow implementation of Taylor JSMA for non-targeted attacks.
    Paper link: https://arxiv.org/pdf/2007.06032

    Parameters
    ----------
    model: cleverhans.model.Model
        The cleverhans model.
    x: tf.Tensor
        The input tf placeholder.
    theta: float
        The amount by which the pixel are modified (can either be positive or negative).
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

    y = tf.one_hot(tf.argmax(original_preds, axis=-1), depth=nb_classes, axis=1)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = tf.constant(tmp, cast)

    increase = theta > 0

    if increase:
        search_domain = tf.reshape(tf.cast(x < clip_max, cast), [-1, nb_features])
    else:
        search_domain = tf.reshape(tf.cast(x > clip_min, cast), [-1, nb_features])

    def condition(x_in, y_in, domain_in, i_in, cond_in):
        return tf.logical_and(tf.less(i_in, max_iter), cond_in)

    def body(x_in, y_in, domain_in, i_in, cond_in):
        logits = model.get_logits(x_in)
        preds = tf.nn.softmax(logits)
        preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

        if use_logits:
            derivatives = tf.reshape(batch_jacobian(logits, x_in), shape=(-1, nb_classes, nb_features))
        else:
            derivatives = tf.reshape(batch_jacobian(preds, x_in), shape=(-1, nb_classes, nb_features))

        derivatives = tf.reshape(1 - x_in, shape=(-1, nb_features)) * tf.transpose(derivatives, perm=(1, 0, 2))

        target_class = tf.reshape(tf.transpose(y_in, perm=(1, 0)), shape=(nb_classes, -1, 1))
        other_classes = tf.cast(tf.not_equal(target_class, 1), cast)

        preds_transpose = tf.transpose(tf.reshape(preds, shape=(-1, nb_classes, 1)), perm=(1, 0, 2))
        other_classes_weighted = other_classes * preds_transpose

        grads_target = tf.reduce_sum(derivatives * target_class, axis=0)
        grads_other = tf.reduce_sum(derivatives * other_classes_weighted, axis=0)

        increase_coef = - (4 * int(increase) - 2) * tf.cast(tf.equal(domain_in, 0), cast)

        target_sum = grads_target - increase_coef * tf.reduce_max(tf.abs(grads_target), axis=1, keep_dims=True)
        target_sum = tf.reshape(target_sum, shape=(-1, nb_features, 1)) + \
            tf.reshape(target_sum, shape=(-1, 1, nb_features))

        other_sum = grads_other + increase_coef * tf.reduce_max(tf.abs(grads_other), axis=1, keep_dims=True)
        other_sum = tf.reshape(other_sum, shape=(-1, nb_features, 1)) + \
            tf.reshape(other_sum, shape=(-1, 1, nb_features))

        if increase:
            scores_masks = ((target_sum < 0) & (other_sum > 0))
        else:
            scores_masks = ((target_sum > 0) & (other_sum < 0))

        scores = tf.cast(scores_masks, cast) * (-target_sum * other_sum) * zero_diagonal
        best = tf.argmax(tf.reshape(scores, shape=(-1, nb_features * nb_features)), axis=1)

        p1 = tf.mod(best, nb_features)
        p2 = tf.floor_div(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)

        if not non_stop:
            cond = tf.equal(tf.reduce_sum(y_in * preds_onehot, axis=1), 1) & (tf.reduce_sum(domain_in, axis=1) >= 2)
        else:
            cond = tf.reduce_sum(domain_in, axis=1) >= 2

        cond_float = tf.reshape(tf.cast(cond, cast), shape=(-1, 1))

        to_mod = (p1_one_hot + p2_one_hot) * cond_float
        to_mod_reshape = tf.reshape(to_mod, shape=(-1, width, height, depth))

        if increase:
            x_out = tf.minimum(clip_max, x_in + to_mod_reshape * theta)
        else:
            x_out = tf.maximum(clip_min, x_in - to_mod_reshape * theta)

        i_out = tf.add(i_in, 1)
        cond_out = tf.reduce_any(cond)

        domain_out = domain_in - to_mod

        return x_out, y_in, domain_out, i_out, cond_out

    x_adv, _, _, _, _ = tf.while_loop(condition, body, [x, y, search_domain, 0, True], parallel_iterations=1)

    return x_adv
