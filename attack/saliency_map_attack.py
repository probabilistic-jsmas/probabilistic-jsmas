"""
The SalienceMapMethod attack with a weighted parameter and returning the adversarial sample and the prediction
probabilities for each iteration. Method inspired from the SalienceMapMethod of the cleverhans module.
"""

import warnings

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks import Attack
from cleverhans.compat import reduce_sum, reduce_max, reduce_any

tf_dtype = tf.as_dtype('float32')


class SaliencyMapMethod(Attack):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Initializes the SaliencyMapMethod.

        Note
        ----
            The model parameter should be an instance of the cleverhans.model.Model abstraction provided by CleverHans.

        Parameters
        ----------
        model: cleverhans.model.Model
            An instance of the cleverhans.model.Model class.
        sess: tf.Session, optional
            The (possibly optional) tf.Session to run graphs in.
        dtypestr: str, optional
            Floating point precision to use (change to float64 to avoid numerical instabilities).
        """

        super(SaliencyMapMethod, self).__init__(model, sess, dtypestr, **kwargs)

        self.feedable_kwargs = ('y_target',)
        self.structural_kwargs = [
            'theta', 'gamma', 'clip_max', 'clip_min', 'symbolic_impl', 'attack'
        ]

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        See Also
        --------
            For kwargs, see `parse_params`.

        Parameters
        ----------
        x:
            The model's symbolic inputs.

        Returns
        -------
        x_adv:
            A symbolic representation of the adversarial examples.
        """

        assert self.parse_params(**kwargs)

        if self.symbolic_impl:
            if self.y_target is None:
                raise NotImplementedError("Non targeted JSMA/WJSMA/TJSMA are not implemented.")

            x_adv = jsma_symbolic(
                x,
                model=self.model,
                y_target=self.y_target,
                theta=self.theta,
                gamma=self.gamma,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                attack=self.attack
            )

        else:
            raise NotImplementedError("The jsma_batch function has been removed."
                                      " The symbolic_impl argument to SaliencyMapMethod will be removed"
                                      " on 2019-07-18 or after. Any code that depends on the non-symbolic"
                                      " implementation of the JSMA should be revised. Consider using"
                                      " SaliencyMapMethod.generate_np() instead."
                                      )

        return x_adv

    def parse_params(self, theta=1., gamma=1., clip_min=0., clip_max=1., y_target=None, attack="jsma",
                     symbolic_impl=True, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        Parameters
        ----------
        theta: float, optional
            Perturbation introduced to modified components (can be positive or negative).
        gamma: float, optional
            Maximum percentage of perturbed features.
        clip_min: float, optional
            Minimum component value for clipping.
        clip_max: float, optional
            Maximum component value for clipping.
        y_target: tf.Tensor, optional
            Target tensor if the attack is targeted.
        attack: str, optional
            The type of used attack (either "jsma", "wjsma" or "tjsma").
        symbolic_impl: bool, optional
            Uses the symbolic version of the attack if set to True (must be True)

        Returns
        -------
        success: bool
            True when parsing was successful
        """

        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y_target = y_target
        self.symbolic_impl = symbolic_impl
        self.attack = attack

        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26."
                          )

        return True


def jsma_batch(*args, **kwargs):
    raise NotImplementedError(
        "The jsma_batch function has been removed. Any code that depends on it should be revised."
    )


def jsma_symbolic(x, y_target, model, theta, gamma, clip_min, clip_max, attack):
    """
    Modified version of the TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528 for details
    about the algorithm design choices).

    Parameters
    ----------
    x: tf.Tensor
        The input tf placeholder.
    y_target: tf.Tensor
        The target tensor.
    model: cleverhans.model.Model
        The cleverhans model obejct.
    theta: float
        The amount by which the pixel are modified.
    gamma: float
        Between 0 and 1, it specifies the maximum distortion percentage.
    clip_min: float
        Minimum component value for clipping.
    clip_max: float
        Maximum component value for clipping.
    attack: str
        The type of used attack (either "jsma", "wjsma" or "tjsma").

    Returns
    -------
    x_adv: tf.Tensor
        The tensor of the adversarial samples.
    """

    nb_classes = int(y_target.shape[-1].value)
    nb_features = int(np.product(x.shape[1:]).value)

    if x.dtype == tf.float32 and y_target.dtype == tf.int64:
        y_target = tf.cast(y_target, tf.int32)

    if x.dtype == tf.float32 and y_target.dtype == tf.float64:
        warnings.warn("Downcasting labels---this should be harmless unless they are smoothed")
        y_target = tf.cast(y_target, tf.float32)

    max_iters = np.floor(nb_features * gamma / 2)
    increase = bool(theta > 0)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = tf.constant(tmp, tf_dtype)

    if increase:
        search_domain = tf.reshape(
            tf.cast(x < clip_max, tf_dtype), [-1, nb_features])
    else:
        search_domain = tf.reshape(
            tf.cast(x > clip_min, tf_dtype), [-1, nb_features])

    def condition(x_in, y_in, domain_in, i_in, cond_in):
        return tf.logical_and(tf.less(i_in, max_iters), cond_in)

    def body(x_in, y_in, domain_in, i_in, cond_in):
        logits = model.get_logits(x_in)
        preds = tf.nn.softmax(logits)
        preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

        list_derivatives = []

        for class_ind in xrange(nb_classes):
            derivatives = tf.gradients(logits[:, class_ind], x_in)
            list_derivatives.append(derivatives[0])

        grads = tf.reshape(tf.stack(list_derivatives), shape=(nb_classes, -1, nb_features))

        if attack == "tjsma":
            grads = tf.reshape(1 - x_in, shape=(-1, nb_features)) * grads

        target_class = tf.reshape(tf.transpose(y_in, perm=(1, 0)), shape=(nb_classes, -1, 1))
        other_classes = tf.cast(tf.not_equal(target_class, 1), tf_dtype)

        grads_target = reduce_sum(grads * target_class, axis=0)

        if attack == "tjsma" or attack == "wjsma":
            preds_transpose = tf.transpose(tf.reshape(preds, shape=(-1, nb_classes, 1)), perm=(1, 0, 2))

            grads_other = reduce_sum(grads * other_classes * preds_transpose, axis=0)
        else:
            grads_other = reduce_sum(grads * other_classes, axis=0)

        increase_coef = (4 * int(increase) - 2) * tf.cast(tf.equal(domain_in, 0), tf_dtype)

        target_tmp = grads_target
        target_tmp -= increase_coef * reduce_max(tf.abs(grads_target), axis=1, keepdims=True)
        target_sum = tf.reshape(target_tmp, shape=(-1, nb_features, 1)) + \
            tf.reshape(target_tmp, shape=(-1, 1, nb_features))

        other_tmp = grads_other
        other_tmp += increase_coef * reduce_max(tf.abs(grads_other), axis=1, keepdims=True)
        other_sum = tf.reshape(other_tmp, shape=(-1, nb_features, 1)) + \
            tf.reshape(other_tmp, shape=(-1, 1, nb_features))

        if increase:
            scores_mask = ((target_sum > 0) & (other_sum < 0))
        else:
            scores_mask = ((target_sum < 0) & (other_sum > 0))

        scores = tf.cast(scores_mask, tf_dtype) * (-target_sum * other_sum) * zero_diagonal

        best = tf.argmax(tf.reshape(scores, shape=(-1, nb_features * nb_features)), axis=1)

        p1 = tf.mod(best, nb_features)
        p2 = tf.floordiv(best, nb_features)
        p1_one_hot = tf.one_hot(p1, depth=nb_features)
        p2_one_hot = tf.one_hot(p2, depth=nb_features)

        mod_not_done = tf.equal(reduce_sum(y_in * preds_onehot, axis=1), 0)
        cond = mod_not_done & (reduce_sum(domain_in, axis=1) >= 2)

        cond_float = tf.reshape(tf.cast(cond, tf_dtype), shape=(-1, 1))
        to_mod = (p1_one_hot + p2_one_hot) * cond_float

        domain_out = domain_in - to_mod

        to_mod_reshape = tf.reshape(to_mod, shape=((-1,) + tuple(x_in.shape[1:])))

        if increase:
            x_out = tf.minimum(clip_max, x_in + to_mod_reshape * theta)
        else:
            x_out = tf.maximum(clip_min, x_in - to_mod_reshape * theta)

        i_out = tf.add(i_in, 1)
        cond_out = reduce_any(cond)

        return x_out, y_in, domain_out, i_out, cond_out

    x_adv, _, _, _, _ = tf.while_loop(
        condition,
        body, [x, y_target, search_domain, 0, True],
        parallel_iterations=1
    )

    return x_adv
