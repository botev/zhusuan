#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from numpy import inf

from zhusuan.distributions.base import Distribution
from zhusuan.distributions.utils import assert_same_float_and_int_dtype, \
    maybe_explicit_broadcast


__all__ = [
    'Empirical',
    'Implicit',
    'FlowDistribution'
]


class Empirical(Distribution):
    """
    The class of Empirical distribution.
    See :class:`~zhusuan.distributions.base.Empirical` for details.

    :param shape: A list or tuple describing the `batch_shape` of the distribution.
        The entries of the list can either be int, Dimension or Tensor.
    :param dtype: The value type of samples from the distribution.
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    :param is_continuous: Whether the distribution is continous or not.
        If None will consider it continuous only if `dtype` is a float type.
    """

    def __init__(self, shape, dtype,
                 group_ndims=0,
                 is_continuous=None,
                 **kwargs):
        try:
            self.static_shape = tf.TensorShape(shape[0])
        except Exception:
            self.static_shape = tf.TensorShape(tf.Dimension(None))
        for s in shape[1:]:
            try:
                self.static_shape = self.static_shape.concatenate(tf.TensorShape(s))
            except Exception:
                self.static_shape = self.static_shape.concatenate(tf.Dimension(None))

        if dtype is None:
            dtype = tf.float32
        assert_same_float_and_int_dtype([], dtype)

        if is_continuous is None:
            is_continuous = dtype.is_floating

        super(Empirical, self).__init__(
            dtype=dtype,
            param_dtype=None,
            is_continuous=is_continuous,
            is_reparameterized=False,
            group_ndims=group_ndims,
            **kwargs)

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        raise ValueError("The Empirical distribution has no dynamic batch shape.")

    def _get_batch_shape(self):
        return self.static_shape

    def _sample(self, n_samples):
        raise ValueError("You can not sample from an Empirical distribution.")

    def _log_prob(self, given):
        raise ValueError("An empirical distribution has no log-probability measure.")

    def _prob(self, given):
        raise ValueError("An empirical distribution has no probability measure.")


class Implicit(Distribution):
    """
    The class of Implicit distribution.
    See :class:`~zhusuan.distributions.base.Implicit` for details.
    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param implicit: A N-D (N >= 1) `float` Tensor
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    """

    def __init__(self,
                 implicit,
                 group_ndims=0,
                 **kwargs):
        self.implicit = implicit
        super(Implicit, self).__init__(
            dtype=implicit.dtype,
            param_dtype=implicit.dtype,
            is_continuous=implicit.dtype.is_floating,
            group_ndims=group_ndims,
            is_reparameterized=False,
            **kwargs)

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.implicit)

    def _get_batch_shape(self):
        return self.implicit.get_shape()

    def _sample(self, n_samples):
        implicit = tf.expand_dims(self.implicit, 0)
        return tf.tile(implicit, [n_samples] + [1] * self.implicit.shape.ndims)

    def _log_prob(self, given):
        return tf.log(self.prob(given))

    def _prob(self, given):
        given = tf.cast(given, self.param_dtype)
        given, implicit = maybe_explicit_broadcast(given, self.implicit, 'given', 'implicit')
        prob = tf.cast(tf.equal(given, implicit), tf.float32)
        if self.is_continuous:
            return (2 * prob - 1) * inf
        else:
            return prob


class FlowDistribution(Distribution):
    """
    The class of Implicit distribution.
    See :class:`~zhusuan.distributions.base.Implicit` for details.
    :param name: A string. The name of the `StochasticTensor`. Must be unique
        in the `BayesianNet` context.
    :param implicit: A N-D (N >= 1) `float` Tensor
    :param group_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`~zhusuan.distributions.base.Distribution` for more detailed
        explanation.
    """

    def __init__(self,
                 base,
                 forward,
                 inverse=None,
                 group_ndims=0,
                 **kwargs):
        self.base = base
        self.forward = forward
        self.inverse = inverse
        super(FlowDistribution, self).__init__(
            dtype=base.dtype,
            param_dtype=base.dtype,
            is_continuous=base.dtype.is_floating,
            group_ndims=group_ndims,
            is_reparameterized=False,
            **kwargs)

    def _value_shape(self):
        return self.base.value_shape()

    def _get_value_shape(self):
        return self.base.get_value_shape()

    def _batch_shape(self):
        return self.base.batch_shape()

    def _get_batch_shape(self):
        return self.base.get_batch_shape()

    def _sample(self, n_samples):
        raise ValueError("FlowDistribution can only sample through `sample_and_log_prob`.")

    def _log_prob(self, given):
        if self.inverse is None:
            raise ValueError("FlowDistribution can only calculate log_prob through `sample_and_log_prob`, "
                             "when inverse=None.")
        else:
            given_base, log_det_j = self.inverse(given)
            log_prob = self.base.log_prob(given)
            return log_prob + log_det_j

    def _prob(self, given):
        return tf.exp(self.log_prob(given))

    def sample_and_log_prob(self, n_samples=None):
        samples, log_prob = self.base.sample_and_log_prob(n_samples)
        samples, log_det_j = self.forward(samples, log_prob)
        return samples, log_prob - log_det_j
