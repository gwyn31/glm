"""
Common functions and classes for all linear models
"""

from abc import ABCMeta, abstractmethod

from numpy import ndarray, ones, hstack, exp
from numpy.random import RandomState


def add_bias(x: ndarray) -> ndarray:
    """
    Add bias term into the training feature matrix
    :param x: feature matrix, should have shape: [n_samples, n_features]
    :return: feature matrix with bias term at first column
    """
    m = x.shape[0]
    o = ones(shape=(m, 1), dtype=float)
    return hstack((o, x))


def lm_predict(x: ndarray, b_w: ndarray) -> ndarray:
    """
    Linear model's prediction: y_^hat = <w, x> + b
    :param x: feature matrix, should have shape: [n_samples, n_features]
    :param b_w: model parameters, the first element should be the intercept
    :return: predicted values array
    """
    x_with_bias = add_bias(x)
    return x_with_bias.dot(b_w)


def sigmoid_prob_predict(x: ndarray, b_w: ndarray) -> ndarray:
    """
    Sigmoid probability predictions
    :param x: feature matrix, first column must be all 1.0
    :param b_w: model parameters, the first element should be the intercept
    :return: predicted probabilities of the positive class
    """
    return 1 / (1 + exp(-x.dot(b_w)))


def genr_init_params(n: int) -> ndarray:
    """
    Randomly initialize the linear model's parameters
    :param n: dimension
    :return: the initialized model parameters array
    """
    rng = RandomState()
    init_params = rng.normal(loc=0.0, scale=1.0, size=(n, 1))
    init_params[0] = 1.0
    return init_params


class BaseLM(metaclass=ABCMeta):

    """
    Super class for all linear models
    """

    @abstractmethod
    def _check_hyperparameters(self):
        """
        Check model's hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, x: ndarray, y: ndarray):
        """
        Train the generalized linear model
        :param x: feature matrix, should have shape: [n_samples, n_features]
        :param y: target array, should have equal number of samples of `x`
        :return: self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: ndarray) -> ndarray:
        """
        Predict the target
        :param x: feature matrix, should have shape: [n_samples, n_features]
        :return: the predicted array
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> ndarray:
        """
        Get model's parameters
        :return: model's parameter array, the first element is the intercept
        """
        raise NotImplementedError

