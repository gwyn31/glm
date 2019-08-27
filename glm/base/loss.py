"""
Loss functions for linear models
"""

from abc import ABCMeta, abstractmethod

from numpy import ndarray, zeros, log, argwhere, sign, abs as np_abs, sum as np_sum
from numpy.linalg import norm

from glm.base.common import sigmoid_prob_predict


def _regularize(param: ndarray, reg_norm: int, penalty: float):
    if 0 < reg_norm <= 2:
        reg_term = (penalty / reg_norm) * ((norm(param[1:], ord=reg_norm)) ** 2)
    else:
        reg_term = 0.0
    return reg_term


def _reg_grad(x: ndarray, param: ndarray, reg_norm: int, penalty: float) -> ndarray:
    param_reg = param.copy()
    param_reg[0] = 0.0
    n = x.shape[1]
    if reg_norm == 1:
        raise ValueError("L1 regularization doesn't support gradient")
    elif reg_norm == 0:
        reg_grad = zeros((n, 1), dtype=float)
    else:
        reg_grad = penalty * param_reg
    return reg_grad


class LossFunction(metaclass=ABCMeta):

    @abstractmethod
    def get_loss(self, x: ndarray, y: ndarray, param: ndarray) -> float:
        """
        Compute loss on features `x` and target `y` with linear model parameter `param`
        :param x: feature matrix, first column must be all 1.0
        :param y: target array, should have equal length with `x`
        :param param: model parameter
        :return: loss value
        """
        raise NotImplementedError

    @abstractmethod
    def get_grad(self, x: ndarray, y: ndarray, param: ndarray) -> ndarray:
        """
        Compute the gradient of parameters on features `x` and target `y` with linear model parameter `param`,

            Even some loss function doesn't have gradient, subclass should also implement this method
        by raising an AttributeError
        :param x: feature matrix, first column must be all 1.0
        :param y: target array, should have equal length with `x`
        :param param: linear model's parameters, the first element must be the intercept
        :return: corresponding gradient of the parameters
        """
        raise NotImplementedError


class MSE(LossFunction):

    """
    Mean Squared Error regression loss
    """

    def __init__(self, reg_norm=2, penalty=0.01):
        """
        Constructor
        :param reg_norm: regularization norm, can only be 0, 1 or 2
        :param penalty: regularization coefficient, must > 0
        """
        self.reg_norm = reg_norm
        self.penalty = penalty

    def get_loss(self, x: ndarray, y: ndarray, param: ndarray) -> float:
        m = len(y)
        pred = x.dot(param)
        resid = pred.ravel() - y.ravel()
        loss_val = (1 / (2 * m)) * (norm(resid, ord=2)) ** 2
        reg_term = _regularize(param, self.reg_norm, self.penalty)
        return loss_val + reg_term

    def get_grad(self, x: ndarray, y: ndarray, param: ndarray) -> ndarray:
        reg_grad = _reg_grad(x, param, self.reg_norm, self.penalty)
        pred = x.dot(param)
        m = x.shape[0]
        resid = pred.ravel() - y.ravel()
        loss_grad = (1 / m) * x.transpose().dot(resid.reshape((m, 1)))
        return loss_grad + reg_grad


class HuberLoss(LossFunction):

    def __init__(self, eps=1.35, penalty=0.1):
        self.eps = eps
        self.penalty = penalty

    def _get_indices(self, pred: ndarray, y: ndarray) -> (ndarray, ndarray, ndarray):
        abs_resid = np_abs(pred.ravel() - y.ravel())
        sq_idx = argwhere(abs_resid <= self.eps)
        l_idx = argwhere(abs_resid > self.eps)
        return abs_resid, sq_idx.ravel(), l_idx.ravel()

    def get_loss(self, x: ndarray, y: ndarray, param: ndarray) -> float:
        m = len(y)
        pred = x.dot(param)
        abs_resid, sq_idx, l_idx = self._get_indices(pred, y)
        loss_val = 0.5 * np_sum(abs_resid[sq_idx] ** 2) + self.eps * np_sum((abs_resid[l_idx] - 0.5 * self.eps))
        return loss_val / m

    def get_grad(self, x: ndarray, y: ndarray, param: ndarray) -> ndarray:
        m = len(y)
        pred = x.dot(param)
        y_col = y.reshape((m, 1))
        resid = pred - y_col
        reg_grad = _reg_grad(x, param, 2, self.penalty)
        abs_resid, sq_idx, l_idx = self._get_indices(pred, y)
        loss_grad = (1 / m) * ( x[sq_idx, :].transpose().dot(resid[sq_idx]) +
                                self.eps * x[l_idx, :].transpose().dot(sign(resid[l_idx])) )
        return loss_grad + reg_grad


class LogLoss(LossFunction):

    """
    Log-likelihood classification loss
    """

    def __init__(self, reg_norm=2, penalty=0.01):
        self.reg_norm = reg_norm
        self.penalty = penalty

    def get_loss(self, x: ndarray, y: ndarray, param: ndarray) -> float:
        m = len(y)
        pred = sigmoid_prob_predict(x, param)
        y_col = y.reshape((m, 1))
        temp = 1 - y_col
        reg_term = _regularize(param, self.reg_norm, self.penalty)
        loss_val = (-1 / m) * (y_col.transpose().dot(log(pred)) + temp.transpose().dot(log(1 - pred)))
        return loss_val + reg_term

    def get_grad(self, x: ndarray, y: ndarray, param: ndarray) -> ndarray:
        m = len(y)
        pred = sigmoid_prob_predict(x, param)
        y_col = y.reshape((m, 1))
        loss_grad = (1 / m) * x.transpose().dot(pred - y_col)
        reg_grad = _regularize(param, self.reg_norm, self.penalty)
        return loss_grad + reg_grad

