"""
Implement some practical optimization algorithms
"""

from abc import ABCMeta, abstractmethod

from numpy import ndarray, array
from numpy.random import RandomState

from glm.base.loss import LossFunction


class IterativeOptimizer(metaclass=ABCMeta):

    """
    Super class for all iterative optimization algorithms
    """

    @abstractmethod
    def check_configs(self):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, loss_obj: LossFunction, x: ndarray, y: ndarray, init_param: ndarray) -> (ndarray, ndarray):
        """
        The optimization process
        :param loss_obj: loss function object
        :param x: training samples, should have a shape of [n_samples, n_features]
        :param y: targets, should have a length of `x.shape[0]`
        :param init_param: initialized model parameters
        :return: optimized parameters, loss history
        """
        raise NotImplementedError


class SGD(IterativeOptimizer):

    """
    Mini-batch gradient descent algorithm.
    In practice sometimes also referring to `stochastic gradient descent`.
    """

    def __init__(self, lr=0.01, batch_size=32, max_iters=1000, tol=1e-3):
        """
        Constructor
        :param lr: float, learning rate
        :param batch_size: int, mini batch's size
        :param max_iters: int, max number of iteration
        :param tol: float, stopping criterion
        """
        self.lr = lr
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.tol = tol
        self.n_iters = 0  # actual number of iterations

    def check_configs(self):
        for k, v in self.__dict__.items():
            if k != "n_iters" and v <= 0:
                raise ValueError("Attribute {0} should be positive, input: {1}".format(k, v))

    def _get_shuffled_mini_batches(self, m: int):
        """
        Get mini batches of indices
        :param m: total number of samples
        """
        rng = RandomState()
        shuffled_indices = rng.permutation(range(m))
        for b in range(0, m, self.batch_size):
            yield shuffled_indices[b:b + self.batch_size]

    def optimize(self, loss_obj: LossFunction, x: ndarray, y: ndarray, init_param: ndarray) -> (ndarray, ndarray):
        """
        Converge when loss < tolerance or reaching the max number of iterations
        :param loss_obj:
        :param x: feature matrix, first column must be all 1.0
        :param y: target array, should have equal length with `x`
        :param init_param:
        :return: trained parameters, loss function history
        """
        loss_history = []
        m = x.shape[0]
        loss = loss_obj.get_loss(x, y, init_param)
        param = init_param.copy()
        n_iters = 0
        while loss > self.tol and n_iters < self.max_iters:  # repeat until convergence
            for batch in self._get_shuffled_mini_batches(m):  # traverse all batches
                x_batch, y_batch = x[batch, :], y[batch]
                # compute gradient
                grad = loss_obj.get_grad(x_batch, y_batch, param)
                # update the parameters
                param = param - self.lr * grad
                # compute loss on the whole training set
                loss = loss_obj.get_loss(x, y, param)
                loss_history.append(loss)
                n_iters += 1
        self.n_iters = n_iters
        return param, array(loss_history)


class CoordinateDescent(IterativeOptimizer):

    def __init__(self, max_iters=10000): pass

    def check_configs(self): pass

