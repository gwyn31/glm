"""
Linear regression models
"""

from numpy import ndarray, eye
from numpy.linalg import inv

from glm.base import MSE, BaseLM, SGD, lm_predict, add_bias, genr_init_params


class RidgeRegression(BaseLM):

    """
    Ridge regression(L2-Regularized Linear Regression)
    """

    def __init__(self, alpha=0.1, solver="analytic", lr=0.01, batch_size=32, max_iters=1000, tol=1e-3):
        self.alpha = alpha
        self.__params = None
        if solver == "analytic":
            self.solver = solver
        elif solver == "sgd":
            self.solver = SGD(lr=lr, batch_size=batch_size, max_iters=max_iters, tol=tol)
        else:
            raise ValueError("`solver` only supports `analytic` or `sgd`")
        self.loss_history = []

    def _check_hyperparameters(self):
        try:
            assert self.alpha > 0
        except AssertionError:
            raise ValueError("Error input `alpha`, must be a positive float")
        if hasattr(self.solver, "_check_configs"):
            self.solver.check_configs()

    def train(self, x: ndarray, y: ndarray):
        self._check_hyperparameters()
        x_bias = add_bias(x)
        n = x_bias.shape[1]
        if self.solver == "analytic":
            m, n = x_bias.shape
            inv_mat = inv(x_bias.transpose().dot(x_bias) + self.alpha * eye(n, dtype=float))
            self.__params = inv_mat.dot(x_bias.transpose().dot(y))
        else:
            init_params = genr_init_params(n)
            mse = MSE(reg_norm=2, penalty=self.alpha)
            params, loss_history = self.solver.optimize(mse, x_bias, y, init_params)
            self.__params = params
            self.loss_history = loss_history
        return self

    def predict(self, x: ndarray) -> ndarray:
        return lm_predict(x, self.__params)

    def get_params(self):
        return self.__params

    @property
    def n_iters(self):
        if hasattr(self.solver, "n_iters"):
            return self.solver.n_iters
        else:
            return 0

