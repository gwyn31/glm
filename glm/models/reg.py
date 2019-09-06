"""
Linear regression models
"""

from numpy import array, ndarray, eye
from numpy.linalg import inv, norm

from glm.base import MSE, HuberLoss, BaseLM, SGD, CoordinateDescent, lm_predict, add_bias, genr_init_params


class _CDLasso(CoordinateDescent):

    """
    Coordinate Descent subclass for Lasso regression
    """

    def __init__(self, tol=1e-2, max_iters=1000, alpha=0.1):
        self.alpha = alpha
        super(_CDLasso, self).__init__(tol=tol, max_iters=max_iters)

    def genr_vars_seq(self, x: ndarray, y: ndarray, params: ndarray, loss_obj=None):
        """
        Sequentially optimizing each variable
        """
        m, n = x.shape
        for j in range(n):
            yield j

    def get_single_var_solution(self, j: int, x: ndarray, y: ndarray, params: ndarray):
        m, n = x.shape
        idx = array(range(n)) != j
        y_col = y.reshape((m, 1))
        mj = x[:, j].transpose().dot(y_col - x[:, idx].dot(params[idx]))
        nj = norm(x[:, j], ord=2) ** 2
        determ = mj - m * self.alpha
        if determ > 0:
            return determ / nj
        elif determ < 0:
            return (mj + m * self.alpha) / nj
        else:
            return 0.0


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
        self.__loss_history = []

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
            self.__loss_history = loss_history
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

    @property
    def loss_history(self):
        return self.__loss_history


class LassoRegression(BaseLM):
    """
    Lasso regression(L1-Regularized Linear Regression)
    """

    def __init__(self, alpha=0.1, tol=1e-2, max_iters=1000):
        self.alpha = alpha
        self.__params = None
        self.solver = _CDLasso(tol=tol, max_iters=max_iters, alpha=alpha)
        self.__loss_history = []

    def _check_hyperparameters(self):
        try:
            assert self.alpha > 0
        except AssertionError:
            raise ValueError("Error input `alpha`, must be a positive float")
        self.solver.check_configs()

    def train(self, x: ndarray, y: ndarray):
        """
        Train Lasso regression model by a simple coordinate descent algorithm
        """
        self._check_hyperparameters()
        x_bias = add_bias(x)
        n = x_bias.shape[1]
        init_params = genr_init_params(n)
        mse = MSE(reg_norm=1, penalty=self.alpha)
        params, loss_history = self.solver.optimize(mse, x_bias, y, init_params)
        self.__params = params
        self.__loss_history = loss_history
        return self

    def predict(self, x: ndarray) -> ndarray:
        return lm_predict(x, self.__params)

    def get_params(self):
        return self.__params

    @property
    def n_iters(self):
        return self.solver.n_iters

    @property
    def loss_history(self):
        return self.__loss_history


class HuberRegression(BaseLM):

    def __init__(self, eps=1.35, alpha=0.1, lr=0.01, batch_size=32, max_iters=1000, tol=1e-3):
        self.eps = eps
        self.alpha = alpha
        self.__params = None
        self.solver = SGD(lr=lr, batch_size=batch_size, max_iters=max_iters, tol=tol)
        self.__loss_history = []

    def _check_hyperparameters(self):
        try:
            assert self.alpha > 0 and self.eps > 0
        except AssertionError:
            raise ValueError("Error input, `alpha` and `eps` must be positive floats")
        self.solver.check_configs()

    def train(self, x: ndarray, y: ndarray):
        self._check_hyperparameters()
        x_bias = add_bias(x)
        n = x_bias.shape[1]
        init_params = genr_init_params(n)
        huber = HuberLoss(eps=self.eps, penalty=self.alpha)
        params, loss_history = self.solver.optimize(huber, x_bias, y, init_params)
        self.__params = params
        self.__loss_history = loss_history
        return self

    def predict(self, x: ndarray) -> ndarray:
        return lm_predict(x, self.__params)

    def get_params(self):
        return self.__params

    @property
    def n_iters(self):
        return self.solver.n_iters

