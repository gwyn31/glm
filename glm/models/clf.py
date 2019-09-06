"""
Linear classifiers
"""

from numpy import ndarray, hstack, argmax

from glm.base import LogLoss, BaseLM, SGD, CoordinateDescent, add_bias, sigmoid_prob_predict, genr_init_params


class _CDLogistic(CoordinateDescent):

    def genr_vars_seq(self, x: ndarray, y: ndarray, params: ndarray, loss_obj=None):
        pass

    def get_single_var_solution(self, j: int, x: ndarray, y: ndarray, params: ndarray):
        pass


class LogisticRegression(BaseLM):

    """
    Logistic regression model for binary classification
    """

    def __init__(self, alpha=0.1, reg_norm=2, lr=0.01, batch_size=32, max_iters=1000, tol=1e-3):
        self.alpha = alpha
        self.reg_norm = reg_norm
        if reg_norm == 2:
            self.solver = SGD(lr=lr, batch_size=batch_size, max_iters=max_iters, tol=tol)
        else:
            self.solver = _CDLogistic(tol=tol, max_iters=max_iters)
        self.__params = None
        self.__loss_history = []

    def _check_hyperparameters(self):
        try:
            assert self.alpha > 0
        except AssertionError:
            raise ValueError("Error input `alpha`, must be a positive float")
        self.solver.check_configs()

    def train(self, x: ndarray, y: ndarray):
        self._check_hyperparameters()
        x_bias = add_bias(x)
        n = x_bias.shape[1]
        init_params = genr_init_params(n)
        logloss = LogLoss(reg_norm=self.reg_norm, penalty=self.alpha)
        params, loss_history = self.solver.optimize(logloss, x_bias, y, init_params)
        self.__params = params
        self.__loss_history = loss_history
        return self

    def predict(self, x: ndarray) -> ndarray:
        prob = self.predict_prob(x)
        return argmax(prob, axis=1).reshape((x.shape[0], 1))

    def predict_prob(self, x: ndarray) -> ndarray:
        x_bias = add_bias(x)
        pos_pred = sigmoid_prob_predict(x_bias, self.__params)
        neg_pred = 1 - pos_pred
        return hstack((neg_pred, pos_pred))

    def get_params(self) -> ndarray:
        return self.__params

    @property
    def n_iters(self):
        return self.solver.n_iters

    @property
    def loss_history(self):
        return self.__loss_history

