"""
Linear classifiers
"""

from numpy import ndarray, hstack, argmax

from glm.base import LogLoss, BaseLM, SGD, add_bias, sigmoid_prob_predict, genr_init_params


class LogisticRegression(BaseLM):

    """
    Logistic regression model for binary classification
    """

    def __init__(self, alpha=0.1, reg_norm=2, solver="sgd", **solver_cfg):
        self.alpha = alpha
        self.reg_norm = reg_norm
        if solver == "sgd":
            self.solver = SGD(**solver_cfg)
        else:
            raise ValueError("`solver` only supports `sgd` or `cd`")
        self.__params = None

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
        self.loss_history = loss_history
        return self

    def get_params(self) -> ndarray:
        return self.__params

    def predict(self, x: ndarray) -> ndarray:
        prob = self.predict_prob(x)
        return argmax(prob, axis=1).reshape((x.shape[0], 1))

    def predict_prob(self, x: ndarray) -> ndarray:
        x_bias = add_bias(x)
        pos_pred = sigmoid_prob_predict(x_bias, self.__params)
        neg_pred = 1 - pos_pred
        return hstack((neg_pred, pos_pred))

