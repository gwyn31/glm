
from numpy import linspace, hstack
from sklearn.linear_model import Ridge

from glm.models.reg import RidgeRegression


if __name__ == '__main__':
    x1 = linspace(1.0, 10.0, num=50).reshape((50, 1))
    x2 = linspace(1.0, 10.0, num=50).reshape((50, 1))
    y = x1 + x2 - 3.0
    x = hstack((x1, x2))

    my_ridge = RidgeRegression(alpha=0.1, solver="sgd", batch_size=10, lr=0.01, tol=1e-2, max_iters=10000)
    my_ridge.train(x, y)
    print(my_ridge.get_params())

    my_ridge = RidgeRegression(alpha=0.1, solver="analytic")
    my_ridge.train(x, y)
    print(my_ridge.get_params())

    ridge = Ridge(alpha=0.1)
    ridge.fit(x, y)
    print(ridge.intercept_)
    print(ridge.coef_)
