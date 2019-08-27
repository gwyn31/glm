
from numpy import linspace, hstack
from numpy.random import RandomState
from sklearn.linear_model import HuberRegressor

from glm.models.reg import HuberRegression


if __name__ == '__main__':
    rng = RandomState(seed=12)
    x1 = linspace(1.0, 10.0, num=50).reshape((50, 1))
    x2 = linspace(1.0, 10.0, num=50).reshape((50, 1))
    y = x1 + x2 - 3.0 + rng.normal(0.35, 0.5)
    x = hstack((x1, x2))

    my_huber = HuberRegression(eps=1.0, alpha=0.1, batch_size=10, lr=0.1, tol=1e-2, max_iters=10000)
    my_huber.train(x, y)
    print(my_huber.get_params())

    huber = HuberRegressor(epsilon=1.0, alpha=0.1)
    huber.fit(x, y)
    print(huber.intercept_)
    print(huber.coef_)