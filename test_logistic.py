
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from glm import LogisticRegression as MyLog


if __name__ == '__main__':
    x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=12)
    clf = LogisticRegression(penalty='l2', C=0.1, solver="sag")
    clf.fit(x, y)
    print(clf.intercept_)
    print(clf.coef_)
    print(roc_auc_score(y, clf.predict_proba(x)[:, -1]))

    clf = MyLog(alpha=0.1, reg_norm=2, solver="sgd", lr=0.1, batch_size=10, max_iters=15000, tol=1e-2)
    clf.train(x, y)
    print(clf.get_params())
    print(roc_auc_score(y, clf.predict_prob(x)[:, -1]))
