import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

from .score_divergence import tvd_div, jsd_div, skl_div


def div_tv_clf(reference, query, clf=LogisticRegression()):
    x = np.concatenate([reference, query], axis=0)
    y = np.concatenate([np.zeros(reference.shape[0]), np.ones(query.shape[0])], axis=0)
    cv_proba = cross_val_predict(clf, x, y, cv=5, method="predict_proba")
    tv = tvd_div(cv_proba[:, 1], y)
    return tv, cv_proba


def div_all_clf(reference, query, clf=LogisticRegression()):
    x = np.concatenate([reference, query], axis=0)
    y = np.concatenate([np.zeros(reference.shape[0]), np.ones(query.shape[0])], axis=0)
    cv_proba = cross_val_predict(clf, x, y, cv=5, method="predict_proba")
    tv = tvd_div(cv_proba[:, 1], y)
    jsd = jsd_div(cv_proba[:, 1], y)
    jsd_inv = jsd_div(cv_proba[:, 0], y)
    skl = skl_div(cv_proba[:, 1], y)
    return tv, jsd, jsd_inv, skl, cv_proba