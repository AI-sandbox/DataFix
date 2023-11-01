import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.kernel_approximation import Nystroem
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
import time
from sklearn import pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier

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


def div_multiple_clf(reference, query, clf_list=None, verbose=False):
    if clf_list is None:
        clf_list = [
            KNeighborsClassifier(),
            LogisticRegression(),
            QuadraticDiscriminantAnalysis(),
            pipeline.Pipeline(
                [
                    ("feature_map", Nystroem(n_components=reference.shape[1])),
                    ("svm", LogisticRegression()),
                ]
            ),
            MLPClassifier(),
            MLPClassifier(
                hidden_layer_sizes=reference.shape[1] * 3
            ),  # MLPClassifier(hidden_layer_sizes=(1024,1024,)),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            HistGradientBoostingClassifier(),
            LGBMClassifier(),
            XGBClassifier(),
            CatBoostClassifier(verbose=False),
        ]

    tv_list = []
    for clf in clf_list:
        tv, _ = div_tv_clf(reference, query, clf=clf)
        tv_list.append(tv)
        if verbose:
            print(clf, tv)
    tv_list = np.array(tv_list)
    return tv_list, np.mean(tv_list), np.max(tv_list)


def div_all_multiple_clf(reference, query, clf_list=None, verbose=False):
    if clf_list is None:
        # Fast
        clf_list = [
            KNeighborsClassifier(),
            LogisticRegression(),
            QuadraticDiscriminantAnalysis(),
            pipeline.Pipeline(
                [
                    ("feature_map", Nystroem(n_components=reference.shape[1])),
                    ("svm", LogisticRegression()),
                ]
            ),
        ]

    metrics_dict = {}
    (
        metrics_dict["jsd"],
        metrics_dict["jsd_inv"],
        metrics_dict["tv"],
        metrics_dict["skl"],
    ) = ([], [], [], [])
    for clf in clf_list:
        tic = time.time()
        tv, jsd, jsd_inv, skl, _ = div_all_clf(reference, query, clf=clf)
        toc = time.time()
        metrics_dict["tv"].append(tv)
        metrics_dict["jsd"].append(jsd)
        metrics_dict["jsd_inv"].append(jsd_inv)
        metrics_dict["skl"].append(skl)
        if verbose:
            print(clf, tv, jsd, jsd_inv, skl, toc - tic)
    keys = list(metrics_dict.keys())
    for k in keys:
        metrics_dict[k] = np.array(metrics_dict[k])
        metrics_dict[k + "_mean"] = np.mean(metrics_dict[k])
        metrics_dict[k + "_max"] = np.max(metrics_dict[k])
        metrics_dict[k + "_mean_no_cat"] = np.mean(metrics_dict[k][0:-2])
        metrics_dict[k + "_max_no_cat"] = np.max(metrics_dict[k][0:-2])
    return metrics_dict
