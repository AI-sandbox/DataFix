################################################################################
# Functions to evaluate the discriminator.
################################################################################

import numpy as np

from joblib import Parallel, delayed
from numpy import ndarray
from sklearn.metrics import get_scorer
from typing import Dict, List, Union

SCORE_PROB_CLIP_LOW = 0.0001
SCORE_PROB_CLIP_HIGH = 0.9999
SCORE_DIV_CLIP_LOW = 0
SCORE_DIV_CLIP_HIGH = None


def hard_tahn_to_half(p):
    """
    Convert values in array p to half by applying a hard tanh 
    activation function.
    """
    p = p.copy()
    p[p > 0] = 0.5
    p[p < 0] = -0.5
    return p


def inv_sigmoid(p):
    """
    Apply inverse sigmoid.
    """
    return np.log(p / (1 - p))


def JD_score(y_true, y_prob):
    """
    Scorer to compute JD-Divergence: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    Parameters
    ----------
    y_true : ndarray
        Array of true labels.
    y_prob: ndarray
        Array of probabilities made by the discriminator model.

    Returns
    -------
    jd : float
        JD-Divergence score.
    """
    prob = np.clip(y_prob[:, 1], SCORE_PROB_CLIP_LOW, SCORE_PROB_CLIP_HIGH)
    jd = np.clip(
        0.5
        * (
            np.mean(np.log(prob[y_true == 1]))
            + np.mean(np.log(1 - prob[y_true == 0]))
            + np.log(4)
        ),
        SCORE_DIV_CLIP_LOW,
        SCORE_DIV_CLIP_HIGH,
    )
    return jd


def KL_score(y_true, y_prob):
    """
    sklearn scorer to compute KL-Divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Parameters
    ----------
    y_true : ndarray
        Array of true labels.
    y_prob: ndarray
        Array of probabilities made by the discriminator model.

    Returns
    -------
    kl : float
        KL-Divergence score.
    """
    prob = np.clip(y_prob[:, 1], SCORE_PROB_CLIP_LOW, SCORE_PROB_CLIP_HIGH)
    r = np.exp(inv_sigmoid(prob))
    kl = np.clip(
        np.mean(np.log(r[y_true == 1])) - np.mean(r[y_true == 0] - 1),
        SCORE_DIV_CLIP_LOW,
        SCORE_DIV_CLIP_HIGH,
    )
    return kl


def TV_score(y_true, y_prob):
    """
    sklearn scorer to compute TV-Divergence: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures

    Parameters
    ----------
    y_true : ndarray
        Array of true labels.
    y_prob: ndarray
        Array of probabilities made by the discriminator model.

    Returns
    -------
    tv : float
        TV-Divergence score.
    """
    prob = np.clip(y_prob[:, 1], SCORE_PROB_CLIP_LOW, SCORE_PROB_CLIP_HIGH)
    r = np.exp(inv_sigmoid(prob))
    r_ = hard_tahn_to_half(r - 1)
    tv = np.clip(
        np.mean(r_[y_true == 1]) - np.mean(r_[y_true == 0]),
        SCORE_DIV_CLIP_LOW,
        SCORE_DIV_CLIP_HIGH,
    )
    return tv


def _single_evaluation_discriminator(
    y_true: ndarray, y_pred: ndarray, y_prob: Union[ndarray, None], metric: str
):
    """
    Evaluate the discriminator performance using a single metric.

    Parameters
    ----------
    y_true : ndarray
        Array of true labels.
    y_pred : ndarray
        Array of predictions made by the discriminator model.
    y_prob : None or ndarray
        Array of probabilities made by the discriminator model.
    metric : str
        Metric to be used for evaluation.
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.

    Returns
    -------
    score : float
        Score for the specified metric.
    """
    # Compute the mean of the y_prob for each class
    if metric in ["D1", "D2", "D3"]:
        p_0 = y_prob[:, 0].mean()
        p_1 = y_prob[:, 1].mean()

    if metric == "D1":
        return np.log(p_1) + np.log(1 - p_0)
    elif metric == "D2":
        return np.log(p_1 / p_0)
    elif metric == "D3":
        return -np.log((1 / p_0) - 1)
    elif metric == "JD":
        return JD_score(y_true, y_prob)
    elif metric == "KL":
        return KL_score(y_true, y_prob)
    elif metric == "TV":
        return TV_score(y_true, y_prob)
    else:
        scorer = get_scorer(metric)
        return scorer._score_func(y_true, y_pred)


def _evaluate_discriminator(
    y_true: ndarray,
    y_pred: ndarray,
    y_prob: Union[ndarray, None],
    scoring: List[str],
    n_jobs,
):
    """
    Evaluate discriminator model performance using specified metrics.

    Parameters
    ----------
    y_true : ndarray
        Array of true labels.
    y_pred : ndarray
        Array of predictions made by the discriminator model.
    y_prob : None or ndarray
        Array of probabilities made by the discriminator model.
    scoring : list of str
        List of metrics to be used for evaluation.
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.

    Returns
    -------
    scores : dict
        Dictionary containing the score for each specified metric.
    """
    # Perform parallelized evaluation on each metric
    score_values = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_single_evaluation_discriminator)(y_true, y_pred, y_prob, metric)
        for metric in scoring
    )

    # Define dictionary containing the computed score for each specified metric
    # The keys are the scoring names, the values the score values
    scores = dict(zip(scoring, score_values))

    return scores
