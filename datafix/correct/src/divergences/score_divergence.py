import numpy as np


SCORE_PROB_CLIP_LOW = 0.001
SCORE_PROB_CLIP_HIGH = 0.999
SCORE_DIV_CLIP_LOW = 0
SCORE_DIV_CLIP_HIGH = None


def hard_tahn_to_half(p):
    p = p.copy()
    p[p > 0] = 0.5
    p[p < 0] = -0.5
    return p


def inv_sigmoid(p):
    return np.log(p / (1 - p))


def kl_div(prob, y):
    prob = np.clip(prob, SCORE_PROB_CLIP_LOW, SCORE_PROB_CLIP_HIGH)
    r = np.exp(inv_sigmoid(prob))
    kl = np.clip(
        np.mean(np.log(r[y == 1])) - np.mean(r[y == 0]),
        SCORE_DIV_CLIP_LOW,
        SCORE_DIV_CLIP_HIGH,
    )
    return kl


def skl_div(prob, y):
    kl1 = kl_div(prob, y)
    kl2 = kl_div(1 - prob, 1 - y)
    skl = kl1 + kl2
    return skl


def jsd_div(prob, y):
    prob = np.clip(prob, SCORE_PROB_CLIP_LOW, SCORE_PROB_CLIP_HIGH)
    jd = np.clip(
        0.5
        * (
            np.mean(np.log(prob[y == 1]))
            + np.mean(np.log(1 - prob[y == 0]))
            + np.log(4)
        ),
        SCORE_DIV_CLIP_LOW,
        SCORE_DIV_CLIP_HIGH,
    )
    return jd


def JSD_score(estimator, x, y):
    """
    sklearn scorer to compute JD-Divergence: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    estimator: a binary classifier with .predict_proba
    x: input data
    y: input labels
    """
    prob = estimator.predict_proba(x)[:, 1]
    jd = jsd_div(prob, y)
    return jd


def tvd_div(prob, y):
    prob = np.clip(prob, SCORE_PROB_CLIP_LOW, SCORE_PROB_CLIP_HIGH)
    r = np.exp(inv_sigmoid(prob))
    r_ = hard_tahn_to_half(r - 1)
    tv = np.clip(
        np.mean(r_[y == 1]) - np.mean(r_[y == 0]),
        SCORE_DIV_CLIP_LOW,
        SCORE_DIV_CLIP_HIGH,
    )
    return tv


def TVD_score(estimator, x, y):
    """
    sklearn scorer to compute TV-Divergence: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    estimator: a binary classifier with .predict_proba
    x: input data
    y: input labels
    """
    prob = estimator.predict_proba(x)[:, 1]
    tv = tvd_div(prob, y)
    return tv
