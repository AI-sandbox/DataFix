import numpy as np
from .BER_estimator_KDtree import ensemble_bg_estimator as BER


def ber_chevy_estimator(reference, query):
    x = np.concatenate([reference, query], axis=0)
    y = np.concatenate([np.zeros(reference.shape[0]), np.ones(query.shape[0])], axis=0)
    ber = BER(x, y)
    return ber
