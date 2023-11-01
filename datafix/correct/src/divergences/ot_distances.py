import ot
import numpy as np
import torch
import time


def ot_dist(reference, query):
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="numItermax reached before optimality. Try to increase numItermax.",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swd = ot.sliced_wasserstein_distance(
            torch.tensor(reference), torch.tensor(query), n_projections=200
        )

        size_batch = reference.shape[0]
        ab = torch.ones(size_batch) / size_batch

        M = ot.dist(torch.tensor(reference), torch.tensor(query))
        w2 = ot.emd2(ab, ab, M, numItermax=100000 * 2)
    return swd.cpu().numpy(), w2.cpu().numpy()
