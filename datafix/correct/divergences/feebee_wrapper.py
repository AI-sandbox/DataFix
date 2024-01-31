import numpy as np
from absl import flags

# FLAGS = flags.FLAGS
# FLAGS.ghp_approx = 1
# flags.FLAGS.ghp_approx = 1

from .feebee.methods.ghp import eval_from_matrix as eval_from_matrix_ghp
from .feebee.methods.kde import eval_from_matrix_kde
from .feebee.methods.knn import eval_from_matrices as eval_from_matrices_knn


def ghp_ber(reference, query, verbose=False):
    if verbose:
        print("running ghp")

    x = np.concatenate([reference, query], axis=0)
    y = np.concatenate([np.zeros(reference.shape[0]), np.ones(query.shape[0])], axis=0)

    pred = eval_from_matrix_ghp(x, y)
    if verbose:
        print(pred)

    return pred["default"][0], pred["default"][1]


def knn_ber(reference, query, verbose=False):
    def _knn_ber(train_x, test_x, train_y, test_y, verbose=False):
        pred = eval_from_matrices_knn(train_x, test_x, train_y, test_y)
        if verbose:
            print(pred)

        pred_list = []
        for k in pred.keys():
            pred_list.append(np.array(pred[k]))
        pred_list = np.mean(np.stack(pred_list), axis=0)
        pred["default"] = pred_list

        if verbose:
            print("running knn")
        return pred["default"]

    x = np.concatenate([reference, query], axis=0)
    y = np.concatenate([np.zeros(reference.shape[0]), np.ones(query.shape[0])], axis=0)

    pred1 = _knn_ber(x[0::2, :], x[1::2, :], y[0::2], y[1::2], verbose=verbose)
    pred2 = _knn_ber(x[1::2, :], x[0::2, :], y[1::2], y[0::2], verbose=verbose)

    pred = {}
    pred["default"] = [0.5 * (pred1[0] + pred2[0]), 0.5 * (pred1[1] + pred2[1])]

    return pred["default"][0], pred["default"][1]


def kde_ber(reference, query, verbose=False):
    # Not working well
    if verbose:
        print("running kde")

    x = np.concatenate([reference, query], axis=0)
    y = np.concatenate([np.zeros(reference.shape[0]), np.ones(query.shape[0])], axis=0)

    pred = eval_from_matrix_kde(x, y)
    if verbose:
        print(pred)

    return pred["default"][0], pred["default"][1]
