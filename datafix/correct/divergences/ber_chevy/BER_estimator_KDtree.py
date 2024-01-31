import matplotlib

# matplotlib.use('TKAgg')
import numpy as np
import math
import cvxpy as cvx  # Need to install CVXPY package
import time
from scipy.special import *
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn import svm, datasets
from sklearn import neighbors
from sklearn.datasets import make_classification
from .get_weights import (
    get_Chebyshev_weights,
    get_arithmetic_weights,
    get_uniform_weights,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def ensemble_bg_estimator(
    X,
    Y,
    L=3,
    k0=10,
    h_range=[1.0, 1.5],
    bw_selection="auto",
    ensemble_weights="Chebyshev",
    N_est_all=5000,
    U=0.1,
):
    N, d = X.shape[0], X.shape[1]

    if L <= d + 1:
        intrinsic_dim = L - 2
    else:
        intrinsic_dim = d
    # I = np.dot(I_vec, weights)
    # print('version 112')

    if ensemble_weights == "arithmetic":
        S, W = get_arithmetic_weights(L, intrinsic_dim)
        W = W.reshape(
            L,
        )
        # print(ensemble_weights, 'arithmetic')

    elif ensemble_weights == "Chebyshev":
        S, W = get_Chebyshev_weights(L)
        # print(ensemble_weights)

    elif ensemble_weights == "decaying":
        P = np.arange(L).astype(float)
        W = np.power(2, (-P)) / np.sum(np.power(2, (-P)))

    elif ensemble_weights == "uniform":
        S, W = get_uniform_weights(L)
        # print(ensemble_weights, 'uniform')

    # Scale the nodes
    h = S * (h_range[1] - h_range[0]) + h_range[0]

    # cross-validation for computing bandwidth
    if bw_selection == "auto":
        if X.shape[0] >= 2000 or X.shape[1] >= 10:
            base = NN_bw(X, k0)
        else:
            base = cross_valid_bw(X)
        # bprint(base)
        h_opt = h * base

    elif bw_selection == "KNN":
        base = NN_bw(X, k0=10)
        h_opt = h * base

    elif bw_selection == "manual":
        h_opt = h * np.linalg.norm(np.std(X, axis=0))

    # Find base estimates:
    B = np.array([bg_estimator(X, Y, h_opt[l], N_est_all, U) for l in range(L)])

    # print('Base estimators',B)
    # print('weights',W)
    E = np.dot(W, B)

    return E


def bg_estimator(X, Y, h, N_est_all=5000, U=0.005):
    # U = 5 # Upper bound on the density ratio

    # print(X.shape)
    Y = Y.reshape((-1, 1))
    N = X.shape[0]

    m = max(Y) + 1
    m = int(np.asscalar(m))
    N_est_all = np.min([N_est_all, N])
    N_est = int(N_est_all / m)
    # N_per_class = np.array([len(Y[(Y==i)]) for i in range(m)])
    # print(len(Y[(Y==0)])/N)
    prior = np.array([len(Y[(Y == i)]) / N for i in range(m)])
    # print('prioir', prior)

    D = 0
    for i in range(1, m):
        # print('i',i)
        # print('progress: /10', i)

        # I_class=np.where(Y==i)[0]
        I_class_all = np.where(Y == i)[0]
        I_class = np.random.choice(I_class_all, size=N_est, replace=True)
        X_class = X[I_class, :]

        # the matrix of prior * densities (j, x_i), where the density is computed for x_i only based on the data with label j
        f_mat = np.array(
            [
                prior[r]
                * eps_neighbor_count(
                    X[np.where(Y == r)[0], :], X[np.where(Y == i)[0], :], h=h
                )
                for r in range(i + 1)
            ]
        )
        # print('i ',i,' f_mat.size ', f_mat.shape)
        max_ratio = np.max(f_mat[: f_mat.shape[0] - 1, :], axis=0)
        # print('max_ratio.shape',max_ratio.shape)
        # print('f_mat[:,:5]')
        # print(f_mat[:,:5])
        # print('f_mat ', np.mean(f_mat) )
        # print('max_ratio ', np.mean(max_ratio))
        # print('ratio', max_ratio/f_mat[-1,:])
        # print('density shapes' , max_ratio.shape, f_mat.shape)
        # print('mean max_ratio', np.mean(max_ratio), 'mean f_mat', np.mean(f_mat[-1,:]))
        # print('partial sum', np.mean(np.maximum((1-max_ratio/f_mat[-1,:]),0)),'i ',i)
        D += np.mean(prior[i] * np.maximum((1 - max_ratio / f_mat[-1, :]), 0))
        # print('D',D)

    D += prior[0]
    # print('D',D)
    # print('base estimator ', 1-D)
    return 1 - D


# compute bg density estimates at the points X based on the the points Y
def eps_neighbor_count(X, X_test, h):
    # print('h_opt',h_opt)
    # kde =KernelDensity(kernel='tophat', bandwidth=h).fit(X)
    tree = KDTree(X)
    count = tree.query_radius(X_test, r=h, count_only=True)
    # dnsty = kde.score_samples(X_test)
    # print('dnsty', np.exp(dnsty) )
    # print('density, done')
    return count


##### Quadratic Program for Ensemble Estimation ####
def compute_weights(L, d, T, N):
    # Create optimization variables.
    cvx_eps = cvx.Variable()
    cvx_w = cvx.Variable(L)

    # Create constraints:
    constraints = [cvx.sum(cvx_w) == 1, cvx.pnorm(cvx_w, 2) - cvx_eps / 2 <= 0]
    for i in range(1, L):
        Tp = (1.0 * T / N) ** (1.0 * i / (2 * d))
        cvx_mult = cvx_w.T * Tp
        constraints.append(cvx.sum(cvx_mult) - cvx_eps * 2 <= 0)

    # Form objective.
    obj = cvx.Minimize(cvx_eps)

    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    sol = np.array(cvx_w.value)

    return sol.T


def cross_valid_bw(X):
    std_data = np.linalg.norm(np.std(X, axis=0))
    print("bw_CV")
    grid = GridSearchCV(
        KernelDensity(kernel="tophat"),
        {"bandwidth": np.linspace(0.2 * std_data, 1.5 * std_data, 15)},
        cv=20,
    )  # 20-fold cross-validation
    grid.fit(X)
    h_cv = grid.best_params_["bandwidth"]

    print("done", std_data, h_cv)
    return h_cv


def NN_bw(X, k0):
    nbrs = NearestNeighbors(n_neighbors=k0, algorithm="ball_tree").fit(X)

    if X.shape[0] > 1000:
        X_test = X[:1000]
    else:
        X_test = X

    distances, indices = nbrs.kneighbors(X_test)
    # print('KNN distance', np.mean(distances[:,-1]))
    return np.mean(distances[:, -1])


# Testing:
if __name__ == "__main__":
    N = 2000

    # X, Y = make_classification(n_samples=2*N,n_features=2, n_redundant=0, n_informative=2,
    #                       n_classes=2, class_sep=0,flip_y=0.0, n_clusters_per_class=1)
    d = 2
    D = np.zeros(d)
    D[0] = 0.1  # mean shift of the classes

    X = np.random.randn(2 * N, d)
    X[N + 1 : 2 * N, :] = X[N + 1 : 2 * N, :] + D

    Y = np.zeros(2 * N)
    Y[N + 1 : 2 * N] = 1
    # print(X.shape)
    # print(Y.shape)
    # print(Y)
    e = ensemble_bg_estimator(
        X, Y, L=5, h_range=[1.0, 1.2], ensemble_weights="Chebyshev", N_est_all=5000, U=1
    )
    print(e)

    e = ensemble_bg_estimator(
        X,
        Y,
        L=5,
        h_range=[1.0, 1.2],
        ensemble_weights="arithmetic",
        N_est_all=5000,
        U=1,
    )
    print(e)

    e = ensemble_bg_estimator(
        X, Y, L=5, h_range=[1.0, 1.2], ensemble_weights="uniform", N_est_all=5000, U=1
    )
    print(e)
