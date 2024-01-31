import numpy as np
import math
import cvxpy as cvx  # Need to install CVXPY package,
from scipy.special import *


def get_Chebyshev_weights(L):
    d = L - 2
    C = np.zeros(d)

    I = np.arange(0, L)
    # compute Chebyshev nodes
    S = 0.5 * np.cos(math.pi / L * (I + 0.5)) + 0.5
    # print(S)
    # simplified form of w_i:
    W_mat = np.zeros((L, d + 1))
    for k in range(d + 1):
        W_mat[:, k] = 2 / L * Ts(k, 0) * Ts(k, S)
    Wi = np.sum(W_mat, axis=1) - 1 / L
    return np.flip(S, axis=0), np.flip(Wi, axis=0)


# print(Wi)


def Ts(i, x):
    C = np.zeros(i + 1)
    C[-1] = 1
    return np.polynomial.chebyshev.chebval(2 * x - 1, C)


def get_arithmetic_weights(L, d):
    # Create optimization variables.
    cvx_eps = cvx.Variable()
    cvx_w = cvx.Variable(L)

    # Create constraints:
    constraints = [cvx.sum(cvx_w) == 1, cvx.pnorm(cvx_w, 2) - cvx_eps / 2 <= 0]
    for i in range(1, L):
        Tp = (1.0 * np.arange(1, L + 1)) ** (1.0 * i / d)
        cvx_mult = cvx.multiply(cvx_w.T, Tp)
        constraints.append(cvx.sum(cvx_mult) - cvx_eps * 2 <= 0)

    # Form objective.
    obj = cvx.Minimize(cvx_eps)

    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    sol = np.array(cvx_w.value)

    # Find points
    S = np.arange(1, L + 1) / L
    return S, sol.T


# Uniform non-optimal Weights
def get_uniform_weights(L):
    # Find points
    S = np.arange(1, L + 1) / L
    W = np.ones(L) / L
    return S, W


# Testing:
if __name__ == "__main__":
    # a = get_Chebyshev_weights(10)
    # print(a)

    a = get_Chebyshev_weights(10)
    print(a)

    a = get_arithmetic_weights(10, 9)
    print(a)

    a = get_uniform_weights(10)
    print(a)
