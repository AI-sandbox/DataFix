################################################################################
# Functions to smooth the curve with the balanced accuracy vs the number
# of corrupted features and find the knee from the smoothed curve.
################################################################################

import numpy as np
import warnings

from kneed import KneeLocator
from numpy import ndarray
from scipy.interpolate import UnivariateSpline
from typing import List


def _interpolate(x: List[int], y: List[float]):
    """
    Given two lists `x` and `y` with sorted integer values in increasing order,
    this function performs interpolation so that `x` contains all values
    ranging from 0 to its maximum value and `y` contains their corresponding
    interpolated values.

    Parameters
    ----------
    x : list of int
        Sorted integer values in increasing order to be interpolated in the
        x-axis.
    y : list of float
        Values corresponding to `x` to be interpolated in the y-axis.

    Returns
    -------
    x_inter : ndarray
        Interpolated values for the x-axis.
    y_inter : ndarray
        Interpolated values for the y-axis corresponding to `x_inter`.
    """
    if len(x) == 1:
        # If x has only one value, return x and y as is
        x_inter = x
        y_inter = y
    else:
        if len(x) < 4:
            # In case the length of x is less than 4, set k to 1
            k = 1

        # Fit a spline model to the data points (x, y)
        f = UnivariateSpline(x, y, s=0, k=1)

        # Obtain x values after interpolation
        x_inter = range(x[-1] + 1)

        # Obtain y values for the interpolated x values
        y_inter = np.array(f(x_inter))

    return x_inter, y_inter


def _opening_left_right(y: ndarray):
    """
    Ensure that each point in the right side of the input array is less than or
    equal to each point in the left side.

    Parameters
    ----------
    y : ndarray
        Array to apply the opening operation on.

    Returns
    -------
    x : ndarray
        Array after opening operation.
    """
    # Convert the input array to a 1-D list
    y = np.squeeze(y).tolist()

    x = [y[0]]
    old = y[0]

    # Iterate over the elements of the input list, starting from the second
    # element
    for actual in y[1:]:
        if actual >= old:
            x.append(old)
            old = old
        else:
            x.append(actual)
            old = actual

    return np.array(x)


def _knee_locator(
    x: ndarray, y: ndarray, curve="convex", direction="decreasing", online=False, S=1.0
):
    """
    Find the knee point (or "elbow" ) in a curve. The knee point is the point of
    maximum curvature and is used as a heuristic to determine the optimal number
    of features to be removed.

    Parameters
    ----------
    x : ndarray
        Sorted x-axis values in increasing order.
    y : ndarray
        Corresponding y-axis values.
    curve : str, default='convex'
        Type of curve to fit. Can be "concave" or "convex".
    direction : str, default='decreasing'
        The direction to look for a knee point. Can be "increasing" or
        "decreasing".
    online : default=False
        When set to True, it updates the old knee values if necessary.
    S : default=1.0
        Sensitivity for knee location. It is a measure of how many "flat" points
        are expected in the unmodified data curve before declaring a knee.
    
    Returns
    -------
    knee : int
        The x-axis coordinate of the knee point.
    w : str
        Warning text used for debugging knee locator.
    """
    x = np.round(x, 2)
    y = np.round(y, 2)

    x_ = [x[0]]
    y_ = [y[0]]
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            x_.append(x[i])
            y_.append(y[i])

    x = np.array(x_)
    y = np.array(y_)

    # Find the first index such that y < 0.8 in order to locate the knee after this
    # If no index is found, the knee is found across all points
    index = np.argwhere(np.array(y) < 0.8)[0][0] if any(np.array(y) < 0.8) else 0

    if index > 0:
        max_y = np.max(y)

    # Filter the curve to only consider points beyond the threshold
    x, y = x[index:], y[index:]

    S_ = S + 50
    extend = True

    if extend:
        if len(x) == 1:
            m = 1
        else:
            m = int(max([x[i + 1] - x[i] for i in range(len(x) - 1)]))

        last_value = x[-1] + m * S_

        x = np.concatenate((x, list(range(int(x[-1] + m), last_value + 1, m))))
        y = np.concatenate((y, np.full(S_, y[-1])))

    x = np.array([x[0] - 1] + list(x))
    y = np.array([1.0] + list(y))

    # Define warning message initially empty
    warning_knee = ""

    # If there is only one point, return it as the knee
    if len(x) == 1:
        return x[1], warning_knee

    # Suppress specific warning message from KneeLocator
    with warnings.catch_warnings(record=True) as w:
        # Find the knee using the KneeLocator package
        knee_locator = KneeLocator(
            x, y, curve=curve, direction=direction, online=online, S=S
        )
        knee = knee_locator.knee

    # Concatenate all warning messages from knee location
    warning_knee = "\n".join([str(warning.message) for warning in w])

    # If no knee was found, return the last point
    if knee is not None:
        if knee == -1:
            knee = x[1]

        return knee, warning_knee

    print("Knee not found.")

    # If no knee was found, return the last point
    return x[1], warning_knee
