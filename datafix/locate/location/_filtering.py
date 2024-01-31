################################################################################
# Functions to localize the corrupted features at each iteration.
################################################################################

import numpy as np

from numpy import ndarray
from typing import Dict, List, Union


def _compute_mean_feature_importances(estimators: List):
    """
    Compute the mean of the feature importances across estimators.
    Normalize the feature importances to be between 0 and 1 and add 1.

    Parameters
    ----------
    estimators : list
        The fitted estimators across all folds at iteration `it`.

    Returns
    -------
    normalized_importances : ndarray
        Normalized feature importances.
    """
    # Obtain average importance across estimators fitted on different folds
    for k, estimator in enumerate(estimators):
        # Obtain feature importances or importances of an estimator
        if hasattr(estimator, "coef_"):
            importances = estimator.coef_
            importances = np.squeeze(importances, 0)
        elif hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            exit(f"{estimator} model does not have coef_ nor feature_importances_.")

        # Stack importances for all estimators
        if k == 0:
            all_importances = importances
            all_importances = np.expand_dims(all_importances, axis=1)
        else:
            all_importances = np.column_stack((all_importances, importances))

    # Calculate the mean importances across all estimators
    mean_importances = np.mean(all_importances, axis=1)

    # Normalize the feature importances to be between 0 and 1 and add 1
    normalized_importances = abs(mean_importances) / sum(abs(mean_importances))

    return normalized_importances


def _localize_corrupted_features(
    estimators: List,
    step: Union[int, None],
    percentage: Union[float, None],
    threshold: Union[float, int, None],
    max_features_to_filter_it: Union[int, None],
):
    """
    Localize the index of the features that contribute most to a distribution
    shift in current iteration.

    Parameters
    ----------
    estimators : list
        The fitted estimators across all folds at iteration `it`.
    step : None or int
        Maximum number of features to remove at each iteration. None means
        there is no limit in the (integer) number of features to remove at
        each iteration.
    percentage : None or float
        To determine the maximum percentage of the total feature importances
        that can be removed at each iteration. The removal percentage is
        calculated as the specified percentage multiplied by
        `2·[balanced_accuracy_(t) - 0.5]`, where balanced_accuracy_(t) is the
        balanced accuracy of the estimator at iteration t. At a maximum, when
        the balanced accuracy is 1.0, the specified percentage of the total
        importances is removed. At a minimum, when the balanced accuracy is 0.5
        or lower, only a single feature is removed. None means there is no limit
        in the percentage of the total importances removed at each iteration.
    threshold : None, int or float
        Only features with importance over the threshold are removed. Unlike
        ``alpha``, the threshold is static. None means that the minimum feature
        importance of a feature to be removed needs to be above 0.
    max_features_to_filter_it : None or int
        The maximum number of features to remove at current iteration.

    Returns
    -------
    to_discard : ndarray
        Indexes of the localized features as being corrupted.
    detected_features : int
        Number of localized feautres as being corrupted.
    importances: ndarray
        The normalized importance of each feature detected as being
        corrupted. The normalized importance is averaged over all folds.
    """
    # Compute the mean of the feature importances across estimators
    # Normalize the feature importances to be between 0 and 1 and add 1
    normalized_importances = _compute_mean_feature_importances(estimators)

    # Sort the features by decreasing importance
    sorted_indexes = np.argsort(-normalized_importances)
    sorted_importances = -np.sort(-normalized_importances)

    to_discard = []  # Indexes of the top corrupted features
    importances = []  # Importances of the top corrupted features
    detected_features = 0  # Amount of localized feautres as being corrupted.
    importance_cum = 0  # Captured importance

    for idx, importance in zip(sorted_indexes, sorted_importances):
        # Update the captured importance
        importance_cum += importance

        # Store the index of the feature if detected as being corrupted
        if (
            (step is None or detected_features < step)
            and (
                max_features_to_filter_it is None
                or detected_features < max_features_to_filter_it
            )
            and (importance > threshold)
        ):
            to_discard.append(idx)
            detected_features += 1
            importances.append(importance)
        else:
            break

        # Break if already surpassed the percentage of the total feature importance
        # to remove at current iteration
        if percentage is not None and importance_cum > percentage:
            break

    return to_discard, detected_features, importances
