################################################################################
# Functions to assist with different utilities.
################################################################################

import numpy as np
import pandas as pd


from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from typing import Iterable, List, Union


def _check_arguments(datafix):
    """
    Check arguments of datafix object.

    Attributes
    ----------
    datafix : ``DFLocate`` instancce.
        ``DFLocate`` instance.
    """
    # Ensure "scoring" is a list of unique strings which contains 'balanced_accuracy'
    assert isinstance(
        datafix["scoring"], list
    ), f'scoring = {datafix["scoring"]} is not a list.'
    assert (
        "balanced_accuracy" in datafix["scoring"]
    ), f"balanced_accuracy not in scoring."
    assert len(datafix["scoring"]) == len(set(datafix["scoring"])), (
        f'Scoring = {datafix["scoring"]} is not a list ' "of unique strings."
    )

    # Ensure "step" and "percentage" are not defined at a time
    assert not all(
        datafix.get(key) for key in ("step", "percentage")
    ), '"step" and "percentage" cannot be defined at a time.'

    # Ensure "alpha" and "threshold" are not defined at a time
    assert not all(
        datafix.get(key) for key in ("alpha", "threshold")
    ), '"alpha" and "threshold" cannot be defined at a time.'

    # Ensure "cv" and "test_size" are not defined at a time
    assert not all(
        datafix.get(key) for key in ("cv", "test_size")
    ), '"cv" and "test_size" cannot be defined at a time.'

    # Ensure return_estimator is a boolean
    assert isinstance(
        datafix["return_estimator"], bool
    ), '"return_estimator" must be a boolean.'

    # Ensure "percentage" is None or a float between 0 and 1
    assert (
        datafix["percentage"] is None or 0 < datafix["percentage"] < 1
    ), '"Percentage" must be None or a float between 0 and 1.'


def _check_data(
    reference: Union[DataFrame, ndarray], query: Union[DataFrame, ndarray]
):
    """
    Verify that the reference and query datasets have the same number of
    features. If both the reference and query are DataFrames, also ensure that
    the order of the features and column names match.

    Attributes
    ----------
    reference : array-like
        The reference dataset.
    query : array-like
        The query dataset.

    Returns
    -------
    n_features_in_ : int
        Number of features in the reference and query datasets.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features. Defined only when reference and query have feature
        names that are all strings and that are matching.
    """
    # Obtain number of features in reference and query
    n_features_in_ = reference.shape[1]
    n_features_in_query = query.shape[1]

    # Check reference and query have the same number of features
    assert n_features_in_ == n_features_in_query, (
        "Reference and query have "
        f"different number of features. {n_features_in_} != {n_features_in_query}."
    )

    # Check reference and query have the same feature names
    if isinstance(reference, pd.DataFrame) and isinstance(query, pd.DataFrame):
        # Obtain column names
        feature_names_in_ = reference.columns
        feature_names_in_query = query.columns

        assert (feature_names_in_ == feature_names_in_query).all(), (
            "Reference and " "query have different column names."
        )
    else:
        feature_names_in_ = None

    return n_features_in_, np.asarray(feature_names_in_)

def _sanity_check(reference : Union[DataFrame, ndarray],
                  query : Union[DataFrame, ndarray]):
    """
    Perform a sanity check on the reference and query datasets.
    
    Attributes
    ----------
    reference : array-like
        The reference dataset.
    query : array-like
        The query dataset.
        
    Returns
    -------
    reference : array-like
        The reference dataset with missing values replaced by -1 if any.
    query : array-like
        The query dataset with missing values replaced by -1 if any.
    """
    missings_reference = np.isnan(reference)
    missings_query = np.isnan(query)
    
    if missings_reference.any():
        # Replace missing values with -1 in reference
        reference[missings_reference] = -1
    
    if missings_query.any():
        # Replace missing values with -1 in query
        query[missings_query] = -1
    
    return reference, query

def _obtain_cv(
    cv: Union[None, int, str, Iterable],
    test_size: Union[float, int],
    random_state: Union[None, int],
):
    """
    Create a cross-validation object to be used in machine learning model
    evaluation based on input parameters.

    Parameters
    ----------
    cv : None, int, str, or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default single stratified train/test split.
        - integer, to specify the number of folds of stratified K-fold
          cross-validation.
        - An iterable yielding (train, test) splits as arrays of indices.
        If None, :class:`StratifiedShuffleSplit` is used, with n_splits=1,
        and for the specified test_size. If an integer is provided, then
        :class:`StratifiedKFold` is used.
    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
    random_state : None or int, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
    """
    if cv is None:
        cv = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
    elif isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    return cv
