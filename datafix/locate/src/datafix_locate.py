################################################################################
# DataFix Location.
################################################################################

import copy
import math
import random
import re
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

from joblib import Parallel, delayed
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
from pandas import DataFrame
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Union

from .location._early_stopping import EarlyStopping
from .location._filtering import _localize_corrupted_features
from .location._knee_location import (
    _interpolate,
    _knee_locator,
    _opening_left_right
)
from .location._misc import (
    _check_arguments,
    _check_data,
    _obtain_cv,
    _sanity_check
)
from .preprocessing._data_manipulation import (
    _convert_to_dataframe,
    _create_source_labels,
    _delete_columns
)
from .utils._evaluation import _evaluate_discriminator


class DFLocate:
    """
    DataFix shift detection and corrupt feature localization using a supervised
    learning estimator that acts as a discriminator.

    Parameters
    ----------
    estimator : ``Estimator`` instance, default=RandomForestClassifier(random_state=0)
        A supervised learning classifier with a ``fit`` method that provides
        information about feature importance through a ``coef_`` attribute or a
        ``feature_importances_`` attribute. The default estimator is
        `RandomForestClassifier` with a random state of 0.
    cv : None, int, str, or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for ``cv`` are:
        - None, to use the default single stratified train/test split
        :class:`StratifiedShuffleSplit`, with ``n_splits`` = 1, and
        ``test_size`` = ``test_size``.
        - int, to specify the number of folds of stratified K-fold
          cross-validation :class:`StratifiedKFold`.
        - An iterable with ``split`` method yielding (train, test) splits as
        arrays of indexes.
    test_size : float or int, default=0.2
        Useful only when ``cv`` is None. If float, should be between 0.0 and 1.0
        and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
    scoring : str or list, default='balanced_accuracy'.
        It specifies the metric(s) to evaluate the performance of the estimator. 
        The supported metrics can be found in `sklearn.metrics` or alternatively, 
        it can be one of the following:
        - ``D1``: log(p1) + log(1-p0).
        - ``D2``: log(p1 / p0).
        - ``D3``: -log((1/p0) - 1).
    n_jobs : int, default=None
        Number of jobs to run in parallel. None means 1, unless in a
        joblib.parallel_backend context. -1 means using all processors.
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split and iteration.
    step : None, int or float, default=None
        If greater than or equal to 1, then step corresponds to the (integer)
        number of features to remove at each iteration. If within (0.0, 1.0),
        then step corresponds to the percentage (rounded down) of features to
        remove at each iteration, based on the number of remaining features.
        Note that the last iteration may remove fewer than step features in order
        to reach ``max_features_to_filter``. None means there is no limit in the
        (integer) number of features removed in each iteration.
    percentage : None or float, default=0.1.
        To determine the maximum percentage of the total feature importances
        that can be removed at each iteration. The removal percentage is
        calculated as the specified percentage multiplied by
        `2·[balanced_accuracy_(t) - 0.5]`, where balanced_accuracy_(t) is the
        balanced accuracy of the estimator at iteration t. At a maximum, when
        the balanced accuracy is 1.0, the specified percentage of the total
        importances is removed. At a minimum, when the balanced accuracy is 0.5
        or lower, only a single feature is removed. None means there is no limit
        in the percentage of the total importances removed at each iteration.
    alpha : None, int or float, default=1
        Used to define a threshold at ``alpha``/``n_features_t``, with
        ``alpha`` ≥ 1 and ``n_features_t`` being the number of features at
        iteration t. Only features with importance over the threshold are removed.
        The threshold adjusts dynamically based on the number of available
        features at each iteration. None means that the minimum feature
        importance of a feature to be removed needs to be above 0.
    threshold : None, int or float, default=None
        Only features with importance over the threshold are removed. Unlike
        ``alpha``, the threshold is static. None means that the minimum feature
        importance of a feature to be removed needs to be above 0.
    margin : None or float, default=0.01
        Stop condition based on balanced accuracy being less than 0.5 + ``margin``.
        None means balanced accuracy is not used as a stop condition, except
        if ``patience`` is not None.
    max_features_to_filter : None, int or float, default=None
        If greater than or equal to 1, then max_features_to_filter corresponds
        to the maximum (integer) number of features to remove in total. If within
        (0.0, 1.0), then max_features_to_filter corresponds to the percentage
        (rounded down) of maximum features to remove in total. None means there
        is no limit in the maximum (integer) number of features to remove in
        total.
    max_it : None or int, default=None
        Maximum number of iterations. None means there is no limit in the
        number of iterations.
    patience : None or int, default=None
        Maximum number of iterations without improvement in balanced accuracy.
        None means balanced accuracy is not used as a stop condition, except
        if ``margin`` is not None.
    random_state : None or int, default=None
        Controls randomness by passing an integer for reproducible output.
    find_best : None or 'knee-balanced', default='knee-balanced'
        If 'knee-balanced', the optimal number of features to eliminate is
        determined by finding the knee of the curve representing the balanced
        accuracy of the estimator vs the number of removed features. None means
        the algorithm does not search for the optimal iteration; instead, it 
        returns the last iteration as the optimal one.
    window_length : None or int, default=5
        Useful only when ``find_best`` == 'knee-balanced'. Used to determine the
        length of the filter window for Savitzky-Golay filter. The window length
        is computed as: `max(5, (delta*window_length)//2*2+1)`, where delta is
        the mean distance between ``corrupted_features_`` points.
    polyorder : None or int, default=4
        Useful only when ``find_best`` == 'knee-balanced'. The polyorder used to
        fit the samples for Savitzky-Golay filter.
    S : None or int, default=5
        Useful only when ``find_best`` == 'knee-balanced'. Sensitity for knee
        location. It is a measure of how many “flat” points are expected in
        the unmodified data curve before declaring a knee.  If the algorithm
        fails to detect a knee point, the sensitivity is gradually decreased
        until a solution is found. If no solution is found with the minimum
        sensitivity of 1, then the last iteration is considered the knee point.
    online : None or bool, default=False
        Useful only when ``find_best`` == 'knee-balanced'. When set to True, it
        "corrects" old knee values if necessary.
    verbose : bool, default=False
        Verbosity.

    Attributes
    ----------
    scores_ : dict
        split(k)_test_score : list of length (n_iters_)
            The cross-validation scores across (k)th fold.
        mean_test_score : list of length (n_iters_)
            Mean of scores over all folds.
        std_test_score : list of length (n_iters_)
            Standard deviation of scores over all folds.
        mean_test_balanced_accuracy_smooth : list of length (n_iters_)
            Mean balanced accuracy after smoothing with Savitzky-Golay smoothing,
            opening and truncation to 0.5. Only available after ``knee_location()``.
    estimators_ : dict
        split(k)_estimator : list of length (n_iters_)
            The fitted estimator across (k)th fold. Only stored when
            ``return_estimator`` == True.
    runtime_ : list of length (n_iters_)
        Runtime of each ``shift_detection()`` or ``shift_location()``
        iteration.
    corrupted_features_ : list of length (n_iters_)
        Total number (cummulative) of features detected as being corrupted at
        each ``shift_location()`` iteration.
    mask_ : list of length (n_features_in_)
        The mask of corrupted features, where 1 indicates a variable is
        corrupted and 0 otherwise.
    ranking_ : list of length (n_features_in_)
        The iteration number when each feature is detected as being corrupted.
        Features not identified as corrupted at any iteration have zero value.
    importances_ : list of length (n_features_in_)
        The normalized importance of a feature at the iteration when it is
        detected as being corrupted. The normalized importance is averaged
        over all folds. Features not identified as corrupted at any iteration
        have zero value.
    n_corrupted_features_ : int
        Total number of features detected as being corrupted after
        ``shift_location()``.
    n_iters_ : int
        Number of iterations performed.
    n_features_in_ : int
        Number of input reference/query features.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of input reference/query features. Only defined when the feature
        names are all strings and match between reference and query.
    n_samples_reference_ : int
        Number of samples in reference.
    n_samples_query_ : int
        Number of samples in query.
    """
    def __init__(
        self,
        estimator=RandomForestClassifier(random_state=0),
        cv=None,
        test_size=0.2,
        scoring="balanced_accuracy",
        n_jobs=1,
        return_estimator=False,
        step=None,
        percentage=0.1,
        alpha=1,
        threshold=None,
        margin=0.01,
        max_it=None,
        max_features_to_filter=None,
        patience=None,
        find_best="knee-balanced",
        window_length=2,
        polyorder=4,
        S=5,
        online=False,
        random_state=None,
        verbose=False
    ):
        self.estimator = estimator
        self.cv = cv
        self.test_size = test_size
        self.scoring = [scoring] if isinstance(scoring, str) else scoring
        self.n_jobs = n_jobs
        self.return_estimator = return_estimator
        self.step = step
        self.percentage = percentage
        self.alpha = alpha
        self.threshold = threshold
        self.margin = margin
        self.max_it = max_it
        self.max_features_to_filter = max_features_to_filter
        self.patience = patience
        self.find_best = find_best
        self.window_length = window_length
        self.polyorder = polyorder
        self.S = S
        self.online = online
        self.random_state = random_state
        self.verbose = verbose
        
        _check_arguments(vars(self))

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def copy(self):
        """
        Returns a deep copy of the current object.
        """
        return copy.deepcopy(self)

    def _update_scores(self, scores: Dict, k: int):
        """
        Update ``scores_`` attribute with the test scores for (k)th fold in
        the current iteration.

        Parameters
        ----------
        scores : dict
            A dictionary containing the computed score for each specified
            scoring.
        k : int
            Number of the split or fold.
        """
        # Append each score to list if existing key in ``self.scores_``,
        # create it otherwise
        for score_name, score_value in scores.items():
            key = f"split{k}_test_{score_name}"
            if key in self.scores_.keys():
                self.scores_[key].append(score_value)
            else:
                self.scores_[key] = [score_value]

    def _update_estimators(self, estimator: BaseEstimator, k: int):
        """
        Update ``estimators_`` attribute with the fitted estimator for (k)th fold in
        the current iteration.

        Parameters
        ----------
        estimator : ``Estimator`` instance.
            A fitted supervised learning estimator with a ``fit`` method that
            provides information about feature importance either through a
            ``coef_`` attribute or through a ``feature_importances_`` attribute.
            A dictionary containing the computed score for each specified
            scoring.
        k : int
            Number of the split or fold.
        """
        if self.return_estimator:
            # Append estimator to list if existing key in ``self.estimators_``,
            # create it otherwise
            key = f"split{k}_estimator"
            if key in self.estimators_.keys():
                self.estimators_[key].append(estimator)
            else:
                self.estimators_[key] = [estimator]

    def _update_mean_std_scores(self):
        """
        Update the attribute ``self.scores_`` with the mean and standard
        deviation of test scores across all folds in the current iteration.
        """
        for score_name in self.scoring:
            # For each scoring...
            # Obtain score values of last split across all folds
            score_values = [
                self.scores_[key][-1]
                for key in self.scores_.keys()
                if re.match(f"split\d+_test_{score_name}", key)
            ]
            # Compute mean and standard deviation of test scores across all splits
            mean_score_value = np.mean(score_values)
            std_score_value = np.std(score_values)
            # Append aggregated score to list if existing key in ``self.scores_``,
            # create it otherwise
            for key, value in [
                (f"mean_test_{score_name}", mean_score_value),
                (f"std_test_{score_name}", std_score_value),
            ]:
                if key in self.scores_:
                    self.scores_[key].append(value)
                else:
                    self.scores_[key] = [value]

    def _stop_condition(
        self, n_features_it: int, early_stopping: EarlyStopping, detected_features: int
    ):
        """
        Check if the termination criteria for ``shift_location()`` are met.

        Parameters
        ----------
        n_features_it : int
            Number of features at the current iteration.
        early_stopping : None or ``EarlyStopping`` instance
            Instance of ``EarlyStopping`` used to determine early stop based on
            the balanced accuracy improvement.
        detected_features : int
            Number of detected corrupted features.

        Returns
        -------
        stop_condition : bool
            True if termination criteria are met, False otherwise.
        """
        if detected_features == 0:
            # No detected corrupted feature
            return True

        if n_features_it == 0:
            # No feature left at current iteration
            return True

        if self.margin is not None:
            if self.scores_["mean_test_balanced_accuracy"][-1] < (0.5 + self.margin):
                # Balanced accuracy below 0.5+margin
                return True

        if self.max_features_to_filter is not None:
            # Filtered features at current iteration
            filtered_features = self.n_features_in_ - n_features_it
            # Maximum number of features to filter in total
            max_features_to_filter = (
                self.max_features_to_filter
                if self.max_features_to_filter >= 1
                else math.floor(self.max_features_to_filter * self.n_features_in_)
            )
            if filtered_features == max_features_to_filter:
                # Maximum number of filtered features reached
                return True

        if self.max_it is not None and self.n_iters_ == self.max_it:
            # Maximum iteration number reached
            return True

        if self.patience is not None:
            early_stopping(self.scores_["mean_test_balanced_accuracy"][-1])
            if early_stopping.early_stop:
                # Patience limit reached
                return True

        return False

    def _single_shift_detection(
        self,
        X: DataFrame,
        y: ndarray,
        train_index: ndarray,
        test_index: ndarray,
        k: int,
    ):
        """
        Perform a single iteration of shift detection for the (k)th fold. Fit
        the estimator on the training set and evaluate its performance on the
        test set.

        Parameters
        ----------
        X : DataFrame
            Concatenated reference and query samples.
        y : ndarray
            Label array indicating source (reference/query) of each sample
            (0 for reference, 1 for query).
        train_index : ndarray
            Indexes of the samples in the training set.
        test_index : ndarray
            Indexes of the samples in the testing set.
        k : int
            Split or fold number.
        """
        # Filter train and test samples based on (k)th fold indexes
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate the estimator
        estimator = clone(self.estimator)

        # Fit the estimator on the training set
        estimator = estimator.fit(np.array(X_train), y_train)

        # Predict the labels for the test set
        y_test_predicted = estimator.predict(np.array(X_test))

        if set(self.scoring) & {"D1", "D2", "D3", "JD", "KL", "TV"}:
            # Predict the probabilities for the test set
            y_test_probs = estimator.predict_proba(np.array(X_test))
        else:
            # The probabilities for the test set are not needed
            y_test_probs = None

        # Evaluate the estimator on the test set
        scores = _evaluate_discriminator(
            y_test, y_test_predicted, y_test_probs, self.scoring, self.n_jobs
        )

        # Update ``scores_`` attribute with the test scores for (k)th fold
        self._update_scores(scores, k)

        # Update ``estimators_`` attribute with the fitted estimator for (k)th split
        self._update_estimators(estimator, k)
        
        return estimator

    def shift_detection(
        self, reference: Union[DataFrame, ndarray], query: Union[DataFrame, ndarray]
    ):
        """
        Fit an estimator that acts as a discriminator aimed at distinguishing
        between samples from the reference or the query to identify if there
        is a distribution shift between the two datasets.

        Parameters
        ----------
        reference : array-like
            The reference dataset is assumed to contain high-quality data.
        query : array-like
            The query dataset might contain some – partially or completely –
            corrupted features. It must contain contain the same features
            as the reference appearing in the same order.

        Returns
        -------
        self : object
            DFLocate with computed attributes.
        """
        # Measure runtime to perform shift detection
        start_time = time.time()

        # Verify that the reference and query datasets have the same number of
        # features. If both the reference and query are DataFrames, also ensure
        # that the order of the features and column names match
        self.n_features_in_, self.feature_names_in_ = _check_data(reference, query)

        # Replace missing values by -1 if any
        self.reference, self.query = _sanity_check(reference, query)
        
        # Define number of samples in reference and query
        self.n_samples_reference_ = reference.shape[0]
        self.n_samples_query_ = query.shape[0]

        # Convert reference and query datasets to dataframe
        # such that the name of the columns are the index of each column
        reference, query = _convert_to_dataframe(reference, query)

        # Obtain the concatenation of all samples from the reference and query
        # datasets, and an array of labels indicating the source of each sample
        # (0 for reference, 1 for query)
        X, y = _create_source_labels(reference, query)

        # Define empty dictionaries to store discriminator scores and fitted
        # estimators
        self.scores_ = {}
        self.estimators_ = {}

        # Define empty list to store shift detection runtime
        self.runtime_ = []

        # Initialize cross-validation generator
        cv = _obtain_cv(self.cv, self.test_size, self.random_state)
        folds = list(cv.split(X, y))

        # Perform parallelized shift detection for each fold
        _ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._single_shift_detection)(X, y, train_index, test_index, k)
            for k, (train_index, test_index) in enumerate(folds)
        )

        # Update the attribute ``self.scores_`` with the mean and standard
        # deviation of test scores across all splits
        self._update_mean_std_scores()

        # Measure runtime to perform shift detection
        end_time = time.time()

        self.runtime_.append(end_time - start_time)

        if self.verbose:
            print(f"balanced_acc={self.scores_['mean_test_balanced_accuracy'][-1]}")
        
        return self

    def shift_location(
        self, reference: Union[DataFrame, ndarray], query: Union[DataFrame, ndarray]
    ):
        """
        Iteratively localize the corrupted features in the query causing a distribution shift.

        Parameters
        ----------
        reference : array-like
            The reference dataset is assumed to contain high-quality data.
        query : array-like
            The query dataset might contain some – partially or completely –
            corrupted features. It must contain contain the same features
            as the reference appearing in the same order.

        Returns
        -------
        self : object
            DFLocate with computed attributes.
        """
        # Verify that the reference and query datasets have the same number of
        # features. If both the reference and query are DataFrames, also ensure
        # that the order of the features and column names match
        self.n_features_in_, self.feature_names_in_ = _check_data(reference, query)
        
        # Replace missing values by -1 if any
        self.reference, self.query = _sanity_check(reference, query)
        
        # Define number of samples in reference and query
        self.n_samples_reference_ = reference.shape[0]
        self.n_samples_query_ = query.shape[0]

        # Convert reference and query datasets to dataframe
        # such that the name of the columns are the index of each column
        reference, query = _convert_to_dataframe(reference, query)

        # Obtain the concatenation of all samples from the reference and query
        # datasets, and an array of labels indicating the source of each sample
        # (0 for reference, 1 for query)
        X, y = _create_source_labels(reference, query)

        # Define empty dictionaries to store discriminator scores and fitted
        # estimators
        self.scores_ = {}
        self.estimators_ = {}

        # Define empty list to store shift detection runtime
        self.runtime_ = []

        # Initialize cross-validation generator
        cv = _obtain_cv(self.cv, self.test_size, self.random_state)
        folds = list(cv.split(X, y))

        self.n_iters_ = 0  # Iteration counter
        self.n_corrupted_features_ = 0  # Counter for removed features
        self.corrupted_features_ = [0]  # List of removed features counters

        # Create early stopping object if patience is set
        early_stopping = EarlyStopping(self.patience) if self.patience else None

        # Store iteration number for each feature when it is detected as corrupted
        self.ranking_ = np.zeros(self.n_features_in_).astype(int)

        # Store the mask of corrupted features
        # (1 = corrupted, 0 = not corrupted)
        self.mask_ = np.zeros(self.n_features_in_).astype(int)

        # Store the importance of corrupted features
        self.importances_ = np.zeros(self.n_features_in_).astype(float)

        # Stop codition boolean
        stop_condition = False

        while not stop_condition:
            # Measure time to perform iteration
            iteration_start_time = time.time()

            if self.n_iters_ > 0:
                # Obtain indexes of the corrupted features detected in this
                # iteration. The indexes are obtained from the column names of
                # the features to discard from the training and testing data
                indexes_corrupted = [int(idx) for idx in X.columns[to_discard]]

                # Filter corrupted features from the training and testing data
                X = _delete_columns(X, to_discard)

                # Update attributes with information on removed features
                self.ranking_[indexes_corrupted] = self.n_iters_
                self.mask_[indexes_corrupted] = 1
                self.importances_[indexes_corrupted] = importances
                self.n_corrupted_features_ += detected_features
                self.corrupted_features_.append(self.n_corrupted_features_)

            # Perform parallelized shift detection on each fold
            estimators = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(self._single_shift_detection)(X, y, train_index, test_index, k)
                for k, (train_index, test_index) in enumerate(folds)
            )

            # Update the attribute ``self.scores_`` with the mean and standard
            # deviation of test scores across all splits
            self._update_mean_std_scores()

            # Obtain balanced accuracy and update early stopping if defined
            balanced_accuracy = self.scores_["mean_test_balanced_accuracy"][-1]

            # Get step, percentage, threshold, and max_features_to_filter values
            step = self.step
            percentage = self.percentage
            threshold = self.threshold
            max_features_to_filter = self.max_features_to_filter
            max_features_to_filter_it = self.max_features_to_filter

            # If threshold is not set, set it to 0
            if threshold is None:
                threshold = 0

            # If step is a fraction, floor it to nearest integer
            if step is not None and step < 1:
                step = math.floor(step * X.shape[1])

            # If alpha is set, calculate threshold based on it
            if self.alpha is not None:
                threshold = 1 / X.shape[1] * self.alpha

            # If percentage is set, adjust it based on mean balanced accuracy
            if self.percentage is not None:
                percentage *= 2 * (balanced_accuracy - 0.5)

            # If max_features_to_filter is set as a fraction, floor it to nearest integer
            if self.max_features_to_filter is not None:
                if self.max_features_to_filter < 1:
                    max_features_to_filter = math.floor(
                        self.max_features_to_filter * self.n_features_in_
                    )
                # Compute maximum number of features to filter at current iteration
                max_features_to_filter_it = (
                    max_features_to_filter - self.n_corrupted_features_
                )

            # Obtain corrupted features to removed in next iteration, the number of
            # corrupted features and their importance
            to_discard, detected_features, importances = _localize_corrupted_features(
                estimators, step, percentage, threshold, max_features_to_filter_it
            )

            # Increase iteration counter
            self.n_iters_ += 1
            
            if self.verbose:
                print(
                    f"iteration={self.n_iters_} |",
                    f"balanced_acc={balanced_accuracy} |",
                    f"corrupted_features={detected_features}"
                )

            # Update stop condition
            stop_condition = self._stop_condition(
                X.shape[1], early_stopping, detected_features
            )

            # Measure time to perform iteration
            iteration_end_time = time.time()
            self.runtime_.append(iteration_end_time - iteration_start_time)

        if self.find_best == "knee-balanced":
            # Find optimal iteration from the curve with the balanced
            # accuracy of the estimator vs the number of removed features
            self.knee_location(
                window_length=self.window_length,
                polyorder=self.polyorder,
                S=self.S,
                online=self.online,
            )

        return self

    def knee_location(
        self, window_length: int=2, polyorder: int=4, S: int=5, online: bool=False
    ):
        """
        Find the correct number of corrupted features by finding the knee of the
        curve representing the balanced accuracy of the estimator vs the number
        of removed features.

        Parameters
        ----------
        window_length : None or int, default=2
            Used to determine the length of the filter window for Savitzky-Golay 
            filter. The window length is computed as: `max(5, (delta*window_length)// 2*2+1)`, 
            where delta is the mean distance between ``corrupted_features_`` points.
        polyorder : None or int, default=4
            The polyorder used to fit the samples for Savitzky-Golay filter.
        S : None or int, default=5
            Sensitity for knee location. It is a measure of how many “flat” points are expected 
            in the unmodified data curve before declaring a knee.
        online : None or bool
            When set to True, it "corrects" old knee values if necessary.

        Returns
        -------
        self : object
            DFLocate with computed attributes.
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.S = S
        self.online = online

        if self.n_iters_ == 1:
            # Make a copy of attributes needed for plotting
            self.plot_ = {}
            self.plot_["corrupted_features"] = self.corrupted_features_
            self.plot_["mean_test_balanced_accuracy"] = self.scores_[
                "mean_test_balanced_accuracy"
            ]
            self.plot_["mean_test_balanced_accuracy_smooth"] = self.scores_[
                "mean_test_balanced_accuracy"
            ]
            self.warning_knee_ = ""
            return self

        # Compute the mean distance between the points in ``corrupted_features_``
        delta = self.corrupted_features_[-1] / (self.n_iters_ - 1)

        # Compute the window size for the Savitzky-Golay smoothing
        window = max(5, (delta * self.window_length) // 2 * 2 + 1)

        # Interpolate the data so that the distance between
        # ``interpolated_features`` points is 1
        interpolated_features, interpolated_accuracies = _interpolate(
            self.corrupted_features_, self.scores_["mean_test_balanced_accuracy"]
        )

        # Apply the Savitzky-Golay smoothing
        interpolated_accuracies_smooth = savgol_filter(
            interpolated_accuracies, window, self.polyorder, mode="nearest"
        )

        # Force each point in the right to be <= than each point in the left
        interpolated_accuracies_smooth = _opening_left_right(
            interpolated_accuracies_smooth
        )

        # Truncate all values below 0.5 to 0.5
        interpolated_accuracies_smooth[interpolated_accuracies_smooth < 0.5] = 0.5

        # Remove interpolation values
        balanced_accuracy_smooth = interpolated_accuracies_smooth[
            self.corrupted_features_
        ]

        # Find the knee from the smoothed balanced accuracy curve
        corrupted_features_knee, warning_knee = _knee_locator(
            self.corrupted_features_,
            balanced_accuracy_smooth,
            curve="convex",
            direction="decreasing",
            online=self.online,
            S=self.S,
        )

        # Find the iteration with the correct number of corrupted features
        iteration_knee = self.corrupted_features_.index(corrupted_features_knee) + 1

        # Make a copy of attributes needed for plotting
        self.plot_ = {}
        self.plot_["corrupted_features"] = self.corrupted_features_
        self.plot_["mean_test_balanced_accuracy"] = self.scores_[
            "mean_test_balanced_accuracy"
        ]
        self.plot_["mean_test_balanced_accuracy_smooth"] = balanced_accuracy_smooth

        # Store smoothed balanced accuracy and update relevant attributes
        # by removing iterations after the knee point
        self.runtime_ = self.runtime_[:iteration_knee]
        self.corrupted_features_ = self.corrupted_features_[:iteration_knee]
        self.importances_[self.ranking_ >= iteration_knee] = 0
        self.mask_[self.ranking_ >= iteration_knee] = 0
        self.ranking_[self.ranking_ >= iteration_knee] = 0
        self.n_corrupted_features_ = corrupted_features_knee
        self.n_iters_ = iteration_knee
        self.warning_knee_ = warning_knee

        for key in self.scores_.keys():
            self.scores_[key] = self.scores_[key][:iteration_knee]

        if self.return_estimator:
            for key in self.estimators_.keys():
                self.estimators_[key] = self.estimators_[key][:iteration_knee]

        return self

    def plot_evolution(self, f1_score=None):
        """
        Plot the evolution curve of balanced accuracy and smoothed balanced
        accuracy vs the number of corrupted features removed. Also plot the
        iteration or knee with the correct number of corrupted features.
        Optionally, plot the f1 score curve.

        Parameters
        ----------
        f1_score : None or list of length (n_iters_)
            An optional list of F1 Scores in the corrupted feature localization.
        """
        def fmt(x, pos):
            return f"{int(x)}\n{round(x/self.n_features_in_*100, 1)}%"

        # Set plot size and background color
        plt.figure(figsize=(25, 15))
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"

        # Update font size and axis width
        plt.rcParams.update({"font.size": 40})
        plt.tick_params("both", length=16, width=2.5, which="major")
        plt.rcParams["xtick.major.pad"] = "18"
        plt.rcParams["ytick.major.pad"] = "12"
        plt.rcParams.update({"axes.linewidth": "2.5"})
        plt.rcParams.update({"patch.linewidth": "2.5"})

        # Plot curves with the original and smoothed balanced accuracy vs
        # the number of corrupted features removed
        plt.plot(
            self.plot_["corrupted_features"],
            self.plot_["mean_test_balanced_accuracy"],
            markersize=6,
            linewidth=4.0,
            marker="o",
            alpha=1,
            color="#1f77b4",
            label="Balanced Accuracy",
        )
        plt.plot(
            self.plot_["corrupted_features"],
            self.plot_["mean_test_balanced_accuracy_smooth"],
            markersize=6,
            linewidth=4.0,
            marker="o",
            alpha=1,
            color="darkorange",
            label="Smoothed Balanced Accuracy",
        )

        # Plot the F1 score in the corrupted feature localization if provided
        if f1_score is not None:
            plt.plot(
                self.plot_["corrupted_features"],
                f1_score,
                markersize=6,
                linewidth=4.0,
                marker="o",
                alpha=1,
                color="green",
                label="F1 Score",
            )

        plt.axvline(
            x=self.n_corrupted_features_,
            linewidth=6.0,
            ls="--",
            color="black",
            label="Knee",
        )

        # Add x and y axis labels
        plt.xlabel("\nRemoved Features", labelpad=-30)
        plt.ylabel("\nBalanced Accuracy", labelpad=10)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))

        # Plot legend and grid
        plt.legend(loc="lower right", fontsize=32)
        plt.grid()
        plt.show()
        plt.close()
