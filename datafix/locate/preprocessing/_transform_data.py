################################################################################
# Preprocessing functions to transform the data.
################################################################################

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn

from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List, Union


def reference_query_split(data: Union[DataFrame, ndarray], random_state: Union[None, int]=0):
    """
    Split input data into reference and query datasets with 50% of samples each.
    The input data is normalized so all features contain values in the range
    [0, 1].

    Parameters
    ----------
    data : array-like
        Input data.
    random_state : None or int, default=0
        Controls randomness by passing an integer for reproducible output.

    Returns
    -------
    reference : array-like
        Reference dataset.
    query : array-like
        Query dataset before manipulation.
    """
    # Set the random seed for reproducibility
    random.seed(random_state)

    # Normalize the dataset to have values between 0 and 1
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    print("The data was normalized to have all values between 0 and 1.")

    # Randomly partition the data into two datasets (reference and query)
    reference, query = train_test_split(data, random_state=random_state, test_size=0.5)

    # Define number of features in the original dataset
    n_features = reference.shape[1]
    print(
        f"The reference dataset contains {reference.shape[0]} samples and "
        f"{n_features} features."
    )
    print(
        f"The query dataset contains {query.shape[0]} samples and "
        f"{n_features} features."
    )

    return reference, query


def indexes_to_manipulate(
    query: Union[DataFrame, ndarray], fraction: float=0.1, maxStd: bool=False, random_state: Union[None, int]=0
):
    """
    Obtain indexes of the features to be manipulated in the query dataset.

    Parameters
    ----------
    query : array-like
        Query dataset before manipulation.
    fraction : float
        Percentage of features to be manipulated.
    maxStd : bool, default=False
        If True, manipulate the features with more variation.
    random_state : None or int, default=0
        Controls randomness by passing an integer for reproducible output.

    Returns
    -------
    manipulated_idxs : list
        Indexes of the features to be manipulated in the query dataset.
    """
    # Set the random seed for reproducibility
    random.seed(random_state)

    # Define number of features in the query dataset
    n_features = query.shape[1]

    # Define list with the indexes of all the features: 0, ..., n_features-1
    idxs = list(range(n_features))

    if maxStd:
        # Compute the standard deviation of each feature in the query dataset
        std = pd.DataFrame(query).std()

        # Define list with the indexes of the features with highest standard
        # deviation which will be manipulated
        indexes = np.argsort(std)[-round(n_features * fraction) :]
        print(
            f"{len(indexes)} features with highest standard deviation "
            "will be manipulated in the query dataset."
        )
    else:
        # Define list with a random selection of the indexes of the features
        # which will be manipulated
        indexes = random.sample(idxs, round(n_features * fraction))
        print(
            f"{len(indexes)} features selected randomly will be "
            "manipulated in the query dataset."
        )

    return indexes


class MLP(nn.Module):
    def __init__(self, input_size):
        """
        Parameters
        ----------
        input_size : int
            Input/output size. It is the number of feature to manipulate in the
            query.
        """
        # Set random seed for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
        )

    def compute_binary(self, x):
        """
        Compute if each feature in x is binary or not.
        """
        self.is_binary = (x.eq(0) | x.eq(1)).all(dim=0)

    def forward(self, x):
        """
        Transform the data forwarding it through the MLP. Apply min-max
        normalization to continuous features so their values are in the range
        [0, 1]. Apply binarization to binary features.
        """
        self.compute_binary(x)
        x = self.net(x)
        # Binarization with sigmoid
        x_bin = torch.sigmoid(x)
        x_bin[x_bin >= 0.5] = 1
        x_bin[x_bin < 0.5] = 0
        # Min-max normalization
        x_min, x_max = x.min(dim=0)[0], x.max(dim=0)[0]
        x_norm = (x - x_min) / (x_max - x_min)

        # Combine binarized and normalized features using self.is_binary
        x = x_bin * self.is_binary + x_norm * (~self.is_binary)
        return x


def manipulate_features(
    query_Y: Union[DataFrame, ndarray], 
    transformation: Union[int, float], 
    mlp_path: None=None, 
    random_state: Union[None, int]=0
):
    """
    Apply the specified transformation to all features in query_Y.

    Parameters
    ----------
    query_Y : array-like
        Selection of features from query dataset to be manipulated. It is assumed
        to contain min-max normalized values between 0.0 and 1.0.
    transformation : int or float
        Type of manipulation applied to the features.
        If 1, substitute each value with a random number between 0 and 1.
        If 2, substitute each value with its complement (1-x).
        If 3, the values of each feature are permuted (row permutation).
        If 4.1, add or subtract 0.02 from each value with a 50% probability.
        If 4.2, add or subtract 0.05 from each value with a 50% probability.
        If 4.3, add or subtract 0.1 from each value with a 50% probability.
        If 5, round each value to the nearest integer, with ties rounded to 0.
        If 6.1, flip the binary value of each feature element with 20% probability.
        If 6.2, flip the binary value of each feature element with 40% probability.
        If 6.3, flip the binary value of each feature element with 60% probability.
        If 6.4, flip the binary value of each feature element with 80% probability.
        If 7, transform the data forwarding it through a MLP. Apply min-max
        normalization to continuous features so their values are in the range
        [0, 1]. Apply binarization to binary features.
        If 8, the values of the features are permuted in the same order (row permutation).
    mlp_path : str, default=None
        Path to MLP model to use for manipulation type 7.
    random_state : None or int, default=0
        Controls randomness by passing an integer for reproducible output.

    Returns
    -------
    query_Y : array-like
        Transformed features in query dataset.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)

    print(f"Applying transformation {transformation}.")
    if transformation == 1:
        # Replace each value with a random number between 0 and 1
        query_Y = np.random.random((query_Y.shape[0], query_Y.shape[1]))
    elif transformation == 2:
        # Replace each value with its complement (1-x)
        query_Y = 1.0 - query_Y
    elif transformation == 3:
        # Permute the values of each feature independently among row elements
        for i in range(query_Y.shape[1]):
            random.shuffle(query_Y[:, i])
    elif transformation == 4.1:
        # Add or subtract 0.02 from each value with probability 50%
        A = np.random.choice([-1, 1], size=(query_Y.shape[0], query_Y.shape[1])) * 0.02
        query_Y += A
        # Clip to range [0, 1]
        query_Y = np.clip(query_Y, 0, 1)
    elif transformation == 4.2:
        # Add or subtract 0.05 from each value with probability 50%
        A = np.random.choice([-1, 1], size=(query_Y.shape[0], query_Y.shape[1])) * 0.05
        query_Y += A
        # Clip to range [0, 1]
        query_Y = np.clip(query_Y, 0, 1)
    elif transformation == 4.3:
        # Add or subtract 0.1 from each value with probability 50%
        A = np.random.choice([-1, 1], size=(query_Y.shape[0], query_Y.shape[1])) * 0.1
        query_Y += A
        # Clip to range [0, 1]
        query_Y = np.clip(query_Y, 0, 1)
    elif transformation == 5:
        # Round each value to the nearest integer
        query_Y = np.where(query_Y > 0.5, 1, 0)
    elif transformation == 6.1:
        # Generate a binary mask where positions with a True value will be flipped
        # Each position has a 20% probability of being set to True
        manipulation_mask = np.random.rand(*query_Y.shape) < 0.2
        query_Y[manipulation_mask] = 1 - query_Y[manipulation_mask]
    elif transformation == 6.2:
        # Generate a binary mask where positions with a True value will be flipped
        # Each position has a 40% probability of being set to True
        manipulation_mask = np.random.rand(*query_Y.shape) < 0.4
        query_Y[manipulation_mask] = 1 - query_Y[manipulation_mask]
    elif transformation == 6.3:
        # Generate a binary mask where positions with a True value will be flipped
        # Each position has a 60% probability of being set to True
        manipulation_mask = np.random.rand(*query_Y.shape) < 0.6
        query_Y[manipulation_mask] = 1 - query_Y[manipulation_mask]
    elif transformation == 6.4:
        # Generate a binary mask where positions with a True value will be flipped
        # Each position has a 80% probability of being set to True
        manipulation_mask = np.random.rand(*query_Y.shape) < 0.8
        query_Y[manipulation_mask] = 1 - query_Y[manipulation_mask]
    elif transformation == 7:
        mlp = MLP(query_Y.shape[1])
        if mlp_path:
            if os.path.exists(mlp_path):
                print(f"Using MLP saved in {mlp_path}")
                mlp.load_state_dict(torch.load(mlp_path))
        query_Y = torch.tensor(query_Y).float()
        query_Y = mlp.forward(query_Y).detach().numpy()
    elif transformation == 8:
        # Generate a random permutation of row indices
        permuted_indices = np.random.permutation(query_Y.shape[0])

        # Use the permutation to shuffle the rows of the matrix
        query_Y = query_Y[permuted_indices, :]
    else:
        raise ValueError(
            f"{transformation} transformation not recognized, "
            "please pass either: 1, 2, 3, 4.1, 4.2, 4.3, 5, 6.1, "
            "6.2, 6.3, 6.4, 7, 8."
        )

    return query_Y


def impute_features(
    reference: Union[DataFrame, ndarray], 
    query: Union[DataFrame, ndarray], 
    manipulated_idxs: List, 
    model: BaseEstimator
):
    """
    Apply the specified transformation to the specified features in the query.

    Parameters
    ----------
    reference : array-like
        Reference dataset.
    query : array-like
        Query dataset before manipulation.
    manipulated_idxs : list
        Indexes of the features to be imputed in the query dataset.
    model : ``Estimator`` instance
        A supervised learning estimator with ``fit`` and ``predict``  methods.

    Returns
    -------
    query_Y : array-like
        Transformed features in query dataset.
    """
    print(f"Applying transformation {model}.")

    # Define list with the indexes of the features to not be manipulated
    clean_idxs = list(set(range(reference.shape[1])) - set(manipulated_idxs))

    # Make a subset of the reference with clean features X and manipulated features Y
    reference_X, reference_Y = reference[:, clean_idxs], reference[:, manipulated_idxs]

    # Fit model on clean features to predict manipulated features
    model.fit(reference_X, reference_Y)

    # Make a subset of the query with clean features X
    query_X = query[:, clean_idxs]

    # Predict manipulated features in query dataset
    query_Y = model.predict(query_X)

    return query_Y


def compute_rmse_of_manipulation(query_clean: Union[DataFrame, ndarray], query: Union[DataFrame, ndarray]):
    """
    Measure the level of corruption of each feature in the query by computing
    the Root Mean Square Error (RMSE) of each feature in the query before and
    after manipulation.

    Parameters
    ----------
    query_clean : array-like of shape (n_samples, n_features)
        The query data before manipulation.
    query : array-like of shape (n_samples, n_features)
        The query data after manipulation.

    Returns
    -------
    rmse : array-like of shape (n_features,)
        A list containing the RMSE between each pair of clean and manipulated
        features.
    """
    # Compute the squared difference between each pair of corresponding features
    # in query_clean and query
    diff_squared = (query_clean - query) ** 2

    # Compute the mean of the squared differences along the first axis (i.e.,
    # across rows)
    mean_diff_squared = diff_squared.mean(axis=0)

    # Compute the square root of the mean squared differences to obtain the RMSE
    # between each pair of clean and manipulated features
    rmse = np.sqrt(mean_diff_squared)

    return rmse
