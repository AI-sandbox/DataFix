################################################################################
# Preprocessing functions to treat the data, convert from one data format to 
# another and generate labels for training and testing the estimator 
# (discriminator).
################################################################################

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from typing import Union


def _convert_to_dataframe(
    reference: Union[DataFrame, ndarray], query: Union[DataFrame, ndarray]
):
    """
    Convert the reference and query datasets to pandas DataFrames. The column
    names are the indexes of the respective columns.

    Parameters
    ----------
    reference : array-like
        Reference dataset.
    query : array-like
        Query dataset. It is assumed to contain the same features as the
        reference appearing in the same order.

    Returns
    -------
    reference : DataFrame
    query : DataFrame
    """
    # Convert reference and query datasets to DataFrames such that the name of
    # the columns are the index of each column
    reference = pd.DataFrame(
        reference, columns=list(map(str, range(reference.shape[1])))
    )
    query = pd.DataFrame(query, columns=list(map(str, range(query.shape[1]))))

    return reference, query


def _create_source_labels(
    reference: DataFrame, query: DataFrame
):
    """
    Concatenate the reference and query datasets and create a label array
    indicating the source of each sample (0 for reference, 1 for query).

    Parameters
    ----------
    reference : DataFrame
        Reference dataset.
    query : DataFrame
        Query dataset.

    Returns
    -------
    X : DataFrame
        Concatenated reference and query samples.
    y : ndarray
        Label array indicating source (reference/query) of each sample.
    """
    # Concatenate reference and query samples
    X = pd.concat((reference, query), axis=0)

    # Create labels for the samples (0 for reference, 1 for query)
    y = [0] * reference.shape[0] + [1] * query.shape[0]

    return X, np.array(y)


def _delete_columns(X: DataFrame, column_indexes: ndarray):
    """
    Remove specified columns from a dataframe by their indexes.

    Parameters
    ----------
    X : DataFrame
        Input dataframe with all columns.
    column_indexes : ndarray
        Indexes of columns to be removed from X.

    Returns
    -------
    DataFrame
        Dataframe with removed columns.
    """
    return X.drop(X.columns[column_indexes], axis=1)
