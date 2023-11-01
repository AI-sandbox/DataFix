################################################################################
# Util functions to save output files.
################################################################################

import os
import pandas as pd
import pickle

from pandas import DataFrame

def save_pkl(output_path: str, output_name: str, new_rows: DataFrame):
    """
    Saves new rows with performance results in a .pkl file in output_path.
    If the file does not exist, it is created.

    Parameters
    ----------
    output_path : str
        Path to the output directory that will contain the .pkl file with metrics.
    output_name : str
        Name of the .pkl file.
    new_rows : DataFrame
        Dataframe with new rows with performance results.

    Returns
    -------
    None
    """
    if not os.path.exists(output_path):
        # If the output directory does not already exist, create it
        os.makedirs(output_path)

    # Define the full path to the output file
    final_path = os.path.join(output_path, output_name)

    if os.path.isfile(final_path):
        # Read the first dataframe from the Pickle file
        with open(final_path, "rb") as f:
            df = pickle.load(f)

        # Concatenate the two dataframes vertically
        df_concat = pd.concat([df, new_rows], axis=0)

        # Write the concatenated dataframe to a new Pickle file
        with open(final_path, "wb") as f:
            pickle.dump(df_concat, f)

    else:
        # Create new Pickle file
        with open(final_path, "wb") as f:
            pickle.dump(new_rows, f)
