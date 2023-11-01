import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import numpy as np
import json
import os
import os.path as path

directory = "."

with open("config.json") as json_file:
    config = json.load(json_file)

config["checkerboard_medium"] = {"classes": 2, "sota": 0.00}

factor = None
# factor = 0.75

if factor:
    for dataset in config.keys():
        config[dataset]["sota"] = config[dataset]["sota"] * factor

# Prepare df_results
df_results = pd.read_csv("results.csv")
df_results = df_results[df_results.dataset.isin(config.keys())]
df_results.drop(columns=df_results.columns[0], axis=1, inplace=True)
df_results["results"] = df_results.results.apply(
    lambda x: [float(z) for z in x[1:-1].split(", ")]
    if x.startswith("[")
    else [float(x)]
)


# Get transformation from identifier
def split_identifiers(x):
    for v in ["_l2_", "_beta_", "_measure_"]:
        if v in x:
            x = x.split(v)[0]
    return x


df_results["transformation"] = df_results.identifier.apply(split_identifiers)

# Split into upper and lower
df_results["value"] = df_results.results.apply(lambda x: x[0])
df_results["value_type"] = "upperbound"

scaling_constant = 0.95
cp_df = df_results.copy()
cp_df["value"] = cp_df.results.apply(
    lambda x: x[1] if len(x) > 1 else x[0] * scaling_constant
)
cp_df["value_type"] = "lowerbound"

df_results = pd.concat([df_results, cp_df], ignore_index=True)

# Add 1NN
cp_df = df_results[
    (df_results.method == "knn") & (df_results.variant.str.endswith(", k=1"))
].copy()

cp_df["method"] = "1nn"
cp_df["variant"] = cp_df.variant.apply(lambda x: x[: -len(", k=1")])

df_results = pd.concat([df_results, cp_df], ignore_index=True)

# Change LR and add other constant
df_results["method"] = df_results.method.apply(
    lambda x: x if x != "lr_model" else f"lr_model_{scaling_constant}"
)

scaling_constant = 0.8
cp_df = df_results[
    (df_results.method.str.startswith("lr_model"))
    & (df_results.value_type == "upperbound")
].copy()
cp_df["method"] = f"lr_model_{scaling_constant}"

cp_df2 = cp_df.copy()
cp_df2["value"] = cp_df2.results.apply(
    lambda x: x[1] if len(x) > 1 else x[0] * scaling_constant
)
cp_df2["value_type"] = "lowerbound"
df_results = pd.concat([df_results, cp_df, cp_df2], ignore_index=True)

# Set label
df_results["label"] = (
    df_results.method + "/" + df_results.variant + "/" + df_results.transformation
)

# DROP NaNs
df_results.dropna(subset=["value"], inplace=True)

# Get results

df = df_results

columns = [
    "dataset",
    "method",
    "variant",
    "transformation",
    "upperbound",
    "lowerbound",
    "eu_top",
    "eu_bottom",
    "eu_sum",
    "el_top",
    "el_bottom",
    "el_sum",
    "time",
]
rows = []


def comp_sub_score(upper, lower, v, max_c):
    if max_c < v:
        v = max_c
    if 0 > v:
        v = 0.0
    diff_top = 0.0
    diff_bottom = 0.0
    if upper < v:
        diff_top = v - upper
    elif lower > v:
        diff_bottom = lower - v
    return (diff_top, diff_bottom, diff_bottom + diff_top)


for (dataset, method, variant, transformation), df_grp in df.groupby(
    ["dataset", "method", "variant", "transformation"]
):
    print(f"Running for {dataset}, {method}, {variant}, {transformation}")
    sota_value = config[dataset]["sota"]
    classes = config[dataset]["classes"]

    upperbound = df_grp[
        (df_grp.value_type == "upperbound") & (df_grp.noise == 0.0)
    ].iloc[0]["value"]
    lowerbound = df_grp[
        (df_grp.value_type == "lowerbound") & (df_grp.noise == 0.0)
    ].iloc[0]["value"]

    max_c = (classes - 1.0) / classes

    eus = []
    els = []
    times = []

    for noise, noise_df in df_grp.groupby("noise"):
        upper = sota_value + noise * (max_c - sota_value)
        lower = noise * max_c

        upper_bounds = noise_df[noise_df.value_type == "upperbound"].value
        lower_bounds = noise_df[noise_df.value_type == "lowerbound"].value

        upperbound_errors = upper_bounds.apply(
            lambda v: comp_sub_score(upper, lower, v, max_c)
        )
        lowerbound_errors = lower_bounds.apply(
            lambda v: comp_sub_score(upper, lower, v, max_c)
        )

        t, b, s = upperbound_errors.apply(pd.Series).mean()
        eus.append([t, b, s])
        t, b, s = lowerbound_errors.apply(pd.Series).mean()
        els.append([t, b, s])
        times.append(float(noise_df.time.mean()))

    assert len(eus) > 0 and len(els) > 0, "No values for upper and/or lower bounds"

    eu = np.mean(eus, axis=0)
    el = np.mean(els, axis=0)
    time = np.mean(times)

    assert (
        eu[2] >= 0.0 and eu[2] <= 1.0
    ), f"error for {dataset}, {method}, {variant}, {transformation} and eu:{eu[2]}"
    assert (
        el[2] >= 0.0 and el[2] <= 1.0
    ), f"error for {dataset}, {method}, {variant}, {transformation} and el:{el[2]}"

    rows.append(
        [
            dataset,
            method,
            variant,
            transformation,
            upperbound,
            lowerbound,
            eu[0],
            eu[1],
            eu[2],
            el[0],
            el[1],
            el[2],
            time,
        ]
    )

    suffix = ", k=1"
    if method == "knn" and variant.endswith(suffix):
        rows.append(
            [
                dataset,
                "1nn",
                variant[: -(len(suffix))],
                transformation,
                upperbound,
                lowerbound,
                eu[0],
                eu[1],
                eu[2],
                el[0],
                el[1],
                el[2],
                time,
            ]
        )

df_areas = pd.DataFrame(rows, columns=columns)
filename = "areas.csv" if not factor else "areas_{0:.2f}.csv".format(factor)
df_areas.to_csv(path.join(directory, filename))
