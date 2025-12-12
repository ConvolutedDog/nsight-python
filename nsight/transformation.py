# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data transformation utilities for Nsight Python profiling output.

This module contains functions that process raw profiling results, aggregate metrics,
normalize them, and prepare the data for visualization or further statistical analysis.
"""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _value_aggregator(agg_func_name: str) -> Callable[[pd.Series], NDArray[Any]]:
    """Factory function to create value aggregators.

    Args:
        agg_func_name: Name of the aggregation function ('mean', 'std', 'min', 'max')

    Returns:
        A function that aggregates a pandas Series into a numpy array

    Raises:
        ValueError: If agg_func_name is not supported
    """
    # Map aggregation names to numpy functions
    AGG_FUNCTIONS = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    if agg_func_name not in AGG_FUNCTIONS:
        raise ValueError(
            f"Unsupported aggregation: '{agg_func_name}'. "
            f"Supported: {list(AGG_FUNCTIONS.keys())}"
        )

    numpy_agg_func = AGG_FUNCTIONS[agg_func_name]

    def aggregator(series: pd.Series) -> NDArray[Any]:
        # Convert None to np.nan
        cleaned_series = series.apply(lambda x: np.nan if x is None else x)
        # Convert to numpy array, handling tuples
        arrays = np.array(
            [
                np.array(item) if isinstance(item, tuple) else item
                for item in cleaned_series
            ]
        )
        # Apply aggregation along axis 0
        return numpy_agg_func(arrays, axis=0)  # type: ignore[no-any-return,operator]

    return aggregator


def aggregate_data(
    df: pd.DataFrame,
    func: Callable[..., Any],
    normalize_against: str | None,
    output_progress: bool,
) -> pd.DataFrame:
    """
    Groups and aggregates profiling data by configuration and annotation.

    Args:
        df: The raw profiling results.
        func: Function representing kernel configuration parameters.
        normalize_against: Name of the annotation to normalize against.
        output_progress: Toggles the display of data processing logs

    Returns:
        Aggregated DataFrame and the (possibly normalized) metric name.
    """
    if output_progress:
        print("[NSIGHT-PYTHON] Processing profiled data")

    # Get the number of arguments in the signature of func
    num_args = len(inspect.signature(func).parameters)

    # Get the last N fields of the dataframe where N is the number of arguments
    # Note: When num_args=0, we need an empty list (not all columns via [-0:])
    func_fields = df.columns[-num_args:].tolist() if num_args > 0 else []

    # Function to convert non-sortable columns to tuples or strings
    def convert_non_sortable_columns(dframe: pd.DataFrame) -> pd.DataFrame:
        for col in dframe.columns:
            try:
                # Try sorting the column to check if it's sortable.
                sorted(dframe[col].dropna())
            except (TypeError, ValueError):
                # If the column is np.ndarray/list, convert them to tuples (hashable and comparable).
                if (
                    hasattr(dframe[col], "apply")
                    and dframe[col].apply(lambda x: isinstance(x, np.ndarray)).any()
                ):
                    dframe[col] = dframe[col].apply(lambda x: tuple(x))
                else:
                    # Convert the column to string.
                    dframe[col] = dframe[col].astype(str)
        return dframe

    # Convert non-sortable columns before grouping
    df = convert_non_sortable_columns(df)

    # Preserve original order by adding an index column
    df = df.reset_index(drop=True)
    df["_original_order"] = df.index

    # Build named aggregation dict for static fields
    named_aggs = {
        "AvgValue": ("Value", _value_aggregator("mean")),
        "StdDev": ("Value", _value_aggregator("std")),
        "MinValue": ("Value", _value_aggregator("min")),
        "MaxValue": ("Value", _value_aggregator("max")),
        "NumRuns": ("Value", "count"),
        "_original_order": (
            "_original_order",
            "min",
        ),  # Use min to preserve first occurrence
    }

    # Add assertion-based unique selection for remaining fields
    remaining_fields = [
        col
        for col in df.columns
        if col not in ["Value", "Annotation", "_original_order"] + func_fields
    ]

    for col in remaining_fields:
        if col == "Kernel":
            named_aggs[col] = (col, "first")
        else:
            named_aggs[col] = (
                col,
                (
                    lambda colname: lambda x: (
                        x.unique()[0]
                        if len(x.unique()) == 1
                        else (_ for _ in ()).throw(
                            AssertionError(
                                f"Column '{colname}' has multiple values in group: {x.unique()}"
                            )
                        )
                    )
                )(col),
            )

    # Apply aggregation with named aggregation
    agg_df = df.groupby(["Annotation"] + func_fields).agg(**named_aggs).reset_index()

    # Compute 95% confidence intervals
    agg_df["CI95_Lower"] = agg_df["AvgValue"] - 1.96 * (
        agg_df["StdDev"] / np.sqrt(agg_df["NumRuns"])
    )
    agg_df["CI95_Upper"] = agg_df["AvgValue"] + 1.96 * (
        agg_df["StdDev"] / np.sqrt(agg_df["NumRuns"])
    )

    # Compute relative standard deviation as a percentage
    agg_df["RelativeStdDevPct"] = (agg_df["StdDev"] / agg_df["AvgValue"]) * 100

    # Flag measurements as stable if relative stddev is less than 2%
    agg_df["StableMeasurement"] = agg_df["RelativeStdDevPct"].apply(
        lambda x: np.all(x < 2.0)
    )

    # Flatten the multi-index columns
    agg_df.columns = [col if isinstance(col, str) else col[0] for col in agg_df.columns]

    # Sort by original order to preserve user-provided configuration order
    agg_df = agg_df.sort_values("_original_order").reset_index(drop=True)
    agg_df = agg_df.drop("_original_order", axis=1)  # Remove the helper column

    do_normalize = normalize_against is not None
    if do_normalize:
        assert (
            normalize_against in agg_df["Annotation"].values
        ), f"Annotation '{normalize_against}' not found in data."

        # Create a DataFrame to hold the normalization values
        normalization_df = agg_df[agg_df["Annotation"] == normalize_against][
            func_fields + ["AvgValue"]
        ]
        normalization_df = normalization_df.rename(
            columns={"AvgValue": "NormalizationValue"}
        )

        # Merge with the original DataFrame to apply normalization
        agg_df = pd.merge(agg_df, normalization_df, on=func_fields)

        # Normalize the AvgValue by the values of the normalization annotation
        agg_df["AvgValue"] = agg_df["NormalizationValue"] / agg_df["AvgValue"]

        # Update the metric name to reflect the normalization
        agg_df["Metric"] = (
            agg_df["Metric"].astype(str) + f" relative to {normalize_against}"
        )

    # Calculate the geometric mean of the AvgValue column for each annotation
    def compute_group_geomean(valid_values: pd.Series) -> Any:
        arrays = np.vstack(valid_values.values)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_vals = np.log(arrays)
            return np.exp(np.mean(log_vals, axis=0))

    geomean_values = {}
    for annotation in agg_df["Annotation"].unique():
        annotation_data = agg_df[agg_df["Annotation"] == annotation]
        valid_values = annotation_data["AvgValue"].dropna()
        if not valid_values.empty:
            geomean = compute_group_geomean(valid_values)
            geomean_values[annotation] = geomean
        else:
            geomean_values[annotation] = np.nan

    # Add geomean values to the DataFrame
    agg_df["Geomean"] = agg_df["Annotation"].map(geomean_values)

    return agg_df
