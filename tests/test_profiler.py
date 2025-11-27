# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for Nsight Python profiler functionality.
"""

import os
import shutil
import tempfile
from typing import Any

import pytest
import torch

import nsight
from nsight import exceptions


def _simple_kernel_impl(x: int, y: int, annotation: str = "test") -> None:
    """Shared kernel implementation for testing."""
    a = torch.randn(x, y, device="cuda")
    b = torch.randn(x, y, device="cuda")
    with nsight.annotate(annotation):
        _ = a + b


# ============================================================================
# Decorator syntax tests
# ============================================================================


@nsight.analyze.kernel
def decorator_without_parens(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel without parentheses."""
    _simple_kernel_impl(x, y)


def test_decorator_without_parens() -> None:
    """Test that decorator works without parentheses."""
    decorator_without_parens(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel()
def decorator_with_empty_parens(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel() with empty parentheses."""
    _simple_kernel_impl(x, y)


def test_decorator_with_empty_parens() -> None:
    """Test that decorator works with empty parentheses."""
    decorator_with_empty_parens(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel(runs=3)
def decorator_with_args(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel(args) with arguments."""
    _simple_kernel_impl(x, y)


def test_decorator_with_args() -> None:
    """Test that decorator works with arguments."""
    decorator_with_args(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel
def decorator_without_parens_with_configs(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel without parentheses, configs at call time."""
    _simple_kernel_impl(x, y)


def test_decorator_without_parens_with_configs() -> None:
    """Test that decorator without parens works with configs at call time."""
    decorator_without_parens_with_configs(configs=[(42, 23), (12, 13)])


# ============================================================================
# Configuration handling tests
# ============================================================================


@nsight.analyze.kernel()
def config_at_call_time_positional(x: int, y: int) -> None:
    _simple_kernel_impl(x, y)


def test_config_at_call_time_positional() -> None:
    """Test providing configuration as positional arguments at call time."""
    config_at_call_time_positional(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel()
def config_at_call_time_configs(x: int, y: int) -> None:
    _simple_kernel_impl(x, y)


def test_config_at_call_time_configs() -> None:
    """Test providing configuration as configs list at call time."""
    config_at_call_time_configs(configs=[(42, 23), (12, 13)])


# ----------------------------------------------------------------------------


def test_config_at_call_time_with_kwargs() -> None:
    """Test that keyword arguments raise appropriate error."""
    with pytest.raises(
        exceptions.ProfilerException, match="Keyword arguments are not supported yet"
    ):
        config_at_call_time_configs(42, y=23)


# ----------------------------------------------------------------------------
# Tests for functions with no arguments
# ----------------------------------------------------------------------------


@nsight.analyze.kernel
def no_args_function_no_parens() -> None:
    """Test function with no arguments using decorator without parentheses."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_no_args_function_no_parens() -> None:
    """Test that function with no args works without providing configs."""
    # Should work without configs since the function takes no arguments
    # and we just want to run it once (or with default runs).
    no_args_function_no_parens()


# ----------------------------------------------------------------------------


@nsight.analyze.kernel()
def no_args_function_with_parens() -> None:
    """Test function with no arguments using decorator with empty parentheses."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_no_args_function_with_parens() -> None:
    """Test that function with no args works with empty parentheses."""
    # Should work without configs since the function takes no arguments
    # and we just want to run it once (or with default runs).
    result = no_args_function_with_parens()

    # Verify the dataframe structure
    assert result is not None, "ProfileResults should be returned"
    df = result.to_dataframe()

    # Should have exactly 1 row (1 annotation, 1 config with no params)
    assert len(df) == 1, f"Expected 1 row in dataframe, got {len(df)}"

    # Verify annotation name
    assert df["Annotation"].iloc[0] == "test", "Expected annotation 'test'"

    # Verify metric value is reasonable (should be positive)
    assert df["AvgValue"].iloc[0] > 0, "Expected positive metric value"


# ----------------------------------------------------------------------------


@nsight.analyze.kernel(runs=3)
def no_args_function_with_kwargs() -> None:
    """Test function with no arguments using decorator with keyword arguments."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_no_args_function_with_kwargs() -> None:
    """Test that function with no args works when decorator has kwargs."""
    # Should work without configs since the function takes no arguments
    # and we just want to run it multiple times with the specified runs.
    result = no_args_function_with_kwargs()

    # Verify the dataframe structure
    assert result is not None, "ProfileResults should be returned"
    df = result.to_dataframe()

    # Should have exactly 1 row (1 annotation, 1 config with no params)
    assert len(df) == 1, f"Expected 1 row in dataframe, got {len(df)}"

    # Verify that runs=3 was respected
    assert df["NumRuns"].iloc[0] == 3, f"Expected 3 runs, got {df['NumRuns'].iloc[0]}"


# ----------------------------------------------------------------------------


def test_no_args_vs_with_args_dataframe_comparison() -> None:
    """Compare dataframe structure for functions with and without arguments."""

    # Test function with no args
    @nsight.analyze.kernel(output="quiet")
    def no_args() -> None:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        with nsight.annotate("test"):
            _ = a + b

    # Test function with args
    @nsight.analyze.kernel(configs=[(32,), (64,)], output="quiet")
    def with_args(size: int) -> None:
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        with nsight.annotate("test"):
            _ = a + b

    result_no_args = no_args()
    result_with_args = with_args()

    assert result_no_args is not None
    assert result_with_args is not None

    df_no_args = result_no_args.to_dataframe()
    df_with_args = result_with_args.to_dataframe()

    # No-args function should have 1 row (1 config)
    assert (
        len(df_no_args) == 1
    ), f"No-args function should have 1 row, got {len(df_no_args)}"

    # With-args function should have 2 rows (2 configs)
    assert (
        len(df_with_args) == 2
    ), f"With-args function should have 2 rows, got {len(df_with_args)}"

    assert (
        "size" in df_with_args.columns
    ), "With-args function should have 'size' column"

    # Verify the size values in the dataframe match the configs
    assert set(df_with_args["size"].values) == {32, 64}


# ----------------------------------------------------------------------------


def test_no_args_function_with_derive_metric() -> None:
    """Test that derive_metric works with functions that have no arguments."""

    # Define a derive_metric function that only takes the metric value
    # (no config parameters since the function has no args)
    def custom_metric(time_ns: float) -> float:
        """Convert time to arbitrary custom metric."""
        return time_ns / 1e6  # Convert to milliseconds

    @nsight.analyze.kernel(runs=2, output="quiet", derive_metric=custom_metric)
    def no_args_with_transform() -> None:
        a = torch.randn(128, 128, device="cuda")
        b = torch.randn(128, 128, device="cuda")
        with nsight.annotate("test"):
            _ = a @ b

    result = no_args_with_transform()

    assert result is not None, "ProfileResults should be returned"
    df = result.to_dataframe()

    # Should have exactly 1 row
    assert len(df) == 1, f"Expected 1 row in dataframe, got {len(df)}"

    # Verify the transformation was applied
    assert (
        df["Transformed"].iloc[0] == "custom_metric"
    ), f"Expected 'custom_metric' in Transformed column, got {df['Transformed'].iloc[0]}"

    # Verify the value is positive (transformed metric should still be positive)
    assert df["AvgValue"].iloc[0] > 0, "Expected positive transformed metric value"

    # Verify runs parameter was respected
    assert df["NumRuns"].iloc[0] == 2, f"Expected 2 runs, got {df['NumRuns'].iloc[0]}"


# ----------------------------------------------------------------------------


# ============================================================================
# Parameter validation tests
# ============================================================================


@nsight.analyze.kernel(configs=[(1,), (2,)])
def function_with_default_parameter(x: int, y: Any = None) -> None:
    a = torch.randn(x, x, device="cuda")
    b = torch.randn(x, x, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_function_with_default_parameter() -> None:
    """Test that calling function with defaults without providing all args raises error."""
    with pytest.raises(exceptions.ProfilerException):
        function_with_default_parameter()


# ============================================================================
# Kernel execution tests
# ============================================================================

configs = [(i,) for i in range(5)]


@nsight.analyze.plot()
@nsight.analyze.kernel(configs=configs, runs=7, output="verbose")
def simple(x: int) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_simple() -> None:
    """Test basic kernel execution with multiple configurations."""
    simple()


# ----------------------------------------------------------------------------
# Conditional execution tests
# ----------------------------------------------------------------------------


@nsight.analyze.plot()
@nsight.analyze.kernel(configs=configs, runs=7, output="quiet")
def different_kernels(x: int) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        if x % 2 == 0:
            _ = a - b
        else:
            _ = a + b


def test_different_kernels() -> None:
    """Test kernel with conditional execution paths."""
    different_kernels()


# ----------------------------------------------------------------------------
# Multiple kernels per run tests
# ----------------------------------------------------------------------------


@nsight.analyze.plot()
@nsight.analyze.kernel(
    configs=configs,
    runs=7,
    combine_kernel_metrics=lambda x, y: x + y,
)
def multiple_kernels_per_run(x: int) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a @ b
        _ = a @ b
        _ = a @ b


def test_multiple_kernels_per_run() -> None:
    """Test kernel that launches multiple operations per execution."""
    multiple_kernels_per_run()


# ----------------------------------------------------------------------------
# Varying kernels per run tests (currently unsupported)
# ----------------------------------------------------------------------------


@nsight.analyze.plot()
@nsight.analyze.kernel(
    configs=((False,), (True,)),
    runs=3,
    combine_kernel_metrics=lambda x, y: x + y,
)
def varying_multiple_kernels_per_run(flag: bool) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b
        if flag:
            _ = a + b


@pytest.mark.skip("don't yet support varying number of kernels per run")  # type: ignore[misc]
def test_varying_multiple_kernels_per_run() -> None:
    """Test kernel with varying number of operations per run (currently unsupported)."""
    varying_multiple_kernels_per_run()


@nsight.analyze.kernel(
    configs=(
        (1,),
        (2,),
        (3,),
    ),
    runs=3,
    normalize_against="annotation1",
)
def normalize_against(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("annotation1"):
        _ = a + b

    with nsight.annotate("annotation2"):
        _ = a - b


def test_parameter_normalize_against() -> None:
    profile_output = normalize_against()
    if profile_output is not None:
        df = profile_output.to_dataframe()

        # Basic validation for normalize_against: AvgValue for the annotation being used as normalization factor should be 1
        assert (df.loc[df["Annotation"] == "annotation1", "AvgValue"] == 1).all()


# ============================================================================
# Output prefix tests
# ============================================================================


@nsight.analyze.kernel(
    configs=[(32,)],
    runs=1,
    output="quiet",
    output_prefix="/tmp/test_output_prefix/test_prefix_",
)
def output_prefix_func(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_parameter_output_prefix() -> None:
    """Test that output_prefix creates directories and files with correct prefix."""
    output_dir = "/tmp/test_output_prefix"
    try:
        result = output_prefix_func()

        if result is not None:
            expected_files = [
                "/tmp/test_output_prefix/test_prefix_ncu-output-output_prefix_func-0.ncu-rep",
                "/tmp/test_output_prefix/test_prefix_ncu-output-output_prefix_func-0.log",
            ]

            for file_path in expected_files:
                assert os.path.exists(
                    file_path
                ), f"Expected file not found: {file_path}"
    finally:
        # Cleanup after test
        if "NSPY_NCU_PROFILE" not in os.environ:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


# ----------------------------------------------------------------------------


@pytest.mark.parametrize("output_csv", [True, False])  # type: ignore[misc]
def test_parameter_output_csv(output_csv: bool) -> None:
    """Test the output_csv parameter to control CSV file generation."""
    output_dir = "/tmp/test_output_csv/"

    try:

        @nsight.analyze.kernel(
            configs=[(42, 23)],
            output_prefix=f"{output_dir}test_",
            output_csv=output_csv,
        )
        def output_csv_func(x: int, y: int) -> None:
            _simple_kernel_impl(x, y, annotation=f"output_csv={output_csv}")

        # Run the profiling
        profile_output = output_csv_func()

        # Verify that ProfileResults is returned (even if CSV is not dumped)
        if "NSPY_NCU_PROFILE" not in os.environ:

            # Check for CSV files based on output_csv value
            csv_files = [
                f"{output_dir}test_profiled_data-output_csv_func-0.csv",
                f"{output_dir}test_processed_data-output_csv_func-0.csv",
            ]

            for file_path in csv_files:
                if output_csv:
                    assert os.path.exists(
                        file_path
                    ), f"CSV file should exist when output_csv=True: {file_path}"
                else:
                    assert not os.path.exists(
                        file_path
                    ), f"CSV file should not exist when output_csv=False: {file_path}"

            # NCU report files should always exist regardless of output_csv
            ncu_files = [
                f"{output_dir}test_ncu-output-output_csv_func-0.ncu-rep",
                f"{output_dir}test_ncu-output-output_csv_func-0.log",
            ]

            for file_path in ncu_files:
                assert os.path.exists(
                    file_path
                ), f"NCU file should always exist: {file_path}"
    finally:
        # Cleanup after test
        if "NSPY_NCU_PROFILE" not in os.environ:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
