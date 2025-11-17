# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.util
from collections.abc import Callable
from typing import Any

import nvtx

import nsight.utils as utils
from nsight.exceptions import CUDA_CORE_UNAVAILABLE_MSG


class annotate(nvtx.annotate):  # type: ignore[misc]
    """
    A decorator/context-manager hybrid for marking profiling regions.
    The encapsulated code will be profiled and associated with an NVTX
    range of the given annotate name.

    Example usage::

        # as context manager
        with nsight.annotate("name"):
            # your kernel launch here

        # as decorator
        @nsight.annotate("name")
        def your_kernel_launcher(...):
            ...

    Args:
        name: Name of the annotation to be used for profiling.
        ignore_failures: Flag indicating whether to ignore
            failures in the annotate context. If set to ``True``, any exceptions
            raised within the context will be ignored, and the profiling will
            continue. The measured metric for this run will be set to NaN.
            Default: ``False``

    Note:
        All annotations are created under the NVTX domain ``"nsight-python"``.
        This domain is used internally to filter and identify Nsight Python
        annotations in profiling tools.

    Note:
        Nested annotations are currently not supported. However, since each
        annotation is expected to contain a single kernel launch by default,
        nested annotations should not be necessary in typical usage scenarios.

    """

    def __init__(self, name: str, ignore_failures: bool = False):
        self.name = name
        self.ignore_failures = ignore_failures

        # Check if cuda-core is available when ignore_failures is True
        if ignore_failures and not utils.CUDA_CORE_AVAILABLE:
            raise ImportError(CUDA_CORE_UNAVAILABLE_MSG)

        super().__init__(name, domain=utils.NVTX_DOMAIN)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            if exc_type and self.ignore_failures:
                utils.launch_dummy_kernel_module()
        finally:
            super().__exit__(exc_type, exc_value, traceback)

        if exc_type and not self.ignore_failures:
            return False  # propagate the exception

        return True

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapped
