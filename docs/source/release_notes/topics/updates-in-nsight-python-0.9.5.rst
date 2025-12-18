.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Updates in Nsight Python 0.9.5
==============================

Fixes
-----

- `Github Issue #2 <https://github.com/NVIDIA/nsight-python/issues/2>`_:
  Function must have arguments otherwise you get an error

- `Github Issue #4 <https://github.com/NVIDIA/nsight-python/issues/4>`_:
  Allow scalar configs if function only takes one argument

Enhancements
------------

- Added support for **non-sized iterables**.
- Added support for collecting **multiple metrics** using ``nsight.analyze.kernel`` (`GitHub issue #8 <https://github.com/NVIDIA/nsight-python/issues/8>`_).  
  ``nsight.analyze.plot`` continues to support only a single metric and will raise an error if multiple metrics are passed.



Other Changes
-------------

- Warn when a decorated function declares a **return type**.
- Raise an error when **annotation names are duplicated**.
- Disallow **nested annotations**.
