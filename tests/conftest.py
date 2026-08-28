"""Shared test configuration.

The environment variables below are set *before* NumPy, PyTorch or
scikit-learn are imported, because OpenMP reads them at runtime
initialization. On macOS a conda PyTorch and a conda scikit-learn frequently
load two different OpenMP runtimes into one process, and scikit-learn's
k-means then segfaults inside its parallel region — which takes the whole
pytest process down with it. Running the native pools single-threaded avoids
that region entirely.

This affects only the test suite. Library behavior is controlled by
``cluster_y(threads=...)`` / ``AugGBCRegressor(cluster_threads=...)``, which
default to 1 for the same reason.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pytest  # noqa: E402


def pytest_report_header(config):
    from gbc.diagnostics import diagnose

    r = diagnose(verbose=False)
    line = (f"gbc: torch {r['versions']['torch']}, "
            f"sklearn {r['versions']['sklearn']}, "
            f"openmp {r['openmp_families'] or 'none'}")
    if r["duplicate_openmp"]:
        line += "  [mismatched OpenMP runtimes - see python -m gbc]"
    return line
