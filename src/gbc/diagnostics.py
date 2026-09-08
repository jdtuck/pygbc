"""Environment diagnostics — mainly for the macOS OpenMP crash.

Run it directly::

    python -m gbc.diagnose

or from Python::

    import gbc; gbc.diagnose()
"""

from __future__ import annotations

import os
import platform
import sys
from collections import defaultdict

__all__ = ["diagnose"]


def _accelerators() -> dict:
    """What compute devices this PyTorch build can actually reach."""
    info = {"cuda": False, "cuda_devices": [], "mps_built": False,
            "mps_available": False, "resolved": "cpu", "torch_threads": None}
    try:
        import torch
    except ImportError:  # pragma: no cover
        return info

    info["cuda"] = torch.cuda.is_available()
    if info["cuda"]:
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        info["mps_built"] = bool(mps.is_built())
        info["mps_available"] = bool(mps.is_available())
    info["torch_threads"] = torch.get_num_threads()

    from .iqn import resolve_device
    info["resolved"] = str(resolve_device())
    return info


def _openmp_runtimes():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    return [d for d in threadpool_info() if d.get("user_api") == "openmp"]


def diagnose(verbose: bool = True) -> dict:
    """Report library versions and check for duplicate OpenMP runtimes.

    Two OpenMP runtimes in one process (typically PyTorch's ``libomp``
    alongside a conda/MKL ``libiomp5`` pulled in by NumPy or scikit-learn)
    can segfault scikit-learn's k-means — which
    :class:`~sklearn.mixture.GaussianMixture`, and therefore
    :class:`~gbc.AugGBCRegressor`, uses to initialize.

    Args:
        verbose: print a human-readable report as well as returning it.

    Returns:
        A dict with ``python``, ``platform``, ``versions``, ``openmp``
        (list of loaded runtimes), ``duplicate_openmp`` (bool) and
        ``advice`` (list of suggested fixes, empty when nothing looks wrong).
    """
    versions = {}
    for name in ("numpy", "scipy", "sklearn", "torch", "threadpoolctl"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as exc:  # pragma: no cover
            versions[name] = f"<not importable: {exc}>"

    accel = _accelerators()
    runtimes = _openmp_runtimes()
    advice = []
    duplicate = False

    if not accel["cuda"] and accel["mps_available"]:
        advice.append(
            "An Apple GPU (MPS) is present but gbc will not use it: the "
            "automatic device choice is CUDA-or-CPU, so device=None resolves "
            "to CPU here. Pass device='mps' explicitly to try it. Check what "
            "a fitted model actually used with model.device_.")

    families = []
    if runtimes is None:  # pragma: no cover
        advice.append("threadpoolctl is not installed; cannot inspect "
                      "OpenMP runtimes. pip install threadpoolctl")
    else:
        by_family = defaultdict(list)
        for d in runtimes:
            by_family[d.get("prefix") or "?"].append(
                d.get("filepath") or "?")
        families = sorted(by_family)
        # Two copies of the SAME runtime (e.g. two libgomp.so) is common and
        # usually harmless. Two DIFFERENT runtimes (libomp + libiomp5, the
        # PyTorch + MKL combination on macOS) is the dangerous case.
        duplicate = len(by_family) > 1
        if duplicate:
            advice += [
                f"Two different OpenMP runtimes are loaded: "
                f"{', '.join(families)}. This is the usual cause of "
                "segfaults in scikit-learn's k-means when PyTorch is also "
                "imported (typically macOS + conda).",
                "Try, in order: (1) AugGBCRegressor already defaults to "
                "cluster_threads=1, which avoids the parallel region; "
                "(2) set OMP_NUM_THREADS=1 before starting Python; "
                "(3) use AugGBCRegressor(init_params='random_from_data') to "
                "skip k-means entirely; (4) rebuild the environment so only "
                "one OpenMP runtime is present, e.g. `conda install nomkl` "
                "or installing numpy/scipy/scikit-learn from PyPI wheels "
                "instead of the MKL conda builds.",
            ]
        elif any(len(v) > 1 for v in by_family.values()):
            advice.append(
                "Several copies of the same OpenMP runtime are loaded. This "
                "is common and usually harmless; if you are seeing "
                "segfaults anyway, follow the same steps as for mismatched "
                "runtimes (OMP_NUM_THREADS=1, or "
                "init_params='random_from_data').")

    report = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": versions,
        "accelerators": accel,
        "openmp": runtimes,
        "openmp_families": families,
        "duplicate_openmp": duplicate,
        "env": {k: os.environ[k] for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_DUPLICATE_LIB_OK")
                if k in os.environ},
        "advice": advice,
    }

    if verbose:
        print(f"python      {report['python']}")
        print(f"platform    {report['platform']}")
        for k, v in versions.items():
            print(f"{k:<12}{v}")
        if report["env"]:
            print("env         " + ", ".join(f"{k}={v}" for k, v
                                             in report["env"].items()))

        print("\nCompute")
        print(f"  CUDA available   {accel['cuda']}"
              + (f"  ({', '.join(accel['cuda_devices'])})"
                 if accel["cuda_devices"] else ""))
        print(f"  MPS (Apple GPU)  built={accel['mps_built']} "
              f"available={accel['mps_available']}")
        print(f"  torch threads    {accel['torch_threads']}")
        print(f"  device=None ->   {accel['resolved']}"
              + ("   <-- GPU NOT in use"
                 if accel["resolved"] == "cpu" else "   <-- GPU in use"))
        print(f"\nOpenMP runtimes loaded: "
              f"{len(runtimes) if runtimes is not None else '?'}")
        for d in runtimes or []:
            print(f"  {d.get('filepath', d.get('prefix'))}  "
                  f"threads={d.get('num_threads')}")
        if duplicate:
            print("\n*** Mismatched OpenMP runtimes detected "
                  f"({', '.join(families)}) ***")
        elif runtimes is not None:
            print("\nOpenMP runtimes look consistent.")
        for line in advice:
            print(f"\n- {line}")

    return report


def main() -> None:  # pragma: no cover
    # Import in the order that provokes the conflict, so the report reflects
    # what a real gbc session loads.
    import torch  # noqa: F401
    import sklearn.mixture  # noqa: F401

    diagnose()


if __name__ == "__main__":  # pragma: no cover
    main()
