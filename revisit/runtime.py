"""Thread limits and determinism switches used for every reported run.

The reported numbers were produced with two CPU threads per process and deterministic cuDNN/cuBLAS
settings. Several runs shared one machine, so the limits also keep concurrent runs from contending.
"""
import os
import random

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def limit_threads(n: int = 2) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for var in THREAD_VARS:
        os.environ[var] = str(n)
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(n)


def seed_everything(seed: int = 42) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
