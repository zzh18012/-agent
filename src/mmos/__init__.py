"""MathModelingOS: evidence-driven mathematical modeling automation."""

import os

# Small tabular runs do not benefit from BLAS thread pools, and multi-threaded
# buffers can fail to allocate on memory-constrained Windows hosts. Existing
# environment settings still win.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
del _variable

__version__ = "0.1.0"
