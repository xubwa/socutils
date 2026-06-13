# Bundled zquatev (ctypes)

Toru Shiozaki's quaternionic eigensolver
([zquatev](https://github.com/qsimulate-open/zquatev)), vendored into socutils
and exposed through `ctypes` so that no external `zquatev` package or pybind11
is required.

Import it as:

```python
from socutils.lib import zquatev
e, v = zquatev.eigh(mat)                 # Hermitian quaternionic (A,-B*;B,A*)
e, c = zquatev.solve_KR_FCSCE(mol, f, s) # Kramers-restricted FC=SCE
```

This is a drop-in replacement for the external `zquatev` Python module (same
`eigh`, `geigh`, `solve_KR_FCSCE`, `check_kramers_structure`).

## Building

The solver is a small C++ library built with CMake (BLAS/LAPACK required):

```sh
./build.sh                                   # auto-detect BLAS/LAPACK
# or pass an explicit BLAS, e.g. OpenBLAS / MKL:
./build.sh -DBLAS_LIBRARIES=/path/to/libopenblas.so
```

The resulting `libzquatev.so` is placed in this directory, next to the ctypes
loader (`__init__.py`). At runtime the loader searches, in order:

1. `$SOCUTILS_ZQUATEV_LIBRARY`
2. `libzquatev.so` next to `__init__.py`
3. `ctypes.util.find_library("zquatev")`

## Licensing

- `csrc/` — verbatim ZQUATEV, **BSD-2-Clause**, © 2013 Toru Shiozaki (see
  `LICENSE` and the per-file headers). Reference: T. Shiozaki, *Mol. Phys.*
  **115**, 5 (2017).
- `capi/zquatev_capi.cc` and `__init__.py` — part of socutils, **Apache-2.0**.
