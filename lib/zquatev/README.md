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

## Local patch (`csrc/f77.h`)

One change was made to the vendored source: `zdotc` is now computed directly
in C++ instead of calling the Fortran BLAS `zdotc_`.  A `COMPLEX*16` Fortran
*function* returns its value through a platform/library-dependent ABI (a hidden
first pointer argument for gfortran-built reference LAPACK / OpenBLAS, vs.
return-by-register for MKL).  The upstream wrapper hard-coded the
hidden-pointer convention, so against OpenBLAS or reference LAPACK `zdotc`
silently returned garbage and corrupted the Householder reduction — producing
wrong eigenvalues (and `KRHF`/`RDHF` failures) on everything but MKL.
Computing the dot product in C++ removes the ABI dependency, so the solver is
correct against any BLAS (verified against OpenBLAS, reference LAPACK 3.9.1 and
3.12, and MKL).

## Licensing

- `csrc/` — ZQUATEV, **BSD-2-Clause**, © 2013 Toru Shiozaki (see `LICENSE`
  and the per-file headers), with the local `f77.h` patch noted above.
  Reference: T. Shiozaki, *Mol. Phys.* **115**, 5 (2017).
- `capi/zquatev_capi.cc` and `__init__.py` — part of socutils, **Apache-2.0**.
