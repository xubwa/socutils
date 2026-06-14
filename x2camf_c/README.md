# x2camf-c

A **pure-C reimplementation** of the [X2CAMF](https://github.com/Warlocat/x2camf)
code, with a Python binding based on **ctypes**.

The original X2CAMF is C++ (Eigen + pybind11). This project is a faithful,
function-by-function C99 translation of its computational core, so the
result is plain C that links only against BLAS/LAPACK and libm — no C++
runtime, no Eigen, no pybind11. That makes it straightforward to
contribute upstream to projects that only accept C extensions (e.g. the
PySCF main line) and to call from any language with a C FFI.

## What is translated

The C sources in `csrc/` correspond 1:1 to the upstream C++ sources:

| C++ (`x2camf/src`) | C (`csrc`) | contents |
|---|---|---|
| Eigen `MatrixXd`, `SelfAdjointEigenSolver`, ... | `x2c_mat.{h,c}` | dense matrix layer over LAPACK (`dsyev`, `dgetrf/dgetri`, `dgesv`) |
| `general.cpp` | `x2c_general.{h,c}` | Wigner 3j/6j/9j, `factorial`, X2C transforms, `Rotate::unite_irrep*`, `elem_list` |
| `int_sph_basic.cpp` | `x2c_int_sph_basic.c` | `INT_SPH` ctor, auxiliary radial integrals |
| `int_sph.cpp` | `x2c_int_sph.c` | 1e integrals, Coulomb 2e (J/K) |
| `int_sph_gaunt.cpp` | `x2c_int_sph_gaunt.c` | Gaunt integrals (incl. spin-free) |
| `int_sph_gauge.cpp` | `x2c_int_sph_gauge.c` | gauge/Breit integrals |
| `dhf_sph.cpp` | `x2c_dhf_sph.c` | 4c/2c Dirac-HF SCF, DIIS, AMFI |
| `dhf_sph_ca.cpp` | `x2c_dhf_sph_ca.c` | average-of-configuration open-shell DHF |
| `dhf_sph_pcc.cpp` | `x2c_dhf_sph_pcc.c` | x2c2e picture-change correction |
| `executables/pyx2camf.cpp` | `c_api/x2camf_c.{h,c}` | the public C ABI (`x2camf_amfi`, `x2camf_atm_integrals`, `x2camf_pcc_k`) |

Members not reachable from the three entry points (GENBAS reader,
`basisGenerator`, `radialDensity`, `coreIonization`, CFOUR interface) are
intentionally not translated.

## Building

Requirements: a C99 compiler, BLAS + LAPACK, OpenMP (optional), numpy
(for the Python binding). `pyscf` is only needed for the high-level
molecular interface.

```bash
# direct build of the shared library
./build_c.sh                 # -> build_c/libx2camf_c.so

# or via CMake
mkdir build && cd build && cmake .. && make -j
```

BLAS/LAPACK selection follows the PySCF conventions. The default
auto-detects a BLAS and LAPACK. To use an optimized, threaded library
(which also parallelizes the dense linear algebra on top of the OpenMP
integral loops), either name it explicitly or pick a vendor:

```bash
cmake .. -DBLAS_LIBRARIES=mkl_rt          # Intel MKL single dynamic lib
cmake .. -DBLA_VENDOR=OpenBLAS            # OpenBLAS
cmake .. -DBLA_VENDOR=Intel10_64lp        # MKL (LP64) via FindBLAS
```

A user-supplied `BLAS_LIBRARIES` (e.g. `mkl_rt`) is assumed to also
provide LAPACK, so no separate reference LAPACK is linked. OpenMP is on by
default; disable with `-DENABLE_OPENMP=OFF`.

```bash
# or as a Python package
pip install .
```

If the library is not inside the Python package directory, point the
binding at it:

```bash
export X2CAMF_C_LIBRARY=/path/to/libx2camf_c.so
```

## Usage

The Python API is a drop-in replacement for the pybind11 `x2camf`
package, so downstream code such as
[socutils](https://github.com/xubwa/socutils) works unchanged:

```python
from pyscf import gto
from pyscf.x2c import x2c
import x2camf

mol = gto.M(atom='S 0 0 0', basis='unc-cc-pvtz')
amf = x2camf.amfi(x2c.X2C(mol), with_gaunt=True, with_gauge=True)
```

Low level (numpy only) and from C — see `tests/` and `examples/`.

All output matrices are row-major (C order); 2c matrices are `n2c x n2c`,
4c matrices `2*n2c x 2*n2c` with `n2c = sum_i (4*l_i + 2)`. Functions
return `0` on success; `x2camf_error_message()` decodes error codes.

## Validation

`tests/test_against_cpp.py` compares every entry point and flavor against
the original pybind11 module (built from upstream x2camf and importable as
`libx2camf`):

```bash
PYTHONPATH=/path/to/x2camf/build \
X2CAMF_C_LIBRARY=$PWD/build_c/libx2camf_c.so \
python tests/test_against_cpp.py          # O smoke test + heavy d/f atom
# stress an open f-shell (AOC) atom:
X2CAMF_TEST_Z=60 python tests/test_against_cpp.py
```

Verified agreement with the C++/Eigen implementation to better than
3e-9 (LAPACK-vs-Eigen round-off through the SCF) for O, Xe and Nd across
all flavors — Coulomb, Gaunt, gauge, 4c, average-of-configuration, PT,
PCC, SD-Gaunt — and all 13 `atm_integrals` matrices (spin-free and
spin-dependent). `tests/test_smoke.py` runs without pyscf or the C++
reference.
