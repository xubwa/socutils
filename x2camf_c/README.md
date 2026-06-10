# x2camf-c

A plain-C (`extern "C"`) interface to the
[X2CAMF](https://github.com/Warlocat/x2camf) code, together with a Python
binding based on **ctypes** instead of pybind11.

The upstream X2CAMF C++ core is reused unchanged as a git submodule
(`external/x2camf`); this repository adds

- `c_api/` — a C ABI layer (`x2camf_c.h` / `x2camf_c.cpp`) exposing the
  three entry points of the pybind11 module (`amfi`, `atm_integrals`,
  `pcc_K`) with only `int`/`double` scalars and arrays. The resulting
  shared library `libx2camf_c.so` has **no Python or pybind11 dependency**
  and can be called from any language with a C FFI (C, Fortran, Julia,
  Rust, ...).
- `x2camf/` — a Python package that loads the library via `ctypes` and is
  a **drop-in replacement** for the pybind11-based `x2camf` package: it
  provides the same `x2camf.amfi(x2cobj, ...)`, `x2camf.x2camf.pcc_k`,
  `x2camf.x2camf.construct_molecular_matrix` and `x2camf.libx2camf.*`
  APIs, so downstream code such as
  [socutils](https://github.com/xubwa/socutils) works without any change.

Because the build no longer embeds CPython, one compiled library works for
every Python version (no more rebuilds when switching interpreters), and
the same binary can serve non-Python consumers.

## Installation

```bash
git clone --recurse-submodules <this-repo>
# or, after a plain clone:
git submodule update --init external/x2camf
cd external/x2camf && git submodule update --init eigen && cd ../..

pip install .
```

Requirements: a C++11 compiler, CMake >= 3.9, numpy. OpenMP is used when
available. `pyscf` is only needed for the high-level molecular interface
(`x2camf.amfi(x2cobj, ...)`); the low-level `x2camf.libx2camf` works with
numpy alone.

### Library only (no Python)

```bash
mkdir build && cd build
cmake .. && make -j
# -> libx2camf_c.so, header in c_api/x2camf_c.h
```

If the library is not installed inside the Python package directory, point
the binding at it with the environment variable

```bash
export X2CAMF_C_LIBRARY=/path/to/libx2camf_c.so
```

## Usage

### From Python (identical to the pybind11 package)

```python
from pyscf import gto
from pyscf.x2c import x2c
import x2camf

mol = gto.M(atom='S 0 0 0', basis='unc-cc-pvtz')
amf = x2camf.amfi(x2c.X2C(mol), with_gaunt=True, with_gauge=True)
```

Low level, numpy only:

```python
from x2camf import libx2camf
results = libx2camf.amfi(soc_int_flavor, atom_number, nshell, nbas,
                         print_level, shell, exp_a)
```

### From C

See `examples/amfi_from_c.c`:

```c
#include "x2camf_c.h"
int n2c = x2camf_n2c(nbas, shell);
double *amfi = malloc(n2c * n2c * sizeof(double));
int ierr = x2camf_amfi(flavor, atom_number, nshell, nbas, 0,
                       shell, exp_a, X2CAMF_SPEED_OF_LIGHT_DEFAULT, amfi);
```

All output matrices are written row-major (C order) into caller-allocated
buffers; two-component matrices are `n2c x n2c`, four-component ones
`2*n2c x 2*n2c` with `n2c = sum_i (4*l_i + 2)`. Functions return `0` on
success; `x2camf_error_message()` decodes error codes.

## Testing

```bash
python tests/test_smoke.py        # numpy only
```

The ctypes interface has been verified to produce bit-for-bit identical
matrices to the pybind11 module for `amfi` (Coulomb / Gaunt / gauge / 4c /
average-of-configuration flavors), `pcc_K` and all 13 matrices of
`atm_integrals` (including `spin_free=True`).

## Note: extracting to a standalone repository

This directory is staged inside socutils because the session could not
create a new GitHub repository. To publish it standalone:

```bash
# from a socutils checkout of this branch
git subtree split --prefix=x2camf_c -b x2camf-c-main
git push git@github.com:<you>/x2camf-c.git x2camf-c-main:main
# then, in the new repo:
git submodule update --init external/x2camf
cd external/x2camf && git submodule update --init eigen
```
