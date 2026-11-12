# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`socutils` is a set of Python modules that extend [PySCF](https://pyscf.org)
with relativistic, two-/four-component quantum chemistry — primarily spin-orbit
coupling via the X2C atomic-mean-field (X2CAMF) Hamiltonian. The repo root **is**
the `socutils` Python package (the top-level `__init__.py` is the package init),
so code imports it as `from socutils.scf import spinor_hf`, etc.

To make it importable, the *parent* of this directory must be on `PYTHONPATH`
**and the directory must be named `socutils`** (the checkout is `socutils-git`,
so collaborators symlink/clone it to `socutils`). There is no `setup.py` at the
root; it is used in-tree, not pip-installed.

The recommended way to make it importable is a symlink wrapper: a directory
named `socutils` pointing at this checkout, with its parent on `PYTHONPATH`
(e.g. `socutils-git-env/socutils -> socutils-git`). Edit the modules at the repo
root (`scf/`, `cc/`, `x2camf_c/`, `lib/`, …) — the checkout root *is* the
package. (A non-git nested `socutils/` copy from clone day once shadowed imports
here; it has been removed.)

## Building native libraries

The C/C++ libraries must be compiled before the features that use them, by
**one** top-level CMake under `lib/` (`lib/CMakeLists.txt`), modelled on
`pyscf/lib`: it centralises the common config (BLAS/LAPACK, OpenMP, flags,
`$ORIGIN` RPATH) and `add_subdirectory`s the in-tree libraries `lib/ccsdt`,
`lib/ao2mo`, `lib/vhf`, `lib/dft`, `lib/pbcdf`, `lib/thc`, `lib/qdhf` (thin
subdir CMakeLists, like `pyscf/lib/vhf`).
`lib/ao2mo` (spinor AO2MO e1/e2 kernels, `nrr_fast.py`) reuses C already
compiled into an *installed* pyscf (`libao2mo`/`libnp_helper`/`libcvhf`/
`libcint`), located via the configuring Python interpreter.
`lib/dft` (`libsocdft.so`) holds the OpenMP XC kernels: the multi-collinear
kernel of `dft/noncollinear.py`, the fused scalar / two-component Vxc, f_xc and
XC-gradient passes (`vxc_fast.py`, `vxc_2c.py`, LDA/GGA/meta-GGA), the RKS XC
Hessian grid pass (`xc_hess.c`, `hessian/rhf_fast.py`), COSX (`scf/cosx.py`) and
the grid response; they fall back to numpy / pyscf when not built.
`lib/vhf` (`libsocvhf.so`: prescreen functions that skip one-centre shell
quartets, used by `somf/x2cmp.py`; the unique-quartet Hessian drivers
`hess2e.c` (2e second derivatives), `fock1.c` (first-order J/K of all atoms),
`jk_multi.c` (direct J/K of many densities, `scf/jk_multi.py`), `ri_j_deriv.c`
(three-centre derivative contractions of the density-fitted J Hessian), `md_j.c`
(McMurchie-Davidson J engine on decontracted primitives: exact J gradient and J
matrices, `scf/md_j.py`, whose `apply(mf)` also routes J + K to `jk_multi` with
incremental Fock builds), `rij_direct.c` (direct RI-J without the three-centre
tensor: MD on primitive Hermite pair densities, auxiliary primitives as pure Hermite
Gaussians, far-field auxiliary multipoles, SIMD lane batches; `scf/rij_direct.py`,
used by `md_j`'s RIJ when L^-1 (P|ij) does not fit, `md_j.RIJ_DIRECT`)) links the
installed pyscf's `libcvhf` the same way; without it `.x2cmp()` falls back to
the full molecular J/K build and the Hessians to pyscf's integral passes.
These `.so` are emitted **flat into `lib/`**.

The **x2camf C library** (C99 translation of the C++ X2CAMF, Apache-2.0,
repo github.com/xubwa/x2camf-c) and **zquatev** (BSD-2, sunqm/zquatev; pinned to
upstream commit 4b397dd = PR #2, merged 2026-11-09, which adds BLAS_LIBRARIES and
install() to its CMake and computes zdotc in C++, so no CBLAS/cblas.h is needed)
are separate repositories, built with `ExternalProject` (like pyscf's
libcint/libxc) with the same BLAS into `lib/deps/lib/` (`libx2camf_c.so`,
`libzquatev.so`). Cache variables `X2CAMF_GIT`/`X2CAMF_VERSION`,
`ZQUATEV_GIT`/`ZQUATEV_VERSION`, or `X2CAMF_SOURCE_DIR`/`ZQUATEV_SOURCE_DIR`
for a local checkout; `BUILD_X2CAMF`/`BUILD_ZQUATEV=OFF` to skip.
All libraries are optional — two-component code that doesn't call them runs
without.

```bash
make            # build all -> lib/lib{ccsdt_clib,nrr_opt,socvhf,socdft,socpbcdf,socthc,qdhf}.so, lib/deps/lib/lib{x2camf_c,zquatev}.so
make test       # build + no-dependency x2camf smoke test
make compare    # build + A/B compare C backend vs C++ reference
                #   (needs X2CAMF_CPP_LIBRARY=/path/to/libx2camf*.so)
make clean
# pass BLAS/LAPACK choice through, e.g.:
make CMAKE_ARGS=-DBLA_VENDOR=OpenBLAS      # or -DBLAS_LIBRARIES=mkl_rt
# target instruction set (default: generic); on Rockfish cascadelake (AVX-512)
# runs on every Intel CPU partition and is ~12 % faster on the J / K kernels:
make CMAKE_ARGS=-DSOCUTILS_MARCH=cascadelake
# equivalently, drive the lib/ CMake directly: cd lib && ./build.sh [-D...]
```

The ctypes loaders find their library in `lib/` or `lib/deps/lib/` (and honour a
`SOCUTILS_*_LIBRARY` env override): `lib/__init__.py:load_library`,
`x2camf_c/x2camf/_c_backend.py`, `cc/_ccsdt_clib.py`, `lib/zquatev/__init__.py`,
`lib/ao2mo/nrr_fast.py`.

## Tests

There is no central pytest config. Tests are standalone scripts/`unittest`:

```bash
python gw/test/test_spinor_gw.py        # spinor G0W0 (unittest)
python dft/test/test_noncollinear.py    # ncol / mcol DFT
python gw/test/test_spinor_rpa.py
python x2camf_c/tests/test_smoke.py     # also: make test
python x2camf_c/tests/test_against_cpp.py
```

The `examples/` directory (numbered `00-…` upward) doubles as runnable
integration checks for the SCF/MCSCF/CC entry points.

## Lint / style

`flake8` (config `.flake8`, `max-line-length = 120`, many E/F codes ignored) and
`yapf` (`.style.yapf`, `COLUMN_LIMIT = 119`). The only CI is `.github/workflows/docs.yml`,
which just builds the Sphinx docs — it does not run tests or lint.

## Architecture

The central abstraction mirrors PySCF: where PySCF turns a mean field
scalar-relativistic with `scf.RHF(mol).x2c()`, socutils turns a spinor/GHF mean
field into a two-component, spin-orbit-coupled one with `.x2camf()` / `.x2cmp()`.
These shortcuts attach an AMF X2C spin-orbit Hamiltonian (default: Gaunt + Breit
two-electron corrections on) and only then require the `x2camf` integral backend.

- **`scf/`** — the SCF entry points.
  - `spinor_hf.py`: `SCF`/`UHF` run in a two-component, *j-adapted spinor*
    basis; `RHF` is the Kramers-restricted variant (uses bundled zquatev).
    Old names `SpinorSCF`, `JHF`, `KRHF` are aliases. All carry `.x2camf()`/`.x2cmp()`.
  - `double_group_symm.py`: double-group symmetry (mixin `DoubleGroupSymm`;
    `SymAdaptedUHF/RHF/DHF/RDHF`), on the AO layer `symm/` (`double_group.py`,
    `spinor_basis.py`); molecule in its symmetry frame (`symm.to_symm_frame`).
    Design note: `DOUBLE_GROUP_DESIGN.md`.
  - `ghf.py`: `GHF` is the spin-orbital analogue (same shortcuts, spin-orbital
    basis instead of spinor); spinor and GHF energies agree to numerical precision.
  - `x2camf_hf.py`: `X2CAMF_*` convenience classes + `x2camf_ghf(scf.GHF(mol))`.
  - `frac_dhf.py`: four-component Dirac-HF; `spinor_hf_df.py`: density fitting.
  - `rik.py`: RI-JK (J and K from any three-index factor through `with_df.loop` /
    `iter_factor`: pyscf DF in core or HDF5, socutils CD); `rik.apply(mf)` (also via
    `md_j.apply` for full DF-JK), screened `cholesky_eri` / `cholesky_eri_h5` builders
    (`lib/vhf/int3c_s2.c`), reusable `half_transform` / `mo_transform`.

- **`x2camf_c/x2camf/`** — the bundled `x2camf` Python package: a dispatcher
  with **two interchangeable backends** behind one API (`amfi`/`atm_integrals`/
  `pcc_K`) — the pure-C reimplementation (ctypes onto `lib/deps/lib/libx2camf_c.so`) and
  the upstream pybind11 C++ reference. `backend.py` selects: env
  `X2CAMF_BACKEND=c|cpp`, `backend.set_backend()`, or `with backend.using('cpp'):`.
  Default `c` if built, else falls back to `cpp`. The **C sources** live in
  their own repository (github.com/xubwa/x2camf-c, built into `lib/deps/lib` by
  ExternalProject): a 1:1 translation of the upstream C++ (`x2c_int_sph*.c` ↔
  integrals, `x2c_dhf_sph*.c` ↔ Dirac-HF SCF; see its
  `csrc/TRANSLATION_CONVENTIONS.md`).

- **The root `__init__.py`** decides whether the bundled `x2camf` dispatcher
  (`x2camf_c/`) is put on `sys.path` as the top-level `x2camf`. Controlled by
  `SOCUTILS_X2CAMF`: `bundled` (default, even if an external `x2camf` is
  installed), `auto` (bundled only if no external `x2camf` is installed), or
  `external`. The external package's Python layer places the atomic matrices by
  position and is wrong for segmented / unsorted basis sets, hence the default.
  In PySCF mainline there will be no option: only the C backend.

- **`somf/`** — spin-orbit mean-field machinery (`amf.py`, `snso.py`,
  `somf_pt.py`, `x2cmp.py`) underlying the `.x2camf()`/`.x2cmp()` shortcuts.

- **`dft/`** — two-component Kohn-Sham: `dft.py` (`SpinorDFT`/`KRDFT`, spinor
  basis; `numint_2c.SpinorNumInt2C` rotates to the scalar-AO x spin basis for the
  grid work), `gks.py` (`GKS` on the socutils GHF).  `noncollinear.py` supplies
  the `collinear` schemes beyond pyscf: `'ncol'` (locally collinear, LDA/GGA/MGGA
  Vxc + fxc) and `'mcol'` (native multi-collinear, no mcfun; LDA in closed form,
  GGA/MGGA on a half Lebedev sphere via `lib/libsocdft.so`).
- **`pbc/`** — Gamma-point periodic two-component SCF on cells: `spinor_scf.py`
  (`periodic(mf, rsdf)`, exxdiv / G = 0 conventions), `rsdf.py` (range-separated
  density fitting, `lr='gspace'`, C kernels `lib/pbcdf`), and the cderi-free THC
  route-1 J/K `thc.py` (`THCJK`; with `thc_gapw.py` GAPW J, `thc_local.py` local CX
  fit, OpenMP kernels `lib/thc` -> `libsocthc.so`), MPI SPMD via mpi4py
  (`jk.comm`; `examples/14-periodic_dopant_casci.py --jk thc` under
  `mpirun ... python -m mpi4py`). Status, results and run paths: `pbc/HANDOFF_WNSM.md`.

- **`cc/`** — relativistic coupled cluster on a spinor reference: `zccsd.py`,
  `zccsd_direct.py`, `zccsdt.py` (triples), `gccsd_t.py`, `eom_zccsd_direct.py`,
  `chol_zccsd*.py` (Cholesky-decomposed ERIs). C kernels live in `lib/ccsdt`
  (loaded from `lib/libccsdt_clib.so` via `cc/_ccsdt_clib.py`).

- **`mcscf/`** (`zcasbase`, `zcasci`-style CASCI/CASSCF, `zmcscf`, superCI),
  **`fci/`** (`zfci`), **`gw/`** (spinor G0W0 / RPA), **`tdscf/`**,
  **`prop/`** (properties: HFC, EFG, g-factor, contact density, effective field),
  **`grad/`** (analytic gradients; `grad/rhf_fast.py`: fast RHF / RKS gradients, the
  MD J engine `lib/vhf/md_j.c` for exact J, the order-1 `hess2e_cart` kernel for K, the contracted XC part; `grad/rik_grad.py`: the RI-K
  gradient with auxiliary response, `lib/vhf/ri_k_grad.c`), **`cd/`** (Cholesky decomposition),
  **`lo/`**, **`tools/`** (`spinor2sph`, `fragmo_guess`) — all built on the
  spinor/GHF SCF objects above.

- **`lib/`** — the pyscf-style home for the C/C++ libraries: thin in-tree
  subdirs driven by `lib/CMakeLists.txt` (`.so` flat into `lib/`,
  `lib/__init__.py:load_library`), plus x2camf (C) and zquatev built by
  ExternalProject into `lib/deps/lib`. `lib/zquatev/__init__.py` is the ctypes
  loader of Toru Shiozaki's quaternion eigensolver for Kramers-restricted SCF.

## Environment variables

- `SOCUTILS_X2CAMF` = `bundled` (default) | `auto` | `external` — whether the
  in-tree `x2camf_c` dispatcher is exposed as the top-level `x2camf`.
- `X2CAMF_BACKEND` = `c` (default if built) | `cpp` — which x2camf backend runs.
- `X2CAMF_CPP_LIBRARY` — path to upstream `libx2camf*.so` for `make compare` / cpp backend.
- `SOCUTILS_ZQUATEV_LIBRARY` — explicit path to `libzquatev.so`.
- `SOCUTILS_SOCDFT_LIBRARY` — explicit path to `libsocdft.so`.
