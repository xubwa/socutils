This repository adds utility for PySCF to include SOMF corrections within spinor style and GHF style calculations.
To do a spinor (j-adapted) style calculation, build a spinor SCF and attach an
X2CAMF spin-orbit Hamiltonian with the `.x2camf()` shortcut (the spinor analogue
of PySCF's `scf.RHF(mol).x2c()`):
```python
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom=[["O", (0., 0., 0.)],
                  ["H", (0., -0.757, 0.587)],
                  ["H", (0.,  0.757, 0.587)]],
            basis='ccpvdz', verbose=4)

mf = spinor_hf.SCF(mol).x2camf()          # Gaunt + Breit on by default
e_spinor = mf.kernel()

# turn the Gaunt/Breit two-electron SOC corrections off:
e_dc = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False).kernel()
```
Note: the C/C++ libraries these features need are **bundled** with socutils
(under `socutils/lib`) -- no external packages to install. They are compiled
once with a single `make` at the repo root (needs a BLAS/LAPACK):
```
make            # builds libx2camf_c (X2CAMF SOC integrals),
                #        libzquatev   (Kramers-restricted spinor SCF), and
                #        libccsdt_clib (spinor CCSDT kernels) into socutils/lib
```
- **x2camf** (the SOC integrals) ships as a pure-C reimplementation exposed as
  `import x2camf`; an external upstream
  [warlocat/x2camf](https://github.com/warlocat/x2camf) pybind11 build is
  *optional* (only for A/B comparison, selected via `X2CAMF_BACKEND` /
  `SOCUTILS_X2CAMF`).
- **zquatev** (Kramers-restricted spinor SCF) is the bundled quaternion
  eigensolver -- the former `xubwa/zquatev` pip package is no longer needed.

See `docs/source/install.rst` for details (BLAS/LAPACK vendor selection,
environment overrides).
To do a GHF (spin-orbital) style calculation, use `ghf.GHF` with the same
`.x2camf()` shortcut; it runs in a spin-orbital rather than a spinor basis and
agrees with the spinor result to numerical precision:
```python
from pyscf import gto
from socutils.scf import ghf

mol = gto.M(atom=[["O", (0., 0., 0.)],
                  ["H", (0., -0.757, 0.587)],
                  ["H", (0.,  0.757, 0.587)]],
            basis='ccpvdz', verbose=4)

gmf = ghf.GHF(mol).x2camf()               # Gaunt + Breit on by default
e_ghf = gmf.kernel()
```
