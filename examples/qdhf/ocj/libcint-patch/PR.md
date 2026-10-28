# Add int3c1e_ipvip1_r{2,4,6}_origk

Adds `<nabla i| r_k^{2m} |nabla j>` with the k-centred Gaussian, m = 1, 2, 3, i.e. the derivative
counterparts of `int3c1e_r{2,4,6}_origk` (nine components, index 3a+b for `<d_a i|...|d_b j>`),
in the same way `int3c1e_ipvip1` extends `int3c1e`.

## Changes

- `scripts/auto_intor.cl`: three entries in the `int3c1e.c` block
  `( nabla \, nabla \, r dot r)`, `( ... r dot r r dot r)`, `( ... r dot r r dot r r dot r)`
- `src/autocode/int3c1e.c`: regenerated; the existing functions are unchanged (code appended)
- `include/cint_funcs.h`: declarations, in the format of the neighbouring int3c1e entries
- `testsuite/test_int3c1e.py`: `run_fd_ket` -- central differences of
  `int3c1e_ip1_r{2,4,6}_origk` in the ket centre (j on atom 1, i and k elsewhere, so
  `-d/dR_j,b <d_a i|g|j> = <d_a i|g|d_b j>` exactly); also run on the existing
  `int3c1e_ipvip1` against `int3c1e_ip1` as a control.  All pass (thresholds as the
  existing `run_fd` checks: 1e-5, 2e-4 for r6).

## Motivation

Small-component blocks `<sigma.p i| V |sigma.p j> = sum_ab <d_a i|V|d_b j> sigma_a sigma_b` of
relativistic one-electron operators built from local potentials of the form
`r^{2m} exp(-zeta r^2)` about atomic centres: e.g. the Hartree potential of a spherical atomic
density expands exactly into `erf(sqrt(zeta) r)/r` (`int3c2e_ipvip1`) plus such Gaussians,
used for model-potential X2C Hamiltonians.  Without the new functions the same quantities need
the l+-1 derivative shells to be formed by the caller.

## Notes

- The `_spinor` variants are declared like those of the other int3c1e functions; as for them,
  the spinor driver for 3c1e is not implemented.
