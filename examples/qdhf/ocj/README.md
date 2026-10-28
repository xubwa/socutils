# C kernel for the one-center (neutral-atom) J

`onecenter_j.c` assembles sum_K (mu nu | k_K) without the three-index tensor, screened per
shell pair and atom, OpenMP over shell pairs; `ocj.py` (`j_kernel`) drives it from the
per-atom potential terms of `../40-analytic_onecenter_j.py` (`potential_terms`).

Integrals (real, spherical; spinor matrices assembled once at the end with sparse
sph->spinor maps):

| block | erf part | local r^2m Gaussians | point nucleus |
|---|---|---|---|
| LL | int3c2e | int3c1e, int3c1e_r{2,4,6}_origk | int1e_rinv |
| SS | int3c2e_ipvip1 | int3c1e_ipvip1, **int3c1e_ipvip1_r{2,4,6}_origk** | int1e_iprinvip |

The bold ones are not in libcint: `libcint-patch/0001-*.patch` adds them (generator entries,
regenerated `src/autocode/int3c1e.c`, declarations, finite-difference tests; PR text in
`libcint-patch/PR.md`).  Without it, `j_kernel(..., std=True)` uses pyscf's own libcint
only and assembles those terms in the kernel from cartesian l+-1 shells
(`OCJ_3c_dshell_drv`), with identical results (~1.5x slower on small molecules).  Build:

    git clone https://github.com/sunqm/libcint && cd libcint
    git am .../libcint-patch/0001-Add-int3c1e_ipvip1_r-2-4-6-_origk.patch
    mkdir build && cd build && cmake .. && make
    gcc -O2 -fopenmp -fPIC -shared -o libocj.so onecenter_j.c -lm

`ocj.py` loads that libcint by absolute path (RTLD_LOCAL, next to pyscf's own libcint)
and passes function pointers to the kernel: set `OCJ_LIBCINT` (path of the new
libcint.so) and `OCJ_LIB` (libocj.so), default: next to `ocj.py`.

Each atom's exponents are split into groups of one decade; every erf group carries its
share of the nuclear charge (-sum E erfc(sqrt(z) r)/r, short-ranged) and is screened with
its own smallest exponent.  The screening bound ignores prefactors, so eps = 1e-20
(error 1e-11) is needed; 1e-13 gives 2e-4 in SS.

Timings (4 cores, eps 1e-20; against one scalar direct NR JK build in the same uncontracted
basis, `test_c60.py`, `test_gold.py` -> `test_gold.out`):

| system | n2c | NR JK | X2C decoupling | kernel J (patched libcint) | kernel J (std) | err 2c |
|---|---|---|---|---|---|---|
| C60, unc-cc-pVDZ | 3120 | 575 s | 238 s | 63 s (0.11x) | 75 s (0.13x) | 9e-12 |
| Au4, unc-dyall-v2z | 1632 | 48 s | 42 s | 11.5 s (0.24x) | | 1.8e-8 |
| Au8 (cube), unc-dyall-v2z | 3264 | 293 s | 292 s | 40 s (0.14x) | | 1.8e-8 |

The unscreened pure-Python analytic J (`../40-analytic_onecenter_j.py`) takes 164 s for C60.
