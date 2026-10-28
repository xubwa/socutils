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
share of the nuclear charge (-sum E erfc(sqrt(z) r)/r, short-ranged).  Screening per
(shell pair, k group): the integral is estimated by the s-type Gaussian-product overlap
with shell amplitudes (normalization x max of r^l exp(-a r^2), x (2+l) sqrt(a) for the
gradient integrals) and group amplitudes (sum |E| 2 sqrt(z/pi), sum |L| max r^2m exp(-z r^2));
triples below eps (absolute) are skipped, the erf group and its point charge together.
eps 1e-12 gives ~1e-10 in J, 1e-10 gives ~1e-8 (`screen='decay'`, the earlier decay-only
criterion, needed eps 1e-20 for 1e-11).

Timings (4 cores; against one scalar direct NR JK build in the same uncontracted basis;
`test_c60.py`, `test_gold.py` -> `test_gold.out`):

| system | n2c | NR JK | X2C decoupling | kernel J | max err J |
|---|---|---|---|---|---|
| C60, unc-cc-pVDZ, decay screening eps 1e-20, patched / std | 3120 | 575 s | 238 s | 63 s / 75 s | 9e-12 |
| C60, estimate screening eps 1e-12, patched / std | 3120 | 575 s | 238 s | **49 s / 52 s** (0.09x) | 5e-10 |
| C60, estimate screening eps 1e-10, patched / std | 3120 | 575 s | 238 s | 40 s / 41 s (0.07x) | 8e-9 |
| Au4, unc-dyall-v2z, decay 1e-20, patched | 1632 | 48 s | 42 s | 11.5 s (0.24x) | 1.8e-8 |
| Au8 (cube), unc-dyall-v2z, decay 1e-20, patched | 3264 | 293 s | 292 s | 40 s (0.14x) | 1.8e-8 |

The unscreened pure-Python analytic J (`../40-analytic_onecenter_j.py`) takes 164 s for C60.
