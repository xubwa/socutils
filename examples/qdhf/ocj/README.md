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

The bold ones are not in libcint; `spsp3c1e.cl` adds them (and spinor sigma.p sigma.p
variants, unused: libcint has no spinor 3c1e driver).  Build:

    git clone https://github.com/sunqm/libcint && cd libcint/scripts
    cp .../spsp3c1e.cl . && clisp spsp3c1e.cl && cp int3c1e_spsp.c ../src/autocode/
    # add src/autocode/int3c1e_spsp.c next to src/autocode/int3c1e.c in CMakeLists.txt
    mkdir ../build && cd ../build && cmake .. && make
    gcc -O2 -fopenmp -fPIC -shared -o libocj.so onecenter_j.c -lm

`ocj.py` loads that libcint by absolute path (RTLD_LOCAL, next to pyscf's own libcint)
and passes function pointers to the kernel: set `OCJ_LIBCINT` (path of the new
libcint.so) and `OCJ_LIB` (libocj.so), default: next to `ocj.py`.

Each atom's exponents are split into groups of one decade; every erf group carries its
share of the nuclear charge (-sum E erfc(sqrt(z) r)/r, short-ranged) and is screened with
its own smallest exponent.  The screening bound ignores prefactors, so eps = 1e-20
(error 1e-11) is needed; 1e-13 gives 2e-4 in SS.

C60 (unc-cc-pVDZ, n2c 3120, 4 cores): kernel 63 s against 575 s for one scalar NR JK
build (164 s for the unscreened Python version, 483 s NR J, 238 s X2C decoupling);
9e-12 from the exact J.
