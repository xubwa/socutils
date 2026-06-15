# nrr_outcore optimization — analysis / prototype

Goal: speed up the spinor (j-spinor) AO->MO transform used by socutils
MP2/ADC. Benchmark (H2O/cc-pVTZ) showed each `nrr_outcore.general_iofree`
block transform costs 5-14s and is the dominant cost of EE/EA.

## Why nrr is slow vs nr

Reference C copied here from pyscf `pyscf/lib/ao2mo/`:

* `nr_ao2mo.c` (1306 lines): full AO permutational-symmetry machinery
  -- `AO2MOfill_nr_{s1,s2ij,s2kl,s4}` + matching `AO2MOtranse1_nr_*`
  (which `NPdunpack_tril` the packed AO back to full before the MO
  contraction).
* `nrr_ao2mo.c` (276 lines): **only `s1`**. `AO2MOfill_nrr_s1` loops `jsh`
  over **all** shells (full `nao x nao`), and `AO2MOnrr_e1_drv` allocates the
  full `nao*nao*nkl` AO buffer. No `s2`/`s4`.

The AO integrals are `int2e_sph` -- real and 8-fold symmetric. Using `s4`
(i>=j, k>=l) computes ~4x fewer shell quartets and transforms ~2x fewer kl
pairs. So the s1-only nrr path does ~4x too much integral + transform work.

Confirmed: `libao2mo.so` exports `AO2MOfill_nr_s4` but **not**
`AO2MOfill_nrr_s4` (OSError: undefined symbol).

## Key structural fact (what to reuse)

The `fill` step (AO integral evaluation + symmetry packing) produces a
**real** AO buffer and is identical for nr and nrr. Only the MO contraction
differs: nrr does a complex two-component transform (alpha + beta j-spinor
components, summed) in `AO2MOmmm_nrr_iltj` (6 real `dgemm`s).

So the prototype reuses the existing, compiled nr fill
(`AO2MOnr_e1fill_drv` + `AO2MOfill_nr_s4`) and only adds the complex side:

* `AO2MOmmm_nrr_iltj`         -- complex 2-component MO contraction (copy)
* `AO2MOtranse1_nrr_s2ij/s1`  -- `NPdunpack_tril` then call the nrr mmm
* a driver that fills once (nr s4) then transforms alpha and beta and sums.

The kl (ket) side is already handled in Python by `_ao2mo.r_e2(..., aosym)`
(pass 2), which supports s2kl/s4; only the e1/bra side needed new C.

## Deficiencies of the spinor (nrr / r) outcore path vs nr

1. **No AO permutational symmetry (s2/s4).** `nrr_ao2mo.c` implements only
   `s1`; the bra half-transform and AO integral generation do ~4x too much
   work. (nr has s2ij/s2kl/s4.)
2. **No bra index-order optimization (iltj vs igtj).** The bra half-transform
   step is O(n2c^2 * first_count); the cheaper order transforms the *smaller*
   MO count first. `nrr_ao2mo.c` has only `AO2MOmmm_nrr_iltj` (the igtj choice
   in `nrr_outcore.r_e1` is commented out), and `r_outcore.py` never selects
   between `AO2MOmmm_r_{iltj,igtj}` either -- so both always transform the
   first index first. When bra1 > bra2 this wastes max/min on the dominant
   step (e.g. nv/no ~ 10x for a (v,o) bra at H2O/cc-pVTZ). Fix: add
   `AO2MOmmm_nrr_igtj` and pick iltj/igtj from i_count vs j_count.
3. **Symmetry plumbing inconsistent beyond s1** (see Findings below):
   `_count_naopair` uses `ao_loc_2c`, `general` vs `half_e1` disagree on the
   s2ij `nao_pair`, and `_ao2mo.r_e2`'s s4 is the 2C/Kramers path.

## Prototype (this dir)

* `nrr_ao2mo_opt.c` -- optimized e1 driver `AO2MOnrr_opt_e1_drv`: reuses
  `AO2MOnr_e1fill_drv` + `AO2MOfill_nr_s4` (compiled in libao2mo) for the real
  symmetry-packed AO fill, then does the complex two-component MO contraction
  (`AO2MOmmm_nrr_iltj`) with a tril-unpack `transe1`.
* `nrr_opt.py` -- monkeypatches `nrr_outcore.r_e1` to call the opt driver and
  runs `general_iofree(..., aosym='s4')`.

Build (needs the installed pyscf libs at runtime):

    PI=/usr/local/lib/python3.11/dist-packages/pyscf/lib
    PB=/home/user/pyscf/pyscf/lib
    gcc -shared -fPIC -fopenmp -O2 nrr_ao2mo_opt.c -o libnrr_opt.so \
      -I$PI -I$PB -I$PI/deps/include -I. \
      -L$PI -lao2mo -lnp_helper -lcvhf -L$PI/deps/lib -lcint \
      $(ls $PI/libopenblas*.so) -Wl,-rpath,$PI -Wl,-rpath,$PI/deps/lib

## Findings (the e1 s4 step works; full integration is blocked)

* The **e1 (bra) s4 transform works**: `r_e1_opt` returns the correct shape;
  the AO fill reuse + complex mmm is sound.
* Integration via `nrr_outcore.general(..., aosym='s4')` **hangs in pass-2**
  `_ao2mo.r_e2(..., 's4')`. Root causes -- the nrr symmetry plumbing is
  s1-only and inconsistent beyond it:
  - `_count_naopair` counts AO pairs with `mol.ao_loc_2c()` (2-component) but
    the integrals are spherical `int2e_sph` -> wrong kl column count.
  - `general()` (l.76) treats `s2ij` kl as `nao*nao` while `half_e1()` (l.226)
    treats it as triangular -- the two disagree.
  - `_ao2mo.r_e2`'s s4 kl-unpack is the relativistic (2C/Kramers, time-rev)
    path, not the spherical s4 packing produced by `AO2MOfill_nr_s4`.
* Conclusion: a real fix must port nr's symmetry-aware kl handling
  (`AO2MOtranse2_nr_s4` / a sph-s4 `r_e2`) and fix the `nao_pair` bookkeeping
  for the sph-AO j-spinor path; then both ij (done) and kl get the 4x.

## End-to-end CORRECT (nrr_incore.py); speed needs C

`nrr_incore.general_incore` computes (ij|kl) over j-spinor MOs as
`Bbra . eri_s4 . Bket^T` with `eri_s4 = intor('int2e_sph', aosym='s4')` and
folded two-component transition densities. **Verified correct to machine
precision** vs `int2e_spinor` (oovv/vovv/voov/full, ~1e-14). This addresses:
  * #1 s4: AO integrals via `aosym='s4'` (~4x fewer shell quartets);
  * #2 order: BLAS zgemm picks the contraction order;
  * #3 plumbing: sidesteps the broken outcore s2/s4 machinery entirely.

But it is **not faster** than the stock s1 `nrr_outcore` at moderate sizes
(H2O/cc-pVTZ, 5 ADC blocks: stock s1 outcore 3.9s vs zgemm-incore 8.0s).
Reasons: the numpy/zgemm transform does not beat pyscf's C transform, and the
s4 saving is on AO *generation*, which `intor('int2e_sph','s4')` already does
in ~0s -- it is not the bottleneck. The bottleneck is the MO transform, which
must be in C (BLAS) to win.

A genuine speedup therefore requires the **C-level s4 outcore**:
  * e1 (bra) s4: DONE in nrr_ao2mo_opt.c (reuse nr fill + complex mmm), works.
  * e2 (kl) s4: port nr's `AO2MOtranse2_nr_s2kl` (spherical *shell-block* AO
    unpack via ao_loc -- NOT element-tril; `r_e2`'s s4 instead applies
    time-reversal, the 2C/Kramers route) and pair it with a complex mmm.
  * fix `nao_pair` bookkeeping for the sph shell-block packing.

## C s4 outcore attempt -- packing convention is the wall

Driving the C e1 (`AO2MOnrr_opt_e1_drv`, reuse `AO2MOfill_nr_s4` + complex
bra mmm) directly for the full kl range gives a half-transform whose kl axis
is **nr's shell-block s2kl packing** (count nao*(nao+1)/2 but a shell-block
*ordering*, NOT simple element-tril). Consequences found by experiment:
  * `lib.unpack_tril` (element-tril) on it -> wrong (mis-assigned la,si);
  * a hand-written numpy replica of `AO2MOsortranse2_nr_s2kl`'s shell-block
    unpack also did not reproduce int2e_spinor -- the exact fill ordering
    (s4 fill vs s2kl transe2, and the bra element-tril vs kl shell-block) has
    subtleties that are too error-prone to replicate by hand.

Conclusion: the e2 must be done **in C by copying nr's `AO2MOsortranse2_nr_s2kl`
verbatim with `double` -> `double complex`** (so the unpack ordering is
guaranteed identical to the fill), paired with a complex ket mmm (zsymm on the
symmetric complex AO block + zgemm), and a 2-spin (alpha/beta) driver that
sums -- mirroring `AO2MOnr_e2_drv`. Then end-to-end s4 in C with a real
speedup. The e1 (bra) C side already works; only this e2 C remains.

The verified-correct fallback is `nrr_incore.py` (slow but right).

## Status
- [x] copied nr/nrr/r C + nr/nrr outcore Python here for study
- [x] optimized e1 C driver (s4 fill reuse + complex bra mmm), builds & runs
- [x] **end-to-end CORRECT** spinor s4 transform (nrr_incore.py), verified
- [ ] C e2: copy AO2MOsortranse2_nr_s2kl as double complex + complex mmm +
      2-spin driver (guaranteed packing match) -> end-to-end C s4 + speedup
