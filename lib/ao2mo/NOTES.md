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

## Status
- [x] copied nr/nrr/r C + nr/nrr outcore Python here for study
- [x] optimized e1 C driver (s4 fill reuse + complex mmm), builds & runs
- [x] e1 transform correctness verified in isolation
- [ ] pass-2 (kl) sph-s4: port nr transe2_s4 / fix r_e2 + nao_pair bookkeeping
- [ ] end-to-end correctness vs int2e_spinor + speed test
