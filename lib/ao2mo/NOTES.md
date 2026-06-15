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

## Status
- [x] copied nr/nrr/r C + nr/nrr outcore Python here for study
- [ ] write optimized nrr C (s2ij/s4 via reused nr fill)
- [ ] build a socutils .so against installed libcint/libcvhf/libnp_helper
- [ ] Python driver + correctness (vs int2e_spinor) + speed test
