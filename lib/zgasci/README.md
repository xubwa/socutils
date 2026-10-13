# zgasci

Full CI and generalized-active-space CI (GAS, with RAS and unrestricted FCI as
special cases) over spinors, as kernels in plain C99 + OpenMP, driven from
Python by `socutils.hci.gasci`. No other dependency: no LAPACK, no HDF5, no
input files. As in PySCF's FCI, C only applies the Hamiltonian and builds
density matrices; the CI vectors live in numpy arrays and are diagonalized
with `pyscf.lib.davidson1` (complex Hermitian or real symmetric).

Build: `make` at the top of socutils (lib/CMakeLists.txt), or directly

    gcc -std=c99 -O3 -fopenmp -fPIC -shared -o lib/libzgasci.so lib/zgasci/zgasci.c

`SOCUTILS_ZGASCI_LIBRARY` overrides the library path.

## Algorithm

* **Sub-groups.** The spinors (in the order of the kernels) are cut into
  consecutive sub-groups of at most 16; the first ones form side A, the others
  side B. A determinant is an A string times a B string, a string a sub-string
  of each sub-group of its side. The Python driver reorders the orbitals so
  that every constrained set is a union of sub-groups, half of each set on
  each side, sub-groups of at most `max_subgroup` (default 12) spinors.
* **Classes, buckets, blocks.** The class of a string is its number of
  electrons in each sub-group; within a class strings are numbered by the
  mixed radix of their sub-string ranks. Strings of a class are split into
  buckets by their symmetry labels (several additive label sets, each with an
  optional modulus). A block is (A bucket, the one bucket of a B class with the
  matching label), a dense matrix (A string, B string); the CI vector is the
  blocks one after another. Which classes pair is decided by the electron
  number and the constraints "electrons in this union of sub-groups within
  [min, max]". Tables are per sub-group only (2^m entries); neither
  determinants nor H are stored.
* **Hamiltonian.** `H = sum_{P,R} v[P,R] a+_p a+_q a_s a_r` over pairs
  `P = (p<q)`, `R = (r<s)`, `v` the antisymmetrized integrals with the
  one-electron part folded in (divided by N - 1), built by Python.
* **Reference kernel.** For each determinant K of N - 2 electrons, gather the
  coefficients of the determinants K + R in the space (with their signs),
  `e = v d`, scatter `sign * e[P]` to K + P (atomic adds). Electrons passing
  between A and B need no special case. The pair density
  `<bra|A+_P A_R|ket> = sum_K conj(d_bra[P]) d_ket[R]` is the same loop; dm1
  follows by contraction. The determinants K come in tasks (pairs of N - 2
  electron tuples, larger ones cut) under dynamic scheduling.
* **Block kernel** (`zg_sigma_block_{d,z}`, `sigma_kernel='block'`). Each
  term `v[P,R] a+_p a+_q a_s a_r` falls in one of nine kinds by whether P and
  R lie within A, across, or within B; each kind is exactly
  `sign (-1)^(k_B N_A) <a'|O_A(x)|a> <b'|O_B(y)|b>` with `W[x][y] = ±v[P,R]`
  (one, two electrons passing between the sides are kinds of their own, as
  in Olsen's LUCIAREL). sigma is built by tiles (rows x columns of a target
  block), each written by the thread that owns it: no atomic additions, the
  unit of a later MPI distribution. Strings are 64-bit words on each side
  (each side at most 64 spinors); excitations are done on the words in the
  loop, with only the per-sub-group tables, and no string-level excitation
  table is stored. Tiles are weighted from the classes for the dynamic
  schedule. `ZGASCI_TERMS` (bit mask) restricts the kinds,
  `ZGASCI_STATS` prints their multiply-adds.
* **Phase.** A determinant is `a+_p1 a+_p2 ... |0>`, `p1 < p2 < ...` in the
  order of the kernels; `GASCI.wavefunction` converts to the original order.

## Interface

C (`zgasci.h`): `zg_plan_new` / `zg_plan_free`, `zg_dim`, `zg_dets`,
`zg_info`, and `zg_hdiag_{d,z}`, `zg_sigma_{d,z}`, `zg_sigma_block_{d,z}`
(with `zg_block_ok`), `zg_rdm2_{d,z}` (real and
complex from one source, `zgasci_kernel.inc` compiled twice).

Python (`socutils.hci.gasci`):

* `GASCI(mol)` — spinors; `kernel(h1e, eri, norb, nelec, ci0, nroots, ecore)`
  with `eri[p,q,r,s] = (pq|rs)`, `make_rdm1`, `make_rdm2`, `make_rdm12`,
  `trans_rdm1`, `trans_rdm12` with the conventions of `SHCI` and
  `zfci.FCISolver` (`dm1[p,q] = <p^+ q>`, `dm2[p,q,r,s] = <p^+ r^+ s q>`,
  `t[p,q] = <bra|p^+ q|ket>`); a fcisolver of `socutils.mcscf.zmcscf.CASSCF`.
  Attributes: `gas` and `restrictions` (lists of `dict(orbitals, min, max)`),
  `orbsym` / `orbsym_mod` (as SHCI), `target`, `max_subgroup`, `subgroups`
  (explicit `(groups_a, groups_b)`), Davidson settings (`conv_tol`,
  `conv_tol_residual`, `max_cycle`, `max_space`, `level_shift`), `direct_dim`
  (spaces up to this size are diagonalized as a full matrix).
* `SpatialGASCI(mol)` — non-relativistic, real spatial orbitals, the
  interface of `pyscf.fci.direct_spin1` (`nelec` as `(na, nb)` or a number
  with `spin`; spin-traced and spin-resolved density matrices, `spin_square`,
  `orbsym` with D2h-subgroup irreps, `wfnsym`); a fcisolver of
  `pyscf.mcscf.CASSCF`. Side A holds the alpha, side B the beta spin orbitals,
  N_alpha is one more constraint, constraints name spatial orbitals.
  `trans_rdm1` follows pyscf: `t[p,q] = <bra|q^+ p|ket>`.
* `Plan` — the C plan: `dim`, `dets()`, `occslst()`, `hdiag(v)`,
  `sigma(v, c)`, `rdm2_pairs(bra, ket)`; `antisymmetrized(h1, eri, n, nelec)`
  builds `v`.

## Limits

At most 16 sub-groups of at most 16 spinors (so 256 spinors), 4 label sets,
2^31 strings per side, 4·10^6 electron distributions per side. Vectors are in
memory; OpenMP only; time reversal is not used. The reference kernel does
`sum_K n_conn(K)^2` multiply-adds (`n_conn` up to C(n - N + 2, 2)).

## Tests and measured timings

`python hci/test/test_gasci.py` (21 tests; `ZGASCI_TEST_KERNEL=block` runs
the solvers with the block kernel, and the exhaustive tests check both): kernels against H built from the
operators on bit strings (≤ 8 spinors; real and complex; FCI, GAS, labels with
and without modulus; several divisions), against zfci, zfci.kernel_dets and
SHCI (FCI, GAS by occslst and by restrictions in a selection, labels), the
same problem under five divisions to 1e-10, pyscf `direct_spin1`
(closed and open shell), `direct_spin1_symm` (C2v), SpatialSHCI (GAS),
pyscf CASSCF and zmcscf CASSCF (against SHCI as solver, 6e-11 Eh).

One sigma with the reference kernel, random dense integrals, 12 electrons,
16 threads (Xeon, measured):

| spinors | determinants | complex | real |
|--------:|-------------:|--------:|-----:|
| 20 | 125 970 | 0.08 s | 0.05 s |
| 22 | 646 646 | 0.48 s | — |
| 24 | 2 704 156 | 2.47 s | 1.32 s |

22 spinors: 6.72 s on 1 thread, 14× faster on 16. (Measured before
`-fcx-limited-range`, which makes complex products inline instead of calls
to `__muldc3`: about 25 % faster on one thread.)

Block kernel against the reference (complex, measured): it does the same
number of multiply-adds (1.62e10 at 24 spinors / 12 electrons, both close to
the number of connected determinants, so near the minimum of a direct CI),
and both run at about 2 ns per complex multiply-add per core, limited by the
dependent gathers. One thread, 20 spinors: block 0.97 s, reference 0.74 s;
16 threads, 24 spinors, on a shared node: block 4.8–5.5 s, reference
3.2–4.3 s. So the block kernel is the structure (ownership, tiles, no
atomics), not yet a speed-up; the reference stays the default. O2 (12, 32) has
2.26·10^8 determinants; from the operation count, about 9 min per complex
sigma with this kernel on 16 cores (estimate), the reason for the block
kernel planned next.
