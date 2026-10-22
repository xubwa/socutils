# Validation log

Each entry names the test that pins the result; rerun with
`source ~/apps/use-socutils-git.sh && pytest`.

## Infrastructure

| check | result | test |
|-------|--------|------|
| determinant phases = pyscf cistring (`cre_sign`, `des_sign`) | exact | `test_dets` |
| `{a_p, a_q^+} = delta_pq`, `{a_p, a_q} = 0` on random vectors, all sectors | < 1e-14 | `test_dets` |
| operator-based H vs `zfci.make_hmat` / `make_hmat_dets` (random complex integrals) | < 1e-12 | `test_dets` |
| RDM index order: zmr = `zfci.make_rdm12_dets` = `fci_dhf_slow.make_rdm12` | < 1e-13, no transposition | `test_rdm_bydef` |
| CASCI energy (H2O, frozen core 0/1/2) vs PySCF | < 1e-10 | `test_integrals` |
| generalized Fock vs AO `get_jk` (spin-diagonal random gamma) | < 1e-10 | `test_fock` |
| active-space sector H vs zfci, all electron numbers | < 1e-12 | `test_active` |

## UC-NEVPT2 vs brute force (validation item 1)

Two independent brute-force references:

* `reference/nevpt2_bruteforce.py`: spin-orbital determinants of the whole
  correlated space, generic operators, sparse LU.
* `reference/nevpt2_spinfree.py`: spatial orbitals, full FCI space,
  `pyscf.fci.direct_spin1`, GMRES; shares only the PySCF integrals.

H2O/6-31G, CAS(4,4), 1s frozen, ground state (Hartree):

| class | UC-NEVPT2 | spin-free brute force − UC |
|-------|-----------|----------------------------|
| ccvv | -0.017327210035 | 1.5e-15 |
| ccav | -0.011220568521 | -4.8e-16 |
| cavv | -0.034676969144 | -7.4e-16 |
| ccaa | -0.004344583964 | 3.8e-16 |
| aavv | -0.025568638931 | 5.6e-16 |
| cava | -0.023619998530 | -5.8e-16 |
| caaa | -0.002900040425 | 7.9e-16 |
| aaav | -0.003925030341 | -1.2e-15 |

The spin-orbital brute force agrees class by class to < 1e-10 for random
complex Hamiltonians (1-3 active electrons in 8 spin-orbitals, one and three
states, both E0 options, unequal weights) and for H2O
(`test_uc_exact`, `test_spinfree_reference`).

## Comparison with PySCF (validation item 2)

H2O/6-31G, CAS(4,4), no frozen core (PySCF NEVPT2 does not support it):

| class | SC-NEVPT2 (PySCF) | UC-NEVPT2 (zmr) |
|-------|-------------------|-----------------|
| ccvv (Sijrs) | -0.017851480637 | -0.017851480637 |
| ccav (Sijr) | -0.011363899548 | -0.011365149939 |
| cavv (Srsi) | -0.034956971125 | -0.034969599042 |
| ccaa (Sij) | -0.004351975702 | -0.004355387613 |
| aavv (Srs) | -0.025542895991 | -0.025568638931 |
| cava (Sir) | -0.023669783221 | -0.023684393561 |
| caaa (Si) | -0.002895851648 | -0.002901238937 |
| aaav (Sr) | -0.003916858215 | -0.003925030341 |
| total | -0.124549716087 | -0.124620919001 |

ccvv is identical; every other class obeys `E_UC <= E_SC`, as required by the
Hylleraas principle (the SC perturbers span a subspace of each UC class).
An empty active space reproduces MP2 to < 1e-10 with and without frozen core
(`test_pyscf_compare`).

## Laplace transform

Relative error of the whole second-order matrix vs the exact engine
(random complex Hamiltonian, 3 states): 3e-7 (10 points), 2e-12 (20),
1e-15 (30).  H2O/6-31G CAS(4,4) with frozen 1s (denominator ratios 2-5):
5e-8 (10), 3e-13 (20), 2e-16 (30); without frozen core, 60 points agree to
< 1e-10 (`test_laplace`).

## Multi-state

Equal weights with `e0='average'`: a random unitary rotation of the model
space leaves the energies unchanged to < 1e-10 and transforms `H_eff` as
`U^+ H_eff U`; per-state E0 and unequal weights break the invariance
(`test_multistate`).

## X2C spinor Hamiltonians and O2 (validation items 3 and 4)

System: O2 at 1.207 A on the z axis, X2CAMF (with SOC) or spin-free X2C
('sf1e') in the j-spinor basis; closed-shell O2^2- spinor orbitals;
CAS(2e, 4 pi_g* spinors) for neutral O2; 1s spinors frozen; model space =
all six CASCI states (`test_x2c`).

| check | result |
|-------|--------|
| CASCI energies vs socutils zcasci + zfci (with and without SOC) | < 1e-10 |
| O2/STO-3G spinor UC-NEVPT2 vs brute force, class by class, 6 states, both E0 | < 1e-10 |
| no SOC: spread of the three X 3Sigma_g- components (cc-pVDZ) | 9e-14 (state E0), 6e-14 (average E0) |
| SOC: Omega values from Jz in the model space | -2, -1, 0, 0, 1, 2 |
| SOC: largest H_eff element between different Omega | 2e-14 |
| SOC: largest same-Omega coupling (X 0+ with b 1Sigma_g+) | 4.8e-2 |
| random unitary rotation of the model space (average E0) | energies unchanged to < 1e-10 |

The zero-field splitting obtained in this setup (CASCI 2.60 cm^-1,
UC-NEVPT2 1.80 cm^-1 with average E0, 1.28 cm^-1 with state E0; experiment
about 3.96 cm^-1) is a smoke test of the pipeline only: the orbitals come
from O2^2-, the active space is minimal and the basis small.

### O2 with state-averaged CASSCF orbitals (`examples/01-o2_zfs.py`)

X2CAMF, SA-CASSCF(8e, 12 spinors: 2p sigma, pi, pi*, sigma*) over the six
lowest states with socutils (density-fitted KRHF reference), then UC-NEVPT2
with exact integrals in those orbitals (1s frozen, 2s correlated).  Energies
relative to X 3Sigma_g-(0+), cm^-1:

| basis | method | X(+-1) - X(0+) | a 1Delta_g | b 1Sigma_g+ | E2(X), Eh |
|-------|--------|----------------|------------|-------------|-----------|
| cc-pVDZ | CASCI | 1.90 | 7330.6 | 13597.6 | |
| cc-pVDZ | UC-NEVPT2, average E0 | 1.83 | 8597.5 | 15413.4 | -0.248864 |
| cc-pVDZ | UC-NEVPT2, state E0 | 1.73 | 8187.9 | 14671.1 | -0.247644 |
| cc-pVTZ | CASCI | 2.08 | 7345.9 | 13658.9 | |
| cc-pVTZ | UC-NEVPT2, average E0 | 2.18 | 8398.7 | 15409.9 | -0.341854 |
| cc-pVTZ | UC-NEVPT2, state E0 | 2.03 | 7886.6 | 14481.1 | -0.340331 |
| experiment | | 3.96 (total) | 7918 | 13195 | |

The X2CAMF Hamiltonian contains no electronic spin-spin coupling, so the
splitting is the spin-orbit part of the experimental value (literature
decompositions put that part near 2.4 cm^-1).  Runs: Slurm jobs 31468214/5,
`~/scr4/zmr_runs/o2_zfs`.

## PC-NEVPT2

Spectral PC-NEVPT2 agrees with a brute-force internally contracted
reference (span of all elementary one- and two-body strings acting on the
model states, solved pattern by pattern with the generic `H_D`) to < 1e-10
for both contractions, both E0 choices and one or three states
(`test_pc`).  The Wick-generated RDM formulas reproduce the spectral metric
and `H_act` matrices to < 1e-11 for all classes and the PC energies to
< 1e-12 (`test_pc_rdm`).

H2O/6-31G, CAS(4,4), no frozen core, ground state (Hartree):

| class | SC (PySCF) | PC (zmr) | UC (zmr) |
|-------|------------|----------|----------|
| ccvv | -0.017851480637 | -0.017851480637 | -0.017851480637 |
| ccav | -0.011363899548 | -0.011364123176 | -0.011365149939 |
| cavv | -0.034956971125 | -0.034961316127 | -0.034969599042 |
| ccaa | -0.004351975702 | -0.004355387613 | -0.004355387613 |
| aavv | -0.025542895991 | -0.025568638931 | -0.025568638931 |
| cava | -0.023669783221 | -0.023680126543 | -0.023684393561 |
| caaa | -0.002895851648 | -0.002901238937 | -0.002901238937 |
| aaav | -0.003916858215 | -0.003925030341 | -0.003925030341 |
| total | -0.124549716087 | -0.124607342304 | -0.124620919001 |

`E_UC <= E_PC <= E_SC` holds for every class; for ccaa, aavv, caaa and aaav
the PC functions already span the whole target sector (8 active spinors), so
PC = UC there.

## 4f-scale timing (Pr3+, X2CAMF, x2c-SVPall)

14 active 4f spinors, 18 correlated core spinors (4d 5s 5p), 102 virtual
spinors, 1s-4p frozen; one 48-core node (Slurm), UC-NEVPT2 with Laplace for
ccav/cavv and exact elsewhere, common E0.  Wall time of `kernel()`:

| case | states | largest sector | before | after | peak memory |
|------|--------|----------------|--------|-------|-------------|
| 4f^2 | 13 | 1001 | 281 s | 71 s (after the block-product fix only) | 2.7 GB |
| 6 electrons (stress) | 7 | 3432 | 150 s | 56 s | 6.1 GB |

Profile of the stress case after the fixes: dense diagonalization of the
five sectors 24 s, the single (PQ|PQ) integral transform 7.5 s (previously
ten transforms, 61 s), ccvv 5.9 s (previously 61 s: the zero-row threshold
was defeated by round-off; the invariant-span projection is exact).  The
dense sector diagonalization is now the main cost.

## Performance round: native kernels, CI-vector PC spaces, symmetry blocks

Same system as above (Pr3+ CAHF orbitals, x2c-SVPall: 36 frozen, 18 core,
14 active, 102 virtual spinors); one node with 2 x Xeon Gold 6248R (48
cores), exclusive (Slurm jobs 31488324, 31490479, 31491289, 31491488;
`~/scr4/zmr_runs/pc_bench2 ... pc_bench4`, `bench.py`).  "old" is commit
07d87b5 (numpy / OpenBLAS, sector eigenbasis everywhere; run with
`KMP_BLOCKTIME=0`, its best environment, the default one is within 5 %);
"new" is commit 92f9306.  Wall time of `kernel()` in seconds on 48 threads;
"exact": exact denominators in every class, "Laplace": Laplace engine for
ccav and cavv; "+ sym": after `symmetry.adapt` (4 generators, largest
discarded integral 5e-13 Eh):

| case | method | old | new | new + sym | peak memory old / new |
|------|--------|-----|-----|-----------|-----------------------|
| 4f^2, 9 states | UC, exact | 9.9 | 1.4 | 1.2 | 2.8 / 2.1 GB |
| | PC union, exact | 9.8 | 1.4 | 1.1 | |
| | PC state, exact | 13.4 | 3.7 | 2.0 | |
| | UC, Laplace | 9.6 | 2.1 | | |
| | PC union, Laplace | 10.0 | 2.1 | | |
| 4f^2, 91 states | UC, Laplace | 20.8 | 2.9 | | 5.0 / 2.8 GB |
| | PC union, Laplace | 22.7 | 3.0 | | 6.4 / 2.8 GB |
| | UC, exact | | 2.9 | | |
| | PC union, exact | | 2.6 | 2.4 | |
| | PC state, exact | | 38.5 | 25.7 | |
| 6 electrons, 9 states | UC, exact | 122.1 | 6.8 | 3.7 | 5.9 / 3.9 GB |
| | PC union, exact | 75.6 | 7.4 | 2.3 | 6.2 / 4.6 GB |
| | PC state, exact | 166.7 | 13.9 | 5.7 | 5.0 / 3.7 GB |

Constructing the driver (active Hamiltonian, model states) adds 1.2 - 2.5 s
(5.4 - 10.9 s in the old code); the Hamiltonian setup of socutils (X2CAMF,
frozen-core Fock) takes 8 s and is not part of zmr.  The 6-electron case is
a stress test of the active-space algebra (sectors of dimension 3003, 3432,
2002; the 4f^6 system is unbound in Pr3+ orbitals, so its numbers are not
physical).

Thread scaling of the new code, `kernel()` in seconds:

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---------|---|---|---|---|----|----|----|
| 6 electrons, PC union (dense eigenproblems) | 91.0 | 47.0 | 26.4 | 15.4 | 10.0 | 8.3 | 7.6 |
| 6 electrons, PC union + sym | 16.1 | 9.5 | 5.9 | 4.0 | 2.9 | 2.6 | 2.5 |
| 6 electrons, PC state | 118.0 | 63.0 | 39.7 | 23.9 | 15.6 | 13.6 | 13.2 |
| 6 electrons, UC + sym | 45.2 | 23.2 | 14.1 | 8.1 | 5.1 | 4.2 | 3.7 |
| 4f^2, 91 states, PC union, exact | 27.1 | 14.5 | 8.2 | 4.8 | 3.2 | 2.8 | 2.6 |
| 4f^2, 91 states, PC union, Laplace | 19.4 | 10.7 | 6.7 | 4.3 | 3.2 | 3.0 | 3.0 |
| 4f^2, 91 states, PC state + sym | 88.7 | 51.6 | 36.0 | 27.1 | 23.4 | 25.5 | 30.2 |

The speed-up is 6 - 12 at 16 - 48 threads.  What remains at 48 threads is
the part that does not scale: the Hermitian eigensolver (4.4 of 7.5 s in the
dense 6-electron case, 15 calls of `zheevd`), the integral transformation
(0.5 s), the Fock builds and the Python loop over external blocks.
The state-specific contraction with 91 states is such a loop (91 kets, each
with its own first-order space onto which all 91 bras are projected) and
does not profit from more than 16 threads.

What changed the cost, in the order of the commits:

* MKL for all dense algebra and OpenMP kernels for the string operations
  and the denominator sums; sector Hamiltonians built by a kernel.
* PC first-order spaces from CI vectors in the determinant basis: one span
  and one Koopmans problem per class, no sector spectrum unless the span is
  the whole sector; integrals contracted first in caaa / aaav.
* Non-redundant operator strings; the span from the Gram matrix with a
  Cholesky-QR step instead of an SVD (the SVD took 5.3 s per state in caaa).
* Symmetry-adapted active spinors: 26 blocks of at most 327 determinants
  instead of 3432; sector diagonalization 3.4 s -> 0.35 s.
* The four-index transformation in memory with `s4` symmetry (2.8 s
  instead of 7.6 s for the superblock of all classes).
* The sum over `(k, e)` without storing `u` (1.5 GB per block in UC cavv),
  for one column in the state-specific contraction, and with the external
  labels summed first when there are many states (ccav with 91 states:
  0.86 s -> 0.65 s, cavv 0.76 s -> 0.08 s); the Laplace engine through MKL
  (ccav 9.3 s -> 0.6 s).
* Thread teams sized by the work, and numpy's OpenBLAS limited to one
  thread inside the drivers: on 48 threads the small cases had been slower
  than on 16 (4f^2, 9 states, PC union: 2.5 s -> 1.4 s).

Energies: every 4f^2 number above agrees between old and new code to the
printed digits (`E[0]` and `tr(H2)` to 1e-9).  In the 6-electron case the
union PC energy moved by 7e-5 Eh, entirely through the meaning of `lindep`:
with non-redundant strings the metric eigenvalues of pair strings are half
of what they were, and this case has a metric spectrum without a gap,

| `lindep` | 1e-8 | 1e-10 (default) | 5e-11 | 1e-12 | 1e-14 | UC |
|----------|------|-----------------|-------|-------|-------|----|
| tr(H2) / Eh | -25.539923 | -25.542252 | -25.542823 | -25.543235 | -25.543237 | -25.545822 |

(old code with redundant strings and 1e-10: -25.542821).  In the 4f^2 cases
the union energy is the same for `lindep` from 1e-6 to 1e-12.

Two observations on the multi-state contractions, 3H4 ground level of Pr3+
(nine degenerate states; spread of the nine NEVPT2 energies):

| contraction | model states | spread / Eh | lowest - UC / Eh |
|-------------|--------------|-------------|------------------|
| UC | any | 1.6e-11 | |
| PC union | any | 1.1e-11 | 3.7e-06 |
| PC state | arbitrary mixtures within the level | 8.4e-05 (18 cm^-1) | -2.1e-05 |
| PC state | M_J eigenstates (after `symmetry.adapt`) | 8.1e-06 (1.8 cm^-1) | 1.4e-05 |

The state-specific contraction splits a degenerate level, by an amount that
depends on how the degenerate states happen to be chosen; the union
contraction keeps it degenerate to round-off.  (The total energy itself is
reproducible only to about 1e-8 Eh between runs with different thread
counts, because the X2CAMF atomic mean field of socutils is; `tr(H2)` is
reproducible to 1e-9.)

Non-relativistic code (`zmr.nr`), N2 with CASSCF(10,8) (2 core, 8 active
orbitals; sectors of up to 4900 determinants), wall time in seconds of the
NEVPT2 step, 48 / 16 / 1 threads:

| basis | PySCF SC-NEVPT2 | zmr.nr PC | zmr.nr UC |
|-------|-----------------|-----------|-----------|
| cc-pVTZ (60 functions) | 0.9 / 0.8 / 1.0 | 1.1 / 1.1 / 3.5 | 6.8 / 9.2 / 70 |
| cc-pVQZ (110 functions) | 1.3 / 1.2 / 1.8 | 2.5 / 2.5 / 7.5 | 10.0 / 11.2 / 76 |

Second-order energies (cc-pVTZ): SC -0.22834, PC -0.23194, UC -0.23220 Eh.
UC is the dense diagonalization of 13 sectors (11.5 of 13.9 s on 16
threads).

Matrix-free sectors (job 31491846, commit 8047e61): N2 with CASCI(10,10),
63504 determinants in the `M_S = 0` sector, PC-NEVPT2 from PySCF's CI vector:
13.7 s (cc-pVDZ) and 14.1 s (cc-pVTZ) on 48 threads, 23.4 s and 23.8 s on
16, peak memory 5.3 GB; caaa and aaav (about 1000 sigma vectors each per
spin pattern) take 5 s each.  PySCF's SC-NEVPT2 needs 6.8 s.  Second-order
energies (cc-pVTZ): SC -0.29309, PC -0.29865 Eh; `E_PC <= E_SC` in every
class and ccvv agrees to 4e-15.

## O2 zero-field splitting: active space and state average (`examples/04-o2_zfs_active_spaces.py`)

Orbitals as in the zmcscf O2 inputs: R = 1.20752 A, uncontracted aug-cc-pVTZ
(232 spinors), Gaussian nuclei, X2CAMF, Kramers-restricted SCF of O2^2+ with
Cholesky-decomposed integrals (1e-7), super-CI SA-CASSCF converged to an
orbital gradient of 1e-6.  Active space: 2 electrons in the 4 pi_g* spinors,
or the full valence 2p space, 8 electrons in 12 spinors (natural occupations
0.98 x 6, 0.52 x 4, 0.02 x 2).  State average: the three components of
X 3Sigma_g- (SA-3) or the six lowest states X, a 1Delta_g, b 1Sigma_g+ (SA-6).
zmr: exact integrals in these orbitals, 1s frozen (4 spinors), 10 or 4 core
and 214 or 212 virtual spinors; model states are CASCI eigenstates (the three
components of X, or the six states), symmetry-adapted (Jz) active spinors.
Slurm jobs 31496142, 31496831, 31496911, 31496912 (`express`, 4 cores, 6 - 8
min each for both Hamiltonians, peak memory 1.9 GB),
`~/scr4/zmr_runs/o2_zfs_cas24`.

The CASSCF step reproduces the zmcscf calculations it follows (logs in
`~/scr4/zmcscf/O2`, converged there to gradients of 1e-5 - 6e-5):

| calculation | E(SA) / Eh, here | zmcscf log | splitting / cm^-1, here | zmcscf log |
|-------------|------------------|------------|-------------------------|------------|
| CAS(2,4), SA-3, DCB | -149.7290310550 | -149.7290310538 | 2.1765 | 2.1725 |
| CAS(2,4), SA-6, DCB | -149.6975394137 | | 1.7241 | 1.7251 |
| CAS(8,12), SA-3, DCB | -149.8282305771 | -149.8282305760 | 2.6674 | 2.6673 |
| CAS(8,12), SA-3, DC | -149.8580774388 | -149.8580774383 | 3.4908 | 3.485, 3.495 |

The published X2C-CASSCF values (X. Wang, S. Wang, Y. Wu, L. Cheng,
arXiv:2602.24236, Table 1: CAS(8e, 12 spinors), SA-3, the same basis and bond
length) are, from the one-decimal entries of the table, 6.9 - 3.3 = 3.6 cm^-1
for X2CAMF(DC) and 3.6 - 0.7 + 0.00 = 2.9 cm^-1 for X2CAMF(DCB); here 3.49
and 2.67.  The Dirac-Coulomb value agrees within the rounding of the table;
the Gaunt contribution is -0.82 here against -0.7 there (the zmcscf log of
April 2026 has the 2.667 found here; switching off `with_pcc` in the X2CAMF
helper changes nothing, job 31497524).  The remaining
entries of that table (X2CMP -0.2, QED -0.1, picture-change correction of
the fluctuation potential including the spin-spin coupling +0.9, total 3.6,
experiment 3.96) are not part of the X2CAMF Hamiltonian used here.

X(+-1) - X(0+) in cm^-1.  "3 st." and "6 st." are the model spaces of the
effective Hamiltonian; NEVPT2 columns are union PC-NEVPT2 with state-specific
E0, in parentheses with the common E0:

X2CAMF, Dirac-Coulomb-Breit:

| active space | orbitals | CASSCF | NEVPT2, 3 st. | NEVPT2, 6 st. | UC-NEVPT2, 3 st. |
|--------------|----------|--------|---------------|---------------|------------------|
| CAS(2,4) | SA-3 | 2.1765 | 1.4957 (1.7825) | 1.3514 (1.6206) | 1.4957 |
| CAS(2,4) | SA-6 | 1.7241 | 1.7408 (1.9678) | 1.5982 (1.8249) | 1.7408 |
| CAS(8,12) | SA-3 | 2.6674 | 1.9359 (2.1333) | 1.9787 (2.1625) | 1.9489 |
| CAS(8,12) | SA-6 | 2.0814 | 2.0525 (2.2061) | 2.0362 (2.1897) | 2.0635 |

X2CAMF, Dirac-Coulomb:

| active space | orbitals | CASSCF | NEVPT2, 3 st. | NEVPT2, 6 st. | UC-NEVPT2, 3 st. |
|--------------|----------|--------|---------------|---------------|------------------|
| CAS(2,4) | SA-3 | 2.8456 | 1.9564 (2.3316) | 1.7676 (2.1197) | 1.9564 |
| CAS(2,4) | SA-6 | 2.2542 | 2.2769 (2.5739) | 2.0906 (2.3871) | 2.2769 |
| CAS(8,12) | SA-3 | 3.4908 | 2.5337 (2.7920) | 2.5882 (2.8287) | 2.5490 |
| CAS(8,12) | SA-6 | 2.7250 | 2.6862 (2.8873) | 2.6633 (2.8642) | 2.6989 |

a 1Delta_g and b 1Sigma_g+ in cm^-1 (Dirac-Coulomb-Breit; experiment 7918 and
13195), CASSCF and six-state NEVPT2 with state-specific E0:

| active space | orbitals | CASSCF a, b | NEVPT2 a, b |
|--------------|----------|-------------|-------------|
| CAS(2,4) | SA-3 | 10427, 20855 | 9124, 17462 |
| CAS(2,4) | SA-6 | 10309, 20618 | 9054, 17344 |
| CAS(8,12) | SA-3 | 7418, 13788 | 7868, 14447 |
| CAS(8,12) | SA-6 | 7327, 13626 | 7834, 14398 |

* With the full-valence active space the NEVPT2 splitting is 1.94 - 2.06
  cm^-1 (Dirac-Coulomb-Breit) for every combination of orbitals, model space
  and contraction, although the CASSCF values differ by 0.59 cm^-1: the SA-3
  CASSCF value is lowered by 0.73, the SA-6 value by 0.03.  The second-order
  treatment removes the dependence on the state average.
* The minimal active space is a poor reference for second-order theory: the
  results scatter between 1.35 and 1.74 cm^-1, and the b - X gap stays at
  17300 - 17500 cm^-1.
* Contraction: with four active spinors the contracted functions span the
  whole sectors (4, 1 and 6 determinants for N +- 1, N +- 2 and N electrons)
  and PC = UC to all printed digits; with twelve spinors they differ by
  0.002 - 0.013 cm^-1.
* The RDM route (metric and Koopmans matrices from RDMs up to rank 4) agrees
  with the CI-vector route to 1e-13 Eh for CAS(2,4), where the 3- and 4-RDM
  vanish; it is not run for twelve spinors (62 GB of 4-RDMs for three
  states).
* Contribution of every class to the splitting (Dirac-Coulomb-Breit, three
  model states, which are Omega eigenstates, so that the class matrices are
  diagonal and the contributions add up), state-specific E0:

  | class | CAS(2,4) SA-3 | CAS(2,4) SA-6 | CAS(8,12) SA-3 | CAS(8,12) SA-6 |
  |-------|---------------|---------------|----------------|----------------|
  | ccvv | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | ccav | -0.2092 | -0.1418 | +0.0120 | +0.0076 |
  | cavv | -0.0001 | +0.0954 | +0.0112 | +0.0426 |
  | ccaa | +0.4263 | +0.3602 | +0.1851 | +0.1415 |
  | aavv | +0.2861 | +0.2414 | +0.2226 | +0.2607 |
  | cava | -1.1972 | -0.5494 | -0.3356 | -0.2772 |
  | caaa | -0.0008 | -0.0040 | -0.0388 | -0.0319 |
  | aaav | +0.0140 | +0.0150 | -0.7879 | -0.1722 |
  | sum | -0.6808 | +0.0167 | -0.7315 | -0.0289 |

  The lowering of the SA-3 values comes from the single excitations into the
  virtual space with a rearrangement of the active electrons (cava for the
  minimal, aaav for the full-valence active space).
* E0: with the common E0 the purely external ccvv class, which does not
  involve the active electrons, contributes +0.09 cm^-1 for CAS(2,4) SA-3
  (ccav -0.1368, cavv +0.0455, ccaa +0.4377, aavv +0.2881, cava -1.1323,
  sum -0.3940), and the two E0 choices differ by N times the CASSCF
  splitting, N = 0.132 for CAS(2,4) and 0.074 for CAS(8,12): the separability
  defect of the common E0 (conventions.md section 9).  The same shows in the
  excitation energies (b at 15441 instead of 14447 cm^-1 for CAS(8,12) SA-3).
  State-specific E0 is the meaningful number.
* socutils note: the super-CI step is projected onto Kramers-symmetric
  rotations and needs phase-clean Kramers pairs.  Started from the
  linear-symmetry SCF of the original inputs, whose (omega, -omega) partners
  have arbitrary phases, the optimization stalled (50 iterations, gradient
  0.035, 99 % of every step projected out, energy 1.1 mEh too high) and gave
  a CAS(2,4) SA-3 splitting of 0.97 cm^-1; the example therefore uses KRHF
  and stops if the CASSCF is not converged.  The default gradient threshold
  (1e-4) leaves the splitting uncertain by 0.004 cm^-1 (CASSCF) to 0.01 cm^-1
  (NEVPT2).

## Pr3+ 4f^2 free-ion levels (`examples/02-pr3_levels.py`)

X2CAMF, x2c-SVPall; CAHF orbitals of 4f^2 (from closed-shell Pr5+); 1s-4p
frozen, 4d 5s 5p correlated, 14 active 4f spinors, 102 virtual spinors; model
space = all 91 states of 4f^2, equal weights (Slurm job 31468668,
`~/scr4/zmr_runs/pr3_levels`, with a common E0).

Symmetry: CASCI, UC- and PC-NEVPT2 all give 13 levels with degeneracies
2J+1 (sum 91); the largest spread within a level is 2.9e-11 Eh (CASCI),
2.7e-11 Eh (UC), 2.4e-11 Eh (PC).  With the complete model space the union PC
space is the whole target sector, so PC = UC (E2 of the ground level
-0.525380 Eh in both).

Levels in cm^-1 (experiment: NIST ASD, Pr IV; quoted for orientation).  The
first NEVPT2 column is the common-E0 result of the runs above; the second one
uses state-specific E0 with the same orbitals and model space (Slurm job
31496212, `~/scr4/zmr_runs/pr3_e0`, and the example itself, job 31496517;
degeneracies 2J+1 to 3.1e-11 Eh, E2 of the ground level -0.520385 Eh):

| level | experiment | CASCI | NEVPT2, `e0='average'` | NEVPT2, `e0='state'` |
|-------|------------|-------|------------------------|----------------------|
| 3H4 | 0 | 0 | 0 | 0 |
| 3H5 | 2152 | 2160.6 | 2327.6 | 2140.9 |
| 3H6 | 4389 | 4456.2 | 4783.4 | 4399.2 |
| 3F2 | 4997 | 6758.7 | 6173.1 | 5600.2 |
| 3F3 | 6415 | 8124.0 | 7666.3 | 6975.1 |
| 3F4 | 6855 | 8153.1 | 7972.4 | 7273.7 |
| 1G4 | 9921 | 11073.3 | 11125.4 | 10180.3 |
| 1D2 | 17334 | 23428.3 | 22087.9 | 20058.4 |
| 3P0 | 21390 | 29434.9 | 27230.5 | 24643.2 |
| 3P1 | 22007 | 29989.9 | 27853.5 | 25218.5 |
| 1I6 | 22212 | 28060.7 | 28237.0 | 25817.7 |
| 3P2 | 23161 | 31086.2 | 29091.5 | 26363.8 |
| 1S0 | ~50090 | 64763.0 | 59907.8 | 53978.8 |

CASCI places 1I6 below 3P0; NEVPT2 restores the experimental order of all
13 levels with either E0.  The common E0 enlarges every level energy by a
fixed fraction of its CASCI value (conventions.md section 9, "Separability"):
`(E_average - E_state) / E_CASCI` is 0.085 - 0.092 for all twelve excited
levels, the norm of the first-order function.  It is the reason why the
spin-orbit intervals of 3H came out 8 % too large (3H5 2327.6 against 2152)
although CASCI already has them right; with state-specific E0 they are
2140.9 and 4399.2 (experiment 2152, 4389), and the 3F, 1G, 1D, 3P, 1I and 1S
levels move 500 - 6000 cm^-1 closer to experiment.  State-specific E0 is the
choice for level energies; the model states are CASCI eigenstates and
degenerate states share their E0, so the 2J+1 degeneracies are untouched.
Wall time for the 91-state runs on one node: UC 390 s and PC 277 s before the
engine optimizations, 32 s and 28 s after them (job 31469619), 4 s and 4 s
after the performance round with symmetry-adapted spinors (job 31488324);
identical levels each time.  The
remaining errors (up to 4000 cm^-1 with state-specific E0) are expected for a
basis without f/g correlation functions.

## Comparison with ORCA 6.0.1 FIC-NEVPT2 (`test_orca_reference`)

ORCA's FIC-NEVPT2 is Angeli's PC-NEVPT2 with the Dyall Hamiltonian (ORCA
prints "PT2 = FIC-NEVPT2 [PC-NEVPT2]").  ORCA CASSCF orbitals are exported
with `orca_2mkl -molden`; PySCF's molden reader rebuilds ORCA's basis
incorrectly, but the coefficients are exactly those of PySCF's own cc-pVDZ
(the AO metric implied by the orbitals, `(C C^T)^-1`, equals PySCF's overlap
to 4e-9), so the coefficients are used with PySCF's basis after Loewdin
re-orthonormalization.  CASCI in these orbitals reproduces ORCA's CASSCF
energy to < 1e-8.  Orbitals, ORCA inputs and reference numbers are stored in
`zmr/tests/data/orca`.  The real orbitals are also expanded to spin-orbitals
and rotated by a random complex unitary matrix inside the core, active and
virtual spaces (alpha/beta mixing, complex phases, max |Im h| = 9.5 for H2O).

Three ORCA conventions matter:

1. The 3-/4-RDM intermediates use CI truncation (defaults D3TPre = 1e-14,
   D4TPre = 1e-12), which changes caaa and aaav by 2-4e-9; all reference
   runs use D3TPre = D4TPre = 1e-30.
2. ORCA removes spin-free PC functions whose metric eigenvalue is below 1e-6
   (unchanged by the NThresh and DThresh keywords).  Nearly full or empty
   active orbitals give physical functions with eigenvalues of 4e-8 to 1.4e-6,
   which ORCA drops and zmr keeps (it removes only the round-off null space).
   `zmr/reference/spin_adapted_pc.py` builds ORCA's spin-free functions
   (`E_ai E_tu|0>`, `E_ti E_au|0>`, ...) by brute force: with a null-space
   threshold it reproduces zmr's class energy exactly, with 1e-6 it
   reproduces ORCA's energy and ORCA's count of removed functions.
3. After SA-CASSCF, ORCA's per-root NEVPT2 uses each root's own Fock matrix;
   zmr reproduces it with single-state runs on the SA orbitals.  (zmr's
   multi-state default uses the averaged Fock matrix, which is what makes the
   effective Hamiltonian invariant under model-space rotations.)

H2O/cc-pVDZ, CAS(4,4), all electrons correlated (Hartree):

| class | ORCA FIC | zmr PC | zmr - ORCA | spin-adapted, 1e-6 threshold |
|-------|----------|--------|------------|------------------------------|
| ccvv | -0.033745090396 | -0.033745090307 | 8.9e-11 | |
| ccav | -0.007781872250 | -0.007781872230 | 2.0e-11 | |
| cavv | -0.049678943965 | -0.049678943870 | 9.5e-11 | |
| ccaa | -0.004184198282 | -0.004184198324 | -4.2e-11 | |
| aavv | -0.026920840408 | -0.026920843335 | -2.9e-09 | -0.026920840679 |
| cava | -0.042578042950 | -0.042578878137 | -8.4e-07 | -0.042578042761 (612 of 1632 removed, as in ORCA) |
| caaa | -0.002926006436 | -0.002926006447 | -1.1e-11 | |
| aaav | -0.017915069220 | -0.017915069125 | 9.5e-11 | |

The remaining 1e-10 differences come from the precision of the molden
coefficients (orthonormal to 1.3e-8).  Complex rotation changes no class by
more than 2e-16; the RDM route agrees with the spectral route to 3e-15.

N2/cc-pVDZ, CAS(6,6): ccvv, ccav, cavv, ccaa, aavv and cava agree to
< 4e-10; caaa (-6.1e-9) and aaav (-4.0e-7) are the classes where ORCA's
threshold removes functions.  With the 1e-6 threshold the spin-adapted
reference gives -0.018881381076 for caaa (ORCA -0.018881381110) and
-0.016766309528 for aaav (ORCA -0.016766309647, 756 of 3888 functions
removed as in ORCA); with the null-space threshold it gives zmr's caaa
exactly and zmr's aaav to 9e-11 (zmr's default lindep = 1e-10 also drops a
few functions with eigenvalues between 1e-12 and 1e-10).  Slurm job
31473780.

SA-CASSCF(4,4) over three singlets of H2O, per-root FIC-NEVPT2: roots 1 and
2 agree in all classes to < 1.6e-10; root 0 agrees except ccaa (2.5e-8), aavv
(9.3e-8) and cava (1.8e-5), and the spin-adapted reference with the 1e-6
threshold gives ORCA's root-0 cava to 2.7e-10 (612 functions removed, as in
ORCA).  Complex rotation changes no element of the state-specific
multi-state matrix by more than 5e-14.

For comparison, ORCA's and PySCF's SC-NEVPT2 in the same orbitals (ORCA
default settings) differ by up to 4.5e-7 (aaav) and 1.1e-8 (caaa), the
classes that use the CI-truncated 3-/4-RDM intermediates.

## Non-relativistic spin-free NEVPT2 (`zmr.nr`; `test_nr_active`, `test_nr_nevpt2`)

Active space: the sector Hamiltonians `(n_alpha, n_beta)` equal PySCF's
(`fci.direct_spin1.pspace`, same determinant order and phases) and the
corresponding block of the spinor sector Hamiltonian to 1e-14; PySCF CI
vectors are eigenvectors; the spin-summed 1-RDM equals PySCF's; operator
strings agree with the spinor kernel on the embedded vectors exactly.

Against the spinor code on the spin-orbital form of the same Hamiltonian
(random spin-free model, 2 core + 3 active + 2 virtual spatial orbitals,
CAS(4,3), `M_S = 0`): UC agrees class by class to 5e-13 for three singlets,
for a triplet and for a mixed singlet + triplet model space, both E0
choices; PC (union and state) agrees to 7e-12 for singlets.  Two triplets
computed from their `M_S = 1`, `0`, `-1` components give the same UC,
PC-union and PC-state energies to 3e-14.  H2O / 6-31G CAS(4,4) through
`pyscf_nr.nr_from_casci` and PySCF's CI vector equals the GHF-expanded spinor
route to 1e-9 per class, and `E_UC <= E_PC <= E_SC(PySCF)` holds per class.

ORCA 6.0.1 FIC-NEVPT2 with ORCA's CASSCF orbitals (settings as in the
section above), triplet ground states, `zmr.nr` spin-free PC-NEVPT2 from the
`M_S = 1` component (the `M_S = 0` component gives the same numbers to
1e-12):

O2 X 3Sigma_g-, cc-pVDZ, CASSCF(8,6), all electrons correlated (Hartree;
CASCI energy vs ORCA CASSCF: 5e-11):

| class | ORCA FIC | zmr.nr PC | PC - ORCA | zmr.nr UC |
|-------|----------|-----------|-----------|-----------|
| ccvv | -0.014960404544 | -0.014960404528 | 1.6e-11 | -0.014960404528 |
| ccav | -0.003400579485 | -0.003400579461 | 2.4e-11 | -0.003401436834 |
| cavv | -0.041411012462 | -0.041411012381 | 8.1e-11 | -0.041506356803 |
| ccaa | -0.001316766540 | -0.001316766542 | -2.0e-12 | -0.001316766542 |
| aavv | -0.085096323286 | -0.085096323506 | -2.2e-10 | -0.085372580966 |
| cava | -0.058813340638 | -0.058813340417 | 2.2e-10 | -0.058824778875 |
| caaa | -0.002284616245 | -0.002284616197 | 4.8e-11 | -0.002284616197 |
| aaav | -0.044193533518 | -0.044194024973 | -4.9e-07 | -0.044200123528 |
| total | -0.251476576718 | -0.251477068005 | -4.9e-07 | -0.251867064273 |

CH2 3B1, cc-pVDZ, CASSCF(6,6) (CASCI energy vs ORCA: 9e-10): ccvv, ccav,
cavv, ccaa and cava agree to < 8e-12, aavv to 1.5e-10, caaa to 3.8e-10;
aaav differs by -5.3e-07.  In both molecules aaav is the class where ORCA's
1e-6 metric threshold removes functions (252 of 3888 and 425 of 3672), as in
the closed-shell cases.

Spin-orbital contraction for comparison (the spinor code on the same O2
Hamiltonian; second-order energy of the triplet):

| model space and contraction | E2 / Hartree |
|-----------------------------|--------------|
| spin-free PC (`zmr.nr`, any component) | -0.2514770680 |
| spinor code, `M_S = 0` component alone | -0.2515253054 |
| spinor code, the three components, union contraction | -0.2517135 (degenerate to 3e-14) |
| UC | -0.2518670643 |
| spinor code, `M_S = 1` component alone | -0.2534030759 |

The single strings of the spinor code span more than the spin-free functions
when `S > 0`, so its PC energy lies between spin-free PC and UC, and it is
invariant only if the model space contains every component of the multiplet.
A single `M_S != 0` component additionally gives the spinor code a
spin-polarized Fock matrix, i.e. another `H_0` (last row, below UC of the
spin-free `H_0`; even ccvv changes, by 1.9e-07).  For a singlet the two
contractions coincide.

O2 / cc-pVTZ, SA-CASSCF(8,6) over the four lowest `M_S = 0` states, one
effective Hamiltonian for X 3Sigma_g-, a 1Delta_g (two components) and
b 1Sigma_g+ (`examples/03-nr_o2_states.py`; experiment 0.98 and 1.64 eV):

| | a 1Delta_g / eV | b 1Sigma_g+ / eV |
|--|----------------|------------------|
| SA-CASSCF | 0.905, 0.905 | 1.683 |
| PC-NEVPT2 | 1.032, 1.032 | 1.902 |
| UC-NEVPT2 | 1.040, 1.040 | 1.907 |
