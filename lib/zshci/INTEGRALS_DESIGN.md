# One integral class per kind of orbitals

Status: spinor and restricted (spatial) kinds done (304d6b4); the
unrestricted kind next, as a third class.

## Today

`Integrals<S>` (`src/integrals.h`) serves three kinds of orbitals with one
storage and branches inside the hot accessors:

- spinors: antisymmetrized `<pq||rs>` in label blocks, about n^4/8 numbers;
- restricted spatial orbitals (`Spatial`): `(PQ|RS)` with eightfold
  symmetry over n spatial orbitals, `AntisymOrdered` branching on
  `num_spatial_ > 0` on every call and putting `<pq||rs>` together from two
  spatial integrals;
- unrestricted spin orbitals (`SpinOrbitals`): the spinor storage filled
  from h1_a, h1_b, eri_aa, eri_bb, eri_ab, about 1.5 n^4 numbers.

`is_spatial()` is also asked outside the class: the heat-bath tables
(`heat_bath.cc`, spatial lists), the density matrices (spin-summed for
spatial orbitals, `driver_common.h`, `rdm.h`), the energy from density
matrices (`rdm.h`) and the output (`main.cc`).

## Proposed layout

    IntegralsBase                       coarse, virtual: sizes, energies,
       |                                labels, description, memory, Share
    IntegralTables<S>                   common data, inline: one-body h,
       |                                pair energies, singles bound
       +-- SpinorIntegrals<S>           label blocks of <pq||rs>
       +-- SpatialIntegrals<double>     (PQ|RS), real only, eightfold
       +-- UnrestrictedIntegrals        later: aa, bb (eightfold), ab
                                        (fourfold), real

Every class has the same inline interface, used by the hot paths:

| member | meaning |
|---|---|
| `using Scalar` | double or Complex |
| `num_orbs()`, `num_elec()`, `core_energy()`, `symmetry()` | as today |
| `OneBody(p, q)`, `PairEnergy(p, q)`, `SingleBound(p, q)` | from `IntegralTables` |
| `AntisymOrdered(p, q, r, s)`, `Antisym(p, q, r, s)` | `<pq||rs>`, no branch on the kind |
| `kSpinOrbitals` (static) | orbitals 2P, 2P+1 with the spin projection as first label set |
| `kSpinSummedDensity` (static) | density matrices summed over spin (spatial only) |

and each supplies, as free functions overloaded on the class (no virtual
call): the heat-bath lists (`BuildHeatBathLists`), the energy from density
matrices (`TwoBodyEnergy`), and its reader. `IntegralsBase` keeps what is
called once: `num_orbs`, `num_elec`, `core_energy`, `symmetry`,
`num_stored`, `Describe()` (the line printed at the start),
`MemoryBytes()`, `Share(comm)`.

`Finalize()` (pair energies and the singles bound) is one template over the
class, run by each reader.

## What gets templated

The kernels take the integral class as a template parameter instead of the
number type; the number type follows from it (`typename Ints::Scalar`):

| file | functions |
|---|---|
| `hamiltonian.h` | `DiagonalEnergy`, `OffDiagonalElement`, `Hij`, signs unchanged |
| `excite.h` | `ForEachExcitation` |
| `heat_bath.h/.cc` | the constructor (lists per class) |
| `connect.h`, `connect_groups.h` | `BuildGroupBlocks`, `BuildBlocks`, `BuildBlock`, brute force |
| `direct_h.h` | `DirectH` |
| `spin_flip.h` | `SpinFlipLayout::Diagonal`, `Element` |
| `pt.h` | `PtProblem`, sinks, `PtDeterministic`, pooled and walker estimates, counting, pilot, choice |
| `rdm.h` | `EnergyFromDensityMatrices` |
| `driver_common.h` | `VariationalH`, `GuessVectors`, `CheckHamiltonian`, `AppendDiagonal`, `ComputePerturbation`, `WriteResults` |
| `select.h`, `diag.h` | `RunSelect`, `RunDiag`, the selectors |
| `main.cc` | `Run<W, Ints>`: the one dispatch, from the input |

`template <int W, typename S>` becomes `template <int W, class Ints>` with
`using S = typename Ints::Scalar;`. The fci mode (`fci.*`, `gas_space.*`)
takes `SpinorIntegrals<S>` only and is otherwise unchanged.

Not touched: `distributed_h.*` and the eigensolvers, which see a
`LinearOperator<S>` and never the integrals; `sparse_h.*`, `comm.*`,
`connect_bytes.h` (strings only). The variational-MPI work on
`distributed_h.cc` (branch zshci-varpar) is thus independent of this
refactor; `driver_common.h` changes only in its template parameters.

## Cost

Instantiations of the select/diag drivers: before, 4 widths x 2 number
types = 8; now 4 x (spinor double, spinor complex, spatial double) = 12,
all widths for every kind (cases of 128 spatial orbitals need 4 words).
Measured: `main.o` 311 -> 393 s on one core (550 s with complex spatial
integrals as well, dropped), `bin/zshci` 180 -> 209 MB. The width is a build
option (`make MAX_DET_WORDS=N`, dispatched by a compile-time recursion);
more words cost compile time and size roughly in proportion (width 6: 923 s,
308 MB) and nothing at run time for small determinants. Run time: the branch on the kind leaves
`AntisymOrdered`, otherwise the same arithmetic in the same order, so
energies, PT2 with a fixed seed and density matrices must be bitwise the
same (checked, with the speed on the 1.2e7 Cr2 point, spin_flip and plain).

## Unrestricted, afterwards

`UnrestrictedIntegrals` (done): one-body h_a, h_b; eri_aa, eri_bb with
eightfold symmetry (n^4/8 each), eri_ab with fourfold (n^4/4): n^4/2 numbers
against about 0.73 n^4 antisymmetrized in spin blocks (O2 cc-pVDZ, 26
orbitals: 246753 against 334776). `<pq||rs>` from the block of the spins of its two charge
distributions ((ab) or (ba) for opposite spins, the latter by swapping the
distributions). Heat-bath lists per pair of spatial orbitals: alpha-alpha
and beta-beta for I < J, alpha-beta for all (I, J) (no exchange of the
spins). Density matrices over spin orbitals; `spin_flip` rejected (the
spins see different orbitals).
