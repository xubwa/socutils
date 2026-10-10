// One- and two-body reduced (transition) density matrices.
//
// Definitions, for a bra vector b and a ket vector k:
//   D1(p, q)       = <b| p^+ q |k>
//   D2(p, q, r, s) = <b| p^+ q^+ s r |k>
// so that  E = sum_pq h(p,q) D1(p,q) + 1/4 sum_pqrs <pq||rs> D2(p,q,r,s).
//
// Spin-summed matrices. Over the spin orbitals 2P (alpha), 2P + 1 (beta) of
// spatial orbitals, the same quantities summed over the spins,
//   D1(P, Q)       = sum_s    <b| Ps^+ Qs |k>
//   D2(P, Q, R, S) = sum_s,t  <b| Ps^+ Qt^+ St Rs |k>,
// are accumulated directly, so that nothing of the size of the spin-orbital
// D2, sixteen times larger, is ever held.
//
// The connected pairs of determinants are taken from the sparsity pattern of
// the Hamiltonian, which must therefore have been built without a threshold
// (ConnectOptions::threshold < 0).

#ifndef ZSHCI_SRC_RDM_H_
#define ZSHCI_SRC_RDM_H_

#include <utility>
#include <vector>

#include "comm.h"
#include "det.h"
#include "hamiltonian.h"
#include "integrals.h"
#include "sparse_h.h"

namespace zshci {

// Dense tensors in row-major order: D1[p * n + q], D2[((p * n + q) * n + r) *
// n + s].
template <typename S>
struct DensityMatrices {
  // Number of spinors or, if spin_summed, of spatial orbitals.
  int num_orbs = 0;
  bool spin_summed = false;
  std::vector<S> one;
  std::vector<S> two;  // empty unless requested

  S& One(int p, int q) { return one[p * num_orbs + q]; }
  S& Two(int p, int q, int r, int s) {
    const size_t n = num_orbs;
    return two[((p * n + q) * n + r) * n + s];
  }
  S One(int p, int q) const { return one[p * num_orbs + q]; }
  S Two(int p, int q, int r, int s) const {
    const size_t n = num_orbs;
    return two[((p * n + q) * n + r) * n + s];
  }
};

namespace internal {

// Adds value to D1(p, q) over spinors.
template <typename S>
void AddOne(DensityMatrices<S>* dm, int p, int q, S value) {
  if (!dm->spin_summed) {
    AtomicAdd(&dm->One(p, q), value);
  } else if (((p ^ q) & 1) == 0) {
    AtomicAdd(&dm->One(p >> 1, q >> 1), value);
  }
}

// Adds value to D2(p, q, r, s) over spinors, which stands for the four
// elements related by antisymmetry: D2(q, p, s, r) = value and
// D2(q, p, r, s) = D2(p, q, s, r) = -value.
//
// Without spin summation only the canonical one, p < q and r < s, is stored
// here; the others are filled in at the end. With it, each of the four goes
// to the spatial element it belongs to, if its spins match.
template <typename S>
void AddCanonical(DensityMatrices<S>* dm, int p, int q, int r, int s, S value) {
  if (dm->spin_summed) {
    const int sp = p & 1, sq = q & 1, sr = r & 1, ss = s & 1;
    p >>= 1;
    q >>= 1;
    r >>= 1;
    s >>= 1;
    if (sp == sr && sq == ss) {
      AtomicAdd(&dm->Two(p, q, r, s), value);
      AtomicAdd(&dm->Two(q, p, s, r), value);
    }
    if (sq == sr && sp == ss) {
      AtomicAdd(&dm->Two(q, p, r, s), -value);
      AtomicAdd(&dm->Two(p, q, s, r), -value);
    }
    return;
  }
  if (p > q) {
    std::swap(p, q);
    value = -value;
  }
  if (r > s) {
    std::swap(r, s);
    value = -value;
  }
  AtomicAdd(&dm->Two(p, q, r, s), value);
}

// Contribution of the pair <D_bra| ... |D_ket> with weight conj(b_bra) k_ket.
template <int W, typename S>
void AddPair(const Det<W>& ket, const Excitation& exc,
             const std::vector<int>& ket_occ, S weight, bool with_two,
             DensityMatrices<S>* dm) {
  if (exc.rank == 1) {
    const int i = exc.holes[0];
    const int a = exc.particles[0];
    const S value = SingleSign(ket, i, a) * weight;
    AddOne(dm, a, i, value);
    if (!with_two) return;
    for (int k : ket_occ) {
      if (k != i) AddCanonical(dm, a, k, i, k, value);
    }
  } else if (exc.rank == 2 && with_two) {
    const int i = exc.holes[0];
    const int j = exc.holes[1];
    const int a = exc.particles[0];
    const int b = exc.particles[1];
    AddCanonical(dm, a, b, i, j, DoubleSign(ket, i, j, a, b) * weight);
  }
}

}  // namespace internal

// Computes D1, and D2 if `with_two`, for the vectors `bra` and `ket`. With
// several ranks, `h` is the local part of the Hamiltonian; every rank adds
// the pairs it holds, the root adds the diagonal part, and all ranks return
// the sum. `num_orbs` is the number of spinors; with `spin_summed` they are
// the spin orbitals of num_orbs / 2 spatial orbitals and the matrices are
// summed over the spins.
template <int W, typename S>
DensityMatrices<S> ComputeDensityMatrices(Span<Det<W>> dets,
                                          const SparseH<S>& h, int num_orbs,
                                          Span<S> bra, Span<S> ket,
                                          bool with_two,
                                          const Comm& comm = Comm(),
                                          bool spin_summed = false) {
  const size_t n = spin_summed ? num_orbs / 2 : num_orbs;
  DensityMatrices<S> dm;
  dm.num_orbs = static_cast<int>(n);
  dm.spin_summed = spin_summed;
  dm.one.assign(n * n, S(0.0));
  if (with_two) dm.two.assign(n * n * n * n, S(0.0));

  // Diagonal part: every determinant with itself. One rank does it.
  if (comm.is_root()) {
    const long num_dets = static_cast<long>(dets.size());
#pragma omp parallel
    {
      std::vector<int> occ;
#pragma omp for schedule(static)
      for (long i = 0; i < num_dets; ++i) {
        occ.clear();
        dets[i].AppendOccupied(&occ);
        const S weight = Conj(bra[i]) * ket[i];
        for (size_t m = 0; m < occ.size(); ++m) {
          internal::AddOne(&dm, occ[m], occ[m], weight);
          if (!with_two) continue;
          for (size_t l = 0; l < m; ++l) {
            internal::AddCanonical(&dm, occ[l], occ[m], occ[l], occ[m], weight);
          }
        }
      }
    }
  }

  // Off-diagonal part: the connected pairs this rank holds.
  for (const CsrBlock<S>& block : h.blocks()) {
    const long num_rows = static_cast<long>(block.num_stored_rows());
#pragma omp parallel
    {
      std::vector<int> occ_i;
      std::vector<int> occ_j;
#pragma omp for schedule(dynamic, 64)
      for (long r = 0; r < num_rows; ++r) {
        const DetIndex i = block.row[r];
        occ_i.clear();
        dets[i].AppendOccupied(&occ_i);
        for (uint64_t k = block.row_ptr[r]; k < block.row_ptr[r + 1]; ++k) {
          const DetIndex j = block.col[k];
          occ_j.clear();
          dets[j].AppendOccupied(&occ_j);
          // <D_i| ... |D_j> and its transpose <D_j| ... |D_i>.
          internal::AddPair(dets[j], FindExcitation(dets[i], dets[j]), occ_j,
                            Conj(bra[i]) * ket[j], with_two, &dm);
          internal::AddPair(dets[i], FindExcitation(dets[j], dets[i]), occ_i,
                            Conj(bra[j]) * ket[i], with_two, &dm);
        }
      }
    }
  }

  comm.Sum(reinterpret_cast<double*>(dm.one.data()),
           kDoublesPerScalar<S> * dm.one.size());
  comm.Sum(reinterpret_cast<double*>(dm.two.data()),
           kDoublesPerScalar<S> * dm.two.size());

  if (with_two && !spin_summed) {
    // Expand the canonical elements by antisymmetry.
    const long num = num_orbs;
#pragma omp parallel for schedule(static)
    for (long p = 0; p < num; ++p) {
      for (int q = p + 1; q < num; ++q) {
        for (int r = 0; r < num; ++r) {
          for (int s = r + 1; s < num; ++s) {
            const S value = dm.Two(p, q, r, s);
            dm.Two(p, q, s, r) = -value;
            dm.Two(q, p, r, s) = -value;
            dm.Two(q, p, s, r) = value;
          }
        }
      }
    }
  }
  return dm;
}

// Energy (without the core energy) from the density matrices of one state.
template <typename S>
double EnergyFromDensityMatrices(const DensityMatrices<S>& dm,
                                 const Integrals<S>& ints) {
  const int n = dm.num_orbs;
  if (dm.spin_summed) {
    // sum_PQ h(P,Q) D1(P,Q) + 1/2 sum_PQRS (PR|QS) D2(P,Q,R,S).
    double total = 0.0;
#pragma omp parallel for reduction(+ : total) schedule(static)
    for (int p = 0; p < n; ++p) {
      for (int q = 0; q < n; ++q) {
        total += Real(ints.OneBody(2 * p, 2 * q) * dm.One(p, q));
        for (int r = 0; r < n; ++r) {
          for (int s = 0; s < n; ++s) {
            total += 0.5 *
                     Real(ints.SpatialTwoBody(p, r, q, s) * dm.Two(p, q, r, s));
          }
        }
      }
    }
    return total;
  }
  S energy(0.0);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) energy += ints.OneBody(p, q) * dm.One(p, q);
  }
  // 1/4 sum_pqrs <pq||rs> D2(p,q,r,s), both factors being antisymmetric in
  // p, q and in r, s.
  double re = Real(energy);
#pragma omp parallel for reduction(+ : re) schedule(static)
  for (int q = 1; q < n; ++q) {
    for (int p = 0; p < q; ++p) {
      for (int s = 1; s < n; ++s) {
        for (int r = 0; r < s; ++r) {
          re += Real(ints.AntisymOrdered(p, q, r, s) * dm.Two(p, q, r, s));
        }
      }
    }
  }
  return re;
}

}  // namespace zshci

#endif  // ZSHCI_SRC_RDM_H_
