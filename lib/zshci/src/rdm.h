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
// The connected pairs of determinants come from a pair source: the sparsity
// pattern of a stored Hamiltonian (StoredPairs), which must then have been
// built without a threshold (ConnectOptions::threshold < 0), or the group
// string tables (TablePairs), which need no stored matrix.

#ifndef ZSHCI_SRC_RDM_H_
#define ZSHCI_SRC_RDM_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "comm.h"
#include "connect_groups.h"
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

// The connected pairs held in the blocks of a stored Hamiltonian.
template <typename S>
class StoredPairs {
 public:
  explicit StoredPairs(const SparseH<S>& h) : h_(h), first_row_(1, 0) {
    for (const CsrBlock<S>& block : h.blocks()) {
      first_row_.push_back(first_row_.back() + block.num_stored_rows());
    }
  }

  long num_rows() const { return static_cast<long>(first_row_.back()); }

  // Calls visit(i, j) for the pairs of row r.
  template <class Visitor>
  void ForRow(long r, const Visitor& visit) const {
    const size_t b = std::upper_bound(first_row_.begin(), first_row_.end(),
                                      static_cast<size_t>(r)) -
                     first_row_.begin() - 1;
    const CsrBlock<S>& block = h_.blocks()[b];
    const size_t local = r - first_row_[b];
    const DetIndex i = block.row[local];
    for (uint64_t k = block.row_ptr[local]; k < block.row_ptr[local + 1]; ++k) {
      visit(i, block.col[k]);
    }
  }

 private:
  const SparseH<S>& h_;
  std::vector<size_t> first_row_;  // of every block, among all stored rows
};

// The connected pairs this rank owns, worked out from the group string
// tables.
template <int W>
class TablePairs {
 public:
  TablePairs(const internal::GroupStringTables<W>& tables, size_t num_dets)
      : tables_(tables), num_dets_(num_dets) {}

  // A row is a determinant or, where the tables find the connections of
  // the determinants of a string together, a chunk of them.
  long num_rows() const {
    return static_cast<long>(tables_.has_chunks() ? tables_.num_chunks()
                                                  : num_dets_);
  }

  template <class Visitor>
  void ForRow(long r, const Visitor& visit) const {
    uint64_t tested = 0;
    if (tables_.has_chunks()) {
      tables_.ForEachConnectedOfChunk(
          r, [&](DetIndex j) { return tables_.Owns(j); },
          [&](int /*k*/, DetIndex j, DetIndex i) { visit(j, i); },
          [](const uint32_t* /*indices*/, int /*count*/) {}, &tested);
      return;
    }
    if (!tables_.Owns(r)) return;
    tables_.ForEachConnected(
        r, [&](DetIndex i) { visit(static_cast<DetIndex>(r), i); }, &tested);
  }

 private:
  const internal::GroupStringTables<W>& tables_;
  size_t num_dets_;
};

// Computes D1, and D2 if `with_two`, for the vectors `bra` and `ket`. With
// several ranks, `pairs` holds the connected pairs of this rank; every rank
// adds those, the root adds the diagonal part, and all ranks return the sum.
// `num_orbs` is the number of spinors; with `spin_summed` they are the spin
// orbitals of num_orbs / 2 spatial orbitals and the matrices are summed over
// the spins.
template <int W, typename S, class Pairs>
DensityMatrices<S> ComputeDensityMatricesFrom(Span<Det<W>> dets,
                                              const Pairs& pairs, int num_orbs,
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
  const long num_rows = pairs.num_rows();
#pragma omp parallel
  {
    std::vector<int> occ_i;
    std::vector<int> occ_j;
#pragma omp for schedule(dynamic, 64)
    for (long r = 0; r < num_rows; ++r) {
      pairs.ForRow(r, [&](DetIndex i, DetIndex j) {
        occ_i.clear();
        dets[i].AppendOccupied(&occ_i);
        occ_j.clear();
        dets[j].AppendOccupied(&occ_j);
        // <D_i| ... |D_j> and its transpose <D_j| ... |D_i>.
        internal::AddPair(dets[j], FindExcitation(dets[i], dets[j]), occ_j,
                          Conj(bra[i]) * ket[j], with_two, &dm);
        internal::AddPair(dets[i], FindExcitation(dets[j], dets[i]), occ_i,
                          Conj(bra[j]) * ket[i], with_two, &dm);
      });
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

// The same from the pairs of a stored Hamiltonian.
template <int W, typename S>
DensityMatrices<S> ComputeDensityMatrices(Span<Det<W>> dets,
                                          const SparseH<S>& h, int num_orbs,
                                          Span<S> bra, Span<S> ket,
                                          bool with_two,
                                          const Comm& comm = Comm(),
                                          bool spin_summed = false) {
  return ComputeDensityMatricesFrom(dets, StoredPairs<S>(h), num_orbs, bra, ket,
                                    with_two, comm, spin_summed);
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
