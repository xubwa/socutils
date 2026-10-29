// Sorted tables of double-excitation integrals for heat-bath selection.
//
// For every pair of spinors i < j the table lists the pairs a < b with
//   g = <ab||ij>,
// the magnitude of the matrix element of the double excitation
// (i, j) -> (a, b), in order of decreasing |g|. Generating all double
// excitations of a determinant whose element exceeds a cutoff then costs time
// proportional to their number: walk each list until |g| drops below the
// cutoff.
//
// With integrals over spatial orbitals (Integrals::Spatial) the lists are
// kept per pair of spatial orbitals and serve every spin combination:
//   same spin       I < J:   pairs A < B with g = (AI|BJ) - (AJ|BI),
//                            for alpha-alpha and for beta-beta;
//   opposite spin   I <= J:  pairs (A, B) with g = (AI|BJ), I -> A, J -> B,
//                            for either orbital being the alpha one.
// That is 3/4 n^4 entries for n spatial orbitals instead of the 3/2 n^4 that
// lists per pair of spin orbitals would take.

#ifndef ZSHCI_SRC_HEAT_BATH_H_
#define ZSHCI_SRC_HEAT_BATH_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm.h"
#include "integrals.h"

namespace zshci {

class HeatBath {
 public:
  struct Entry {
    double magnitude;
    uint16_t a;
    uint16_t b;
  };

  // Entries with |g| <= threshold are left out. With several ranks this is
  // collective: the root builds the tables and they are held once per node.
  // Instantiated for the spinor and the spatial integrals, real and
  // complex.
  template <class Ints>
  HeatBath(const Ints& ints, double threshold, const Comm& comm = Comm());

  // Calls visit(a, b, magnitude), a < b, for the double excitations
  // (i, j) -> (a, b) of the spinors i < j with magnitude >= cutoff, in order
  // of decreasing magnitude.
  //
  // With `below` finite only the excitations with magnitude < below are
  // visited (the band [cutoff, below)); the entries above are skipped by
  // bisection, so that walking consecutive bands costs one walk in all.
  template <class Visit>
  void ForEachDouble(int i, int j, double cutoff, const Visit& visit,
                     double below = kNoBound) const {
    if (!spatial_) {
      const Entry* const end = entries_.data() + start_[PairIndex(i, j) + 1];
      for (const Entry* e =
               First(entries_.data() + start_[PairIndex(i, j)], end, below);
           e != end && e->magnitude >= cutoff; ++e) {
        visit(e->a, e->b, e->magnitude);
      }
      return;
    }
    const int spin_i = i & 1;
    const int spin_j = j & 1;
    const int orb_i = i >> 1;
    const int orb_j = j >> 1;
    if (spin_i == spin_j) {
      const Entry* const end =
          entries_.data() + start_[PairIndex(orb_i, orb_j) + 1];
      for (const Entry* e = First(
               entries_.data() + start_[PairIndex(orb_i, orb_j)], end, below);
           e != end && e->magnitude >= cutoff; ++e) {
        visit(2 * e->a + spin_i, 2 * e->b + spin_i, e->magnitude);
      }
      return;
    }
    // Opposite spins. Since i < j the spatial orbital of i is not above
    // that of j, and the entry (a, b) moves i to a and j to b.
    const size_t pair = opposite_offset_ + PairIndexWithDiagonal(orb_i, orb_j);
    const Entry* const end = entries_.data() + start_[pair + 1];
    for (const Entry* e = First(entries_.data() + start_[pair], end, below);
         e != end && e->magnitude >= cutoff; ++e) {
      const int a = 2 * e->a + spin_i;
      const int b = 2 * e->b + spin_j;
      if (a < b) {
        visit(a, b, e->magnitude);
      } else {
        visit(b, a, e->magnitude);
      }
    }
  }

  // Largest magnitude in the whole table.
  double max_magnitude() const { return max_magnitude_; }
  size_t num_entries() const { return entries_.size(); }

  static constexpr double kNoBound = 1e300;

 private:
  // The first entry of [begin, end) below `below`.
  static const Entry* First(const Entry* begin, const Entry* end,
                            double below) {
    if (below >= kNoBound) return begin;
    return std::partition_point(
        begin, end, [&](const Entry& e) { return e.magnitude >= below; });
  }

  // Index of the pair i < j, and of the pair i <= j.
  static size_t PairIndex(int i, int j) {
    return static_cast<size_t>(j) * (j - 1) / 2 + i;
  }
  static size_t PairIndexWithDiagonal(int i, int j) {
    return static_cast<size_t>(j) * (j + 1) / 2 + i;
  }

  template <typename S>
  void BuildSpinorLists(const SpinorIntegrals<S>& ints, double threshold,
                        std::vector<std::vector<Entry>>* lists) const;
  template <typename S>
  void BuildSpatialLists(const SpatialIntegrals<S>& ints, double threshold,
                         std::vector<std::vector<Entry>>* lists) const;

  bool spatial_ = false;
  // With spatial orbitals the opposite-spin lists follow the same-spin ones;
  // this is the index of the first.
  size_t opposite_offset_ = 0;
  std::vector<size_t> start_;
  SharedArray<Entry> entries_;
  double max_magnitude_ = 0.0;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_HEAT_BATH_H_
