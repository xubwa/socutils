// The single place where excitations of a determinant are generated.
//
// ForEachExcitation walks the single and double excitations of a determinant
// whose Hamiltonian element exceeds a cutoff and hands each one to a sink.
// Variational selection, perturbation theory and anything else that needs
// "the important determinants connected to D" differ only in the sink.
//
// A filter decides which excitations are allowed at all. This is the hook for
// symmetry and for occupation restrictions. For every determinant the
// generator asks the filter for a view, filter.ForDeterminant(occ), and puts
// the questions Single(i, a) and Double(i, j, a, b) to that view; a filter
// that needs to know the determinant (RestrictedFilter) prepares it there.
// Any type with this interface can be used.

#ifndef ZSHCI_SRC_EXCITE_H_
#define ZSHCI_SRC_EXCITE_H_

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "det.h"
#include "hamiltonian.h"
#include "heat_bath.h"
#include "integrals.h"
#include "labels.h"

namespace zshci {

// Allows an excitation when it conserves every label set. One set with
// labels 2*omega is the angular momentum projection of a linear molecule
// (mod = 0) or the label of an n-fold axis (mod = 2n); a second set with
// labels 0 and 1 modulo 2 adds inversion; the spin projection together with
// a point group is likewise two sets. The sets are independent, which a
// single label with a single modulus cannot express.
class LabelFilter {
 public:
  explicit LabelFilter(const std::vector<LabelSet>& sets) {
    // A set whose labels are all equal conserves itself.
    for (const LabelSet& set : sets) {
      bool trivial = true;
      for (int label : set.labels) trivial = trivial && label == set.labels[0];
      if (trivial) continue;
      if (num_orbs_ == 0) num_orbs_ = set.labels.size();
      if (set.labels.size() != num_orbs_) {
        throw std::invalid_argument("label sets of different lengths");
      }
      mods_.push_back(set.mod);
    }
    const size_t num_sets = mods_.size();
    labels_.resize(num_orbs_ * num_sets);
    size_t k = 0;
    for (const LabelSet& set : sets) {
      bool trivial = true;
      for (int label : set.labels) trivial = trivial && label == set.labels[0];
      if (trivial) continue;
      for (size_t p = 0; p < num_orbs_; ++p) {
        labels_[p * num_sets + k] = set.labels[p];
      }
      ++k;
    }
  }
  LabelFilter(std::vector<int> labels, int mod)
      : LabelFilter(std::vector<LabelSet>{{std::move(labels), mod}}) {}

  // Declares that the double excitations offered to this filter conserve
  // the labels already, so that Double() need not look. That is the case
  // when they come from heat-bath tables of integrals held in the blocks of
  // the same label sets (Integrals::symmetry()).
  void set_doubles_conserved(bool conserved) { doubles_conserved_ = conserved; }

  // Label conservation does not depend on the determinant.
  const LabelFilter& ForDeterminant(const std::vector<int>& /*occ*/) const {
    return *this;
  }

  bool Single(int i, int a) const {
    const size_t num_sets = mods_.size();
    const int* const li = labels_.data() + i * num_sets;
    const int* const la = labels_.data() + a * num_sets;
    for (size_t k = 0; k < num_sets; ++k) {
      if (!LabelConserved(la[k] - li[k], mods_[k])) return false;
    }
    return true;
  }
  bool Double(int i, int j, int a, int b) const {
    if (doubles_conserved_) return true;
    const size_t num_sets = mods_.size();
    const int* const li = labels_.data() + i * num_sets;
    const int* const lj = labels_.data() + j * num_sets;
    const int* const la = labels_.data() + a * num_sets;
    const int* const lb = labels_.data() + b * num_sets;
    for (size_t k = 0; k < num_sets; ++k) {
      if (!LabelConserved(la[k] + lb[k] - li[k] - lj[k], mods_[k]))
        return false;
    }
    return true;
  }

 private:
  size_t num_orbs_ = 0;
  // labels_[p * (number of sets) + k]: label of spinor p in set k.
  std::vector<int> labels_;
  std::vector<int> mods_;
  bool doubles_conserved_ = false;
};

// A bound on the number of electrons in a set of spinors.
struct OccupationRestriction {
  std::vector<int> orbitals;
  int min_elec = 0;
  int max_elec = 0;
};

// Label conservation plus occupation restrictions: an excitation is allowed
// only if the excited determinant keeps the number of electrons of every
// restricted set within its bounds. With restricted active spaces this
// confines the calculation to a RAS-type space.
class RestrictedFilter {
 public:
  RestrictedFilter(LabelFilter labels,
                   const std::vector<OccupationRestriction>& restrictions,
                   int num_orbs)
      : labels_(std::move(labels)), membership_(num_orbs, 0) {
    if (restrictions.size() > 64) {
      throw std::invalid_argument("at most 64 occupation restrictions");
    }
    for (size_t r = 0; r < restrictions.size(); ++r) {
      min_elec_.push_back(restrictions[r].min_elec);
      max_elec_.push_back(restrictions[r].max_elec);
      for (int p : restrictions[r].orbitals) {
        if (p < 0 || p >= num_orbs) {
          throw std::invalid_argument("restriction: spinor index out of range");
        }
        membership_[p] |= 1ULL << r;
      }
    }
  }

  bool has_restrictions() const { return !min_elec_.empty(); }

  // Whether a determinant with these occupied spinors is inside the bounds.
  bool Allows(const std::vector<int>& occ) const {
    const View view = ForDeterminant(occ);
    for (size_t r = 0; r < min_elec_.size(); ++r) {
      if (!InBounds(r, view.count_[r])) return false;
    }
    return true;
  }

  // The filter as seen from one determinant.
  class View {
   public:
    bool Single(int i, int a) const {
      if (!filter_.labels_.Single(i, a)) return false;
      const uint64_t changed =
          filter_.membership_[i] ^ filter_.membership_[a];
      return changed == 0 || Check(changed, i, -1, a, -1);
    }
    bool Double(int i, int j, int a, int b) const {
      if (!filter_.labels_.Double(i, j, a, b)) return false;
      const uint64_t touched = filter_.membership_[i] | filter_.membership_[j] |
                               filter_.membership_[a] | filter_.membership_[b];
      return touched == 0 || Check(touched, i, j, a, b);
    }

   private:
    friend class RestrictedFilter;
    explicit View(const RestrictedFilter& filter) : filter_(filter) {}

    // Checks the restrictions in `mask` after moving electrons out of i, j
    // and into a, b (-1: none).
    bool Check(uint64_t mask, int i, int j, int a, int b) const {
      while (mask != 0) {
        const int r = __builtin_ctzll(mask);
        mask &= mask - 1;
        const int count = count_[r] - filter_.In(r, i) - filter_.In(r, j) +
                          filter_.In(r, a) + filter_.In(r, b);
        if (!filter_.InBounds(r, count)) return false;
      }
      return true;
    }

    const RestrictedFilter& filter_;
    int count_[64];
  };

  View ForDeterminant(const std::vector<int>& occ) const {
    View view(*this);
    for (size_t r = 0; r < min_elec_.size(); ++r) view.count_[r] = 0;
    for (int p : occ) {
      uint64_t mask = membership_[p];
      while (mask != 0) {
        ++view.count_[__builtin_ctzll(mask)];
        mask &= mask - 1;
      }
    }
    return view;
  }

 private:
  int In(int r, int p) const {
    return p >= 0 && ((membership_[p] >> r) & 1ULL) ? 1 : 0;
  }
  bool InBounds(size_t r, int count) const {
    return count >= min_elec_[r] && count <= max_elec_[r];
  }

  LabelFilter labels_;
  // Bit r of membership_[p]: spinor p belongs to restriction r.
  std::vector<uint64_t> membership_;
  std::vector<int> min_elec_;
  std::vector<int> max_elec_;
};

// Calls, for the determinant `det` with occupied spinors `occ` (increasing):
//   sink.Single(i, a, element)        for i -> a, element = <new|H|det>,
//                                     if |element| > cutoff;
//   sink.Double(i, j, a, b, magnitude) for (i, j) -> (a, b), i < j, a < b,
//                                     if magnitude = |<new|H|det>| >= cutoff.
// Excitations rejected by the filter are skipped.
template <int W, class Filter, class Sink, typename S>
void ForEachExcitation(const Det<W>& det, const std::vector<int>& occ,
                       double cutoff, const Integrals<S>& ints,
                       const HeatBath& heat_bath, const Filter& filter,
                       Sink* sink) {
  const int num_orbs = ints.num_orbs();
  const auto& view = filter.ForDeterminant(occ);
  Excitation single;
  single.rank = 1;
  for (int i : occ) {
    single.holes[0] = i;
    for (int a = 0; a < num_orbs; ++a) {
      if (det.Occupied(a) || !view.Single(i, a)) continue;
      single.particles[0] = a;
      const S element = OffDiagonalElement(det, single, ints);
      if (std::abs(element) > cutoff) sink->Single(i, a, element);
    }
  }

  if (heat_bath.max_magnitude() < cutoff) return;
  for (size_t n = 1; n < occ.size(); ++n) {
    const int j = occ[n];
    for (size_t m = 0; m < n; ++m) {
      const int i = occ[m];
      const HeatBath::Entry* const end = heat_bath.end(i, j);
      for (const HeatBath::Entry* e = heat_bath.begin(i, j); e != end; ++e) {
        if (e->magnitude < cutoff) break;
        if (det.Occupied(e->a) || det.Occupied(e->b)) continue;
        if (!view.Double(i, j, e->a, e->b)) continue;
        sink->Double(i, j, e->a, e->b, e->magnitude);
      }
    }
  }
}

}  // namespace zshci

#endif  // ZSHCI_SRC_EXCITE_H_
