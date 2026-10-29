// Functions of definite behaviour under the exchange of the two spins
// ("spin_flip", non-relativistic calculations with spatial orbitals).
//
// The spin orbitals of spatial orbital P are 2P (alpha) and 2P+1 (beta), and
// a determinant is the product of its creation operators in increasing
// order. The exchange of the spins, P_s: a+(2P) <-> a+(2P+1), commutes with a
// spin-free Hamiltonian and squares to one; it takes a determinant D (alpha
// string a, beta string b) to +/- the determinant Dbar (alpha string b, beta
// string a):
//
//   P_s |D> = p(D) |Dbar>,   p(D) = (-1)^(number of doubly occupied orbitals)
//
// (the two operators of a doubly occupied orbital change places; all others
// keep their order). On the states of Ms = 0 with N electrons, P_s is
// (-1)^(N/2) times the rotation by pi about the y axis, which gives (-1)^S on
// a state of total spin S, so
//
//   P_s |Psi> = lambda |Psi>,  lambda = (-1)^(S + N/2),
//   c(Dbar) = lambda p(D) c(D)  for the coefficients of |Psi>.
//
// "spin_flip": +1 asks for even S, -1 for odd S. The calculation works in
// the basis of the functions
//
//   |Phi_D> = (|D> + lambda p(D) |Dbar>) / sqrt(2)   for D != Dbar,
//   |Phi_D> = |D>                                    for D == Dbar,
//
// one per pair, labelled by its canonical member, the one that is the
// smaller of D and Dbar as a bit string. A determinant equal to its flip
// (alpha string = beta string) has p(D) = (-1)^(N/2) and belongs to the
// space only if lambda p(D) = 1, which is for even S.
//
// The list of determinants keeps both members of every pair, the canonical
// one first and its partner right after it (the layout), so that the string
// tables and the connection search see a list closed under the flip, and so
// that perturbation theory, density matrices and the files work with plain
// determinants and the expanded vector. What is halved is everything over
// the functions: the vectors of the eigensolver and the Hamiltonian, whose
// rows are those of the canonical determinants. With w(j) the coefficient of
// determinant j in its function (1/sqrt(2), lambda p(D)/sqrt(2) or 1) and
// rep(j) the function of j,
//
//   expanded vector:  c_j = w(j) x_rep(j),
//   <Phi_J|H|Phi_I> = (1 / w(c_J)) sum_{i in I} w(i) H(c_J, i),
//
// c_J the canonical member of J; for I < J both members of I precede c_J in
// the list, so the lower triangle of the matrix over the functions comes
// from the lower triangle of the rows of the canonical determinants. The
// diagonal element is H(c,c) + lambda p(c) H(c, cbar).

#ifndef ZSHCI_SRC_SPIN_FLIP_H_
#define ZSHCI_SRC_SPIN_FLIP_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "det.h"
#include "hamiltonian.h"
#include "integrals.h"
#include "parallel.h"
#include "scalar.h"
#include "sparse_h.h"

namespace zshci {

constexpr uint64_t kEvenBits = 0x5555555555555555ULL;

// The determinant with the spins exchanged (without the sign).
template <int W>
Det<W> FlipSpins(const Det<W>& det) {
  Det<W> out;
  for (int k = 0; k < W; ++k) {
    const uint64_t x = det.words[k];
    out.words[k] = ((x & kEvenBits) << 1) | ((x >> 1) & kEvenBits);
  }
  return out;
}

// Number of doubly occupied spatial orbitals.
template <int W>
int NumDoublyOccupied(const Det<W>& det) {
  int n = 0;
  for (int k = 0; k < W; ++k) {
    const uint64_t x = det.words[k];
    n += __builtin_popcountll(x & (x >> 1) & kEvenBits);
  }
  return n;
}

// Alpha minus beta electrons (2 Ms).
template <int W>
int TwiceSpinProjection(const Det<W>& det) {
  int n = 0;
  for (int k = 0; k < W; ++k) {
    n += __builtin_popcountll(det.words[k] & kEvenBits) -
         __builtin_popcountll(det.words[k] & ~kEvenBits);
  }
  return n;
}

// The canonical member of the pair of `det`.
template <int W>
Det<W> CanonicalOfPair(const Det<W>& det) {
  const Det<W> flipped = FlipSpins(det);
  return flipped < det ? flipped : det;
}

// lambda = spin_flip * (-1)^(N/2).
inline int SpinFlipLambda(int spin_flip, int num_elec) {
  return (num_elec / 2) % 2 == 0 ? spin_flip : -spin_flip;
}

// Whether a determinant equal to its flip may be in the space.
template <int W>
bool SelfPairAllowed(const Det<W>& det, int lambda) {
  const int p = NumDoublyOccupied(det) % 2 == 0 ? 1 : -1;
  return lambda * p == 1;
}

// Appends the pairs of `dets` (any members, possibly repeated) to `out` in
// the layout: canonical member, then its partner unless it is its own flip.
// Pairs whose canonical member is in `skip` (sorted) or was seen before, and
// determinants equal to their flip that the symmetry excludes, are left out.
// `source`, if given, receives for every appended determinant its position
// in `dets`, or -1 if only its partner was there.
template <int W>
void AppendPairs(const std::vector<Det<W>>& dets, int lambda,
                 std::vector<Det<W>>* out, std::vector<long>* source = nullptr,
                 const std::vector<Det<W>>* skip = nullptr) {
  // (canonical, position of the first member seen), in order of appearance.
  std::vector<std::pair<Det<W>, long>> seen;
  seen.reserve(dets.size());
  for (size_t d = 0; d < dets.size(); ++d) {
    seen.emplace_back(CanonicalOfPair(dets[d]), static_cast<long>(d));
  }
  std::stable_sort(
      seen.begin(), seen.end(),
      [](const std::pair<Det<W>, long>& a, const std::pair<Det<W>, long>& b) {
        return a.first < b.first;
      });
  // Positions in `dets` of each member, for `source`.
  std::vector<std::pair<Det<W>, long>> where;
  if (source != nullptr) {
    for (size_t d = 0; d < dets.size(); ++d) {
      where.emplace_back(dets[d], static_cast<long>(d));
    }
    std::sort(
        where.begin(), where.end(),
        [](const std::pair<Det<W>, long>& a, const std::pair<Det<W>, long>& b) {
          return a.first < b.first;
        });
  }
  auto position_of = [&](const Det<W>& det) -> long {
    const auto found =
        std::lower_bound(where.begin(), where.end(), det,
                         [](const std::pair<Det<W>, long>& a, const Det<W>& d) {
                           return a.first < d;
                         });
    return found != where.end() && found->first == det ? found->second : -1;
  };
  // Unique canonical members, ordered by their first appearance.
  std::vector<std::pair<long, Det<W>>> unique;
  for (size_t k = 0; k < seen.size(); ++k) {
    if (k > 0 && seen[k].first == seen[k - 1].first) continue;
    unique.emplace_back(seen[k].second, seen[k].first);
  }
  seen = std::vector<std::pair<Det<W>, long>>();
  std::sort(unique.begin(), unique.end(),
            [](const std::pair<long, Det<W>>& a,
               const std::pair<long, Det<W>>& b) { return a.first < b.first; });
  for (const std::pair<long, Det<W>>& item : unique) {
    const Det<W>& canonical = item.second;
    if (skip != nullptr &&
        std::binary_search(skip->begin(), skip->end(), canonical)) {
      continue;
    }
    const Det<W> partner = FlipSpins(canonical);
    if (partner == canonical && !SelfPairAllowed(canonical, lambda)) continue;
    out->push_back(canonical);
    if (source != nullptr) source->push_back(position_of(canonical));
    if (partner != canonical) {
      out->push_back(partner);
      if (source != nullptr) source->push_back(position_of(partner));
    }
  }
}

// What every position of a list in the layout is: the function it belongs
// to and its coefficient in that function.
class SpinFlipLayout {
 public:
  SpinFlipLayout() = default;
  SpinFlipLayout(int spin_flip, int num_elec)
      : lambda_(SpinFlipLambda(spin_flip, num_elec)) {}

  int lambda() const { return lambda_; }
  size_t num_dets() const { return kind_.size(); }
  size_t num_functions() const { return canonical_.size(); }

  // Takes in the determinants from first_new on; throws if they are not in
  // the layout.
  template <int W>
  void Extend(Span<Det<W>> dets, size_t first_new) {
    if (first_new != kind_.size()) {
      throw std::logic_error("SpinFlipLayout: extended out of order");
    }
    const size_t n = dets.size();
    kind_.resize(n);
    rep_.resize(n);
    for (size_t j = first_new; j < n; ++j) {
      const Det<W>& det = dets[j];
      if (TwiceSpinProjection(det) != 0) {
        throw std::runtime_error("spin_flip: determinant " + std::to_string(j) +
                                 " does not have Ms = 0");
      }
      const Det<W> partner = FlipSpins(det);
      if (det < partner || det == partner) {
        if (det == partner) {
          if (!SelfPairAllowed(det, lambda_)) {
            throw std::runtime_error(
                "spin_flip: determinant " + std::to_string(j) +
                " equals its flip and has the other symmetry");
          }
          kind_[j] = kSelf;
        } else {
          if (j + 1 >= n || dets[j + 1] != partner) {
            throw std::runtime_error("spin_flip: determinant " +
                                     std::to_string(j) +
                                     " is not followed by its flip");
          }
          kind_[j] = kFirst;
        }
        rep_[j] = static_cast<DetIndex>(canonical_.size());
        canonical_.push_back(j);
      } else {
        if (j == 0 || dets[j - 1] != partner || kind_[j - 1] != kFirst) {
          throw std::runtime_error("spin_flip: determinant " +
                                   std::to_string(j) +
                                   " does not follow its flip");
        }
        const int p = NumDoublyOccupied(partner) % 2 == 0 ? 1 : -1;
        kind_[j] = lambda_ * p > 0 ? kPartnerPlus : kPartnerMinus;
        rep_[j] = rep_[j - 1];
      }
    }
  }

  DetIndex rep(size_t j) const { return rep_[j]; }
  bool is_canonical(size_t j) const {
    return kind_[j] == kFirst || kind_[j] == kSelf;
  }
  // Position of the canonical member of function J.
  size_t canonical(size_t J) const { return canonical_[J]; }
  // Whether function J has two members.
  bool is_pair(size_t J) const { return kind_[canonical_[J]] == kFirst; }
  // Coefficient of determinant j in its function.
  double weight(size_t j) const {
    switch (kind_[j]) {
      case kFirst:
      case kPartnerPlus:
        return kInvSqrt2;
      case kPartnerMinus:
        return -kInvSqrt2;
      default:
        return 1.0;
    }
  }
  // lambda p(c) of a pair: the sign of the partner relative to the
  // canonical member (that of a single determinant: 1).
  double partner_sign(size_t J) const {
    const size_t c = canonical_[J];
    return kind_[c] == kFirst && kind_[c + 1] == kPartnerMinus ? -1.0 : 1.0;
  }

  // out[j] = w(j) x[rep(j)] over the list.
  template <typename S>
  void Expand(const S* x, S* out) const {
    const long n = static_cast<long>(kind_.size());
#pragma omp parallel for schedule(static)
    for (long j = 0; j < n; ++j) out[j] = weight(j) * x[rep_[j]];
  }
  template <typename S>
  std::vector<S> Expand(const std::vector<S>& x) const {
    std::vector<S> out(kind_.size());
    Expand(x.data(), out.data());
    return out;
  }
  // x[J] = sum_{j in J} w(j) full[j]: the projection of a vector over the
  // list onto the functions.
  template <typename S>
  std::vector<S> Project(const S* full) const {
    std::vector<S> x(canonical_.size());
    const long m = static_cast<long>(canonical_.size());
#pragma omp parallel for schedule(static)
    for (long J = 0; J < m; ++J) {
      const size_t c = canonical_[J];
      S sum = weight(c) * full[c];
      if (kind_[c] == kFirst) sum += weight(c + 1) * full[c + 1];
      x[J] = sum;
    }
    return x;
  }

  // <Phi_J|H|Phi_J>.
  template <int W, class Ints, typename S = typename Ints::Scalar>
  double Diagonal(Span<Det<W>> dets, size_t J, const Ints& ints) const {
    const size_t c = canonical_[J];
    double value = DiagonalEnergy(dets[c], ints);
    if (kind_[c] == kFirst) {
      value += partner_sign(J) * Real(Hij(dets[c], dets[c + 1], ints));
    }
    return value;
  }
  // <Phi_J|H|Phi_I>, J != I.
  template <int W, class Ints, typename S = typename Ints::Scalar>
  S Element(Span<Det<W>> dets, size_t J, size_t I, const Ints& ints) const {
    const size_t c = canonical_[J];
    const size_t d = canonical_[I];
    S value = weight(d) * Hij(dets[c], dets[d], ints);
    if (kind_[d] == kFirst)
      value += weight(d + 1) * Hij(dets[c], dets[d + 1], ints);
    return value / weight(c);
  }

 private:
  enum Kind : uint8_t { kFirst, kSelf, kPartnerPlus, kPartnerMinus };
  static constexpr double kInvSqrt2 = 0.70710678118654752440;

  int lambda_ = 1;
  std::vector<uint8_t> kind_;
  std::vector<DetIndex> rep_;
  std::vector<size_t> canonical_;
};

// Merges the entries of equal column of a row sorted by column (the two
// members of a function).
template <typename S>
void MergeEqualColumns(std::vector<std::pair<DetIndex, S>>* row) {
  size_t out = 0;
  for (size_t k = 0; k < row->size(); ++k) {
    if (out > 0 && (*row)[out - 1].first == (*row)[k].first) {
      (*row)[out - 1].second += (*row)[k].second;
    } else {
      (*row)[out++] = (*row)[k];
    }
  }
  row->resize(out);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_SPIN_FLIP_H_
