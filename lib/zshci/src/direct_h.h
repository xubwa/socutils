// The Hamiltonian applied without being stored.
//
// DirectH works out the connections of every determinant from the group
// string tables (connect_groups.h) each time it is applied, evaluating the
// elements as it goes. Each product costs about as much as building the
// stored matrix, but the memory is that of the tables: the way to go when
// the matrix does not fit.
//
// When the tables were made with the block-wise search (connect_groups.h),
// the connections are found string by string instead of determinant by
// determinant, which is several times faster; a rank then does its share
// of the tasks of that search instead of its share of the rows.
//
// Like the stored matrix, only one triangle is visited: an element H(j,i),
// i < j, found from determinant j, serves y_j += H(j,i) x_i and
// y_i += conj(H(j,i)) x_j. The second goes to an element of y that another
// thread may want to write. As long as it fits a memory budget, every thread
// therefore adds into a vector of its own and the vectors are summed at the
// end; beyond the budget the threads add into y itself, atomically, which
// needs no memory but is slower: the cache lines of y travel between the
// cores.

#ifndef ZSHCI_SRC_DIRECT_H_H_
#define ZSHCI_SRC_DIRECT_H_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "connect_groups.h"
#include "det.h"
#include "hamiltonian.h"
#include "integrals.h"
#include "parallel.h"
#include "scalar.h"
#include "sparse_h.h"
#include "spin_flip.h"

namespace zshci {

template <int W, typename S>
class DirectH : public LinearOperator<S> {
 public:
  // `diagonal` holds <D|H|D> of every determinant on the rank that runs the
  // eigensolver and may be empty elsewhere, where the diagonal elements are
  // then evaluated as needed. Elements with |H(j,i)| <= threshold are left
  // out (never, if threshold < 0). `scratch_bytes` is the memory the
  // per-thread vectors may take together. The arguments must outlive the
  // operator.
  //
  // With `rows` only the elements H(j,i), i < j, of the rows j it marks
  // (one byte per determinant, nonzero: apply) are applied, in both
  // directions, and with with_diagonal = false no diagonal element: the
  // part of a Hamiltonian
  // whose other rows are stored.
  //
  // With `flip` the operator is that over the functions of the layout
  // (spin_flip.h): x and y are over the functions, `diagonal` too, and only
  // the rows of canonical determinants are visited.
  DirectH(Span<Det<W>> dets, const Integrals<S>& ints,
          const internal::GroupStringTables<W>& tables, double threshold,
          const std::vector<double>& diagonal,
          size_t scratch_bytes = kDefaultScratchBytes,
          const std::vector<uint8_t>* rows = nullptr, bool with_diagonal = true,
          const SpinFlipLayout* flip = nullptr)
      : dets_(dets),
        ints_(ints),
        tables_(tables),
        threshold_(threshold),
        diagonal_(diagonal),
        scratch_bytes_(scratch_bytes),
        rows_(rows),
        with_diagonal_(with_diagonal),
        flip_(flip) {}

  static constexpr size_t kDefaultScratchBytes = size_t{1} << 30;

  size_t dim() const override {
    return flip_ != nullptr ? flip_->num_functions() : dets_.size();
  }
  const std::vector<double>& diagonal() const override { return diagonal_; }

  // The part of H x that comes from the rows this rank owns: for each of
  // them the diagonal element and the elements with the determinants before
  // it, used in both directions. The sum over the ranks is H x.
  void Apply(const S* x, S* y) const override {
    if (flip_ != nullptr) return ApplyFlip(x, y);
    const size_t dim = dets_.size();
    const long n = static_cast<long>(dim);
    const bool keep_all = threshold_ < 0.0;
    const bool have_diagonal = diagonal_.size() == dim;
    const size_t num_threads = omp_get_max_threads();
    const bool own_vectors =
        num_threads > 1 && num_threads * dim * sizeof(S) <= scratch_bytes_;
    if (own_vectors && scratch_.size() != num_threads * dim) {
      scratch_ = RawArray<S>(num_threads * dim);
    }
#pragma omp parallel
    {
      // Where this thread adds: its own vector, or y.
      S* const target =
          own_vectors ? scratch_.data() + omp_get_thread_num() * dim : y;
      if (own_vectors) {
        std::fill(target, target + dim, S(0.0));
      } else {
#pragma omp for schedule(static)
        for (long j = 0; j < n; ++j) y[j] = S(0.0);
      }
      if (tables_.has_blocks()) {
        // The diagonal, by rows; the rest, by tasks.
#pragma omp for schedule(static)
        for (long j = 0; j < n; ++j) {
          if (!with_diagonal_ || !tables_.Owns(j)) continue;
          const S term =
              (have_diagonal ? diagonal_[j] : DiagonalEnergy(dets_[j], ints_)) *
              x[j];
          if (own_vectors) {
            target[j] += term;
          } else {
            AtomicAdd(&target[j], term);
          }
        }
        const long num_tasks = static_cast<long>(tables_.num_tasks());
#pragma omp for schedule(dynamic, 16)
        for (long t = 0; t < num_tasks; ++t) {
          if (!tables_.OwnsTask(t)) continue;
          uint64_t tested = 0;
          tables_.ForEachConnectedOfTask(
              t,
              [&](DetIndex j, DetIndex i) {
                if (rows_ != nullptr && !(*rows_)[std::max(j, i)]) return;
                const S value = Hij(dets_[j], dets_[i], ints_);
                if (!keep_all && std::abs(value) <= threshold_) return;
                if (own_vectors) {
                  target[j] += value * x[i];
                  target[i] += Conj(value) * x[j];
                } else {
                  AtomicAdd(&target[j], value * x[i]);
                  AtomicAdd(&target[i], Conj(value) * x[j]);
                }
              },
              &tested);
        }
      }
#pragma omp for schedule(dynamic, 64)
      for (long j = 0; j < n; ++j) {
        if (tables_.has_blocks() || !tables_.Owns(j)) continue;
        if (rows_ != nullptr && !(*rows_)[j]) continue;
        const S xj = x[j];
        S sum = !with_diagonal_
                    ? S(0.0)
                    : (have_diagonal ? diagonal_[j]
                                     : DiagonalEnergy(dets_[j], ints_)) *
                          xj;
        uint64_t tested = 0;
        tables_.ForEachConnected(
            j,
            [&](DetIndex i) {
              const S value = Hij(dets_[j], dets_[i], ints_);
              if (!keep_all && std::abs(value) <= threshold_) return;
              sum += value * x[i];
              if (own_vectors) {
                target[i] += Conj(value) * xj;
              } else {
                AtomicAdd(&target[i], Conj(value) * xj);
              }
            },
            &tested);
        if (own_vectors) {
          target[j] += sum;
        } else {
          AtomicAdd(&target[j], sum);
        }
      }
      if (own_vectors) {
#pragma omp for schedule(static)
        for (long j = 0; j < n; ++j) {
          S total(0.0);
          for (size_t t = 0; t < num_threads; ++t) {
            total += scratch_[t * dim + j];
          }
          y[j] = total;
        }
      }
    }
  }

 private:
  // Apply over the functions of flip_: the pairs (j, i) with j > i and j
  // canonical, folded as in spin_flip.h.
  void ApplyFlip(const S* x, S* y) const {
    const SpinFlipLayout& flip = *flip_;
    const size_t dim = flip.num_functions();
    const long m = static_cast<long>(dim);
    const long n = static_cast<long>(dets_.size());
    const bool keep_all = threshold_ < 0.0;
    const bool have_diagonal = diagonal_.size() == dim;
    const size_t num_threads = omp_get_max_threads();
    const bool own_vectors =
        num_threads > 1 && num_threads * dim * sizeof(S) <= scratch_bytes_;
    if (own_vectors && scratch_.size() != num_threads * dim) {
      scratch_ = RawArray<S>(num_threads * dim);
    }
    auto add = [&](S* target, size_t k, S term) {
      if (own_vectors) {
        target[k] += term;
      } else {
        AtomicAdd(&target[k], term);
      }
    };
    // The element of canonical row hi and column lo < hi, both ways.
    auto pair = [&](S* target, size_t hi, size_t lo) {
      const S value = Hij(dets_[hi], dets_[lo], ints_);
      if (!keep_all && std::abs(value) <= threshold_) return;
      const S folded = value * (flip.weight(lo) / flip.weight(hi));
      const size_t J = flip.rep(hi);
      const size_t I = flip.rep(lo);
      add(target, J, folded * x[I]);
      add(target, I, Conj(folded) * x[J]);
    };
#pragma omp parallel
    {
      S* const target =
          own_vectors ? scratch_.data() + omp_get_thread_num() * dim : y;
      if (own_vectors) {
        std::fill(target, target + dim, S(0.0));
      } else {
#pragma omp for schedule(static)
        for (long J = 0; J < m; ++J) y[J] = S(0.0);
      }
      if (with_diagonal_) {
#pragma omp for schedule(static)
        for (long J = 0; J < m; ++J) {
          if (!tables_.Owns(flip.canonical(J))) continue;
          const double d =
              have_diagonal ? diagonal_[J] : flip.Diagonal(dets_, J, ints_);
          add(target, J, d * x[J]);
        }
      }
      if (tables_.has_blocks()) {
        const long num_tasks = static_cast<long>(tables_.num_tasks());
#pragma omp for schedule(dynamic, 16)
        for (long t = 0; t < num_tasks; ++t) {
          if (!tables_.OwnsTask(t)) continue;
          uint64_t tested = 0;
          tables_.ForEachConnectedOfTask(
              t,
              [&](DetIndex j, DetIndex i) {
                const size_t hi = std::max(j, i);
                const size_t lo = std::min(j, i);
                if (!flip.is_canonical(hi)) return;
                if (rows_ != nullptr && !(*rows_)[hi]) return;
                pair(target, hi, lo);
              },
              &tested);
        }
      } else {
#pragma omp for schedule(dynamic, 64)
        for (long j = 0; j < n; ++j) {
          if (!tables_.Owns(j) || !flip.is_canonical(j)) continue;
          if (rows_ != nullptr && !(*rows_)[j]) continue;
          uint64_t tested = 0;
          tables_.ForEachConnected(
              j, [&](DetIndex i) { pair(target, j, i); }, &tested);
        }
      }
      if (own_vectors) {
#pragma omp for schedule(static)
        for (long J = 0; J < m; ++J) {
          S total(0.0);
          for (size_t t = 0; t < num_threads; ++t) {
            total += scratch_[t * dim + J];
          }
          y[J] = total;
        }
      }
    }
  }

  Span<Det<W>> dets_;
  const Integrals<S>& ints_;
  const internal::GroupStringTables<W>& tables_;
  double threshold_;
  const std::vector<double>& diagonal_;
  size_t scratch_bytes_;
  const std::vector<uint8_t>* rows_;
  bool with_diagonal_;
  const SpinFlipLayout* flip_;
  // One vector per thread, when they fit the budget.
  mutable RawArray<S> scratch_;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_DIRECT_H_H_
