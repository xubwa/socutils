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
  DirectH(Span<Det<W>> dets, const Integrals<S>& ints,
          const internal::GroupStringTables<W>& tables, double threshold,
          const std::vector<double>& diagonal,
          size_t scratch_bytes = kDefaultScratchBytes)
      : dets_(dets),
        ints_(ints),
        tables_(tables),
        threshold_(threshold),
        diagonal_(diagonal),
        scratch_bytes_(scratch_bytes) {}

  static constexpr size_t kDefaultScratchBytes = size_t{1} << 30;

  size_t dim() const override { return dets_.size(); }
  const std::vector<double>& diagonal() const override { return diagonal_; }

  // The part of H x that comes from the rows this rank owns: for each of
  // them the diagonal element and the elements with the determinants before
  // it, used in both directions. The sum over the ranks is H x.
  void Apply(const S* x, S* y) const override {
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
          if (!tables_.Owns(j)) continue;
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
        const S xj = x[j];
        S sum =
            (have_diagonal ? diagonal_[j] : DiagonalEnergy(dets_[j], ints_)) *
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
  Span<Det<W>> dets_;
  const Integrals<S>& ints_;
  const internal::GroupStringTables<W>& tables_;
  double threshold_;
  const std::vector<double>& diagonal_;
  size_t scratch_bytes_;
  // One vector per thread, when they fit the budget.
  mutable RawArray<S> scratch_;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_DIRECT_H_H_
