// The Hamiltonian applied without being stored.
//
// DirectH works out the connections of every determinant from the group
// string tables (connect_groups.h) each time it is applied, evaluating the
// elements as it goes. Each product costs about as much as building the
// stored matrix twice (both triangles are visited, so that every thread only
// writes the elements of y it owns), but the memory is that of the tables:
// the way to go when the matrix does not fit.

#ifndef ZSHCI_SRC_DIRECT_H_H_
#define ZSHCI_SRC_DIRECT_H_H_

#include <cmath>
#include <vector>

#include "connect_groups.h"
#include "det.h"
#include "hamiltonian.h"
#include "integrals.h"
#include "parallel.h"
#include "sparse_h.h"

namespace zshci {

template <int W, typename S>
class DirectH : public LinearOperator<S> {
 public:
  // `diagonal` holds <D|H|D> of every determinant on the rank that runs the
  // eigensolver and may be empty elsewhere, where the diagonal elements are
  // then evaluated as needed. Elements with |H(j,i)| <= threshold are left
  // out (never, if threshold < 0). The arguments must outlive the operator.
  DirectH(Span<Det<W>> dets, const Integrals<S>& ints,
          const internal::GroupStringTables<W>& tables, double threshold,
          const std::vector<double>& diagonal)
      : dets_(dets),
        ints_(ints),
        tables_(tables),
        threshold_(threshold),
        diagonal_(diagonal) {}

  size_t dim() const override { return dets_.size(); }
  const std::vector<double>& diagonal() const override { return diagonal_; }

  // y_j = sum_i H(j,i) x_i for the rows j this rank owns, zero for the
  // others: the sum over the ranks is H x.
  void Apply(const S* x, S* y) const override {
    const long n = static_cast<long>(dets_.size());
    const bool keep_all = threshold_ < 0.0;
    const bool have_diagonal = diagonal_.size() == dets_.size();
#pragma omp parallel for schedule(dynamic, 64)
    for (long j = 0; j < n; ++j) {
      if (!tables_.Owns(j)) {
        y[j] = S(0.0);
        continue;
      }
      S sum = (have_diagonal ? diagonal_[j] : DiagonalEnergy(dets_[j], ints_)) *
              x[j];
      uint64_t tested = 0;
      tables_.ForEachConnected(
          j,
          [&](DetIndex i) {
            const S value = Hij(dets_[j], dets_[i], ints_);
            if (keep_all || std::abs(value) > threshold_) sum += value * x[i];
          },
          &tested, /*both_sides=*/true);
      y[j] = sum;
    }
  }

 private:
  Span<Det<W>> dets_;
  const Integrals<S>& ints_;
  const internal::GroupStringTables<W>& tables_;
  double threshold_;
  const std::vector<double>& diagonal_;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_DIRECT_H_H_
