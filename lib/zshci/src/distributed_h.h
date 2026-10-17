// The Hamiltonian spread over the ranks of a communicator, and the
// eigensolver on top of it.
//
// Every rank holds an operator for its part of the matrix - a SparseH with
// the off-diagonal elements it found, or a DirectH for the rows it owns; the
// product H x is the sum of the local products. The eigensolver runs on the
// root only: the other ranks wait in Serve() and take part in each product.
// They never hold the Davidson basis.

#ifndef ZSHCI_SRC_DISTRIBUTED_H_H_
#define ZSHCI_SRC_DISTRIBUTED_H_H_

#include <vector>

#include "comm.h"
#include "davidson.h"
#include "scalar.h"
#include "sparse_h.h"

namespace zshci {

template <typename S>
class DistributedH : public LinearOperator<S> {
 public:
  // Collective.
  DistributedH(const LinearOperator<S>& local, const Comm& comm)
      : local_(local),
        comm_(comm),
        x_(comm.size() > 1 ? local.dim() : 0, comm),
        y_(comm.is_root() ? 0 : local.dim()) {}

  size_t dim() const override { return local_.dim(); }
  const std::vector<double>& diagonal() const override {
    return local_.diagonal();
  }

  // Root only: y = H x, with the help of the ranks in Serve().
  void Apply(const S* x, S* y) const override;

  // Other ranks: take part in the root's products until it calls Stop().
  void Serve() const;

  // Root only: releases the ranks in Serve().
  void Stop() const;

  // Seconds this rank spent in its local products, and in communication.
  double seconds_local() const { return seconds_local_; }
  double seconds_comm() const { return seconds_comm_; }

 private:
  enum Command : int { kStop = 0, kApply = 1 };

  const LinearOperator<S>& local_;
  const Comm& comm_;
  // The vector being multiplied, one copy per node, and the local product of
  // a serving rank.
  mutable SharedArray<S> x_;
  mutable std::vector<S> y_;
  mutable double seconds_local_ = 0.0;
  mutable double seconds_comm_ = 0.0;
};

// Solves for the lowest eigenpairs with all ranks. On entry the root's
// `vectors` holds the starting guesses and on exit the eigenvectors; the
// other ranks' `vectors` is not touched. Every rank gets the result.
// With `active` (read on the root only, one entry per basis vector), the
// problem is solved in the marked subspace; see MaskedOperator.
// Instantiated for double and Complex.
template <typename S>
DavidsonResult SolveDistributed(const LinearOperator<S>& local,
                                const DavidsonOptions& options,
                                const Comm& comm,
                                std::vector<std::vector<S>>* vectors,
                                const std::vector<uint8_t>* active = nullptr);

}  // namespace zshci

#endif  // ZSHCI_SRC_DISTRIBUTED_H_H_
