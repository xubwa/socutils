// The Hamiltonian spread over the ranks of a communicator, and the
// eigensolver on top of it.
//
// Every rank holds a SparseH with the off-diagonal elements it found; the
// product H x is the sum of the local products. The eigensolver runs on the
// root only: the other ranks wait in Serve() and take part in each product.
// They never hold the Davidson basis.

#ifndef ZSHCI_SRC_DISTRIBUTED_H_H_
#define ZSHCI_SRC_DISTRIBUTED_H_H_

#include <vector>

#include "comm.h"
#include "davidson.h"
#include "integrals.h"
#include "sparse_h.h"

namespace zshci {

class DistributedH : public LinearOperator {
 public:
  // Collective.
  DistributedH(const SparseH& local, const Comm& comm)
      : local_(local), comm_(comm), x_(local.dim(), comm), y_(local.dim()) {}

  size_t dim() const override { return local_.dim(); }
  const std::vector<double>& diagonal() const override {
    return local_.diagonal();
  }

  // Root only: y = H x, with the help of the ranks in Serve().
  void Apply(const Complex* x, Complex* y) const override;

  // Other ranks: take part in the root's products until it calls Stop().
  void Serve() const;

  // Root only: releases the ranks in Serve().
  void Stop() const;

 private:
  enum Command : int { kStop = 0, kApply = 1 };

  const SparseH& local_;
  const Comm& comm_;
  // The vector being multiplied, one copy per node, and the local product of
  // a serving rank.
  mutable SharedArray<Complex> x_;
  mutable std::vector<Complex> y_;
};

// Solves for the lowest eigenpairs with all ranks. On entry the root's
// `vectors` holds the starting guesses and on exit the eigenvectors; the
// other ranks' `vectors` is not touched. Every rank gets the result.
DavidsonResult SolveDistributed(const SparseH& local,
                                const DavidsonOptions& options,
                                const Comm& comm,
                                std::vector<std::vector<Complex>>* vectors);

}  // namespace zshci

#endif  // ZSHCI_SRC_DISTRIBUTED_H_H_
