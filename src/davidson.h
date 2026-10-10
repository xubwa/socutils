// Block Davidson solver for the lowest eigenpairs of a Hermitian operator.

#ifndef ZSHCI_SRC_DAVIDSON_H_
#define ZSHCI_SRC_DAVIDSON_H_

#include <vector>

#include "integrals.h"
#include "sparse_h.h"

namespace zshci {

struct DavidsonOptions {
  // Convergence threshold on the norm of every residual H x - E x.
  double tol = 5e-5;
  // The subspace is collapsed to the current Ritz vectors when it would grow
  // beyond max_space_per_root * num_roots vectors. Small values cost many
  // more iterations: 8 instead of 20 quadrupled them on a 700k-determinant
  // test, while 40 brought nothing further. Each vector takes 32 bytes per
  // determinant.
  int max_space_per_root = 20;
  int max_iterations = 200;
  // Matrices up to this dimension are diagonalized densely instead.
  int dense_dim = 400;
  bool print = false;
};

struct DavidsonResult {
  std::vector<double> energies;  // without the core energy
  std::vector<double> residual_norms;
  int num_iterations = 0;
  bool converged = false;
};

// On entry `vectors` holds one starting guess per root; on exit it holds the
// eigenvectors. Guesses need not be normalized or orthogonal.
DavidsonResult Davidson(const LinearOperator& h, const DavidsonOptions& options,
                        std::vector<std::vector<Complex>>* vectors);

}  // namespace zshci

#endif  // ZSHCI_SRC_DAVIDSON_H_
