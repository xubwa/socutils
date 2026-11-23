// Eigensolvers of the select and diag modes.
//
// Two solvers stand behind SolveDistributed (distributed_h.h):
//
// - "few" (default): the Davidson of the fci mode (fci_davidson.h), which
//   keeps max_space basis vectors per root and their products with H and
//   no other vector of the full length, forms combinations in place and on
//   collapse keeps the Ritz vectors with those of the previous iteration
//   (LOBPCG-like). With 3 vectors per root it needs about as many products
//   as the classic solver with 20 and a seventh of its memory: at 2e8
//   complex determinants 6 vectors of 3.2 GB instead of 40.
// - "classic": the block Davidson below, max_space_per_root vectors.
//
// In a selection the eigenvectors of one iteration, padded with zeros for
// the determinants added since, start the solver of the next; the subspace
// is not carried over (its products would need the new rows anyway). The
// solver runs on the root, where all vectors are held; the other ranks
// take part in each product only (DistributedH::Serve). With a stored, a
// direct or an adapted Hamiltonian alike the solver sees a LinearOperator.
// Vectors stay on the root up to some 1e9 determinants (6 vectors of 16 GB);
// the few-vector solver can hold them as segments on the ranks
// (VectorLayout) when that is needed.
//
// Checkpoints (few only): every checkpoint_every iterations the Ritz
// vectors go to `checkpoint` with a fingerprint of the determinant list;
// with `restart` a matching file is read back as the start, a file of
// another list is ignored. The select mode writes the list itself with the
// vectors to select.h5 (select.h).

#ifndef ZSHCI_SRC_DAVIDSON_H_
#define ZSHCI_SRC_DAVIDSON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "scalar.h"
#include "sparse_h.h"

namespace zshci {

struct DavidsonOptions {
  // Convergence threshold on the norm of every residual H x - E x.
  double tol = 5e-5;
  // The subspace is collapsed to the current Ritz vectors when it would grow
  // beyond max_space_per_root * num_roots vectors. Small values cost many
  // more iterations: 8 instead of 20 quadrupled them on a 700k-determinant
  // test, while 40 brought nothing further. Each vector and its product
  // with H take two numbers per determinant.
  int max_space_per_root = 20;
  // The few-vector solver (default) instead of the classic one, its basis
  // vectors per root, and the shift of its preconditioner
  // 1 / (H_ii - E + level_shift).
  bool few_vectors = true;
  int max_space = 3;
  double level_shift = 1e-3;
  // Few-vector solver: Ritz vectors written to `checkpoint` every
  // checkpoint_every iterations (0: never), read back with `restart` if
  // written for the same fingerprint.
  std::string checkpoint;
  int checkpoint_every = 0;
  bool restart = false;
  uint64_t fingerprint = 0;
  int max_iterations = 200;
  // Matrices up to this dimension are diagonalized densely instead.
  int dense_dim = 400;
  bool print = false;
};

struct DavidsonResult {
  std::vector<double> energies;  // without the core energy
  std::vector<double> residual_norms;
  int num_iterations = 0;
  // Products H x, and the seconds spent in them and in the whole solve.
  int num_products = 0;
  double seconds_products = 0.0;
  double seconds_total = 0.0;
  bool converged = false;
  // The few-vector solver started from a checkpoint.
  bool restarted = false;
};

// On entry `vectors` holds one starting guess per root; on exit it holds the
// eigenvectors. Guesses need not be normalized or orthogonal. Instantiated
// for double and Complex.
template <typename S>
DavidsonResult Davidson(const LinearOperator<S>& h,
                        const DavidsonOptions& options,
                        std::vector<std::vector<S>>* vectors);

}  // namespace zshci

#endif  // ZSHCI_SRC_DAVIDSON_H_
