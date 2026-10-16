// Davidson eigensolver for CI vectors too long to keep many of: the "fci"
// mode, up to some 1e10 determinants.
//
// It keeps at most max_space basis vectors and as many products with H, and
// no other vector of the full length. Linear combinations of the basis are
// formed in place, element by element (the new vectors at an index depend
// only on the old ones at that index), so collapsing the subspace costs no
// memory. On collapse it keeps the Ritz vectors and, where max_space allows
// it, the Ritz vectors of the previous iteration (orthogonalized): with
// max_space = 3 per root this is the locally optimal (LOBPCG-like) scheme,
// which converges much like an unrestricted Davidson.
//
// Vectors are held as the elements of some segments of the index range
// (VectorLayout); one process now holds them all. Every operation goes over
// the local elements and sums over them, so that a distribution over
// processes needs only the sums to be combined.
//
// Checkpoints: every checkpoint_every iterations the current Ritz vectors go
// to the file `checkpoint` (written beside it and renamed, so a job killed
// while writing leaves the previous one), with the energies and a
// fingerprint of the space; with `restart` they are read back as the start.

#ifndef ZSHCI_SRC_FCI_DAVIDSON_H_
#define ZSHCI_SRC_FCI_DAVIDSON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "big_array.h"
#include "scalar.h"

namespace zshci {

struct VectorLayout {
  struct Segment {
    int64_t begin = 0;
    int64_t end = 0;
  };
  // Held here, stored one after another.
  std::vector<Segment> segments;
  int64_t global_dim = 0;

  static VectorLayout Whole(int64_t dim) { return {{{0, dim}}, dim}; }
  int64_t local_size() const {
    int64_t n = 0;
    for (const Segment& s : segments) n += s.end - s.begin;
    return n;
  }
};

// y = H x over the local elements.
template <typename S>
class CiOperator {
 public:
  virtual ~CiOperator() = default;
  virtual void Apply(const S* x, S* y) const = 0;
};

struct FciDavidsonOptions {
  int num_roots = 1;
  // Converged when the norm of every residual H x - E x is below this.
  double tol = 1e-6;
  int max_iterations = 200;
  // Basis vectors (each with its product with H); at least 2 * num_roots.
  int max_space = 3;
  // Preconditioner 1 / (H_ii - E + level_shift).
  double level_shift = 1e-3;
  // Spaces up to this dimension are diagonalized as a full matrix.
  int dense_dim = 400;
  std::string checkpoint;
  int checkpoint_every = 1;
  bool restart = false;
  bool huge_pages = true;
  // Progress lines on stdout.
  bool print = true;
  // Written into the checkpoint and checked on restart.
  uint64_t fingerprint = 0;
};

template <typename S>
struct FciDavidsonResult {
  std::vector<double> energies;  // without the core energy
  std::vector<double> residual_norms;
  int num_iterations = 0;
  int num_products = 0;
  double seconds_products = 0.0;
  double seconds_total = 0.0;
  bool converged = false;
  bool restarted = false;
  // The eigenvectors (Ritz vectors when not converged), local elements.
  std::vector<BigArray<S>> vectors;
};

// `diagonal`: H_ii of the local elements. Instantiated for double and
// Complex.
template <typename S>
FciDavidsonResult<S> FciDavidson(const CiOperator<S>& h, const double* diagonal,
                                 const VectorLayout& layout,
                                 const FciDavidsonOptions& options);

// Reads up to num_roots vectors of a checkpoint file into `vectors` (local
// elements of `layout`, allocated here) and returns their number: 0 if the
// file is missing. Throws if it was written for another space (dimension,
// number type, fingerprint). Fills `energies` if given.
template <typename S>
int ReadCiCheckpoint(const std::string& path, const VectorLayout& layout,
                     uint64_t fingerprint, int num_roots, bool huge_pages,
                     std::vector<BigArray<S>>* vectors,
                     std::vector<double>* energies = nullptr);

}  // namespace zshci

#endif  // ZSHCI_SRC_FCI_DAVIDSON_H_
