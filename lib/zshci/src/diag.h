// "diag" mode: diagonalize the Hamiltonian in a given list of determinants.

#ifndef ZSHCI_SRC_DIAG_H_
#define ZSHCI_SRC_DIAG_H_

#include <omp.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm.h"
#include "connect.h"
#include "davidson.h"
#include "det.h"
#include "distributed_h.h"
#include "driver_common.h"
#include "input.h"
#include "integrals.h"
#include "sparse_h.h"

namespace zshci {

template <int W, typename S>
void RunDiag(const Input& input, const Integrals<S>& ints, const Comm& comm) {
  const int num_orbs = ints.num_orbs();
  std::vector<Det<W>> root_dets;
  if (comm.is_root()) root_dets = LoadDeterminants<W>(input, num_orbs);
  const SharedArray<Det<W>> shared_dets =
      SharedArray<Det<W>>::FromRoot(root_dets, comm);
  root_dets = std::vector<Det<W>>();
  const Span<Det<W>> dets = shared_dets.span();
  const size_t n = dets.size();
  if (static_cast<size_t>(input.nroots) > n) {
    throw std::runtime_error("nroots exceeds the number of determinants");
  }
  std::printf("determinants: %zu, electrons: %d, spinors: %d, words: %d\n", n,
              dets[0].Count(), num_orbs, W);

  // In diag mode the labels are only reported: the space is whatever the
  // list contains.
  const std::vector<int> labels = ResolveLabels(input, ints);
  if (comm.is_root()) {
    const int first_label =
        DeterminantLabel(dets[0], labels, input.symmetry.mod);
    size_t num_other = 0;
    for (const Det<W>& det : dets) {
      if (DeterminantLabel(det, labels, input.symmetry.mod) != first_label) {
        ++num_other;
      }
    }
    if (num_other == 0) {
      std::printf("symmetry label of all determinants: %d\n", first_label);
    } else {
      std::printf("note: %zu determinants carry a symmetry label different "
                  "from that of the first one (%d)\n",
                  num_other, first_label);
    }
  }

  double time = omp_get_wtime();
  SparseH<S> h;
  h.set_apply_diagonal(comm.is_root());
  AppendDiagonal(dets, 0, ints, comm, &h);
  const ConnectOptions connect = MakeConnectOptions(input);
  ConnectStats stats;
  h.AppendBlock(BuildBlock(dets, 0, ints, connect, comm, &stats));
  PrintConnectStats(stats);
  std::printf("Hamiltonian: %zu stored elements (upper triangle) on the "
              "root, %.2f s\n",
              h.num_elements(), omp_get_wtime() - time);
  if (input.check) {
    if (comm.size() == 1) {
      CheckHamiltonian(dets, h, ints, connect);
    } else {
      std::printf("note: \"check\" needs a single rank and is skipped\n");
    }
  }

  time = omp_get_wtime();
  std::vector<std::vector<S>> root_vectors;
  if (comm.is_root()) root_vectors = GuessVectors(dets, h, ints, input.nroots);
  const DavidsonResult result =
      SolveDistributed(h, input.davidson, comm, &root_vectors);
  if (!result.converged) {
    throw std::runtime_error("Davidson did not converge in " +
                             std::to_string(result.num_iterations) +
                             " iterations");
  }
  std::printf("Davidson: %d iterations, %.2f s\n", result.num_iterations,
              omp_get_wtime() - time);
  const SharedVectors<S> vectors = ShareVectors(root_vectors, n, comm);
  root_vectors = std::vector<std::vector<S>>();

  // The list is taken as given, but a perturbative correction on top of it
  // respects the occupation restrictions.
  const RestrictedFilter filter =
      MakeFilter(input, labels, num_orbs, /*with_restrictions=*/true);
  const std::vector<PtResult> pt =
      ComputePerturbation(input, ints, dets, result.energies, vectors,
                          /*heat_bath=*/nullptr, filter, comm);
  WriteResults(input, ints, dets, h, labels, result.energies, vectors, pt,
               comm);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_DIAG_H_
