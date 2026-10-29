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
#include "spin_flip.h"

namespace zshci {

template <int W, class Ints, typename S = typename Ints::Scalar>
void RunDiag(const Input& input, const Ints& ints, const Comm& comm) {
  const int num_orbs = ints.num_orbs();
  std::vector<Det<W>> root_dets;
  // With spin_flip the list is completed by the flips of its determinants
  // and put in the layout (spin_flip.h); loaded_source[j] is the position in
  // the loaded list of determinant j (-1: added).
  size_t num_loaded = 0;
  std::vector<long> loaded_source;
  if (comm.is_root()) {
    root_dets = LoadDeterminants<W>(input, num_orbs);
    num_loaded = root_dets.size();
    if (input.spin_flip != 0) {
      std::vector<Det<W>> paired;
      AppendPairs(root_dets,
                  SpinFlipLambda(input.spin_flip, root_dets[0].Count()),
                  &paired, &loaded_source);
      if (paired.empty()) {
        throw std::runtime_error(
            "spin_flip: no determinant of the list has the asked symmetry");
      }
      std::printf(
          "spin_flip %+d: %zu determinants with their flips make "
          "%zu\n",
          input.spin_flip, root_dets.size(), paired.size());
      root_dets = std::move(paired);
    }
  }
  const SharedArray<Det<W>> shared_dets =
      SharedArray<Det<W>>::FromRoot(root_dets, comm);
  root_dets = std::vector<Det<W>>();
  const Span<Det<W>> dets = shared_dets.span();
  const size_t n = dets.size();
  std::printf("determinants: %zu, electrons: %d, spinors: %d, words: %d\n", n,
              dets[0].Count(), num_orbs, W);

  // In diag mode the labels are only reported: the space is whatever the
  // list contains.
  const std::vector<LabelSet>& labels = ints.symmetry();
  if (comm.is_root()) {
    const std::vector<int> first_label = DeterminantLabels(dets[0], labels);
    size_t num_other = 0;
    for (const Det<W>& det : dets) {
      if (DeterminantLabels(det, labels) != first_label) ++num_other;
    }
    if (num_other == 0) {
      std::printf("symmetry label of all determinants: %s\n",
                  FormatLabels(first_label).c_str());
    } else {
      std::printf(
          "note: %zu determinants carry a symmetry label different "
          "from that of the first one, %s\n",
          num_other, FormatLabels(first_label).c_str());
    }
  }

  double time = omp_get_wtime();
  VariationalH<W, Ints> h(input, ints, comm);
  h.Extend(dets, 0);
  if (static_cast<size_t>(input.nroots) > h.dim()) {
    throw std::runtime_error("nroots exceeds the dimension of the space");
  }
  if (h.flip() != nullptr) {
    std::printf("spin_flip %+d: %zu functions\n", input.spin_flip, h.dim());
  }
  h.PrintStats();
  if (!h.direct()) {
    std::printf(
        "Hamiltonian: %zu stored elements (one triangle) on the "
        "root, %.2f s\n",
        h.stored().num_elements(), omp_get_wtime() - time);
  }
  if (input.check) {
    if (comm.size() == 1 && !h.direct() && h.flip() == nullptr) {
      CheckHamiltonian(dets, h.stored(), ints, h.connect());
    } else {
      std::printf(
          "note: \"check\" needs a single rank and a stored "
          "Hamiltonian and is skipped\n");
    }
  }

  time = omp_get_wtime();
  std::vector<std::vector<S>> root_vectors;
  if (comm.is_root()) {
    // The vectors of the file the determinants came from, if it has them.
    if (!input.detfile.empty() &&
        ReadCoefficients(input.detfile, h.flip() != nullptr ? num_loaded : n,
                         input.nroots, &root_vectors)) {
      if (h.flip() != nullptr) {
        for (std::vector<S>& v : root_vectors) {
          std::vector<S> full(n, S(0.0));
          for (size_t j = 0; j < n; ++j) {
            if (loaded_source[j] >= 0) full[j] = v[loaded_source[j]];
          }
          v = std::move(full);
        }
        root_vectors = h.Project(root_vectors);
      }
      std::printf("starting vectors: read from %s\n", input.detfile.c_str());
    } else {
      root_vectors = h.Guess(input.nroots);
    }
  }
  const DavidsonResult result = h.Solve(input.davidson, &root_vectors);
  if (!result.converged) {
    throw std::runtime_error("Davidson did not converge in " +
                             std::to_string(result.num_iterations) +
                             " iterations");
  }
  std::printf("Davidson: %d iterations, %.2f s\n", result.num_iterations,
              omp_get_wtime() - time);
  const SharedVectors<S> vectors =
      ShareVectors(h.Expand(root_vectors), n, comm);
  root_vectors = std::vector<std::vector<S>>();

  // The list is taken as given, but a perturbative correction on top of it
  // respects the occupation restrictions.
  const RestrictedFilter filter =
      MakeFilter(input, labels, num_orbs, /*with_restrictions=*/true);
  // epsilon1 of the selection that made the list, if its file says.
  double start_threshold = 0.0;
  if (comm.is_root() && !input.detfile.empty()) {
    start_threshold = ReadSelectionThreshold(input.detfile);
  }
  comm.BcastValue(&start_threshold);
  const std::vector<PtResult> pt = ComputePerturbation(
      input, ints, dets, result.energies, vectors,
      /*heat_bath=*/nullptr, filter, comm, h.flip(), start_threshold);
  WriteResults(input, ints, dets, h, labels, result.energies, vectors, pt,
               comm);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_DIAG_H_
