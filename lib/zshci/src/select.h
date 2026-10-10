// "select" mode: heat-bath selected configuration interaction.
//
// Starting from the determinants of the input, each iteration adds every
// determinant D_a with |H(a,i) c_i| > epsilon1 for some D_i already in the
// space, where c_i is the coefficient of D_i (the root-mean-square over the
// roots times sqrt(nroots)), and diagonalizes the Hamiltonian in the enlarged
// space.

#ifndef ZSHCI_SRC_SELECT_H_
#define ZSHCI_SRC_SELECT_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm.h"
#include "connect.h"
#include "davidson.h"
#include "det.h"
#include "distributed_h.h"
#include "driver_common.h"
#include "excite.h"
#include "heat_bath.h"
#include "input.h"
#include "integrals.h"
#include "sparse_h.h"

namespace zshci {

namespace internal {

// Sink of ForEachExcitation that collects the excited determinants.
template <int W>
class CandidateSink {
 public:
  CandidateSink(const Det<W>& parent, std::vector<Det<W>>* out)
      : parent_(parent), out_(out) {}

  void Single(int i, int a, Complex /*element*/) {
    Det<W> det = parent_;
    det.Clear(i);
    det.Set(a);
    out_->push_back(det);
  }
  void Double(int i, int j, int a, int b, double /*magnitude*/) {
    Det<W> det = parent_;
    det.Clear(i);
    det.Clear(j);
    det.Set(a);
    det.Set(b);
    out_->push_back(det);
  }

 private:
  const Det<W>& parent_;
  std::vector<Det<W>>* out_;
};

}  // namespace internal

// Returns, sorted, the determinants reached from `dets` with weights
// `weights` at the cutoff `epsilon` that are not in `sorted_dets` (the same
// determinants as `dets`, sorted).
//
// With several ranks, each rank excites its share of the determinants, the
// candidates are sent to the rank that owns their hash, which removes
// duplicates and known determinants, and the survivors are gathered so that
// every rank returns the same list.
template <int W, class Filter>
std::vector<Det<W>> SelectNewDeterminants(
    Span<Det<W>> dets, Span<Det<W>> sorted_dets, Span<double> weights, double epsilon, const Integrals& ints,
    const HeatBath& heat_bath, const Filter& filter,
    const Comm& comm = Comm()) {
  std::vector<Det<W>> candidates;
  const long num_dets = static_cast<long>(dets.size());
  const long num_ranks = comm.size();
  const long rank = comm.rank();
#pragma omp parallel
  {
    std::vector<Det<W>> local;
    std::vector<int> occ;
#pragma omp for schedule(dynamic, 16) nowait
    for (long i = 0; i < num_dets; ++i) {
      if (i % num_ranks != rank || weights[i] <= 0.0) continue;
      occ.clear();
      dets[i].AppendOccupied(&occ);
      internal::CandidateSink<W> sink(dets[i], &local);
      ForEachExcitation(dets[i], occ, epsilon / weights[i], ints, heat_bath,
                        filter, &sink);
    }
    std::sort(local.begin(), local.end());
    local.erase(std::unique(local.begin(), local.end()), local.end());
#pragma omp critical
    candidates.insert(candidates.end(), local.begin(), local.end());
  }
  if (num_ranks > 1) {
    std::vector<std::vector<Det<W>>> outgoing(num_ranks);
    for (const Det<W>& det : candidates) {
      outgoing[det.Hash() % num_ranks].push_back(det);
    }
    candidates = comm.Exchange(outgoing);
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());

  std::vector<Det<W>> fresh;
  std::set_difference(candidates.begin(), candidates.end(),
                      sorted_dets.begin(), sorted_dets.end(),
                      std::back_inserter(fresh));
  if (num_ranks > 1) {
    fresh = comm.GatherAll(fresh);
    std::sort(fresh.begin(), fresh.end());
  }
  return fresh;
}

// Collective: the selection weight of every determinant, once per node. It is
// the root-sum-square of its coefficients over the roots or, without
// eigenvectors (fewer starting determinants than roots), the value every
// determinant would have if each root were that determinant.
inline SharedArray<double> ShareWeights(
    const std::vector<std::vector<Complex>>& root_vectors, size_t dim,
    int num_roots, const Comm& comm) {
  SharedArray<double> weights(dim, comm);
  if (comm.is_root()) {
    for (size_t i = 0; i < dim; ++i) {
      double sum = 0.0;
      for (const std::vector<Complex>& v : root_vectors) sum += std::norm(v[i]);
      weights[i] = root_vectors.empty() ? std::sqrt(double(num_roots))
                                        : std::sqrt(sum);
    }
  }
  weights.ShareFromRoot(comm);
  return weights;
}

template <int W>
void RunSelect(const Input& input, const Integrals& ints, const Comm& comm) {
  const int num_orbs = ints.num_orbs();
  const int num_roots = input.nroots;
  const std::vector<int> labels = ResolveLabels(input, ints);
  const RestrictedFilter filter =
      MakeFilter(input, labels, num_orbs, /*with_restrictions=*/true);

  // The root owns the growing list; `shared_dets` is its copy on every node.
  std::vector<Det<W>> root_dets;
  if (comm.is_root()) {
    root_dets = LoadDeterminants<W>(input, num_orbs);
    // The starting determinants fix the symmetry block; mixing blocks is
    // almost certainly an input mistake.
    const int block_label =
        DeterminantLabel(root_dets[0], labels, input.symmetry.mod);
    for (size_t d = 0; d < root_dets.size(); ++d) {
      if (DeterminantLabel(root_dets[d], labels, input.symmetry.mod) !=
          block_label) {
        throw std::runtime_error(
            "starting determinant " + std::to_string(d) +
            " has a symmetry label different from that of the first one");
      }
      std::vector<int> occ;
      root_dets[d].AppendOccupied(&occ);
      if (!filter.Allows(occ)) {
        throw std::runtime_error("starting determinant " + std::to_string(d) +
                                 " violates an occupation restriction");
      }
    }
    std::printf("starting determinants: %zu, electrons: %d, spinors: %d, "
                "words: %d, symmetry label: %d\n",
                root_dets.size(), root_dets[0].Count(), num_orbs, W,
                block_label);
  }
  SharedArray<Det<W>> shared_dets =
      SharedArray<Det<W>>::FromRoot(root_dets, comm);

  double time = omp_get_wtime();
  const HeatBath heat_bath(ints, input.select.integral_threshold, comm);
  std::printf("heat-bath tables: %zu entries, %.2f s\n",
              heat_bath.num_entries(), omp_get_wtime() - time);

  const ConnectOptions connect = MakeConnectOptions(input);
  ConnectStats stats;

  SparseH h;
  h.set_apply_diagonal(comm.is_root());
  AppendDiagonal(shared_dets.span(), 0, ints, comm, &h);
  h.AppendBlock(
      BuildBlock(shared_dets.span(), 0, ints, connect, comm, &stats));

  // The eigenvectors live on the root; the other ranks see only what is
  // shared from them (the selection weights, and at the end the vectors).
  // If the starting space can hold all roots, start from its eigenvectors.
  std::vector<std::vector<Complex>> root_vectors;
  std::vector<double> energies;
  DavidsonOptions loose = input.davidson;
  loose.tol = std::max(input.davidson.tol, input.select.davidson_tol_loose);
  bool have_vectors = shared_dets.size() >= static_cast<size_t>(num_roots);
  if (have_vectors) {
    if (comm.is_root()) {
      root_vectors = GuessVectors(shared_dets.span(), h, ints, num_roots);
    }
    energies = SolveDistributed(h, loose, comm, &root_vectors).energies;
  }

  // t_sel, t_ham and t_dav are the seconds spent in this iteration on the
  // selection, the Hamiltonian and the eigensolver.
  std::printf("%5s %10s %12s %5s %20s %9s %8s %8s %8s %9s\n", "iter",
              "epsilon1", "dets", "root", "energy", "davidson", "t_sel",
              "t_ham", "t_dav", "time");
  const double start_time = omp_get_wtime();
  const int last_schedule_start = input.select.schedule.back().first;
  bool finished = false;
  for (int iter = 0; iter < input.select.max_iter && !finished; ++iter) {
    const double epsilon = input.select.Epsilon(iter);
    const size_t old_size = shared_dets.size();

    const double select_start = omp_get_wtime();
    std::vector<Det<W>> fresh;
    {
      const SharedArray<double> weights =
          ShareWeights(root_vectors, old_size, num_roots, comm);
      const SharedArray<Det<W>> sorted_dets =
          ShareSorted(shared_dets.span(), comm);
      fresh = SelectNewDeterminants(shared_dets.span(), sorted_dets.span(),
                                    weights.span(), epsilon, ints, heat_bath,
                                    filter, comm);
    }
    const double build_start = omp_get_wtime();

    const size_t new_size = old_size + fresh.size();
    if (new_size < static_cast<size_t>(num_roots)) {
      throw std::runtime_error(
          "the space holds fewer determinants than roots; give more "
          "starting determinants or a smaller epsilon1");
    }
    if (!fresh.empty()) {
      if (comm.is_root()) {
        root_dets.insert(root_dets.end(), fresh.begin(), fresh.end());
      }
      shared_dets = SharedArray<Det<W>>::FromRoot(root_dets, comm);
      AppendDiagonal(shared_dets.span(), old_size, ints, comm, &h);
      h.AppendBlock(BuildBlock(shared_dets.span(), old_size, ints, connect,
                               comm, &stats));
    }
    fresh = std::vector<Det<W>>();

    if (comm.is_root()) {
      if (!have_vectors) {
        root_vectors = GuessVectors(shared_dets.span(), h, ints, num_roots);
      } else {
        for (std::vector<Complex>& v : root_vectors) v.resize(new_size, 0.0);
      }
    }
    have_vectors = true;

    // The last epsilon1 has been reached and the space or the energies have
    // stopped changing: solve tightly and stop.
    const double solve_start = omp_get_wtime();
    const std::vector<double> previous = energies;
    DavidsonResult result = SolveDistributed(h, loose, comm, &root_vectors);
    bool converged_space = false;
    if (iter >= last_schedule_start && !previous.empty()) {
      double change = 0.0;
      for (int k = 0; k < num_roots; ++k) {
        change += std::abs(result.energies[k] - previous[k]) / num_roots;
      }
      converged_space =
          new_size == old_size || change < input.select.energy_tol;
    }
    finished = converged_space || iter == input.select.max_iter - 1;
    if (finished) {
      result = SolveDistributed(h, input.davidson, comm, &root_vectors);
    }
    if (!result.converged) {
      throw std::runtime_error("Davidson did not converge in iteration " +
                               std::to_string(iter));
    }
    energies = result.energies;
    for (int k = 0; k < num_roots; ++k) {
      std::printf("%5d %10.2e %12zu %5d %20.12f %9d %8.2f %8.2f %8.2f %9.2f\n",
                  iter, epsilon, new_size, k,
                  energies[k] + ints.core_energy(), result.num_iterations,
                  build_start - select_start, solve_start - build_start,
                  omp_get_wtime() - solve_start, omp_get_wtime() - start_time);
    }
    if (finished && !converged_space) {
      std::printf("note: stopped at max_iter = %d before the energies "
                  "converged to energy_tol\n",
                  input.select.max_iter);
    }
  }
  std::printf("variational space: %zu determinants, %zu stored Hamiltonian "
              "elements on the root\n",
              shared_dets.size(), h.num_elements());
  PrintConnectStats(stats);

  root_dets = std::vector<Det<W>>();
  const SharedVectors vectors =
      ShareVectors(root_vectors, shared_dets.size(), comm);
  root_vectors = std::vector<std::vector<Complex>>();
  const std::vector<PtResult> pt =
      ComputePerturbation(input, ints, shared_dets.span(), energies, vectors,
                          &heat_bath, filter, comm);
  WriteResults(input, ints, shared_dets.span(), h, labels, energies, vectors,
               pt, comm);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_SELECT_H_
