// Pieces shared by the "diag" and "select" drivers.

#ifndef ZSHCI_SRC_DRIVER_COMMON_H_
#define ZSHCI_SRC_DRIVER_COMMON_H_

#include <omp.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm.h"
#include "connect.h"
#include "davidson.h"
#include "det.h"
#include "excite.h"
#include "hamiltonian.h"
#include "heat_bath.h"
#include "input.h"
#include "integrals.h"
#include "output.h"
#include "pt.h"
#include "rdm.h"
#include "sparse_h.h"

namespace zshci {

// Builds the determinant list of the input and checks it: every determinant
// must hold the same number of electrons, within the orbital range, and
// appear once. Call on the root only.
template <int W>
std::vector<Det<W>> LoadDeterminants(const Input& input, int num_orbs) {
  std::vector<Det<W>> dets;
  if (!input.detfile.empty()) {
    int file_width = 0;
    const std::vector<uint64_t> words =
        ReadDetWords(input.detfile, &file_width);
    if (file_width < 1) throw std::runtime_error("detfile: empty table");
    dets.resize(words.size() / file_width);
    for (size_t d = 0; d < dets.size(); ++d) {
      for (int k = 0; k < file_width; ++k) {
        const uint64_t word = words[d * file_width + k];
        if (k < W) {
          dets[d].words[k] = word;
        } else if (word != 0) {
          throw std::runtime_error("detfile: determinant " +
                                   std::to_string(d) +
                                   " occupies a spinor beyond NORB");
        }
      }
    }
  } else {
    dets.resize(input.dets.size());
    for (size_t d = 0; d < dets.size(); ++d) {
      for (int p : input.dets[d]) {
        if (p < 0 || p >= num_orbs) {
          throw std::runtime_error("dets: determinant " + std::to_string(d) +
                                   " has spinor index " + std::to_string(p) +
                                   " outside 0.." +
                                   std::to_string(num_orbs - 1));
        }
        if (dets[d].Occupied(p)) {
          throw std::runtime_error("dets: determinant " + std::to_string(d) +
                                   " lists spinor " + std::to_string(p) +
                                   " twice");
        }
        dets[d].Set(p);
      }
    }
  }
  if (dets.empty()) throw std::runtime_error("the determinant list is empty");

  const int num_elec = dets[0].Count();
  for (size_t d = 0; d < dets.size(); ++d) {
    if (dets[d].Count() != num_elec) {
      throw std::runtime_error("determinant " + std::to_string(d) +
                               " does not have " + std::to_string(num_elec) +
                               " electrons");
    }
    for (int p = num_orbs; p < 64 * W; ++p) {
      if (dets[d].Occupied(p)) {
        throw std::runtime_error("determinant " + std::to_string(d) +
                                 " occupies a spinor beyond NORB");
      }
    }
  }
  std::vector<Det<W>> sorted = dets;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    throw std::runtime_error("the determinant list has duplicates");
  }
  return dets;
}

// The value of every label set for a determinant: the sum of the labels of
// its occupied spinors, reduced modulo `mod` if that is nonzero.
template <int W>
std::vector<int> DeterminantLabels(const Det<W>& det,
                                   const std::vector<LabelSet>& sets) {
  std::vector<int> occ;
  det.AppendOccupied(&occ);
  std::vector<int> values;
  for (const LabelSet& set : sets) {
    long sum = 0;
    for (int p : occ) sum += set.labels[p];
    if (set.mod > 0) sum = ((sum % set.mod) + set.mod) % set.mod;
    values.push_back(static_cast<int>(sum));
  }
  return values;
}

// "3", "(3, 1)", or "none" without label sets.
inline std::string FormatLabels(const std::vector<int>& values) {
  if (values.empty()) return "none";
  if (values.size() == 1) return std::to_string(values[0]);
  std::string text = "(";
  for (size_t k = 0; k < values.size(); ++k) {
    text += (k == 0 ? "" : ", ") + std::to_string(values[k]);
  }
  return text + ")";
}

// Starting vectors: eigenvectors of H in the span of the `guess_dim`
// determinants of lowest diagonal energy.
template <int W, typename S>
std::vector<std::vector<S>> GuessVectors(Span<Det<W>> dets, const SparseH<S>& h,
                                         const Integrals<S>& ints,
                                         int num_roots) {
  const size_t n = dets.size();
  const size_t guess_dim =
      std::min(n, static_cast<size_t>(std::max(100, 20 * num_roots)));
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  const std::vector<double>& diagonal = h.diagonal();
  std::partial_sort(order.begin(), order.begin() + guess_dim, order.end(),
                    [&](size_t a, size_t b) {
                      return diagonal[a] != diagonal[b]
                                 ? diagonal[a] < diagonal[b]
                                 : a < b;
                    });
  using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
  Matrix small(guess_dim, guess_dim);
  for (size_t a = 0; a < guess_dim; ++a) {
    small(a, a) = diagonal[order[a]];
    for (size_t b = a + 1; b < guess_dim; ++b) {
      small(a, b) = Hij(dets[order[a]], dets[order[b]], ints);
      small(b, a) = Conj(small(a, b));
    }
  }
  Eigen::SelfAdjointEigenSolver<Matrix> solver(small);
  std::vector<std::vector<S>> guess(num_roots, std::vector<S>(n, S(0.0)));
  for (int k = 0; k < num_roots; ++k) {
    for (size_t a = 0; a < guess_dim; ++a) {
      guess[k][order[a]] = solver.eigenvectors()(a, k);
    }
  }
  return guess;
}

// Compares the block and the threaded product with their slow references.
// Throws if they disagree.
template <int W, typename S>
void CheckHamiltonian(Span<Det<W>> dets, const SparseH<S>& h,
                      const Integrals<S>& ints, const ConnectOptions& options) {
  const CsrBlock<S> reference = BuildBlockBruteForce(dets, 0, ints, options);
  // The blocks of h may be one per thread; compare a single one.
  if (!SameBlock(BuildBlock(dets, 0, ints, options), reference, 0.0)) {
    throw std::runtime_error(
        "check: the bucket search and the brute-force search disagree");
  }

  const size_t n = h.dim();
  std::vector<S> x(n), y(n), y_ref(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = ScalarTraits<S>::Make(std::sin(0.37 * i + 0.1), std::cos(1.3 * i));
  }
  h.Multiply(x.data(), y.data());
  h.MultiplyReference(x.data(), y_ref.data());
  double norm = 0.0;
  double max_diff = 0.0;
  for (size_t i = 0; i < n; ++i) {
    max_diff = std::max(max_diff, std::abs(y[i] - y_ref[i]));
    norm = std::max(norm, std::abs(y_ref[i]));
  }
  if (max_diff > 1e-12 * std::max(1.0, norm)) {
    throw std::runtime_error(
        "check: the threaded and the serial H*x disagree");
  }
  std::printf("check: Hamiltonian block and H*x agree with the references\n");
}

template <int W, typename S>
void PrintLeadingDeterminants(Span<Det<W>> dets, Span<S> vector, int num_orbs,
                              int count) {
  std::vector<size_t> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  const size_t shown = std::min(order.size(), static_cast<size_t>(count));
  std::partial_sort(order.begin(), order.begin() + shown, order.end(),
                    [&](size_t a, size_t b) {
                      return std::abs(vector[a]) > std::abs(vector[b]);
                    });
  for (size_t m = 0; m < shown; ++m) {
    const size_t i = order[m];
    if (ScalarTraits<S>::kIsComplex) {
      std::printf("    %14.10f %14.10f  %s\n", Real(vector[i]), Imag(vector[i]),
                  dets[i].ToString(num_orbs).c_str());
    } else {
      std::printf("    %14.10f  %s\n", Real(vector[i]),
                  dets[i].ToString(num_orbs).c_str());
    }
  }
}

// Extends h to all of `dets`. Only the root, which applies the diagonal and
// runs the eigensolver, stores the diagonal elements of dets[first_new..].
template <int W, typename S>
void AppendDiagonal(Span<Det<W>> dets, size_t first_new,
                    const Integrals<S>& ints, const Comm& comm, SparseH<S>* h) {
  if (!comm.is_root()) {
    h->Grow(dets.size());
    return;
  }
  std::vector<double> diagonal(dets.size() - first_new);
  const long count = static_cast<long>(diagonal.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < count; ++i) {
    diagonal[i] = DiagonalEnergy(dets[first_new + i], ints);
  }
  h->AppendDiagonal(diagonal);
}

// The CI vectors of all roots, held once per node: root k occupies
// [k * dim, (k + 1) * dim).
template <typename S>
struct SharedVectors {
  SharedArray<S> data;
  size_t dim = 0;
  int num_roots = 0;

  Span<S> Root(int k) const { return data.span(k * dim, dim); }
};

// Collective: puts the root's eigenvectors where every rank can read them.
template <typename S>
SharedVectors<S> ShareVectors(const std::vector<std::vector<S>>& vectors,
                              size_t dim, const Comm& comm) {
  SharedVectors<S> shared;
  shared.dim = dim;
  shared.num_roots = static_cast<int>(vectors.size());
  comm.BcastValue(&shared.num_roots);
  shared.data = SharedArray<S>(shared.num_roots * dim, comm);
  if (comm.is_root()) {
    for (int k = 0; k < shared.num_roots; ++k) {
      std::copy(vectors[k].begin(), vectors[k].end(),
                shared.data.data() + k * dim);
    }
  }
  shared.data.ShareFromRoot(comm);
  return shared;
}

// Collective: the determinants of the root, sorted, once per node.
template <int W>
SharedArray<Det<W>> ShareSorted(Span<Det<W>> dets, const Comm& comm) {
  std::vector<Det<W>> sorted;
  if (comm.is_root()) {
    sorted.assign(dets.begin(), dets.end());
    std::sort(sorted.begin(), sorted.end());
  }
  return SharedArray<Det<W>>::FromRoot(sorted, comm);
}

// The connection search requested in the input.
inline ConnectOptions MakeConnectOptions(const Input& input) {
  ConnectOptions options;
  options.threshold = input.hij_threshold;
  options.group_a = input.connect_group_a;
  options.method = input.connect_by_groups     ? ConnectMethod::kGroups
                   : input.connect_by_segments ? ConnectMethod::kSegments
                                               : ConnectMethod::kStrings;
  options.num_segments = input.connect_segments;
  options.num_batches = input.connect_batches;
  return options;
}

// Prints what the connection search of the root did.
inline void PrintConnectStats(const ConnectStats& stats) {
  std::printf("connection search (root): %llu pairs tested, %llu connected "
              "(%.1f%%), largest bucket %llu\n",
              static_cast<unsigned long long>(stats.tested_pairs),
              static_cast<unsigned long long>(stats.connected_pairs),
              stats.tested_pairs == 0
                  ? 100.0
                  : 100.0 * stats.connected_pairs / stats.tested_pairs,
              static_cast<unsigned long long>(stats.largest_bucket));
  if (stats.num_strings > 0) {
    std::printf(
        "connection search (root): %llu distinct strings, %llu "
        "entries in their relation lists\n",
        static_cast<unsigned long long>(stats.num_strings),
        static_cast<unsigned long long>(stats.num_links));
  }
  std::printf(
      "connection search (root): %.2f s preparing, %.2f s counting, "
      "%.2f s filling, %.2f s sorting rows\n",
      stats.seconds_prepare, stats.seconds_count, stats.seconds_fill,
      stats.seconds_sort);
}

// The excitation filter of the calculation: label conservation and, unless
// `with_restrictions` is false, the occupation restrictions of the input.
inline RestrictedFilter MakeFilter(const Input& input,
                                   const std::vector<LabelSet>& symmetry,
                                   int num_orbs, bool with_restrictions) {
  std::vector<OccupationRestriction> restrictions;
  if (with_restrictions) {
    for (const RestrictionInput& item : input.restrictions) {
      restrictions.push_back({item.orbitals, item.min_elec, item.max_elec});
    }
  }
  // The label sets are those of the integrals, whose heat-bath tables then
  // hold label-conserving double excitations only.
  LabelFilter labels(symmetry);
  labels.set_doubles_conserved(true);
  return RestrictedFilter(std::move(labels), restrictions, num_orbs);
}

// Runs the perturbative correction requested in the input for every root.
// `heat_bath` may be null, in which case the tables are built here.
template <int W, typename S>
std::vector<PtResult> ComputePerturbation(
    const Input& input, const Integrals<S>& ints, Span<Det<W>> dets,
    const std::vector<double>& result_energies, const SharedVectors<S>& vectors,
    const HeatBath* heat_bath, const RestrictedFilter& filter,
    const Comm& comm) {
  std::vector<PtResult> results;
  if (input.pt.method == PtMethod::kNone) return results;

  const double time = omp_get_wtime();
  std::unique_ptr<HeatBath> owned;
  if (heat_bath == nullptr) {
    owned = std::make_unique<HeatBath>(ints, input.select.integral_threshold,
                                       comm);
    heat_bath = owned.get();
  }
  const SharedArray<Det<W>> sorted_dets = ShareSorted(dets, comm);

  for (int k = 0; k < input.nroots; ++k) {
    const PtProblem<W, RestrictedFilter, S> problem{
        dets, sorted_dets.span(), vectors.Root(k), result_energies[k],
        ints, *heat_bath,         filter,          input.pt.num_batches,
        comm};
    results.push_back(
        RunPerturbation(problem, input.pt, k, input.output_level > 1));
    const PtResult& pt = results.back();
    if (pt.error > 0.0 || input.pt.method != PtMethod::kDeterministic) {
      std::printf("State %3d  PT correction = %16.10f +/- %.2e  (%d "
                  "iterations, %ld walker samples)\n",
                  k, pt.correction, pt.error, pt.num_iterations,
                  pt.num_estimates);
    } else {
      std::printf("State %3d  PT correction = %16.10f\n", k, pt.correction);
    }
    if (!pt.converged) {
      std::printf("note: state %d did not reach target_error = %.1e in %d "
                  "iterations\n",
                  k, input.pt.target_error, pt.num_iterations);
    }
  }
  std::printf("perturbation theory: %.2f s\n", omp_get_wtime() - time);
  return results;
}

// Prints the states and writes shci.e, wfn.h5 and the requested density
// matrices. `result_energies` are without the core energy; `pt` is empty if
// no perturbative correction was computed. All ranks take part in the
// density matrices; only the root writes.
template <int W, typename S>
void WriteResults(const Input& input, const Integrals<S>& ints,
                  Span<Det<W>> dets, const SparseH<S>& h,
                  const std::vector<LabelSet>& symmetry,
                  const std::vector<double>& result_energies,
                  const SharedVectors<S>& vectors,
                  const std::vector<PtResult>& pt, const Comm& comm) {
  const size_t n = dets.size();
  const int num_orbs = ints.num_orbs();
  const int num_elec = dets[0].Count();
  double time = omp_get_wtime();
  std::vector<double> energies(input.nroots);
  std::vector<double> final_energies(input.nroots);
  for (int k = 0; k < input.nroots; ++k) {
    energies[k] = result_energies[k] + ints.core_energy();
    final_energies[k] = energies[k];
    std::printf("State %3d  Energy = %20.12f\n", k, energies[k]);
    if (!pt.empty()) {
      final_energies[k] += pt[k].correction;
      std::printf("State %3d  Energy + PT = %15.12f +/- %.2e\n", k,
                  final_energies[k], pt[k].error);
    }
    if (input.output_level > 0 && comm.is_root()) {
      PrintLeadingDeterminants(dets, vectors.Root(k), num_orbs, 6);
    }
  }

  if (comm.is_root()) {
    WriteEnergies(input.prefix, final_energies);
    WaveFunction<S> wfn;
    wfn.num_orbs = num_orbs;
    wfn.num_elec = num_elec;
    wfn.det_width = W;
    wfn.det_words.resize(n * W);
    for (size_t d = 0; d < n; ++d) {
      for (int k = 0; k < W; ++k) wfn.det_words[d * W + k] = dets[d].words[k];
    }
    for (int k = 0; k < input.nroots; ++k) {
      const Span<S> root = vectors.Root(k);
      wfn.coefficients.emplace_back(root.begin(), root.end());
    }
    wfn.energies = energies;
    for (const PtResult& result : pt) {
      wfn.pt_corrections.push_back(result.correction);
      wfn.pt_errors.push_back(result.error);
    }
    for (const LabelSet& set : symmetry) {
      wfn.labels.insert(wfn.labels.end(), set.labels.begin(), set.labels.end());
      wfn.label_mods.push_back(set.mod);
    }
    wfn.input = input.resolved;
    WriteWaveFunction(input.prefix + "/wfn.h5", wfn);
  }

  if (input.rdm.one || input.rdm.two) {
    time = omp_get_wtime();
    for (int k = 0; k < input.nroots; ++k) {
      const DensityMatrices<S> dm = ComputeDensityMatrices(
          dets, h, num_orbs, vectors.Root(k), vectors.Root(k), input.rdm.two,
          comm, ints.is_spatial());
      if (!comm.is_root()) continue;
      WriteDensityMatrices(input.prefix, k, k, dm);
      if (input.rdm.two) {
        const double from_rdm =
            EnergyFromDensityMatrices(dm, ints) + ints.core_energy();
        std::printf("State %3d  energy from density matrices = %20.12f "
                    "(difference %.2e)\n",
                    k, from_rdm, from_rdm - energies[k]);
      }
    }
    std::printf("density matrices: %.2f s\n", omp_get_wtime() - time);
  }
  if (input.rdm.transition) {
    for (int k = 1; k < input.nroots; ++k) {
      const DensityMatrices<S> dm = ComputeDensityMatrices(
          dets, h, num_orbs, vectors.Root(0), vectors.Root(k),
          /*with_two=*/false, comm, ints.is_spatial());
      if (comm.is_root()) WriteDensityMatrices(input.prefix, 0, k, dm);
    }
  }
}

}  // namespace zshci

#endif  // ZSHCI_SRC_DRIVER_COMMON_H_
