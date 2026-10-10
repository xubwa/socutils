// Self-checks of the zshci core that need no reference program.
//
// Usage: zshci_test FCIDUMP
//
// The integral file may be any complex FCIDUMP with at most 32 spinors and at
// least two more spinors than electrons.

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "connect.h"
#include "davidson.h"
#include "det.h"
#include "excite.h"
#include "hamiltonian.h"
#include "heat_bath.h"
#include "integrals.h"
#include "pt.h"
#include "rdm.h"
#include "select.h"
#include "sparse_h.h"

namespace zshci {
namespace {

int g_num_failures = 0;

void Expect(bool condition, const std::string& what) {
  std::printf("%s  %s\n", condition ? "ok  " : "FAIL", what.c_str());
  if (!condition) ++g_num_failures;
}

// Random occupation lists: the aufbau determinant and distinct excitations
// of it, of any rank up to max_rank.
std::vector<std::vector<int>> RandomOccupations(int num_orbs, int num_elec,
                                                int count, int max_rank,
                                                unsigned seed) {
  std::mt19937 rng(seed);
  std::set<std::vector<int>> seen;
  std::vector<std::vector<int>> out;
  std::vector<int> reference(num_elec);
  for (int i = 0; i < num_elec; ++i) reference[i] = i;
  seen.insert(reference);
  out.push_back(reference);
  while (static_cast<int>(out.size()) < count) {
    std::set<int> occ(reference.begin(), reference.end());
    const int rank = 1 + rng() % max_rank;
    for (int k = 0; k < rank; ++k) {
      std::vector<int> occupied(occ.begin(), occ.end());
      const int hole = occupied[rng() % occupied.size()];
      int particle;
      do {
        particle = rng() % num_orbs;
      } while (occ.count(particle) != 0);
      occ.erase(hole);
      occ.insert(particle);
    }
    std::vector<int> det(occ.begin(), occ.end());
    if (seen.insert(det).second) out.push_back(det);
  }
  return out;
}

template <int W>
std::vector<Det<W>> ToDets(const std::vector<std::vector<int>>& occupations,
                           int shift) {
  std::vector<Det<W>> dets(occupations.size());
  for (size_t d = 0; d < dets.size(); ++d) {
    for (int p : occupations[d]) dets[d].Set(p + shift);
  }
  return dets;
}

// The same integrals with every spinor index increased by `shift`.
Integrals<Complex> ShiftIntegrals(const Integrals<Complex>& ints, int shift) {
  const int n = ints.num_orbs();
  Integrals<Complex> shifted(n + shift);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      shifted.SetOneBody(p + shift, q + shift, ints.OneBody(p, q));
      for (int s = 1; s < n && p < q; ++s) {
        for (int r = 0; r < s; ++r) {
          shifted.SetAntisymOrdered(p + shift, q + shift, r + shift, s + shift,
                                    ints.AntisymOrdered(p, q, r, s));
        }
      }
    }
  }
  shifted.set_num_elec(ints.num_elec());
  shifted.set_core_energy(ints.core_energy());
  shifted.Finalize();
  return shifted;
}

void TestCountBetween() {
  std::mt19937_64 rng(7);
  bool ok = true;
  for (int trial = 0; trial < 2000 && ok; ++trial) {
    Det<3> det;
    for (uint64_t& word : det.words) word = rng();
    int p = rng() % 192;
    int q = rng() % 192;
    if (p == q) continue;
    if (p > q) std::swap(p, q);
    int naive = 0;
    for (int k = p + 1; k < q; ++k) naive += det.Occupied(k) ? 1 : 0;
    ok = det.CountBetween(p, q) == naive;
  }
  Expect(ok, "Det<3>::CountBetween agrees with a bit-by-bit count");
}

void TestHamiltonian(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<std::vector<int>> occupations =
      RandomOccupations(n, ints.num_elec(), 500, 3, 11);
  ConnectOptions keep_all;
  ConnectOptions batched;
  batched.num_batches = 4;
  ConnectOptions thresholded;
  thresholded.threshold = 1e-6;

  const std::vector<Det<1>> dets1 = ToDets<1>(occupations, 0);
  const CsrBlock<Complex> block1 =
      BuildBlock(MakeSpan(dets1), 0, ints, keep_all);
  Expect(SameBlock(block1, BuildBlockBruteForce(MakeSpan(dets1), 0, ints, keep_all), 0),
         "bucket search equals brute force (one word)");
  Expect(SameBlock(block1, BuildBlock(MakeSpan(dets1), 0, ints, batched), 0),
         "bucket search in four batches gives the same block");
  Expect(SameBlock(BuildBlock(MakeSpan(dets1), 0, ints, thresholded),
                   BuildBlockBruteForce(MakeSpan(dets1), 0, ints, thresholded), 0),
         "bucket search equals brute force with a threshold");

  // The segment method, for several numbers of segments (fewer than five
  // means a single cover, i.e. all pairs in one bucket).
  bool segments_ok = true;
  for (int num_segments : {1, 3, 5, 6, 8, 11, 16}) {
    ConnectOptions segmented;
    segmented.method = ConnectMethod::kSegments;
    segmented.num_segments = num_segments;
    segments_ok = segments_ok &&
                  SameBlock(block1, BuildBlock(MakeSpan(dets1), 0, ints, segmented), 0);
  }
  Expect(segments_ok, "segment search equals the string search");
  {
    ConnectOptions segmented;
    segmented.method = ConnectMethod::kSegments;
    segmented.num_segments = 7;
    segmented.threshold = 1e-6;
    Expect(SameBlock(BuildBlock(MakeSpan(dets1), 0, ints, segmented),
                     BuildBlockBruteForce(MakeSpan(dets1), 0, ints, thresholded), 0),
           "segment search equals brute force with a threshold");
    segmented.threshold = -1.0;
    Expect(SameBlock(BuildBlock(MakeSpan(dets1), 300, ints, segmented),
                     BuildBlock(MakeSpan(dets1), 300, ints, keep_all), 0),
           "segment search builds the same incremental block");
  }

  // Blocks built for a growing space add up to the block of the full space.
  {
    const size_t first_new = 300;
    const std::vector<Det<1>> old_dets(dets1.begin(),
                                       dets1.begin() + first_new);
    const CsrBlock<Complex> old_block =
        BuildBlock(MakeSpan(old_dets), 0, ints, keep_all);
    const CsrBlock<Complex> new_block =
        BuildBlock(MakeSpan(dets1), first_new, ints, keep_all);
    bool ok = old_block.num_elements() + new_block.num_elements() ==
              block1.num_elements();
    for (size_t k = 0; k < new_block.col.size() && ok; ++k) {
      ok = new_block.col[k] >= first_new;
    }
    Expect(ok, "incremental blocks partition the elements of the full block");
  }

  // Shifting all spinors across a word boundary must not change anything.
  const int shift = 66 - n;
  const Integrals<Complex> shifted = ShiftIntegrals(ints, shift);
  const std::vector<Det<2>> dets2 = ToDets<2>(occupations, shift);
  const CsrBlock<Complex> block2 =
      BuildBlock(MakeSpan(dets2), 0, shifted, keep_all);
  Expect(SameBlock(block1, block2, 1e-14),
         "two-word determinants give the same Hamiltonian block");
  {
    ConnectOptions segmented;
    segmented.method = ConnectMethod::kSegments;
    segmented.num_segments = 9;
    Expect(SameBlock(block1, BuildBlock(MakeSpan(dets2), 0, shifted, segmented),
                     1e-14),
           "segment search works across a word boundary");
  }
  bool same_diagonal = true;
  for (size_t d = 0; d < dets1.size(); ++d) {
    same_diagonal = same_diagonal &&
                    std::abs(DiagonalEnergy(dets1[d], ints) -
                             DiagonalEnergy(dets2[d], shifted)) < 1e-11;
  }
  Expect(same_diagonal, "two-word determinants give the same diagonal");

  // Solver and density matrices.
  SparseH<Complex> h;
  std::vector<double> diagonal(dets1.size());
  for (size_t d = 0; d < dets1.size(); ++d) {
    diagonal[d] = DiagonalEnergy(dets1[d], ints);
  }
  h.AppendDiagonal(diagonal);
  h.AppendBlock(BuildBlock(MakeSpan(dets1), 0, ints, keep_all));

  const size_t dim = h.dim();
  std::vector<Complex> x(dim), y(dim), y_ref(dim);
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (Complex& value : x) value = Complex(uniform(rng), uniform(rng));
  h.Multiply(x.data(), y.data());
  h.MultiplyReference(x.data(), y_ref.data());
  double max_diff = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    max_diff = std::max(max_diff, std::abs(y[i] - y_ref[i]));
  }
  Expect(max_diff < 1e-12, "threaded H*x equals the serial product");

  const int num_roots = 3;
  DavidsonOptions dense_options;
  dense_options.dense_dim = 1 << 20;
  std::vector<std::vector<Complex>> exact(num_roots,
                                          std::vector<Complex>(dim, 0.0));
  const DavidsonResult dense = Davidson(h, dense_options, &exact);

  DavidsonOptions iterative_options;
  iterative_options.dense_dim = 0;
  iterative_options.tol = 1e-9;
  std::vector<std::vector<Complex>> vectors(num_roots,
                                            std::vector<Complex>(dim, 0.0));
  for (int k = 0; k < num_roots; ++k) {
    for (size_t i = 0; i < dim; ++i) {
      vectors[k][i] = exact[k][i] + 0.05 * Complex(uniform(rng), uniform(rng));
    }
  }
  const DavidsonResult iterative = Davidson(h, iterative_options, &vectors);
  bool same_energies = iterative.converged;
  for (int k = 0; k < num_roots; ++k) {
    same_energies = same_energies &&
                    std::abs(iterative.energies[k] - dense.energies[k]) < 1e-10;
  }
  Expect(same_energies, "Davidson reproduces the dense eigenvalues");

  const DensityMatrices<Complex> dm = ComputeDensityMatrices(
      MakeSpan(dets1), h, n, MakeSpan(exact[0]), MakeSpan(exact[0]), true);
  Expect(std::abs(EnergyFromDensityMatrices(dm, ints) - dense.energies[0]) <
             1e-10,
         "energy from the density matrices equals the eigenvalue");
  Complex trace(0.0);
  double max_antihermitian = 0.0;
  for (int p = 0; p < n; ++p) {
    trace += dm.One(p, p);
    for (int q = 0; q < n; ++q) {
      max_antihermitian =
          std::max(max_antihermitian,
                   std::abs(dm.One(p, q) - std::conj(dm.One(q, p))));
    }
  }
  Expect(std::abs(trace - Complex(ints.num_elec())) < 1e-10,
         "the one-body density matrix traces to the electron number");
  Expect(max_antihermitian < 1e-12,
         "the one-body density matrix is Hermitian");

  // Partial trace of D2 over the second index pair gives (N - 1) D1.
  double max_trace_diff = 0.0;
  for (int p = 0; p < n; ++p) {
    for (int r = 0; r < n; ++r) {
      Complex sum(0.0);
      for (int q = 0; q < n; ++q) sum += dm.Two(p, q, r, q);
      max_trace_diff = std::max(
          max_trace_diff,
          std::abs(sum - double(ints.num_elec() - 1) * dm.One(p, r)));
    }
  }
  Expect(max_trace_diff < 1e-10,
         "the two-body density matrix contracts to (N-1) times the one-body");

  // Transition density between two eigenvectors: <0|p^+ q|1> = conj(<1|q^+
  // p|0>).
  const DensityMatrices<Complex> t01 = ComputeDensityMatrices(
      MakeSpan(dets1), h, n, MakeSpan(exact[0]), MakeSpan(exact[1]), false);
  const DensityMatrices<Complex> t10 = ComputeDensityMatrices(
      MakeSpan(dets1), h, n, MakeSpan(exact[1]), MakeSpan(exact[0]), false);
  double max_transition_diff = 0.0;
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      max_transition_diff =
          std::max(max_transition_diff,
                   std::abs(t01.One(p, q) - std::conj(t10.One(q, p))));
    }
  }
  Expect(max_transition_diff < 1e-12,
         "transition density matrices obey <0|p^+ q|1> = conj(<1|q^+ p|0>)");
}

// Selection through the heat-bath tables against a direct evaluation of
// |H(a,i)| w_i for every single and double excitation. `restrictions` bound
// the number of electrons in sets of spinors.
void TestSelection(const Integrals<Complex>& ints,
                   const std::vector<LabelSet>& symmetry,
                   const std::vector<OccupationRestriction>& restrictions,
                   const std::string& what) {
  const int n = ints.num_orbs();
  // Counts the electrons of each restricted set directly.
  auto within_bounds = [&](const Det<1>& det) {
    for (const OccupationRestriction& restriction : restrictions) {
      int count = 0;
      for (int p : restriction.orbitals) count += det.Occupied(p) ? 1 : 0;
      if (count < restriction.min_elec || count > restriction.max_elec) {
        return false;
      }
    }
    return true;
  };
  // As in a calculation, every determinant of the space obeys the
  // restrictions; the filter relies on that.
  std::vector<Det<1>> dets;
  for (const Det<1>& det :
       ToDets<1>(RandomOccupations(n, ints.num_elec(), 200, 2, 5), 0)) {
    if (within_bounds(det) && dets.size() < 40) dets.push_back(det);
  }
  std::vector<Det<1>> sorted_dets = dets;
  std::sort(sorted_dets.begin(), sorted_dets.end());
  std::mt19937 rng(17);
  std::uniform_real_distribution<double> uniform(0.01, 1.0);
  std::vector<double> weights(dets.size());
  for (double& w : weights) w = uniform(rng);
  const double epsilon = 2e-3;

  const HeatBath heat_bath(ints, 1e-12);
  const RestrictedFilter filter(LabelFilter(symmetry), restrictions, n);
  const std::vector<Det<1>> selected =
      SelectNewDeterminants(MakeSpan(dets), MakeSpan(sorted_dets),
                            MakeSpan(weights), epsilon, ints, heat_bath, filter);

  // Whether moving electrons from i, j to a, b (-1: none) conserves every
  // label set.
  auto conserved = [&](int i, int j, int a, int b) {
    for (const LabelSet& set : symmetry) {
      auto label = [&](int p) { return p < 0 ? 0 : set.labels[p]; };
      const int change = label(a) + label(b) - label(i) - label(j);
      if (set.mod == 0 ? change != 0 : change % set.mod != 0) return false;
    }
    return true;
  };
  std::set<Det<1>> expected;
  for (size_t d = 0; d < dets.size(); ++d) {
    const double cutoff = epsilon / weights[d];
    std::vector<int> occ;
    dets[d].AppendOccupied(&occ);
    std::vector<int> virt;
    for (int p = 0; p < n; ++p) {
      if (!dets[d].Occupied(p)) virt.push_back(p);
    }
    for (int i : occ) {
      for (int a : virt) {
        if (!conserved(i, -1, a, -1)) continue;
        Det<1> det = dets[d];
        det.Clear(i);
        det.Set(a);
        if (within_bounds(det) && std::abs(Hij(det, dets[d], ints)) > cutoff) {
          expected.insert(det);
        }
      }
    }
    for (size_t x = 0; x < occ.size(); ++x) {
      for (size_t y = x + 1; y < occ.size(); ++y) {
        for (size_t u = 0; u < virt.size(); ++u) {
          for (size_t v = u + 1; v < virt.size(); ++v) {
            if (!conserved(occ[x], occ[y], virt[u], virt[v])) continue;
            Det<1> det = dets[d];
            det.Clear(occ[x]);
            det.Clear(occ[y]);
            det.Set(virt[u]);
            det.Set(virt[v]);
            if (within_bounds(det) &&
                std::abs(Hij(det, dets[d], ints)) >= cutoff) {
              expected.insert(det);
            }
          }
        }
      }
    }
  }
  for (const Det<1>& det : dets) expected.erase(det);
  const std::vector<Det<1>> expected_sorted(expected.begin(), expected.end());
  Expect(dets.size() >= 10 && !selected.empty() && selected == expected_sorted,
         what);
}

// Perturbation theory against a direct sum over all excitations.
void TestPerturbation(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 60, 2, 23), 0);
  std::vector<Det<1>> sorted_dets = dets;
  std::sort(sorted_dets.begin(), sorted_dets.end());

  // The lowest eigenvector in this small space.
  SparseH<Complex> h;
  std::vector<double> diagonal(dets.size());
  for (size_t d = 0; d < dets.size(); ++d) {
    diagonal[d] = DiagonalEnergy(dets[d], ints);
  }
  h.AppendDiagonal(diagonal);
  h.AppendBlock(BuildBlock(MakeSpan(dets), 0, ints, ConnectOptions()));
  std::vector<std::vector<Complex>> vectors(
      1, std::vector<Complex>(dets.size(), 0.0));
  const double energy = Davidson(h, DavidsonOptions(), &vectors).energies[0];
  // An arbitrary global phase must not matter.
  for (Complex& c : vectors[0]) c *= std::polar(1.0, 0.7);
  const std::vector<Complex>& c = vectors[0];

  const double epsilon = 1e-4;
  const double epsilon_large = 2e-3;
  auto direct = [&](double eps) {
    std::map<Det<1>, Complex> numerators;
    for (size_t d = 0; d < dets.size(); ++d) {
      std::vector<int> occ;
      dets[d].AppendOccupied(&occ);
      std::vector<int> virt;
      for (int p = 0; p < n; ++p) {
        if (!dets[d].Occupied(p)) virt.push_back(p);
      }
      auto add = [&](const Det<1>& det, bool strict) {
        const Complex x = Hij(det, dets[d], ints) * c[d];
        if (strict ? std::abs(x) > eps : std::abs(x) >= eps) {
          numerators[det] += x;
        }
      };
      for (int i : occ) {
        for (int a : virt) {
          Det<1> det = dets[d];
          det.Clear(i);
          det.Set(a);
          add(det, true);
        }
      }
      for (size_t x = 0; x < occ.size(); ++x) {
        for (size_t y = x + 1; y < occ.size(); ++y) {
          for (size_t u = 0; u < virt.size(); ++u) {
            for (size_t v = u + 1; v < virt.size(); ++v) {
              Det<1> det = dets[d];
              det.Clear(occ[x]);
              det.Clear(occ[y]);
              det.Set(virt[u]);
              det.Set(virt[v]);
              add(det, false);
            }
          }
        }
      }
    }
    double e2 = 0.0;
    for (const auto& entry : numerators) {
      if (std::binary_search(sorted_dets.begin(), sorted_dets.end(),
                             entry.first)) {
        continue;
      }
      e2 += std::norm(entry.second) /
            (energy - DiagonalEnergy(entry.first, ints));
    }
    return e2;
  };

  const HeatBath heat_bath(ints, 1e-12);
  const LabelFilter filter(std::vector<int>(n, 0), 0);
  const Comm serial;
  PtProblem<1, LabelFilter, Complex> problem{
      dets, sorted_dets, c, energy, ints, heat_bath, filter, 1, serial};
  const double exact = direct(epsilon);
  const double exact_large = direct(epsilon_large);
  const double deterministic = PtDeterministic(problem, epsilon);
  Expect(exact < -1e-4 && std::abs(deterministic - exact) < 1e-12,
         "deterministic PT equals the direct sum");
  problem.num_batches = 3;
  Expect(std::abs(PtDeterministic(problem, epsilon) - exact) < 1e-12,
         "deterministic PT in three batches gives the same energy");
  problem.num_batches = 1;

  // The stochastic estimator is unbiased: its mean over many samples must
  // agree with the deterministic value within the statistical error.
  const DeterminantSampler sampler(MakeSpan(c), serial);
  std::mt19937_64 rng(5);
  WalkerWorkspace<1, Complex> scratch;
  const int num_iterations = 400;
  double sum = 0.0, sum_squares = 0.0;
  double sum_large = 0.0, sum_squares_large = 0.0;
  double max_equal_cutoff_diff = 0.0;
  double max_batch_diff = 0.0;
  for (int iter = 0; iter < num_iterations; ++iter) {
    // The same random state with and without batching must give the same
    // estimate.
    std::mt19937_64 rng_copy = rng;
    const StochasticEstimate estimate = PtWalkerSample(
        problem, epsilon, epsilon_large, 20, sampler, &rng, &scratch);
    problem.num_batches = 3;
    const StochasticEstimate batched = PtWalkerSample(
        problem, epsilon, epsilon_large, 20, sampler, &rng_copy, &scratch);
    problem.num_batches = 1;
    max_batch_diff =
        std::max(max_batch_diff, std::abs(batched.small - estimate.small));
    sum += estimate.small;
    sum_squares += estimate.small * estimate.small;
    sum_large += estimate.large;
    sum_squares_large += estimate.large * estimate.large;
    const StochasticEstimate same = PtWalkerSample(
        problem, epsilon, epsilon, 20, sampler, &rng, &scratch);
    max_equal_cutoff_diff =
        std::max(max_equal_cutoff_diff, std::abs(same.small - same.large));
  }
  auto within_error = [&](double s, double s2, double reference) {
    const double mean = s / num_iterations;
    const double error = std::sqrt((s2 / num_iterations - mean * mean) /
                                   (num_iterations - 1));
    std::printf("      mean %.8f  error %.1e  reference %.8f\n", mean, error,
                reference);
    return std::abs(mean - reference) < 4.0 * error && error < 0.1 * -reference;
  };
  Expect(within_error(sum, sum_squares, exact),
         "stochastic PT averages to the deterministic value");
  Expect(within_error(sum_large, sum_squares_large, exact_large),
         "stochastic PT with the large cutoff averages to its value");
  Expect(max_equal_cutoff_diff == 0.0,
         "equal cutoffs give identical small and large estimates");
  Expect(max_batch_diff < 1e-12,
         "a walker gives the same estimate in one and in three batches");
}

// The second-order formula against exact diagonalization. In a model with
// the coupling between the variational space P and the rest Q scaled by
// lambda, and no coupling inside Q, the exact energy is
// E0 + lambda^2 E2 + O(lambda^4). This fixes which of H(a,i) and its complex
// conjugate enters the numerator, independently of the PT code.
void TestSecondOrderLimit(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 300, 2, 31), 0);
  const int dim = static_cast<int>(dets.size());
  const int num_p = 60;

  Eigen::MatrixXcd full = Eigen::MatrixXcd::Zero(dim, dim);
  for (int i = 0; i < dim; ++i) {
    full(i, i) = DiagonalEnergy(dets[i], ints);
    for (int j = 0; j < i; ++j) {
      if (j < num_p) full(i, j) = Hij(dets[i], dets[j], ints);
      full(j, i) = std::conj(full(i, j));
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> p_solver(
      full.topLeftCorner(num_p, num_p));
  const double e0 = p_solver.eigenvalues()[0];
  const Eigen::VectorXcd c = p_solver.eigenvectors().col(0);

  // (E(lambda) - E0) / lambda^2 for the exact state that continues the
  // P-space ground state.
  auto scaled_shift = [&](double lambda) {
    Eigen::MatrixXcd model = full;
    model.bottomLeftCorner(dim - num_p, num_p) *= lambda;
    model.topRightCorner(num_p, dim - num_p) *= lambda;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(model);
    int best = 0;
    double best_overlap = 0.0;
    for (int k = 0; k < dim; ++k) {
      const double overlap =
          std::abs(c.dot(solver.eigenvectors().col(k).head(num_p)));
      if (overlap > best_overlap) {
        best_overlap = overlap;
        best = k;
      }
    }
    return (solver.eigenvalues()[best] - e0) / (lambda * lambda);
  };
  // Richardson extrapolation removes the lambda^2 term of the scaled shift.
  const double lambda = 0.04;
  const double exact =
      (4.0 * scaled_shift(lambda / 2) - scaled_shift(lambda)) / 3.0;

  double second_order = 0.0;
  double second_order_conjugated = 0.0;
  for (int a = num_p; a < dim; ++a) {
    Complex numerator(0.0);
    Complex conjugated(0.0);
    for (int i = 0; i < num_p; ++i) {
      const Complex element = Hij(dets[a], dets[i], ints);
      numerator += element * c[i];
      conjugated += std::conj(element) * c[i];
    }
    const double denominator = e0 - DiagonalEnergy(dets[a], ints);
    second_order += std::norm(numerator) / denominator;
    second_order_conjugated += std::norm(conjugated) / denominator;
  }
  std::printf("      exact %.8f  sum H(a,i) c_i: %.8f  sum conj(H(a,i)) c_i: "
              "%.8f\n",
              exact, second_order, second_order_conjugated);
  Expect(std::abs(exact - second_order) < 1e-4 * std::abs(exact),
         "the PT numerator sum_i H(a,i) c_i reproduces exact diagonalization "
         "to second order");
}

// Applies the annihilator of spinor p to a determinant: returns false if p is
// empty, otherwise removes it and multiplies `sign` by the fermionic phase,
// (-1) to the number of occupied spinors below p. The determinant stands for
// the creators of its spinors applied to the vacuum in increasing order.
bool Annihilate(int p, Det<1>* det, double* sign) {
  if (!det->Occupied(p)) return false;
  if (__builtin_popcountll(det->words[0] & ((1ULL << p) - 1)) & 1) {
    *sign = -*sign;
  }
  det->Clear(p);
  return true;
}

bool Create(int p, Det<1>* det, double* sign) {
  if (det->Occupied(p)) return false;
  if (__builtin_popcountll(det->words[0] & ((1ULL << p) - 1)) & 1) {
    *sign = -*sign;
  }
  det->Set(p);
  return true;
}

// The density matrices against their definition: every operator string is
// applied to every determinant one operator at a time, with none of the
// logic of rdm.h. Unlike the energy, this tells a density matrix from its
// complex conjugate and from its transpose.
void TestDensityMatricesFromDefinition(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 120, 2, 41), 0);
  std::map<Det<1>, size_t> index;
  for (size_t d = 0; d < dets.size(); ++d) index[dets[d]] = d;

  SparseH<Complex> h;
  std::vector<double> diagonal(dets.size());
  for (size_t d = 0; d < dets.size(); ++d) {
    diagonal[d] = DiagonalEnergy(dets[d], ints);
  }
  h.AppendDiagonal(diagonal);
  h.AppendBlock(BuildBlock(MakeSpan(dets), 0, ints, ConnectOptions()));
  DavidsonOptions dense;
  dense.dense_dim = 1 << 20;
  std::vector<std::vector<Complex>> states(
      2, std::vector<Complex>(dets.size(), 0.0));
  Davidson(h, dense, &states);
  double max_imag = 0.0;
  for (const Complex& c : states[0]) {
    max_imag = std::max(max_imag, std::abs(c.imag()));
  }

  // <bra| p^+ q |ket> and <bra| p^+ q^+ s r |ket> by direct application.
  auto reference = [&](const std::vector<Complex>& bra,
                       const std::vector<Complex>& ket, bool with_two) {
    DensityMatrices<Complex> dm;
    dm.num_orbs = n;
    dm.one.assign(static_cast<size_t>(n) * n, Complex(0.0));
    if (with_two) {
      dm.two.assign(static_cast<size_t>(n) * n * n * n, Complex(0.0));
    }
    for (size_t j = 0; j < dets.size(); ++j) {
      for (int q = 0; q < n; ++q) {
        Det<1> one_removed = dets[j];
        double sign_q = 1.0;
        if (!Annihilate(q, &one_removed, &sign_q)) continue;
        for (int p = 0; p < n; ++p) {
          Det<1> det = one_removed;
          double sign = sign_q;
          if (!Create(p, &det, &sign)) continue;
          const auto found = index.find(det);
          if (found == index.end()) continue;
          dm.One(p, q) += sign * std::conj(bra[found->second]) * ket[j];
        }
        if (!with_two) continue;
        // p^+ q2^+ s r with r = q: remove r, then s, add q2, then p.
        const int r = q;
        for (int s = 0; s < n; ++s) {
          Det<1> two_removed = one_removed;
          double sign_s = sign_q;
          if (!Annihilate(s, &two_removed, &sign_s)) continue;
          for (int q2 = 0; q2 < n; ++q2) {
            Det<1> one_added = two_removed;
            double sign_q2 = sign_s;
            if (!Create(q2, &one_added, &sign_q2)) continue;
            for (int p = 0; p < n; ++p) {
              Det<1> det = one_added;
              double sign = sign_q2;
              if (!Create(p, &det, &sign)) continue;
              const auto found = index.find(det);
              if (found == index.end()) continue;
              dm.Two(p, q2, r, s) +=
                  sign * std::conj(bra[found->second]) * ket[j];
            }
          }
        }
      }
    }
    return dm;
  };
  auto max_difference = [](const std::vector<Complex>& a,
                           const std::vector<Complex>& b) {
    double diff = 0.0;
    for (size_t k = 0; k < a.size(); ++k) {
      diff = std::max(diff, std::abs(a[k] - b[k]));
    }
    return diff;
  };

  const DensityMatrices<Complex> state = ComputeDensityMatrices(
      MakeSpan(dets), h, n, MakeSpan(states[0]), MakeSpan(states[0]), true);
  const DensityMatrices<Complex> state_ref =
      reference(states[0], states[0], true);
  Expect(max_imag > 1e-3, "the test wave function is genuinely complex");
  Expect(max_difference(state.one, state_ref.one) < 1e-12,
         "dm1 equals <p^+ q> applied operator by operator");
  Expect(max_difference(state.two, state_ref.two) < 1e-12,
         "dm2 equals <p^+ q^+ s r> applied operator by operator");
  // The conjugate must not pass: the test has to be able to tell them apart.
  std::vector<Complex> conjugated = state_ref.one;
  for (Complex& value : conjugated) value = std::conj(value);
  Expect(max_difference(state.one, conjugated) > 1e-6,
         "dm1 differs from its complex conjugate in this test");

  const DensityMatrices<Complex> transition = ComputeDensityMatrices(
      MakeSpan(dets), h, n, MakeSpan(states[0]), MakeSpan(states[1]), false);
  const DensityMatrices<Complex> transition_ref =
      reference(states[0], states[1], false);
  Expect(max_difference(transition.one, transition_ref.one) < 1e-12,
         "the transition dm1 equals <0| p^+ q |1> applied operator by operator");
}

// The real part of a Hamiltonian, which is again a Hamiltonian, as numbers
// of type S.
template <typename S>
Integrals<S> RealPart(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  Integrals<S> out(n);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      out.SetOneBody(p, q, S(ints.OneBody(p, q).real()));
      for (int s = 1; s < n && p < q; ++s) {
        for (int r = 0; r < s; ++r) {
          out.SetAntisymOrdered(p, q, r, s,
                                S(ints.AntisymOrdered(p, q, r, s).real()));
        }
      }
    }
  }
  out.set_num_elec(ints.num_elec());
  out.set_core_energy(ints.core_energy());
  out.Finalize();
  return out;
}

// What a small calculation produces, in a form that does not depend on the
// number type.
struct ScalarResults {
  size_t num_elements = 0;
  std::vector<double> energies;
  double pt = 0.0;
  StochasticEstimate sample;
  std::vector<Complex> dm1;
  std::vector<Complex> dm2;
  std::vector<Complex> transition;
  std::vector<double> real_vector;  // the lowest state, if it is real
};

// `pt_vector`, if given, replaces the lowest state in the perturbative part:
// the stochastic sample compares only between runs that draw from exactly
// the same coefficients.
template <typename S>
ScalarResults SmallCalculation(const Integrals<S>& ints,
                               const std::vector<double>* pt_vector) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 500, 2, 41), 0);
  std::vector<Det<1>> sorted_dets = dets;
  std::sort(sorted_dets.begin(), sorted_dets.end());

  ConnectOptions connect;
  connect.threshold = -1.0;
  SparseH<S> h;
  std::vector<double> diagonal(dets.size());
  for (size_t d = 0; d < dets.size(); ++d) {
    diagonal[d] = DiagonalEnergy(dets[d], ints);
  }
  h.AppendDiagonal(diagonal);
  h.AppendBlock(BuildBlock(MakeSpan(dets), 0, ints, connect));

  ScalarResults results;
  results.num_elements = h.num_elements();
  std::vector<std::vector<S>> vectors =
      GuessVectors(MakeSpan(dets), h, ints, 2);
  DavidsonOptions options;
  options.tol = 1e-10;
  results.energies = Davidson<S>(h, options, &vectors).energies;
  // The sign of a real eigenvector is arbitrary; fix it for the transition
  // density, which is linear in each state.
  for (std::vector<S>& v : vectors) {
    size_t largest = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      if (std::abs(v[i]) > std::abs(v[largest])) largest = i;
    }
    const S phase = std::abs(v[largest]) / v[largest];
    for (S& c : v) c *= phase;
  }

  for (const S& c : vectors[0]) results.real_vector.push_back(Real(c));

  const std::vector<S> pt_state =
      pt_vector == nullptr
          ? vectors[0]
          : std::vector<S>(pt_vector->begin(), pt_vector->end());
  const HeatBath heat_bath(ints, 1e-12);
  const LabelFilter filter(std::vector<int>(n, 0), 0);
  const Comm serial;
  const PtProblem<1, LabelFilter, S> problem{
      dets,   sorted_dets, pt_state, results.energies[0], ints, heat_bath,
      filter, 1,           serial};
  results.pt = PtDeterministic(problem, 1e-5);
  const DeterminantSampler sampler(MakeSpan(pt_state), serial);
  std::mt19937_64 rng(11);
  WalkerWorkspace<1, S> workspace;
  results.sample =
      PtWalkerSample(problem, 1e-5, 1e-3, 50, sampler, &rng, &workspace);

  const DensityMatrices<S> dm = ComputeDensityMatrices(
      MakeSpan(dets), h, n, MakeSpan(vectors[0]), MakeSpan(vectors[0]), true);
  const DensityMatrices<S> transition = ComputeDensityMatrices(
      MakeSpan(dets), h, n, MakeSpan(vectors[0]), MakeSpan(vectors[1]), false);
  results.dm1.assign(dm.one.begin(), dm.one.end());
  results.dm2.assign(dm.two.begin(), dm.two.end());
  results.transition.assign(transition.one.begin(), transition.one.end());
  return results;
}

// Integrals held in symmetry blocks against the same integrals held in one
// block.
void TestBlockedIntegrals(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  std::vector<int> axis(n), parity(n);
  for (int p = 0; p < n; ++p) {
    axis[p] = (p * p + 3 * p) % 5 - 2;
    parity[p] = (p / 3) % 2;
  }
  const std::vector<LabelSet> sets = {{axis, 4}, {parity, 2}};

  // Dropping what the labels forbid makes a Hamiltonian with the symmetry.
  Integrals<Complex> blocked(n, sets);
  Integrals<Complex> plain(n);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) blocked.SetOneBody(p, q, ints.OneBody(p, q));
  }
  for (int q = 1; q < n; ++q) {
    for (int p = 0; p < q; ++p) {
      for (int s = 1; s < n; ++s) {
        for (int r = 0; r < s; ++r) {
          blocked.SetAntisymOrdered(p, q, r, s,
                                    ints.AntisymOrdered(p, q, r, s));
        }
      }
    }
  }
  bool same_elements = true;
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) plain.SetOneBody(p, q, blocked.OneBody(p, q));
  }
  for (int q = 1; q < n; ++q) {
    for (int p = 0; p < q; ++p) {
      for (int s = 1; s < n; ++s) {
        for (int r = 0; r < s; ++r) {
          plain.SetAntisymOrdered(p, q, r, s,
                                  blocked.AntisymOrdered(p, q, r, s));
          const Complex original = ints.AntisymOrdered(p, q, r, s);
          const Complex kept = blocked.AntisymOrdered(p, q, r, s);
          same_elements =
              same_elements && (kept == original || kept == Complex(0.0));
        }
      }
    }
  }
  for (Integrals<Complex>* each : {&blocked, &plain}) {
    each->set_num_elec(ints.num_elec());
    each->Finalize();
  }
  std::printf(
      "      %zu numbers in blocks, %zu in one block; largest dropped "
      "integral %.2e\n",
      blocked.num_stored(), plain.num_stored(), blocked.max_forbidden());
  Expect(same_elements && 3 * blocked.num_stored() < plain.num_stored() &&
             blocked.max_forbidden() > 1e-3 && plain.max_forbidden() == 0.0,
         "blocked integrals keep the allowed elements in a third of the space");

  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 300, 2, 9), 0);
  double max_diff = 0.0;
  double max_element = 0.0;
  for (size_t i = 0; i < dets.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(DiagonalEnergy(dets[i], blocked) -
                                           DiagonalEnergy(dets[i], plain)));
    for (size_t j = 0; j < i; ++j) {
      const Complex element = Hij(dets[i], dets[j], blocked);
      max_element = std::max(max_element, std::abs(element));
      max_diff =
          std::max(max_diff, std::abs(element - Hij(dets[i], dets[j], plain)));
    }
  }
  Expect(max_element > 1e-3 && max_diff == 0.0,
         "blocked integrals give the same Hamiltonian elements");

  const HeatBath heat_bath_blocked(blocked, 1e-12);
  const HeatBath heat_bath_plain(plain, 1e-12);
  bool same_tables =
      heat_bath_blocked.num_entries() == heat_bath_plain.num_entries();
  for (int j = 1; j < n && same_tables; ++j) {
    for (int i = 0; i < j && same_tables; ++i) {
      std::vector<HeatBath::Entry> from_blocked, from_plain;
      heat_bath_blocked.ForEachDouble(
          i, j, 0.0, [&](int a, int b, double magnitude) {
            from_blocked.push_back({magnitude, static_cast<uint16_t>(a),
                                    static_cast<uint16_t>(b)});
          });
      heat_bath_plain.ForEachDouble(
          i, j, 0.0, [&](int a, int b, double magnitude) {
            from_plain.push_back({magnitude, static_cast<uint16_t>(a),
                                  static_cast<uint16_t>(b)});
          });
      same_tables = from_blocked.size() == from_plain.size();
      for (size_t k = 0; same_tables && k < from_plain.size(); ++k) {
        same_tables = from_blocked[k].magnitude == from_plain[k].magnitude &&
                      from_blocked[k].a == from_plain[k].a &&
                      from_blocked[k].b == from_plain[k].b;
      }
    }
  }
  Expect(heat_bath_blocked.num_entries() > 1000 && same_tables,
         "blocked integrals give the same heat-bath tables");

  // Selection: the filter that trusts the tables for the double excitations
  // against the filter that checks every one.
  std::vector<Det<1>> start;
  const std::vector<int> block_labels = DeterminantLabels(dets[0], sets);
  for (const Det<1>& det : dets) {
    if (DeterminantLabels(det, sets) == block_labels) start.push_back(det);
  }
  std::vector<Det<1>> sorted_start = start;
  std::sort(sorted_start.begin(), sorted_start.end());
  const std::vector<double> weights(start.size(), 0.3);
  LabelFilter trusting(sets);
  trusting.set_doubles_conserved(true);
  const LabelFilter checking(sets);
  const std::vector<Det<1>> selected_trusting = SelectNewDeterminants(
      MakeSpan(start), MakeSpan(sorted_start), MakeSpan(weights), 1e-3, blocked,
      heat_bath_blocked, trusting);
  const std::vector<Det<1>> selected_checking = SelectNewDeterminants(
      MakeSpan(start), MakeSpan(sorted_start), MakeSpan(weights), 1e-3, plain,
      heat_bath_plain, checking);
  bool one_block = true;
  for (const Det<1>& det : selected_trusting) {
    one_block = one_block && DeterminantLabels(det, sets) == block_labels;
  }
  Expect(start.size() >= 3 && selected_trusting.size() > 100 && one_block &&
             selected_trusting == selected_checking,
         "selection from blocked integrals stays in the block without "
         "checking the double excitations");
  std::printf("      %zu starting determinants, %zu selected\n", start.size(),
              selected_trusting.size());
}

// Integrals over spatial orbitals against the same Hamiltonian written out
// over spin orbitals.
void TestSpatialIntegrals() {
  const int n = 7;  // spatial orbitals
  const int num_elec = 6;
  std::mt19937 rng(29);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  // Random integrals with the eightfold symmetry: (pq|rs) = sum_l L(l,pq)
  // L(l,rs) with symmetric L(l), and a symmetric h.
  std::vector<std::vector<double>> factors(2 * n, std::vector<double>(n * n));
  for (std::vector<double>& factor : factors) {
    for (int p = 0; p < n; ++p) {
      for (int q = 0; q <= p; ++q) {
        factor[p * n + q] = factor[q * n + p] = 0.4 * uniform(rng);
      }
    }
  }
  std::vector<double> h(n * n);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q <= p; ++q) h[p * n + q] = h[q * n + p] = uniform(rng);
  }
  auto eri = [&](int p, int q, int r, int s) {
    double value = 0.0;
    for (const std::vector<double>& factor : factors) {
      value += factor[p * n + q] * factor[r * n + s];
    }
    return value;
  };

  Integrals<double> spatial = Integrals<double>::Spatial(n, {});
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      spatial.SetSpatialOneBody(p, q, h[p * n + q]);
      for (int r = 0; r < n; ++r) {
        for (int s = 0; s < n; ++s) {
          spatial.SetSpatialTwoBody(p, q, r, s, eri(p, q, r, s));
        }
      }
    }
  }
  // Spin orbital 2P + sigma; <pq||rs> = (pr|qs) - (ps|qr) with the spins of
  // each charge distribution equal.
  const int m = 2 * n;
  Integrals<double> spinor(m, spatial.symmetry());
  auto chemist = [&](int p, int q, int r, int s) {
    return (p % 2 == q % 2 && r % 2 == s % 2) ? eri(p / 2, q / 2, r / 2, s / 2)
                                              : 0.0;
  };
  for (int p = 0; p < m; ++p) {
    for (int q = 0; q < m; ++q) {
      spinor.SetOneBody(p, q, p % 2 == q % 2 ? h[(p / 2) * n + q / 2] : 0.0);
    }
  }
  for (int q = 1; q < m; ++q) {
    for (int p = 0; p < q; ++p) {
      for (int s = 1; s < m; ++s) {
        for (int r = 0; r < s; ++r) {
          spinor.SetAntisymOrdered(p, q, r, s,
                                   chemist(p, r, q, s) - chemist(p, s, q, r));
        }
      }
    }
  }
  for (Integrals<double>* each : {&spatial, &spinor}) {
    each->set_num_elec(num_elec);
    each->Finalize();
  }
  std::printf(
      "      %zu numbers over spatial orbitals, %zu over spin "
      "orbitals in spin blocks\n",
      spatial.num_stored(), spinor.num_stored());
  Expect(spinor.max_forbidden() == 0.0 &&
             4 * spatial.num_stored() < spinor.num_stored(),
         "spatial integrals take less than a quarter of the spin-orbital ones");

  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(m, num_elec, 400, 3, 31), 0);
  double max_diff = 0.0;
  double max_element = 0.0;
  for (size_t i = 0; i < dets.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(DiagonalEnergy(dets[i], spatial) -
                                           DiagonalEnergy(dets[i], spinor)));
    for (size_t j = 0; j < i; ++j) {
      const double element = Hij(dets[i], dets[j], spatial);
      max_element = std::max(max_element, std::abs(element));
      max_diff =
          std::max(max_diff, std::abs(element - Hij(dets[i], dets[j], spinor)));
    }
  }
  Expect(max_element > 1e-2 && max_diff < 1e-13,
         "spatial integrals give the Hamiltonian elements of the spin-orbital "
         "ones");

  // The heat-bath lists hold the same excitations (ties may be ordered
  // differently).
  const HeatBath heat_bath_spatial(spatial, 1e-12);
  const HeatBath heat_bath_spinor(spinor, 1e-12);
  bool same_lists = true;
  size_t num_listed = 0;
  for (int j = 1; j < m; ++j) {
    for (int i = 0; i < j; ++i) {
      std::map<std::pair<int, int>, double> from_spatial, from_spinor;
      bool decreasing = true;
      double previous = 1e300;
      heat_bath_spatial.ForEachDouble(
          i, j, 0.0, [&](int a, int b, double magnitude) {
            from_spatial[{a, b}] = magnitude;
            decreasing = decreasing && magnitude <= previous;
            previous = magnitude;
          });
      heat_bath_spinor.ForEachDouble(i, j, 0.0,
                                     [&](int a, int b, double magnitude) {
                                       from_spinor[{a, b}] = magnitude;
                                     });
      num_listed += from_spatial.size();
      same_lists =
          same_lists && decreasing && from_spatial.size() == from_spinor.size();
      for (const auto& entry : from_spinor) {
        const auto found = from_spatial.find(entry.first);
        same_lists = same_lists && found != from_spatial.end() &&
                     std::abs(found->second - entry.second) < 1e-13;
      }
    }
  }
  std::printf(
      "      %zu excitations listed from %zu spatial table entries "
      "(%zu spin-orbital ones)\n",
      num_listed, heat_bath_spatial.num_entries(),
      heat_bath_spinor.num_entries());
  Expect(num_listed > 1000 && same_lists &&
             heat_bath_spatial.num_entries() < heat_bath_spinor.num_entries(),
         "spatial heat-bath tables list the excitations of the spin-orbital "
         "ones from fewer entries");

  // Selection from the closed-shell determinant.
  std::vector<int> closed_shell(num_elec);
  for (int i = 0; i < num_elec; ++i) closed_shell[i] = i;
  const std::vector<Det<1>> start = ToDets<1>({closed_shell}, 0);
  const std::vector<double> weights(1, 1.0);
  LabelFilter filter(spatial.symmetry());
  filter.set_doubles_conserved(true);
  const std::vector<Det<1>> selected_spatial =
      SelectNewDeterminants(MakeSpan(start), MakeSpan(start), MakeSpan(weights),
                            1e-3, spatial, heat_bath_spatial, filter);
  const std::vector<Det<1>> selected_spinor =
      SelectNewDeterminants(MakeSpan(start), MakeSpan(start), MakeSpan(weights),
                            1e-3, spinor, heat_bath_spinor, filter);
  bool spin_conserved = true;
  for (const Det<1>& det : selected_spatial) {
    spin_conserved =
        spin_conserved && DeterminantLabels(det, spatial.symmetry())[0] == 0;
  }
  Expect(selected_spatial.size() > 50 && spin_conserved &&
             selected_spatial == selected_spinor,
         "selection from spatial integrals equals that from spin-orbital "
         "ones and conserves the spin projection");
}

// A real Hamiltonian treated with real numbers and with complex numbers.
void TestRealScalar(const Integrals<Complex>& ints) {
  const ScalarResults real = SmallCalculation(RealPart<double>(ints), nullptr);
  const ScalarResults complex =
      SmallCalculation(RealPart<Complex>(ints), &real.real_vector);
  auto max_difference = [](const std::vector<Complex>& a,
                           const std::vector<Complex>& b) {
    double diff = a.size() == b.size() ? 0.0 : 1.0;
    for (size_t k = 0; k < a.size() && k < b.size(); ++k) {
      diff = std::max(diff, std::abs(a[k] - b[k]));
    }
    return diff;
  };
  Expect(real.num_elements > 1000 && real.num_elements == complex.num_elements,
         "real numbers: the Hamiltonian has the same elements");
  Expect(std::abs(real.energies[0] - complex.energies[0]) < 1e-10 &&
             std::abs(real.energies[1] - complex.energies[1]) < 1e-10,
         "real numbers: the same two lowest energies as with complex numbers");
  Expect(real.pt < -1e-5 && std::abs(real.pt - complex.pt) < 1e-10,
         "real numbers: the same deterministic PT energy");
  Expect(std::abs(real.sample.small - complex.sample.small) < 1e-10 &&
             std::abs(real.sample.large - complex.sample.large) < 1e-10,
         "real numbers: the same stochastic PT sample from the same draws");
  std::printf(
      "      E0 %.10f  PT %.10f  sample %.10f (difference %.1e) and "
      "%.10f (difference %.1e)\n",
      real.energies[0], real.pt, real.sample.small,
      real.sample.small - complex.sample.small, real.sample.large,
      real.sample.large - complex.sample.large);
  // Davidson converged the vectors to about 1e-10 only.
  Expect(max_difference(real.dm1, complex.dm1) < 1e-8 &&
             max_difference(real.dm2, complex.dm2) < 1e-8,
         "real numbers: the same one- and two-body density matrices");
  Expect(max_difference(real.transition, complex.transition) < 1e-7,
         "real numbers: the same transition density matrix");
  std::printf(
      "      largest differences: dm1 %.1e  dm2 %.1e  transition "
      "%.1e\n",
      max_difference(real.dm1, complex.dm1),
      max_difference(real.dm2, complex.dm2),
      max_difference(real.transition, complex.transition));
}

}  // namespace
}  // namespace zshci

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s FCIDUMP\n", argv[0]);
    return EXIT_FAILURE;
  }
  try {
    const zshci::Integrals<zshci::Complex> ints =
        zshci::ReadFcidump<zshci::Complex>(argv[1]);
    if (ints.num_orbs() > 32 || ints.num_orbs() < ints.num_elec() + 2) {
      throw std::runtime_error("the test needs NELEC + 2 <= NORB <= 32");
    }
    zshci::TestCountBetween();
    zshci::TestHamiltonian(ints);
    zshci::TestDensityMatricesFromDefinition(ints);
    zshci::TestPerturbation(ints);
    zshci::TestSecondOrderLimit(ints);
    zshci::TestRealScalar(ints);
    zshci::TestBlockedIntegrals(ints);
    zshci::TestSpatialIntegrals();
    const int n = ints.num_orbs();
    const std::vector<zshci::LabelSet> no_symmetry = {
        {std::vector<int>(n, 0), 0}};
    zshci::TestSelection(ints, no_symmetry, {},
                         "heat-bath selection equals direct evaluation");
    // RAS-like bounds: few holes in the lowest spinors, few electrons in the
    // highest, and an overlapping third set.
    const int nelec = ints.num_elec();
    std::vector<zshci::OccupationRestriction> restrictions(3);
    for (int p = 0; p < nelec / 2; ++p) restrictions[0].orbitals.push_back(p);
    restrictions[0].min_elec = nelec / 2 - 1;
    restrictions[0].max_elec = nelec / 2;
    for (int p = nelec + 4; p < n; ++p) restrictions[1].orbitals.push_back(p);
    restrictions[1].min_elec = 0;
    restrictions[1].max_elec = 1;
    for (int p = nelec - 2; p < nelec + 2; ++p) {
      restrictions[2].orbitals.push_back(p);
    }
    restrictions[2].min_elec = 1;
    restrictions[2].max_elec = 3;
    zshci::TestSelection(
        ints, no_symmetry, restrictions,
        "selection with occupation restrictions equals direct evaluation");
    std::vector<int> labels(n);
    for (int p = 0; p < n; ++p) labels[p] = (p * p + 3 * p) % 5 - 2;
    zshci::TestSelection(
        ints, {{labels, 0}}, {},
        "selection with additive labels equals direct evaluation");
    zshci::TestSelection(
        ints, {{labels, 4}}, {},
        "selection with labels modulo 4 equals direct evaluation");
    // Two independent sets, as for an axis together with inversion.
    std::vector<int> parity(n);
    for (int p = 0; p < n; ++p) parity[p] = (p / 3) % 2;
    zshci::TestSelection(
        ints, {{labels, 4}, {parity, 2}}, {},
        "selection with two label sets equals direct evaluation");
    zshci::TestSelection(
        ints, {{labels, 4}, {parity, 2}}, restrictions,
        "selection with two label sets and occupation restrictions equals "
        "direct evaluation");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "zshci_test: error: %s\n", e.what());
    return EXIT_FAILURE;
  }
  std::printf("%d failure(s)\n", zshci::g_num_failures);
  return zshci::g_num_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
