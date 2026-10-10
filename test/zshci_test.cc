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
Integrals ShiftIntegrals(const Integrals& ints, int shift) {
  const int n = ints.num_orbs();
  Integrals shifted(n + shift);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      shifted.SetOneBody(p + shift, q + shift, ints.OneBody(p, q));
      for (int r = 0; r < n; ++r) {
        for (int s = 0; s < n; ++s) {
          shifted.SetTwoBody(p + shift, q + shift, r + shift, s + shift,
                             ints.TwoBody(p, q, r, s));
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

void TestHamiltonian(const Integrals& ints) {
  const int n = ints.num_orbs();
  const std::vector<std::vector<int>> occupations =
      RandomOccupations(n, ints.num_elec(), 500, 3, 11);
  ConnectOptions keep_all;
  ConnectOptions batched;
  batched.num_batches = 4;
  ConnectOptions thresholded;
  thresholded.threshold = 1e-6;

  const std::vector<Det<1>> dets1 = ToDets<1>(occupations, 0);
  const CsrBlock block1 = BuildBlock(MakeSpan(dets1), 0, ints, keep_all);
  Expect(SameBlock(block1, BuildBlockBruteForce(MakeSpan(dets1), 0, ints, keep_all), 0),
         "bucket search equals brute force (one word)");
  Expect(SameBlock(block1, BuildBlock(MakeSpan(dets1), 0, ints, batched), 0),
         "bucket search in four batches gives the same block");
  Expect(SameBlock(BuildBlock(MakeSpan(dets1), 0, ints, thresholded),
                   BuildBlockBruteForce(MakeSpan(dets1), 0, ints, thresholded), 0),
         "bucket search equals brute force with a threshold");

  // Blocks built for a growing space add up to the block of the full space.
  {
    const size_t first_new = 300;
    const std::vector<Det<1>> old_dets(dets1.begin(),
                                       dets1.begin() + first_new);
    const CsrBlock old_block = BuildBlock(MakeSpan(old_dets), 0, ints, keep_all);
    const CsrBlock new_block = BuildBlock(MakeSpan(dets1), first_new, ints, keep_all);
    bool ok = old_block.num_elements() + new_block.num_elements() ==
              block1.num_elements();
    for (size_t k = 0; k < new_block.col.size() && ok; ++k) {
      ok = new_block.col[k] >= first_new;
    }
    Expect(ok, "incremental blocks partition the elements of the full block");
  }

  // Shifting all spinors across a word boundary must not change anything.
  const int shift = 66 - n;
  const Integrals shifted = ShiftIntegrals(ints, shift);
  const std::vector<Det<2>> dets2 = ToDets<2>(occupations, shift);
  const CsrBlock block2 = BuildBlock(MakeSpan(dets2), 0, shifted, keep_all);
  Expect(SameBlock(block1, block2, 1e-14),
         "two-word determinants give the same Hamiltonian block");
  bool same_diagonal = true;
  for (size_t d = 0; d < dets1.size(); ++d) {
    same_diagonal = same_diagonal &&
                    std::abs(DiagonalEnergy(dets1[d], ints) -
                             DiagonalEnergy(dets2[d], shifted)) < 1e-11;
  }
  Expect(same_diagonal, "two-word determinants give the same diagonal");

  // Solver and density matrices.
  SparseH h;
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

  const DensityMatrices dm =
      ComputeDensityMatrices(MakeSpan(dets1), h, n, exact[0], exact[0], true);
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
  const DensityMatrices t01 =
      ComputeDensityMatrices(MakeSpan(dets1), h, n, exact[0], exact[1], false);
  const DensityMatrices t10 =
      ComputeDensityMatrices(MakeSpan(dets1), h, n, exact[1], exact[0], false);
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
void TestSelection(const Integrals& ints, const std::vector<int>& labels,
                   int mod,
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
  const RestrictedFilter filter(LabelFilter(labels, mod), restrictions, n);
  const std::vector<Det<1>> selected =
      SelectNewDeterminants(MakeSpan(dets), MakeSpan(sorted_dets),
                            MakeSpan(weights), epsilon, ints, heat_bath, filter);

  auto conserved = [&](int change) {
    return mod == 0 ? change == 0 : change % mod == 0;
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
        if (!conserved(labels[a] - labels[i])) continue;
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
            if (!conserved(labels[virt[u]] + labels[virt[v]] -
                           labels[occ[x]] - labels[occ[y]])) {
              continue;
            }
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
void TestPerturbation(const Integrals& ints) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 60, 2, 23), 0);
  std::vector<Det<1>> sorted_dets = dets;
  std::sort(sorted_dets.begin(), sorted_dets.end());

  // The lowest eigenvector in this small space.
  SparseH h;
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
  PtProblem<1, LabelFilter> problem{dets, sorted_dets, c,      energy, ints,
                                    heat_bath, filter, 1, serial};
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
  const DeterminantSampler sampler(c, serial);
  std::mt19937_64 rng(5);
  WalkerWorkspace<1> scratch;
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
void TestSecondOrderLimit(const Integrals& ints) {
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

}  // namespace
}  // namespace zshci

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s FCIDUMP\n", argv[0]);
    return EXIT_FAILURE;
  }
  try {
    const zshci::Integrals ints = zshci::ReadFcidump(argv[1]);
    if (ints.num_orbs() > 32 || ints.num_orbs() < ints.num_elec() + 2) {
      throw std::runtime_error("the test needs NELEC + 2 <= NORB <= 32");
    }
    zshci::TestCountBetween();
    zshci::TestHamiltonian(ints);
    zshci::TestPerturbation(ints);
    zshci::TestSecondOrderLimit(ints);
    const int n = ints.num_orbs();
    zshci::TestSelection(ints, std::vector<int>(n, 0), 0, {},
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
        ints, std::vector<int>(n, 0), 0, restrictions,
        "selection with occupation restrictions equals direct evaluation");
    std::vector<int> labels(n);
    for (int p = 0; p < n; ++p) labels[p] = (p * p + 3 * p) % 5 - 2;
    zshci::TestSelection(
        ints, labels, 0, {},
        "selection with additive labels equals direct evaluation");
    zshci::TestSelection(
        ints, labels, 4, {},
        "selection with labels modulo 4 equals direct evaluation");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "zshci_test: error: %s\n", e.what());
    return EXIT_FAILURE;
  }
  std::printf("%d failure(s)\n", zshci::g_num_failures);
  return zshci::g_num_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
