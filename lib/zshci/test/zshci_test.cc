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
#include "det_set.h"
#include "direct_h.h"
#include "excite.h"
#include "fci_davidson.h"
#include "fci_dist.h"
#include "gas_space.h"
#include "hamiltonian.h"
#include "heat_bath.h"
#include "highfive/H5File.hpp"
#include "integrals.h"
#include "output.h"
#include "pt.h"
#include "rdm.h"
#include "select.h"
#include "sparse_h.h"
#include "spin_flip.h"

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

void TestBalanceWithin() {
  std::mt19937_64 rng(11);
  bool ok = true;
  for (int trial = 0; trial < 300 && ok; ++trial) {
    const int count = 1 + rng() % 60, n = 1 + rng() % 8;
    std::vector<double> weight(count), size(count);
    double total = 0.0, biggest = 0.0;
    for (int i = 0; i < count; ++i) {
      weight[i] = 1.0 + rng() % 100;
      size[i] = 1.0 + rng() % 50;
      total += size[i];
      biggest = std::max(biggest, size[i]);
    }
    const double cap = std::max(biggest, 1.3 * total / n);
    std::vector<int> b;
    try {
      b = BalanceWithin(weight, size, n, cap);
    } catch (const std::runtime_error&) {
      continue;  // a cap the greedy cut cannot meet
    }
    ok = static_cast<int>(b.size()) == n + 1 && b[0] == 0 && b[n] == count;
    for (int r = 0; r < n && ok; ++r) {
      double s = 0.0;
      for (int i = b[r]; i < b[r + 1]; ++i) s += size[i];
      ok = b[r] <= b[r + 1] && s <= cap;
    }
  }
  bool threw = false;
  try {
    BalanceWithin({1.0, 1.0}, {10.0, 10.0}, 1, 15.0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(ok && threw,
         "cut of the vector by work keeps every part within its memory");
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
  // The string method; the group method has tests of its own below.
  ConnectOptions keep_all;
  keep_all.method = ConnectMethod::kStrings;
  ConnectOptions batched = keep_all;
  batched.num_batches = 4;
  ConnectOptions thresholded = keep_all;
  thresholded.threshold = 1e-6;

  const std::vector<Det<1>> dets1 = ToDets<1>(occupations, 0);
  const CsrBlock<Complex> block1 =
      BuildBlock(MakeSpan(dets1), 0, ints, keep_all);
  Expect(SameBlock(block1,
                   BuildBlockBruteForce(MakeSpan(dets1), 0, ints, keep_all), 0),
         "bucket search equals brute force (one word)");
  Expect(SameBlock(block1, BuildBlock(MakeSpan(dets1), 0, ints, batched), 0),
         "bucket search in four batches gives the same block");
  Expect(
      SameBlock(BuildBlock(MakeSpan(dets1), 0, ints, thresholded),
                BuildBlockBruteForce(MakeSpan(dets1), 0, ints, thresholded), 0),
      "bucket search equals brute force with a threshold");

  // The segment method, for several numbers of segments (fewer than five
  // means a single cover, i.e. all pairs in one bucket).
  bool segments_ok = true;
  for (int num_segments : {1, 3, 5, 6, 8, 11, 16}) {
    ConnectOptions segmented;
    segmented.method = ConnectMethod::kSegments;
    segmented.num_segments = num_segments;
    segments_ok =
        segments_ok &&
        SameBlock(block1, BuildBlock(MakeSpan(dets1), 0, ints, segmented), 0);
  }
  Expect(segments_ok, "segment search equals the string search");
  {
    ConnectOptions segmented;
    segmented.method = ConnectMethod::kSegments;
    segmented.num_segments = 7;
    segmented.threshold = 1e-6;
    Expect(SameBlock(
               BuildBlock(MakeSpan(dets1), 0, ints, segmented),
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
    for (size_t r = 0; r < new_block.row.size() && ok; ++r) {
      ok = new_block.row[r] >= first_new;
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
    same_diagonal =
        same_diagonal && std::abs(DiagonalEnergy(dets1[d], ints) -
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
  Expect(
      std::abs(EnergyFromDensityMatrices(dm, ints) - dense.energies[0]) < 1e-10,
      "energy from the density matrices equals the eigenvalue");
  Complex trace(0.0);
  double max_antihermitian = 0.0;
  for (int p = 0; p < n; ++p) {
    trace += dm.One(p, p);
    for (int q = 0; q < n; ++q) {
      max_antihermitian = std::max(
          max_antihermitian, std::abs(dm.One(p, q) - std::conj(dm.One(q, p))));
    }
  }
  Expect(std::abs(trace - Complex(ints.num_elec())) < 1e-10,
         "the one-body density matrix traces to the electron number");
  Expect(max_antihermitian < 1e-12, "the one-body density matrix is Hermitian");

  // Partial trace of D2 over the second index pair gives (N - 1) D1.
  double max_trace_diff = 0.0;
  for (int p = 0; p < n; ++p) {
    for (int r = 0; r < n; ++r) {
      Complex sum(0.0);
      for (int q = 0; q < n; ++q) sum += dm.Two(p, q, r, q);
      max_trace_diff =
          std::max(max_trace_diff,
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

// The few-vector solver of the select and diag modes against the classic one
// and the dense eigenvalues: several roots, a space that grows (vectors
// padded with zeros), determinants out of the space, checkpoint and restart.
void TestFewVectorSolver(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 500, 3, 11), 0);
  ConnectOptions connect;
  connect.method = ConnectMethod::kStrings;
  auto make_h = [&](size_t count) {
    SparseH<Complex> h;
    std::vector<double> diagonal(count);
    for (size_t d = 0; d < count; ++d) {
      diagonal[d] = DiagonalEnergy(dets[d], ints);
    }
    h.AppendDiagonal(diagonal);
    const std::vector<Det<1>> part(dets.begin(), dets.begin() + count);
    h.AppendBlock(BuildBlock(MakeSpan(part), 0, ints, connect));
    return h;
  };
  const SparseH<Complex> h = make_h(dets.size());
  const size_t dim = h.dim();
  const int max_roots = 3;
  DavidsonOptions dense_options;
  dense_options.dense_dim = 1 << 20;
  dense_options.few_vectors = false;
  std::vector<std::vector<Complex>> exact(max_roots,
                                          std::vector<Complex>(dim, 0.0));
  const DavidsonResult dense = Davidson(h, dense_options, &exact);
  // Start vectors: unit vectors on the lowest diagonal elements.
  auto guess = [&](const SparseH<Complex>& op, int roots) {
    std::vector<size_t> order(op.dim());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return op.diagonal()[a] < op.diagonal()[b];
    });
    std::vector<std::vector<Complex>> v(roots,
                                        std::vector<Complex>(op.dim(), 0.0));
    for (int k = 0; k < roots; ++k) v[k][order[k]] = 1.0;
    return v;
  };
  auto error = [&](const DavidsonResult& r, int roots) {
    double e = r.converged ? 0.0 : 1.0;
    for (int k = 0; k < roots; ++k) {
      e = std::max(e, std::abs(r.energies[k] - dense.energies[k]));
    }
    return e;
  };

  for (int roots : {1, 3}) {
    DavidsonOptions few;
    few.dense_dim = 0;
    few.tol = 1e-8;
    DavidsonOptions classic = few;
    classic.few_vectors = false;
    std::vector<std::vector<Complex>> a = guess(h, roots), b = a;
    const DavidsonResult ra = SolveDistributed<Complex>(h, few, Comm(), &a);
    const DavidsonResult rb = SolveDistributed<Complex>(h, classic, Comm(), &b);
    Expect(error(ra, roots) < 1e-10 && error(rb, roots) < 1e-10,
           "few-vector Davidson (3 per root) and the classic one give the "
           "dense eigenvalues, " +
               std::to_string(roots) + " root(s)");
    std::printf(
        "      %d root(s): %d products with 3 vectors per root, %d "
        "with 20\n",
        roots, ra.num_products, rb.num_products);

    // The space grows: the eigenvectors of the first 300 determinants,
    // padded with zeros, start the solver in all 500.
    const SparseH<Complex> small = make_h(300);
    std::vector<std::vector<Complex>> v = guess(small, roots);
    SolveDistributed<Complex>(small, few, Comm(), &v);
    for (std::vector<Complex>& x : v) x.resize(dim, 0.0);
    const DavidsonResult grown = SolveDistributed<Complex>(h, few, Comm(), &v);
    double residual = 0.0;
    std::vector<Complex> hv(dim);
    for (int k = 0; k < roots; ++k) {
      h.Multiply(v[k].data(), hv.data());
      for (size_t i = 0; i < dim; ++i) {
        residual =
            std::max(residual, std::abs(hv[i] - grown.energies[k] * v[k][i]));
      }
    }
    Expect(error(grown, roots) < 1e-10 && residual < 1e-7,
           "few-vector Davidson continues from vectors padded with zeros "
           "after the space grew, " +
               std::to_string(roots) + " root(s)");

    // Determinants out of the space.
    std::vector<uint8_t> active(dim, 1);
    for (size_t i = 0; i < dim; i += 7) active[i] = 0;
    std::vector<std::vector<Complex>> c = guess(h, roots), d = c;
    for (int k = 0; k < roots; ++k) {
      for (size_t i = 0; i < dim; ++i) {
        if (!active[i]) c[k][i] = d[k][i] = 0.0;
      }
      c[k][(k + 1) * 11 % dim] = d[k][(k + 1) * 11 % dim] = 0.5;
    }
    const DavidsonResult rc =
        SolveDistributed<Complex>(h, few, Comm(), &c, &active);
    const DavidsonResult rd =
        SolveDistributed<Complex>(h, classic, Comm(), &d, &active);
    double diff = rc.converged && rd.converged ? 0.0 : 1.0;
    for (int k = 0; k < roots; ++k) {
      diff = std::max(diff, std::abs(rc.energies[k] - rd.energies[k]));
    }
    Expect(diff < 1e-10,
           "few-vector and classic Davidson agree in a masked "
           "space, " +
               std::to_string(roots) + " root(s)");
  }

  // Stopped with checkpoints, then restarted from them; a checkpoint of
  // another list is passed over.
  const std::string path =
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
      "/zshci_test_sci.ckpt";
  std::remove(path.c_str());
  DavidsonOptions options;
  options.dense_dim = 0;
  options.tol = 1e-8;
  options.checkpoint = path;
  options.checkpoint_every = 1;
  options.fingerprint = 12345;
  options.max_iterations = 4;
  std::vector<std::vector<Complex>> v = guess(h, 2);
  const DavidsonResult first =
      SolveDistributed<Complex>(h, options, Comm(), &v);
  options.max_iterations = 200;
  options.restart = true;
  v = guess(h, 2);
  const DavidsonResult second =
      SolveDistributed<Complex>(h, options, Comm(), &v);
  options.fingerprint = 54321;
  v = guess(h, 2);
  const DavidsonResult other =
      SolveDistributed<Complex>(h, options, Comm(), &v);
  Expect(!first.converged && second.restarted && error(second, 2) < 1e-10 &&
             second.num_iterations < other.num_iterations && !other.restarted &&
             error(other, 2) < 1e-10,
         "few-vector Davidson restarts from its checkpoint and passes over "
         "one of another list");
  std::printf("      %d + %d iterations with the checkpoint, %d without\n",
              first.num_iterations, second.num_iterations,
              other.num_iterations);
  std::remove(path.c_str());
}

// The energies of a select-mode run in the directory `prefix`.
std::vector<double> RunSelectCase(const Integrals<Complex>& ints,
                                  const std::string& prefix,
                                  const std::string& extra,
                                  const std::string& solver, int max_iter) {
  const std::string text =
      "{\"mode\": \"select\", \"integrals\": \"unused\", \"nroots\": 2, "
      "\"prefix\": \"" +
      prefix +
      "\", \"dets\": [[0,1,2,3,4,5,6,7,8,9,10,"
      "11]], \"select\": {\"schedule\": [[0, 1e-2], [1, 3e-3]], "
      "\"max_iter\": " +
      std::to_string(max_iter) + ", \"energy_tol\": 1e-12" + extra +
      "}, \"output_level\": 0, "
      "\"davidson\": {\"tol\": 1e-9, \"checkpoint_every\": 1, "
      "\"solver\": \"" +
      solver + "\"}}";
  Input input = ParseInput(text);
  RunSelect<1, Complex>(input, ints, Comm());
  std::vector<double> energies(2, 0.0);
  std::FILE* file = std::fopen((prefix + "/shci.e").c_str(), "rb");
  if (file != nullptr) {
    if (std::fread(energies.data(), sizeof(double), 2, file) != 2) {
      energies.assign(2, 0.0);
    }
    std::fclose(file);
  }
  return energies;
}

// A whole selection with the few-vector solver against the classic one, and
// one stopped after two iterations and resumed from its checkpoint.
void TestSelectSolvers(const Integrals<Complex>& ints) {
  const std::string base =
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
      "/zshci_test_select";
  std::system(("rm -rf " + base + " && mkdir -p " + base + "/a " + base +
               "/b " + base + "/c")
                  .c_str());
  const std::vector<double> few =
      RunSelectCase(ints, base + "/a", "", "few", 5);
  const std::vector<double> classic =
      RunSelectCase(ints, base + "/b", "", "classic", 5);
  RunSelectCase(ints, base + "/c", ", \"checkpoint\": true", "few", 2);
  int iteration = -1, solved = 0;
  const bool state =
      ReadSelectState(base + "/c/select.h5", &iteration, &solved);
  const std::vector<double> resumed = RunSelectCase(
      ints, base + "/c", ", \"checkpoint\": true, \"resume\": true", "few", 5);
  double diff = 0.0, resume_diff = 0.0;
  for (int k = 0; k < 2; ++k) {
    diff = std::max(diff, std::abs(few[k] - classic[k]));
    resume_diff = std::max(resume_diff, std::abs(few[k] - resumed[k]));
  }
  Expect(few[0] != 0.0 && diff < 1e-9,
         "select mode: the few-vector and the classic Davidson give the same "
         "energies (2 roots)");
  Expect(state && iteration == 1 && solved == 1 && resume_diff < 1e-9,
         "select mode resumed from its checkpoint after two iterations "
         "ends with the energies of an uninterrupted run");
  std::printf("      differences %.1e (solvers), %.1e (resumed)\n", diff,
              resume_diff);
  std::system(("rm -rf " + base).c_str());
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
  const std::vector<Det<1>> selected = SelectNewDeterminants(
      MakeSpan(dets), MakeSpan(sorted_dets), MakeSpan(weights), epsilon, ints,
      heat_bath, filter);

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
    const StochasticEstimate same =
        PtWalkerSample(problem, epsilon, epsilon, 20, sampler, &rng, &scratch);
    max_equal_cutoff_diff =
        std::max(max_equal_cutoff_diff, std::abs(same.small - same.large));
  }
  auto within_error = [&](double s, double s2, double reference) {
    const double mean = s / num_iterations;
    const double error =
        std::sqrt((s2 / num_iterations - mean * mean) / (num_iterations - 1));
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
  std::printf(
      "      exact %.8f  sum H(a,i) c_i: %.8f  sum conj(H(a,i)) c_i: "
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
  Expect(
      max_difference(transition.one, transition_ref.one) < 1e-12,
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

  // Spin-summed density matrices against the spin-orbital ones summed
  // afterwards, for two different vectors (a transition density).
  std::vector<Det<1>> space = start;
  space.insert(space.end(), selected_spatial.begin(), selected_spatial.end());
  ConnectOptions keep_all;
  keep_all.threshold = -1.0;
  keep_all.method = ConnectMethod::kGroups;
  SparseH<double> hamiltonian;
  hamiltonian.AppendDiagonal(std::vector<double>(space.size(), 0.0));
  hamiltonian.AppendBlock(BuildBlock(MakeSpan(space), 0, spatial, keep_all));
  std::vector<double> bra(space.size()), ket(space.size());
  for (size_t i = 0; i < space.size(); ++i) {
    bra[i] = uniform(rng);
    ket[i] = uniform(rng);
  }
  const DensityMatrices<double> full = ComputeDensityMatrices(
      MakeSpan(space), hamiltonian, m, MakeSpan(bra), MakeSpan(ket), true);
  const DensityMatrices<double> summed =
      ComputeDensityMatrices(MakeSpan(space), hamiltonian, m, MakeSpan(bra),
                             MakeSpan(ket), true, Comm(), /*spin_summed=*/true);
  double max_one = 0.0;
  double max_two = 0.0;
  double largest = 0.0;
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      double one = 0.0;
      for (int spin = 0; spin < 2; ++spin) {
        one += full.One(2 * p + spin, 2 * q + spin);
      }
      max_one = std::max(max_one, std::abs(one - summed.One(p, q)));
      for (int r = 0; r < n; ++r) {
        for (int s = 0; s < n; ++s) {
          double two = 0.0;
          for (int spin = 0; spin < 2; ++spin) {
            for (int other = 0; other < 2; ++other) {
              two += full.Two(2 * p + spin, 2 * q + other, 2 * r + spin,
                              2 * s + other);
            }
          }
          largest = std::max(largest, std::abs(two));
          max_two = std::max(max_two, std::abs(two - summed.Two(p, q, r, s)));
        }
      }
    }
  }
  Expect(summed.num_orbs == n && largest > 0.1 && max_one < 1e-12 &&
             max_two < 1e-12,
         "spin-summed density matrices equal the spin-orbital ones summed "
         "over the spins");
}

// The alpha/beta string tables against a comparison of all pairs.
template <int W>
void TestSpinStringConnections(int num_spatial, int shift,
                               const std::string& what) {
  const int num_alpha = 4;
  const int num_beta = 3;
  std::mt19937 rng(43);
  // Random integrals; only the pattern of connections is tested here, and
  // every connected pair gets an element since no threshold is set.
  Integrals<double> ints = Integrals<double>::Spatial(num_spatial + shift, {});
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (int p = 0; p < num_spatial + shift; ++p) {
    for (int q = 0; q <= p; ++q) {
      ints.SetSpatialOneBody(p, q, uniform(rng));
      ints.SetSpatialOneBody(q, p, ints.OneBody(2 * p, 2 * q));
      for (int r = 0; r <= p; ++r) {
        for (int s = 0; s <= r; ++s) {
          ints.SetSpatialTwoBody(p, q, r, s, uniform(rng));
        }
      }
    }
  }
  ints.Finalize();

  // Determinants with fixed numbers of alpha and beta electrons, strings
  // drawn from small pools so that many determinants share one.
  auto random_string = [&](int count, int spin) {
    std::set<int> orbitals;
    while (static_cast<int>(orbitals.size()) < count) {
      orbitals.insert(rng() % num_spatial);
    }
    Det<W> string;
    for (int p : orbitals) string.Set(2 * (p + shift) + spin);
    return string;
  };
  std::vector<Det<W>> alpha_pool, beta_pool;
  for (int k = 0; k < 40; ++k)
    alpha_pool.push_back(random_string(num_alpha, 0));
  for (int k = 0; k < 30; ++k) beta_pool.push_back(random_string(num_beta, 1));
  std::set<Det<W>> seen;
  std::vector<Det<W>> dets;
  while (dets.size() < 500) {
    const Det<W>& a = alpha_pool[rng() % alpha_pool.size()];
    const Det<W>& b = beta_pool[rng() % beta_pool.size()];
    Det<W> det;
    for (int k = 0; k < W; ++k) det.words[k] = a.words[k] | b.words[k];
    if (seen.insert(det).second) dets.push_back(det);
  }

  ConnectOptions spin;
  spin.threshold = -1.0;
  spin.method = ConnectMethod::kGroups;
  ConnectOptions reference;
  reference.threshold = -1.0;
  // With stored relation lists (by keys and by sub-groups) and with related
  // strings composed from the byte table.
  for (StringRelations relations :
       {StringRelations::kKeys, StringRelations::kSubgroups,
        StringRelations::kBytes}) {
    spin.relations = relations;
    bool same = true;
    size_t num_elements = 0;
    for (size_t first_new : {size_t{0}, size_t{137}, dets.size() - 1}) {
      const CsrBlock<double> block =
          BuildBlock(MakeSpan(dets), first_new, ints, spin);
      const CsrBlock<double> brute =
          BuildBlockBruteForce(MakeSpan(dets), first_new, ints, reference);
      same = same && SameBlock(block, brute, 0.0);
      num_elements += block.num_elements();
    }
    Expect(num_elements > 5000 && same,
           what + (relations == StringRelations::kKeys        ? " (keys)"
                   : relations == StringRelations::kSubgroups ? " (sub-groups)"
                                                              : " (bytes)"));
  }
}

// The connections of every determinant and of every task from tables with
// the given way of finding related strings, as sorted lists of pairs
// (larger index first).
template <int W>
void TableConnections(const internal::GroupStringTables<W>& tables,
                      size_t num_dets,
                      std::vector<std::pair<DetIndex, DetIndex>>* below,
                      std::vector<std::pair<DetIndex, DetIndex>>* both,
                      std::vector<std::pair<DetIndex, DetIndex>>* by_tasks) {
  below->clear();
  both->clear();
  by_tasks->clear();
  uint64_t tested = 0;
  for (size_t j = 0; j < num_dets; ++j) {
    const DetIndex row = static_cast<DetIndex>(j);
    tables.ForEachConnected(
        j, [&](DetIndex i) { below->emplace_back(row, i); }, &tested);
    tables.ForEachConnected(
        j, [&](DetIndex i) { both->emplace_back(row, i); }, &tested, true);
  }
  for (size_t t = 0; t < tables.num_tasks(); ++t) {
    tables.ForEachConnectedOfTask(
        t,
        [&](DetIndex j, DetIndex i) {
          by_tasks->emplace_back(std::max(j, i), std::min(j, i));
        },
        &tested);
  }
  std::sort(below->begin(), below->end());
  std::sort(both->begin(), both->end());
  std::sort(by_tasks->begin(), by_tasks->end());
}

// Related strings composed from the byte table (connect_bytes.h) against
// the stored relation lists and against a comparison of all pairs: the
// connections of every determinant, below it and on both sides, and those
// found string by string. `labels`, if not empty, restricts the
// determinants to those whose labels sum to that of the first one.
template <int W>
void TestByteConnections(int num_orbs, int num_elec, int shift, int count,
                         int max_rank, const std::vector<int>& labels,
                         const std::string& what) {
  std::vector<std::vector<int>> occupations =
      RandomOccupations(num_orbs, num_elec, count, max_rank, 97);
  if (!labels.empty()) {
    auto sum = [&](const std::vector<int>& occ) {
      int total = 0;
      for (int p : occ) total += labels[p];
      return total;
    };
    const int wanted = sum(occupations[0]);
    std::vector<std::vector<int>> kept;
    for (const std::vector<int>& occ : occupations) {
      if (sum(occ) == wanted) kept.push_back(occ);
    }
    occupations = kept;
  }
  const std::vector<Det<W>> dets = ToDets<W>(occupations, shift);
  const size_t n = dets.size();
  const int num_spinors = num_orbs + shift;
  std::vector<std::pair<DetIndex, DetIndex>> brute, brute_both;
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      const int differing = dets[i].NumDifferentBits(dets[j]);
      if (differing != 2 && differing != 4) continue;
      brute_both.emplace_back(j, i);
      if (i < j) brute.emplace_back(j, i);
    }
  }
  // The usual groups; groups of unequal size whose last bytes are partial;
  // the lower and the upper half; a group of one spinor; and one group
  // only, which leaves the other without any spinor.
  std::vector<ConnectOptions> groupings(5);
  for (int p = 0; p < num_spinors; ++p) {
    if (p % 7 == 0) groupings[1].group_a.push_back(p);
    if (p < shift + num_orbs / 2) groupings[2].group_a.push_back(p);
    groupings[4].group_a.push_back(p);
  }
  groupings[3].group_a = {shift};
  const Comm serial;
  bool same = true;
  bool same_as_lists = true;
  bool same_stepwise = true;
  size_t num_strings = 0;
  for (const ConnectOptions& grouping : groupings) {
    const Det<W> mask = GroupMask<W>(grouping, num_spinors);
    std::vector<std::pair<DetIndex, DetIndex>> below, both, by_tasks;
    internal::GroupStringTables<W> composed(mask, serial,
                                            StringRelations::kBytes);
    composed.set_block_search(true);
    composed.Extend(MakeSpan(dets), 0);
    num_strings += composed.num_strings(0) + composed.num_strings(1);
    TableConnections(composed, n, &below, &both, &by_tasks);
    same = same && below == brute && both == brute_both && by_tasks == brute &&
           composed.num_links() == 0;
    std::vector<std::pair<DetIndex, DetIndex>> list_below, list_both,
        list_tasks;
    internal::GroupStringTables<W> listed(mask, serial,
                                          StringRelations::kSubgroups);
    listed.set_block_search(true);
    listed.Extend(MakeSpan(dets), 0);
    TableConnections(listed, n, &list_below, &list_both, &list_tasks);
    same_as_lists = same_as_lists && below == list_below && both == list_both &&
                    by_tasks == list_tasks &&
                    composed.has_transfers() == listed.has_transfers() &&
                    composed.num_strings(0) == listed.num_strings(0) &&
                    composed.num_strings(1) == listed.num_strings(1);
    // Determinants added in steps, the first step a single one.
    internal::GroupStringTables<W> stepwise(mask, serial,
                                            StringRelations::kBytes);
    stepwise.set_block_search(true);
    size_t previous = 0;
    for (size_t size : {size_t{1}, size_t{40}, size_t{41}, n / 2, n}) {
      stepwise.Extend(Span<Det<W>>(dets.data(), size), previous);
      previous = size;
    }
    std::vector<std::pair<DetIndex, DetIndex>> step_below, step_both,
        step_tasks;
    TableConnections(stepwise, n, &step_below, &step_both, &step_tasks);
    same_stepwise = same_stepwise && step_below == brute &&
                    step_both == brute_both && step_tasks == brute;
  }
  Expect(brute.size() > 2000 && same, what);
  Expect(same_as_lists, "  and equal those from the stored relation lists");
  Expect(same_stepwise,
         "  and the same after determinants were added in "
         "steps");
  std::printf(
      "      %zu determinants, %zu connected pairs, %zu strings over "
      "five groupings\n",
      n, brute.size(), num_strings);
}

// The group string tables for spinors, where electrons move between the
// groups, against a comparison of all pairs.
void TestGroupConnections(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<std::vector<int>> occupations =
      RandomOccupations(n, ints.num_elec(), 700, 3, 61);
  ConnectOptions reference;
  reference.threshold = -1.0;
  std::vector<std::vector<int>> groups(4);  // the first: even spinors
  for (int p = 0; p < n / 2; ++p) groups[1].push_back(p);
  for (int p = 0; p < n; ++p) {
    if ((p * 7 + p / 3) % 3 == 0) groups[2].push_back(p);
  }
  groups[3] = {0};
  bool same = true;
  size_t num_elements = 0;
  for (const std::vector<int>& group_a : groups) {
    ConnectOptions grouped = reference;
    grouped.method = ConnectMethod::kGroups;
    grouped.group_a = group_a;
    const std::vector<Det<1>> dets = ToDets<1>(occupations, 0);
    for (size_t first_new : {size_t{0}, size_t{251}, dets.size() - 1}) {
      const CsrBlock<Complex> block =
          BuildBlock(MakeSpan(dets), first_new, ints, grouped);
      same = same && SameBlock(block,
                               BuildBlockBruteForce(MakeSpan(dets), first_new,
                                                    ints, reference),
                               0.0);
      num_elements += block.num_elements();
    }
  }
  Expect(num_elements > 20000 && same,
         "group string tables find the connected pairs of spinor "
         "determinants for four choices of the groups");

  // The same blocks with the related strings composed from the byte table,
  // with the values and with the columns only.
  {
    const std::vector<Det<1>> dets = ToDets<1>(occupations, 0);
    const Comm serial;
    bool same_blocks = true;
    bool same_columns = true;
    size_t num_composed = 0;
    for (const std::vector<int>& group_a : groups) {
      ConnectOptions composed = reference;
      composed.method = ConnectMethod::kGroups;
      composed.relations = StringRelations::kBytes;
      composed.group_a = group_a;
      for (double threshold : {-1.0, 1e-3}) {
        composed.threshold = threshold;
        ConnectOptions brute = reference;
        brute.threshold = threshold;
        for (size_t first_new : {size_t{0}, size_t{251}, dets.size() - 1}) {
          const CsrBlock<Complex> wanted =
              BuildBlockBruteForce(MakeSpan(dets), first_new, ints, brute);
          const CsrBlock<Complex> block =
              BuildBlock(MakeSpan(dets), first_new, ints, composed);
          same_blocks = same_blocks && SameBlock(block, wanted, 0.0);
          num_composed += block.num_elements();
          // "connections": the columns without the values.
          const internal::GroupStringTables<1> tables(
              MakeSpan(dets), GroupMask<1>(composed, n), serial,
              StringRelations::kBytes);
          std::map<DetIndex, std::vector<DetIndex>> columns, wanted_columns;
          for (const CsrBlock<Complex>& part : internal::BuildGroupBlocks(
                   MakeSpan(dets), first_new, ints, threshold, tables, nullptr,
                   /*store_values=*/false)) {
            same_columns = same_columns && !part.has_values();
            for (size_t r = 0; r < part.num_stored_rows(); ++r) {
              columns[part.row[r]].assign(
                  part.col.begin() + part.row_ptr[r],
                  part.col.begin() + part.row_ptr[r + 1]);
            }
          }
          for (size_t r = 0; r < wanted.num_stored_rows(); ++r) {
            wanted_columns[wanted.row[r]].assign(
                wanted.col.begin() + wanted.row_ptr[r],
                wanted.col.begin() + wanted.row_ptr[r + 1]);
          }
          same_columns = same_columns && columns == wanted_columns;
        }
      }
    }
    Expect(num_composed > 20000 && same_blocks,
           "the same blocks with related strings composed from the byte "
           "table, with and without a threshold");
    Expect(same_columns, "  and the same columns when no values are stored");
  }

  const int shift = 66 - n;
  const Integrals<Complex> shifted = ShiftIntegrals(ints, shift);
  const std::vector<Det<2>> dets2 = ToDets<2>(occupations, shift);
  ConnectOptions grouped = reference;
  grouped.method = ConnectMethod::kGroups;
  Expect(SameBlock(BuildBlock(MakeSpan(dets2), 0, shifted, grouped),
                   BuildBlockBruteForce(MakeSpan(dets2), 0, shifted, reference),
                   0.0),
         "group string tables work across a word boundary");
  {
    ConnectOptions composed = grouped;
    composed.relations = StringRelations::kBytes;
    Expect(
        SameBlock(BuildBlock(MakeSpan(dets2), 0, shifted, composed),
                  BuildBlockBruteForce(MakeSpan(dets2), 0, shifted, reference),
                  0.0),
        "and with related strings composed from the byte table");
  }
  grouped.threshold = 1e-3;
  ConnectOptions thresholded = reference;
  thresholded.threshold = 1e-3;
  Expect(SameBlock(
             BuildBlock(MakeSpan(dets2), 300, shifted, grouped),
             BuildBlockBruteForce(MakeSpan(dets2), 300, shifted, thresholded),
             0.0),
         "group string tables with a threshold equal brute force");

  // Related strings found through sub-groups against those found by keys:
  // the same connections for every determinant, for tables made at once
  // and step by step, with the usual groups and with groups of 5 and 27
  // spinors, whose sub-groups are of unequal size.
  {
    const std::vector<Det<1>> dets = ToDets<1>(occupations, 0);
    const Comm serial;
    std::vector<Det<1>> masks = {GroupMask<1>(ConnectOptions(), n)};
    ConnectOptions uneven;
    for (int p = 0; p < n; ++p) {
      if (p % 7 == 0) uneven.group_a.push_back(p);
    }
    masks.push_back(GroupMask<1>(uneven, n));
    bool same_connections = true;
    size_t num_connections = 0;
    for (const Det<1>& mask : masks) {
      const internal::GroupStringTables<1> by_keys(MakeSpan(dets), mask, serial,
                                                   StringRelations::kKeys);
      const internal::GroupStringTables<1> at_once(MakeSpan(dets), mask, serial,
                                                   StringRelations::kSubgroups);
      internal::GroupStringTables<1> stepwise(mask, serial,
                                              StringRelations::kSubgroups);
      size_t previous = 0;
      for (size_t size :
           {size_t{1}, size_t{40}, size_t{41}, size_t{300}, dets.size()}) {
        stepwise.Extend(Span<Det<1>>(dets.data(), size), previous);
        previous = size;
      }
      same_connections = same_connections &&
                         by_keys.num_links() == at_once.num_links() &&
                         by_keys.num_links() == stepwise.num_links();
      for (size_t j = 0; j < dets.size() && same_connections; ++j) {
        std::vector<DetIndex> a, b, c;
        uint64_t tested = 0;
        by_keys.ForEachConnected(
            j, [&](DetIndex i) { a.push_back(i); }, &tested, true);
        at_once.ForEachConnected(
            j, [&](DetIndex i) { b.push_back(i); }, &tested, true);
        stepwise.ForEachConnected(
            j, [&](DetIndex i) { c.push_back(i); }, &tested, true);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        std::sort(c.begin(), c.end());
        same_connections = a == b && a == c;
        num_connections += a.size();
      }
    }
    Expect(same_connections && num_connections > 1000,
           "related strings from sub-groups equal those from keys, also "
           "for groups of unequal size and tables extended step by step");
    std::printf("      %d spinors; %zu connections compared\n", n,
                num_connections);
  }

  // Tables extended step by step against tables made at once: the same
  // connections for every determinant. The first step holds one
  // determinant, so the lists for electrons moving between the groups only
  // come into being later.
  {
    const std::vector<Det<1>> dets = ToDets<1>(occupations, 0);
    const Comm serial;
    const Det<1> mask = GroupMask<1>(ConnectOptions(), n);
    const internal::GroupStringTables<1> at_once(MakeSpan(dets), mask, serial);
    internal::GroupStringTables<1> stepwise(mask, serial);
    size_t previous = 0;
    for (size_t size :
         {size_t{1}, size_t{40}, size_t{41}, size_t{300}, dets.size()}) {
      stepwise.Extend(Span<Det<1>>(dets.data(), size), previous);
      previous = size;
    }
    bool same_connections = true;
    size_t num_connections = 0;
    for (size_t j = 0; j < dets.size() && same_connections; ++j) {
      std::vector<DetIndex> a, b;
      uint64_t tested = 0;
      at_once.ForEachConnected(j, [&](DetIndex i) { a.push_back(i); }, &tested);
      stepwise.ForEachConnected(
          j, [&](DetIndex i) { b.push_back(i); }, &tested);
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      same_connections = a == b;
      num_connections += a.size();
    }
    Expect(num_connections > 5000 && same_connections &&
               at_once.num_links() == stepwise.num_links() &&
               at_once.has_transfers(),
           "group string tables extended step by step equal those made at "
           "once");
  }

  // The Hamiltonian applied from the tables against the stored one, and
  // density matrices from the tables against those from the stored pairs.
  const std::vector<Det<1>> dets = ToDets<1>(occupations, 0);
  const Comm serial;
  for (double threshold : {-1.0, 1e-3}) {
    ConnectOptions options;
    options.threshold = threshold;
    options.method = ConnectMethod::kGroups;
    SparseH<Complex> stored;
    std::vector<double> diagonal(dets.size());
    for (size_t d = 0; d < dets.size(); ++d) {
      diagonal[d] = DiagonalEnergy(dets[d], ints);
    }
    stored.AppendDiagonal(diagonal);
    stored.AppendBlocks(BuildBlocks(MakeSpan(dets), 0, ints, options));
    // Both ways of multiplying the stored matrix against the serial one.
    if (threshold < 0.0) {
      std::vector<Complex> x(dets.size()), y(dets.size()), y_ref(dets.size());
      for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Complex(std::cos(0.7 * i + 0.3), std::sin(2.1 * i));
      }
      stored.MultiplyReference(x.data(), y_ref.data());
      double max_diff[2] = {0.0, 0.0};
      double largest = 0.0;
      int which = 0;
      for (SparseH<Complex>::Product product :
           {SparseH<Complex>::Product::kOneSweep,
            SparseH<Complex>::Product::kTwoSweeps}) {
        stored.set_product(product);
        stored.Multiply(x.data(), y.data());
        for (size_t i = 0; i < x.size(); ++i) {
          max_diff[which] =
              std::max(max_diff[which], std::abs(y[i] - y_ref[i]));
          largest = std::max(largest, std::abs(y_ref[i]));
        }
        ++which;
      }
      stored.set_product(SparseH<Complex>::Product::kAutomatic);
      Expect(largest > 1.0 && max_diff[0] < 1e-12 * largest &&
                 max_diff[1] < 1e-12 * largest,
             "the one-sweep and the two-sweep product of the stored matrix "
             "equal the serial one");

      // The same with the values dropped and worked out in every product.
      struct Source : ElementSource<Complex> {
        const std::vector<Det<1>>* dets;
        const Integrals<Complex>* ints;
        void Fill(DetIndex row, const DetIndex* cols, size_t count,
                  Complex* values) const override {
          for (size_t k = 0; k < count; ++k) {
            values[k] = Hij((*dets)[row], (*dets)[cols[k]], *ints);
          }
        }
      };
      Source source;
      source.dets = &dets;
      source.ints = &ints;
      const size_t bytes_with_values = stored.element_bytes();
      stored.set_element_source(&source);
      stored.DropValues();
      double worst = 0.0;
      for (SparseH<Complex>::Product product :
           {SparseH<Complex>::Product::kOneSweep,
            SparseH<Complex>::Product::kTwoSweeps}) {
        stored.set_product(product);
        stored.Multiply(x.data(), y.data());
        for (size_t i = 0; i < x.size(); ++i) {
          worst = std::max(worst, std::abs(y[i] - y_ref[i]));
        }
      }
      stored.MultiplyReference(x.data(), y.data());
      for (size_t i = 0; i < x.size(); ++i) {
        worst = std::max(worst, std::abs(y[i] - y_ref[i]));
      }
      stored.set_product(SparseH<Complex>::Product::kAutomatic);
      Expect(stored.num_values() == 0 &&
                 5 * stored.element_bytes() == bytes_with_values &&
                 worst < 1e-12 * largest,
             "products with the columns only, the values worked out each "
             "time, equal those of the stored matrix");
      // The matrix with its values again, for what follows.
      stored = SparseH<Complex>();
      stored.AppendDiagonal(diagonal);
      stored.AppendBlocks(BuildBlocks(MakeSpan(dets), 0, ints, options));
    }
    const internal::GroupStringTables<1> tables(
        MakeSpan(dets), GroupMask<1>(options, n), serial);
    const std::vector<double> no_diagonal;
    const std::vector<const std::vector<double>*> diagonals = {&diagonal,
                                                               &no_diagonal};
    double max_diff = 0.0;
    double norm = 0.0;
    for (const std::vector<double>* given : diagonals) {
      // With the diagonal given: per-thread vectors; without: atomic
      // additions into y.
      const DirectH<1, Complex> direct(
          MakeSpan(dets), ints, tables, threshold, *given,
          given->empty() ? 0 : DirectH<1, Complex>::kDefaultScratchBytes);
      std::vector<Complex> x(dets.size()), y(dets.size()), y_ref(dets.size());
      for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Complex(std::sin(0.3 * i + 0.2), std::cos(1.7 * i));
      }
      direct.Apply(x.data(), y.data());
      stored.Multiply(x.data(), y_ref.data());
      for (size_t i = 0; i < x.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(y[i] - y_ref[i]));
        norm = std::max(norm, std::abs(y_ref[i]));
      }
    }
    Expect(norm > 1.0 && max_diff < 1e-12 * norm,
           threshold < 0.0
               ? "the Hamiltonian applied from the tables equals the stored one"
               : "the same with a threshold on the elements");
    {
      // The same with the connections found string by string, for the
      // usual groups and for groups of 5 and 27 spinors.
      ConnectOptions uneven = options;
      for (int p = 0; p < n; ++p) {
        if (p % 7 == 0) uneven.group_a.push_back(p);
      }
      double worst = 0.0, largest = 0.0;
      size_t num_tasks = 0;
      for (const ConnectOptions& grouping : {options, uneven}) {
        internal::GroupStringTables<1> blocked(GroupMask<1>(grouping, n),
                                               serial);
        blocked.set_block_search(true);
        blocked.Extend(MakeSpan(dets), 0);
        num_tasks += blocked.num_tasks();
        for (size_t budget :
             {size_t{0}, DirectH<1, Complex>::kDefaultScratchBytes}) {
          const DirectH<1, Complex> direct(MakeSpan(dets), ints, blocked,
                                           threshold, diagonal, budget);
          std::vector<Complex> x(dets.size()), y(dets.size()),
              y_ref(dets.size());
          for (size_t i = 0; i < x.size(); ++i) {
            x[i] = Complex(std::sin(0.3 * i + 0.2), std::cos(1.7 * i));
          }
          direct.Apply(x.data(), y.data());
          stored.Multiply(x.data(), y_ref.data());
          for (size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, std::abs(y[i] - y_ref[i]));
            largest = std::max(largest, std::abs(y_ref[i]));
          }
        }
      }
      Expect(num_tasks > 0 && largest > 1.0 && worst < 1e-12 * largest,
             threshold < 0.0
                 ? "the Hamiltonian applied string by string equals the "
                   "stored one"
                 : "the same with a threshold on the elements");
    }
    {
      // Both again with the related strings composed from the byte table.
      ConnectOptions uneven = options;
      for (int p = 0; p < n; ++p) {
        if (p % 7 == 0) uneven.group_a.push_back(p);
      }
      double worst = 0.0, largest = 0.0;
      size_t num_tasks = 0;
      std::vector<Complex> x(dets.size()), y(dets.size()), y_ref(dets.size());
      for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Complex(std::sin(0.3 * i + 0.2), std::cos(1.7 * i));
      }
      stored.Multiply(x.data(), y_ref.data());
      for (const ConnectOptions& grouping : {options, uneven}) {
        for (bool by_blocks : {false, true}) {
          internal::GroupStringTables<1> composed(
              GroupMask<1>(grouping, n), serial, StringRelations::kBytes);
          composed.set_block_search(by_blocks);
          composed.Extend(MakeSpan(dets), 0);
          num_tasks += composed.num_tasks();
          for (size_t budget :
               {size_t{0}, DirectH<1, Complex>::kDefaultScratchBytes}) {
            const DirectH<1, Complex> direct(MakeSpan(dets), ints, composed,
                                             threshold, diagonal, budget);
            direct.Apply(x.data(), y.data());
            for (size_t i = 0; i < x.size(); ++i) {
              worst = std::max(worst, std::abs(y[i] - y_ref[i]));
              largest = std::max(largest, std::abs(y_ref[i]));
            }
          }
        }
      }
      Expect(num_tasks > 0 && largest > 1.0 && worst < 1e-12 * largest,
             threshold < 0.0
                 ? "the Hamiltonian applied with related strings from the "
                   "byte table, by determinants and by strings, equals the "
                   "stored one"
                 : "the same with a threshold on the elements");
    }
    if (threshold >= 0.0) continue;
    std::vector<Complex> bra(dets.size()), ket(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
      bra[i] = Complex(std::cos(0.9 * i), std::sin(0.4 * i + 1.0));
      ket[i] = Complex(std::sin(1.1 * i), std::cos(0.6 * i));
    }
    const DensityMatrices<Complex> from_stored = ComputeDensityMatrices(
        MakeSpan(dets), stored, n, MakeSpan(bra), MakeSpan(ket), true);
    const DensityMatrices<Complex> from_tables = ComputeDensityMatricesFrom(
        MakeSpan(dets), TablePairs<1>(tables, dets.size()), n, MakeSpan(bra),
        MakeSpan(ket), true);
    double max_one = 0.0;
    double max_two = 0.0;
    for (size_t k = 0; k < from_stored.one.size(); ++k) {
      max_one =
          std::max(max_one, std::abs(from_stored.one[k] - from_tables.one[k]));
    }
    for (size_t k = 0; k < from_stored.two.size(); ++k) {
      max_two =
          std::max(max_two, std::abs(from_stored.two[k] - from_tables.two[k]));
    }
    Expect(max_one < 1e-10 && max_two < 1e-10,
           "density matrices from the tables equal those from the stored "
           "pairs");
    const internal::GroupStringTables<1> composed(
        MakeSpan(dets), GroupMask<1>(options, n), serial,
        StringRelations::kBytes);
    const DensityMatrices<Complex> from_bytes = ComputeDensityMatricesFrom(
        MakeSpan(dets), TablePairs<1>(composed, dets.size()), n, MakeSpan(bra),
        MakeSpan(ket), true);
    max_one = 0.0;
    max_two = 0.0;
    for (size_t k = 0; k < from_stored.one.size(); ++k) {
      max_one =
          std::max(max_one, std::abs(from_stored.one[k] - from_bytes.one[k]));
    }
    for (size_t k = 0; k < from_stored.two.size(); ++k) {
      max_two =
          std::max(max_two, std::abs(from_stored.two[k] - from_bytes.two[k]));
    }
    Expect(max_one < 1e-10 && max_two < 1e-10,
           "and with related strings composed from the byte table");
  }
}

// The selectors that score external determinants by perturbation theory
// against scores worked out determinant by determinant.
void TestPtScoreSelection(const Integrals<Complex>& ints) {
  const int n = ints.num_orbs();
  const std::vector<Det<1>> dets =
      ToDets<1>(RandomOccupations(n, ints.num_elec(), 60, 2, 71), 0);
  std::vector<Det<1>> sorted_dets = dets;
  std::sort(sorted_dets.begin(), sorted_dets.end());
  SparseH<Complex> h;
  std::vector<double> diagonal(dets.size());
  for (size_t d = 0; d < dets.size(); ++d) {
    diagonal[d] = DiagonalEnergy(dets[d], ints);
  }
  h.AppendDiagonal(diagonal);
  h.AppendBlock(BuildBlock(MakeSpan(dets), 0, ints, ConnectOptions()));
  std::vector<std::vector<Complex>> root_vectors =
      GuessVectors(MakeSpan(dets), h, ints, 2);
  const std::vector<double> energies =
      Davidson<Complex>(h, DavidsonOptions(), &root_vectors).energies;
  const Comm serial;
  const SharedVectors<Complex> vectors =
      ShareVectors(root_vectors, dets.size(), serial);
  const SharedArray<double> weights =
      ShareWeights(root_vectors, dets.size(), 2, serial);

  // Every determinant one or two excitations from the space, with V_a of
  // both roots from all determinants of the space.
  std::set<Det<1>> external;
  for (const Det<1>& det : dets) {
    std::vector<int> occ, virt;
    for (int p = 0; p < n; ++p) (det.Occupied(p) ? occ : virt).push_back(p);
    for (size_t x = 0; x < occ.size(); ++x) {
      for (size_t u = 0; u < virt.size(); ++u) {
        Det<1> single = det;
        single.Clear(occ[x]);
        single.Set(virt[u]);
        external.insert(single);
        for (size_t y = x + 1; y < occ.size(); ++y) {
          for (size_t v = u + 1; v < virt.size(); ++v) {
            Det<1> doubly = single;
            doubly.Clear(occ[y]);
            doubly.Set(virt[v]);
            external.insert(doubly);
          }
        }
      }
    }
  }
  for (const Det<1>& det : dets) external.erase(det);
  struct Reference {
    Det<1> det;
    double energy_score;     // sum over roots of |V|^2 / gap
    double amplitude_score;  // largest over roots of |V| / gap
  };
  std::vector<Reference> reference;
  for (const Det<1>& det : external) {
    Reference item{det, 0.0, 0.0};
    const double energy_a = DiagonalEnergy(det, ints);
    for (int k = 0; k < 2; ++k) {
      Complex v(0.0);
      for (size_t i = 0; i < dets.size(); ++i) {
        v += Hij(det, dets[i], ints) * root_vectors[k][i];
      }
      const double gap = energy_a - energies[k];
      if (gap <= 0.0) continue;
      item.energy_score += std::norm(v) / gap;
      item.amplitude_score = std::max(item.amplitude_score, std::abs(v) / gap);
    }
    reference.push_back(item);
  }

  const HeatBath heat_bath(ints, 0.0);
  const RestrictedFilter filter(LabelFilter(std::vector<int>(n, 0), 0), {}, n);
  const std::vector<uint8_t> all_active(dets.size(), 1);
  const SelectionContext<1, Complex> context{0,
                                             MakeSpan(dets),
                                             MakeSpan(sorted_dets),
                                             weights.span(),
                                             &vectors,
                                             energies,
                                             ints,
                                             heat_bath,
                                             filter,
                                             serial,
                                             MakeSpan(all_active),
                                             dets.size()};

  // By count: the space grows from 60 to 150 determinants.
  for (SelectMethod method : {SelectMethod::kCipsi, SelectMethod::kAsci}) {
    SelectInput select;
    select.method = method;
    select.growth = 2.5;
    select.target_dets = 1000;
    PtScoreSelector<1, Complex> selector(select);
    const std::vector<Det<1>> chosen = selector.Select(context).add;
    const bool amplitude = method == SelectMethod::kAsci;
    std::sort(reference.begin(), reference.end(),
              [&](const Reference& a, const Reference& b) {
                return amplitude ? a.amplitude_score > b.amplitude_score
                                 : a.energy_score > b.energy_score;
              });
    std::vector<Det<1>> expected;
    for (size_t k = 0; k < 90; ++k) expected.push_back(reference[k].det);
    std::sort(expected.begin(), expected.end());
    Expect(reference.size() > 1000 && chosen == expected,
           amplitude ? "asci selection takes the determinants of largest "
                       "first-order amplitude"
                     : "cipsi selection takes the determinants of largest "
                       "second-order energy");
  }

  // By sum: everything except the smallest energies that sum to sigma.
  {
    std::sort(reference.begin(), reference.end(),
              [](const Reference& a, const Reference& b) {
                return a.energy_score < b.energy_score;
              });
    double total = 0.0;
    for (const Reference& item : reference) total += item.energy_score;
    SelectInput select;
    select.method = SelectMethod::kAci;
    select.sigma = 0.05 * total;
    std::vector<Det<1>> expected;
    double left_out = 0.0;
    for (const Reference& item : reference) {
      if (item.energy_score <= 0.0) continue;
      if (left_out + item.energy_score <= select.sigma) {
        left_out += item.energy_score;
      } else {
        expected.push_back(item.det);
      }
    }
    std::sort(expected.begin(), expected.end());
    PtScoreSelector<1, Complex> selector(select);
    const std::vector<Det<1>> chosen = selector.Select(context).add;
    Expect(expected.size() > 20 && expected.size() + 20 < reference.size() &&
               chosen == expected,
           "aci selection leaves out the smallest energies that sum to "
           "sigma");
    std::printf(
        "      %zu candidates, second-order energy %.6f, %zu kept "
        "with sigma %.6f\n",
        reference.size(), total, chosen.size(), select.sigma);
  }

  // asci with pruning: the determinants of the space are ranked with the
  // candidates, by the largest magnitude of their coefficients.
  {
    SelectInput select;
    select.method = SelectMethod::kAsci;
    select.prune = true;
    select.growth = 2.5;
    select.target_dets = 100;
    struct Ranked {
      double value;
      bool internal;
      size_t index;  // into dets or into reference
    };
    std::vector<Ranked> ranking;
    for (size_t i = 0; i < dets.size(); ++i) {
      ranking.push_back(
          {std::max(std::abs(root_vectors[0][i]), std::abs(root_vectors[1][i])),
           true, i});
    }
    for (size_t a = 0; a < reference.size(); ++a) {
      ranking.push_back({reference[a].amplitude_score, false, a});
    }
    std::sort(
        ranking.begin(), ranking.end(),
        [](const Ranked& a, const Ranked& b) { return a.value > b.value; });
    std::vector<Det<1>> expected_add;
    std::vector<size_t> expected_drop;
    for (size_t r = 0; r < ranking.size(); ++r) {
      const bool kept = r < select.target_dets;
      if (kept && !ranking[r].internal) {
        expected_add.push_back(reference[ranking[r].index].det);
      } else if (!kept && ranking[r].internal) {
        expected_drop.push_back(ranking[r].index);
      }
    }
    std::sort(expected_add.begin(), expected_add.end());
    std::sort(expected_drop.begin(), expected_drop.end());
    PtScoreSelector<1, Complex> selector(select);
    const Selection<1> selection = selector.Select(context);
    Expect(!expected_drop.empty() && expected_drop.size() < dets.size() &&
               selection.add == expected_add && selection.drop == expected_drop,
           "asci with pruning keeps the highest-ranked determinants of the "
           "space and of the candidates");
    std::printf("      %zu of %zu determinants leave the space, %zu come in\n",
                selection.drop.size(), dets.size(), selection.add.size());

    // The eigensolver in the determinants that stay.
    std::vector<uint8_t> active(dets.size(), 1);
    for (size_t i : selection.drop) active[i] = 0;
    std::vector<Det<1>> staying;
    for (size_t i = 0; i < dets.size(); ++i) {
      if (active[i]) staying.push_back(dets[i]);
    }
    SparseH<Complex> h_staying;
    std::vector<double> diagonal_staying(staying.size());
    for (size_t d = 0; d < staying.size(); ++d) {
      diagonal_staying[d] = DiagonalEnergy(staying[d], ints);
    }
    h_staying.AppendDiagonal(diagonal_staying);
    h_staying.AppendBlock(
        BuildBlock(MakeSpan(staying), 0, ints, ConnectOptions()));
    DavidsonOptions dense;
    dense.dense_dim = 1 << 20;
    std::vector<std::vector<Complex>> exact(
        2, std::vector<Complex>(staying.size(), 0.0));
    const std::vector<double> exact_energies =
        Davidson<Complex>(h_staying, dense, &exact).energies;
    const MaskedOperator<Complex> masked(h, active);
    bool same = true;
    for (int dense_dim : {1 << 20, 0}) {
      DavidsonOptions options;
      options.dense_dim = dense_dim;
      options.tol = 1e-9;
      std::vector<std::vector<Complex>> guess = root_vectors;
      for (std::vector<Complex>& v : guess) {
        for (size_t i : selection.drop) v[i] = 0.0;
      }
      const DavidsonResult result = Davidson<Complex>(masked, options, &guess);
      for (int k = 0; k < 2; ++k) {
        same = same && result.converged &&
               std::abs(result.energies[k] - exact_energies[k]) < 1e-9;
        for (size_t i : selection.drop) {
          same = same && std::abs(guess[k][i]) < 1e-12;
        }
      }
    }
    Expect(same,
           "the eigensolver in the marked determinants gives the "
           "eigenpairs of the Hamiltonian in them, with zeros "
           "elsewhere");
  }

  // asci with pruning and a subset generating: every determinant, of the
  // space or not, is ranked by its amplitude from the generators.
  {
    SelectInput select;
    select.method = SelectMethod::kAsci;
    select.prune = true;
    select.growth = 2.5;
    select.target_dets = 100;
    select.core_dets = 20;
    std::vector<size_t> by_weight(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) by_weight[i] = i;
    std::sort(by_weight.begin(), by_weight.end(),
              [&](size_t a, size_t b) { return weights[a] > weights[b]; });
    by_weight.resize(select.core_dets);
    std::vector<Det<1>> pool(external.begin(), external.end());
    pool.insert(pool.end(), dets.begin(), dets.end());
    std::vector<std::pair<double, Det<1>>> ranking;
    for (const Det<1>& det : pool) {
      const double energy_a = DiagonalEnergy(det, ints);
      double score = 0.0;
      for (int k = 0; k < 2; ++k) {
        Complex v(0.0);
        for (size_t j : by_weight) {
          if (dets[j] == det) continue;
          v += Hij(det, dets[j], ints) * root_vectors[k][j];
        }
        const double gap = energy_a - energies[k];
        if (gap > 0.0) score = std::max(score, std::abs(v) / gap);
      }
      if (score > 0.0) ranking.push_back({score, det});
    }
    std::sort(
        ranking.begin(), ranking.end(),
        [](const std::pair<double, Det<1>>& a,
           const std::pair<double, Det<1>>& b) { return a.first > b.first; });
    std::vector<Det<1>> kept;
    for (size_t r = 0; r < select.target_dets; ++r) {
      kept.push_back(ranking[r].second);
    }
    std::sort(kept.begin(), kept.end());
    std::vector<Det<1>> expected_add;
    std::set_difference(kept.begin(), kept.end(), sorted_dets.begin(),
                        sorted_dets.end(), std::back_inserter(expected_add));
    std::vector<size_t> expected_drop;
    for (size_t i = 0; i < dets.size(); ++i) {
      if (!std::binary_search(kept.begin(), kept.end(), dets[i])) {
        expected_drop.push_back(i);
      }
    }
    PtScoreSelector<1, Complex> selector(select);
    const Selection<1> selection = selector.Select(context);
    Expect(ranking.size() > 200 && !expected_drop.empty() &&
               selection.add == expected_add && selection.drop == expected_drop,
           "asci with pruning and a generating subset ranks all "
           "determinants by their amplitude from the generators");
    std::printf("      %zu of %zu determinants leave the space, %zu come in\n",
                selection.drop.size(), dets.size(), selection.add.size());
  }

  // aci with pruning: the determinants of largest weight that hold all but
  // sigma of the squared norm generate and stay; the others are scored
  // from them, and the smallest energies, summing to sigma, are left out.
  {
    SelectInput select;
    select.method = SelectMethod::kAci;
    select.prune = true;
    select.sigma = 0.02;
    std::vector<size_t> by_weight(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) by_weight[i] = i;
    std::sort(by_weight.begin(), by_weight.end(),
              [&](size_t a, size_t b) { return weights[a] > weights[b]; });
    double total = 0.0;
    for (size_t i = 0; i < dets.size(); ++i) total += weights[i] * weights[i];
    std::vector<size_t> generators;
    double held = 0.0;
    for (size_t i : by_weight) {
      generators.push_back(i);
      held += weights[i] * weights[i];
      if (held >= (1.0 - select.sigma) * total) break;
    }
    std::vector<uint8_t> generates(dets.size(), 0);
    for (size_t i : generators) generates[i] = 1;
    std::vector<Det<1>> pool(external.begin(), external.end());
    for (size_t i = 0; i < dets.size(); ++i) {
      if (!generates[i]) pool.push_back(dets[i]);
    }
    std::vector<std::pair<double, Det<1>>> ranking;
    for (const Det<1>& det : pool) {
      const double energy_a = DiagonalEnergy(det, ints);
      double score = 0.0;
      for (int k = 0; k < 2; ++k) {
        Complex v(0.0);
        for (size_t j : generators) {
          v += Hij(det, dets[j], ints) * root_vectors[k][j];
        }
        const double gap = energy_a - energies[k];
        if (gap > 0.0) score += std::norm(v) / gap;
      }
      if (score > 0.0) ranking.push_back({score, det});
    }
    std::sort(
        ranking.begin(), ranking.end(),
        [](const std::pair<double, Det<1>>& a,
           const std::pair<double, Det<1>>& b) { return a.first < b.first; });
    std::vector<Det<1>> kept;
    double left_out = 0.0;
    for (const std::pair<double, Det<1>>& item : ranking) {
      if (left_out + item.first <= select.sigma) {
        left_out += item.first;
      } else {
        kept.push_back(item.second);
      }
    }
    std::sort(kept.begin(), kept.end());
    std::vector<Det<1>> expected_add;
    std::set_difference(kept.begin(), kept.end(), sorted_dets.begin(),
                        sorted_dets.end(), std::back_inserter(expected_add));
    std::vector<size_t> expected_drop;
    for (size_t i = 0; i < dets.size(); ++i) {
      if (!generates[i] &&
          !std::binary_search(kept.begin(), kept.end(), dets[i])) {
        expected_drop.push_back(i);
      }
    }
    PtScoreSelector<1, Complex> selector(select);
    const Selection<1> selection = selector.Select(context);
    Expect(generators.size() < dets.size() && !expected_drop.empty() &&
               !expected_add.empty() && kept.size() < ranking.size() &&
               selection.add == expected_add && selection.drop == expected_drop,
           "aci with pruning keeps the generating determinants and the "
           "others of largest energy");
    std::printf(
        "      %zu of %zu determinants generate, %zu leave the "
        "space, %zu come in\n",
        generators.size(), dets.size(), selection.drop.size(),
        selection.add.size());
  }
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

// The first m spinors of `ints`; the real parts only if S is double.
template <typename S>
Integrals<S> FirstSpinors(const Integrals<Complex>& ints, int m) {
  Integrals<S> out(m);
  auto take = [](Complex z) {
    return ScalarTraits<S>::Make(z.real(), z.imag());
  };
  for (int p = 0; p < m; ++p) {
    for (int q = 0; q < m; ++q) {
      out.SetOneBody(p, q, take(ints.OneBody(p, q)));
      for (int s = 1; s < m && p < q; ++s) {
        for (int r = 0; r < s; ++r) {
          out.SetAntisymOrdered(p, q, r, s,
                                take(ints.AntisymOrdered(p, q, r, s)));
        }
      }
    }
  }
  out.set_num_elec(ints.num_elec());
  out.Finalize();
  return out;
}

template <typename S>
using DenseMatrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

// H over the determinants of a GAS space in the order and with the phases
// of the plan (creation operators in the order of the plan's spinors), from
// the Slater-Condon rules of hamiltonian.h, without the core energy.
template <typename S>
DenseMatrix<S> ExplicitGasH(const Integrals<S>& ints, const GasSpace& space) {
  int nwords = 0;
  const std::vector<uint64_t> words = space.PlanDets(&nwords);
  const std::vector<int>& perm = space.layout().perm;
  const long dim = space.dim();
  std::vector<Det<1>> dets(dim);
  std::vector<double> sign(dim, 1.0);
  for (long d = 0; d < dim; ++d) {
    std::vector<int> occ;
    for (int p = 0; p < space.num_orbs(); ++p) {
      if (words[d * nwords + p / 64] >> (p % 64) & 1) occ.push_back(perm[p]);
    }
    int inversions = 0;
    for (size_t i = 0; i < occ.size(); ++i) {
      for (size_t j = i + 1; j < occ.size(); ++j) inversions += occ[i] > occ[j];
      dets[d].Set(occ[i]);
    }
    sign[d] = inversions % 2 ? -1.0 : 1.0;
  }
  DenseMatrix<S> h(dim, dim);
#pragma omp parallel for schedule(dynamic, 16)
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < dim; ++j) {
      const S value = i == j ? S(DiagonalEnergy(dets[i], ints))
                             : Hij(dets[i], dets[j], ints);
      h(i, j) = sign[i] * sign[j] * value;
    }
  }
  return h;
}

template <typename S>
struct GasCase {
  int nelec = 0;
  std::vector<GasConstraint> constraints;
  std::vector<LabelSet> labels;
  int max_subgroup = 12;
};

template <typename S>
S RandomScalar(std::mt19937* rng);
template <>
double RandomScalar<double>(std::mt19937* rng) {
  return std::uniform_real_distribution<double>(-1.0, 1.0)(*rng);
}
template <>
Complex RandomScalar<Complex>(std::mt19937* rng) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  return Complex(u(*rng), u(*rng));
}

// The kernels against the explicit matrix: the diagonal and H c for a
// random c, element by element, with each of the three kernels.
template <typename S>
void TestGasKernels(const Integrals<S>& ints, const GasCase<S>& c,
                    const std::string& what, long min_dim) {
  const GasLayout layout =
      MakeGasLayout(ints.num_orbs(), c.constraints, c.max_subgroup);
  std::vector<int> targets;
  for (const LabelSet& set : c.labels) {
    int sum = 0;
    for (int p = 0; p < c.nelec; ++p) sum += set.labels[p];
    targets.push_back(sum);
  }
  const GasSpace space(ints.num_orbs(), c.nelec, layout, c.labels, targets);
  const long dim = space.dim();
  const DenseMatrix<S> h = ExplicitGasH(ints, space);
  const std::vector<S> v = AntisymmetrizedPairs(ints, layout.perm, c.nelec);
  std::vector<double> diag(dim);
  space.Diagonal(v, diag.data());
  double diag_error = 0.0;
  for (long i = 0; i < dim; ++i) {
    diag_error = std::max(diag_error, std::abs(diag[i] - Real(h(i, i))));
  }
  std::mt19937 rng(7);
  std::vector<S> x(dim), y(dim);
  for (S& value : x) value = RandomScalar<S>(&rng);
  Eigen::Map<const Eigen::Matrix<S, Eigen::Dynamic, 1>> xv(x.data(), dim);
  const Eigen::Matrix<S, Eigen::Dynamic, 1> reference = h * xv;
  double error[3] = {0, 0, 0};
  const GasKernel kernels[3] = {GasKernel::kBlas, GasKernel::kBlock,
                                GasKernel::kReference};
  for (int k = 0; k < 3; ++k) {
    space.Sigma(v, x.data(), y.data(), kernels[k]);
    for (long i = 0; i < dim; ++i) {
      error[k] = std::max(error[k], std::abs(y[i] - reference[i]));
    }
  }
  const double scale = reference.cwiseAbs().maxCoeff();
  Expect(dim >= min_dim && diag_error < 1e-10 &&
             std::max({error[0], error[1], error[2]}) < 1e-10 * scale,
         what + " (" + ScalarTraits<S>::kName + ", " + std::to_string(dim) +
             " determinants)");
  std::printf(
      "      largest differences: diagonal %.1e, H c by the BLAS, block, "
      "reference kernel %.1e %.1e %.1e (largest element %.1e)\n",
      diag_error, error[0], error[1], error[2], scale);
}

template <typename S>
class DenseOperator : public CiOperator<S> {
 public:
  DenseOperator(const GasSpace& space, const std::vector<S>& v)
      : space_(space), v_(v) {}
  void Apply(const S* x, S* y) const override {
    space_.Sigma(v_, x, y);
    ++products;
  }
  mutable int products = 0;

 private:
  const GasSpace& space_;
  const std::vector<S>& v_;
};

// The Davidson of the fci mode against the eigenvalues of the explicit
// matrix, several roots, a small subspace, a checkpoint and a restart.
template <typename S>
void TestFciDavidson(const Integrals<S>& ints, const GasCase<S>& c,
                     const std::string& path) {
  const GasLayout layout =
      MakeGasLayout(ints.num_orbs(), c.constraints, c.max_subgroup);
  std::vector<int> targets;
  for (const LabelSet& set : c.labels) {
    int sum = 0;
    for (int p = 0; p < c.nelec; ++p) sum += set.labels[p];
    targets.push_back(sum);
  }
  const GasSpace space(ints.num_orbs(), c.nelec, layout, c.labels, targets);
  const long dim = space.dim();
  const std::vector<S> v = AntisymmetrizedPairs(ints, layout.perm, c.nelec);
  std::vector<double> diag(dim);
  space.Diagonal(v, diag.data());
  Eigen::SelfAdjointEigenSolver<DenseMatrix<S>> exact(
      ExplicitGasH(ints, space));
  const DenseOperator<S> h(space, v);
  const VectorLayout whole = VectorLayout::Whole(dim);
  const std::string name = std::string(ScalarTraits<S>::kName) + ", " +
                           std::to_string(dim) + " determinants";

  for (int roots : {1, 3}) {
    for (int space_per_root : {2, 3}) {
      FciDavidsonOptions options;
      options.num_roots = roots;
      options.tol = 1e-7;
      options.max_space = space_per_root * roots;
      options.dense_dim = 0;
      options.max_iterations = 300;
      options.print = false;
      const FciDavidsonResult<S> r =
          FciDavidson(h, diag.data(), whole, options);
      double error = 0.0, overlap = 1.0;
      for (int k = 0; k < roots; ++k) {
        error =
            std::max(error, std::abs(r.energies[k] - exact.eigenvalues()[k]));
        if (k == 0) {
          S dot(0.0);
          for (long i = 0; i < dim; ++i) {
            dot += Conj(exact.eigenvectors()(i, 0)) * r.vectors[0][i];
          }
          overlap = std::abs(dot);
        }
      }
      Expect(r.converged && error < 1e-11 && std::abs(overlap - 1.0) < 1e-10,
             "fci Davidson with " + std::to_string(roots) + " root(s) and " +
                 std::to_string(options.max_space) +
                 " vectors equals the full diagonalization (" + name + ")");
      std::printf("      %d iterations, largest energy difference %.1e\n",
                  r.num_iterations, error);
    }
  }

  // Stopped after a few iterations with checkpoints, then restarted.
  std::remove(path.c_str());
  FciDavidsonOptions options;
  options.num_roots = 2;
  options.tol = 1e-7;
  options.max_space = 6;
  options.dense_dim = 0;
  options.checkpoint = path;
  options.fingerprint = space.Fingerprint();
  options.max_iterations = 4;
  options.print = false;
  const FciDavidsonResult<S> first =
      FciDavidson(h, diag.data(), whole, options);
  std::vector<BigArray<S>> back;
  std::vector<double> saved;
  const int count = ReadCiCheckpoint<S>(path, whole, space.Fingerprint(), 2,
                                        false, &back, &saved);
  double vector_error = 0.0;
  for (int k = 0; k < count; ++k) {
    for (long i = 0; i < dim; ++i) {
      vector_error =
          std::max(vector_error, std::abs(back[k][i] - first.vectors[k][i]));
    }
  }
  Expect(!first.converged && count == 2 && vector_error < 1e-12 &&
             saved[1] == first.energies[1],
         "fci checkpoint holds the Ritz vectors and energies (" + name + ")");
  options.restart = true;
  options.max_iterations = 300;
  const FciDavidsonResult<S> second =
      FciDavidson(h, diag.data(), whole, options);
  options.max_iterations = 300;
  const FciDavidsonResult<S> again =
      FciDavidson(h, diag.data(), whole, options);
  const double error =
      std::max(std::abs(second.energies[0] - exact.eigenvalues()[0]),
               std::abs(second.energies[1] - exact.eigenvalues()[1]));
  Expect(second.restarted && second.converged && error < 1e-11 &&
             again.restarted && again.num_iterations == 1,
         "fci restart from the checkpoint converges to the same roots, and at "
         "once from a converged one (" +
             name + ")");
  std::printf("      %d + %d iterations, then %d; energy difference %.1e\n",
              first.num_iterations, second.num_iterations, again.num_iterations,
              error);
  bool refused = false;
  try {
    ReadCiCheckpoint<S>(path, whole, space.Fingerprint() + 1, 2, false, &back);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  Expect(refused, "fci checkpoint of another space is refused");
  std::remove(path.c_str());
}

// The distributed sigma on one rank, with the vector sent round in several
// slices, against the kernel in one piece; and the Davidson on it.
template <typename S>
void TestDistributedSigma(const Integrals<S>& ints, const GasCase<S>& c) {
  const GasLayout layout =
      MakeGasLayout(ints.num_orbs(), c.constraints, c.max_subgroup);
  std::vector<int> targets;
  for (const LabelSet& set : c.labels) {
    int sum = 0;
    for (int p = 0; p < c.nelec; ++p) sum += set.labels[p];
    targets.push_back(sum);
  }
  const GasSpace space(ints.num_orbs(), c.nelec, layout, c.labels, targets);
  const long dim = space.dim();
  const std::vector<S> v = AntisymmetrizedPairs(ints, layout.perm, c.nelec);
  std::mt19937 rng(11);
  std::vector<S> x(dim), y(dim), z(dim);
  for (S& value : x) value = RandomScalar<S>(&rng);
  space.Sigma(v, x.data(), y.data());
  std::vector<double> diag(dim), diag_dist(dim);
  space.Diagonal(v, diag.data());
  const Comm serial;
  double error = 0.0, diag_error = 0.0, scale = 0.0;
  int most_slices = 0;
  for (int64_t slice : {int64_t(0), int64_t(dim / 7 + 1)}) {
    for (bool by_work : {true, false}) {
      const GasDistribution dist(space, 1, by_work, slice);
      most_slices = std::max<int>(most_slices, dist.Slices(0, 0).size());
      const DistributedGasOperator<S> op(space, v, dist, serial, false);
      op.Apply(x.data(), z.data());
      op.Apply(x.data(), z.data());  // twice: nothing left over
      op.Diagonal(diag_dist.data());
      for (long i = 0; i < dim; ++i) {
        error = std::max(error, std::abs(z[i] - y[i]));
        scale = std::max(scale, std::abs(y[i]));
        diag_error = std::max(diag_error, std::abs(diag_dist[i] - diag[i]));
      }
    }
  }
  Expect(most_slices > 3 && error < 1e-12 * scale && diag_error < 1e-12,
         std::string("distributed sigma in slices equals the kernel (") +
             ScalarTraits<S>::kName + ", " + std::to_string(dim) +
             " determinants)");
  std::printf("      up to %d slices; largest difference %.1e of %.1e\n",
              most_slices, error, scale);
}

// A space for the kernel tests with random v: n spinors, sub-groups given
// explicitly (groups_a, groups_b) or cut at max_subgroup.
struct RandomGasCase {
  int n = 0;
  int nelec = 0;
  std::vector<GasConstraint> constraints;
  std::vector<LabelSet> labels;
  std::vector<std::vector<int>> groups_a, groups_b;
  int max_subgroup = 12;
};

// Consecutive spinors first .. first + size - 1.
std::vector<int> Range(int first, int size) {
  std::vector<int> out(size);
  for (int i = 0; i < size; ++i) out[i] = first + i;
  return out;
}

// The BLAS, block and distributed kernels against the reference kernel for
// random v and c (the kernels are linear in v; no Hamiltonian needed), so
// that spaces wider than the explicit matrix allows can be checked.
template <typename S>
void TestGasKernelsRandom(const RandomGasCase& c, const std::string& what,
                          long min_dim) {
  const GasLayout layout =
      MakeGasLayout(c.n, c.constraints, c.max_subgroup, c.groups_a, c.groups_b);
  std::vector<int> targets;
  for (const LabelSet& set : c.labels) {
    int sum = 0;
    for (int p = 0; p < c.nelec; ++p) sum += set.labels[layout.perm[p]];
    targets.push_back(sum);
  }
  const GasSpace space(c.n, c.nelec, layout, c.labels, targets);
  const long dim = space.dim();
  // For few spinors, the space against all determinants: those of the plan
  // distinct and allowed, as many as are allowed.
  bool same_space = true;
  if (c.n <= 24) {
    auto allowed = [&](uint64_t occ) {
      if (__builtin_popcountll(occ) != c.nelec) return false;
      for (const GasConstraint& con : c.constraints) {
        int sum = 0;
        for (size_t k = 0; k < con.orbitals.size(); ++k) {
          if (occ >> con.orbitals[k] & 1) {
            sum += con.weights.empty() ? 1 : con.weights[k];
          }
        }
        if (sum < con.min_elec || sum > con.max_elec) return false;
      }
      for (size_t s = 0; s < c.labels.size(); ++s) {
        int sum = 0;
        for (int p = 0; p < c.n; ++p) {
          if (occ >> p & 1) sum += c.labels[s].labels[p];
        }
        if (!LabelConserved(sum - targets[s], c.labels[s].mod)) return false;
      }
      return true;
    };
    long count = 0;
    for (uint64_t occ = 0; occ < (uint64_t(1) << c.n); ++occ) {
      count += allowed(occ);
    }
    int nwords = 0;
    const std::vector<uint64_t> words = space.PlanDets(&nwords);
    std::set<uint64_t> seen;
    for (long d = 0; d < dim; ++d) {
      uint64_t occ = 0;
      for (int p = 0; p < c.n; ++p) {
        if (words[d * nwords] >> p & 1) occ |= uint64_t(1) << layout.perm[p];
      }
      same_space = same_space && allowed(occ) && seen.insert(occ).second;
    }
    same_space = same_space && count == dim;
  }
  int widest[2] = {0, 0};
  for (size_t g = 0; g < layout.sub_sizes.size(); ++g) {
    widest[static_cast<int>(g) >= layout.num_sub_a] += layout.sub_sizes[g];
  }
  const long npair = static_cast<long>(c.n) * (c.n - 1) / 2;
  std::mt19937 rng(17);
  std::vector<S> v(static_cast<size_t>(npair) * npair);
  for (S& value : v) value = RandomScalar<S>(&rng);
  std::vector<S> x(dim), reference(dim), y(dim);
  for (S& value : x) value = RandomScalar<S>(&rng);
  space.Sigma(v, x.data(), reference.data(), GasKernel::kReference);
  double scale = 0.0;
  for (const S& value : reference) scale = std::max(scale, std::abs(value));
  double error[3] = {0, 0, 0};
  const GasKernel kernels[2] = {GasKernel::kBlas, GasKernel::kBlock};
  for (int k = 0; k < 2; ++k) {
    space.Sigma(v, x.data(), y.data(), kernels[k]);
    for (long i = 0; i < dim; ++i) {
      error[k] = std::max(error[k], std::abs(y[i] - reference[i]));
    }
  }
  const Comm serial;
  const GasDistribution dist(space, 1, true, dim / 5 + 1);
  const DistributedGasOperator<S> op(space, v, dist, serial, false);
  op.Apply(x.data(), y.data());
  for (long i = 0; i < dim; ++i) {
    error[2] = std::max(error[2], std::abs(y[i] - reference[i]));
  }
  Expect(space.block_ok() && dim >= min_dim && same_space &&
             std::max({error[0], error[1], error[2]}) < 1e-12 * scale,
         what + " (" + ScalarTraits<S>::kName + ", sides of " +
             std::to_string(widest[0]) + " and " + std::to_string(widest[1]) +
             " spinors, " + std::to_string(dim) + " determinants)");
  std::printf(
      "      largest differences to the reference kernel: BLAS, block, "
      "distributed %.1e %.1e %.1e (largest element %.1e)\n",
      error[0], error[1], error[2], scale);
}

template <typename S>
void TestGasKernelsWide() {
  // Small spaces cut into many small sub-groups: operators that span
  // several sub-groups of a side.
  RandomGasCase fci;
  fci.n = 16;
  fci.nelec = 6;
  fci.groups_a = {{0}, Range(1, 2), Range(3, 3), Range(6, 3)};
  fci.groups_b = {Range(9, 2), Range(11, 5)};
  TestGasKernelsRandom<S>(fci,
                          "fci kernels, random v: full CI, sub-groups "
                          "of 1 to 5 spinors",
                          8008);
  std::vector<int> axis(80), parity(80);
  for (int p = 0; p < 80; ++p) {
    axis[p] = (p * p + 3 * p) % 5 - 2;
    parity[p] = (p / 3) % 2;
  }
  fci.labels = {{{axis.begin(), axis.begin() + 16}, 0},
                {{parity.begin(), parity.begin() + 16}, 2}};
  TestGasKernelsRandom<S>(fci, "the same with two label sets", 500);
  // GAS with weights, one of them negative (no bound on single sub-groups
  // from that constraint), and labels.
  RandomGasCase gas;
  gas.n = 20;
  gas.nelec = 6;
  GasConstraint holes, level, balance;
  holes.orbitals = Range(0, 6);
  holes.min_elec = 4;
  holes.max_elec = 6;
  level.orbitals = Range(6, 14);
  for (int p = 6; p < 20; ++p) level.weights.push_back(p < 10 ? 1 : 2);
  level.max_elec = 3;
  balance.orbitals = {0, 1, 2, 3, 6, 7};
  balance.weights = {1, 1, 1, 1, -1, -1};
  balance.min_elec = 2;
  balance.max_elec = 4;
  gas.constraints = {holes, level, balance};
  gas.max_subgroup = 3;
  gas.labels = {{{axis.begin(), axis.begin() + 20}, 4}};
  TestGasKernelsRandom<S>(gas,
                          "fci kernels, random v: weighted GAS, labels "
                          "modulo 4, sub-groups of at most 3",
                          200);
  // More than 64 spinors on side A, then on side B: 72 spinors, at most two
  // electrons in 62 of them, with labels.
  RandomGasCase wide;
  wide.n = 72;
  wide.nelec = 4;
  GasConstraint virt;
  virt.orbitals = Range(4, 62);
  virt.max_elec = 2;
  wide.constraints = {virt};
  wide.labels = {{{axis.begin(), axis.begin() + 72}, 0},
                 {{parity.begin(), parity.begin() + 72}, 2}};
  wide.groups_a = {Range(0, 4),   Range(4, 16),  Range(20, 16),
                   Range(36, 12), Range(48, 10), Range(58, 8)};
  wide.groups_b = {Range(66, 3), Range(69, 3)};
  TestGasKernelsRandom<S>(wide,
                          "fci kernels, random v: more than 64 "
                          "spinors on side A",
                          2000);
  std::swap(wide.groups_a, wide.groups_b);
  TestGasKernelsRandom<S>(wide,
                          "fci kernels, random v: more than 64 "
                          "spinors on side B",
                          2000);
}

template <typename S>
void TestGasCi(const Integrals<Complex>& complex_ints) {
  const int n = complex_ints.num_orbs();
  // Full CI: 6 electrons in the first 12 spinors, no constraint.
  {
    const Integrals<S> small = FirstSpinors<S>(complex_ints, 12);
    GasCase<S> fci;
    fci.nelec = 6;
    TestGasKernels(small, fci,
                   "fci kernels equal the explicit Hamiltonian: full CI", 924);
    fci.max_subgroup = 4;
    TestGasKernels(small, fci, "the same with sub-groups of 4 spinors", 924);
  }
  const Integrals<S> ints = FirstSpinors<S>(complex_ints, n);
  const int nelec = complex_ints.num_elec();
  // GAS: at most two holes in the lowest nelec spinors, at most one electron
  // above nelec + 4; labels as of an axis and of inversion.
  GasCase<S> gas;
  gas.nelec = nelec;
  GasConstraint holes, high, level;
  for (int p = 0; p < nelec; ++p) holes.orbitals.push_back(p);
  holes.min_elec = nelec - 2;
  holes.max_elec = nelec;
  for (int p = nelec + 4; p < n; ++p) high.orbitals.push_back(p);
  high.max_elec = 1;
  gas.constraints = {holes, high};
  TestGasKernels(ints, gas, "fci kernels: two GAS constraints", 4000);
  std::vector<int> axis(n), parity(n);
  for (int p = 0; p < n; ++p) {
    axis[p] = (p * p + 3 * p) % 5 - 2;
    parity[p] = (p / 3) % 2;
  }
  GasCase<S> labelled = gas;
  labelled.labels = {{axis, 0}, {parity, 2}};
  TestGasKernels(ints, labelled, "fci kernels: GAS and two label sets", 500);
  labelled.labels = {{axis, 4}};
  TestGasKernels(ints, labelled, "fci kernels: GAS and labels modulo 4", 500);
  // Weighted: electrons above nelec count once, above nelec + 4 twice, at
  // most two (an excitation level), and spinors 0-3 hold four electrons.
  GasCase<S> weighted;
  weighted.nelec = nelec;
  for (int p = nelec; p < n; ++p) {
    level.orbitals.push_back(p);
    level.weights.push_back(p < nelec + 4 ? 1 : 2);
  }
  level.max_elec = 2;
  GasConstraint frozen;
  frozen.orbitals = {0, 1, 2, 3};
  frozen.min_elec = frozen.max_elec = 4;
  weighted.constraints = {level, frozen};
  TestGasKernels(ints, weighted, "fci kernels: weighted constraint", 300);

  TestGasKernelsWide<S>();
  TestDistributedSigma(ints, gas);
  labelled.labels = {{axis, 0}, {parity, 2}};
  TestDistributedSigma(ints, labelled);
  TestFciDavidson(
      ints, labelled,
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
          "/zshci_test_" + ScalarTraits<S>::kName + ".ckpt");
}

// ---------------------------------------------------------------------------
// Spin-flip symmetry ("spin_flip", spin_flip.h).

// Random real integrals over n spatial orbitals with increasing orbital
// energies, so that the low states are of the usual kind.
Integrals<double> RandomSpatialIntegrals(int n, unsigned seed) {
  Integrals<double> ints = Integrals<double>::Spatial(n, {});
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q <= p; ++q) {
      const double h = p == q ? 0.5 * p - 2.0 : 0.08 * uniform(rng);
      ints.SetSpatialOneBody(p, q, h);
      ints.SetSpatialOneBody(q, p, h);
      for (int r = 0; r <= p; ++r) {
        for (int s = 0; s <= r; ++s) {
          const double v = p == q && r == s ? 0.25 + 0.05 * uniform(rng)
                                            : 0.04 * uniform(rng);
          ints.SetSpatialTwoBody(p, q, r, s, v);
        }
      }
    }
  }
  ints.Finalize();
  return ints;
}

struct SpinFlipRun {
  std::vector<double> energies;  // with the perturbative correction if any
  std::vector<Det<1>> dets;      // of wfn.h5
  std::vector<std::vector<double>> vectors;
  std::string prefix;
};

// A select-mode run over spatial orbitals; reads shci.e and wfn.h5.
SpinFlipRun RunSpinFlipCase(const Integrals<double>& ints,
                            const std::string& prefix, const std::string& dets,
                            int nroots, int spin_flip, double epsilon1,
                            const std::string& extra, int max_iter = 12,
                            const std::string& select_extra = "") {
  std::system(("mkdir -p " + prefix).c_str());
  char schedule[128];
  std::snprintf(schedule, sizeof(schedule), "[[0, %.3e]]",
                std::max(epsilon1, 1e-14));
  const std::string text =
      "{\"mode\": \"select\", \"integrals\": \"unused\", \"orbitals\": "
      "\"spatial\", \"nroots\": " +
      std::to_string(nroots) + ", \"prefix\": \"" + prefix +
      "\", \"dets\": " + dets +
      ", \"spin_flip\": " + std::to_string(spin_flip) +
      ", \"select\": {\"schedule\": " + schedule +
      ", \"max_iter\": " + std::to_string(max_iter) +
      ", \"energy_tol\": 1e-13" + select_extra +
      "}, \"davidson\": {\"tol\": 1e-10, \"checkpoint_every\": 1}, "
      "\"output_level\": 0" +
      extra + "}";
  RunSelect<1, double>(ParseInput(text), ints, Comm());
  SpinFlipRun run;
  run.prefix = prefix;
  run.energies.assign(nroots, 0.0);
  std::FILE* file = std::fopen((prefix + "/shci.e").c_str(), "rb");
  if (file != nullptr) {
    if (std::fread(run.energies.data(), sizeof(double), nroots, file) !=
        static_cast<size_t>(nroots)) {
      run.energies.assign(nroots, 0.0);
    }
    std::fclose(file);
  }
  int width = 0;
  const std::vector<uint64_t> words = ReadDetWords(prefix + "/wfn.h5", &width);
  for (uint64_t word : words) {
    Det<1> det;
    det.words[0] = word;
    run.dets.push_back(det);
  }
  ReadCoefficients(prefix + "/wfn.h5", run.dets.size(), nroots, &run.vectors);
  return run;
}

// <psi|S^2|psi> for a normalized vector over determinants of Ms = 0:
// |S+ psi|^2, S+ = sum_P a+(2P) a(2P+1), which has no signs in this order.
double SpinSquared(const std::vector<Det<1>>& dets,
                   const std::vector<double>& c) {
  std::map<uint64_t, double> raised;
  for (size_t i = 0; i < dets.size(); ++i) {
    const uint64_t x = dets[i].words[0];
    for (int p = 0; p < 32; ++p) {
      const uint64_t alpha = 1ULL << (2 * p), beta = 1ULL << (2 * p + 1);
      if ((x & beta) && !(x & alpha)) raised[(x ^ beta) | alpha] += c[i];
    }
  }
  double sum = 0.0;
  for (const auto& item : raised) sum += item.second * item.second;
  return sum;
}

double MaxDatasetDifference(const std::string& a, const std::string& b,
                            const std::string& name) {
  HighFive::File fa(a, HighFive::File::ReadOnly);
  HighFive::File fb(b, HighFive::File::ReadOnly);
  HighFive::DataSet da = fa.getDataSet(name), db = fb.getDataSet(name);
  if (da.getDimensions() != db.getDimensions()) return 1e300;
  size_t size = 1;
  for (size_t d : da.getDimensions()) size *= d;
  std::vector<double> va(size), vb(size);
  da.read(va.data());
  db.read(vb.data());
  double diff = 0.0;
  for (size_t k = 0; k < size; ++k) {
    diff = std::max(diff, std::abs(va[k] - vb[k]));
  }
  return diff;
}

void TestSpinFlip() {
  const std::string base =
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
      "/zshci_test_flip";
  std::system(("rm -rf " + base).c_str());

  // Layout: pairs come together, canonical first; expanding and projecting
  // a vector over the functions returns it.
  {
    std::vector<Det<1>> dets(4);
    for (int p : {0, 1, 2, 5}) dets[0].Set(p);  // (0a 0b 1a 2b)
    for (int p : {0, 1, 2, 3}) dets[1].Set(p);  // closed shell
    for (int p : {0, 1, 3, 4}) dets[2].Set(p);  // flip of dets[0]
    for (int p : {0, 2, 4, 7}) dets[3].Set(p);  // (0a 1a 2a 3b): Ms != 0
    std::vector<Det<1>> paired;
    std::vector<long> source;
    AppendPairs(std::vector<Det<1>>(dets.begin(), dets.begin() + 3),
                SpinFlipLambda(1, 4), &paired, &source);
    SpinFlipLayout layout(1, 4);
    layout.Extend(MakeSpan(paired), 0);
    std::vector<double> x = {0.3, -0.7};
    const std::vector<double> back = layout.Project(layout.Expand(x).data());
    bool ok = paired.size() == 3 && layout.num_functions() == 2 &&
              std::abs(back[0] - x[0]) + std::abs(back[1] - x[1]) < 1e-15;
    for (size_t j = 0; j < paired.size(); ++j) {
      ok = ok && source[j] >= 0 && dets[source[j]] == paired[j];
    }
    std::vector<Det<1>> triplet;
    AppendPairs(std::vector<Det<1>>{dets[1]}, SpinFlipLambda(-1, 4), &triplet);
    bool rejects_ms = false;
    try {
      SpinFlipLayout other(1, 4);
      other.Extend(Span<Det<1>>(&dets[3], 1), 0);
    } catch (const std::exception&) {
      rejects_ms = true;
    }
    Expect(ok && triplet.empty() && rejects_ms,
           "spin_flip layout: pairs together, expand and project invert each "
           "other, a closed shell has no odd-spin function, Ms != 0 rejected");
  }

  // Full spaces (epsilon1 = 0 fills the space): 6 electrons in 8 orbitals.
  const Integrals<double> ints = RandomSpatialIntegrals(8, 7);
  const std::string closed = "[[0,1,2,3,4,5]]";
  const std::string open = "[[0,1,2,3,4,7]]";  // 0, 1 doubly, 2a 3b
  const std::string high = "[[0,1,2,3,4,6]]";  // Ms = 1
  const std::string rdm = ", \"rdm\": {\"one\": true, \"two\": true}";
  const SpinFlipRun plain =
      RunSpinFlipCase(ints, base + "/plain", closed, 6, 0, 0.0, rdm);
  // The lowest singlet and triplet of the plain run.
  int singlet = -1, triplet = -1;
  std::string spins;
  for (int k = 0; k < 6; ++k) {
    const double s2 = SpinSquared(plain.dets, plain.vectors[k]);
    char text[64];
    std::snprintf(text, sizeof(text), " %.4f", s2);
    spins += text;
    if (singlet < 0 && std::abs(s2) < 1e-6) singlet = k;
    if (triplet < 0 && std::abs(s2 - 2.0) < 1e-6) triplet = k;
  }
  std::printf("      plain roots: S^2 =%s\n", spins.c_str());
  const SpinFlipRun even =
      RunSpinFlipCase(ints, base + "/even", closed, 1, 1, 0.0, rdm);
  const SpinFlipRun odd =
      RunSpinFlipCase(ints, base + "/odd", open, 1, -1, 0.0, "");
  const SpinFlipRun ms1 =
      RunSpinFlipCase(ints, base + "/ms1", high, 1, 0, 0.0, "");
  const bool found = singlet >= 0 && triplet >= 0;
  const double e_singlet = found ? plain.energies[singlet] : 0.0;
  const double e_triplet = found ? plain.energies[triplet] : 0.0;
  std::printf(
      "      singlet %.12f (+1: %.12f), triplet %.12f (-1: %.12f, "
      "Ms = 1: %.12f)\n",
      e_singlet, even.energies[0], e_triplet, odd.energies[0], ms1.energies[0]);
  Expect(found && std::abs(even.energies[0] - e_singlet) < 1e-10 &&
             even.dets.size() == plain.dets.size(),
         "spin_flip +1 in the full space gives the lowest singlet of the "
         "plain calculation");
  Expect(found && std::abs(odd.energies[0] - e_triplet) < 1e-10 &&
             std::abs(ms1.energies[0] - e_triplet) < 1e-10,
         "spin_flip -1 gives the lowest triplet, as the Ms = 1 calculation");
  const double s2_even = SpinSquared(even.dets, even.vectors[0]);
  const double s2_odd = SpinSquared(odd.dets, odd.vectors[0]);
  Expect(std::abs(s2_even) < 1e-8 && std::abs(s2_odd - 2.0) < 1e-8,
         "the expanded vectors have S^2 = 0 and 2");
  // The convention: c(Dbar) = lambda p(D) c(D), lambda = (-1)^(S + N/2), in
  // the eigenvectors of the plain calculation.
  {
    std::map<uint64_t, size_t> where;
    for (size_t i = 0; i < plain.dets.size(); ++i) {
      where[plain.dets[i].words[0]] = i;
    }
    double worst = 0.0;
    for (int k : {singlet, triplet}) {
      if (k < 0) continue;
      const int lambda = SpinFlipLambda(k == singlet ? 1 : -1, 6);
      for (size_t i = 0; i < plain.dets.size(); ++i) {
        const Det<1> flipped = FlipSpins(plain.dets[i]);
        const int p = NumDoublyOccupied(plain.dets[i]) % 2 == 0 ? 1 : -1;
        const double expected = lambda * p * plain.vectors[k][i];
        worst = std::max(
            worst,
            std::abs(plain.vectors[k][where[flipped.words[0]]] - expected));
      }
    }
    Expect(found && worst < 1e-8,
           "plain eigenvectors obey c(Dbar) = (-1)^(S + N/2) p(D) c(D)");
  }
  {
    const std::string a = base + "/plain/rdm_" + std::to_string(singlet) + "_" +
                          std::to_string(singlet) + ".h5";
    const std::string b = base + "/even/rdm_0_0.h5";
    const double d1 = found ? MaxDatasetDifference(a, b, "dm1") : 1.0;
    const double d2 = found ? MaxDatasetDifference(a, b, "dm2") : 1.0;
    std::printf("      density matrices: largest differences %.1e, %.1e\n", d1,
                d2);
    Expect(d1 < 1e-10 && d2 < 1e-10,
           "density matrices of the expanded vector equal those of the plain "
           "singlet");
  }

  // Every way of holding the Hamiltonian, the one-function-per-pair space
  // being half the size.
  {
    double worst = 0.0;
    for (const std::string& mode : {"connections", "direct", "adapt"}) {
      const std::string extra = ", \"hamiltonian\": \"" + mode + "\"" +
                                (mode == "adapt" ? ", \"memory\": 0.0012" : "");
      const SpinFlipRun run =
          RunSpinFlipCase(ints, base + "/" + mode, closed, 1, 1, 0.0, extra);
      worst = std::max(worst, std::abs(run.energies[0] - e_singlet));
    }
    Expect(found && worst < 1e-10,
           "spin_flip: stored, connections, direct and adapt agree");
  }

  // Selected spaces, perturbation theory, resuming: 8 electrons in 12
  // orbitals.
  const Integrals<double> big = RandomSpatialIntegrals(12, 11);
  const std::string hf = "[[0,1,2,3,4,5,6,7]]";
  const std::string det_pt =
      ", \"pt\": {\"method\": \"deterministic\", \"epsilon2\": 1e-7}";
  const SpinFlipRun sel_plain =
      RunSpinFlipCase(big, base + "/sel_plain", hf, 1, 0, 2e-3, det_pt, 6);
  const SpinFlipRun sel_flip =
      RunSpinFlipCase(big, base + "/sel_flip", hf, 1, 1, 2e-3, det_pt, 6);
  std::set<uint64_t> set_plain, set_flip;
  bool closed_under_flip = true;
  for (const Det<1>& d : sel_plain.dets) set_plain.insert(d.words[0]);
  for (const Det<1>& d : sel_flip.dets) set_flip.insert(d.words[0]);
  for (const Det<1>& d : sel_flip.dets) {
    closed_under_flip =
        closed_under_flip && set_flip.count(FlipSpins(d).words[0]) == 1;
  }
  // Energies from shci.e include PT; the variational ones are in wfn.h5.
  std::printf(
      "      selected: %zu determinants (plain %zu), E + PT %.12f "
      "(plain %.12f)\n",
      sel_flip.dets.size(), sel_plain.dets.size(), sel_flip.energies[0],
      sel_plain.energies[0]);
  Expect(set_plain == set_flip && closed_under_flip &&
             sel_plain.dets.size() > 1000 &&
             std::abs(sel_plain.energies[0] - sel_flip.energies[0]) < 1e-10,
         "selected space closed under the flip and equal to the plain one, "
         "with the same energy and deterministic PT correction");
  {
    // A selected space is closed under the flip but not complete in spin:
    // S^2 is near, not at, its value.
    const double s2 = SpinSquared(sel_flip.dets, sel_flip.vectors[0]);
    std::printf("      selected spin_flip +1: S^2 = %.2e\n", s2);
    Expect(std::abs(s2) < 0.05, "selected spin_flip +1 vector has S^2 near 0");
  }
  {
    const std::string ss =
        ", \"pt\": {\"method\": \"semistochastic\", \"epsilon2\": 1e-7, "
        "\"epsilon2_large\": 3e-4, \"target_error\": 2e-6, "
        "\"num_samples\": 100, \"estimator\": \"walkers\"}";
    const SpinFlipRun stochastic =
        RunSpinFlipCase(big, base + "/sel_ss", hf, 1, 1, 2e-3, ss, 6);
    const double diff = stochastic.energies[0] - sel_plain.energies[0];
    std::printf("      semistochastic - deterministic: %.2e\n", diff);
    Expect(std::abs(diff) < 1e-5,
           "spin_flip semistochastic PT agrees with the deterministic one");
    // The pooled estimator, without and with spin_flip.
    const std::string pooled =
        ", \"pt\": {\"method\": \"semistochastic\", \"epsilon2\": 1e-7, "
        "\"epsilon2_large\": 3e-4, \"target_error\": 2e-6, "
        "\"num_samples\": 400, \"estimator\": \"pooled\"}";
    double worst = 0.0;
    for (int flip : {0, 1}) {
      const SpinFlipRun run =
          RunSpinFlipCase(big, base + "/sel_pooled" + std::to_string(flip), hf,
                          1, flip, 2e-3, pooled, 6);
      const double d = run.energies[0] - sel_plain.energies[0];
      std::printf("      pooled (spin_flip %d) - deterministic: %.2e\n", flip,
                  d);
      worst = std::max(worst, std::abs(d));
    }
    Expect(worst < 1e-5,
           "pooled semistochastic PT agrees with the deterministic one, "
           "without and with spin_flip");
  }
  {
    RunSpinFlipCase(big, base + "/sel_resume", hf, 1, 1, 2e-3, det_pt, 3,
                    ", \"checkpoint\": true");
    const SpinFlipRun resumed =
        RunSpinFlipCase(big, base + "/sel_resume", hf, 1, 1, 2e-3, det_pt, 6,
                        ", \"checkpoint\": true, \"resume\": true");
    Expect(std::abs(resumed.energies[0] - sel_flip.energies[0]) < 1e-10 &&
               resumed.dets.size() == sel_flip.dets.size(),
           "spin_flip selection resumed from its checkpoint ends as an "
           "uninterrupted one");
  }
  {
    // Odd spin from an open-shell start.
    const SpinFlipRun sel_odd = RunSpinFlipCase(
        big, base + "/sel_odd", "[[0,1,2,3,4,5,6,9]]", 1, -1, 2e-3, "", 6);
    const double s2 = SpinSquared(sel_odd.dets, sel_odd.vectors[0]);
    std::printf("      selected spin_flip -1: S^2 = %.4f, %zu determinants\n",
                s2, sel_odd.dets.size());
    Expect(std::abs(s2 - 2.0) < 0.05 && sel_odd.dets.size() > 1000,
           "selected spin_flip -1 vector has S^2 near 2");
  }
  std::system(("rm -rf " + base).c_str());
}

// ---------------------------------------------------------------------------
// epsilon2_large and num_samples chosen by the program (pt.h, "auto").

// The pieces of the faster PT2 kernels: the bound that screens singles, the
// diagonal element carried from the parent, the hashed set of the space.
void TestPtKernels(const Integrals<Complex>& complex_ints) {
  const Integrals<double> ints = RandomSpatialIntegrals(12, 5);
  std::mt19937 rng(9);
  double worst_bound = -1.0, worst_energy = 0.0;
  size_t num_singles = 0, num_doubles = 0;
  std::vector<Det<1>> dets;
  for (int trial = 0; trial < 200; ++trial) {
    // Random determinant of 4 alpha and 4 beta electrons in 12 orbitals.
    Det<1> det;
    for (int spin = 0; spin < 2; ++spin) {
      int placed = 0;
      while (placed < 4) {
        const int p = 2 * (rng() % 12) + spin;
        if (!det.Occupied(p)) {
          det.Set(p);
          ++placed;
        }
      }
    }
    dets.push_back(det);
    std::vector<int> occ;
    det.AppendOccupied(&occ);
    for (int i : occ) {
      for (int a = 0; a < ints.num_orbs(); ++a) {
        if (det.Occupied(a)) continue;
        Det<1> other = det;
        other.Clear(i);
        other.Set(a);
        const double element = std::abs(Hij(other, det, ints));
        worst_bound = std::max(worst_bound, element - ints.SingleBound(i, a));
      }
    }
    auto visit = [&](const Det<1>& d, double, bool, double energy) {
      worst_energy =
          std::max(worst_energy, std::abs(energy - DiagonalEnergy(d, ints)));
      ++(d.NumDifferentBits(det) == 2 ? num_singles : num_doubles);
    };
    const HeatBath heat_bath(ints, 1e-14);
    const LabelFilter filter(std::vector<int>(ints.num_orbs(), 0), 0);
    internal::ElementEnergySink<1, decltype(visit), double> sink(det, occ, ints,
                                                                 0.0, visit);
    ForEachExcitation(det, occ, 1e-12, ints, heat_bath, filter, &sink);
  }
  std::printf(
      "      single elements exceed their bound by at most %.1e; "
      "diagonal elements from the parent off by %.1e (%zu singles, "
      "%zu doubles)\n",
      worst_bound, worst_energy, num_singles, num_doubles);
  Expect(worst_bound <= 0.0 && worst_energy < 1e-11 && num_singles > 100 &&
             num_doubles > 1000,
         "singles stay within their screening bound and the diagonal "
         "elements carried from the parent are exact");
  // Complex spinor integrals: the bound holds too.
  {
    const int n = complex_ints.num_orbs();
    const int nelec = complex_ints.num_elec();
    double worst = -1.0;
    for (int trial = 0; trial < 50; ++trial) {
      Det<1> det;
      int placed = 0;
      while (placed < nelec) {
        const int p = rng() % n;
        if (!det.Occupied(p)) {
          det.Set(p);
          ++placed;
        }
      }
      std::vector<int> occ;
      det.AppendOccupied(&occ);
      for (int i : occ) {
        for (int a = 0; a < n; ++a) {
          if (det.Occupied(a)) continue;
          Det<1> other = det;
          other.Clear(i);
          other.Set(a);
          worst = std::max(worst, std::abs(Hij(other, det, complex_ints)) -
                                      complex_ints.SingleBound(i, a));
        }
      }
    }
    Expect(worst <= 0.0,
           "the singles bound holds for complex spinor integrals");
  }
  // The hashed set against the sorted list.
  {
    std::sort(dets.begin(), dets.end());
    dets.erase(std::unique(dets.begin(), dets.end()), dets.end());
    std::vector<Det<1>> half(dets.begin(), dets.begin() + dets.size() / 2);
    const DetSet<1> set(MakeSpan(half), Comm());
    std::sort(half.begin(), half.end());
    bool same = true;
    for (const Det<1>& d : dets) {
      same = same &&
             set.Contains(d) == std::binary_search(half.begin(), half.end(), d);
    }
    Expect(same, "the hashed set of determinants answers as the sorted list");
  }
}

void TestPtAuto() {
  const std::string base =
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
      "/zshci_test_ptauto";
  std::system(("rm -rf " + base).c_str());
  const Integrals<double> ints = RandomSpatialIntegrals(12, 11);
  const int n = ints.num_orbs();
  const HeatBath heat_bath(ints, 1e-12);
  const LabelFilter filter(std::vector<int>(n, 0), 0);
  const Comm serial;
  for (int flip_mode : {0, 1}) {
    const std::string tag = flip_mode ? " (spin_flip)" : " (plain)";
    const SpinFlipRun run =
        RunSpinFlipCase(ints, base + "/sel" + std::to_string(flip_mode),
                        "[[0,1,2,3,4,5,6,7]]", 1, flip_mode, 2e-3, "", 6);
    std::vector<Det<1>> sorted = run.dets;
    std::sort(sorted.begin(), sorted.end());
    SpinFlipLayout layout(1, 8);
    if (flip_mode) layout.Extend(MakeSpan(run.dets), 0);
    const double energy = run.energies[0] - ints.core_energy();
    PtProblem<1, LabelFilter, double> problem{
        run.dets,       sorted,
        run.vectors[0], energy,
        ints,           heat_bath,
        filter,         1,
        serial,         flip_mode ? &layout : nullptr};

    // Counting against the terms themselves: exact from every determinant,
    // within its error from every 7th.
    {
      const std::vector<double> thresholds = {1e-3, 1e-4, 3e-5};
      double worst_exact = 0.0, worst_sigmas = 0.0;
      bool errors_positive = true;
      for (double t : thresholds) {
        double terms = 0.0;
        for (size_t i = 0; i < run.dets.size(); ++i) {
          const double c = std::abs(run.vectors[0][i]);
          if (c == 0.0 || internal::GeneratorWeight(problem, i) == 0.0) {
            continue;
          }
          std::vector<int> occ;
          run.dets[i].AppendOccupied(&occ);
          auto visit = [&](const Det<1>& det, double, bool) {
            Det<1> canonical = det;
            if (flip_mode && internal::FoldOntoCanonical(det, layout.lambda(),
                                                         &canonical) == 0.0) {
              return;
            }
            terms += 1.0;
          };
          internal::ElementSink<1, decltype(visit), double> sink(
              run.dets[i], ints, 0.0, visit);
          ForEachExcitation(run.dets[i], occ, t / c, ints, heat_bath, filter,
                            &sink);
        }
        const TermCount all = CountTerms(problem, t, 1, 0);
        const TermCount some = CountTerms(problem, t, 7, 3);
        worst_exact =
            std::max(worst_exact, std::abs(all.terms - terms) / terms);
        worst_sigmas = std::max(worst_sigmas, std::abs(some.terms - terms) /
                                                  std::max(some.error, 1e-300));
        errors_positive = errors_positive && some.error > 0.0;
      }
      std::printf(
          "      counting%s: exact off by %.1e (relative), sampled "
          "1 in 7 off by %.1f standard errors at most\n",
          tag.c_str(), worst_exact, worst_sigmas);
      Expect(worst_exact < 1e-9 && worst_sigmas < 4.0 && errors_positive,
             "terms counted from all determinants equal those generated, "
             "and from every 7th within 4 standard errors" +
                 tag);
    }

    // Batches and the guard do not change the deterministic sum.
    {
      const double one = PtDeterministic(problem, 1e-5);
      PtProblem<1, LabelFilter, double> batched = problem;
      batched.num_batches = 3;
      const double three = PtDeterministic(batched, 1e-5);
      internal::SumStats stats;
      const double guarded =
          PtDeterministic(problem, 1e-5, nullptr, /*max_terms=*/2000, &stats);
      const int slices = stats.slices;
      const double diff =
          std::max(std::abs(three - one), std::abs(guarded - one));
      std::printf(
          "      batches%s: 3 passes and %d guarded slices differ by "
          "%.1e\n",
          tag.c_str(), slices, diff);
      Expect(diff < 1e-12 * std::abs(one) && slices > 1,
             "the deterministic sum in 3 passes and in slices split by the "
             "memory guard equals that in one pass" +
                 tag);
    }

    // auto against the chosen values given explicitly, same seed.
    {
      PtInput pt;
      pt.method = PtMethod::kSemistochastic;
      pt.epsilon2 = 1e-7;
      pt.auto_large = true;
      pt.auto_samples = true;
      pt.pooled = true;
      pt.memory_gb = 0.05;  // several candidates, the smaller in passes
      pt.min_iterations = 8;
      pt.max_iterations = 8;
      pt.target_error = 1e-12;
      pt.pilot_samples = 200;
      pt.pilot_batches = 4;
      pt.max_samples = 5000;
      const DeterminantSampler sampler(problem.coefficients, serial,
                                       problem.flip);
      const PtAutoChoice choice = ChoosePtParameters(problem, pt, sampler, 0);
      const PtResult automatic = RunPerturbation(problem, pt, 0, false);
      PtInput fixed = pt;
      fixed.auto_large = false;
      fixed.auto_samples = false;
      fixed.epsilon2_large = choice.epsilon_large;
      fixed.num_samples = choice.num_samples;
      const PtResult given = RunPerturbation(problem, fixed, 0, false);
      const double det_diff =
          std::abs(PtDeterministic(problem, choice.epsilon_large) - [&] {
            PtProblem<1, LabelFilter, double> b = problem;
            b.num_batches = choice.det_batches;
            return PtDeterministic(b, choice.epsilon_large);
          }());
      std::printf(
          "      auto%s: epsilon2_large %.1e, %d samples, %d "
          "pass(es); auto %.12f, given %.12f\n",
          tag.c_str(), choice.epsilon_large, choice.num_samples,
          choice.det_batches, automatic.correction, given.correction);
      Expect(choice.det_batches == 1 &&
                 automatic.num_iterations >= pt.min_iterations &&
                 automatic.error > 0.0,
             "the chosen deterministic part fits one pass and the stochastic "
             "part runs with an error bar" +
                 tag);
      Expect(det_diff < 1e-12 * std::abs(automatic.correction) &&
                 std::abs(automatic.correction - given.correction) <
                     1e-10 * std::abs(given.correction) &&
                 choice.epsilon_large > pt.epsilon2,
             "PT with epsilon2_large and num_samples chosen by the program "
             "equals PT with the chosen values given, same seed" +
                 tag);
      // With room and a tight target the stochastic part dominates and a
      // small epsilon2_large must win (regression: the choice once stopped
      // at the largest candidate).
      PtInput roomy = pt;
      roomy.memory_gb = 1.0;
      roomy.target_error = 1e-7;
      const PtAutoChoice small = ChoosePtParameters(problem, roomy, sampler, 0);
      std::printf("      auto%s with room: epsilon2_large %.1e\n", tag.c_str(),
                  small.epsilon_large);
      Expect(small.epsilon_large <= 1e-5,
             "with memory to spare a small epsilon2_large is chosen" + tag);

      // The thresholds visited start at epsilon1 / 10 and never go below
      // epsilon2; when epsilon2 itself fits, all is deterministic.
      PtInput start = pt;
      start.start_threshold = 2e-3;
      start.epsilon2 = 2e-5;
      const PtAutoChoice bounded =
          ChoosePtParameters(problem, start, sampler, 0);
      PtInput all_fits = start;
      all_fits.memory_gb = 4.0;
      const PtAutoChoice whole =
          ChoosePtParameters(problem, all_fits, sampler, 0);
      const PtResult whole_pt = RunPerturbation(problem, all_fits, 0, false);
      const double exact = PtDeterministic(problem, all_fits.epsilon2);
      std::printf(
          "      auto%s from epsilon1 2e-3: epsilon2_large %.2e; with "
          "4 GB: deterministic only %d, %.12f against %.12f\n",
          tag.c_str(), bounded.epsilon_large, whole.deterministic_only,
          whole_pt.correction, exact);
      Expect(
          bounded.epsilon_large >= start.epsilon2 &&
              bounded.epsilon_large <= 2e-4 && whole.deterministic_only &&
              whole_pt.error == 0.0 &&
              std::abs(whole_pt.correction - exact) < 1e-12 * std::abs(exact),
          "the thresholds stay between epsilon1/10 and epsilon2, and with "
          "room for epsilon2 the correction is deterministic" +
              tag);
    }
  }
  std::system(("rm -rf " + base).c_str());
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
    zshci::TestBalanceWithin();
    zshci::TestHamiltonian(ints);
    zshci::TestFewVectorSolver(ints);
    zshci::TestSelectSolvers(ints);
    zshci::TestDensityMatricesFromDefinition(ints);
    zshci::TestPerturbation(ints);
    zshci::TestSecondOrderLimit(ints);
    zshci::TestGroupConnections(ints);
    {
      // Labels as of an axis: values -3, -1, 1, 3 in turn by pairs.
      std::vector<int> axis(44);
      for (int p = 0; p < 44; ++p) axis[p] = 2 * ((p / 2) % 4) - 3;
      zshci::TestByteConnections<1>(
          44, 14, 0, 900, 3, {},
          "related strings composed from the byte table give the connections "
          "of 14 electrons in 44 spinors");
      zshci::TestByteConnections<1>(
          44, 14, 0, 4000, 3, axis,
          "the same for determinants of one symmetry label");
      zshci::TestByteConnections<2>(
          44, 14, 46, 900, 3, {},
          "the same with two-word determinants, in spinors 46 to 89");
      zshci::TestByteConnections<2>(100, 20, 0, 900, 4, {},
                                    "the same for 20 electrons in 100 spinors");
    }
    zshci::TestPtScoreSelection(ints);
    zshci::TestRealScalar(ints);
    zshci::TestGasCi<double>(ints);
    zshci::TestGasCi<zshci::Complex>(ints);
    zshci::TestBlockedIntegrals(ints);
    zshci::TestSpatialIntegrals();
    zshci::TestSpinFlip();
    zshci::TestPtKernels(ints);
    zshci::TestPtAuto();
    zshci::TestSpinStringConnections<1>(
        10, 0,
        "group string tables find the connected pairs of spin-conserving "
        "determinants");
    zshci::TestSpinStringConnections<2>(
        10, 27, "group string tables find the same with two-word determinants");
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
