// Second-order Epstein-Nesbet perturbation theory on top of a variational
// wave function |psi> = sum_i c_i |D_i> with energy E0.
//
// Deterministic:
//   E2(eps) = sum_a |sum_i' H(a,i) c_i|^2 / (E0 - E_a),
// over the determinants D_a outside the variational space, the inner sum
// being restricted to the terms with |H(a,i) c_i| > eps.
//
// Stochastic: a walker draws N determinants with probability
// p_i = |c_i| / sum|c|, D_i being drawn w_i times. With x_i = H(a,i) c_i,
//   E2(eps) ~ 1/(N(N-1)) sum_a [ |sum_i' w_i x_i / p_i|^2
//             + sum_i' (w_i (N-1) / p_i - w_i^2 / p_i^2) |x_i|^2 ] / (E0 - E_a)
// is an unbiased estimate of the deterministic expression with the same eps.
//
// Semistochastic: E2(eps) = E2(eps_large) + [E2(eps) - E2(eps_large)], the
// first term evaluated deterministically and the bracket stochastically, both
// of its parts from the same samples so that their fluctuations largely
// cancel.
//
// With spin_flip (spin_flip.h) the vector is symmetric under the exchange
// of the spins, c(Dbar) = lambda p(D) c(D), and so is V_a = sum_i H(a,i) c_i:
// V(abar) = lambda p(a) V_a, and the sum over D_a may run over canonical
// determinants A only, each counted twice unless it is its own flip. V_A
// itself comes from the canonical generators D_i alone: the term of the
// partner ibar for A equals lambda p(Abar) times that of i for Abar, so a
// term generated from i for a determinant a counts for its canonical member
// A with the factor
//   1 (a = A != Abar),   lambda p(a) (a != A),   1 + lambda p(a) (a = abar),
// and the terms of a generator that is its own flip with one half (it is
// then reached twice). The stochastic estimator draws the canonical
// generators with probability proportional to |c_i| times their number of
// members and sums, before squaring, what one generator gives to one A.

#ifndef ZSHCI_SRC_PT_H_
#define ZSHCI_SRC_PT_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "comm.h"
#include "det.h"
#include "excite.h"
#include "hamiltonian.h"
#include "heat_bath.h"
#include "input.h"
#include "integrals.h"
#include "parallel.h"
#include "spin_flip.h"

namespace zshci {

namespace internal {

// Sink of ForEachExcitation that evaluates the exact element <D_a|H|D_i> of
// every excitation and calls visit(D_a, element, is_large), is_large telling
// whether the element also passes the larger cutoff.
template <int W, class Visit, typename S>
class ElementSink {
 public:
  ElementSink(const Det<W>& parent, const Integrals<S>& ints,
              double large_cutoff, const Visit& visit)
      : parent_(parent),
        ints_(ints),
        large_cutoff_(large_cutoff),
        visit_(visit) {}

  void Single(int i, int a, S element) {
    Det<W> det = parent_;
    det.Clear(i);
    det.Set(a);
    visit_(det, element, std::abs(element) > large_cutoff_);
  }
  void Double(int i, int j, int a, int b, double magnitude) {
    Det<W> det = parent_;
    det.Clear(i);
    det.Clear(j);
    det.Set(a);
    det.Set(b);
    Excitation exc;
    exc.rank = 2;
    exc.holes[0] = i;
    exc.holes[1] = j;
    exc.particles[0] = a;
    exc.particles[1] = b;
    visit_(det, OffDiagonalElement(parent_, exc, ints_),
           magnitude >= large_cutoff_);
  }

 private:
  const Det<W>& parent_;
  const Integrals<S>& ints_;
  double large_cutoff_;
  const Visit& visit_;
};

// What the deterministic sum accumulates for one external determinant:
// sum_i H(a,i) c_i.
template <int W, typename S>
struct DetTerm {
  Det<W> det;
  S sum = S(0.0);

  void Add(const DetTerm& other) { sum += other.sum; }
};

// What the stochastic estimator accumulates for one external determinant.
// The `large` members hold the part from the terms that pass the larger
// cutoff; the `square` members hold the diagonal sum of the estimator.
template <int W, typename S>
struct SampleTerm {
  Det<W> det;
  S sum = S(0.0);
  S sum_large = S(0.0);
  double square = 0.0;
  double square_large = 0.0;

  void Add(const SampleTerm& other) {
    sum += other.sum;
    sum_large += other.sum_large;
    square += other.square;
    square_large += other.square_large;
  }
};

// Runs generate(parent, emit) for parent = 0 .. num_parents-1, where emit
// takes a Term (DetTerm or SampleTerm); merges the terms of equal determinants; drops the
// determinants of the variational space; and returns the sum over the
// remaining ones of contribution(term, E_a).
//
// Every determinant is owned by one rank and, there, by one part, both
// chosen from its hash: each rank generates the terms of its share of the
// parents, sends them to their owners, and the parts are merged in parallel.
// To bound the memory, the whole procedure is repeated `num_batches` times,
// each pass keeping one slice of the hash range.
template <int W, class Term, class Generate, class Contribution, typename S>
std::vector<double> SumOverExternalDets(
    size_t num_parents, Span<Det<W>> sorted_dets, const Integrals<S>& ints,
    int num_batches, int num_values, const Comm& comm, const Generate& generate,
    const Contribution& contribution) {
  const uint64_t num_ranks = comm.size();
  const uint64_t batches = std::max(1, num_batches);
  // More parts than threads, so that uneven parts even out.
  const uint64_t num_parts = 8 * static_cast<uint64_t>(omp_get_max_threads());
  std::vector<double> total(num_values, 0.0);

  for (uint64_t batch = 0; batch < batches; ++batch) {
    // Generate, one chunk per thread.
    std::vector<std::vector<Term>> chunks(omp_get_max_threads());
#pragma omp parallel
    {
      std::vector<Term>& mine = chunks[omp_get_thread_num()];
      auto emit = [&](const Term& term) {
        if (term.det.Hash() % batches == batch) mine.push_back(term);
      };
      const long count = static_cast<long>(num_parents);
      const long rank = comm.rank();
      const long ranks = comm.size();
#pragma omp for schedule(dynamic, 8)
      for (long parent = 0; parent < count; ++parent) {
        if (parent % ranks == rank) generate(parent, emit);
      }
    }

    // Send every term to the rank that owns its determinant, then group
    // what arrived into parts that one thread each can merge.
    auto part_of = [&](const Term& term) {
      return (term.det.Hash() / batches / num_ranks) % num_parts;
    };
    Partitioned<Term> terms;
    if (num_ranks == 1) {
      terms = PartitionChunks(&chunks, num_parts, part_of);
    } else {
      Partitioned<Term> by_rank =
          PartitionChunks(&chunks, num_ranks, [&](const Term& term) {
            return (term.det.Hash() / batches) % num_ranks;
          });
      const RawArray<Term> received = comm.ExchangeParts(&by_rank);
      terms = PartitionArray(received, num_parts, part_of);
    }

    // Merge and sum.
#pragma omp parallel
    {
      std::vector<double> local(num_values, 0.0);
      std::vector<double> values(num_values);
#pragma omp for schedule(dynamic, 1) nowait
      for (long part = 0; part < static_cast<long>(num_parts); ++part) {
        Term* const first = terms.begin(part);
        Term* const last = terms.end(part);
        std::sort(first, last, [](const Term& a, const Term& b) {
          return a.det < b.det;
        });
        Term* run = first;
        while (run != last) {
          Term merged = *run;
          Term* next = run + 1;
          while (next != last && next->det == merged.det) {
            merged.Add(*next);
            ++next;
          }
          run = next;
          if (std::binary_search(sorted_dets.begin(), sorted_dets.end(),
                                 merged.det)) {
            continue;
          }
          contribution(merged, DiagonalEnergy(merged.det, ints), &values);
          for (int v = 0; v < num_values; ++v) local[v] += values[v];
        }
      }
#pragma omp critical
      for (int v = 0; v < num_values; ++v) total[v] += local[v];
    }
  }
  comm.Sum(total.data(), total.size());
  return total;
}

// With spin_flip: the canonical member of `det` and the factor with which a
// term of det counts for it (see above); lambda as in spin_flip.h.
template <int W>
double FoldOntoCanonical(const Det<W>& det, int lambda, Det<W>* canonical) {
  const Det<W> flipped = FlipSpins(det);
  const int p = NumDoublyOccupied(det) % 2 == 0 ? 1 : -1;
  if (flipped == det) {
    *canonical = det;
    return 1.0 + lambda * p;
  }
  if (det < flipped) {
    *canonical = det;
    return 1.0;
  }
  *canonical = flipped;
  return lambda * p;
}

}  // namespace internal

// Everything the perturbative routines need to know about the wave function.
template <int W, class Filter, typename S>
struct PtProblem {
  Span<Det<W>> dets;
  Span<Det<W>> sorted_dets;
  Span<S> coefficients;
  double energy;  // variational energy, without the core energy
  const Integrals<S>& ints;
  const HeatBath& heat_bath;
  const Filter& filter;
  int num_batches;
  const Comm& comm;
  // With spin_flip: the layout of `dets`, whose coefficients are then those
  // of the expanded vector; the sums use the symmetry (see above).
  const SpinFlipLayout* flip = nullptr;
};

namespace internal {

// The weight of generator i: 0 for a partner, 1/2 for a determinant that is
// its own flip, 1 otherwise (and always 1 without spin_flip).
template <int W, class Filter, typename S>
double GeneratorWeight(const PtProblem<W, Filter, S>& problem, size_t i) {
  if (problem.flip == nullptr) return 1.0;
  if (!problem.flip->is_canonical(i)) return 0.0;
  return problem.flip->is_pair(problem.flip->rep(i)) ? 1.0 : 0.5;
}

// The number of members of the pair of a canonical external determinant.
template <int W>
double ExternalWeight(const Det<W>& det, bool with_flip) {
  return with_flip && FlipSpins(det) != det ? 2.0 : 1.0;
}

}  // namespace internal

// E2(epsilon), deterministically.
template <int W, class Filter, typename S>
double PtDeterministic(const PtProblem<W, Filter, S>& problem, double epsilon) {
  using Term = internal::DetTerm<W, S>;
  const SpinFlipLayout* flip = problem.flip;
  const int lambda = flip != nullptr ? flip->lambda() : 1;
  auto generate = [&](long i, const auto& emit) {
    const S ci = problem.coefficients[i];
    const double g = internal::GeneratorWeight(problem, i);
    if (std::abs(ci) == 0.0 || g == 0.0) return;
    std::vector<int> occ;
    problem.dets[i].AppendOccupied(&occ);
    auto visit = [&](const Det<W>& det, S element, bool /*is_large*/) {
      Term term;
      term.det = det;
      term.sum = element * ci;
      if (flip != nullptr) {
        const double factor =
            g * internal::FoldOntoCanonical(det, lambda, &term.det);
        if (factor == 0.0) return;
        term.sum *= factor;
      }
      emit(term);
    };
    internal::ElementSink<W, decltype(visit), S> sink(problem.dets[i],
                                                      problem.ints, 0.0, visit);
    ForEachExcitation(problem.dets[i], occ, epsilon / std::abs(ci),
                      problem.ints, problem.heat_bath, problem.filter, &sink);
  };
  auto contribution = [&](const Term& term, double energy_a,
                          std::vector<double>* values) {
    (*values)[0] = internal::ExternalWeight(term.det, flip != nullptr) *
                   std::norm(term.sum) / (problem.energy - energy_a);
  };
  return internal::SumOverExternalDets<W, Term>(
             problem.dets.size(), problem.sorted_dets, problem.ints,
             problem.num_batches, 1, problem.comm, generate, contribution)[0];
}

// One stochastic estimate of E2(epsilon) and of E2(epsilon_large), both from
// the same sampled determinants.
struct StochasticEstimate {
  double small = 0.0;  // estimate of E2(epsilon)
  double large = 0.0;  // estimate of E2(epsilon_large)
};

// Draws determinants with probability proportional to |c_i|. Drawing does
// not modify the sampler, so all walkers share one; its table is held once
// per node. Construction is collective.
class DeterminantSampler {
 public:
  // With `flip`, only canonical determinants are drawn, with |c_i| times
  // the number of members of their pair.
  template <typename S>
  DeterminantSampler(Span<S> coefficients, const Comm& comm,
                     const SpinFlipLayout* flip = nullptr)
      : cumulative_(coefficients.size(), comm) {
    if (comm.is_root()) {
      double sum = 0.0;
      for (size_t i = 0; i < coefficients.size(); ++i) {
        double weight = std::abs(coefficients[i]);
        if (flip != nullptr) {
          weight = !flip->is_canonical(i)        ? 0.0
                   : flip->is_pair(flip->rep(i)) ? 2.0 * weight
                                                 : weight;
        }
        sum += weight;
        cumulative_[i] = sum;
      }
    }
    cumulative_.ShareFromRoot(comm);
    total_ = cumulative_.size() == 0 ? 0.0 : cumulative_[cumulative_.size() - 1];
  }

  double Probability(size_t i) const {
    return (cumulative_[i] - (i == 0 ? 0.0 : cumulative_[i - 1])) / total_;
  }

  // Returns the distinct determinants drawn in num_samples draws and how
  // often each was drawn. `draws` is scratch space.
  template <class Rng>
  void Draw(int num_samples, Rng* rng, std::vector<size_t>* draws,
            std::vector<size_t>* indices,
            std::vector<int>* multiplicities) const {
    std::uniform_real_distribution<double> uniform(0.0, total_);
    const double* const begin = cumulative_.data();
    const double* const end = begin + cumulative_.size();
    draws->resize(num_samples);
    for (size_t& draw : *draws) {
      const size_t i = std::upper_bound(begin, end, uniform(*rng)) - begin;
      draw = std::min(i, cumulative_.size() - 1);
    }
    std::sort(draws->begin(), draws->end());
    indices->clear();
    multiplicities->clear();
    for (size_t draw : *draws) {
      if (!indices->empty() && indices->back() == draw) {
        ++multiplicities->back();
      } else {
        indices->push_back(draw);
        multiplicities->push_back(1);
      }
    }
  }

 private:
  SharedArray<double> cumulative_;
  double total_ = 0.0;
};

// Everything a walker writes to. Each walker owns one, created before the
// sampling starts and reused for every sample, so that walkers running as
// threads share no mutable state - not even the memory allocator - and
// behave like walkers running as separate processes.
template <int W, typename S>
struct WalkerWorkspace {
  std::vector<internal::SampleTerm<W, S>> terms;
  std::vector<size_t> draws;
  std::vector<size_t> indices;
  std::vector<int> multiplicities;
  std::vector<int> occ;
  // spin_flip: the terms of one generator, merged before they are squared.
  std::vector<internal::SampleTerm<W, S>> of_generator;
};

// One walker: draws `num_samples` determinants and evaluates the estimator
// from them alone. The terms are generated, merged and summed inside the
// caller's workspace, so any number of walkers run side by side without
// exchanging data, on threads and on ranks alike.
template <int W, class Filter, class Rng, typename S>
StochasticEstimate PtWalkerSample(const PtProblem<W, Filter, S>& problem,
                                  double epsilon, double epsilon_large,
                                  int num_samples,
                                  const DeterminantSampler& sampler, Rng* rng,
                                  WalkerWorkspace<W, S>* workspace) {
  using Term = internal::SampleTerm<W, S>;
  if (num_samples < 2) {
    throw std::invalid_argument("stochastic PT needs at least two samples");
  }
  std::vector<Term>* const terms = &workspace->terms;
  std::vector<size_t>& indices = workspace->indices;
  std::vector<int>& multiplicities = workspace->multiplicities;
  std::vector<int>& occ = workspace->occ;
  sampler.Draw(num_samples, rng, &workspace->draws, &indices, &multiplicities);
  const double n = num_samples;
  const uint64_t batches = std::max(1, problem.num_batches);
  const SpinFlipLayout* flip = problem.flip;
  const int lambda = flip != nullptr ? flip->lambda() : 1;
  std::vector<Term>& of_generator = workspace->of_generator;

  StochasticEstimate estimate;
  for (uint64_t batch = 0; batch < batches; ++batch) {
    terms->clear();
    for (size_t sample = 0; sample < indices.size(); ++sample) {
      const size_t i = indices[sample];
      const S ci = problem.coefficients[i];
      const double g = internal::GeneratorWeight(problem, i);
      const double w = multiplicities[sample];
      const double p = sampler.Probability(i);
      const double linear = w / p;
      const double quadratic = w * (n - 1.0) / p - w * w / (p * p);
      occ.clear();
      problem.dets[i].AppendOccupied(&occ);
      if (flip != nullptr) of_generator.clear();
      auto visit = [&](const Det<W>& det, S element, bool is_large) {
        S x = element * ci;
        Term term;
        term.det = det;
        if (flip != nullptr) {
          const double factor =
              g * internal::FoldOntoCanonical(det, lambda, &term.det);
          if (factor == 0.0) return;
          x *= factor;
        }
        if (batches > 1 && term.det.Hash() % batches != batch) return;
        if (flip != nullptr) {
          // Squared below, once the terms of this generator are summed.
          term.sum = x;
          if (is_large) term.sum_large = x;
          of_generator.push_back(term);
          return;
        }
        term.sum = linear * x;
        term.square = quadratic * std::norm(x);
        if (is_large) {
          term.sum_large = term.sum;
          term.square_large = term.square;
        }
        terms->push_back(term);
      };
      internal::ElementSink<W, decltype(visit), S> sink(
          problem.dets[i], problem.ints, epsilon_large / std::abs(ci), visit);
      ForEachExcitation(problem.dets[i], occ, epsilon / std::abs(ci),
                        problem.ints, problem.heat_bath, problem.filter, &sink);
      if (flip != nullptr) {
        std::sort(of_generator.begin(), of_generator.end(),
                  [](const Term& a, const Term& b) { return a.det < b.det; });
        size_t k = 0;
        while (k < of_generator.size()) {
          Term merged = of_generator[k];
          size_t next = k + 1;
          while (next < of_generator.size() &&
                 of_generator[next].det == merged.det) {
            merged.sum += of_generator[next].sum;
            merged.sum_large += of_generator[next].sum_large;
            ++next;
          }
          k = next;
          Term term;
          term.det = merged.det;
          term.sum = linear * merged.sum;
          term.square = quadratic * std::norm(merged.sum);
          term.sum_large = linear * merged.sum_large;
          term.square_large = quadratic * std::norm(merged.sum_large);
          terms->push_back(term);
        }
      }
    }

    std::sort(terms->begin(), terms->end(),
              [](const Term& a, const Term& b) { return a.det < b.det; });
    size_t run = 0;
    while (run < terms->size()) {
      Term merged = (*terms)[run];
      size_t next = run + 1;
      while (next < terms->size() && (*terms)[next].det == merged.det) {
        merged.Add((*terms)[next]);
        ++next;
      }
      run = next;
      if (std::binary_search(problem.sorted_dets.begin(),
                             problem.sorted_dets.end(), merged.det)) {
        continue;
      }
      const double denominator =
          (problem.energy - DiagonalEnergy(merged.det, problem.ints)) * n *
          (n - 1.0) / internal::ExternalWeight(merged.det, flip != nullptr);
      estimate.small += (std::norm(merged.sum) + merged.square) / denominator;
      estimate.large +=
          (std::norm(merged.sum_large) + merged.square_large) / denominator;
    }
  }
  return estimate;
}

// One estimate from num_samples determinants drawn with `rng` (the same
// stream on every rank, so that all draw the same), the terms of all ranks
// and threads merged as in the deterministic sum. Collective.
template <int W, class Filter, class Rng, typename S>
StochasticEstimate PtPooledSample(const PtProblem<W, Filter, S>& problem,
                                  double epsilon, double epsilon_large,
                                  int num_samples,
                                  const DeterminantSampler& sampler, Rng* rng) {
  using Term = internal::SampleTerm<W, S>;
  std::vector<size_t> draws, indices;
  std::vector<int> multiplicities;
  sampler.Draw(num_samples, rng, &draws, &indices, &multiplicities);
  const double n = num_samples;
  const SpinFlipLayout* flip = problem.flip;
  const int lambda = flip != nullptr ? flip->lambda() : 1;
  auto generate = [&](long sample, const auto& emit) {
    const size_t i = indices[sample];
    const S ci = problem.coefficients[i];
    const double g = internal::GeneratorWeight(problem, i);
    const double w = multiplicities[sample];
    const double p = sampler.Probability(i);
    const double linear = w / p;
    const double quadratic = w * (n - 1.0) / p - w * w / (p * p);
    std::vector<int> occ;
    problem.dets[i].AppendOccupied(&occ);
    // What this generator gives to one determinant is summed before it is
    // squared; with spin_flip two of its terms can fold onto one.
    std::vector<Term> mine;
    auto visit = [&](const Det<W>& det, S element, bool is_large) {
      S x = element * ci;
      Term term;
      term.det = det;
      if (flip != nullptr) {
        const double factor =
            g * internal::FoldOntoCanonical(det, lambda, &term.det);
        if (factor == 0.0) return;
        x *= factor;
      }
      term.sum = x;
      if (is_large) term.sum_large = x;
      mine.push_back(term);
    };
    internal::ElementSink<W, decltype(visit), S> sink(
        problem.dets[i], problem.ints, epsilon_large / std::abs(ci), visit);
    ForEachExcitation(problem.dets[i], occ, epsilon / std::abs(ci),
                      problem.ints, problem.heat_bath, problem.filter, &sink);
    if (flip != nullptr) {
      std::sort(mine.begin(), mine.end(),
                [](const Term& a, const Term& b) { return a.det < b.det; });
    }
    size_t k = 0;
    while (k < mine.size()) {
      Term merged = mine[k];
      size_t next = k + 1;
      while (flip != nullptr && next < mine.size() &&
             mine[next].det == merged.det) {
        merged.sum += mine[next].sum;
        merged.sum_large += mine[next].sum_large;
        ++next;
      }
      k = next;
      Term term;
      term.det = merged.det;
      term.sum = linear * merged.sum;
      term.square = quadratic * std::norm(merged.sum);
      term.sum_large = linear * merged.sum_large;
      term.square_large = quadratic * std::norm(merged.sum_large);
      emit(term);
    }
  };
  auto contribution = [&](const Term& term, double energy_a,
                          std::vector<double>* values) {
    const double denominator =
        (problem.energy - energy_a) * n * (n - 1.0) /
        internal::ExternalWeight(term.det, flip != nullptr);
    (*values)[0] = (std::norm(term.sum) + term.square) / denominator;
    (*values)[1] =
        (std::norm(term.sum_large) + term.square_large) / denominator;
  };
  const std::vector<double> sums = internal::SumOverExternalDets<W, Term>(
      indices.size(), problem.sorted_dets, problem.ints, problem.num_batches, 2,
      problem.comm, generate, contribution);
  StochasticEstimate estimate;
  estimate.small = sums[0];
  estimate.large = sums[1];
  return estimate;
}

struct PtResult {
  double correction = 0.0;  // E2
  double error = 0.0;       // standard error of the mean; 0 if deterministic
  int num_iterations = 0;
  long num_estimates = 0;   // walker samples behind a stochastic result
  bool converged = true;
};

// Runs the method selected in the input for one state.
//
// The stochastic methods use one walker per core: every thread of every rank
// draws its own samples with its own random stream and produces its own
// estimate in each iteration, without communication. The result is the mean
// over all walker estimates, and its error their standard error.
template <int W, class Filter, typename S>
PtResult RunPerturbation(const PtProblem<W, Filter, S>& problem,
                         const PtInput& pt, int root, bool print) {
  PtResult result;
  if (pt.method == PtMethod::kDeterministic) {
    result.correction = PtDeterministic(problem, pt.epsilon2);
    return result;
  }

  double deterministic_part = 0.0;
  double epsilon_large = pt.epsilon2;  // pure stochastic: nothing subtracted
  if (pt.method == PtMethod::kSemistochastic) {
    epsilon_large = pt.epsilon2_large;
    deterministic_part = PtDeterministic(problem, epsilon_large);
    if (print) {
      std::printf("  root %d: deterministic part (epsilon2_large = %.1e): "
                  "%.10f\n",
                  root, epsilon_large, deterministic_part);
    }
  }

  const DeterminantSampler sampler(problem.coefficients, problem.comm,
                                   problem.flip);
  if (pt.pooled) {
    // One estimate per iteration from the samples drawn alike on every rank.
    std::mt19937_64 stream(Mix64(pt.seed) ^
                           Mix64(0x9e3779b97f4a7c15ULL * (root + 1)));
    double sums[3] = {0.0, 0.0, 0.0};
    result.converged = false;
    for (int iter = 1; iter <= pt.max_iterations; ++iter) {
      const StochasticEstimate estimate =
          PtPooledSample(problem, pt.epsilon2, epsilon_large, pt.num_samples,
                         sampler, &stream);
      const double value =
          pt.method == PtMethod::kSemistochastic
              ? deterministic_part + estimate.small - estimate.large
              : estimate.small;
      sums[0] += 1.0;
      sums[1] += value;
      sums[2] += value * value;
      result.num_iterations = iter;
      result.num_estimates = iter;
      result.correction = sums[1] / sums[0];
      if (iter < pt.min_iterations) continue;
      const double variance = std::max(
          0.0, (sums[2] - sums[1] * sums[1] / sums[0]) / (sums[0] - 1.0));
      result.error = std::sqrt(variance / sums[0]);
      if (print) {
        std::printf(
            "  root %d iter %5d  this iteration %16.10f  average "
            "%16.10f  error %.2e\n",
            root, iter, value, result.correction, result.error);
      }
      if (result.error < pt.target_error) {
        result.converged = true;
        break;
      }
    }
    return result;
  }
  const int num_threads = omp_get_max_threads();
  // One random stream per walker, different on every rank and for every
  // root.
  std::vector<std::mt19937_64> streams;
  for (int t = 0; t < num_threads; ++t) {
    const uint64_t walker =
        static_cast<uint64_t>(problem.comm.rank()) * num_threads + t;
    streams.emplace_back(Mix64(pt.seed) ^ Mix64(walker + 1) ^
                         Mix64(0x9e3779b97f4a7c15ULL * (root + 1)));
  }
  std::vector<WalkerWorkspace<W, S>> workspaces(num_threads);

  // sums[0..2]: number of estimates, their sum, the sum of their squares.
  double sums[3] = {0.0, 0.0, 0.0};
  result.converged = false;
  for (int iter = 1; iter <= pt.max_iterations; ++iter) {
    double count = 0.0, sum = 0.0, sum_squares = 0.0;
#pragma omp parallel reduction(+ : count, sum, sum_squares)
    {
      const int t = omp_get_thread_num();
      const StochasticEstimate estimate =
          PtWalkerSample(problem, pt.epsilon2, epsilon_large, pt.num_samples,
                         sampler, &streams[t], &workspaces[t]);
      const double value =
          pt.method == PtMethod::kSemistochastic
              ? deterministic_part + estimate.small - estimate.large
              : estimate.small;
      count += 1.0;
      sum += value;
      sum_squares += value * value;
    }
    double iteration[3] = {count, sum, sum_squares};
    problem.comm.Sum(iteration, 3);
    for (int k = 0; k < 3; ++k) sums[k] += iteration[k];

    result.num_iterations = iter;
    result.num_estimates = static_cast<long>(sums[0]);
    result.correction = sums[1] / sums[0];
    if (sums[0] < 2.0 || iter < pt.min_iterations) continue;
    const double variance = std::max(
        0.0, (sums[2] - sums[1] * sums[1] / sums[0]) / (sums[0] - 1.0));
    result.error = std::sqrt(variance / sums[0]);
    if (print) {
      std::printf("  root %d iter %5d  walkers %4.0f  this iteration %16.10f  "
                  "average %16.10f  error %.2e\n",
                  root, iter, iteration[0], iteration[1] / iteration[0],
                  result.correction, result.error);
    }
    if (result.error < pt.target_error) {
      result.converged = true;
      break;
    }
  }
  return result;
}

}  // namespace zshci

#endif  // ZSHCI_SRC_PT_H_
