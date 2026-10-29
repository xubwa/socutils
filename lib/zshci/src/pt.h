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
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "comm.h"
#include "det.h"
#include "det_set.h"
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
template <int W, class Visit, class Ints, typename S = typename Ints::Scalar>
class ElementSink {
 public:
  ElementSink(const Det<W>& parent, const Ints& ints, double large_cutoff,
              const Visit& visit)
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
  const Ints& ints_;
  double large_cutoff_;
  const Visit& visit_;
};

// As ElementSink, and visit(D_a, element, is_large, <D_a|H|D_a>): the
// diagonal element follows from that of the parent in a few operations,
// with the sums row[p] = sum over the occupied k != p of the pair energies
// <pk||pk> made once per parent.
template <int W, class Visit, class Ints, typename S = typename Ints::Scalar>
class ElementEnergySink {
 public:
  ElementEnergySink(const Det<W>& parent, const std::vector<int>& occ,
                    const Ints& ints, double large_cutoff, const Visit& visit)
      : parent_(parent),
        ints_(ints),
        large_cutoff_(large_cutoff),
        visit_(visit),
        energy_(DiagonalEnergy(parent, ints)),
        row_(ints.num_orbs(), 0.0) {
    for (int p = 0; p < ints.num_orbs(); ++p) {
      double sum = 0.0;
      for (int k : occ) sum += ints.PairEnergy(p, k);  // <pp||pp> = 0
      row_[p] = sum;
    }
  }

  void Single(int i, int a, S element) {
    Det<W> det = parent_;
    det.Clear(i);
    det.Set(a);
    const double energy =
        energy_ - H(i) - row_[i] + H(a) + row_[a] - ints_.PairEnergy(a, i);
    visit_(det, element, std::abs(element) > large_cutoff_, energy);
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
    const double energy = energy_ - H(i) - H(j) - row_[i] - row_[j] +
                          ints_.PairEnergy(i, j) + H(a) + H(b) + row_[a] -
                          ints_.PairEnergy(a, i) - ints_.PairEnergy(a, j) +
                          row_[b] - ints_.PairEnergy(b, i) -
                          ints_.PairEnergy(b, j) + ints_.PairEnergy(a, b);
    visit_(det, OffDiagonalElement(parent_, exc, ints_),
           magnitude >= large_cutoff_, energy);
  }

 private:
  double H(int p) const { return Real(ints_.OneBody(p, p)); }

  const Det<W>& parent_;
  const Ints& ints_;
  double large_cutoff_;
  const Visit& visit_;
  double energy_;
  std::vector<double> row_;
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
  // <D_a|H|D_a>, worked out when the term is made (pooled estimator).
  double energy = 0.0;

  void Add(const SampleTerm& other) {
    sum += other.sum;
    sum_large += other.sum_large;
    square += other.square;
    square_large += other.square_large;
  }
};

// The diagonal element of the determinant of a term: carried by the terms
// of the pooled estimator, worked out for the others.
template <int W, class Ints, typename S = typename Ints::Scalar, class Term>
double TermEnergy(const Term& term, const Ints& ints) {
  return DiagonalEnergy(term.det, ints);
}
template <int W, class Ints, typename S = typename Ints::Scalar>
double TermEnergy(const SampleTerm<W, S>& term, const Ints&) {
  return term.energy;
}

// Buffers of SumOverExternalDets kept from one call to the next (the
// batches of the stochastic part), so that their pages are not faulted in
// again every time.
template <class Term>
struct SumWorkspace {
  std::vector<std::vector<Term>> chunks;
  Partitioned<Term> parts;
};

// Whether a determinant is in the variational space: by the hashed set if
// there is one, else by bisection in the sorted list.
template <int W>
bool InSpace(const Det<W>& det, Span<Det<W>> sorted_dets,
             const DetSet<W>* set) {
  if (set != nullptr) return set->Contains(det);
  return std::binary_search(sorted_dets.begin(), sorted_dets.end(), det);
}

// Runs generate(parent, emit) for parent = 0 .. num_parents-1, where emit
// takes a Term (DetTerm or SampleTerm); merges the terms of equal determinants;
// drops the determinants of the variational space; and returns the sum over the
// remaining ones of contribution(term, E_a).
//
// Every determinant is owned by one rank and, there, by one part, both
// chosen from its hash: each rank generates the terms of its share of the
// parents, sends them to their owners, and the parts are merged in parallel.
// To bound the memory, the whole procedure is repeated `num_batches` times,
// each pass keeping one slice of the hash range (hash % num_batches).
//
// With max_terms > 0 a rank stops storing the terms of a slice once it
// holds more than that many (the guard of the memory estimates, pt_auto in
// RunPerturbation); the slice is then split in two (hash % 2m == r and
// r + m) and done again, so that an estimate that was too low costs time
// instead of the run. `stats`, if given, receives the number of slices
// done, the terms generated (with repetitions) and the distinct external
// determinants outside the variational space, over all ranks.
struct SumStats {
  int slices = 0;
  double terms = 0.0;
  double distinct = 0.0;
};

template <int W, class Term, class Generate, class Contribution, class Ints,
          typename S = typename Ints::Scalar>
std::vector<double> SumOverExternalDets(
    size_t num_parents, Span<Det<W>> sorted_dets, const Ints& ints,
    int num_batches, int num_values, const Comm& comm, const Generate& generate,
    const Contribution& contribution, uint64_t max_terms = 0,
    SumStats* stats = nullptr, const DetSet<W>* space = nullptr,
    SumWorkspace<Term>* workspace = nullptr) {
  const uint64_t num_ranks = comm.size();
  const uint64_t batches = std::max(1, num_batches);
  // More parts than threads, so that uneven parts even out.
  const uint64_t num_parts = 8 * static_cast<uint64_t>(omp_get_max_threads());
  std::vector<double> total(num_values, 0.0);
  // Slices still to do: (modulus, residue).
  std::vector<std::pair<uint64_t, uint64_t>> todo;
  for (uint64_t b = batches; b-- > 0;) todo.emplace_back(batches, b);
  int num_slices = 0;
  double num_terms = 0.0, num_distinct = 0.0;

  while (!todo.empty()) {
    const uint64_t modulus = todo.back().first;
    const uint64_t residue = todo.back().second;
    todo.pop_back();
    // Generate, one chunk per thread.
    std::vector<std::vector<Term>> own_chunks;
    std::vector<std::vector<Term>>& chunks =
        workspace != nullptr ? workspace->chunks : own_chunks;
    chunks.resize(omp_get_max_threads());
    for (std::vector<Term>& chunk : chunks) chunk.clear();
    uint64_t stored = 0;
    bool full = false;
#pragma omp parallel
    {
      std::vector<Term>& mine = chunks[omp_get_thread_num()];
      uint64_t unreported = 0;
      auto emit = [&](const Term& term) {
        if (term.det.Hash() % modulus != residue) return;
        mine.push_back(term);
        if (max_terms > 0 && ++unreported == 4096) {
          uint64_t now;
#pragma omp atomic capture
          now = stored += unreported;
          unreported = 0;
          if (now > max_terms) {
#pragma omp atomic write
            full = true;
          }
        }
      };
      const long count = static_cast<long>(num_parents);
      const long rank = comm.rank();
      const long ranks = comm.size();
#pragma omp for schedule(dynamic, 8)
      for (long parent = 0; parent < count; ++parent) {
        bool stop;
#pragma omp atomic read
        stop = full;
        if (!stop && parent % ranks == rank) generate(parent, emit);
      }
    }
    if (max_terms > 0 && modulus < (uint64_t{1} << 20)) {
      double any = full ? 1.0 : 0.0;
      comm.Sum(&any, 1);
      if (any > 0.0) {
        for (std::vector<Term>& chunk : chunks) std::vector<Term>().swap(chunk);
        todo.emplace_back(2 * modulus, residue + modulus);
        todo.emplace_back(2 * modulus, residue);
        if (comm.is_root()) {
          std::printf(
              "note: a slice of the external determinants held more "
              "than the %llu terms estimated per rank; split in two "
              "(1/%llu of the hash range each)\n",
              static_cast<unsigned long long>(max_terms),
              static_cast<unsigned long long>(2 * modulus));
        }
        continue;
      }
    }
    ++num_slices;
    for (const std::vector<Term>& chunk : chunks) num_terms += chunk.size();

    // Send every term to the rank that owns its determinant, then group
    // what arrived into parts that one thread each can merge.
    auto part_of = [&](const Term& term) {
      return (term.det.Hash() / modulus / num_ranks) % num_parts;
    };
    Partitioned<Term> own_terms;
    Partitioned<Term>& terms =
        workspace != nullptr ? workspace->parts : own_terms;
    if (num_ranks == 1 && workspace != nullptr) {
      // The chunks keep their memory for the next call.
      std::vector<Span<Term>> spans;
      for (const std::vector<Term>& chunk : chunks) spans.emplace_back(chunk);
      PartitionSpansInto(spans, num_parts, part_of, &terms);
    } else if (num_ranks == 1) {
      terms = PartitionChunks(&chunks, num_parts, part_of);
    } else {
      Partitioned<Term> by_rank =
          PartitionChunks(&chunks, num_ranks, [&](const Term& term) {
            return (term.det.Hash() / modulus) % num_ranks;
          });
      const RawArray<Term> received = comm.ExchangeParts(&by_rank);
      terms = PartitionArray(received, num_parts, part_of);
    }

    // Merge and sum.
#pragma omp parallel reduction(+ : num_distinct)
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
          if (InSpace(merged.det, sorted_dets, space)) continue;
          num_distinct += 1.0;
          contribution(merged, TermEnergy<W>(merged, ints), &values);
          for (int v = 0; v < num_values; ++v) local[v] += values[v];
        }
      }
#pragma omp critical
      for (int v = 0; v < num_values; ++v) total[v] += local[v];
    }
  }
  if (stats != nullptr) {
    double counts[2] = {num_terms, num_distinct};
    comm.Sum(counts, 2);
    stats->slices = num_slices;
    stats->terms = counts[0];
    stats->distinct = counts[1];
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

namespace internal {

// ---------------------------------------------------------------------------
// Profile of one sum over external determinants ("pt": {"profile": true}).
//
// The same generate and contribution as SumOverExternalDets, in one batch,
// with every step timed on its own: the singles of the excitation generator
// (whose elements it evaluates for every candidate), the walk of the
// heat-bath lists for the doubles, the Slater-Condon elements of the doubles
// with the fold onto canonical determinants (spin_flip), storing the terms,
// partitioning by owner, the exchange between ranks, sorting, merging equal
// determinants, removing those of the variational space, the denominators
// and the sum. The first three are measured by generating again with sinks
// that stop after the step. Times: thread-seconds (busy, summed over all
// threads and ranks) and wall seconds (largest rank).

struct SumProfile {
  std::vector<std::string> steps;
  std::vector<double> busy;  // thread-seconds
  std::vector<double> wall;
  double generated = 0.0, distinct = 0.0, external = 0.0;
  double singles = 0.0, doubles = 0.0;
  double bytes_partitioned = 0.0, bytes_exchanged = 0.0;
  size_t term_bytes = 0;

  void Add(const std::string& step, double thread_seconds, double seconds) {
    steps.push_back(step);
    busy.push_back(thread_seconds);
    wall.push_back(seconds);
  }
};

template <class S>
struct CountSink {
  double singles = 0.0, doubles = 0.0;
  void Single(int, int, S) { singles += 1.0; }
  void Double(int, int, int, int, double) { doubles += 1.0; }
};

// Sink of the third step: the elements of the doubles and the fold.
template <int W, class Ints, typename S = typename Ints::Scalar>
struct ElementOnlySink {
  const Det<W>& parent;
  const Ints& ints;
  const SpinFlipLayout* flip;
  int lambda;
  double sum = 0.0;
  void Single(int, int, S element) { sum += std::abs(element); }
  void Double(int i, int j, int a, int b, double /*magnitude*/) {
    Excitation exc;
    exc.rank = 2;
    exc.holes[0] = i;
    exc.holes[1] = j;
    exc.particles[0] = a;
    exc.particles[1] = b;
    S value = OffDiagonalElement(parent, exc, ints);
    if (flip != nullptr) {
      Det<W> det = parent, canonical;
      det.Clear(i);
      det.Clear(j);
      det.Set(a);
      det.Set(b);
      value *= FoldOntoCanonical(det, lambda, &canonical);
      sum += static_cast<double>(canonical.words[0] & 1);
    }
    sum += std::abs(value);
  }
};

// parent(p, &i, &cutoff) gives the determinant of parent p and the cutoff
// on |H(a,i)| (false: the parent generates nothing).
template <int W, class Term, class Parent, class Generate, class Contribution,
          class Filter, class Ints, typename S = typename Ints::Scalar>
std::vector<double> ProfileSumOverExternalDets(
    size_t num_parents, Span<Det<W>> dets, Span<Det<W>> sorted_dets,
    const Ints& ints, const HeatBath& heat_bath, const Filter& filter,
    const SpinFlipLayout* flip, const DetSet<W>* space, int num_values,
    const Comm& comm, const Parent& parent, const Generate& generate,
    const Contribution& contribution, SumProfile* profile) {
  const int num_threads = omp_get_max_threads();
  const long count = static_cast<long>(num_parents);
  const long rank = comm.rank(), ranks = comm.size();
  const int lambda = flip != nullptr ? flip->lambda() : 1;
  profile->term_bytes = sizeof(Term);
  std::vector<double> busy(num_threads);
  // Runs body(p) over this rank's parents; returns the wall seconds and adds
  // the busy thread-seconds to *thread_seconds.
  auto timed_loop = [&](const auto& body, double* thread_seconds) {
    std::fill(busy.begin(), busy.end(), 0.0);
    const double start = omp_get_wtime();
#pragma omp parallel
    {
      const int t = omp_get_thread_num();
#pragma omp for schedule(dynamic, 8)
      for (long p = 0; p < count; ++p) {
        if (p % ranks != rank) continue;
        const double t0 = omp_get_wtime();
        body(p, t);
        busy[t] += omp_get_wtime() - t0;
      }
    }
    *thread_seconds = 0.0;
    for (double b : busy) *thread_seconds += b;
    return omp_get_wtime() - start;
  };
  std::vector<double> step_busy, step_wall;
  auto record = [&](double thread_seconds, double wall) {
    step_busy.push_back(thread_seconds);
    step_wall.push_back(wall);
  };
  // Steps 1-3: generation stopped after singles, after the doubles' walk,
  // after their elements.
  std::vector<double> singles(num_threads, 0.0), doubles(num_threads, 0.0),
      dummy(num_threads, 0.0);
  double ts = 0.0, wall = 0.0;
  for (int variant = 0; variant < 3; ++variant) {
    wall = timed_loop(
        [&](long p, int t) {
          size_t i;
          double cutoff;
          if (!parent(p, &i, &cutoff)) return;
          std::vector<int> occ;
          dets[i].AppendOccupied(&occ);
          if (variant == 0) {
            CountSink<S> sink;
            ForEachExcitation(dets[i], occ, cutoff, ints, heat_bath, filter,
                              &sink, /*doubles=*/false);
          } else if (variant == 1) {
            CountSink<S> sink;
            ForEachExcitation(dets[i], occ, cutoff, ints, heat_bath, filter,
                              &sink);
            singles[t] += sink.singles;
            doubles[t] += sink.doubles;
          } else {
            ElementOnlySink<W, Ints> sink{dets[i], ints, flip, lambda};
            ForEachExcitation(dets[i], occ, cutoff, ints, heat_bath, filter,
                              &sink);
            dummy[t] += sink.sum;
          }
        },
        &ts);
    record(ts, wall);
  }
  for (int t = 0; t < num_threads; ++t) {
    profile->singles += singles[t];
    profile->doubles += doubles[t];
  }
  // Step 4: the real generation, terms stored.
  std::vector<std::vector<Term>> chunks(num_threads);
  wall = timed_loop(
      [&](long p, int t) {
        generate(p, [&](const Term& term) { chunks[t].push_back(term); });
      },
      &ts);
  record(ts, wall);
  double generated = 0.0;
  for (const std::vector<Term>& chunk : chunks) generated += chunk.size();
  // Step 5: waiting for the slowest rank; step 6: partition by owner and
  // exchange; then the parts.
  double start = omp_get_wtime();
  comm.Barrier();
  record(0.0, omp_get_wtime() - start);
  const uint64_t num_parts = 8 * static_cast<uint64_t>(num_threads);
  auto part_of = [&](const Term& term) {
    return (term.det.Hash() / ranks) % num_parts;
  };
  Partitioned<Term> terms;
  start = omp_get_wtime();
  double exchanged = 0.0;
  if (ranks == 1) {
    terms = PartitionChunks(&chunks, num_parts, part_of);
    record(0.0, omp_get_wtime() - start);
    record(0.0, 0.0);
  } else {
    Partitioned<Term> by_rank = PartitionChunks(
        &chunks, ranks,
        [&](const Term& term) { return term.det.Hash() % ranks; });
    exchanged = (generated - by_rank.size(rank)) * sizeof(Term);
    record(0.0, omp_get_wtime() - start);
    start = omp_get_wtime();
    const RawArray<Term> received = comm.ExchangeParts(&by_rank);
    const double exchange_wall = omp_get_wtime() - start;
    start = omp_get_wtime();
    terms = PartitionArray(received, num_parts, part_of);
    step_wall.back() += omp_get_wtime() - start;
    record(0.0, exchange_wall);
  }
  // Steps 8-12, timed per part.
  std::vector<double> t_sort(num_threads, 0.0), t_merge(num_threads, 0.0),
      t_space(num_threads, 0.0), t_diag(num_threads, 0.0),
      t_sum(num_threads, 0.0);
  std::vector<double> total(num_values, 0.0);
  double distinct = 0.0, external = 0.0;
  start = omp_get_wtime();
#pragma omp parallel reduction(+ : distinct, external)
  {
    const int t = omp_get_thread_num();
    std::vector<double> local(num_values, 0.0), values(num_values);
    std::vector<Term> merged_terms;
#pragma omp for schedule(dynamic, 1) nowait
    for (long part = 0; part < static_cast<long>(num_parts); ++part) {
      Term* const first = terms.begin(part);
      Term* const last = terms.end(part);
      double t0 = omp_get_wtime();
      std::sort(first, last,
                [](const Term& a, const Term& b) { return a.det < b.det; });
      double t1 = omp_get_wtime();
      t_sort[t] += t1 - t0;
      merged_terms.clear();
      Term* run = first;
      while (run != last) {
        Term merged = *run;
        Term* next = run + 1;
        while (next != last && next->det == merged.det) {
          merged.Add(*next);
          ++next;
        }
        run = next;
        merged_terms.push_back(merged);
      }
      t0 = omp_get_wtime();
      t_merge[t] += t0 - t1;
      distinct += merged_terms.size();
      size_t kept = 0;
      for (const Term& term : merged_terms) {
        if (!InSpace(term.det, sorted_dets, space)) {
          merged_terms[kept++] = term;
        }
      }
      merged_terms.resize(kept);
      external += kept;
      t1 = omp_get_wtime();
      t_space[t] += t1 - t0;
      std::vector<double> energies(kept);
      for (size_t k = 0; k < kept; ++k) {
        energies[k] = TermEnergy<W>(merged_terms[k], ints);
      }
      t0 = omp_get_wtime();
      t_diag[t] += t0 - t1;
      for (size_t k = 0; k < kept; ++k) {
        contribution(merged_terms[k], energies[k], &values);
        for (int v = 0; v < num_values; ++v) local[v] += values[v];
      }
      t_sum[t] += omp_get_wtime() - t0;
    }
#pragma omp critical
    for (int v = 0; v < num_values; ++v) total[v] += local[v];
  }
  const double parts_wall = omp_get_wtime() - start;
  auto sum_of = [](const std::vector<double>& v) {
    double x = 0.0;
    for (double y : v) x += y;
    return x;
  };
  const double parts_busy = sum_of(t_sort) + sum_of(t_merge) + sum_of(t_space) +
                            sum_of(t_diag) + sum_of(t_sum);
  // Wall of each part step in proportion to its busy time.
  for (const std::vector<double>* v :
       {&t_sort, &t_merge, &t_space, &t_diag, &t_sum}) {
    const double b = sum_of(*v);
    record(b, parts_busy > 0.0 ? parts_wall * b / parts_busy : 0.0);
  }
  start = omp_get_wtime();
  comm.Sum(total.data(), total.size());
  record(0.0, omp_get_wtime() - start);

  // Over the ranks: busy and counts summed, wall the largest.
  std::vector<double> sums = step_busy;
  sums.insert(sums.end(),
              {generated, distinct, external, profile->singles,
               profile->doubles, generated * 2.0 * sizeof(Term), exchanged});
  comm.Sum(sums.data(), sums.size());
  std::vector<double> walls = comm.GatherAll(step_wall);
  const size_t n = step_wall.size();
  static const char* const kNames[] = {
      "singles (screened by bound, then evaluated)",
      "doubles: heat-bath walk",
      "doubles: Slater-Condon elements, fold",
      "storing the terms (incl. per-generator merge)",
      "waiting for the slowest rank",
      "partition by owner",
      "exchange between ranks",
      "sort",
      "merge equal determinants",
      "remove variational determinants",
      "denominators H_aa",
      "contributions, sum",
      "reduction over ranks"};
  for (size_t k = 0; k < n; ++k) {
    double w = 0.0;
    for (int r = 0; r < comm.size(); ++r) w = std::max(w, walls[r * n + k]);
    profile->Add(kNames[k], sums[k], w);
  }
  // The generation steps were run cumulatively: report the differences.
  for (int k = 3; k >= 1; --k) {
    profile->busy[k] = std::max(0.0, profile->busy[k] - profile->busy[k - 1]);
    profile->wall[k] = std::max(0.0, profile->wall[k] - profile->wall[k - 1]);
  }
  profile->generated = sums[n];
  profile->distinct = sums[n + 1];
  profile->external = sums[n + 2];
  profile->singles = sums[n + 3];
  profile->doubles = sums[n + 4];
  profile->bytes_partitioned = sums[n + 5];
  profile->bytes_exchanged = sums[n + 6];
  if (sum_of(dummy) == -1.0) std::printf(" ");  // keep the work
  return total;
}

inline void PrintSumProfile(const char* title, const SumProfile& p,
                            int num_threads_total, const Comm& comm) {
  if (!comm.is_root()) return;
  double busy = 0.0, wall = 0.0;
  for (size_t k = 0; k < p.steps.size(); ++k) {
    busy += p.busy[k];
    wall += p.wall[k];
  }
  std::printf("profile: %s\n", title);
  std::printf("  %-46s %12s %10s\n", "step", "thread-s", "wall s");
  for (size_t k = 0; k < p.steps.size(); ++k) {
    std::printf("  %-46s %12.2f %10.3f\n", p.steps[k].c_str(), p.busy[k],
                p.wall[k]);
  }
  std::printf("  %-46s %12.2f %10.3f  (idle: %.1f%% of %d threads x wall)\n",
              "total", busy, wall,
              100.0 * (1.0 - busy / (num_threads_total * wall)),
              num_threads_total);
  std::printf(
      "  terms generated %.4g (%zu bytes each), distinct %.4g, outside the "
      "space %.4g; singles passing %.4g, doubles passing %.4g; bytes "
      "partitioned %.4g, exchanged %.4g\n",
      p.generated, p.term_bytes, p.distinct, p.external, p.singles, p.doubles,
      p.bytes_partitioned, p.bytes_exchanged);
}

// Bytes per second of a parallel copy of a large array (read + write).
inline double MeasureBandwidth() {
  const size_t n = size_t{1} << 28;  // 2 GB of doubles
  RawArray<double> a(n), b(n);
  const long count = static_cast<long>(n);
#pragma omp parallel for schedule(static)
  for (long k = 0; k < count; ++k) a[k] = static_cast<double>(k);
  double best = 1e300;
  for (int rep = 0; rep < 3; ++rep) {
    const double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long k = 0; k < count; ++k) b[k] = a[k];
    best = std::min(best, omp_get_wtime() - start);
  }
  return 2.0 * n * sizeof(double) / best;
}

}  // namespace internal

// Everything the perturbative routines need to know about the wave function.
template <int W, class Filter, class Ints, typename S = typename Ints::Scalar>
struct PtProblem {
  Span<Det<W>> dets;
  Span<Det<W>> sorted_dets;
  Span<S> coefficients;
  double energy;  // variational energy, without the core energy
  const Ints& ints;
  const HeatBath& heat_bath;
  const Filter& filter;
  int num_batches;
  const Comm& comm;
  // With spin_flip: the layout of `dets`, whose coefficients are then those
  // of the expanded vector; the sums use the symmetry (see above).
  const SpinFlipLayout* flip = nullptr;
  // The determinants of `dets` as a hashed set, for the membership tests
  // (null: bisection in sorted_dets).
  const DetSet<W>* space = nullptr;
};

namespace internal {

// The weight of generator i: 0 for a partner, 1/2 for a determinant that is
// its own flip, 1 otherwise (and always 1 without spin_flip).
template <int W, class Filter, class Ints, typename S>
double GeneratorWeight(const PtProblem<W, Filter, Ints, S>& problem, size_t i) {
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
template <int W, class Filter, class Ints, typename S>
double PtDeterministic(const PtProblem<W, Filter, Ints, S>& problem,
                       double epsilon, internal::SumProfile* profile = nullptr,
                       uint64_t max_terms = 0,
                       internal::SumStats* stats = nullptr, long stride = 1) {
  // With stride > 1 only every stride-th determinant generates (a probe of
  // the cost per term; the sum is then not E2).
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
    internal::ElementSink<W, decltype(visit), Ints> sink(
        problem.dets[i], problem.ints, 0.0, visit);
    ForEachExcitation(problem.dets[i], occ, epsilon / std::abs(ci),
                      problem.ints, problem.heat_bath, problem.filter, &sink);
  };
  auto contribution = [&](const Term& term, double energy_a,
                          std::vector<double>* values) {
    (*values)[0] = internal::ExternalWeight(term.det, flip != nullptr) *
                   std::norm(term.sum) / (problem.energy - energy_a);
  };
  if (profile != nullptr) {
    auto parent = [&](long p, size_t* i, double* cutoff) {
      *i = p;
      const double c = std::abs(problem.coefficients[p]);
      *cutoff = c > 0.0 ? epsilon / c : 0.0;
      return c > 0.0 && internal::GeneratorWeight(problem, p) > 0.0;
    };
    return internal::ProfileSumOverExternalDets<W, Term>(
        problem.dets.size(), problem.dets, problem.sorted_dets, problem.ints,
        problem.heat_bath, problem.filter, problem.flip, problem.space, 1,
        problem.comm, parent, generate, contribution, profile)[0];
  }
  if (stride > 1) {
    const long n = static_cast<long>(problem.dets.size());
    auto strided = [&](long p, const auto& emit) {
      generate(p * stride, emit);
    };
    return internal::SumOverExternalDets<W, Term>(
        (n + stride - 1) / stride, problem.sorted_dets, problem.ints,
        problem.num_batches, 1, problem.comm, strided, contribution, max_terms,
        stats, problem.space)[0];
  }
  return internal::SumOverExternalDets<W, Term>(
      problem.dets.size(), problem.sorted_dets, problem.ints,
      problem.num_batches, 1, problem.comm, generate, contribution, max_terms,
      stats, problem.space)[0];
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
template <int W, class Filter, class Rng, class Ints, typename S>
StochasticEstimate PtWalkerSample(const PtProblem<W, Filter, Ints, S>& problem,
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
      internal::ElementSink<W, decltype(visit), Ints> sink(
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
template <int W, class Filter, class Rng, class Ints, typename S>
StochasticEstimate PtPooledSample(
    const PtProblem<W, Filter, Ints, S>& problem, double epsilon,
    double epsilon_large, int num_samples, const DeterminantSampler& sampler,
    Rng* rng, internal::SumProfile* profile = nullptr, uint64_t max_terms = 0,
    internal::SumWorkspace<internal::SampleTerm<W, S>>* workspace = nullptr) {
  using Term = internal::SampleTerm<W, S>;
  std::vector<size_t> draws, indices;
  std::vector<int> multiplicities;
  const double draw_start = omp_get_wtime();
  sampler.Draw(num_samples, rng, &draws, &indices, &multiplicities);
  if (profile != nullptr) {
    profile->Add("drawing the samples", 0.0, omp_get_wtime() - draw_start);
  }
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
    auto visit = [&](const Det<W>& det, S element, bool is_large,
                     double energy) {
      S x = element * ci;
      Term term;
      term.det = det;
      term.energy = energy;
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
    internal::ElementEnergySink<W, decltype(visit), Ints> sink(
        problem.dets[i], occ, problem.ints, epsilon_large / std::abs(ci),
        visit);
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
      term.energy = merged.energy;
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
  std::vector<double> sums;
  if (profile != nullptr) {
    auto parent = [&](long p, size_t* i, double* cutoff) {
      *i = indices[p];
      *cutoff = epsilon / std::abs(problem.coefficients[*i]);
      return true;
    };
    sums = internal::ProfileSumOverExternalDets<W, Term>(
        indices.size(), problem.dets, problem.sorted_dets, problem.ints,
        problem.heat_bath, problem.filter, problem.flip, problem.space, 2,
        problem.comm, parent, generate, contribution, profile);
  } else {
    sums = internal::SumOverExternalDets<W, Term>(
        indices.size(), problem.sorted_dets, problem.ints, problem.num_batches,
        2, problem.comm, generate, contribution, max_terms, nullptr,
        problem.space, workspace);
  }
  StochasticEstimate estimate;
  estimate.small = sums[0];
  estimate.large = sums[1];
  return estimate;
}

// ---------------------------------------------------------------------------
// epsilon2_large and num_samples chosen by the program ("auto").
//
// Memory model: a sum over external determinants holds, per rank, its terms
// about three times (the per-thread lists, which grow by doubling, and the
// partitioned copy, or the exchanged one), so kTermCopies * sizeof(Term)
// bytes per term held. The terms of the deterministic part at a threshold
// t, counted with their repetitions (the strict bound of what is held), are
// estimated from every n-th determinant (term counts add up over the
// generating determinants, so the scaled count is unbiased, and its
// sampling error is printed); those of a stochastic batch per sample from a
// pilot.
//
// Thresholds visited: from epsilon1 / 10 (the last heat-bath threshold of
// the selection; 1e-4 if unknown), never below epsilon2. When the count at
// t uses the share f of the terms one pass may hold (with kMemorySafety),
// the next threshold is t f^(1/alpha), alpha the exponent of terms ~ t^-alpha
// (1 at first, then from the last two points), at most a factor 10 per
// step and within the thresholds known to fit and not to fit, until f lies
// in [0.5, 1] or epsilon2 is reached. If epsilon2 itself fits the whole
// correction is deterministic.
//
// Choice: a pilot at two batch sizes gives, for every threshold visited
// that fits, the spread of the stochastic correction E2(eps2) - E2(t) (each
// term carries the band of its strength |H(a,i) c_i|) and the time per
// batch; a probe (the deterministic part from every n-th determinant at the
// smallest threshold that fits) the time per term. Of the thresholds within
// kCloseEnough of the least predicted time to target_error the largest is
// taken (compared at the batches each needs for target_error); the
// stochastic part always runs, at least min_iterations batches. Passes over
// the hash range are only the guard's fallback. num_samples: the
// largest batch whose terms fit the memory per rank, at most max_samples.

constexpr double kTermCopies = 3.0;
// The deterministic part is chosen to fit one pass with this margin on the
// counted bound; a threshold within kCloseEnough of the least predicted
// time is preferred if it is larger.
constexpr double kMemorySafety = 1.5;
constexpr double kCloseEnough = 1.2;
constexpr int kMaxCandidates = 9;

namespace internal {

// What a stochastic pilot term accumulates, by band of its strength: band b
// < num_bands - 1 holds the terms above candidate b and not above b - 1,
// the last band those below all candidates.
template <int W, typename S>
struct BandTerm {
  // Single precision is enough for the spreads the pilot estimates and
  // halves the bytes the pilot moves.
  using Value = std::conditional_t<std::is_same<S, double>::value, float, S>;
  Det<W> det;
  std::array<Value, kMaxCandidates + 1> sum{};
  std::array<float, kMaxCandidates + 1> square{};
  void Add(const BandTerm& other) {
    for (int b = 0; b <= kMaxCandidates; ++b) {
      sum[b] += other.sum[b];
      square[b] += other.square[b];
    }
  }
};

inline int BandOf(double strength, const std::vector<double>& thresholds) {
  int b = 0;
  while (b < static_cast<int>(thresholds.size()) && strength <= thresholds[b]) {
    ++b;
  }
  return b;
}

// Bytes this rank may use for one sum over external determinants.
inline double PtMemoryBudget(const PtInput& pt, const Comm& comm) {
  if (pt.memory_gb > 0.0) return pt.memory_gb * 1024.0 * 1024.0 * 1024.0;
  double available = 0.0;
  if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
    char key[64];
    double value;
    char unit[16];
    while (std::fscanf(f, "%63s %lf %15s", key, &value, unit) == 3) {
      if (std::string(key) == "MemAvailable:") available = value * 1024.0;
    }
    std::fclose(f);
  }
  double nodes = comm.is_node_leader() ? 1.0 : 0.0;
  comm.Sum(&nodes, 1);
  const double ranks_per_node = comm.size() / std::max(1.0, nodes);
  // The smallest over the ranks.
  const std::vector<double> all =
      comm.GatherAll(std::vector<double>{available / ranks_per_node});
  return 0.6 * *std::min_element(all.begin(), all.end());
}

}  // namespace internal

// The terms of the deterministic sum at `threshold`, with repetitions,
// from every stride-th determinant (offset `offset`), scaled up. Collective.
struct TermCount {
  double threshold = 0.0;
  double terms = 0.0;
  double error = 0.0;  // standard error of the scaled count
  double seconds = 0.0;
  long stride = 1;
};

template <int W, class Filter, class Ints, typename S>
TermCount CountTerms(const PtProblem<W, Filter, Ints, S>& problem,
                     double threshold, long stride, long offset) {
  const double start = omp_get_wtime();
  const SpinFlipLayout* flip = problem.flip;
  const int lambda = flip != nullptr ? flip->lambda() : 1;
  const long n = static_cast<long>(problem.dets.size());
  const long sampled = n > offset ? (n - offset + stride - 1) / stride : 0;
  const long rank = problem.comm.rank(), ranks = problem.comm.size();
  double sum = 0.0, sum2 = 0.0;
#pragma omp parallel reduction(+ : sum, sum2)
  {
    std::vector<int> occ;
    auto keeps = [&](const Det<W>& det) {
      if (flip == nullptr) return true;
      Det<W> canonical;
      return internal::FoldOntoCanonical(det, lambda, &canonical) != 0.0;
    };
#pragma omp for schedule(dynamic, 16)
    for (long p = 0; p < sampled; ++p) {
      if (p % ranks != rank) continue;
      const long i = offset + p * stride;
      const double c = std::abs(problem.coefficients[i]);
      if (c == 0.0 || internal::GeneratorWeight(problem, i) == 0.0) continue;
      const Det<W>& det = problem.dets[i];
      occ.clear();
      det.AppendOccupied(&occ);
      const auto& view = problem.filter.ForDeterminant(occ);
      double count = 0.0;
      Excitation single;
      single.rank = 1;
      for (int h : occ) {
        single.holes[0] = h;
        for (int a = 0; a < problem.ints.num_orbs(); ++a) {
          if (det.Occupied(a) ||
              problem.ints.SingleBound(h, a) * c <= threshold ||
              !view.Single(h, a)) {
            continue;
          }
          single.particles[0] = a;
          if (std::abs(OffDiagonalElement(det, single, problem.ints)) * c <=
              threshold) {
            continue;
          }
          Det<W> excited = det;
          excited.Clear(h);
          excited.Set(a);
          count += keeps(excited);
        }
      }
      const double lower = threshold / c;
      if (problem.heat_bath.max_magnitude() >= lower) {
        for (size_t m2 = 1; m2 < occ.size(); ++m2) {
          const int j = occ[m2];
          for (size_t m1 = 0; m1 < m2; ++m1) {
            const int i2 = occ[m1];
            problem.heat_bath.ForEachDouble(
                i2, j, lower, [&](int a, int b, double /*magnitude*/) {
                  if (det.Occupied(a) || det.Occupied(b)) return;
                  if (!view.Double(i2, j, a, b)) return;
                  if (flip != nullptr) {
                    Det<W> excited = det;
                    excited.Clear(i2);
                    excited.Clear(j);
                    excited.Set(a);
                    excited.Set(b);
                    if (!keeps(excited)) return;
                  }
                  count += 1.0;
                });
          }
        }
      }
      sum += count;
      sum2 += count * count;
    }
  }
  double sums[2] = {sum, sum2};
  problem.comm.Sum(sums, 2);
  TermCount result;
  result.threshold = threshold;
  result.stride = stride;
  result.terms = stride * sums[0];
  if (sampled > 1 && stride > 1) {
    const double mean = sums[0] / sampled;
    const double variance =
        std::max(0.0, (sums[1] - sampled * mean * mean) / (sampled - 1));
    result.error =
        stride * std::sqrt(sampled * variance *
                           (1.0 - 1.0 / static_cast<double>(stride)));
  }
  result.seconds = omp_get_wtime() - start;
  return result;
}

// The pilot: batches of num_samples samples at epsilon2, each giving, for
// every threshold, the stochastic correction E2(eps2) - E2(threshold).
struct PilotResult {
  std::vector<double>
      sigma;  // spread of the correction per batch, by threshold
  std::vector<double> mean;
  double sigma_total = 0.0;       // spread of E2(eps2) itself per batch
  double terms_per_sample = 0.0;  // over all ranks
  double seconds_per_batch = 0.0;
  double seconds = 0.0;
};

template <int W, class Filter, class Rng, class Ints, typename S>
PilotResult PtPilot(const PtProblem<W, Filter, Ints, S>& problem,
                    double epsilon, const std::vector<double>& thresholds,
                    int num_samples, int num_batches,
                    const DeterminantSampler& sampler, Rng* rng) {
  using Term = internal::BandTerm<W, S>;
  const int num = static_cast<int>(thresholds.size());
  const SpinFlipLayout* flip = problem.flip;
  const int lambda = flip != nullptr ? flip->lambda() : 1;
  const double n = num_samples;
  const double start_all = omp_get_wtime();
  std::vector<std::vector<double>> values(num_batches);
  double terms = 0.0;
  for (int batch = 0; batch < num_batches; ++batch) {
    std::vector<size_t> draws, indices;
    std::vector<int> multiplicities;
    sampler.Draw(num_samples, rng, &draws, &indices, &multiplicities);
    std::vector<double> generated(omp_get_max_threads(), 0.0);
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
      std::vector<Term> mine;
      auto visit = [&](const Det<W>& det, S element, bool /*is_large*/) {
        const double strength = std::abs(element) * std::abs(ci);
        S x = element * ci;
        Term term;
        term.det = det;
        if (flip != nullptr) {
          const double factor =
              g * internal::FoldOntoCanonical(det, lambda, &term.det);
          if (factor == 0.0) return;
          x *= factor;
        }
        term.sum[internal::BandOf(strength, thresholds)] =
            static_cast<typename Term::Value>(x);
        mine.push_back(term);
      };
      internal::ElementSink<W, decltype(visit), Ints> sink(
          problem.dets[i], problem.ints, 0.0, visit);
      ForEachExcitation(problem.dets[i], occ, epsilon / std::abs(ci),
                        problem.ints, problem.heat_bath, problem.filter, &sink);
      generated[omp_get_thread_num()] += mine.size();
      // Two terms of one generator meet at one determinant only through
      // the fold.
      if (flip != nullptr) {
        std::sort(mine.begin(), mine.end(),
                  [](const Term& a, const Term& b) { return a.det < b.det; });
      }
      size_t k = 0;
      while (k < mine.size()) {
        Term merged = mine[k];
        size_t next = k + 1;
        while (next < mine.size() && mine[next].det == merged.det) {
          for (int b = 0; b <= num; ++b) merged.sum[b] += mine[next].sum[b];
          ++next;
        }
        k = next;
        for (int b = 0; b <= num; ++b) {
          merged.square[b] =
              static_cast<float>(quadratic * std::norm(S(merged.sum[b])));
          merged.sum[b] =
              static_cast<typename Term::Value>(linear * S(merged.sum[b]));
        }
        emit(merged);
      }
    };
    auto contribution = [&](const Term& term, double energy_a,
                            std::vector<double>* out) {
      const double denominator =
          (problem.energy - energy_a) * n * (n - 1.0) /
          internal::ExternalWeight(term.det, flip != nullptr);
      S partial(0.0);
      double squares = 0.0;  // sums in double
      // out[m]: the part above threshold m; out[num]: all of it.
      for (int b = 0; b <= num; ++b) {
        partial += S(term.sum[b]);
        squares += term.square[b];
        (*out)[b] = (std::norm(partial) + squares) / denominator;
      }
    };
    values[batch] = internal::SumOverExternalDets<W, Term>(
        indices.size(), problem.sorted_dets, problem.ints, 1, num + 1,
        problem.comm, generate, contribution, 0, nullptr, problem.space);
    for (double x : generated) terms += x;
  }
  problem.comm.Sum(&terms, 1);
  PilotResult result;
  result.seconds = omp_get_wtime() - start_all;
  result.seconds_per_batch = result.seconds / num_batches;
  result.terms_per_sample = terms / (double(num_samples) * num_batches);
  for (int m = 0; m <= num; ++m) {
    double sum = 0.0, sum2 = 0.0;
    for (const std::vector<double>& v : values) {
      const double x = m == num ? v[num] : v[num] - v[m];
      sum += x;
      sum2 += x * x;
    }
    const double mean = sum / num_batches;
    const double sigma = std::sqrt(
        std::max(0.0, (sum2 - num_batches * mean * mean) / (num_batches - 1)));
    if (m == num) {
      result.sigma_total = sigma;
    } else {
      result.mean.push_back(mean);
      result.sigma.push_back(sigma);
    }
  }
  return result;
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

// What the program chose (see "auto" above).
struct PtAutoChoice {
  double epsilon_large = 0.0;
  int num_samples = 0;
  bool deterministic_only = false;  // epsilon2 itself fits
  int det_batches = 1;
  uint64_t det_max_terms = 0;    // guard of the deterministic part
  uint64_t batch_max_terms = 0;  // guard of a stochastic batch
  double counted_terms = 0.0;    // at epsilon_large, for the output
};

template <int W, class Filter, class Ints, typename S>
PtAutoChoice ChoosePtParameters(const PtProblem<W, Filter, Ints, S>& problem,
                                const PtInput& pt,
                                const DeterminantSampler& sampler, int root) {
  using DetTermT = internal::DetTerm<W, S>;
  using SampleTermT = internal::SampleTerm<W, S>;
  const Comm& comm = problem.comm;
  const bool semistochastic = pt.method == PtMethod::kSemistochastic;
  const bool talk = comm.is_root();
  const double ranks = comm.size();
  const double imbalance = 1.1;  // the largest rank over the mean
  const double gb = 1024.0 * 1024.0 * 1024.0;
  const double budget = internal::PtMemoryBudget(pt, comm);
  PtAutoChoice choice;
  choice.epsilon_large = pt.epsilon2_large;
  choice.num_samples = pt.num_samples;
  if (talk) {
    std::printf("pt auto (root %d): memory per rank %.1f GB (%s)\n", root,
                budget / gb,
                pt.memory_gb > 0.0 ? "pt.memory"
                                   : "60 % of what the node has available");
  }

  // Thresholds of the deterministic part, from epsilon1 / 10 down.
  const double det_bytes = kTermCopies * sizeof(DetTermT) * imbalance / ranks;
  const double one_pass = budget / (kMemorySafety * det_bytes);  // terms
  std::vector<TermCount> visited;
  if (semistochastic && pt.auto_large) {
    const long n = static_cast<long>(problem.dets.size());
    // About a twentieth of the determinants, at least 2000 of them.
    const long stride =
        std::max<long>(1, std::min<long>(20, n / std::max<long>(1, 2000)));
    const long offset = static_cast<long>(Mix64(pt.seed + root) % stride);
    double t =
        std::max(pt.epsilon2,
                 pt.start_threshold > 0.0 ? 0.1 * pt.start_threshold : 1e-4);
    double alpha = 1.0;
    double fits = 0.0,
           too_big = 0.0;  // the smallest that fits, the largest not
    for (int step = 0; step < 12; ++step) {
      const TermCount count = CountTerms(problem, t, stride, offset);
      visited.push_back(count);
      const double f = count.terms / one_pass;
      if (f <= 1.0) {
        fits = fits == 0.0 ? t : std::min(fits, t);
      } else {
        too_big = std::max(too_big, t);
      }
      if (f <= 1.0 && (f >= 0.5 || t <= pt.epsilon2)) break;
      const size_t m = visited.size();
      if (m >= 2) {
        const TermCount& a = visited[m - 2];
        const TermCount& b = visited[m - 1];
        if (a.terms > 0.0 && b.terms > 0.0 && a.threshold != b.threshold) {
          alpha =
              std::log(b.terms / a.terms) / std::log(a.threshold / b.threshold);
          alpha = std::min(4.0, std::max(0.3, alpha));
        }
      }
      double next = t * std::pow(std::max(f, 1e-12), 1.0 / alpha);
      next = std::min(10.0 * t, std::max(0.1 * t, next));
      next = std::max(next, pt.epsilon2);
      // Over the budget: up by at least a tenth, so that one that fits is
      // found.
      if (f > 1.0) next = std::max(next, 1.1 * t);
      // Within what is known: below a threshold that fits, above one that
      // does not.
      if (too_big > 0.0 && next <= too_big) {
        next = fits > 0.0 ? std::sqrt(too_big * fits) : 2.0 * too_big;
      }
      if (fits > 0.0 && next >= fits) {
        next = too_big > 0.0 ? std::sqrt(too_big * fits) : 0.5 * fits;
      }
      next = std::max(next, pt.epsilon2);
      if (f <= 1.0 && std::abs(next / t - 1.0) < 0.05) break;
      t = next;
    }
    if (talk) {
      std::printf(
          "pt auto: terms of the deterministic part (with repetitions, from 1 "
          "in %ld determinants), one pass holds %.4g terms with the margin "
          "%.1f:\n  %10s %12s %10s %9s %9s\n",
          stride, one_pass, kMemorySafety, "threshold", "terms", "error",
          "of pass", "seconds");
      for (const TermCount& c : visited) {
        std::printf("  %10.3e %12.4g %10.2g %8.0f%% %9.2f\n", c.threshold,
                    c.terms, c.error, 100.0 * c.terms / one_pass, c.seconds);
      }
    }
    // epsilon2 itself fits: all of it deterministically.
    for (const TermCount& c : visited) {
      if (c.threshold <= pt.epsilon2 && c.terms <= one_pass) {
        choice.deterministic_only = true;
        choice.epsilon_large = pt.epsilon2;
        choice.counted_terms = c.terms;
        choice.det_max_terms =
            static_cast<uint64_t>(kMemorySafety * c.terms * imbalance / ranks);
        if (talk) {
          std::printf(
              "pt auto: epsilon2 = %.2e fits one pass: the whole "
              "correction is deterministic\n",
              pt.epsilon2);
        }
        return choice;
      }
    }
  }

  // The candidates: the thresholds visited that fit, largest first (if none
  // fits, the largest visited, in passes).
  std::vector<TermCount> candidates;
  if (semistochastic && pt.auto_large) {
    std::vector<TermCount> sorted = visited;
    std::sort(sorted.begin(), sorted.end(),
              [](const TermCount& a, const TermCount& b) {
                return a.threshold > b.threshold;
              });
    for (const TermCount& c : sorted) {
      if (c.terms <= one_pass && c.threshold > pt.epsilon2 &&
          static_cast<int>(candidates.size()) < kMaxCandidates) {
        candidates.push_back(c);
      }
    }
    if (candidates.empty()) candidates.push_back(sorted.front());
  } else if (semistochastic) {
    TermCount given;
    given.threshold = pt.epsilon2_large;
    candidates.push_back(given);
  }
  std::vector<double> thresholds;
  for (const TermCount& c : candidates) thresholds.push_back(c.threshold);

  // The pilot, at two batch sizes.
  std::mt19937_64 stream(Mix64(pt.seed ^ 0x51ed270b7a3e0f1dULL) ^
                         Mix64(root + 1));
  const PilotResult small_pilot =
      PtPilot(problem, pt.epsilon2, thresholds, pt.pilot_samples,
              pt.pilot_batches, sampler, &stream);
  const double n0 = 4.0 * pt.pilot_samples;
  const PilotResult pilot =
      PtPilot(problem, pt.epsilon2, thresholds, static_cast<int>(n0),
              pt.pilot_batches, sampler, &stream);
  const double bytes_per_sample = pilot.terms_per_sample / ranks * imbalance *
                                  kTermCopies * sizeof(SampleTermT);
  const double time_exponent =
      small_pilot.seconds_per_batch > 0.0
          ? std::min(1.0,
                     std::max(0.7, std::log(pilot.seconds_per_batch /
                                            small_pilot.seconds_per_batch) /
                                       std::log(4.0)))
          : 1.0;
  if (talk) {
    std::printf(
        "pt auto: pilot of %d + %d batches of %d and %.0f samples in %.2f s: "
        "%.0f terms per sample, %.0f bytes per sample and rank in a batch; "
        "%.2f and %.2f s per batch; spread of E2 per batch %.2e and %.2e\n",
        pt.pilot_batches, pt.pilot_batches, pt.pilot_samples, n0,
        small_pilot.seconds + pilot.seconds, pilot.terms_per_sample,
        bytes_per_sample, small_pilot.seconds_per_batch,
        pilot.seconds_per_batch, small_pilot.sigma_total, pilot.sigma_total);
  }
  if (pt.auto_samples) {
    double n = std::floor(budget / bytes_per_sample / 100.0) * 100.0;
    n = std::min<double>(n, pt.max_samples);
    n = std::max(n, 100.0);
    choice.num_samples = static_cast<int>(n);
    if (talk) {
      std::printf("pt auto: num_samples = %d, %s\n", choice.num_samples,
                  n >= pt.max_samples ? "max_samples"
                  : n <= 100.0        ? "the least allowed (memory is short)"
                                      : "the largest batch that fits");
    }
  }
  const double n = choice.num_samples;
  choice.batch_max_terms = static_cast<uint64_t>(
      kMemorySafety * pilot.terms_per_sample * n / ranks * imbalance);
  // Batches to the target and their time: beyond the larger pilot the
  // variance is taken to fall as 1/samples only (conservative), and the
  // stochastic part always runs at least min_iterations batches.
  const double seconds_per_batch =
      pilot.seconds_per_batch * std::pow(n / n0, time_exponent);
  auto sigma_at = [&](double sigma0) { return sigma0 * std::sqrt(n0 / n); };
  // Candidates are compared at the batches each needs for target_error
  // (at least one); min_iterations bounds the run, not the comparison.
  auto batches_for = [&](double sigma0) {
    return std::max(1.0, std::pow(sigma_at(sigma0) / pt.target_error, 2));
  };
  if (!semistochastic) {
    if (talk) {
      std::printf("pt auto: about %.0f batches of %.1f s to %.1e\n",
                  batches_for(pilot.sigma_total), seconds_per_batch,
                  pt.target_error);
    }
    return choice;
  }
  if (!pt.auto_large) return choice;

  // The probe of the deterministic part at the smallest candidate.
  const TermCount& smallest = candidates.back();
  const double wanted = 0.05 * budget / det_bytes;
  const long stride = static_cast<long>(
      std::max(1.0, std::ceil(smallest.terms / std::max(1.0, wanted))));
  double probe_seconds = omp_get_wtime();
  PtDeterministic(problem, smallest.threshold, nullptr, 0, nullptr, stride);
  probe_seconds = omp_get_wtime() - probe_seconds;
  // A full walk at threshold t costs about the counting time scaled up; the
  // rest of the probe is storing and merging its terms.
  const double gen_over_count = 1.6;
  auto walk_seconds = [&](const TermCount& c) {
    return gen_over_count * c.seconds * c.stride;
  };
  const double probe_terms = smallest.terms / stride;
  const double seconds_per_term =
      std::max(0.0, probe_seconds - walk_seconds(smallest) / stride) /
      std::max(1.0, probe_terms);
  if (talk) {
    std::printf(
        "pt auto: probe of the deterministic part at %.2e (1 in %ld "
        "determinants, about %.3g terms): %.2f s, %.3g s per term\n",
        smallest.threshold, stride, probe_terms, probe_seconds,
        seconds_per_term);
  }
  const int num = static_cast<int>(candidates.size());
  std::vector<double> det_time(num), stoch_time(num), batches(num);
  double least = 0.0;
  for (int k = 0; k < num; ++k) {
    const double passes =
        std::max(1.0, std::ceil(candidates[k].terms / one_pass));
    det_time[k] = walk_seconds(candidates[k]) * passes +
                  seconds_per_term * candidates[k].terms;
    batches[k] = batches_for(pilot.sigma[k]);
    stoch_time[k] = batches[k] * seconds_per_batch;
    const double total = det_time[k] + stoch_time[k];
    if (k == 0 || total < least) least = total;
  }
  int best = 0;
  while (best < num - 1 &&
         det_time[best] + stoch_time[best] > kCloseEnough * least) {
    ++best;
  }
  if (talk) {
    std::printf(
        "pt auto: thresholds that fit one pass (predicted deterministic "
        "seconds, spread of the stochastic correction per batch of %d "
        "samples, batches to %.1e, predicted stochastic seconds, total):\n"
        "  %10s %12s %9s %10s %9s %9s %9s\n",
        choice.num_samples, pt.target_error, "threshold", "terms", "det s",
        "sigma", "batches", "stoch s", "total s");
    for (int k = 0; k < num; ++k) {
      std::printf("  %10.3e %12.4g %9.1f %10.2e %9.0f %9.1f %9.1f%s\n",
                  candidates[k].threshold, candidates[k].terms, det_time[k],
                  sigma_at(pilot.sigma[k]), batches[k], stoch_time[k],
                  det_time[k] + stoch_time[k],
                  k == best ? "  <- chosen (the largest within 20 % of the "
                              "least)"
                            : "");
    }
  }
  choice.epsilon_large = candidates[best].threshold;
  choice.counted_terms = candidates[best].terms;
  choice.det_batches = static_cast<int>(
      std::max(1.0, std::ceil(candidates[best].terms / one_pass)));
  choice.det_max_terms =
      static_cast<uint64_t>(kMemorySafety * candidates[best].terms * imbalance /
                            ranks / choice.det_batches);
  return choice;
}

template <int W, class Filter, class Ints, typename S>
PtResult RunPerturbation(const PtProblem<W, Filter, Ints, S>& problem,
                         const PtInput& pt, int root, bool print) {
  PtResult result;
  if (pt.method == PtMethod::kDeterministic) {
    result.correction = PtDeterministic(problem, pt.epsilon2);
    return result;
  }

  double deterministic_part = 0.0;
  double epsilon_large = pt.epsilon2;  // pure stochastic: nothing subtracted
  int num_samples = pt.num_samples;
  const DeterminantSampler sampler(problem.coefficients, problem.comm,
                                   problem.flip);
  const bool semistochastic = pt.method == PtMethod::kSemistochastic;
  if (semistochastic) epsilon_large = pt.epsilon2_large;
  PtProblem<W, Filter, Ints, S> det_problem = problem;
  uint64_t det_max_terms = 0, batch_max_terms = 0;
  double counted_terms = 0.0;
  if (pt.auto_large || pt.auto_samples) {
    const PtAutoChoice choice = ChoosePtParameters(problem, pt, sampler, root);
    if (choice.deterministic_only) {
      // epsilon2 fits one pass: no stochastic part is needed.
      internal::SumStats stats;
      const double start = omp_get_wtime();
      PtProblem<W, Filter, Ints, S> whole = problem;
      result.correction = PtDeterministic(whole, pt.epsilon2, nullptr,
                                          choice.det_max_terms, &stats);
      if (problem.comm.is_root()) {
        std::printf(
            "  root %d: all of it deterministic (epsilon2 = %.1e): %.10f, "
            "%.2f s, %d pass(es), %.4g terms (counted %.4g), %.4g distinct "
            "outside the space\n",
            root, pt.epsilon2, result.correction, omp_get_wtime() - start,
            stats.slices, stats.terms, choice.counted_terms, stats.distinct);
      }
      return result;
    }
    counted_terms = choice.counted_terms;
    if (semistochastic) epsilon_large = choice.epsilon_large;
    num_samples = choice.num_samples;
    det_problem.num_batches = choice.det_batches;
    det_max_terms = choice.det_max_terms;
    batch_max_terms = choice.batch_max_terms;
  }
  if (semistochastic) {
    const double start = omp_get_wtime();
    internal::SumStats stats;
    deterministic_part = PtDeterministic(det_problem, epsilon_large, nullptr,
                                         det_max_terms, &stats);
    if ((print || pt.auto_large) && problem.comm.is_root()) {
      char counted[64] = "";
      if (counted_terms > 0.0) {
        std::snprintf(counted, sizeof(counted), " (counted %.4g)",
                      counted_terms);
      }
      std::printf(
          "  root %d: deterministic part (epsilon2_large = %.2e): %.10f, "
          "%.2f s, %d pass(es), %.4g terms%s, %.4g distinct outside the "
          "space\n",
          root, epsilon_large, deterministic_part, omp_get_wtime() - start,
          stats.slices, stats.terms, counted, stats.distinct);
    }
  }

  if (pt.profile) {
    // Step by step, before the calculation proper: the deterministic part
    // and two pooled iterations.
    const int threads = omp_get_max_threads() * problem.comm.size();
    const double bandwidth = internal::MeasureBandwidth();
    if (problem.comm.is_root()) {
      std::printf(
          "profile: copy bandwidth %.1f GB/s per node (read + "
          "write)\n",
          bandwidth / 1e9);
    }
    char title[160];
    if (det_problem.num_batches > 1) {
      if (problem.comm.is_root()) {
        std::printf(
            "profile: the deterministic part takes %d passes; its "
            "replay in one is skipped\n",
            det_problem.num_batches);
      }
    } else {
      internal::SumProfile det_profile;
      const double det_start = omp_get_wtime();
      const double det_value =
          PtDeterministic(problem, epsilon_large, &det_profile);
      std::snprintf(
          title, sizeof(title),
          "deterministic part, epsilon2_large = %.1e: %.10f in %.2f s",
          epsilon_large, det_value, omp_get_wtime() - det_start);
      internal::PrintSumProfile(title, det_profile, threads, problem.comm);
    }
    std::mt19937_64 probe(Mix64(pt.seed + 12345));
    for (int k = 0; k < 2; ++k) {
      internal::SumProfile profile;
      const double start = omp_get_wtime();
      const StochasticEstimate e =
          PtPooledSample(problem, pt.epsilon2, epsilon_large, num_samples,
                         sampler, &probe, &profile);
      // The draw was recorded first; move it after the generation steps.
      std::snprintf(title, sizeof(title),
                    "pooled stochastic iteration %d, %d samples, epsilon2 = "
                    "%.1e: %.10f - %.10f in %.2f s",
                    k + 1, num_samples, pt.epsilon2, e.small, e.large,
                    omp_get_wtime() - start);
      internal::PrintSumProfile(title, profile, threads, problem.comm);
    }
  }
  if (pt.pooled) {
    // One estimate per iteration from the samples drawn alike on every rank,
    // in buffers kept from batch to batch.
    internal::SumWorkspace<internal::SampleTerm<W, S>> workspace;
    std::mt19937_64 stream(Mix64(pt.seed) ^
                           Mix64(0x9e3779b97f4a7c15ULL * (root + 1)));
    double sums[3] = {0.0, 0.0, 0.0};
    result.converged = false;
    for (int iter = 1; iter <= pt.max_iterations; ++iter) {
      const double iter_start = omp_get_wtime();
      const StochasticEstimate estimate = PtPooledSample(
          problem, pt.epsilon2, epsilon_large, num_samples, sampler, &stream,
          nullptr, batch_max_terms, &workspace);
      const double iter_seconds = omp_get_wtime() - iter_start;
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
            "%16.10f  error %.2e  %.2f s\n",
            root, iter, value, result.correction, result.error, iter_seconds);
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
