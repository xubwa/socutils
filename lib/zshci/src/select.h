// "select" mode: selected configuration interaction.
//
// Starting from the determinants of the input, each iteration asks a
// selector which determinants to add and diagonalizes the Hamiltonian in the
// enlarged space. The loop (RunSelect) is the same for every kind of
// selected CI; what differs between them is the selector, a class derived
// from Selector that says which determinants come in. The heat-bath
// selector adds every determinant D_a with |H(a,i) c_i| > epsilon1 for some
// D_i already in the space, where c_i is the coefficient of D_i (the
// root-mean-square over the roots times sqrt(nroots)).
//
// Selectors, like everything here, are templates in the number of words of
// a determinant and in the number type, so one selector serves real and
// complex, spinor and spatial-orbital calculations; which selector runs is
// decided when the program runs, through a virtual call once per iteration.

#ifndef ZSHCI_SRC_SELECT_H_
#define ZSHCI_SRC_SELECT_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iterator>
#include <memory>
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
template <int W, typename S>
class CandidateSink {
 public:
  CandidateSink(const Det<W>& parent, std::vector<Det<W>>* out)
      : parent_(parent), out_(out) {}

  void Single(int i, int a, S /*element*/) {
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
template <int W, class Filter, typename S>
std::vector<Det<W>> SelectNewDeterminants(
    Span<Det<W>> dets, Span<Det<W>> sorted_dets, Span<double> weights,
    double epsilon, const Integrals<S>& ints, const HeatBath& heat_bath,
    const Filter& filter, const Comm& comm = Comm()) {
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
      internal::CandidateSink<W, S> sink(dets[i], &local);
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
template <typename S>
SharedArray<double> ShareWeights(
    const std::vector<std::vector<S>>& root_vectors, size_t dim, int num_roots,
    const Comm& comm) {
  SharedArray<double> weights(dim, comm);
  if (comm.is_root()) {
    for (size_t i = 0; i < dim; ++i) {
      double sum = 0.0;
      for (const std::vector<S>& v : root_vectors) sum += std::norm(v[i]);
      weights[i] = root_vectors.empty() ? std::sqrt(double(num_roots))
                                        : std::sqrt(sum);
    }
  }
  weights.ShareFromRoot(comm);
  return weights;
}

// What a selector is given in one iteration. Everything is the same on every
// rank.
template <int W, typename S>
struct SelectionContext {
  int iteration;
  // The list of determinants. All of them are in the space unless a
  // selector has taken some out (see Selection): those stay in the list,
  // with weight and coefficients zero, and count as outside the space.
  Span<Det<W>> dets;
  // The determinants of the space, sorted.
  Span<Det<W>> sorted_dets;
  // For every determinant of the list, the root-sum-square of its
  // coefficients over the roots (see ShareWeights).
  Span<double> weights;
  // The eigenvectors and their energies (without the core energy); only for
  // selectors that ask for them (needs_vectors), otherwise null and empty.
  const SharedVectors<S>* vectors;
  const std::vector<double>& energies;
  const Integrals<S>& ints;
  const HeatBath& heat_bath;
  const RestrictedFilter& filter;
  const Comm& comm;
  // For every determinant of the list, whether it is in the space, and how
  // many are.
  Span<uint8_t> active;
  size_t num_active;
};

// What a selector decides in one iteration; the same on every rank.
template <int W>
struct Selection {
  // Determinants to bring into the space, sorted, none of them in it now:
  // new ones, or ones of the list that were taken out before.
  std::vector<Det<W>> add;
  // Positions in the list of the determinants to take out of the space,
  // increasing.
  std::vector<size_t> drop;
};

// One kind of selected CI: the rule by which the variational space changes.
template <int W, typename S>
class Selector {
 public:
  virtual ~Selector() = default;

  // The quantity that controls the selection in an iteration, and its name,
  // for the progress table.
  virtual const char* parameter_name() const = 0;
  virtual double Parameter(int iteration) const = 0;
  // Whether the selection has reached its last stage, given the iteration,
  // the size of the space and how many determinants the last Select brought
  // in or took out; the calculation can only be called converged from there
  // on.
  virtual bool InFinalStage(int iteration, size_t num_dets,
                            size_t num_changed) const = 0;
  // Whether Select uses the eigenvectors.
  virtual bool needs_vectors() const { return false; }

  // Collective: what to bring into the space and what to take out of it.
  virtual Selection<W> Select(const SelectionContext<W, S>& context) = 0;
};

// Heat-bath selection with the epsilon1 schedule of the input.
template <int W, typename S>
class HeatBathSelector : public Selector<W, S> {
 public:
  explicit HeatBathSelector(const SelectInput& select) : select_(select) {}

  const char* parameter_name() const override { return "epsilon1"; }
  double Parameter(int iteration) const override {
    return select_.Epsilon(iteration);
  }
  bool InFinalStage(int iteration, size_t /*num_dets*/,
                    size_t /*num_changed*/) const override {
    return iteration >= select_.schedule.back().first;
  }

  Selection<W> Select(const SelectionContext<W, S>& context) override {
    Selection<W> selection;
    selection.add = SelectNewDeterminants(
        context.dets, context.sorted_dets, context.weights,
        select_.Epsilon(context.iteration), context.ints, context.heat_bath,
        context.filter, context.comm);
    return selection;
  }

 private:
  const SelectInput& select_;
};

// Selection by a score from first-order perturbation theory. For every
// determinant D_a outside the space and every root k,
//   V_a(k) = sum_i H(a,i) c_i(k),
// the sum running over the determinants of the space (for asci: over the
// core_dets ones of largest weight) and, if epsilon_generate is set, over
// the terms above it. The score of D_a is
//   cipsi, aci:  sum_k |V_a(k)|^2 / (E_a - E_k)     (its second-order energy)
//   asci:        max_k |V_a(k)| / (E_a - E_k)       (its first-order amplitude)
// and the determinants with the largest scores are added:
//   cipsi, asci: as many as make the space grow by the factor `growth`, up
//                to target_dets;
//   aci:         all but the lowest ones whose scores sum to at most sigma.
// Either rule amounts to a threshold on the score, which is found by
// bisection over all ranks.
//
// With "prune" (asci), the determinants of the space take part in the
// ranking, with the same amplitude - the sum for D_a then runs over the
// generating determinants other than D_a - and the space becomes the
// highest-ranked determinants of both kinds: growth times its present size,
// up to target_dets. Those of the space that fall below are taken out. If
// all determinants generate, the amplitude of one of the space is its
// coefficient (the eigenvalue equation says so) and that is used. Without
// "prune" no determinant is ever removed.
template <int W, typename S>
class PtScoreSelector : public Selector<W, S> {
 public:
  explicit PtScoreSelector(const SelectInput& select) : select_(select) {}

  const char* parameter_name() const override {
    return select_.method == SelectMethod::kAci ? "sigma" : "target";
  }
  double Parameter(int /*iteration*/) const override {
    return select_.method == SelectMethod::kAci
               ? select_.sigma
               : static_cast<double>(select_.target_dets);
  }
  bool InFinalStage(int /*iteration*/, size_t num_dets,
                    size_t num_changed) const override {
    // Also when nothing is left to add before the target is reached.
    return select_.method == SelectMethod::kAci ||
           num_dets >= select_.target_dets || num_changed == 0;
  }
  bool needs_vectors() const override { return true; }

  Selection<W> Select(const SelectionContext<W, S>& context) override {
    const Comm& comm = context.comm;
    const size_t num_dets = context.num_active;
    const bool by_count = select_.method != SelectMethod::kAci;
    // Ranking the determinants of the space takes their coefficients.
    const bool prune = select_.prune && context.vectors != nullptr;
    // The number of determinants to add or, with pruning, the size the
    // space is to have.
    size_t wanted = 0;
    if (by_count) {
      if (!prune && num_dets >= select_.target_dets) return {};
      const size_t grown =
          static_cast<size_t>(std::ceil(select_.growth * num_dets));
      wanted = std::min(select_.target_dets, std::max(grown, num_dets + 1));
      if (!prune) wanted -= num_dets;
    }

    // asci: only determinants of weight at least this generate candidates.
    double core_weight = 0.0;
    if (select_.method == SelectMethod::kAsci && select_.core_dets > 0 &&
        select_.core_dets < num_dets) {
      if (comm.is_root()) {
        std::vector<double> sorted(context.weights.begin(),
                                   context.weights.end());
        std::nth_element(sorted.begin(), sorted.begin() + select_.core_dets - 1,
                         sorted.end(), std::greater<double>());
        core_weight = sorted[select_.core_dets - 1];
      }
      comm.BcastValue(&core_weight);
    }

    // With pruning and a subset generating, the determinants of the space
    // are scored like the others; nothing is excluded from the sums.
    const bool rank_all = prune && core_weight > 0.0;
    const Span<Det<W>> excluded =
        rank_all ? Span<Det<W>>() : context.sorted_dets;

    // The scores of the determinants this rank owns, root by root.
    std::vector<Scored> scores;
    if (context.vectors == nullptr) {
      // No eigenvectors yet: the space holds fewer determinants than roots.
      // Score with one stand-in state, all determinants with their weight
      // and the lowest diagonal energy, to get the space started.
      const std::vector<S> stand_in(context.weights.begin(),
                                    context.weights.end());
      double lowest = DiagonalEnergy(context.dets[0], context.ints);
      for (const Det<W>& det : context.dets) {
        lowest = std::min(lowest, DiagonalEnergy(det, context.ints));
      }
      scores = ScoresOfRoot(context, MakeSpan(stand_in), lowest, core_weight,
                            excluded);
    } else {
      for (int k = 0; k < context.vectors->num_roots; ++k) {
        std::vector<Scored> of_root =
            ScoresOfRoot(context, context.vectors->Root(k), context.energies[k],
                         core_weight, excluded);
        if (k == 0) {
          scores = std::move(of_root);
        } else {
          scores = Combine(scores, of_root);
        }
      }
    }

    // With pruning and all determinants generating: the rank of every
    // determinant of the space, zero for the others of the list. The root
    // counts them.
    std::vector<double> internal;
    if (prune && !rank_all) {
      internal.assign(context.dets.size(), 0.0);
      for (int k = 0; k < context.vectors->num_roots; ++k) {
        const Span<S> root = context.vectors->Root(k);
        for (size_t i = 0; i < internal.size(); ++i) {
          if (context.active[i]) {
            internal[i] = std::max(internal[i], std::abs(root[i]));
          }
        }
      }
    }
    const std::vector<double> none;
    const double threshold =
        by_count ? ThresholdForCount(scores, comm.is_root() ? internal : none,
                                     wanted, comm)
                 : ThresholdForSum(scores, select_.sigma, comm);
    Selection<W> selection;
    for (const Scored& item : scores) {
      if (item.score >= threshold) selection.add.push_back(item.det);
    }
    selection.add = comm.GatherAll(selection.add);
    std::sort(selection.add.begin(), selection.add.end());
    if (rank_all) {
      // What was gathered is the new space: split it into what comes in and
      // what stays.
      const std::vector<Det<W>> kept = std::move(selection.add);
      selection.add.clear();
      std::set_difference(kept.begin(), kept.end(), context.sorted_dets.begin(),
                          context.sorted_dets.end(),
                          std::back_inserter(selection.add));
      for (size_t i = 0; i < context.dets.size(); ++i) {
        if (context.active[i] &&
            !std::binary_search(kept.begin(), kept.end(), context.dets[i])) {
          selection.drop.push_back(i);
        }
      }
    } else if (prune) {
      for (size_t i = 0; i < internal.size(); ++i) {
        if (context.active[i] && internal[i] < threshold) {
          selection.drop.push_back(i);
        }
      }
    }
    return selection;
  }

 private:
  struct Scored {
    Det<W> det;
    double score;
  };

  // The scores from one state with the given coefficients and energy, of
  // the determinants not in `excluded` (sorted), sorted by determinant.
  std::vector<Scored> ScoresOfRoot(const SelectionContext<W, S>& context,
                                   Span<S> coefficients, double energy,
                                   double core_weight,
                                   Span<Det<W>> excluded) const {
    using Term = internal::DetTerm<W, S>;
    const bool amplitude = select_.method == SelectMethod::kAsci;
    const double epsilon = select_.epsilon_generate;
    auto generate = [&](long i, const auto& emit) {
      const S ci = coefficients[i];
      if (std::abs(ci) == 0.0 || context.weights[i] < core_weight) return;
      std::vector<int> occ;
      context.dets[i].AppendOccupied(&occ);
      auto visit = [&](const Det<W>& det, S element, bool /*is_large*/) {
        Term term;
        term.det = det;
        term.sum = element * ci;
        emit(term);
      };
      internal::ElementSink<W, decltype(visit), S> sink(
          context.dets[i], context.ints, 0.0, visit);
      ForEachExcitation(context.dets[i], occ, epsilon / std::abs(ci),
                        context.ints, context.heat_bath, context.filter, &sink);
    };
    // The merged terms arrive here from several threads; each keeps its own
    // list.
    std::vector<std::vector<Scored>> per_thread(omp_get_max_threads());
    auto contribution = [&](const Term& term, double energy_a,
                            std::vector<double>* values) {
      (*values)[0] = 0.0;
      const double gap = energy_a - energy;
      if (gap <= 0.0) return;  // not a candidate for a perturbative score
      const double score =
          amplitude ? std::abs(term.sum) / gap : std::norm(term.sum) / gap;
      if (score > 0.0) {
        per_thread[omp_get_thread_num()].push_back({term.det, score});
      }
    };
    internal::SumOverExternalDets<W, Term>(context.dets.size(), excluded,
                                           context.ints, 1, 1, context.comm,
                                           generate, contribution);
    std::vector<Scored> all;
    for (std::vector<Scored>& list : per_thread) {
      all.insert(all.end(), list.begin(), list.end());
      list = std::vector<Scored>();
    }
    std::sort(all.begin(), all.end(),
              [](const Scored& a, const Scored& b) { return a.det < b.det; });
    return all;
  }

  // Two lists sorted by determinant into one: energies add, amplitudes take
  // the larger.
  std::vector<Scored> Combine(const std::vector<Scored>& a,
                              const std::vector<Scored>& b) const {
    const bool amplitude = select_.method == SelectMethod::kAsci;
    std::vector<Scored> out;
    out.reserve(std::max(a.size(), b.size()));
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
      if (j == b.size() || (i < a.size() && a[i].det < b[j].det)) {
        out.push_back(a[i++]);
      } else if (i == a.size() || b[j].det < a[i].det) {
        out.push_back(b[j++]);
      } else {
        out.push_back({a[i].det, amplitude ? std::max(a[i].score, b[j].score)
                                           : a[i].score + b[j].score});
        ++i;
        ++j;
      }
    }
    return out;
  }

  // The smallest and the largest positive value among the scores and
  // `others` over all ranks; both zero if there are none.
  static void Range(const std::vector<Scored>& scores,
                    const std::vector<double>& others, const Comm& comm,
                    double* smallest, double* largest) {
    // Largest value and largest inverse value of this rank.
    std::vector<double> local(2, 0.0);
    for (const Scored& item : scores) {
      local[0] = std::max(local[0], item.score);
      local[1] = std::max(local[1], 1.0 / item.score);
    }
    for (double value : others) {
      if (value <= 0.0) continue;
      local[0] = std::max(local[0], value);
      local[1] = std::max(local[1], 1.0 / value);
    }
    const std::vector<double> all = comm.GatherAll(local);
    double max_score = 0.0, max_inverse = 0.0;
    for (size_t r = 0; r + 1 < all.size(); r += 2) {
      max_score = std::max(max_score, all[r]);
      max_inverse = std::max(max_inverse, all[r + 1]);
    }
    *largest = max_score;
    *smallest = max_inverse > 0.0 ? 1.0 / max_inverse : 0.0;
  }

  // The value such that about `wanted` of the scores, over all ranks, and
  // of `others` (values that take part in the ranking; a rank passes its
  // own, so shared ones come from one rank only) lie at or above it - all
  // the positive ones if there are no more.
  static double ThresholdForCount(const std::vector<Scored>& scores,
                                  const std::vector<double>& others,
                                  size_t wanted, const Comm& comm) {
    double low, high;
    Range(scores, others, comm, &low, &high);
    if (high == 0.0) return 1.0;  // nothing to choose from
    auto count_at = [&](double threshold) {
      double count = 0.0;
      for (const Scored& item : scores) count += item.score >= threshold;
      for (double value : others) count += value >= threshold;
      comm.Sum(&count, 1);
      return count;
    };
    if (count_at(low) <= static_cast<double>(wanted)) return low;
    // Invariant: more than `wanted` at `low`, at most `wanted` at `high`.
    high *= 1.0 + 1e-12;
    for (int step = 0; step < 100 && high > low * (1.0 + 1e-14); ++step) {
      const double middle = std::sqrt(low * high);
      if (count_at(middle) > static_cast<double>(wanted)) {
        low = middle;
      } else {
        high = middle;
      }
    }
    return high;
  }

  // The largest score t such that the scores below t, over all ranks, sum to
  // at most `budget`.
  static double ThresholdForSum(const std::vector<Scored>& scores,
                                double budget, const Comm& comm) {
    double low, high;
    Range(scores, std::vector<double>(), comm, &low, &high);
    if (high == 0.0) return 1.0;
    auto sum_below = [&](double threshold) {
      double sum = 0.0;
      for (const Scored& item : scores) {
        if (item.score < threshold) sum += item.score;
      }
      comm.Sum(&sum, 1);
      return sum;
    };
    high *= 1.0 + 1e-12;
    if (sum_below(high) <= budget) return high;  // everything may be left out
    // Invariant: the sum below `low` fits the budget, that below `high` not.
    for (int step = 0; step < 100 && high > low * (1.0 + 1e-14); ++step) {
      const double middle = std::sqrt(low * high);
      if (sum_below(middle) <= budget) {
        low = middle;
      } else {
        high = middle;
      }
    }
    return low;
  }

  const SelectInput& select_;
};

// The selector the input asks for.
template <int W, typename S>
std::unique_ptr<Selector<W, S>> MakeSelector(const Input& input) {
  switch (input.select.method) {
    case SelectMethod::kHeatBath:
      return std::make_unique<HeatBathSelector<W, S>>(input.select);
    case SelectMethod::kCipsi:
    case SelectMethod::kAsci:
    case SelectMethod::kAci:
      return std::make_unique<PtScoreSelector<W, S>>(input.select);
  }
  throw std::logic_error("unknown selection method");
}

template <int W, typename S>
void RunSelect(const Input& input, const Integrals<S>& ints, const Comm& comm) {
  const int num_orbs = ints.num_orbs();
  const int num_roots = input.nroots;
  const std::vector<LabelSet>& labels = ints.symmetry();
  const RestrictedFilter filter =
      MakeFilter(input, labels, num_orbs, /*with_restrictions=*/true);

  // The root owns the growing list; `shared_dets` is its copy on every node.
  std::vector<Det<W>> root_dets;
  if (comm.is_root()) {
    root_dets = LoadDeterminants<W>(input, num_orbs);
    // The starting determinants fix the symmetry block; mixing blocks is
    // almost certainly an input mistake.
    const std::vector<int> block_label =
        DeterminantLabels(root_dets[0], labels);
    for (size_t d = 0; d < root_dets.size(); ++d) {
      if (DeterminantLabels(root_dets[d], labels) != block_label) {
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
    std::printf(
        "starting determinants: %zu, electrons: %d, spinors: %d, "
        "words: %d, symmetry label: %s\n",
        root_dets.size(), root_dets[0].Count(), num_orbs, W,
        FormatLabels(block_label).c_str());
  }
  SharedArray<Det<W>> shared_dets =
      SharedArray<Det<W>>::FromRoot(root_dets, comm);

  double time = omp_get_wtime();
  const HeatBath heat_bath(ints, input.select.integral_threshold, comm);
  std::printf("heat-bath tables: %zu entries, %.2f s\n",
              heat_bath.num_entries(), omp_get_wtime() - time);

  VariationalH<W, S> h(input, ints, comm);
  h.Extend(shared_dets.span(), 0);

  // The eigenvectors live on the root; the other ranks see only what is
  // shared from them (the selection weights, and at the end the vectors).
  // If the starting space can hold all roots, start from its eigenvectors;
  // the guesses for them are the vectors of the file the determinants came
  // from, if it has them.
  std::vector<std::vector<S>> root_vectors;
  std::vector<double> energies;
  DavidsonOptions loose = input.davidson;
  loose.tol = std::max(input.davidson.tol, input.select.davidson_tol_loose);
  bool have_vectors = shared_dets.size() >= static_cast<size_t>(num_roots);
  if (have_vectors) {
    if (comm.is_root()) {
      const bool from_file = !input.detfile.empty() &&
                             ReadCoefficients(input.detfile, shared_dets.size(),
                                              num_roots, &root_vectors);
      if (from_file) {
        std::printf("starting vectors: read from %s\n", input.detfile.c_str());
      } else {
        root_vectors =
            GuessVectors(shared_dets.span(), h.stored(), ints, num_roots);
      }
    }
    energies = h.Solve(loose, &root_vectors).energies;
  }

  // t_sel, t_ham and t_dav are the seconds spent in this iteration on the
  // selection, the Hamiltonian and the eigensolver.
  const std::unique_ptr<Selector<W, S>> selector = MakeSelector<W, S>(input);
  std::printf("%5s %10s %12s %5s %20s %9s %8s %8s %8s %9s\n", "iter",
              selector->parameter_name(), "dets", "root", "energy", "davidson",
              "t_sel", "t_ham", "t_dav", "time");
  const double start_time = omp_get_wtime();
  bool finished = false;
  int num_products = 0;
  double seconds_products = 0.0;
  double seconds_solver = 0.0;

  // A selector may take determinants out of the space. They stay in the
  // list, so that neither the list nor the stored Hamiltonian is reordered
  // or rebuilt; the root marks them here, the eigenvectors are zero on them
  // and the eigensolver works in the others.
  std::vector<uint8_t> root_active;
  if (comm.is_root()) root_active.assign(shared_dets.size(), 1);
  size_t num_inactive = 0;

  // Collective: removes the determinants out of the space from the list and
  // makes the Hamiltonian afresh. The eigenvectors must be those of the
  // list as it is.
  auto compact = [&]() {
    const double compact_start = omp_get_wtime();
    if (comm.is_root()) {
      size_t kept = 0;
      for (size_t i = 0; i < root_dets.size(); ++i) {
        if (!root_active[i]) continue;
        root_dets[kept] = root_dets[i];
        for (std::vector<S>& v : root_vectors) v[kept] = v[i];
        ++kept;
      }
      root_dets.resize(kept);
      for (std::vector<S>& v : root_vectors) v.resize(kept);
      root_active.assign(kept, 1);
    }
    shared_dets = SharedArray<Det<W>>::FromRoot(root_dets, comm);
    h.Rebuild(shared_dets.span());
    std::printf(
        "list compacted: %zu determinants out of the space removed, "
        "Hamiltonian made afresh in %.2f s\n",
        num_inactive, omp_get_wtime() - compact_start);
    num_inactive = 0;
  };

  for (int iter = 0; iter < input.select.max_iter && !finished; ++iter) {
    const double epsilon = selector->Parameter(iter);
    const size_t old_size = shared_dets.size();

    const double select_start = omp_get_wtime();
    Selection<W> selection;
    {
      const SharedArray<double> weights =
          ShareWeights(root_vectors, old_size, num_roots, comm);
      const SharedArray<uint8_t> active =
          SharedArray<uint8_t>::FromRoot(root_active, comm);
      const SharedArray<Det<W>> sorted_dets =
          ShareSortedActive(shared_dets.span(), root_active, comm);
      // The eigenvectors go to all ranks only for selectors that use them.
      SharedVectors<S> vectors;
      const bool with_vectors = selector->needs_vectors() && have_vectors;
      if (with_vectors) vectors = ShareVectors(root_vectors, old_size, comm);
      const SelectionContext<W, S> context{iter,
                                           shared_dets.span(),
                                           sorted_dets.span(),
                                           weights.span(),
                                           with_vectors ? &vectors : nullptr,
                                           energies,
                                           ints,
                                           heat_bath,
                                           filter,
                                           comm,
                                           active.span(),
                                           old_size - num_inactive};
      selection = selector->Select(context);
    }
    const double build_start = omp_get_wtime();

    // The root applies the selection to the list: determinants taken out
    // are marked, those brought in are either marked again, if the list
    // has them, or appended.
    std::vector<Det<W>> fresh;
    size_t num_fresh = 0, num_back = 0, num_out = 0;
    if (comm.is_root()) {
      for (size_t i : selection.drop) {
        root_active[i] = 0;
        for (std::vector<S>& v : root_vectors) v[i] = S(0.0);
      }
      num_out = selection.drop.size();
      if (num_inactive + num_out == 0) {
        fresh = std::move(selection.add);
      } else {
        // The positions of the list in the order of the determinants.
        std::vector<size_t> order(old_size);
        for (size_t i = 0; i < old_size; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
          return root_dets[a] < root_dets[b];
        });
        for (const Det<W>& det : selection.add) {
          const auto found = std::lower_bound(
              order.begin(), order.end(), det,
              [&](size_t i, const Det<W>& d) { return root_dets[i] < d; });
          if (found == order.end() || root_dets[*found] < det ||
              det < root_dets[*found]) {
            fresh.push_back(det);
          } else if (!root_active[*found]) {
            root_active[*found] = 1;
            ++num_back;
          }
        }
      }
      num_fresh = fresh.size();
    }
    selection = Selection<W>();
    comm.BcastValue(&num_fresh);
    comm.BcastValue(&num_back);
    comm.BcastValue(&num_out);
    num_inactive = num_inactive + num_out - num_back;
    const size_t num_changed = num_fresh + num_back + num_out;

    const size_t new_size = old_size + num_fresh;
    const size_t num_active = new_size - num_inactive;
    if (num_active < static_cast<size_t>(num_roots)) {
      throw std::runtime_error(
          "the space holds fewer determinants than roots; give more "
          "starting determinants or select more");
    }
    if (num_fresh > 0) {
      if (comm.is_root()) {
        root_dets.insert(root_dets.end(), fresh.begin(), fresh.end());
        root_active.resize(new_size, 1);
      }
      shared_dets = SharedArray<Det<W>>::FromRoot(root_dets, comm);
      h.Extend(shared_dets.span(), old_size);
    }
    fresh = std::vector<Det<W>>();

    if (comm.is_root()) {
      if (!have_vectors) {
        root_vectors =
            GuessVectors(shared_dets.span(), h.stored(), ints, num_roots);
      } else {
        for (std::vector<S>& v : root_vectors) v.resize(new_size, 0.0);
      }
    }
    have_vectors = true;

    // The last epsilon1 has been reached and the space or the energies have
    // stopped changing: solve tightly and stop.
    const double solve_start = omp_get_wtime();
    const std::vector<double> previous = energies;
    const std::vector<uint8_t>* mask =
        num_inactive > 0 ? &root_active : nullptr;
    DavidsonResult result = h.Solve(loose, &root_vectors, mask);
    num_products += result.num_products;
    seconds_products += result.seconds_products;
    seconds_solver += result.seconds_total;
    bool converged_space = false;
    if (selector->InFinalStage(iter, num_active, num_changed) &&
        !previous.empty()) {
      double change = 0.0;
      for (int k = 0; k < num_roots; ++k) {
        change += std::abs(result.energies[k] - previous[k]) / num_roots;
      }
      converged_space = num_changed == 0 || change < input.select.energy_tol;
    }
    finished = converged_space || iter == input.select.max_iter - 1;
    if (finished) {
      result = h.Solve(input.davidson, &root_vectors, mask);
      num_products += result.num_products;
      seconds_products += result.seconds_products;
      seconds_solver += result.seconds_total;
    }
    if (!result.converged) {
      throw std::runtime_error("Davidson did not converge in iteration " +
                               std::to_string(iter));
    }
    energies = result.energies;
    for (int k = 0; k < num_roots; ++k) {
      std::printf("%5d %10.2e %12zu %5d %20.12f %9d %8.2f %8.2f %8.2f %9.2f\n",
                  iter, epsilon, num_active, k,
                  energies[k] + ints.core_energy(), result.num_iterations,
                  build_start - select_start, solve_start - build_start,
                  omp_get_wtime() - solve_start, omp_get_wtime() - start_time);
    }
    if (num_back + num_out + num_inactive > 0) {
      std::printf(
          "      space: %zu new, %zu back in, %zu taken out; %zu of "
          "the %zu listed are out of the space\n",
          num_fresh, num_back, num_out, num_inactive, new_size);
    }
    if (finished && !converged_space) {
      std::printf("note: stopped at max_iter = %d before the energies "
                  "converged to energy_tol\n",
                  input.select.max_iter);
    }
    if (!finished && input.select.compact_above > 0.0 &&
        num_inactive > input.select.compact_above * new_size) {
      compact();
    }
  }
  if (num_inactive > 0) compact();
  std::printf(
      "variational space: %zu determinants, %zu stored Hamiltonian "
      "elements on the root\n",
      shared_dets.size(), h.stored().num_elements());
  h.PrintStats();
  std::printf(
      "eigensolver (root): %d products H*x in %.2f s, %.2f s in the "
      "rest of the solver\n",
      num_products, seconds_products, seconds_solver - seconds_products);

  root_dets = std::vector<Det<W>>();
  const SharedVectors<S> vectors =
      ShareVectors(root_vectors, shared_dets.size(), comm);
  root_vectors = std::vector<std::vector<S>>();
  const std::vector<PtResult> pt =
      ComputePerturbation(input, ints, shared_dets.span(), energies, vectors,
                          &heat_bath, filter, comm);
  WriteResults(input, ints, shared_dets.span(), h, labels, energies, vectors,
               pt, comm);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_SELECT_H_
