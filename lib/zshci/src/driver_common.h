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
#include "direct_h.h"
#include "distributed_h.h"
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

// Collective: the determinants of the root that are marked in `active` (read
// on the root), sorted, once per node.
template <int W>
SharedArray<Det<W>> ShareSortedActive(Span<Det<W>> dets,
                                      const std::vector<uint8_t>& active,
                                      const Comm& comm) {
  std::vector<Det<W>> sorted;
  if (comm.is_root()) {
    for (size_t i = 0; i < dets.size(); ++i) {
      if (active[i]) sorted.push_back(dets[i]);
    }
    std::sort(sorted.begin(), sorted.end());
  }
  return SharedArray<Det<W>>::FromRoot(sorted, comm);
}

// The connection search requested in the input.
inline ConnectOptions MakeConnectOptions(const Input& input) {
  ConnectOptions options;
  options.threshold = input.hij_threshold;
  options.group_a = input.connect_group_a;
  options.subgroups = input.connect_subgroups;
  options.blocks = input.connect_blocks;
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

// The Hamiltonian in the determinants of a calculation, as the drivers use
// it: grown when determinants are added, diagonalized, and asked for density
// matrices. It is either stored (SparseH) or, with "hamiltonian": "direct",
// applied from the group string tables without being stored (DirectH).
template <int W, typename S>
class VariationalH : private ElementSource<S> {
 public:
  VariationalH(const Input& input, const Integrals<S>& ints, const Comm& comm)
      : ints_(ints),
        comm_(comm),
        direct_(input.direct_hamiltonian),
        store_values_(input.store_values),
        adapt_(input.adapt_hamiltonian),
        memory_limit_(
            static_cast<size_t>(input.memory_gb * 1024.0 * 1024.0 * 1024.0)),
        // Basis vectors and their products of the eigensolver, the
        // eigenvectors, and two for a product.
        num_vectors_((2 * std::max(2, input.davidson.max_space_per_root) + 1) *
                         input.nroots +
                     2),
        incremental_(input.incremental),
        connect_(MakeConnectOptions(input)) {
    ResetStored();
  }
  VariationalH(const VariationalH&) = delete;
  VariationalH& operator=(const VariationalH&) = delete;

  bool direct() const { return direct_; }
  // The diagonal (on the root) and, unless direct, the stored elements.
  const SparseH<S>& stored() const { return stored_; }
  const ConnectOptions& connect() const { return connect_; }

  // Collective: makes the Hamiltonian that of `dets`, of which those from
  // first_new on are new.
  void Extend(Span<Det<W>> dets, size_t first_new) {
    if (!incremental_ && first_new > 0) {
      // Everything is made afresh for the whole list.
      first_new = 0;
      ResetStored();
    }
    dets_ = dets;
    AppendDiagonal(dets, first_new, ints_, comm_, &stored_);
    if (connect_.method != ConnectMethod::kGroups) {
      stored_.AppendBlocks(
          BuildBlocks(dets, first_new, ints_, connect_, comm_, &stats_));
      if (!store_values_) stored_.DropValues();
      return;
    }
    // The string tables live as long as the calculation and are extended
    // by what the new determinants bring.
    const double time = omp_get_wtime();
    if (tables_ == nullptr) {
      tables_ = std::make_unique<internal::GroupStringTables<W>>(
          GroupMask<W>(connect_, ints_.num_orbs()), comm_, connect_.subgroups);
      // A Hamiltonian that is not stored wants all connections, again and
      // again.
      tables_->set_block_search((direct_ || adapt_) && connect_.blocks);
    }
    tables_->Extend(dets, first_new);
    table_memory_ = tables_->memory();
    if (!incremental_) tables_->DropExtensionState();
    stats_.seconds_prepare += omp_get_wtime() - time;
    stats_.num_strings = tables_->num_strings(0) + tables_->num_strings(1);
    stats_.num_links = tables_->num_links();
    if (adapt_) {
      ExtendWithinMemory(first_new);
    } else if (!direct_) {
      stored_.AppendBlocks(
          internal::BuildGroupBlocks(dets, first_new, ints_, connect_.threshold,
                                     *tables_, &stats_, store_values_));
    }
  }

  // Collective: makes the Hamiltonian of `dets` from nothing, forgetting
  // what was there.
  void Rebuild(Span<Det<W>> dets) {
    ResetStored();
    tables_.reset();
    Extend(dets, 0);
  }

  // Collective: the lowest eigenpairs (see SolveDistributed), with `active`
  // in the determinants it marks.
  DavidsonResult Solve(const DavidsonOptions& options,
                       std::vector<std::vector<S>>* root_vectors,
                       const std::vector<uint8_t>* active = nullptr) const {
    if (direct_) {
      const DirectH<W, S> h(dets_, ints_, *tables_, connect_.threshold,
                            stored_.diagonal());
      return SolveDistributed<S>(h, options, comm_, root_vectors, active);
    }
    if (!all_rows_stored()) {
      // The stored rows, which include the diagonal, and the others.
      const DirectH<W, S> rest(dets_, ints_, *tables_, connect_.threshold,
                               stored_.diagonal(),
                               DirectH<W, S>::kDefaultScratchBytes,
                               &direct_rows_, /*with_diagonal=*/false);
      const SumOperator sum(stored_, rest);
      return SolveDistributed<S>(sum, options, comm_, root_vectors, active);
    }
    return SolveDistributed<S>(stored_, options, comm_, root_vectors, active);
  }

  // Collective: the density matrices between two vectors.
  DensityMatrices<S> Density(Span<S> bra, Span<S> ket, bool with_two) const {
    const int num_orbs = ints_.num_orbs();
    if (!direct_ && all_rows_stored()) {
      return ComputeDensityMatricesFrom(dets_, StoredPairs<S>(stored_),
                                        num_orbs, bra, ket, with_two, comm_,
                                        ints_.is_spatial());
    }
    return ComputeDensityMatricesFrom(
        dets_, TablePairs<W>(*tables_, dets_.size()), num_orbs, bra, ket,
        with_two, comm_, ints_.is_spatial());
  }

  void PrintStats() const {
    if (tables_ != nullptr) {
      // How the determinants divide by the electrons in the first group.
      std::string text;
      const std::vector<uint64_t>& sizes = tables_->sector_sizes();
      for (size_t in_a = 0; in_a < sizes.size(); ++in_a) {
        if (sizes[in_a] == 0) continue;
        text += (text.empty() ? "" : ", ") + std::to_string(in_a) + ": " +
                std::to_string(sizes[in_a]);
      }
      std::printf(
          "sectors (electrons in the first group: determinants): "
          "%s\n",
          text.c_str());
    }
    const double mb = 1.0 / (1024.0 * 1024.0);
    if (tables_ != nullptr) {
      // As of the last extension; without incremental updates what is
      // "kept" is released as soon as the tables are made.
      const auto& memory = table_memory_;
      std::printf(
          "string tables (root): %.1f MB held per node, %.1f MB %s "
          "(keys %.1f MB), at most %.1f MB more while extending\n",
          memory.shared * mb, memory.kept * mb,
          incremental_ ? "kept to extend them"
                       : "needed only while they are made",
          memory.keys * mb, memory.transient * mb);
    }
    if (!direct_) {
      std::printf(
          "stored Hamiltonian (root): %.1f MB of elements, %.1f MB "
          "of row lists and thread tables\n",
          stored_.element_bytes() * mb, stored_.overhead_bytes() * mb);
      if (adapt_) {
        const size_t n = dets_.size();
        std::printf(
            "memory (root): limit %.1f MB; %.1f MB counted for the string "
            "tables, the determinants and the larger of extending the tables "
            "and %zu vectors%s, which leaves %.1f MB for the Hamiltonian\n",
            memory_limit_ * mb, OtherBytes(n, one_sweep_) * mb, num_vectors_,
            one_sweep_ ? " with one more per thread" : " (two-sweep product)",
            HamiltonianBudget(n, one_sweep_) * mb);
        if (!all_rows_stored()) {
          std::printf(
              "stored Hamiltonian (root): the rows of %zu of %zu determinants "
              "are not stored and are applied directly\n",
              num_direct_rows_, n);
        }
      }
      if (stored_.num_values() < stored_.num_elements()) {
        std::printf(
            "stored Hamiltonian (root): values held for %zu of %zu elements; "
            "the others are worked out in every product\n",
            stored_.num_values(), stored_.num_elements());
      }
    }
    std::printf("determinants: %.1f MB\n", dets_.size() * sizeof(Det<W>) * mb);
    if (direct_) {
      std::printf(
          "Hamiltonian: not stored; %llu distinct strings, %llu "
          "entries in their relation lists, %.2f s for the tables\n",
          static_cast<unsigned long long>(stats_.num_strings),
          static_cast<unsigned long long>(stats_.num_links),
          stats_.seconds_prepare);
    } else {
      PrintConnectStats(stats_);
    }
  }

 private:
  // The sum of two operators of the same dimension; the diagonal is that of
  // the first.
  class SumOperator : public LinearOperator<S> {
   public:
    SumOperator(const LinearOperator<S>& a, const LinearOperator<S>& b)
        : a_(a), b_(b) {}
    size_t dim() const override { return a_.dim(); }
    const std::vector<double>& diagonal() const override {
      return a_.diagonal();
    }
    void Apply(const S* x, S* y) const override {
      const long n = static_cast<long>(dim());
      a_.Apply(x, y);
      std::vector<S> other(n);
      b_.Apply(x, other.data());
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; ++i) y[i] += other[i];
    }

   private:
    const LinearOperator<S>& a_;
    const LinearOperator<S>& b_;
  };

  bool all_rows_stored() const { return num_direct_rows_ == 0; }

  // Bytes of what this process holds besides the Hamiltonian for n
  // determinants. All the time: the string tables, the determinants (the
  // list, the root's copy, the sorted copy) and a twentieth of the limit
  // for everything else. In addition, at different times: what extending
  // the tables takes, while the Hamiltonian is being extended; and the
  // vectors of the eigensolver while it runs, with, for the one-sweep
  // product of the stored part, a vector per thread. Only the larger of
  // the two counts.
  size_t OtherBytes(size_t n, bool one_sweep) const {
    const size_t solving =
        num_vectors_ * n * sizeof(S) +
        (one_sweep ? static_cast<size_t>(omp_get_max_threads()) * n * sizeof(S)
                   : 0);
    return table_memory_.shared + table_memory_.kept + 3 * n * sizeof(Det<W>) +
           memory_limit_ / 20 + std::max(table_memory_.transient, solving);
  }
  size_t HamiltonianBudget(size_t n, bool one_sweep) const {
    const size_t other = OtherBytes(n, one_sweep);
    return memory_limit_ > other ? memory_limit_ - other : 0;
  }

  // Adds the rows of the determinants from first_new on within the memory
  // limit (see "adapt" in input.h). Collective.
  void ExtendWithinMemory(size_t first_new) {
    const size_t n = dets_.size();
    // The connections are judged against the larger budget, that of the
    // two-sweep product, which needs no vectors of its own.
    const size_t budget_two_sweeps = HamiltonianBudget(n, false);
    // The connections of the new rows, as far as they fit: the budget less
    // the connections already held (values can make way). The rows that do
    // not fit are applied directly from then on. The blocks are appended
    // even if empty, so that the stored part knows of the new determinants.
    // The two-sweep product keeps, for every stored row, where its columns
    // pass from one thread's rows to the next.
    const size_t held =
        stored_.num_elements() * sizeof(DetIndex) + stored_.overhead_bytes() +
        (n - first_new) * static_cast<size_t>(omp_get_max_threads()) *
            sizeof(uint32_t);
    const size_t room = budget_two_sweeps > held ? budget_two_sweeps - held : 0;
    std::vector<uint8_t> left_out(n - first_new, 0);
    std::vector<CsrBlock<S>> blocks = internal::BuildGroupBlocks(
        dets_, first_new, ints_, connect_.threshold, *tables_, &stats_,
        /*store_values=*/false, room, left_out.data());
    // Every rank must know which rows the others left out.
    if (comm_.size() > 1) {
      std::vector<double> flags(left_out.begin(), left_out.end());
      comm_.Sum(flags.data(), flags.size());
      for (size_t k = 0; k < flags.size(); ++k) left_out[k] = flags[k] > 0.0;
    }
    direct_rows_.resize(first_new);
    direct_rows_.insert(direct_rows_.end(), left_out.begin(), left_out.end());
    num_direct_rows_ = 0;
    for (uint8_t flag : direct_rows_) num_direct_rows_ += flag;
    stored_.AppendBlocks(std::move(blocks));
    // The one-sweep product, which reads the matrix once but takes a vector
    // per thread, if the connections leave room for those.
    one_sweep_ =
        stored_.num_elements() * sizeof(DetIndex) + stored_.overhead_bytes() <=
        HamiltonianBudget(n, true);
    stored_.set_product(one_sweep_ ? SparseH<S>::Product::kAutomatic
                                   : SparseH<S>::Product::kTwoSweeps);
    const size_t budget = HamiltonianBudget(n, one_sweep_);
    // Values: released from the newest blocks until the budget is kept,
    // then worked out for the blocks that have none while they fit.
    const size_t num_blocks = stored_.blocks().size();
    size_t bytes = stored_.element_bytes() + stored_.overhead_bytes();
    for (size_t b = num_blocks; b-- > 0 && bytes > budget;) {
      const size_t freed = stored_.blocks()[b].val.size() * sizeof(S);
      stored_.DropBlockValues(b);
      bytes -= freed;
    }
    for (size_t b = 0; b < num_blocks; ++b) {
      const CsrBlock<S>& block = stored_.blocks()[b];
      if (block.has_values()) continue;
      const size_t needed = block.col.size() * sizeof(S);
      if (bytes + needed > budget) continue;
      stored_.FillBlockValues(b);
      bytes += needed;
    }
  }

  // The elements of stored blocks that hold columns only.
  void Fill(DetIndex row, const DetIndex* cols, size_t count,
            S* values) const override {
    const Det<W>& bra = dets_[row];
    for (size_t k = 0; k < count; ++k) {
      values[k] = Hij(bra, dets_[cols[k]], ints_);
    }
  }

  void ResetStored() {
    direct_rows_.clear();
    num_direct_rows_ = 0;
    stored_ = SparseH<S>();
    stored_.set_apply_diagonal(comm_.is_root());
    stored_.set_element_source(this);
  }

  const Integrals<S>& ints_;
  const Comm& comm_;
  bool direct_;
  bool store_values_;
  // "adapt": the memory limit, the number of vectors of the eigensolver
  // counted against it, and the rows that are not stored (one byte per
  // determinant, 1: applied directly) with their number.
  bool adapt_;
  size_t memory_limit_;
  size_t num_vectors_;
  std::vector<uint8_t> direct_rows_;
  size_t num_direct_rows_ = 0;
  // Whether the stored part may use the one-sweep product.
  bool one_sweep_ = true;
  bool incremental_;
  typename internal::GroupStringTables<W>::Memory table_memory_;
  ConnectOptions connect_;
  Span<Det<W>> dets_;
  SparseH<S> stored_;
  std::unique_ptr<internal::GroupStringTables<W>> tables_;
  ConnectStats stats_;
};

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
                  Span<Det<W>> dets, const VariationalH<W, S>& h,
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
      const DensityMatrices<S> dm =
          h.Density(vectors.Root(k), vectors.Root(k), input.rdm.two);
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
      const DensityMatrices<S> dm =
          h.Density(vectors.Root(0), vectors.Root(k), /*with_two=*/false);
      if (comm.is_root()) WriteDensityMatrices(input.prefix, 0, k, dm);
    }
  }
}

}  // namespace zshci

#endif  // ZSHCI_SRC_DRIVER_COMMON_H_
