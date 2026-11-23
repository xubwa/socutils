#include "distributed_h.h"

#include <omp.h>

#include <algorithm>
#include <cstdio>

#include "fci_davidson.h"
#include "profile.h"

namespace zshci {

namespace {

template <typename S>
double* AsDoubles(S* data) {
  return reinterpret_cast<double*>(data);
}

template <typename S>
class OperatorAdapter : public CiOperator<S> {
 public:
  explicit OperatorAdapter(const LinearOperator<S>& h) : h_(h) {}
  void Apply(const S* x, S* y) const override { h_.Apply(x, y); }

 private:
  const LinearOperator<S>& h_;
};

// Options of the few-vector solver from those of the eigensolver.
FciDavidsonOptions FewVectorOptions(const DavidsonOptions& options,
                                    int num_roots) {
  FciDavidsonOptions few;
  few.num_roots = num_roots;
  few.tol = options.tol;
  few.max_iterations = options.max_iterations;
  few.max_space = options.max_space * few.num_roots;
  few.level_shift = options.level_shift;
  few.dense_dim = options.dense_dim;
  few.checkpoint = options.checkpoint_every > 0 ? options.checkpoint : "";
  few.checkpoint_every = std::max(1, options.checkpoint_every);
  few.restart = options.restart && !few.checkpoint.empty();
  few.restart_if_matching = true;
  few.fingerprint = options.fingerprint;
  few.huge_pages = true;
  few.print = options.print;
  few.label = "davidson";
  return few;
}

// The few-vector solver (fci_davidson.h) on this process alone. The start
// vectors are moved into its basis one by one and the eigenvectors back,
// so that at most one vector more than the solver's is held.
template <typename S>
DavidsonResult FewVectorDavidson(const LinearOperator<S>& h,
                                 const DavidsonOptions& options,
                                 std::vector<std::vector<S>>* vectors) {
  const size_t n = h.dim();
  const FciDavidsonOptions few =
      FewVectorOptions(options, static_cast<int>(vectors->size()));
  std::vector<BigArray<S>> start;
  for (std::vector<S>& v : *vectors) {
    if (v.size() != n) throw std::invalid_argument("Davidson: bad guess size");
    BigArray<S> copy(n, few.huge_pages);
    std::copy(v.begin(), v.end(), copy.data());
    v = std::vector<S>();
    start.push_back(std::move(copy));
  }
  const double start_time = omp_get_wtime();
  const OperatorAdapter<S> adapter(h);
  FciDavidsonResult<S> r =
      FciDavidson<S>(adapter, h.diagonal().data(), VectorLayout::Whole(n), few,
                     Comm(), &start);
  start.clear();
  for (size_t k = 0; k < vectors->size(); ++k) {
    (*vectors)[k].assign(r.vectors[k].data(), r.vectors[k].data() + n);
    r.vectors[k] = BigArray<S>();
  }
  DavidsonResult result;
  result.energies = r.energies;
  result.residual_norms = r.residual_norms;
  result.num_iterations = r.num_iterations;
  result.num_products = r.num_products;
  result.seconds_products = r.seconds_products;
  result.seconds_total = omp_get_wtime() - start_time;
  result.converged = r.converged;
  result.restarted = r.restarted;
  return result;
}

// H on vectors held in segments (rank r: elements [begin[r], begin[r+1])):
// x is gathered once per node, every rank multiplies with its part of H,
// and the products are summed onto the owners of the segments. With
// `active`, as MaskedOperator: P H P plus `shift` on the elements out of
// the space.
template <typename S>
class SegmentOperator : public CiOperator<S> {
 public:
  SegmentOperator(const LinearOperator<S>& local, const Comm& comm,
                  const std::vector<uint64_t>& begin,
                  const std::vector<uint8_t>* active, double shift)
      : local_(local),
        comm_(comm),
        begin_(begin),
        active_(active),
        shift_(shift),
        x_(begin.back(), comm),
        y_(begin.back()) {
    for (uint64_t b : begin) begin_doubles_.push_back(kDoublesPerScalar<S> * b);
  }

  void Apply(const S* x, S* y) const override {
    double t = omp_get_wtime();
    const long first = static_cast<long>(begin_[comm_.rank()]);
    const long count = static_cast<long>(begin_[comm_.rank() + 1]) - first;
    S* const mine = x_.data() + first;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < count; ++i) {
      mine[i] = active_ == nullptr || (*active_)[i] ? x[i] : S(0.0);
    }
    comm_.GatherShared(x_.data(), begin_, sizeof(S));
    double now = omp_get_wtime();
    seconds_share_ += now - t;
    t = now;
    local_.Apply(x_.data(), y_.data());
    now = omp_get_wtime();
    seconds_local_ += now - t;
    t = now;
    comm_.SumScatter(AsDoubles(y_.data()), AsDoubles(y), begin_doubles_);
    if (active_ != nullptr) {
#pragma omp parallel for schedule(static)
      for (long i = 0; i < count; ++i) {
        if (!(*active_)[i]) y[i] = shift_ * x[i];
      }
    }
    seconds_reduce_ += omp_get_wtime() - t;
  }

  double seconds_local() const { return seconds_local_; }
  double seconds_share() const { return seconds_share_; }
  double seconds_reduce() const { return seconds_reduce_; }

 private:
  const LinearOperator<S>& local_;
  const Comm& comm_;
  const std::vector<uint64_t>& begin_;
  std::vector<uint64_t> begin_doubles_;
  const std::vector<uint8_t>* active_;
  const double shift_;
  // The whole x, once per node, and this rank's product.
  mutable SharedArray<S> x_;
  mutable RawArray<S> y_;
  mutable double seconds_local_ = 0.0;
  mutable double seconds_share_ = 0.0;
  mutable double seconds_reduce_ = 0.0;
};

// Whether the few-vector solver runs with its vectors in segments over the
// ranks: with several ranks, the ranks of each node together, and a space
// that is not diagonalized as a full matrix and is large enough to divide.
template <typename S>
bool UseSegments(const LinearOperator<S>& h, const DavidsonOptions& options,
                 const Comm& comm) {
  const size_t n = h.dim();
  return options.few_vectors && comm.size() > 1 && comm.nodes_contiguous() &&
         n > static_cast<size_t>(options.dense_dim) &&
         n >= 4096 * static_cast<size_t>(comm.size());
}

// The few-vector solver with its vectors in segments over all ranks
// (collective). The root's start vectors are handed out and the
// eigenvectors collected on the root; the diagonal and `active` are read
// on the root only. Prints how the products divided.
template <typename S>
DavidsonResult SegmentedDavidson(const LinearOperator<S>& local,
                                 const DavidsonOptions& options,
                                 const Comm& comm,
                                 std::vector<std::vector<S>>* vectors,
                                 const std::vector<uint8_t>* active) {
  const double start_time = omp_get_wtime();
  const size_t n = local.dim();
  const int p = comm.size();
  const int rank = comm.rank();
  std::vector<uint64_t> begin(p + 1);
  for (int r = 0; r <= p; ++r) begin[r] = n * r / p;
  const size_t local_n = begin[rank + 1] - begin[rank];

  int num_roots = comm.is_root() ? static_cast<int>(vectors->size()) : 0;
  comm.BcastValue(&num_roots);
  int masked = active != nullptr ? 1 : 0;
  comm.BcastValue(&masked);
  FciDavidsonOptions few = FewVectorOptions(options, num_roots);
  comm.BcastValue(&few.fingerprint);
  few.print = options.print && comm.is_root();

  // The diagonal (shifted out of the space, as MaskedOperator) and the
  // mask, in segments.
  std::vector<double> diagonal(local_n);
  std::vector<uint8_t> active_part(masked ? local_n : 0);
  double shift = 0.0;
  {
    std::vector<double> whole;
    if (comm.is_root()) {
      whole = local.diagonal();
      if (masked) {
        for (double d : whole) shift = std::max(shift, d);
        shift += 1.0;
        for (size_t i = 0; i < n; ++i) {
          if (!(*active)[i]) whole[i] = shift;
        }
      }
    }
    comm.BcastValue(&shift);
    comm.Scatter(whole.data(), diagonal.data(), begin, sizeof(double));
    if (masked) {
      comm.Scatter(comm.is_root() ? active->data() : nullptr,
                   active_part.data(), begin, sizeof(uint8_t));
    }
  }

  std::vector<BigArray<S>> start;
  for (int k = 0; k < num_roots; ++k) {
    BigArray<S> part(local_n, few.huge_pages);
    if (comm.is_root() && (*vectors)[k].size() != n) {
      throw std::invalid_argument("Davidson: bad guess size");
    }
    comm.Scatter(comm.is_root() ? (*vectors)[k].data() : nullptr,
                 part.data(), begin, sizeof(S));
    if (comm.is_root()) (*vectors)[k] = std::vector<S>();
    start.push_back(std::move(part));
  }

  VectorLayout layout;
  layout.segments.push_back({static_cast<int64_t>(begin[rank]),
                             static_cast<int64_t>(begin[rank + 1])});
  layout.global_dim = static_cast<int64_t>(n);
  const SegmentOperator<S> h(local, comm, begin,
                             masked ? &active_part : nullptr, shift);
  FciDavidsonResult<S> r =
      FciDavidson<S>(h, diagonal.data(), layout, few, comm, &start);
  start.clear();
  for (int k = 0; k < num_roots; ++k) {
    if (comm.is_root()) (*vectors)[k].resize(n);
    comm.Gather(r.vectors[k].data(),
                comm.is_root() ? (*vectors)[k].data() : nullptr, begin,
                sizeof(S));
    r.vectors[k] = BigArray<S>();
  }

  const std::vector<double> times = comm.GatherAll(std::vector<double>{
      h.seconds_local(), h.seconds_share(), h.seconds_reduce()});
  if (comm.is_root() && r.num_products > 0) {
    std::vector<double> work;
    for (size_t i = 0; i < times.size(); i += 3) work.push_back(times[i]);
    std::printf(
        "distributed products: %d, local work per rank %.1f to %.1f s, "
        "root communicating and waiting %.1f s (vectors in segments)\n",
        r.num_products, *std::min_element(work.begin(), work.end()),
        *std::max_element(work.begin(), work.end()),
        h.seconds_share() + h.seconds_reduce());
    if (ProfileOn()) {
      std::printf("  profile dav: per rank local/share/reduce (s):");
      for (size_t i = 0; i < times.size(); i += 3) {
        std::printf(" [%.1f %.1f %.1f]", times[i], times[i + 1], times[i + 2]);
      }
      std::printf("\n");
    }
  }
  DavidsonResult result;
  result.energies = r.energies;
  result.residual_norms = r.residual_norms;
  result.num_iterations = r.num_iterations;
  result.num_products = r.num_products;
  result.seconds_products = r.seconds_products;
  result.seconds_total = omp_get_wtime() - start_time;
  result.converged = r.converged;
  result.restarted = r.restarted;
  return result;
}

template <typename S>
DavidsonResult RunSolver(const LinearOperator<S>& h,
                         const DavidsonOptions& options,
                         std::vector<std::vector<S>>* vectors) {
  if (options.few_vectors) return FewVectorDavidson(h, options, vectors);
  return Davidson(h, options, vectors);
}

}  // namespace

template <typename S>
void DistributedH<S>::Apply(const S* x, S* y) const {
  double t = omp_get_wtime();
  int command = kApply;
  comm_.BcastValue(&command);
  if (comm_.size() > 1) {
    const long n = static_cast<long>(dim());
    S* shared = x_.data();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) shared[i] = x[i];
    x_.ShareFromRoot(comm_);
  }
  double now = omp_get_wtime();
  seconds_share_ += now - t;
  t = now;
  local_.Apply(x, y);
  now = omp_get_wtime();
  seconds_local_ += now - t;
  t = now;
  comm_.SumToRoot(AsDoubles(y), kDoublesPerScalar<S> * dim());
  seconds_reduce_ += omp_get_wtime() - t;
}

template <typename S>
void DistributedH<S>::Serve() const {
  while (true) {
    int command = kStop;
    double t = omp_get_wtime();
    comm_.BcastValue(&command);
    if (command == kStop) return;
    x_.ShareFromRoot(comm_);
    double now = omp_get_wtime();
    seconds_share_ += now - t;
    t = now;
    local_.Apply(x_.data(), y_.data());
    now = omp_get_wtime();
    seconds_local_ += now - t;
    t = now;
    comm_.SumToRoot(AsDoubles(y_.data()), kDoublesPerScalar<S> * dim());
    seconds_reduce_ += omp_get_wtime() - t;
  }
}

template <typename S>
void DistributedH<S>::Stop() const {
  int command = kStop;
  comm_.BcastValue(&command);
}

template <typename S>
DavidsonResult SolveDistributed(const LinearOperator<S>& local,
                                const DavidsonOptions& options,
                                const Comm& comm,
                                std::vector<std::vector<S>>* vectors,
                                const std::vector<uint8_t>* active) {
  if (UseSegments(local, options, comm)) {
    return SegmentedDavidson(local, options, comm, vectors, active);
  }
  DavidsonResult result;
  const DistributedH<S> h(local, comm);
  if (comm.is_root()) {
    if (active != nullptr) {
      const MaskedOperator<S> masked(h, *active);
      result = RunSolver<S>(masked, options, vectors);
    } else {
      result = RunSolver<S>(h, options, vectors);
    }
    h.Stop();
  } else {
    h.Serve();
  }

  if (comm.size() > 1) {
    // How the products divided: the local work of every rank and, on the
    // root, the communication (with the wait for the slowest rank).
    const std::vector<double> times = comm.GatherAll(std::vector<double>{
        h.seconds_local(), h.seconds_share(), h.seconds_reduce()});
    if (comm.is_root() && result.num_products > 0) {
      std::vector<double> local;
      for (size_t r = 0; r < times.size(); r += 3) local.push_back(times[r]);
      std::printf(
          "distributed products: %d, local work per rank %.1f to "
          "%.1f s, root communicating and waiting %.1f s\n",
          result.num_products, *std::min_element(local.begin(), local.end()),
          *std::max_element(local.begin(), local.end()), h.seconds_comm());
      if (ProfileOn()) {
        std::printf("  profile dav: per rank local/share/reduce (s):");
        for (size_t r = 0; r < times.size(); r += 3) {
          std::printf(" [%.1f %.1f %.1f]", times[r], times[r + 1],
                      times[r + 2]);
        }
        std::printf("\n");
      }
    }
  }
  comm.Bcast(&result.energies);
  comm.Bcast(&result.residual_norms);
  comm.BcastValue(&result.num_iterations);
  comm.BcastValue(&result.num_products);
  int converged = result.converged ? 1 : 0;
  comm.BcastValue(&converged);
  result.converged = converged != 0;
  return result;
}

template class DistributedH<double>;
template class DistributedH<Complex>;
template DavidsonResult SolveDistributed(const LinearOperator<double>&,
                                         const DavidsonOptions&, const Comm&,
                                         std::vector<std::vector<double>>*,
                                         const std::vector<uint8_t>*);
template DavidsonResult SolveDistributed(const LinearOperator<Complex>&,
                                         const DavidsonOptions&, const Comm&,
                                         std::vector<std::vector<Complex>>*,
                                         const std::vector<uint8_t>*);

}  // namespace zshci
