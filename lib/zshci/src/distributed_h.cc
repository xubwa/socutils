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

// The few-vector solver (fci_davidson.h) on this process alone. The start
// vectors are moved into its basis one by one and the eigenvectors back,
// so that at most one vector more than the solver's is held.
template <typename S>
DavidsonResult FewVectorDavidson(const LinearOperator<S>& h,
                                 const DavidsonOptions& options,
                                 std::vector<std::vector<S>>* vectors) {
  const size_t n = h.dim();
  FciDavidsonOptions few;
  few.num_roots = static_cast<int>(vectors->size());
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
