#include "distributed_h.h"

#include <algorithm>

namespace zshci {

namespace {

template <typename S>
double* AsDoubles(S* data) {
  return reinterpret_cast<double*>(data);
}

}  // namespace

template <typename S>
void DistributedH<S>::Apply(const S* x, S* y) const {
  int command = kApply;
  comm_.BcastValue(&command);
  std::copy(x, x + dim(), x_.data());
  x_.ShareFromRoot(comm_);
  local_.Multiply(x, y);
  comm_.SumToRoot(AsDoubles(y), kDoublesPerScalar<S> * dim());
}

template <typename S>
void DistributedH<S>::Serve() const {
  while (true) {
    int command = kStop;
    comm_.BcastValue(&command);
    if (command == kStop) return;
    x_.ShareFromRoot(comm_);
    local_.Multiply(x_.data(), y_.data());
    comm_.SumToRoot(AsDoubles(y_.data()), kDoublesPerScalar<S> * dim());
  }
}

template <typename S>
void DistributedH<S>::Stop() const {
  int command = kStop;
  comm_.BcastValue(&command);
}

template <typename S>
DavidsonResult SolveDistributed(const SparseH<S>& local,
                                const DavidsonOptions& options,
                                const Comm& comm,
                                std::vector<std::vector<S>>* vectors) {
  DavidsonResult result;
  const DistributedH<S> h(local, comm);
  if (comm.is_root()) {
    result = Davidson(h, options, vectors);
    h.Stop();
  } else {
    h.Serve();
  }

  comm.Bcast(&result.energies);
  comm.Bcast(&result.residual_norms);
  comm.BcastValue(&result.num_iterations);
  int converged = result.converged ? 1 : 0;
  comm.BcastValue(&converged);
  result.converged = converged != 0;
  return result;
}

template class DistributedH<double>;
template class DistributedH<Complex>;
template DavidsonResult SolveDistributed(const SparseH<double>&,
                                         const DavidsonOptions&, const Comm&,
                                         std::vector<std::vector<double>>*);
template DavidsonResult SolveDistributed(const SparseH<Complex>&,
                                         const DavidsonOptions&, const Comm&,
                                         std::vector<std::vector<Complex>>*);

}  // namespace zshci
