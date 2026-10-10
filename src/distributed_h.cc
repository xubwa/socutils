#include "distributed_h.h"

#include <algorithm>

namespace zshci {

namespace {

double* AsDoubles(Complex* data) { return reinterpret_cast<double*>(data); }

}  // namespace

void DistributedH::Apply(const Complex* x, Complex* y) const {
  int command = kApply;
  comm_.BcastValue(&command);
  std::copy(x, x + dim(), x_.data());
  x_.ShareFromRoot(comm_);
  local_.Multiply(x, y);
  comm_.SumToRoot(AsDoubles(y), 2 * dim());
}

void DistributedH::Serve() const {
  while (true) {
    int command = kStop;
    comm_.BcastValue(&command);
    if (command == kStop) return;
    x_.ShareFromRoot(comm_);
    local_.Multiply(x_.data(), y_.data());
    comm_.SumToRoot(AsDoubles(y_.data()), 2 * dim());
  }
}

void DistributedH::Stop() const {
  int command = kStop;
  comm_.BcastValue(&command);
}

DavidsonResult SolveDistributed(const SparseH& local,
                                const DavidsonOptions& options,
                                const Comm& comm,
                                std::vector<std::vector<Complex>>* vectors) {
  DavidsonResult result;
  const DistributedH h(local, comm);
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

}  // namespace zshci
