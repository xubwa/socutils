#include "fci.h"

#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "big_array.h"
#include "fci_davidson.h"
#include "gas_space.h"
#include "output.h"

// MKL's single dynamic library chooses its threading at run time. The kernel
// calls BLAS from inside its own parallel region, one thread per call, so
// the sequential layer is what it needs; it also keeps a second OpenMP
// runtime out of the process.
extern "C" int MKL_Set_Threading_Layer(int) __attribute__((weak));

namespace zshci {
namespace {

constexpr int kMklThreadingSequential = 1;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// Bytes this process may use: the memory limit of its control group (a
// Slurm job's allocation) or of the machine, whichever is lower.
double AvailableBytes() {
  double limit = std::numeric_limits<double>::infinity();
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  double kb = 0.0;
  while (meminfo >> key >> kb) {
    if (key == "MemTotal:") {
      limit = kb * 1024.0;
      break;
    }
    meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  std::ifstream cgroup("/proc/self/cgroup");
  std::string line;
  while (std::getline(cgroup, line)) {
    const size_t a = line.find(':');
    const size_t b = line.find(':', a + 1);
    if (a == std::string::npos || b == std::string::npos) continue;
    const std::string controllers = line.substr(a + 1, b - a - 1);
    const std::string path = line.substr(b + 1);
    std::string file;
    if (controllers == "memory") {
      file = "/sys/fs/cgroup/memory" + path + "/memory.limit_in_bytes";
    } else if (controllers.empty()) {
      file = "/sys/fs/cgroup" + path + "/memory.max";
    } else {
      continue;
    }
    std::ifstream in(file);
    double value = 0.0;
    if (in >> value && value > 0.0) limit = std::min(limit, value);
  }
  return limit;
}

double PeakGiB() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss * 1024.0 / kGiB;
}

std::string Join(const std::vector<int>& values) {
  std::ostringstream out;
  for (size_t k = 0; k < values.size(); ++k) out << (k ? " " : "") << values[k];
  return out.str();
}

template <typename S>
class GasOperator : public CiOperator<S> {
 public:
  GasOperator(const GasSpace& space, const std::vector<S>& v, GasKernel kernel)
      : space_(space), v_(v), kernel_(kernel) {}
  void Apply(const S* x, S* y) const override {
    space_.Sigma(v_, x, y, kernel_);
  }

 private:
  const GasSpace& space_;
  const std::vector<S>& v_;
  GasKernel kernel_;
};

}  // namespace

template <typename S>
void RunFci(const Input& input, const Integrals<S>& ints, const Comm& comm) {
  if (comm.size() > 1) {
    throw std::runtime_error("fci mode runs on one rank (threads only)");
  }
  if (MKL_Set_Threading_Layer) {
    MKL_Set_Threading_Layer(kMklThreadingSequential);
  }
  const FciInput& in = input.fci;
  const int n = ints.num_orbs();
  const int nelec = in.nelec > 0 ? in.nelec : ints.num_elec();

  std::vector<GasConstraint> constraints = in.gas;
  for (const RestrictionInput& r : input.restrictions) {
    GasConstraint c;
    c.orbitals = r.orbitals;
    c.min_elec = r.min_elec;
    c.max_elec = r.max_elec;
    constraints.push_back(c);
  }
  const GasLayout layout =
      MakeGasLayout(n, constraints, in.max_subgroup, in.groups_a, in.groups_b);
  const std::vector<LabelSet>& labels = ints.symmetry();
  std::vector<int> targets = in.target;
  if (targets.empty()) {
    for (const LabelSet& set : labels) {
      int sum = 0;
      for (int p = 0; p < nelec; ++p) sum += set.labels[p];
      targets.push_back(sum);
    }
  }
  if (targets.size() != labels.size()) {
    throw std::runtime_error("fci.target needs one entry per label set (" +
                             std::to_string(labels.size()) + ")");
  }

  double time = omp_get_wtime();
  const GasSpace space(n, nelec, layout, labels, targets);
  const std::vector<int64_t> info = space.Info();
  const int64_t dim = space.dim();
  std::printf("fci: %d electrons in %d spinors, label targets [%s]\n", nelec, n,
              Join(targets).c_str());
  std::printf("fci: spinors in the order of the kernel: %s\n",
              Join(layout.perm).c_str());
  std::printf("fci: sub-groups %s, the first %d on side A\n",
              Join(layout.sub_sizes).c_str(), layout.num_sub_a);
  std::printf(
      "fci: %ld determinants; strings %ld / %ld, classes %ld / %ld, "
      "buckets %ld / %ld, blocks %ld; plan %.2f s\n",
      static_cast<long>(dim), static_cast<long>(info[0]),
      static_cast<long>(info[1]), static_cast<long>(info[2]),
      static_cast<long>(info[3]), static_cast<long>(info[4]),
      static_cast<long>(info[5]), static_cast<long>(info[6]),
      omp_get_wtime() - time);

  GasKernel kernel = in.kernel == "reference" ? GasKernel::kReference
                     : in.kernel == "block"   ? GasKernel::kBlock
                                              : GasKernel::kBlas;
  if (kernel != GasKernel::kReference && !space.block_ok()) {
    std::printf("fci: a side holds more than 64 spinors: reference kernel\n");
    kernel = GasKernel::kReference;
  }

  // Memory: the Davidson vectors, the diagonal, the kernel's workspace, the
  // integrals and the plan.
  const double vector_bytes = static_cast<double>(dim) * sizeof(S);
  const bool dense = dim <= input.davidson.dense_dim;
  const int num_vectors = in.benchmark_sigmas > 0 ? 2
                          : dense                 ? 0
                                                  : 2 * in.max_space;
  const int kernel_vectors = kernel == GasKernel::kBlas ? 2 : 0;
  const double npair = n * (n - 1) / 2.0;
  const double other_bytes =
      3.0 * npair * npair * sizeof(S) + 40.0 * (info[0] + info[1]) + (1 << 30);
  const double need =
      (num_vectors + kernel_vectors) * vector_bytes + 8.0 * dim + other_bytes;
  const double available =
      input.memory_gb > 0.0 ? input.memory_gb * kGiB : AvailableBytes();
  std::printf(
      "fci memory: one vector %.2f GiB (%s); %d Davidson vectors (max_space "
      "%d, each with its product) %.2f GiB, diagonal %.2f GiB, %d vectors "
      "of kernel workspace %.2f GiB, integrals, plan and the rest about "
      "%.2f GiB; total %.2f GiB of %.2f GiB available\n",
      vector_bytes / kGiB, ScalarTraits<S>::kName, num_vectors, in.max_space,
      num_vectors * vector_bytes / kGiB, 8.0 * dim / kGiB, kernel_vectors,
      kernel_vectors * vector_bytes / kGiB, other_bytes / kGiB, need / kGiB,
      available / kGiB);
  if (in.dry_run) return;
  if (need > available) {
    throw std::runtime_error(
        "fci: the calculation needs more memory than is available (see "
        "\"fci memory\" above; \"memory\" sets the gigabytes to assume)");
  }

  time = omp_get_wtime();
  const std::vector<S> v = AntisymmetrizedPairs(ints, layout.perm, nelec);
  BigArray<double> diagonal(dim, in.huge_pages);
  space.Diagonal(v, diagonal.data());
  std::printf("fci: integrals over pairs and the diagonal, %.2f s\n",
              omp_get_wtime() - time);
  const GasOperator<S> h(space, v, kernel);

  if (in.benchmark_sigmas > 0) {
    BigArray<S> x(dim, in.huge_pages), y(dim, in.huge_pages);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(dim); ++i) {
      x[i] = S(1.0 / (1.0 + i % 97));
    }
    double best = 1e300;
    for (int k = 0; k < in.benchmark_sigmas; ++k) {
      time = omp_get_wtime();
      h.Apply(x.data(), y.data());
      const double seconds = omp_get_wtime() - time;
      best = std::min(best, seconds);
      std::printf("fci benchmark: sigma %d, %.2f s, peak memory %.2f GiB\n", k,
                  seconds, PeakGiB());
    }
    std::printf("fci benchmark: best %.2f s per sigma, %d threads\n", best,
                omp_get_max_threads());
    return;
  }

  FciDavidsonOptions options;
  options.num_roots = input.nroots;
  options.tol = input.davidson.tol;
  options.max_iterations = input.davidson.max_iterations;
  options.max_space = in.max_space;
  options.level_shift = in.level_shift;
  options.dense_dim = input.davidson.dense_dim;
  if (in.checkpoint != "none") options.checkpoint = in.checkpoint;
  options.checkpoint_every = in.checkpoint_every;
  options.restart = in.restart;
  options.huge_pages = in.huge_pages;
  options.fingerprint = space.Fingerprint();
  const FciDavidsonResult<S> result =
      FciDavidson(h, diagonal.data(), VectorLayout::Whole(dim), options);

  std::vector<double> energies;
  for (int k = 0; k < input.nroots; ++k) {
    energies.push_back(result.energies[k] + ints.core_energy());
  }
  std::printf(
      "fci: %s after %d iterations, %d products with H in %.1f s, %.1f s "
      "in all\n",
      result.converged ? "converged" : "NOT converged", result.num_iterations,
      result.num_products, result.seconds_products, result.seconds_total);
  for (int k = 0; k < input.nroots; ++k) {
    std::printf("fci: root %d  E = %.12f  residual %.2e\n", k, energies[k],
                result.residual_norms[k]);
  }
  std::printf("fci: peak memory %.2f GiB\n", PeakGiB());
  WriteEnergies(input.prefix, energies);
}

template void RunFci(const Input&, const Integrals<double>&, const Comm&);
template void RunFci(const Input&, const Integrals<Complex>&, const Comm&);

}  // namespace zshci
