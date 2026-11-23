#include "fci.h"

#include <omp.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "big_array.h"
#include "fci_davidson.h"
#include "fci_dist.h"
#include "gas_space.h"
#include "output.h"
#include "rdm.h"

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

// Number of ranks on this rank's node (they share its memory).
int RanksOnNode(const Comm& comm) {
  char name[256] = {};
  gethostname(name, sizeof(name) - 1);
  uint64_t hash = 1469598103934665603ull;
  for (const char* c = name; *c; ++c) hash = (hash ^ *c) * 1099511628211ull;
  const std::vector<uint64_t> all = comm.GatherAll(std::vector<uint64_t>{hash});
  return static_cast<int>(std::count(all.begin(), all.end(), hash));
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

// Density matrices over the original spinors from the pair density of the
// kernel, in the convention of rdm.h: D2(p,q,r,s) = <bra| p^+ q^+ s r |ket>.
template <typename S>
DensityMatrices<S> GasDensity(const GasSpace& space, const S* bra, const S* ket,
                              bool with_two) {
  const int n = space.num_orbs();
  const size_t npair = static_cast<size_t>(n) * (n - 1) / 2;
  const std::vector<int>& perm = space.layout().perm;
  std::vector<S> g(npair * npair);
  space.PairDensity(bra, ket, g.data());
  DensityMatrices<S> dm;
  dm.num_orbs = n;
  dm.one.assign(static_cast<size_t>(n) * n, S(0.0));
  if (with_two) dm.two.assign(static_cast<size_t>(n) * n * n * n, S(0.0));
  std::vector<int> first(npair), second(npair);
  for (int q = 1, k = 0; q < n; ++q) {
    for (int p = 0; p < q; ++p, ++k) {
      first[k] = perm[p];
      second[k] = perm[q];
    }
  }
  for (size_t bra_pair = 0; bra_pair < npair; ++bra_pair) {
    const int a = first[bra_pair], b = second[bra_pair];
    for (size_t ket_pair = 0; ket_pair < npair; ++ket_pair) {
      const S value = g[bra_pair * npair + ket_pair];
      if (value == S(0.0)) continue;
      const int c = first[ket_pair], d = second[ket_pair];
      if (with_two) {
        dm.Two(a, b, c, d) = value;
        dm.Two(b, a, c, d) = -value;
        dm.Two(a, b, d, c) = -value;
        dm.Two(b, a, d, c) = value;
      }
      if (b == d) dm.One(a, c) += value;
      if (a == d) dm.One(b, c) -= value;
      if (b == c) dm.One(a, d) -= value;
      if (a == c) dm.One(b, d) += value;
    }
  }
  for (S& x : dm.one) x /= static_cast<double>(space.num_elec() - 1);
  return dm;
}

// wfn.h5 as the other modes write it: the determinants over the original
// spinors and the coefficients in their phase convention (creation
// operators in increasing order of the original spinors).
template <class Ints, typename S = typename Ints::Scalar>
void WriteGasWaveFunction(const Input& input, const Ints& ints,
                          const GasSpace& space,
                          const std::vector<BigArray<S>>& vectors,
                          const std::vector<double>& energies) {
  int nwords = 0;
  const std::vector<uint64_t> plan_words = space.PlanDets(&nwords);
  const std::vector<int>& perm = space.layout().perm;
  const int n = space.num_orbs();
  const int64_t dim = space.dim();
  WaveFunction<S> wfn;
  wfn.num_orbs = n;
  wfn.num_elec = space.num_elec();
  wfn.det_width = (n + 63) / 64;
  wfn.det_words.assign(static_cast<size_t>(dim) * wfn.det_width, 0);
  std::vector<double> sign(dim);
#pragma omp parallel for schedule(static)
  for (int64_t d = 0; d < dim; ++d) {
    int occ[256], count = 0, inversions = 0;
    for (int p = 0; p < n; ++p) {
      if (!(plan_words[d * nwords + p / 64] >> (p % 64) & 1)) continue;
      const int orbital = perm[p];
      for (int i = 0; i < count; ++i) inversions += occ[i] > orbital;
      occ[count++] = orbital;
      wfn.det_words[d * wfn.det_width + orbital / 64] |= uint64_t(1)
                                                         << (orbital % 64);
    }
    sign[d] = inversions % 2 ? -1.0 : 1.0;
  }
  for (const BigArray<S>& v : vectors) {
    std::vector<S> c(dim);
    for (int64_t d = 0; d < dim; ++d) c[d] = sign[d] * v[d];
    wfn.coefficients.push_back(std::move(c));
  }
  wfn.energies = energies;
  for (const LabelSet& set : ints.symmetry()) {
    wfn.labels.insert(wfn.labels.end(), set.labels.begin(), set.labels.end());
    wfn.label_mods.push_back(set.mod);
  }
  wfn.input = input.resolved;
  WriteWaveFunction(input.prefix + "/wfn.h5", wfn);
}

}  // namespace

template <class Ints, typename S = typename Ints::Scalar>
void RunFci(const Input& input, const Ints& ints, const Comm& comm) {
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
    std::printf("fci: the block kernels do not apply: reference kernel\n");
    kernel = GasKernel::kReference;
  }

  // With several ranks (or "distributed"), each holds a part of the vectors
  // (fci_dist.h).
  const bool distributed = comm.size() > 1 || in.distributed;
  const int me = comm.rank();
  std::unique_ptr<GasDistribution> dist;
  int64_t local = dim, most_local = dim, most_t = 0, slice = 0;
  if (distributed) {
    if (kernel != GasKernel::kBlas) {
      throw std::runtime_error("fci: the distributed sigma is the BLAS kernel");
    }
    dist = std::make_unique<GasDistribution>(
        space, comm.size(), in.balance == "work",
        static_cast<int64_t>(in.slice_gb * kGiB / sizeof(S)));
    local = dist->End(me) - dist->Begin(me);
    most_local = most_t = 0;
    for (int r = 0; r < comm.size(); ++r) {
      most_local = std::max(most_local, dist->End(r) - dist->Begin(r));
      most_t = std::max(most_t, dist->EndT(r) - dist->BeginT(r));
    }
    slice = comm.size() > 1 ? dist->MaxSlice() : 0;
    std::printf(
        "fci: distributed over %d rank(s) by %s; largest part %.3f of the "
        "vector (size %.2f, work %.2f times the mean); slices of up to %ld "
        "elements\n",
        comm.size(), in.balance.c_str(), static_cast<double>(most_local) / dim,
        dist->SizeImbalance(), dist->WorkImbalance(),
        static_cast<long>(dist->MaxSlice()));
  }

  // Memory of the rank with the largest part: the Davidson vectors, the
  // diagonal, the kernel's workspace, the integrals and the plan.
  const double vector_bytes = static_cast<double>(most_local) * sizeof(S);
  const bool dense = !distributed && dim <= input.davidson.dense_dim;
  const int num_vectors = in.benchmark_sigmas > 0 ? 2
                          : dense                 ? 0
                                                  : 2 * in.max_space;
  // The kernel: the transposed c and sigma, a slice and exchange buffers.
  const double kernel_bytes =
      distributed                  ? (2.0 * most_t + 2.0 * slice) * sizeof(S) +
                                         2.0 * (1 << 26) * sizeof(S)
      : kernel == GasKernel::kBlas ? 2.0 * dim * sizeof(S)
                                   : 0.0;
  const double npair = n * (n - 1) / 2.0;
  // The plan: per string about 40 bytes, per A bucket and B class (and B
  // bucket and A class) 12.
  const double other_bytes =
      3.0 * npair * npair * sizeof(S) + 40.0 * (info[0] + info[1]) +
      12.0 * (info[4] * info[3] + info[5] * info[2]) + (1 << 30);
  const double need = num_vectors * vector_bytes + 8.0 * most_local +
                      kernel_bytes + other_bytes;
  double available = input.memory_gb > 0.0
                         ? input.memory_gb * kGiB
                         : AvailableBytes() / RanksOnNode(comm);
  std::printf(
      "fci memory per rank: one vector %.2f GiB (%s); %d Davidson vectors "
      "(max_space %d, each with its product) %.2f GiB, diagonal %.2f GiB, "
      "kernel workspace %.2f GiB, integrals, plan and the rest about %.2f "
      "GiB; total %.2f GiB of %.2f GiB available\n",
      vector_bytes / kGiB, ScalarTraits<S>::kName, num_vectors, in.max_space,
      num_vectors * vector_bytes / kGiB, 8.0 * most_local / kGiB,
      kernel_bytes / kGiB, other_bytes / kGiB, need / kGiB, available / kGiB);
  if (in.dry_run) return;
  if (need > available) {
    throw std::runtime_error(
        "fci: the calculation needs more memory than is available (see "
        "\"fci memory\" above; \"memory\" sets the gigabytes per rank to "
        "assume)");
  }

  time = omp_get_wtime();
  const std::vector<S> v = AntisymmetrizedPairs(ints, layout.perm, nelec);
  std::unique_ptr<CiOperator<S>> h;
  if (distributed && comm.size() > 1 && in.calibrate) {
    // One product with the parts as estimated, timing the target rows of
    // every bucket; the parts are then cut by these times.
    const int64_t slice_elements =
        static_cast<int64_t>(in.slice_gb * kGiB / sizeof(S));
    std::vector<double> a_seconds, b_seconds;
    {
      const DistributedGasOperator<S> op(space, v, *dist, comm, in.huge_pages);
      BigArray<S> x(local, in.huge_pages), y(local, in.huge_pages);
      const long first = dist->Begin(me);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < static_cast<long>(local); ++i) {
        x[i] = S(1.0 / (1.0 + (first + i) % 97));
      }
      op.MeasureBuckets(true);
      op.Apply(x.data(), y.data());
      a_seconds = op.BucketSeconds(0);
      b_seconds = op.BucketSeconds(1);
      comm.Sum(a_seconds.data(), a_seconds.size());
      comm.Sum(b_seconds.data(), b_seconds.size());
    }
    const double before = dist->Imbalance(a_seconds, b_seconds);
    // A rank's range of either half may not outgrow its memory: the
    // Davidson vectors and the diagonal per element of the vector, the
    // transposed c and sigma per element of the transpose.
    const double fixed =
        other_bytes + (2.0 * slice + 2.0 * (1 << 26)) * sizeof(S);
    const double per_element = num_vectors * sizeof(S) + 8.0 + 2.0 * sizeof(S);
    const int64_t max_elements = static_cast<int64_t>(
        std::max(0.0, (0.85 * available - fixed) / per_element));
    dist = std::make_unique<GasDistribution>(
        space, comm.size(), a_seconds, b_seconds, slice_elements, max_elements);
    local = dist->End(me) - dist->Begin(me);
    most_local = most_t = 0;
    for (int r = 0; r < comm.size(); ++r) {
      most_local = std::max(most_local, dist->End(r) - dist->Begin(r));
      most_t = std::max(most_t, dist->EndT(r) - dist->BeginT(r));
    }
    const double need_now = fixed +
                            (num_vectors * sizeof(S) + 8.0) * most_local +
                            2.0 * sizeof(S) * most_t;
    std::printf(
        "fci: parts cut by measured times: largest over mean %.2f before, "
        "%.2f now (%.1f s); largest part %.3f of the vector, about %.2f GiB "
        "per rank of %.2f GiB\n",
        before, dist->Imbalance(a_seconds, b_seconds), omp_get_wtime() - time,
        static_cast<double>(most_local) / dim, need_now / kGiB,
        available / kGiB);
    if (need_now > available) {
      throw std::runtime_error(
          "fci: after the cut by measured times a rank needs more memory than "
          "is available; use more ranks or a smaller max_space");
    }
  }
  BigArray<double> diagonal(local, in.huge_pages);
  if (distributed) {
    auto op = std::make_unique<DistributedGasOperator<S>>(space, v, *dist, comm,
                                                          in.huge_pages);
    op->Diagonal(diagonal.data());
    h = std::move(op);
  } else {
    space.Diagonal(v, diagonal.data());
    h = std::make_unique<GasOperator<S>>(space, v, kernel);
  }
  std::printf("fci: integrals over pairs and the diagonal, %.2f s\n",
              omp_get_wtime() - time);
  const VectorLayout vector_layout =
      distributed ? dist->Layout(me) : VectorLayout::Whole(dim);

  if (in.benchmark_sigmas > 0) {
    BigArray<S> x(local, in.huge_pages), y(local, in.huge_pages);
    const long first = vector_layout.segments[0].begin;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(local); ++i) {
      x[i] = S(1.0 / (1.0 + (first + i) % 97));
    }
    double best = 1e300;
    for (int k = 0; k < in.benchmark_sigmas; ++k) {
      comm.Barrier();
      time = omp_get_wtime();
      h->Apply(x.data(), y.data());
      comm.Barrier();
      const double seconds = omp_get_wtime() - time;
      best = std::min(best, seconds);
      std::printf("fci benchmark: sigma %d, %.2f s, peak memory %.2f GiB\n", k,
                  seconds, PeakGiB());
      if (distributed) {
        const auto* op = static_cast<const DistributedGasOperator<S>*>(h.get());
        const std::vector<double> parts = comm.GatherAll(
            std::vector<double>{op->seconds_exchange(), op->seconds_bcast(),
                                op->seconds_kernel(), PeakGiB()});
        for (int r = 0; r < comm.size(); ++r) {
          std::printf(
              "fci benchmark:   rank %d: transpositions %.2f s, slices "
              "waited for %.2f s, kernel %.2f s, peak memory %.2f GiB\n",
              r, parts[4 * r], parts[4 * r + 1], parts[4 * r + 2],
              parts[4 * r + 3]);
        }
      }
      double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
      for (long i = 0; i < static_cast<long>(local); ++i) {
        sum += Real(Conj(x[i]) * y[i]);
      }
      comm.Sum(&sum, 1);
      std::printf("fci benchmark:   <x|H|x> = %.12e\n", sum);
    }
    std::printf(
        "fci benchmark: best %.2f s per sigma, %d rank(s) x %d threads\n", best,
        comm.size(), omp_get_max_threads());
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
      FciDavidson(*h, diagonal.data(), vector_layout, options, comm);

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
  const std::vector<double> peaks =
      comm.GatherAll(std::vector<double>{PeakGiB()});
  std::printf("fci: peak memory %.2f GiB (largest over the ranks)\n",
              *std::max_element(peaks.begin(), peaks.end()));
  if (comm.is_root()) WriteEnergies(input.prefix, energies);

  // The wave function and density matrices, held whole by one rank.
  if (distributed) {
    if (input.rdm.one || input.rdm.two || input.rdm.transition) {
      std::printf(
          "fci: density matrices are not made from distributed "
          "vectors yet; none written\n");
    }
    return;
  }
  if (in.write_wfn > 0 && dim <= in.write_wfn) {
    time = omp_get_wtime();
    WriteGasWaveFunction(input, ints, space, result.vectors, energies);
    std::printf("fci: wfn.h5 written, %.1f s\n", omp_get_wtime() - time);
  }
  if (input.rdm.one || input.rdm.two) {
    time = omp_get_wtime();
    for (int k = 0; k < input.nroots; ++k) {
      const DensityMatrices<S> dm =
          GasDensity(space, result.vectors[k].data(), result.vectors[k].data(),
                     input.rdm.two);
      WriteDensityMatrices(input.prefix, k, k, dm);
      if (input.rdm.two) {
        const double from_rdm =
            EnergyFromDensityMatrices(dm, ints) + ints.core_energy();
        std::printf(
            "fci: root %d  energy from density matrices %.12f (difference "
            "%.2e)\n",
            k, from_rdm, from_rdm - energies[k]);
      }
    }
    std::printf("fci: density matrices, %.1f s\n", omp_get_wtime() - time);
  }
  if (input.rdm.transition) {
    for (int k = 1; k < input.nroots; ++k) {
      const DensityMatrices<S> dm = GasDensity(space, result.vectors[0].data(),
                                               result.vectors[k].data(), false);
      WriteDensityMatrices(input.prefix, 0, k, dm);
    }
  }
}

template void RunFci(const Input&, const SpinorIntegrals<double>&, const Comm&);
template void RunFci(const Input&, const SpinorIntegrals<Complex>&,
                     const Comm&);
template void RunFci(const Input&, const SpatialIntegrals<double>&,
                     const Comm&);
template void RunFci(const Input&, const UnrestrictedIntegrals&, const Comm&);

}  // namespace zshci
