#include "fci_dist.h"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

extern "C" {
#include "gas_dist.h"
#include "zgasci/zgasci_plan.h"
}

namespace zshci {
namespace {

// Elements per message of the transpositions.
constexpr int64_t kExchangeElements = int64_t(1) << 26;

// Work of a row of half h with kr electrons on the row side, as the kernel
// weighs its pieces (blas_row_weight in zgasci.c).
double RowWeight(const zg_plan* p, int h, int kr) {
  const double n = p->nside[h], nc = p->nside[1 - h], k = kr, u = n - k;
  const double pairs = 0.5 * k * (k - 1);
  double w = pairs * 0.5 * (u + 2) * (u + 1) + pairs * (u + 2) * nc +
             k * 0.5 * (u + 1) * u * nc + pairs * 0.5 * nc * (nc - 1);
  if (h == 0) w += k * (u + 1) * nc * nc;
  return w + 1.0;
}

// Bounds [0 = b_0 <= b_1 <= ... <= b_n = count) that cut `weight` into n
// nearly equal parts.
std::vector<int> Balance(const std::vector<double>& weight, int n) {
  double total = 0.0;
  for (double w : weight) total += w;
  std::vector<int> bounds(n + 1, static_cast<int>(weight.size()));
  bounds[0] = 0;
  double sum = 0.0;
  int r = 1;
  for (size_t i = 0; i < weight.size() && r < n; ++i) {
    // Close part r - 1 before item i if that is nearer to its share.
    while (r < n && sum + 0.5 * weight[i] > total * r / n) bounds[r++] = i;
    sum += weight[i];
  }
  return bounds;
}

double PartImbalance(const std::vector<double>& weight,
                     const std::vector<int>& bounds) {
  const int n = static_cast<int>(bounds.size()) - 1;
  double total = 0.0, most = 0.0;
  for (int r = 0; r < n; ++r) {
    double part = 0.0;
    for (int i = bounds[r]; i < bounds[r + 1]; ++i) part += weight[i];
    total += part;
    most = std::max(most, part);
  }
  return total > 0.0 ? most * n / total : 1.0;
}

struct Block {
  int ba, bb;
  int64_t na, nb;
  int64_t native, transposed;  // global offsets
  int owner, owner_t;
};

std::vector<Block> Blocks(const zg_plan* p, const GasDistribution& dist) {
  const zg_side& sa = p->s[0];
  const zg_side& sb = p->s[1];
  std::vector<int> owner_a(sa.nbucket), owner_b(sb.nbucket);
  for (int r = 0; r < dist.num_ranks(); ++r) {
    for (int b = dist.a_bounds()[r]; b < dist.a_bounds()[r + 1]; ++b) {
      owner_a[b] = r;
    }
    for (int b = dist.b_bounds()[r]; b < dist.b_bounds()[r + 1]; ++b) {
      owner_b[b] = r;
    }
  }
  std::vector<Block> blocks;
  for (int ba = 0; ba < sa.nbucket; ++ba) {
    for (int cb = 0; cb < sb.nclass; ++cb) {
      const size_t k = static_cast<size_t>(ba) * sb.nclass + cb;
      const int bb = p->match[k];
      if (bb < 0) continue;
      Block b;
      b.ba = ba;
      b.bb = bb;
      b.na = sa.bucket_size[ba];
      b.nb = sb.bucket_size[bb];
      b.native = p->blockoff[k];
      b.transposed = p->blockoff_t[static_cast<size_t>(bb) * sa.nclass +
                                   sa.bucket_class[ba]];
      b.owner = owner_a[ba];
      b.owner_t = owner_b[bb];
      blocks.push_back(b);
    }
  }
  return blocks;
}

}  // namespace

GasDistribution::GasDistribution(const GasSpace& space, int num_ranks,
                                 const std::vector<double>& a_weight,
                                 const std::vector<double>& b_weight,
                                 int64_t slice_elements)
    : GasDistribution(space, num_ranks, true, slice_elements) {
  a_work_ = a_weight;
  b_work_ = b_weight;
  a_bounds_ = Balance(a_work_, num_ranks);
  b_bounds_ = Balance(b_work_, num_ranks);
}

GasDistribution::GasDistribution(const GasSpace& space, int num_ranks,
                                 bool by_work, int64_t slice_elements)
    : slice_elements_(slice_elements) {
  const zg_plan* p = space.plan();
  if (!p->block_ok) {
    throw std::runtime_error(
        "fci: the distributed kernel needs each side to hold at most 64 "
        "spinors");
  }
  const zg_side& sa = p->s[0];
  const zg_side& sb = p->s[1];
  a_start_.assign(sa.nbucket + 1, 0);
  b_start_.assign(sb.nbucket + 1, 0);
  a_work_.assign(sa.nbucket, 0.0);
  b_work_.assign(sb.nbucket, 0.0);
  std::vector<double> a_size(sa.nbucket, 0.0), b_size(sb.nbucket, 0.0);
  int64_t at = 0;
  for (int ba = 0; ba < sa.nbucket; ++ba) {
    a_start_[ba] = at;
    const int kr = sa.tuple_elec[sa.class_tuple[sa.bucket_class[ba]]];
    for (int cb = 0; cb < sb.nclass; ++cb) {
      const size_t k = static_cast<size_t>(ba) * sb.nclass + cb;
      const int bb = p->match[k];
      if (bb < 0) continue;
      if (p->blockoff[k] != at) throw std::logic_error("fci: block order");
      const int64_t size = sa.bucket_size[ba] * sb.bucket_size[bb];
      at += size;
      a_size[ba] += size;
      a_work_[ba] += size * RowWeight(p, 0, kr);
    }
  }
  a_start_[sa.nbucket] = at;
  at = 0;
  for (int bb = 0; bb < sb.nbucket; ++bb) {
    b_start_[bb] = at;
    const int kr = sb.tuple_elec[sb.class_tuple[sb.bucket_class[bb]]];
    for (int ca = 0; ca < sa.nclass; ++ca) {
      const size_t k = static_cast<size_t>(bb) * sa.nclass + ca;
      const int ba = p->match_t[k];
      if (ba < 0) continue;
      if (p->blockoff_t[k] != at) throw std::logic_error("fci: block order");
      const int64_t size = sa.bucket_size[ba] * sb.bucket_size[bb];
      at += size;
      b_size[bb] += size;
      b_work_[bb] += size * RowWeight(p, 1, kr);
    }
  }
  b_start_[sb.nbucket] = at;
  a_bounds_ = Balance(by_work ? a_work_ : a_size, num_ranks);
  b_bounds_ = Balance(by_work ? b_work_ : b_size, num_ranks);
}

VectorLayout GasDistribution::Layout(int rank) const {
  VectorLayout layout;
  layout.global_dim = a_start_.back();
  layout.segments.push_back({Begin(rank), End(rank)});
  return layout;
}

std::vector<std::pair<int, int>> GasDistribution::Slices(int rank,
                                                         int half) const {
  const std::vector<int>& bounds = half == 0 ? a_bounds_ : b_bounds_;
  const std::vector<int64_t>& start = half == 0 ? a_start_ : b_start_;
  std::vector<std::pair<int, int>> out;
  int b0 = bounds[rank];
  const int end = bounds[rank + 1];
  while (b0 < end) {
    int b1 = b0 + 1;
    while (b1 < end && (slice_elements_ <= 0 ||
                        start[b1 + 1] - start[b0] <= slice_elements_)) {
      ++b1;
    }
    out.emplace_back(b0, b1);
    b0 = b1;
  }
  return out;
}

int64_t GasDistribution::MaxSlice() const {
  int64_t most = 0;
  for (int half = 0; half < 2; ++half) {
    const std::vector<int64_t>& start = half == 0 ? a_start_ : b_start_;
    for (int r = 0; r < num_ranks(); ++r) {
      for (const auto& s : Slices(r, half)) {
        most = std::max(most, start[s.second] - start[s.first]);
      }
    }
  }
  return most;
}

double GasDistribution::SizeImbalance() const {
  std::vector<double> size(a_start_.size() - 1);
  for (size_t b = 0; b + 1 < a_start_.size(); ++b) {
    size[b] = static_cast<double>(a_start_[b + 1] - a_start_[b]);
  }
  return PartImbalance(size, a_bounds_);
}

double GasDistribution::Imbalance(const std::vector<double>& a_weight,
                                  const std::vector<double>& b_weight) const {
  return std::max(PartImbalance(a_weight, a_bounds_),
                  PartImbalance(b_weight, b_bounds_));
}

double GasDistribution::WorkImbalance() const {
  // Both halves: the larger of the two per rank, which is what waits.
  return std::max(PartImbalance(a_work_, a_bounds_),
                  PartImbalance(b_work_, b_bounds_));
}

template <typename S>
DistributedGasOperator<S>::DistributedGasOperator(const GasSpace& space,
                                                  const std::vector<S>& v,
                                                  const GasDistribution& dist,
                                                  const Comm& comm,
                                                  bool huge_pages)
    : space_(space), v_(v), dist_(dist), comm_(comm) {
  if (dist.num_ranks() != comm.size()) {
    throw std::logic_error("fci: distribution for another number of ranks");
  }
  const int me = comm.rank();
  xt_ = BigArray<S>(dist.EndT(me) - dist.BeginT(me), huge_pages);
  yt_ = BigArray<S>(dist.EndT(me) - dist.BeginT(me), huge_pages);
  bucket_seconds_[0].assign(space.plan()->s[0].nbucket, 0.0);
  bucket_seconds_[1].assign(space.plan()->s[1].nbucket, 0.0);
  if (comm.size() > 1) {
    slice_[0] = BigArray<S>(dist.MaxSlice(), huge_pages);
    slice_[1] = BigArray<S>(dist.MaxSlice(), huge_pages);
  }
}

template <typename S>
DistributedGasOperator<S>::~DistributedGasOperator() {
  zgd_matrices_free(static_cast<zgd_matrices*>(matrices_));
}

template <>
void DistributedGasOperator<double>::Diagonal(double* out) const {
  const int me = comm_.rank();
  zgd_hdiag_d(space_.plan(), v_.data(), dist_.a_bounds()[me],
              dist_.a_bounds()[me + 1], out, dist_.Begin(me));
}

template <>
void DistributedGasOperator<Complex>::Diagonal(double* out) const {
  const int me = comm_.rank();
  zgd_hdiag_z(
      space_.plan(), reinterpret_cast<const double _Complex*>(v_.data()),
      dist_.a_bounds()[me], dist_.a_bounds()[me + 1], out, dist_.Begin(me));
}

template <typename S>
void DistributedGasOperator<S>::Transpose(const S* x, S* xt) const {
  const zg_plan* p = space_.plan();
  const std::vector<Block> blocks = Blocks(p, dist_);
  const int me = comm_.rank(), n = comm_.size();
  const int64_t base = dist_.Begin(me), base_t = dist_.BeginT(me);
  for (int round = 0; round < n; ++round) {
    const int dest = (me + round) % n, source = (me - round + n) % n;
    // My blocks that dest holds transposed, and dest's blocks I hold so,
    // both in the order of the vector and in batches of whole blocks.
    std::vector<const Block*> out, in;
    for (const Block& b : blocks) {
      if (b.owner == me && b.owner_t == dest) out.push_back(&b);
      if (b.owner == source && b.owner_t == me) in.push_back(&b);
    }
    size_t o = 0, i = 0;
    while (o < out.size() || i < in.size()) {
      std::vector<int64_t> out_at(1, 0), in_at(1, 0);
      size_t o1 = o, i1 = i;
      while (o1 < out.size() &&
             (o1 == o ||
              out_at.back() + out[o1]->na * out[o1]->nb <= kExchangeElements)) {
        out_at.push_back(out_at.back() + out[o1]->na * out[o1]->nb);
        ++o1;
      }
      while (i1 < in.size() &&
             (i1 == i ||
              in_at.back() + in[i1]->na * in[i1]->nb <= kExchangeElements)) {
        in_at.push_back(in_at.back() + in[i1]->na * in[i1]->nb);
        ++i1;
      }
      if (dest == me) {
        // Straight into place.
#pragma omp parallel for schedule(dynamic)
        for (long k = static_cast<long>(o); k < static_cast<long>(o1); ++k) {
          const Block& b = *out[k];
          const S* from = x + (b.native - base);
          S* to = xt + (b.transposed - base_t);
          for (int64_t j0 = 0; j0 < b.nb; j0 += 64) {
            const int64_t j1 = std::min(b.nb, j0 + 64);
            for (int64_t a = 0; a < b.na; ++a) {
              for (int64_t j = j0; j < j1; ++j)
                to[j * b.na + a] = from[a * b.nb + j];
            }
          }
        }
      } else {
        send_.resize(out_at.back());
        receive_.resize(in_at.back());
#pragma omp parallel for schedule(dynamic)
        for (long k = static_cast<long>(o); k < static_cast<long>(o1); ++k) {
          const Block& b = *out[k];
          const S* from = x + (b.native - base);
          S* to = send_.data() + out_at[k - o];
          for (int64_t j0 = 0; j0 < b.nb; j0 += 64) {
            const int64_t j1 = std::min(b.nb, j0 + 64);
            for (int64_t a = 0; a < b.na; ++a) {
              for (int64_t j = j0; j < j1; ++j)
                to[j * b.na + a] = from[a * b.nb + j];
            }
          }
        }
        comm_.SendReceive(send_.data(), out_at.back(), dest, receive_.data(),
                          in_at.back(), source, sizeof(S));
#pragma omp parallel for schedule(dynamic)
        for (long k = static_cast<long>(i); k < static_cast<long>(i1); ++k) {
          const Block& b = *in[k];
          std::memcpy(xt + (b.transposed - base_t),
                      receive_.data() + in_at[k - i], sizeof(S) * b.na * b.nb);
        }
      }
      o = o1;
      i = i1;
    }
  }
}

template <typename S>
void DistributedGasOperator<S>::TransposeAdd(const S* yt, S* y) const {
  const zg_plan* p = space_.plan();
  const std::vector<Block> blocks = Blocks(p, dist_);
  const int me = comm_.rank(), n = comm_.size();
  const int64_t base = dist_.Begin(me), base_t = dist_.BeginT(me);
  auto add_transposed = [](const Block& b, const S* from, S* to) {
    for (int64_t j0 = 0; j0 < b.nb; j0 += 64) {
      const int64_t j1 = std::min(b.nb, j0 + 64);
      for (int64_t a = 0; a < b.na; ++a) {
        for (int64_t j = j0; j < j1; ++j)
          to[a * b.nb + j] += from[j * b.na + a];
      }
    }
  };
  for (int round = 0; round < n; ++round) {
    const int dest = (me + round) % n, source = (me - round + n) % n;
    std::vector<const Block*> out, in;
    for (const Block& b : blocks) {
      if (b.owner_t == me && b.owner == dest) out.push_back(&b);
      if (b.owner_t == source && b.owner == me) in.push_back(&b);
    }
    size_t o = 0, i = 0;
    while (o < out.size() || i < in.size()) {
      std::vector<int64_t> out_at(1, 0), in_at(1, 0);
      size_t o1 = o, i1 = i;
      while (o1 < out.size() &&
             (o1 == o ||
              out_at.back() + out[o1]->na * out[o1]->nb <= kExchangeElements)) {
        out_at.push_back(out_at.back() + out[o1]->na * out[o1]->nb);
        ++o1;
      }
      while (i1 < in.size() &&
             (i1 == i ||
              in_at.back() + in[i1]->na * in[i1]->nb <= kExchangeElements)) {
        in_at.push_back(in_at.back() + in[i1]->na * in[i1]->nb);
        ++i1;
      }
      if (dest == me) {
#pragma omp parallel for schedule(dynamic)
        for (long k = static_cast<long>(o); k < static_cast<long>(o1); ++k) {
          const Block& b = *out[k];
          add_transposed(b, yt + (b.transposed - base_t),
                         y + (b.native - base));
        }
      } else {
        send_.resize(out_at.back());
        receive_.resize(in_at.back());
#pragma omp parallel for schedule(dynamic)
        for (long k = static_cast<long>(o); k < static_cast<long>(o1); ++k) {
          const Block& b = *out[k];
          std::memcpy(send_.data() + out_at[k - o],
                      yt + (b.transposed - base_t), sizeof(S) * b.na * b.nb);
        }
        comm_.SendReceive(send_.data(), out_at.back(), dest, receive_.data(),
                          in_at.back(), source, sizeof(S));
#pragma omp parallel for schedule(dynamic)
        for (long k = static_cast<long>(i); k < static_cast<long>(i1); ++k) {
          const Block& b = *in[k];
          add_transposed(b, receive_.data() + in_at[k - i],
                         y + (b.native - base));
        }
      }
      o = o1;
      i = i1;
    }
  }
}

namespace {

void HalfKernel(const zg_plan* p, zgd_matrices* m, int h, int t0, int t1,
                int s0, int s1, const double* cv, int64_t cbase, double* sv,
                int64_t sbase, double* seconds, void (*progress)()) {
  zgd_half_timed_d(p, m, h, t0, t1, s0, s1, cv, cbase, sv, sbase, seconds,
                   progress);
}

void HalfKernel(const zg_plan* p, zgd_matrices* m, int h, int t0, int t1,
                int s0, int s1, const Complex* cv, int64_t cbase, Complex* sv,
                int64_t sbase, double* seconds, void (*progress)()) {
  zgd_half_timed_z(
      p, m, h, t0, t1, s0, s1, reinterpret_cast<const double _Complex*>(cv),
      cbase, reinterpret_cast<double _Complex*>(sv), sbase, seconds, progress);
}

}  // namespace

template <typename S>
void DistributedGasOperator<S>::Half(int h, const S* local, S* target) const {
  const zg_plan* p = space_.plan();
  zgd_matrices* m = static_cast<zgd_matrices*>(matrices_);
  const int me = comm_.rank(), n = comm_.size();
  const std::vector<int>& bounds = h == 0 ? dist_.a_bounds() : dist_.b_bounds();
  const std::vector<int64_t>& start =
      h == 0 ? dist_.a_start() : dist_.b_start();
  const int64_t my_base = start[bounds[me]];
  auto kernel = [&](const std::pair<int, int>& s, const S* source) {
    const double t = omp_get_wtime();
    HalfKernel(p, m, h, bounds[me], bounds[me + 1], s.first, s.second, source,
               start[s.first], target, my_base,
               measure_ ? bucket_seconds_[h].data() : nullptr,
               n > 1 ? &Comm::Progress : nullptr);
    seconds_kernel_ += omp_get_wtime() - t;
  };
  // Own slices first, from place; then those of the other ranks, starting
  // with the next one, each read one-sidedly from its owner while the
  // previous one is worked on (the first while the own ones are).
  if (n == 1) {
    for (const std::pair<int, int>& s : dist_.Slices(me, h)) {
      kernel(s, local + (start[s.first] - my_base));
    }
    return;
  }
  const RemoteArray remote(comm_, local,
                           sizeof(S) * (start[bounds[me + 1]] - my_base));
  std::vector<std::pair<int, std::pair<int, int>>> others;
  for (int k = 1; k < n; ++k) {
    const int owner = (me + k) % n;
    for (const std::pair<int, int>& s : dist_.Slices(owner, h)) {
      others.emplace_back(owner, s);
    }
  }
  auto fetch = [&](size_t i) {
    const int owner = others[i].first;
    const std::pair<int, int>& s = others[i].second;
    const int64_t owner_base = start[bounds[owner]];
    remote.Get(slice_[i % 2].data(), owner,
               sizeof(S) * (start[s.first] - owner_base),
               sizeof(S) * (start[s.second] - start[s.first]));
  };
  if (!others.empty()) fetch(0);
  for (const std::pair<int, int>& s : dist_.Slices(me, h)) {
    kernel(s, local + (start[s.first] - my_base));
  }
  for (size_t i = 0; i < others.size(); ++i) {
    const double t = omp_get_wtime();
    remote.Wait(others[i].first);
    seconds_bcast_ += omp_get_wtime() - t;
    if (i + 1 < others.size()) fetch(i + 1);
    kernel(others[i].second, slice_[i % 2].data());
  }
}

template <typename S>
void DistributedGasOperator<S>::Apply(const S* x, S* y) const {
  seconds_exchange_ = seconds_bcast_ = seconds_kernel_ = 0.0;
  if (matrices_ == nullptr) {
    if constexpr (ScalarTraits<S>::kIsComplex) {
      const_cast<DistributedGasOperator*>(this)->matrices_ = zgd_matrices_new_z(
          space_.plan(), reinterpret_cast<const double _Complex*>(v_.data()));
    } else {
      const_cast<DistributedGasOperator*>(this)->matrices_ =
          zgd_matrices_new_d(space_.plan(), v_.data());
    }
  }
  const int me = comm_.rank();
  const long n = static_cast<long>(dist_.End(me) - dist_.Begin(me));
  const long nt = static_cast<long>(dist_.EndT(me) - dist_.BeginT(me));
  double t = omp_get_wtime();
  Transpose(x, xt_.data());
  seconds_exchange_ += omp_get_wtime() - t;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) y[i] = S(0.0);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < nt; ++i) yt_[i] = S(0.0);
  // The buffers of the slices are shared; the halves go one after the other.
  Half(1, xt_.data(), yt_.data());
  Half(0, x, y);
  t = omp_get_wtime();
  TransposeAdd(yt_.data(), y);
  seconds_exchange_ += omp_get_wtime() - t;
}

template class DistributedGasOperator<double>;
template class DistributedGasOperator<Complex>;

}  // namespace zshci
