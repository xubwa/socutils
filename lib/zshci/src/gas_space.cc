#include "gas_space.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

extern "C" {
#include "zgasci/zgasci.h"
}

namespace zshci {
namespace {

// `orbitals` in ceil(size / max_size) consecutive pieces of nearly equal
// size, cut where numpy.linspace(...).round() cuts.
std::vector<std::vector<int>> Chunks(const std::vector<int>& orbitals,
                                     int max_size) {
  std::vector<std::vector<int>> out;
  const int n = static_cast<int>(orbitals.size());
  if (n == 0) return out;
  const int k = (n + max_size - 1) / max_size;
  const double step = static_cast<double>(n) / k;
  int begin = 0;
  for (int i = 1; i <= k; ++i) {
    const int end = i == k ? n : static_cast<int>(std::nearbyint(i * step));
    out.emplace_back(orbitals.begin() + begin, orbitals.begin() + end);
    begin = end;
  }
  return out;
}

uint64_t Mix(uint64_t hash, int64_t value) {
  for (int b = 0; b < 8; ++b) {
    hash ^= static_cast<uint64_t>(value >> (8 * b)) & 0xff;
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace

GasLayout MakeGasLayout(int num_orbs,
                        const std::vector<GasConstraint>& constraints,
                        int max_subgroup,
                        const std::vector<std::vector<int>>& groups_a,
                        const std::vector<std::vector<int>>& groups_b) {
  // The weight of every orbital in every constraint.
  std::vector<std::vector<int>> weight(constraints.size(),
                                       std::vector<int>(num_orbs, 0));
  for (size_t c = 0; c < constraints.size(); ++c) {
    const GasConstraint& con = constraints[c];
    if (!con.weights.empty() && con.weights.size() != con.orbitals.size()) {
      throw std::runtime_error("gas: weights need one entry per orbital");
    }
    std::set<int> seen;
    for (size_t k = 0; k < con.orbitals.size(); ++k) {
      const int p = con.orbitals[k];
      if (p < 0 || p >= num_orbs) {
        throw std::runtime_error("gas: orbital " + std::to_string(p) +
                                 " out of 0.." + std::to_string(num_orbs - 1));
      }
      if (!seen.insert(p).second) {
        throw std::runtime_error("gas: orbital " + std::to_string(p) +
                                 " named twice in a constraint");
      }
      weight[c][p] = con.weights.empty() ? 1 : con.weights[k];
    }
  }

  std::vector<std::vector<int>> groups;
  int num_sub_a = 0;
  if (!groups_a.empty() || !groups_b.empty()) {
    groups = groups_a;
    groups.insert(groups.end(), groups_b.begin(), groups_b.end());
    num_sub_a = static_cast<int>(groups_a.size());
    std::vector<int> flat;
    for (const std::vector<int>& g : groups) {
      flat.insert(flat.end(), g.begin(), g.end());
    }
    std::sort(flat.begin(), flat.end());
    for (int p = 0; p < num_orbs; ++p) {
      if (static_cast<int>(flat.size()) != num_orbs || flat[p] != p) {
        throw std::runtime_error("subgroups must hold every orbital once");
      }
    }
  } else {
    if (max_subgroup < 1 || max_subgroup > 16) {
      throw std::runtime_error("max_subgroup must be 1 to 16");
    }
    // Orbitals of equal weights in all constraints, in the order of their
    // first orbital.
    std::map<std::vector<int>, std::vector<int>> by_key;
    std::vector<std::vector<int>> keys;
    for (int p = 0; p < num_orbs; ++p) {
      std::vector<int> key(constraints.size());
      for (size_t c = 0; c < constraints.size(); ++c) key[c] = weight[c][p];
      if (by_key.count(key) == 0) keys.push_back(key);
      by_key[key].push_back(p);
    }
    std::vector<std::vector<int>> side_a, side_b;
    bool extra_to_a = true;
    for (const std::vector<int>& key : keys) {
      const std::vector<int>& atom = by_key[key];
      int half = static_cast<int>(atom.size()) / 2;
      if (atom.size() % 2) {
        half += extra_to_a ? 1 : 0;
        extra_to_a = !extra_to_a;
      }
      for (auto& g :
           Chunks({atom.begin(), atom.begin() + half}, max_subgroup)) {
        side_a.push_back(g);
      }
      for (auto& g : Chunks({atom.begin() + half, atom.end()}, max_subgroup)) {
        side_b.push_back(g);
      }
    }
    if (side_b.empty() && !side_a.empty()) {
      side_b.push_back(side_a.back());
      side_a.pop_back();
    }
    groups = side_a;
    groups.insert(groups.end(), side_b.begin(), side_b.end());
    num_sub_a = static_cast<int>(side_a.size());
  }

  GasLayout layout;
  layout.num_sub_a = num_sub_a;
  for (const std::vector<int>& g : groups) {
    layout.perm.insert(layout.perm.end(), g.begin(), g.end());
    layout.sub_sizes.push_back(static_cast<int>(g.size()));
  }
  for (size_t c = 0; c < constraints.size(); ++c) {
    GasLayout::Constraint con;
    con.min_elec = constraints[c].min_elec;
    con.max_elec = constraints[c].max_elec;
    for (size_t k = 0; k < groups.size(); ++k) {
      std::set<int> w;
      for (int p : groups[k]) w.insert(weight[c][p]);
      if (w.size() > 1) {
        throw std::runtime_error(
            "the sub-groups must not cut a constraint: sub-group " +
            std::to_string(k) + " and constraint " + std::to_string(c));
      }
      if (!w.empty() && *w.begin() != 0) {
        con.subs.push_back(static_cast<int>(k));
        con.weights.push_back(*w.begin());
      }
    }
    layout.constraints.push_back(con);
  }
  return layout;
}

GasSpace::GasSpace(int num_orbs, int num_elec, const GasLayout& layout,
                   const std::vector<LabelSet>& labels,
                   const std::vector<int>& targets)
    : num_orbs_(num_orbs), num_elec_(num_elec), layout_(layout) {
  if (num_elec < 2 || num_elec > num_orbs) {
    throw std::runtime_error(
        "fci: needs at least two electrons and no more than the spinors");
  }
  if (targets.size() != labels.size()) {
    throw std::runtime_error("fci: target needs one entry per label set");
  }
  std::vector<int> gas_ptr(1, 0), gas_sub, gas_weight, gas_min, gas_max;
  for (const GasLayout::Constraint& con : layout.constraints) {
    gas_sub.insert(gas_sub.end(), con.subs.begin(), con.subs.end());
    gas_weight.insert(gas_weight.end(), con.weights.begin(), con.weights.end());
    gas_ptr.push_back(static_cast<int>(gas_sub.size()));
    gas_min.push_back(con.min_elec);
    gas_max.push_back(con.max_elec);
  }
  std::vector<int> plan_labels, mods;
  for (const LabelSet& set : labels) {
    for (int p : layout.perm) plan_labels.push_back(set.labels.at(p));
    mods.push_back(set.mod);
  }
  // The kernel reads one element past empty arrays.
  for (std::vector<int>* a :
       {&gas_sub, &gas_weight, &gas_min, &gas_max, &plan_labels, &mods}) {
    a->push_back(0);
  }
  std::vector<int> target_list = targets;
  target_list.push_back(0);
  plan_ =
      zg_plan_new(num_orbs, num_elec, static_cast<int>(layout.sub_sizes.size()),
                  layout.num_sub_a, layout.sub_sizes.data(),
                  static_cast<int>(layout.constraints.size()), gas_ptr.data(),
                  gas_sub.data(), gas_weight.data(), gas_min.data(),
                  gas_max.data(), static_cast<int>(labels.size()),
                  plan_labels.data(), mods.data(), target_list.data());
  if (plan_ == nullptr) {
    throw std::runtime_error(std::string("fci: zgasci: ") + zg_error());
  }
  dim_ = zg_dim(plan_);

  uint64_t hash = 1469598103934665603ull;
  hash = Mix(hash, num_orbs);
  hash = Mix(hash, num_elec);
  hash = Mix(hash, layout.num_sub_a);
  for (int x : layout.perm) hash = Mix(hash, x);
  for (int x : layout.sub_sizes) hash = Mix(hash, x);
  for (const GasLayout::Constraint& con : layout.constraints) {
    for (int x : con.subs) hash = Mix(hash, x);
    for (int x : con.weights) hash = Mix(hash, x);
    hash = Mix(hash, con.min_elec);
    hash = Mix(hash, con.max_elec);
  }
  for (int x : plan_labels) hash = Mix(hash, x);
  for (int x : mods) hash = Mix(hash, x);
  for (int x : targets) hash = Mix(hash, x);
  hash = Mix(hash, dim_);
  fingerprint_ = hash;
}

GasSpace::~GasSpace() {
  if (plan_ != nullptr) zg_plan_free(plan_);
}

std::vector<int64_t> GasSpace::Info() const {
  std::vector<int64_t> out(8);
  zg_info(plan_, out.data());
  return out;
}

bool GasSpace::block_ok() const { return zg_block_ok(plan_) != 0; }

std::vector<uint64_t> GasSpace::PlanDets(int* nwords) const {
  *nwords = zg_nwords(plan_);
  std::vector<uint64_t> words(static_cast<size_t>(dim_) * *nwords);
  zg_dets(plan_, words.data());
  return words;
}

template <>
void GasSpace::Diagonal(const std::vector<double>& v, double* out) const {
  zg_hdiag_d(plan_, v.data(), out);
}

template <>
void GasSpace::Diagonal(const std::vector<Complex>& v, double* out) const {
  zg_hdiag_z(plan_, reinterpret_cast<const double _Complex*>(v.data()), out);
}

template <>
void GasSpace::PairDensity(const double* bra, const double* ket,
                           double* out) const {
  zg_rdm2_d(plan_, bra, ket, out);
}

template <>
void GasSpace::PairDensity(const Complex* bra, const Complex* ket,
                           Complex* out) const {
  zg_rdm2_z(plan_, reinterpret_cast<const double _Complex*>(bra),
            reinterpret_cast<const double _Complex*>(ket),
            reinterpret_cast<double _Complex*>(out));
}

template <>
void GasSpace::Sigma(const std::vector<double>& v, const double* c,
                     double* sigma, GasKernel kernel) const {
  switch (kernel) {
    case GasKernel::kBlas:
      return zg_sigma_blas_d(plan_, v.data(), c, sigma);
    case GasKernel::kBlock:
      if (block_ok()) return zg_sigma_block_d(plan_, v.data(), c, sigma);
      break;
    case GasKernel::kReference:
      break;
  }
  zg_sigma_d(plan_, v.data(), c, sigma);
}

template <>
void GasSpace::Sigma(const std::vector<Complex>& v, const Complex* c,
                     Complex* sigma, GasKernel kernel) const {
  using Z = double _Complex;
  const Z* zv = reinterpret_cast<const Z*>(v.data());
  const Z* zc = reinterpret_cast<const Z*>(c);
  Z* zs = reinterpret_cast<Z*>(sigma);
  switch (kernel) {
    case GasKernel::kBlas:
      return zg_sigma_blas_z(plan_, zv, zc, zs);
    case GasKernel::kBlock:
      if (block_ok()) return zg_sigma_block_z(plan_, zv, zc, zs);
      break;
    case GasKernel::kReference:
      break;
  }
  zg_sigma_z(plan_, zv, zc, zs);
}

template <class Ints, typename S = typename Ints::Scalar>
std::vector<S> AntisymmetrizedPairs(const Ints& ints,
                                    const std::vector<int>& perm,
                                    int num_elec) {
  const int n = static_cast<int>(perm.size());
  const long npair = static_cast<long>(n) * (n - 1) / 2;
  std::vector<int> first(npair), second(npair);
  for (int q = 1, k = 0; q < n; ++q) {
    for (int p = 0; p < q; ++p, ++k) {
      first[k] = p;
      second[k] = q;
    }
  }
  const double scale = 1.0 / (num_elec - 1);
  auto h = [&](int a, int b) { return ints.OneBody(perm[a], perm[b]); };
  std::vector<S> v(static_cast<size_t>(npair) * npair);
#pragma omp parallel for schedule(dynamic, 16)
  for (long bra = 0; bra < npair; ++bra) {
    const int p = first[bra], q = second[bra];
    for (long ket = 0; ket < npair; ++ket) {
      const int r = first[ket], s = second[ket];
      S value = ints.Antisym(perm[p], perm[q], perm[r], perm[s]);
      S one(0.0);
      if (q == s) one += h(p, r);
      if (q == r) one -= h(p, s);
      if (p == s) one -= h(q, r);
      if (p == r) one += h(q, s);
      v[static_cast<size_t>(bra) * npair + ket] = value + scale * one;
    }
  }
  return v;
}

template std::vector<double> AntisymmetrizedPairs(
    const SpinorIntegrals<double>&, const std::vector<int>&, int);
template std::vector<Complex> AntisymmetrizedPairs(
    const SpinorIntegrals<Complex>&, const std::vector<int>&, int);
template std::vector<double> AntisymmetrizedPairs(
    const SpatialIntegrals<double>&, const std::vector<int>&, int);

}  // namespace zshci
