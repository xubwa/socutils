// The CI space of the "fci" mode: full CI or generalized-active-space CI
// over spinors, held by the plan of the kernel src/zgasci (a copy of
// socutils' lib/zgasci, see src/zgasci/UPSTREAM).
//
// The kernel cuts the spinors into consecutive sub-groups of at most 16, the
// first ones forming side A, the others side B; a determinant is a string of
// each side. The spinors are therefore reordered first (GasLayout, as
// socutils.hci.gasci does it): every constrained set must be a union of
// sub-groups, and half of every set goes to each side. The Hamiltonian is
// handed over as
//     H = sum_{P,R} v[P,R] a+_p a+_q a_s a_r,   P = (p < q), R = (r < s),
// over the reordered spinors, v the antisymmetrized integrals with the
// one-electron part folded in (AntisymmetrizedPairs). CI vectors are in the
// order of the plan; nothing of the determinants or of H is stored.

#ifndef ZSHCI_SRC_GAS_SPACE_H_
#define ZSHCI_SRC_GAS_SPACE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "input.h"
#include "integrals.h"
#include "labels.h"
#include "scalar.h"

struct zg_plan;

namespace zshci {

// The reordering and the division into sub-groups.
struct GasLayout {
  // The original spinor at each place of the plan.
  std::vector<int> perm;
  // Spinors of each sub-group; the first num_sub_a are side A.
  std::vector<int> sub_sizes;
  int num_sub_a = 0;
  // Each constraint as sub-groups with weights.
  struct Constraint {
    std::vector<int> subs;
    std::vector<int> weights;
    int min_elec = 0;
    int max_elec = 0;
  };
  std::vector<Constraint> constraints;
};

// As socutils.hci.gasci.GASCI.layout: the spinors are grouped by their
// weights in the constraints; each group is split in halves (an odd spinor
// going to A and B in turn), each half cut into nearly equal sub-groups of at
// most max_subgroup spinors. `groups_a` and `groups_b`, if not empty, give
// the sub-groups explicitly instead. Throws if a sub-group cuts a constraint.
GasLayout MakeGasLayout(int num_orbs,
                        const std::vector<GasConstraint>& constraints,
                        int max_subgroup,
                        const std::vector<std::vector<int>>& groups_a = {},
                        const std::vector<std::vector<int>>& groups_b = {});

enum class GasKernel { kBlas, kBlock, kReference };

// The plan of the kernel: the addressing of a CI space. Not copyable.
class GasSpace {
 public:
  // `labels`: the label sets over the original spinors; `targets`: the sum
  // of the labels of the occupied spinors, one per set. Throws if the
  // kernel refuses the space.
  GasSpace(int num_orbs, int num_elec, const GasLayout& layout,
           const std::vector<LabelSet>& labels,
           const std::vector<int>& targets);
  ~GasSpace();
  GasSpace(const GasSpace&) = delete;
  GasSpace& operator=(const GasSpace&) = delete;

  int num_orbs() const { return num_orbs_; }
  int num_elec() const { return num_elec_; }
  int64_t dim() const { return dim_; }
  const GasLayout& layout() const { return layout_; }
  // Strings of A and of B, classes of A and of B, buckets of A and of B,
  // blocks, tasks of the reference kernel.
  std::vector<int64_t> Info() const;
  // Whether the block and BLAS kernels apply (always now); the reference
  // kernel runs otherwise.
  bool block_ok() const;
  // A number that changes with anything that changes the order of the
  // determinants: for checkpoint files.
  uint64_t Fingerprint() const { return fingerprint_; }
  // The kernel's plan (src/zgasci/zgasci_plan.h), for gas_dist.h.
  const zg_plan* plan() const { return plan_; }

  // The determinants in the order of the vector, as occupation words over
  // the spinors of the plan (bit p: spinor perm[p]); nwords per determinant.
  std::vector<uint64_t> PlanDets(int* nwords) const;

  // out[dim] = the diagonal of H. v: AntisymmetrizedPairs.
  template <typename S>
  void Diagonal(const std::vector<S>& v, double* out) const;
  // out[P * npair + R] = <bra| a+_p a+_q a_s a_r |ket> over the spinors of
  // the plan (reference kernel: slow for large spaces).
  template <typename S>
  void PairDensity(const S* bra, const S* ket, S* out) const;
  // sigma = H c.
  template <typename S>
  void Sigma(const std::vector<S>& v, const S* c, S* sigma,
             GasKernel kernel = GasKernel::kBlas) const;

 private:
  zg_plan* plan_ = nullptr;
  int num_orbs_ = 0;
  int num_elec_ = 0;
  int64_t dim_ = 0;
  GasLayout layout_;
  uint64_t fingerprint_ = 0;
};

// v[P * npair + R] over the spinors of the plan (perm), P = q (q - 1) / 2 + p:
// <pq||rs> + (h_pr d_qs - h_ps d_qr - h_qr d_ps + h_qs d_pr) / (N - 1).
template <class Ints, typename S = typename Ints::Scalar>
std::vector<S> AntisymmetrizedPairs(const Ints& ints,
                                    const std::vector<int>& perm, int num_elec);

}  // namespace zshci

#endif  // ZSHCI_SRC_GAS_SPACE_H_
