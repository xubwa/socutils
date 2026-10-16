// H c for a CI vector of the fci mode distributed over ranks.
//
// Every rank owns a range of A buckets of the plan, that is a contiguous
// segment of the vector (blocks are stored A bucket after A bucket), and
// writes those elements of sigma. The kernel builds sigma in two halves
// (src/zgasci/zgasci_blas.inc): half 0 with A strings as rows on the vector,
// half 1 with B strings as rows on the transposed vector. For half 1 every
// rank also owns a range of B buckets of the transposed vector; c is
// transposed into that distribution by pairwise exchanges, and the result of
// half 1 transposed back and added. In each half the source vector goes
// round slice by slice (a rank's range, or pieces of it of at most
// slice_elements): every rank reads the slices of the others one-sidedly,
// the next one while it works on the present one, and applies the kernel
// restricted to its own target rows and the source rows of the slice
// (gas_dist.h). Memory per rank: the local parts of c and sigma, of their
// transposes, two slices, and the exchange buffers.
//
// With one rank this is the same computation without communication; the
// tests use it with several slices.

#ifndef ZSHCI_SRC_FCI_DIST_H_
#define ZSHCI_SRC_FCI_DIST_H_

#include <cstdint>
#include <vector>

#include "big_array.h"
#include "comm.h"
#include "fci_davidson.h"
#include "gas_space.h"

namespace zshci {

// Who owns which part of the vector and of its transpose.
class GasDistribution {
 public:
  // Ranges balanced by the work of the kernel's rows ("work") or by the
  // number of elements ("size"). slice_elements 0: whole ranges.
  GasDistribution(const GasSpace& space, int num_ranks, bool by_work,
                  int64_t slice_elements);

  int num_ranks() const { return static_cast<int>(a_bounds_.size()) - 1; }
  // A buckets [a_bounds[r], a_bounds[r + 1]) are rank r's, their elements
  // [a_start[a_bounds[r]], a_start[a_bounds[r + 1]]); likewise B buckets of
  // the transposed vector.
  const std::vector<int>& a_bounds() const { return a_bounds_; }
  const std::vector<int>& b_bounds() const { return b_bounds_; }
  const std::vector<int64_t>& a_start() const { return a_start_; }
  const std::vector<int64_t>& b_start() const { return b_start_; }
  int64_t Begin(int rank) const { return a_start_[a_bounds_[rank]]; }
  int64_t End(int rank) const { return a_start_[a_bounds_[rank + 1]]; }
  int64_t BeginT(int rank) const { return b_start_[b_bounds_[rank]]; }
  int64_t EndT(int rank) const { return b_start_[b_bounds_[rank + 1]]; }
  VectorLayout Layout(int rank) const;
  // The slices of a rank's range of half h, as bucket ranges.
  std::vector<std::pair<int, int>> Slices(int rank, int half) const;
  int64_t MaxSlice() const;
  // Largest over mean of the elements and of the estimated work per rank.
  double SizeImbalance() const;
  double WorkImbalance() const;

 private:
  std::vector<int> a_bounds_, b_bounds_;
  std::vector<int64_t> a_start_, b_start_;
  std::vector<double> a_work_, b_work_;  // per bucket
  int64_t slice_elements_ = 0;
};

// S is double or Complex.
template <typename S>
class DistributedGasOperator : public CiOperator<S> {
 public:
  DistributedGasOperator(const GasSpace& space, const std::vector<S>& v,
                         const GasDistribution& dist, const Comm& comm,
                         bool huge_pages);
  ~DistributedGasOperator() override;

  // The local elements of the diagonal.
  void Diagonal(double* out) const;
  // y = H x over this rank's elements.
  void Apply(const S* x, S* y) const override;

  // Seconds of the last Apply: transpositions, waiting for slices, kernel.
  double seconds_exchange() const { return seconds_exchange_; }
  double seconds_bcast() const { return seconds_bcast_; }
  double seconds_kernel() const { return seconds_kernel_; }

 private:
  // xt (local transposed) = the transpose of x (local native).
  void Transpose(const S* x, S* xt) const;
  // y (local native) += the transpose of yt (local transposed).
  void TransposeAdd(const S* yt, S* y) const;
  void Half(int h, const S* local, S* target) const;

  const GasSpace& space_;
  const std::vector<S>& v_;
  const GasDistribution& dist_;
  const Comm& comm_;
  void* matrices_ = nullptr;
  mutable BigArray<S> xt_, yt_, slice_[2];
  mutable std::vector<S> send_, receive_;
  mutable double seconds_exchange_ = 0.0;
  mutable double seconds_bcast_ = 0.0;
  mutable double seconds_kernel_ = 0.0;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_FCI_DIST_H_
