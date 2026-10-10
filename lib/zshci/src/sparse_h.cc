#include "sparse_h.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace zshci {

template <typename S>
bool SameBlock(const CsrBlock<S>& a, const CsrBlock<S>& b, double tolerance) {
  if (a.row_ptr != b.row_ptr || a.row.size() != b.row.size() ||
      a.col.size() != b.col.size()) {
    return false;
  }
  for (size_t r = 0; r < a.row.size(); ++r) {
    if (a.row[r] != b.row[r]) return false;
  }
  for (size_t k = 0; k < a.col.size(); ++k) {
    if (a.col[k] != b.col[k]) return false;
    if (std::abs(a.val[k] - b.val[k]) > tolerance) return false;
  }
  return true;
}

template <typename S>
size_t SparseH<S>::num_elements() const {
  size_t n = 0;
  for (const CsrBlock<S>& block : blocks_) n += block.num_elements();
  return n;
}

template <typename S>
void SparseH<S>::AppendDiagonal(const std::vector<double>& diagonal) {
  diagonal_.insert(diagonal_.end(), diagonal.begin(), diagonal.end());
  dim_ = diagonal_.size();
}

template <typename S>
void SparseH<S>::AppendBlock(CsrBlock<S> block) {
  blocks_.push_back(std::move(block));
  PartitionRows();
}

// Splits the rows into one contiguous segment per thread such that every
// segment touches about the same number of stored elements. An element (i, j)
// counts for both i and j, which is the work Multiply does for a segment.
template <typename S>
void SparseH<S>::PartitionRows() {
  const size_t n = dim_;
  const int num_threads = omp_get_max_threads();
  segment_.assign(num_threads + 1, n);
  segment_[0] = 0;
  if (num_threads > 1) {
    std::vector<uint64_t> weight(n, 1);
    for (const CsrBlock<S>& block : blocks_) {
      const long num_rows = static_cast<long>(block.num_stored_rows());
#pragma omp parallel for schedule(static)
      for (long r = 0; r < num_rows; ++r) {
        const uint64_t count = block.row_ptr[r + 1] - block.row_ptr[r];
#pragma omp atomic
        weight[block.row[r]] += count;
        for (uint64_t k = block.row_ptr[r]; k < block.row_ptr[r + 1]; ++k) {
#pragma omp atomic
          ++weight[block.col[k]];
        }
      }
    }
    uint64_t total = 0;
    for (uint64_t w : weight) total += w;
    uint64_t running = 0;
    int next = 1;
    for (size_t i = 0; i < n && next < num_threads; ++i) {
      running += weight[i];
      while (next < num_threads && running * num_threads >= total * next) {
        segment_[next++] = i + 1;
      }
    }
  }

  splits_.resize(blocks_.size());
  for (size_t b = 0; b < blocks_.size(); ++b) {
    BuildSplitTable(blocks_[b], &splits_[b]);
  }
}

template <typename S>
void SparseH<S>::BuildSplitTable(const CsrBlock<S>& block,
                                 SplitTable* table) const {
  const int num_segments = static_cast<int>(segment_.size()) - 1;
  // The segments that contain the first and the last possible column.
  auto segment_of = [&](size_t index) {
    int s = 0;
    while (s + 1 < num_segments && segment_[s + 1] <= index) ++s;
    return s;
  };
  table->first_segment = segment_of(block.first_col);
  table->last_segment =
      segment_of(block.dim == 0 ? block.first_col : block.dim - 1);
  const size_t num_boundaries = table->num_boundaries();
  table->offset = RawArray<uint32_t>(block.num_stored_rows() * num_boundaries);
  if (num_boundaries == 0) return;

  const long num_rows = static_cast<long>(block.num_stored_rows());
#pragma omp parallel for schedule(static)
  for (long r = 0; r < num_rows; ++r) {
    const uint64_t row_begin = block.row_ptr[r];
    const uint64_t row_end = block.row_ptr[r + 1];
    if (row_end - row_begin > UINT32_MAX) {
      throw std::length_error("a row of H has more than 2^32 elements");
    }
    uint64_t k = row_begin;
    for (size_t b = 0; b < num_boundaries; ++b) {
      const size_t boundary = segment_[table->first_segment + b + 1];
      while (k < row_end && block.col[k] < boundary) ++k;
      table->offset[r * num_boundaries + b] =
          static_cast<uint32_t>(k - row_begin);
    }
  }
}

template <typename S>
void SparseH<S>::Multiply(const S* x, S* y) const {
  const int num_segments = static_cast<int>(segment_.size()) - 1;
#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < num_segments; ++t) {
    const size_t begin = segment_[t];
    const size_t end = segment_[t + 1];
    for (size_t i = begin; i < end; ++i) {
      y[i] = apply_diagonal_ ? diagonal_[i] * x[i] : S(0.0);
    }

    for (size_t b = 0; b < blocks_.size(); ++b) {
      const CsrBlock<S>& block = blocks_[b];
      const SplitTable& split = splits_[b];
      // The stored rows below `begin` and below `end`.
      const DetIndex* const rows = block.row.data();
      const size_t num_rows = block.num_stored_rows();
      const size_t first_row =
          std::lower_bound(rows, rows + num_rows, begin) - rows;
      const size_t last_row =
          std::lower_bound(rows + first_row, rows + num_rows, end) - rows;

      // Rows owned by this thread: y_i += H(i,j) x_j.
      for (size_t r = first_row; r < last_row; ++r) {
        S sum(0.0);
        for (uint64_t k = block.row_ptr[r]; k < block.row_ptr[r + 1]; ++k) {
          sum += block.val[k] * x[block.col[k]];
        }
        y[rows[r]] += sum;
      }
      // Columns owned by this thread: y_j += conj(H(i,j)) x_i. Since j > i,
      // only rows below `end` can hold such a column.
      if (t < split.first_segment || t > split.last_segment) continue;
      const size_t num_boundaries = split.num_boundaries();
      const int local = t - split.first_segment;
      for (size_t r = 0; r < last_row; ++r) {
        const uint32_t* offsets = split.offset.data() + r * num_boundaries;
        const uint64_t first =
            local == 0 ? block.row_ptr[r]
                       : block.row_ptr[r] + offsets[local - 1];
        const uint64_t last = t == split.last_segment
                                  ? block.row_ptr[r + 1]
                                  : block.row_ptr[r] + offsets[local];
        const S xi = x[rows[r]];
        for (uint64_t k = first; k < last; ++k) {
          y[block.col[k]] += Conj(block.val[k]) * xi;
        }
      }
    }
  }
}

template <typename S>
void SparseH<S>::MultiplyReference(const S* x, S* y) const {
  const size_t n = dim_;
  for (size_t i = 0; i < n; ++i) {
    y[i] = apply_diagonal_ ? diagonal_[i] * x[i] : S(0.0);
  }
  for (const CsrBlock<S>& block : blocks_) {
    for (size_t r = 0; r < block.num_stored_rows(); ++r) {
      const DetIndex i = block.row[r];
      for (uint64_t k = block.row_ptr[r]; k < block.row_ptr[r + 1]; ++k) {
        const DetIndex j = block.col[k];
        y[i] += block.val[k] * x[j];
        y[j] += Conj(block.val[k]) * x[i];
      }
    }
  }
}

template bool SameBlock(const CsrBlock<double>&, const CsrBlock<double>&,
                        double);
template bool SameBlock(const CsrBlock<Complex>&, const CsrBlock<Complex>&,
                        double);
template class SparseH<double>;
template class SparseH<Complex>;

}  // namespace zshci
