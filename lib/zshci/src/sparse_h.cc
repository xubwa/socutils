#include "sparse_h.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

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
  if (a.has_values() != b.has_values()) return false;
  for (size_t k = 0; k < a.col.size(); ++k) {
    if (a.col[k] != b.col[k]) return false;
    if (a.has_values() && std::abs(a.val[k] - b.val[k]) > tolerance) {
      return false;
    }
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
size_t SparseH<S>::num_values() const {
  size_t n = 0;
  for (const CsrBlock<S>& block : blocks_) n += block.val.size();
  return n;
}

template <typename S>
size_t SparseH<S>::element_bytes() const {
  return num_elements() * sizeof(DetIndex) + num_values() * sizeof(S);
}

template <typename S>
void SparseH<S>::DropValues() {
  for (CsrBlock<S>& block : blocks_) block.val = RawArray<S>();
}

template <typename S>
void SparseH<S>::FillBlockValues(size_t b) {
  CsrBlock<S>& block = blocks_[b];
  if (block.has_values()) return;
  if (source_ == nullptr) {
    throw std::logic_error("a block without values needs an element source");
  }
  block.val = RawArray<S>(block.col.size());
  const long num_rows = static_cast<long>(block.num_stored_rows());
#pragma omp parallel for schedule(dynamic, 64)
  for (long r = 0; r < num_rows; ++r) {
    const uint64_t first = block.row_ptr[r];
    source_->Fill(block.row[r], block.col.data() + first,
                  block.row_ptr[r + 1] - first, block.val.data() + first);
  }
}

namespace {

// The values of elements [first, last) of stored row r of `block`: where
// they are stored, or in `buffer` after being worked out by `source`.
template <typename S>
const S* ValuesOfRow(const CsrBlock<S>& block, size_t r, uint64_t first,
                     uint64_t last, const ElementSource<S>* source,
                     std::vector<S>* buffer) {
  if (block.has_values()) return block.val.data() + first;
  if (source == nullptr) {
    throw std::logic_error("a block without values needs an element source");
  }
  if (buffer->size() < last - first) buffer->resize(last - first);
  source->Fill(block.row[r], block.col.data() + first, last - first,
               buffer->data());
  return buffer->data();
}

}  // namespace

template <typename S>
size_t SparseH<S>::overhead_bytes() const {
  size_t bytes = diagonal_.size() * sizeof(double);
  for (const CsrBlock<S>& block : blocks_) {
    bytes += block.row.size() * sizeof(DetIndex) +
             block.row_ptr.size() * sizeof(uint64_t);
  }
  for (const SplitTable& split : splits_) {
    bytes += split.offset.size() * sizeof(uint32_t);
  }
  return bytes;
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

template <typename S>
void SparseH<S>::AppendBlocks(std::vector<CsrBlock<S>> blocks) {
  for (CsrBlock<S>& block : blocks) blocks_.push_back(std::move(block));
  PartitionRows();
}

// Splits the rows into one contiguous segment per thread such that every
// segment has about the same work in Multiply: with one sweep the elements
// of its rows, with two sweeps an element (j, i) counts for both j and i.
// An element whose value is worked out in the product counts as several
// (measured: products with no value stored take 2.5 to 4 times as long).
template <typename S>
void SparseH<S>::PartitionRows() {
  const size_t n = dim_;
  const int num_threads = omp_get_max_threads();
  one_sweep_ =
      num_threads > 1 && product_ != Product::kTwoSweeps &&
      (product_ == Product::kOneSweep ||
       // Judged by what the elements would take with their values: values
       // that are worked out are worth even more not to be visited twice.
       4 * static_cast<size_t>(num_threads) * n * sizeof(S) <=
           num_elements() * (sizeof(DetIndex) + sizeof(S)));
  segment_.assign(num_threads + 1, n);
  segment_[0] = 0;
  if (num_threads > 1 && n > 0) {
    // The work is counted in buckets of consecutive rows, 256 per thread,
    // by every thread on its own: an atomic addition per element (with two
    // sweeps every element counts for its column too) took 14 s for 1.1e10
    // elements. The segments end at bucket boundaries.
    const size_t num_buckets =
        std::min(n, static_cast<size_t>(256) * num_threads);
    const size_t bucket = (n + num_buckets - 1) / num_buckets;
    std::vector<uint64_t> weight(num_buckets, 0);
    for (size_t b = 0; b < num_buckets; ++b) {
      weight[b] = std::min(n, (b + 1) * bucket) - std::min(n, b * bucket);
    }
#pragma omp parallel
    {
      std::vector<uint64_t> local(num_buckets, 0);
      for (const CsrBlock<S>& block : blocks_) {
        const long num_rows = static_cast<long>(block.num_stored_rows());
        const uint64_t cost = block.has_values() ? 1 : 3;
#pragma omp for schedule(static) nowait
        for (long r = 0; r < num_rows; ++r) {
          local[block.row[r] / bucket] +=
              cost * (block.row_ptr[r + 1] - block.row_ptr[r]);
          if (one_sweep_) continue;
          for (uint64_t k = block.row_ptr[r]; k < block.row_ptr[r + 1]; ++k) {
            local[block.col[k] / bucket] += cost;
          }
        }
      }
#pragma omp critical
      for (size_t b = 0; b < num_buckets; ++b) weight[b] += local[b];
    }
    uint64_t total = 0;
    for (uint64_t w : weight) total += w;
    uint64_t running = 0;
    int next = 1;
    for (size_t b = 0; b < num_buckets && next < num_threads; ++b) {
      running += weight[b];
      while (next < num_threads && running * num_threads >= total * next) {
        segment_[next++] = std::min(n, (b + 1) * bucket);
      }
    }
  }

  // The split tables serve the two-sweep product only.
  splits_.clear();
  if (one_sweep_) return;
  scratch_ = RawArray<S>();
  splits_.resize(blocks_.size());
  for (size_t b = 0; b < blocks_.size(); ++b) {
    BuildSplitTable(blocks_[b], &splits_[b]);
  }
}

template <typename S>
void SparseH<S>::BuildSplitTable(const CsrBlock<S>& block,
                                 SplitTable* table) const {
  const int num_segments = static_cast<int>(segment_.size()) - 1;
  // The segments that can contain a column: from the first determinant to
  // the one before the last row.
  auto segment_of = [&](size_t index) {
    int s = 0;
    while (s + 1 < num_segments && segment_[s + 1] <= index) ++s;
    return s;
  };
  const size_t stored_rows = block.num_stored_rows();
  table->first_segment = 0;
  table->last_segment =
      stored_rows == 0 ? 0 : segment_of(block.row[stored_rows - 1]);
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
  if (one_sweep_) {
    const size_t n = dim_;
    if (scratch_.size() != static_cast<size_t>(num_segments) * n) {
      scratch_ = RawArray<S>(static_cast<size_t>(num_segments) * n);
    }
#pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < num_segments; ++t) {
      const size_t begin = segment_[t];
      const size_t end = segment_[t + 1];
      // The columns of this thread's rows lie below `end`.
      S* const mine = scratch_.data() + static_cast<size_t>(t) * n;
      std::vector<S> buffer;
      std::fill(mine, mine + end, S(0.0));
      for (size_t i = begin; i < end; ++i) {
        y[i] = apply_diagonal_ ? diagonal_[i] * x[i] : S(0.0);
      }
      for (const CsrBlock<S>& block : blocks_) {
        const DetIndex* const rows = block.row.data();
        const size_t num_rows = block.num_stored_rows();
        const size_t first_row =
            std::lower_bound(rows, rows + num_rows, begin) - rows;
        const size_t last_row =
            std::lower_bound(rows + first_row, rows + num_rows, end) - rows;
        for (size_t r = first_row; r < last_row; ++r) {
          const uint64_t first = block.row_ptr[r];
          const uint64_t count = block.row_ptr[r + 1] - first;
          const DetIndex* const col = block.col.data() + first;
          const S* const val =
              ValuesOfRow(block, r, first, first + count, source_, &buffer);
          const S xj = x[rows[r]];
          S sum(0.0);
          for (uint64_t k = 0; k < count; ++k) {
            sum += val[k] * x[col[k]];
            mine[col[k]] += Conj(val[k]) * xj;
          }
          y[rows[r]] += sum;
        }
      }
    }
    // Element i received contributions from the threads whose range ends
    // above i: its owner and those after it. The first range thus sums all
    // vectors and the last range one, so that a thread per range would
    // leave most threads waiting; and going through the vectors element by
    // element touches as many places in memory as there are vectors. So
    // the ranges are summed in pieces, vector after vector, by whichever
    // thread is free. Each element is summed in the same order as before.
    constexpr size_t kPiece = 2048;
    std::vector<size_t> first_task(num_segments + 1, 0);
    for (int t = 0; t < num_segments; ++t) {
      first_task[t + 1] =
          first_task[t] + (segment_[t + 1] - segment_[t] + kPiece - 1) / kPiece;
    }
    const long num_tasks = static_cast<long>(first_task[num_segments]);
#pragma omp parallel
    {
      std::vector<S> sum(kPiece);
#pragma omp for schedule(dynamic, 1)
      for (long k = 0; k < num_tasks; ++k) {
        const int t = static_cast<int>(
            std::upper_bound(first_task.begin(), first_task.end(),
                             static_cast<size_t>(k)) -
            first_task.begin() - 1);
        const size_t begin = segment_[t] + (k - first_task[t]) * kPiece;
        const size_t count = std::min(kPiece, segment_[t + 1] - begin);
        std::fill(sum.begin(), sum.begin() + count, S(0.0));
        for (int u = t; u < num_segments; ++u) {
          const S* const part =
              scratch_.data() + static_cast<size_t>(u) * n + begin;
          for (size_t i = 0; i < count; ++i) sum[i] += part[i];
        }
        for (size_t i = 0; i < count; ++i) y[begin + i] += sum[i];
      }
    }
    return;
  }
#pragma omp parallel for schedule(static, 1)
  for (int t = 0; t < num_segments; ++t) {
    const size_t begin = segment_[t];
    const size_t end = segment_[t + 1];
    std::vector<S> buffer;
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

      // Rows owned by this thread: y_j += H(j,i) x_i.
      for (size_t r = first_row; r < last_row; ++r) {
        const uint64_t first = block.row_ptr[r];
        const uint64_t count = block.row_ptr[r + 1] - first;
        const DetIndex* const col = block.col.data() + first;
        const S* const val =
            ValuesOfRow(block, r, first, first + count, source_, &buffer);
        S sum(0.0);
        for (uint64_t k = 0; k < count; ++k) sum += val[k] * x[col[k]];
        y[rows[r]] += sum;
      }
      // Columns owned by this thread: y_i += conj(H(j,i)) x_j. Since i < j,
      // only rows above `begin` can hold such a column.
      if (t < split.first_segment || t > split.last_segment) continue;
      const size_t num_boundaries = split.num_boundaries();
      const int local = t - split.first_segment;
      const size_t first_candidate =
          std::upper_bound(rows, rows + num_rows, begin) - rows;
      for (size_t r = first_candidate; r < num_rows; ++r) {
        const uint32_t* offsets = split.offset.data() + r * num_boundaries;
        const uint64_t first =
            local == 0 ? block.row_ptr[r]
                       : block.row_ptr[r] + offsets[local - 1];
        const uint64_t last = t == split.last_segment
                                  ? block.row_ptr[r + 1]
                                  : block.row_ptr[r] + offsets[local];
        if (first == last) continue;
        const DetIndex* const col = block.col.data() + first;
        const S* const val =
            ValuesOfRow(block, r, first, last, source_, &buffer);
        const S xi = x[rows[r]];
        for (uint64_t k = 0; k < last - first; ++k) {
          y[col[k]] += Conj(val[k]) * xi;
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
  std::vector<S> buffer;
  for (const CsrBlock<S>& block : blocks_) {
    for (size_t r = 0; r < block.num_stored_rows(); ++r) {
      const DetIndex i = block.row[r];
      const uint64_t first = block.row_ptr[r];
      const uint64_t count = block.row_ptr[r + 1] - first;
      const S* const val =
          ValuesOfRow(block, r, first, first + count, source_, &buffer);
      for (uint64_t k = 0; k < count; ++k) {
        const DetIndex j = block.col[first + k];
        y[i] += val[k] * x[j];
        y[j] += Conj(val[k]) * x[i];
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
