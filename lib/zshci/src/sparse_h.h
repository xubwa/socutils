// Sparse Hermitian Hamiltonian in the determinant basis.
//
// Only one triangle is stored, so the matrix costs half the memory of a full
// CSR: for every determinant, its elements with the determinants before it. The
// matrix is a list of blocks because a selected-CI calculation grows the
// determinant list: each block holds the elements whose column lies in the
// range of determinants added in one iteration.

#ifndef ZSHCI_SRC_SPARSE_H_H_
#define ZSHCI_SRC_SPARSE_H_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "parallel.h"
#include "scalar.h"

namespace zshci {

// A Hermitian operator the eigensolver can work with.
template <typename S>
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;
  virtual size_t dim() const = 0;
  virtual const std::vector<double>& diagonal() const = 0;
  // y = H x.
  virtual void Apply(const S* x, S* y) const = 0;
};

// Index of a determinant in the variational space.
using DetIndex = uint32_t;

// Compressed storage of elements H(j, i) = <D_j|H|D_i> with i < j < dim and
// j >= first_row: the rows are the determinants added in one iteration, the
// columns those before each of them. A determinant's whole row is thus known
// as soon as its connections are, and one thread can write it down by
// itself. Only the rows that hold elements are listed, so a rank or a thread
// that holds a small part of the matrix does not pay for all its rows:
// row[r] is the r-th such row, in increasing order, and its elements are
// col/val[row_ptr[r] .. row_ptr[r + 1]), sorted by column.
template <typename S>
struct CsrBlock {
  size_t dim = 0;
  size_t first_row = 0;
  RawArray<DetIndex> row;
  std::vector<uint64_t> row_ptr;  // row.size() + 1 entries
  RawArray<DetIndex> col;
  RawArray<S> val;

  size_t num_stored_rows() const { return row.size(); }
  size_t num_elements() const { return col.size(); }
};

// Whether two blocks hold the same elements, the values within `tolerance`.
template <typename S>
bool SameBlock(const CsrBlock<S>& a, const CsrBlock<S>& b, double tolerance);

template <typename S>
class SparseH : public LinearOperator<S> {
 public:
  size_t dim() const override { return dim_; }
  // Empty on ranks that were given no diagonal (see Grow).
  const std::vector<double>& diagonal() const override { return diagonal_; }
  void Apply(const S* x, S* y) const override { Multiply(x, y); }
  const std::vector<CsrBlock<S>>& blocks() const { return blocks_; }
  size_t num_elements() const;
  // Bytes of the stored elements (columns and values), and of what is held
  // besides them: the row lists and the tables that tell every thread where
  // its columns lie in each row.
  size_t element_bytes() const;
  size_t overhead_bytes() const;

  // Appends the diagonal elements of newly added determinants.
  void AppendDiagonal(const std::vector<double>& diagonal);

  // Enlarges the matrix to `dim` rows without storing diagonal elements, for
  // the ranks that do not apply the diagonal.
  void Grow(size_t dim) { dim_ = dim; }

  // Adds a block, or several, and re-partitions the rows among the threads.
  void AppendBlock(CsrBlock<S> block);
  void AppendBlocks(std::vector<CsrBlock<S>> blocks);

  // y = H x. Each thread owns a contiguous range of y and is the only writer
  // to it: an element H(j, i) is read once by the owner of j, for
  // y_j += H(j,i) x_i, and once by the owner of i, for y_i += conj(H(j,i)) x_j.
  // No thread-private copy of y is made. For the second kind of access every
  // block keeps, per row, the positions where the columns pass from one
  // thread's range to the next.
  void Multiply(const S* x, S* y) const;

  // When the matrix is spread over several ranks, each holding part of the
  // off-diagonal elements, exactly one of them must add the diagonal.
  void set_apply_diagonal(bool apply) { apply_diagonal_ = apply; }

  // Plain serial product, used to check Multiply.
  void MultiplyReference(const S* x, S* y) const;

 private:
  // Where the columns of each row of one block cross segment boundaries.
  struct SplitTable {
    // The segments that hold columns of this block.
    int first_segment = 0;
    int last_segment = 0;
    // For stored row r and boundary b = 0 .. last_segment - first_segment - 1,
    // the offset within the row of the first column in segment
    // first_segment + b + 1.
    RawArray<uint32_t> offset;

    int num_boundaries() const { return last_segment - first_segment; }
  };

  void PartitionRows();
  void BuildSplitTable(const CsrBlock<S>& block, SplitTable* table) const;

  size_t dim_ = 0;
  std::vector<double> diagonal_;
  std::vector<CsrBlock<S>> blocks_;
  std::vector<SplitTable> splits_;  // one per block
  // Thread t owns rows [segment_[t], segment_[t + 1]).
  std::vector<size_t> segment_;
  bool apply_diagonal_ = true;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_SPARSE_H_H_
