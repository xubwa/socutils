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

// An operator restricted to the basis vectors marked in `active`: P H P,
// where P projects on them, plus a shift on the others that puts them above
// every eigenvalue of the restricted operator. Its lowest eigenvectors are
// those of H in the marked subspace, with zeros elsewhere (as long as the
// subspace holds that many).
template <typename S>
class MaskedOperator : public LinearOperator<S> {
 public:
  // `inner` and `active` (one entry per basis vector) must outlive this.
  MaskedOperator(const LinearOperator<S>& inner,
                 const std::vector<uint8_t>& active)
      : inner_(inner), active_(active), diagonal_(inner.diagonal()) {
    double largest = 0.0;
    for (double d : diagonal_) largest = d > largest ? d : largest;
    shift_ = largest + 1.0;
    for (size_t i = 0; i < diagonal_.size(); ++i) {
      if (!active_[i]) diagonal_[i] = shift_;
    }
  }

  size_t dim() const override { return inner_.dim(); }
  const std::vector<double>& diagonal() const override { return diagonal_; }
  void Apply(const S* x, S* y) const override {
    const long n = static_cast<long>(dim());
    std::vector<S> masked(n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) masked[i] = active_[i] ? x[i] : S(0.0);
    inner_.Apply(masked.data(), y);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
      if (!active_[i]) y[i] = shift_ * x[i];
    }
  }

 private:
  const LinearOperator<S>& inner_;
  const std::vector<uint8_t>& active_;
  std::vector<double> diagonal_;
  double shift_ = 0.0;
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

  // y = H x. Each thread owns a contiguous range of rows. An element
  // H(j, i) serves y_j += H(j,i) x_i, which its row owner writes, and
  // y_i += conj(H(j,i)) x_j, which belongs to another thread's range. Two
  // ways of doing the second without two threads writing one element:
  //
  // One sweep (the default when it is affordable): the row owner adds the
  // second contribution into a vector of its own, and the vectors are summed
  // into y at the end. The matrix, which is what does not fit a cache, is
  // read once. The vectors take threads x dim numbers; this way is used
  // while that is at most a quarter of the stored elements.
  //
  // Two sweeps: every element is read a second time by the owner of its
  // column, for which every block keeps, per row, the positions where the
  // columns pass from one thread's range to the next. No extra vectors, but
  // the matrix is read twice.
  void Multiply(const S* x, S* y) const;

  // Which product Multiply uses: chosen as described above, or always one
  // of the two (for tests and comparisons).
  enum class Product { kAutomatic, kOneSweep, kTwoSweeps };
  void set_product(Product product) {
    product_ = product;
    PartitionRows();
  }

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
  Product product_ = Product::kAutomatic;
  bool one_sweep_ = false;
  // One sweep: thread t adds into scratch_[t * dim .. t * dim + its range
  // end), the columns its rows can have.
  mutable RawArray<S> scratch_;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_SPARSE_H_H_
