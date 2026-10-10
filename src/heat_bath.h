// Sorted tables of double-excitation integrals for heat-bath selection.
//
// For every pair of spinors i < j the table lists the pairs a < b with
//   g = (ai|bj) - (aj|bi),
// the magnitude of the matrix element of the double excitation
// (i, j) -> (a, b), in order of decreasing |g|. Generating all double
// excitations of a determinant whose element exceeds a cutoff then costs time
// proportional to their number: walk each list until |g| drops below the
// cutoff.

#ifndef ZSHCI_SRC_HEAT_BATH_H_
#define ZSHCI_SRC_HEAT_BATH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm.h"
#include "integrals.h"

namespace zshci {

class HeatBath {
 public:
  struct Entry {
    double magnitude;
    uint16_t a;
    uint16_t b;
  };

  // Entries with |g| <= threshold are left out. With several ranks this is
  // collective: the root builds the tables and they are held once per node.
  HeatBath(const Integrals& ints, double threshold, const Comm& comm = Comm());

  // The list of the pair i < j, by decreasing magnitude.
  const Entry* begin(int i, int j) const {
    return entries_.data() + start_[PairIndex(i, j)];
  }
  const Entry* end(int i, int j) const {
    return entries_.data() + start_[PairIndex(i, j) + 1];
  }

  // Largest magnitude in the whole table.
  double max_magnitude() const { return max_magnitude_; }
  size_t num_entries() const { return entries_.size(); }

 private:
  static size_t PairIndex(int i, int j) {
    return static_cast<size_t>(j) * (j - 1) / 2 + i;
  }

  std::vector<size_t> start_;
  SharedArray<Entry> entries_;
  double max_magnitude_ = 0.0;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_HEAT_BATH_H_
