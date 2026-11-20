// Additive symmetry labels.

#ifndef ZSHCI_SRC_LABELS_H_
#define ZSHCI_SRC_LABELS_H_

#include <vector>

namespace zshci {

// One conserved quantity: an integer label per spinor whose sum over the
// occupied spinors, modulo `mod` if that is nonzero, is the same for all
// determinants of a calculation. Examples: twice the angular momentum
// projection of a linear molecule (mod = 0) or of an n-fold axis (mod = 2n);
// 0 and 1 modulo 2 for inversion; +1 and -1 for the spin projection.
// Several sets are independent of each other.
struct LabelSet {
  std::vector<int> labels;
  int mod = 0;
};

// Whether a change of the summed label is no change.
inline bool LabelConserved(int change, int mod) {
  return mod == 0 ? change == 0 : change % mod == 0;
}

}  // namespace zshci

#endif  // ZSHCI_SRC_LABELS_H_
