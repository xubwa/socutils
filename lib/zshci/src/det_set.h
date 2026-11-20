// A set of determinants for membership tests: open addressing on the hash
// of a determinant, linear probing, at most half full, with the hash kept
// beside each entry. One lookup touches
// one or two cache lines, against some 24 for a binary search in a sorted
// list of 1e7 determinants. Built once; held once per node.

#ifndef ZSHCI_SRC_DET_SET_H_
#define ZSHCI_SRC_DET_SET_H_

#include <cstdint>

#include "comm.h"
#include "det.h"
#include "parallel.h"

namespace zshci {

template <int W>
class DetSet {
 public:
  DetSet() = default;
  // Collective.
  DetSet(Span<Det<W>> dets, const Comm& comm) {
    size_t size = 16;
    while (size < 2 * dets.size()) size *= 2;
    mask_ = size - 1;
    keys_ = SharedArray<uint64_t>(size, comm);
    table_ = SharedArray<Det<W>>(size, comm);
    if (comm.is_node_leader()) {
      // Filled on every node by its leader: cheaper than sending it. A slot
      // is claimed by setting its key (the hash, never 0) atomically.
      uint64_t* const keys = keys_.data();
      Det<W>* const table = table_.data();
      const long count = static_cast<long>(size);
#pragma omp parallel for schedule(static)
      for (long k = 0; k < count; ++k) keys[k] = 0;
      const long n = static_cast<long>(dets.size());
#pragma omp parallel for schedule(static)
      for (long d = 0; d < n; ++d) {
        const Det<W>& det = dets[d];
        const uint64_t key = det.Hash() | 1;
        size_t slot = det.Hash() & mask_;
        while (true) {
          uint64_t expected = 0;
          if (__atomic_compare_exchange_n(&keys[slot], &expected, key, false,
                                          __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            table[slot] = det;
            break;
          }
          slot = (slot + 1) & mask_;
        }
      }
    }
    comm.Barrier();
  }

  bool Contains(const Det<W>& det) const {
    const uint64_t hash = det.Hash();
    const uint64_t key = hash | 1;
    size_t slot = hash & mask_;
    while (true) {
      const uint64_t k = keys_[slot];
      if (k == 0) return false;
      if (k == key && table_[slot] == det) return true;
      slot = (slot + 1) & mask_;
    }
  }

 private:
  SharedArray<uint64_t> keys_;
  SharedArray<Det<W>> table_;
  size_t mask_ = 0;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_DET_SET_H_
