// Sorting with the threads of the process, for the long lists that one
// rank sorts while the others wait (the determinants of the space, the
// candidates of a selection).

#ifndef ZSHCI_SRC_PARALLEL_SORT_H_
#define ZSHCI_SRC_PARALLEL_SORT_H_

#include <omp.h>

#include <algorithm>
#include <functional>
#include <parallel/algorithm>

namespace zshci {

// As std::sort; the order of equal elements is unspecified, so the result
// is the same as std::sort's whenever no two elements are equivalent
// without being identical.
template <class It, class Less = std::less<>>
void ParallelSort(It first, It last, Less less = Less()) {
  if (last - first < (1 << 16) || omp_get_max_threads() == 1 ||
      omp_in_parallel()) {
    std::sort(first, last, less);
    return;
  }
  __gnu_parallel::sort(first, last, less,
                       __gnu_parallel::multiway_mergesort_tag());
}

}  // namespace zshci

#endif  // ZSHCI_SRC_PARALLEL_SORT_H_
