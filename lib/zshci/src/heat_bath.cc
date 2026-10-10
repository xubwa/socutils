#include "heat_bath.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace zshci {

template <typename S>
HeatBath::HeatBath(const Integrals<S>& ints, double threshold,
                   const Comm& comm) {
  const int n = ints.num_orbs();
  if (n > UINT16_MAX) throw std::length_error("HeatBath: too many spinors");
  const size_t num_pairs = static_cast<size_t>(n) * (n - 1) / 2;
  std::vector<std::vector<Entry>> lists(comm.is_root() ? num_pairs : 0);

  if (comm.is_root()) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 1; j < n; ++j) {
      for (int i = 0; i < j; ++i) {
        std::vector<Entry>& list = lists[PairIndex(i, j)];
        // Only the pairs in the symmetry block of (i, j) can be reached.
        const auto* const partners_end = ints.PartnersEnd(i, j);
        for (const auto* pair = ints.PartnersBegin(i, j); pair != partners_end;
             ++pair) {
          const double magnitude =
              std::abs(ints.AntisymOrdered(pair->first, pair->second, i, j));
          if (magnitude > threshold) {
            list.push_back({magnitude, pair->first, pair->second});
          }
        }
        // Ties are broken by the orbital pair so that the order, and with it
        // the selected space, does not depend on the sort implementation.
        std::sort(list.begin(), list.end(),
                  [](const Entry& x, const Entry& y) {
                    if (x.magnitude != y.magnitude) {
                      return x.magnitude > y.magnitude;
                    }
                    return x.b != y.b ? x.b < y.b : x.a < y.a;
                  });
      }
    }
    start_.assign(num_pairs + 1, 0);
    for (size_t k = 0; k < num_pairs; ++k) {
      start_[k + 1] = start_[k] + lists[k].size();
      if (!lists[k].empty()) {
        max_magnitude_ = std::max(max_magnitude_, lists[k][0].magnitude);
      }
    }
  }
  comm.Bcast(&start_);
  comm.BcastValue(&max_magnitude_);

  entries_ = SharedArray<Entry>(start_[num_pairs], comm);
  if (comm.is_root()) {
    for (size_t k = 0; k < num_pairs; ++k) {
      std::copy(lists[k].begin(), lists[k].end(),
                entries_.data() + start_[k]);
    }
  }
  entries_.ShareFromRoot(comm);
}

template HeatBath::HeatBath(const Integrals<double>&, double, const Comm&);
template HeatBath::HeatBath(const Integrals<Complex>&, double, const Comm&);

}  // namespace zshci
