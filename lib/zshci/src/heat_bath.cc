#include "heat_bath.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace zshci {
namespace {

// Ties are broken by the orbital pair so that the order, and with it the
// selected space, does not depend on the sort implementation.
void SortList(std::vector<HeatBath::Entry>* list) {
  std::sort(list->begin(), list->end(),
            [](const HeatBath::Entry& x, const HeatBath::Entry& y) {
              if (x.magnitude != y.magnitude) {
                return x.magnitude > y.magnitude;
              }
              return x.b != y.b ? x.b < y.b : x.a < y.a;
            });
}

}  // namespace

template <typename S>
void HeatBath::BuildSpinorLists(const Integrals<S>& ints, double threshold,
                                std::vector<std::vector<Entry>>* lists) const {
  const int n = ints.num_orbs();
  lists->resize(static_cast<size_t>(n) * (n - 1) / 2);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 1; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      std::vector<Entry>& list = (*lists)[PairIndex(i, j)];
      // Only the pairs in the symmetry block of (i, j) can be reached.
      const auto* const partners_end = ints.PartnersEnd(i, j);
      for (const auto* pair = ints.PartnersBegin(i, j); pair != partners_end;
           ++pair) {
        const double magnitude =
            Abs(ints.AntisymOrdered(pair->first, pair->second, i, j));
        if (magnitude > threshold) {
          list.push_back({magnitude, pair->first, pair->second});
        }
      }
      SortList(&list);
    }
  }
}

template <typename S>
void HeatBath::BuildSpatialLists(const Integrals<S>& ints, double threshold,
                                 std::vector<std::vector<Entry>>* lists) const {
  const int n = ints.num_spatial();
  const size_t num_same = static_cast<size_t>(n) * (n - 1) / 2;
  const size_t num_opposite = static_cast<size_t>(n) * (n + 1) / 2;
  lists->resize(num_same + num_opposite);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= j; ++i) {
      if (i < j) {
        std::vector<Entry>& same = (*lists)[PairIndex(i, j)];
        for (int b = 1; b < n; ++b) {
          for (int a = 0; a < b; ++a) {
            const double magnitude = Abs(ints.SpatialTwoBody(a, i, b, j) -
                                         ints.SpatialTwoBody(a, j, b, i));
            if (magnitude > threshold) {
              same.push_back({magnitude, static_cast<uint16_t>(a),
                              static_cast<uint16_t>(b)});
            }
          }
        }
        SortList(&same);
      }
      std::vector<Entry>& opposite =
          (*lists)[num_same + PairIndexWithDiagonal(i, j)];
      for (int b = 0; b < n; ++b) {
        for (int a = 0; a < n; ++a) {
          const double magnitude = Abs(ints.SpatialTwoBody(a, i, b, j));
          if (magnitude > threshold) {
            opposite.push_back({magnitude, static_cast<uint16_t>(a),
                                static_cast<uint16_t>(b)});
          }
        }
      }
      SortList(&opposite);
    }
  }
}

template <typename S>
HeatBath::HeatBath(const Integrals<S>& ints, double threshold,
                   const Comm& comm) {
  if (ints.num_orbs() > UINT16_MAX) {
    throw std::length_error("HeatBath: too many spinors");
  }
  spatial_ = ints.is_spatial();
  if (spatial_) {
    const size_t n = ints.num_spatial();
    opposite_offset_ = n * (n - 1) / 2;
  }
  std::vector<std::vector<Entry>> lists;
  size_t num_lists = 0;
  if (comm.is_root()) {
    if (spatial_) {
      BuildSpatialLists(ints, threshold, &lists);
    } else {
      BuildSpinorLists(ints, threshold, &lists);
    }
    num_lists = lists.size();
    start_.assign(num_lists + 1, 0);
    for (size_t k = 0; k < num_lists; ++k) {
      start_[k + 1] = start_[k] + lists[k].size();
      if (!lists[k].empty()) {
        max_magnitude_ = std::max(max_magnitude_, lists[k][0].magnitude);
      }
    }
  }
  comm.Bcast(&start_);
  comm.BcastValue(&max_magnitude_);
  num_lists = start_.size() - 1;

  entries_ = SharedArray<Entry>(start_[num_lists], comm);
  if (comm.is_root()) {
    for (size_t k = 0; k < num_lists; ++k) {
      std::copy(lists[k].begin(), lists[k].end(),
                entries_.data() + start_[k]);
    }
  }
  entries_.ShareFromRoot(comm);
}

template HeatBath::HeatBath(const Integrals<double>&, double, const Comm&);
template HeatBath::HeatBath(const Integrals<Complex>&, double, const Comm&);

}  // namespace zshci
