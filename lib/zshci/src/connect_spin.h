// Connected pairs of determinants from tables of alpha and beta strings.
//
// With spin orbitals 2P (alpha) and 2P + 1 (beta) and a conserved spin
// projection, a determinant is an alpha string and a beta string, and a
// connected pair differs by
//   a single excitation of one string, the other string being the same;
//   a single excitation of each string;
//   a double excitation of one string, the other being the same.
// Far fewer distinct strings occur than determinants. The tables hold, for
// every distinct string, the determinants that contain it, ordered by their
// other string, and the distinct strings one single excitation away. The
// pairs of the first two kinds then follow from look-ups, the relations
// between strings being worked out once and used by every determinant that
// contains the string; those of the third kind are found by comparing the
// determinants that share the other string. This is the scheme of the
// non-relativistic Dice program.
//
// It relies on the number of electrons in each string being the same for
// all determinants; without that, excitations that move an electron from
// one string to the other would be missed.

#ifndef ZSHCI_SRC_CONNECT_SPIN_H_
#define ZSHCI_SRC_CONNECT_SPIN_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "comm.h"
#include "det.h"
#include "parallel.h"
#include "sparse_h.h"

namespace zshci {
namespace internal {

template <int W>
class SpinStringTables {
 public:
  // Collective: the root builds the tables of all determinants and they are
  // held once per node.
  SpinStringTables(Span<Det<W>> dets, const Comm& comm)
      : dets_(dets), rank_(comm.rank()), num_ranks_(comm.size()) {
    if (dets.size() > static_cast<size_t>(UINT32_MAX)) {
      throw std::length_error("too many determinants for a 32-bit index");
    }
    for (int k = 0; k < W; ++k) alpha_mask_.words[k] = 0x5555555555555555ULL;
    std::vector<uint32_t> string_of_det[2];
    std::vector<uint64_t> det_start[2];
    std::vector<uint32_t> det_other[2];
    std::vector<uint32_t> det_index[2];
    std::vector<uint64_t> single_start[2];
    std::vector<uint32_t> single_list[2];
    if (comm.is_root()) {
      std::vector<Det<W>> strings[2];
      for (int spin = 0; spin < 2; ++spin) {
        NumberStrings(spin, &strings[spin], &string_of_det[spin]);
      }
      for (int spin = 0; spin < 2; ++spin) {
        GroupDets(strings[spin].size(), string_of_det[spin],
                  string_of_det[1 - spin], &det_start[spin], &det_other[spin],
                  &det_index[spin]);
        FindSingles(strings[spin], &single_start[spin], &single_list[spin]);
      }
    }
    for (int spin = 0; spin < 2; ++spin) {
      string_of_det_[spin] =
          SharedArray<uint32_t>::FromRoot(string_of_det[spin], comm);
      det_start_[spin] = SharedArray<uint64_t>::FromRoot(det_start[spin], comm);
      det_other_[spin] = SharedArray<uint32_t>::FromRoot(det_other[spin], comm);
      det_index_[spin] = SharedArray<uint32_t>::FromRoot(det_index[spin], comm);
      single_start_[spin] =
          SharedArray<uint64_t>::FromRoot(single_start[spin], comm);
      single_list_[spin] =
          SharedArray<uint32_t>::FromRoot(single_list[spin], comm);
    }
  }

  size_t num_strings(int spin) const { return det_start_[spin].size() - 1; }
  size_t num_single_links(int spin) const { return single_list_[spin].size(); }

  // Whether this rank is responsible for the row of determinant j.
  bool Owns(size_t j) const {
    return static_cast<long>(j) % num_ranks_ == rank_;
  }

  // Calls visit(i) once for every determinant i < j connected to
  // determinant j. Adds to *tested_pairs the number of determinants that
  // were compared with j bit by bit.
  template <class Visitor>
  void ForEachConnected(size_t j, const Visitor& visit,
                        uint64_t* tested_pairs) const {
    uint64_t tested = 0;
    {
      const uint32_t self = static_cast<uint32_t>(j);
      for (int spin = 0; spin < 2; ++spin) {
        const int other = 1 - spin;
        const uint32_t mine = string_of_det_[spin][j];
        const uint32_t partner = string_of_det_[other][j];
        // The determinants that contain the partner string, by this string.
        const uint32_t* const with_partner =
            det_other_[other].data() + det_start_[other][partner];
        const uint32_t* const with_partner_end =
            det_other_[other].data() + det_start_[other][partner + 1];
        const uint32_t* const with_partner_dets =
            det_index_[other].data() + det_start_[other][partner];

        const uint32_t* const singles =
            single_list_[spin].data() + single_start_[spin][mine];
        const uint32_t* const singles_end =
            single_list_[spin].data() + single_start_[spin][mine + 1];

        // A single excitation of this string.
        for (const uint32_t* s = singles; s != singles_end; ++s) {
          const uint32_t* const found =
              std::lower_bound(with_partner, with_partner_end, *s);
          if (found == with_partner_end || *found != *s) continue;
          const uint32_t i = with_partner_dets[found - with_partner];
          if (i < self) visit(i);
        }

        // A double excitation of this string: compare with the determinants
        // that share the partner string.
        for (const uint32_t* d = with_partner_dets;
             d != with_partner_dets + (with_partner_end - with_partner); ++d) {
          const uint32_t i = *d;
          if (i >= self) continue;
          ++tested;
          if (dets_[i].NumDifferentBits(dets_[j]) == 4) visit(i);
        }

        // A single excitation of each string; counted from the alpha side.
        if (spin != 0) continue;
        const uint32_t* const partner_singles =
            single_list_[other].data() + single_start_[other][partner];
        const uint32_t* const partner_singles_end =
            single_list_[other].data() + single_start_[other][partner + 1];
        for (const uint32_t* s = singles; s != singles_end; ++s) {
          // The determinants that contain the excited string, by their
          // other string: merge with the singles of the partner.
          const uint32_t* with = det_other_[spin].data() + det_start_[spin][*s];
          const uint32_t* const with_end =
              det_other_[spin].data() + det_start_[spin][*s + 1];
          const uint32_t* const with_dets =
              det_index_[spin].data() + det_start_[spin][*s];
          const uint32_t* const with_begin = with;
          const uint32_t* t = partner_singles;
          while (with != with_end && t != partner_singles_end) {
            if (*with < *t) {
              ++with;
            } else if (*t < *with) {
              ++t;
            } else {
              const uint32_t i = with_dets[with - with_begin];
              if (i < self) visit(i);
              ++with;
              ++t;
            }
          }
        }
      }
    }
    *tested_pairs += tested;
  }

 private:
  Det<W> StringOf(const Det<W>& det, int spin) const {
    Det<W> out;
    for (int k = 0; k < W; ++k) {
      out.words[k] = det.words[k] &
                     (spin == 0 ? alpha_mask_.words[k] : ~alpha_mask_.words[k]);
    }
    return out;
  }

  // The distinct strings of one spin, sorted, and the number of its string
  // for every determinant.
  void NumberStrings(int spin, std::vector<Det<W>>* strings,
                     std::vector<uint32_t>* string_of_det) const {
    const size_t n = dets_.size();
    std::vector<Det<W>> all(n);
    const long count = static_cast<long>(n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < count; ++i) all[i] = StringOf(dets_[i], spin);
    *strings = all;
    std::sort(strings->begin(), strings->end());
    strings->erase(std::unique(strings->begin(), strings->end()),
                   strings->end());
    string_of_det->resize(n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < count; ++i) {
      (*string_of_det)[i] = static_cast<uint32_t>(
          std::lower_bound(strings->begin(), strings->end(), all[i]) -
          strings->begin());
    }
  }

  // For every string, the determinants that contain it: the number of their
  // other string, increasing, and their index.
  static void GroupDets(size_t num_strings,
                        const std::vector<uint32_t>& string_of_det,
                        const std::vector<uint32_t>& other_of_det,
                        std::vector<uint64_t>* start,
                        std::vector<uint32_t>* other,
                        std::vector<uint32_t>* index) {
    const size_t n = string_of_det.size();
    start->assign(num_strings + 1, 0);
    for (size_t i = 0; i < n; ++i) ++(*start)[string_of_det[i] + 1];
    for (size_t s = 0; s < num_strings; ++s) (*start)[s + 1] += (*start)[s];
    std::vector<uint64_t> cursor(start->begin(), start->end() - 1);
    std::vector<std::pair<uint32_t, uint32_t>> items(n);
    for (size_t i = 0; i < n; ++i) {
      items[cursor[string_of_det[i]]++] = {other_of_det[i],
                                           static_cast<uint32_t>(i)};
    }
    const long count = static_cast<long>(num_strings);
#pragma omp parallel for schedule(dynamic, 256)
    for (long s = 0; s < count; ++s) {
      std::sort(items.begin() + (*start)[s], items.begin() + (*start)[s + 1]);
    }
    other->resize(n);
    index->resize(n);
    for (size_t k = 0; k < n; ++k) {
      (*other)[k] = items[k].first;
      (*index)[k] = items[k].second;
    }
  }

  // For every string, the strings one single excitation away, increasing.
  // Two such strings share exactly one string with one electron less, so
  // sorting the (shortened string, string) pairs brings them together.
  static void FindSingles(const std::vector<Det<W>>& strings,
                          std::vector<uint64_t>* start,
                          std::vector<uint32_t>* list) {
    struct Shortened {
      Det<W> key;
      uint32_t string;
    };
    std::vector<Shortened> shortened;
    for (size_t s = 0; s < strings.size(); ++s) {
      for (int k = 0; k < W; ++k) {
        uint64_t word = strings[s].words[k];
        while (word != 0) {
          Shortened item{strings[s], static_cast<uint32_t>(s)};
          item.key.Clear(64 * k + __builtin_ctzll(word));
          shortened.push_back(item);
          word &= word - 1;
        }
      }
    }
    std::sort(shortened.begin(), shortened.end(),
              [](const Shortened& a, const Shortened& b) {
                return a.key != b.key ? a.key < b.key : a.string < b.string;
              });
    start->assign(strings.size() + 1, 0);
    for (int fill = 0; fill < 2; ++fill) {
      std::vector<uint64_t> cursor(start->begin(), start->end() - 1);
      size_t run = 0;
      while (run < shortened.size()) {
        size_t end = run + 1;
        while (end < shortened.size() &&
               shortened[end].key == shortened[run].key) {
          ++end;
        }
        for (size_t a = run; a < end; ++a) {
          for (size_t b = run; b < end; ++b) {
            if (a == b) continue;
            if (fill == 0) {
              ++(*start)[shortened[a].string + 1];
            } else {
              (*list)[cursor[shortened[a].string]++] = shortened[b].string;
            }
          }
        }
        run = end;
      }
      if (fill == 0) {
        for (size_t s = 0; s < strings.size(); ++s) {
          (*start)[s + 1] += (*start)[s];
        }
        list->resize((*start)[strings.size()]);
      }
    }
    const long count = static_cast<long>(strings.size());
#pragma omp parallel for schedule(dynamic, 256)
    for (long s = 0; s < count; ++s) {
      std::sort(list->begin() + (*start)[s], list->begin() + (*start)[s + 1]);
    }
  }

  Span<Det<W>> dets_;
  long rank_;
  long num_ranks_;
  Det<W> alpha_mask_;
  // Index 0: alpha strings, 1: beta strings.
  SharedArray<uint32_t> string_of_det_[2];
  // The determinants that contain string s are det_index_[det_start_[s] ..
  // det_start_[s + 1]), ordered by the number of their other string, which
  // is in det_other_.
  SharedArray<uint64_t> det_start_[2];
  SharedArray<uint32_t> det_other_[2];
  SharedArray<uint32_t> det_index_[2];
  // The strings one single excitation from string s are
  // single_list_[single_start_[s] .. single_start_[s + 1]), increasing.
  SharedArray<uint64_t> single_start_[2];
  SharedArray<uint32_t> single_list_[2];
};

}  // namespace internal
}  // namespace zshci

#endif  // ZSHCI_SRC_CONNECT_SPIN_H_
