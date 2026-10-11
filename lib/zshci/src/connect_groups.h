// Connected pairs of determinants from tables of strings.
//
// The spinors are divided into two groups, A and B, and a determinant is an
// A string and a B string. Far fewer distinct strings occur than
// determinants. The tables hold, for every distinct string, the determinants
// that contain it, ordered by their other string, and lists of the distinct
// strings that stand in certain relations to it; the connected pairs then
// follow from look-ups, each relation between strings being worked out once
// and used by every determinant that contains the string.
//
// Two determinants are connected (differ by at most a double excitation)
// exactly when their A strings and their B strings are related as follows,
// the columns giving how many electrons leave and enter a string:
//
//     A string          B string
//     same              single (1 out, 1 in) or double (2 out, 2 in)
//     single            same or single
//     double            same
//     1 out             1 in;  or 1 out, 2 in
//     2 out, 1 in       1 in
//     2 out             2 in
//
// and the same with A and B exchanged. The first three lines move no
// electron between the groups. When the number of electrons in each group
// is the same for all determinants - spin orbitals with a conserved spin
// projection, A the alpha and B the beta ones - the other lines never occur,
// no list for them is built, and this is the scheme of the non-relativistic
// Dice program. Spinors without such a conserved quantity need them.
//
// Related strings are found by sorting: two strings are related when they
// share a string of one or two electrons less, so each string contributes
// such shortened strings as keys and equal keys bring the related strings
// together. Doubles of one string with the other string unchanged are found
// by comparing the determinants that share the other string.

#ifndef ZSHCI_SRC_CONNECT_GROUPS_H_
#define ZSHCI_SRC_CONNECT_GROUPS_H_

#include <omp.h>

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
class GroupStringTables {
 public:
  // `group_a` has the bits of the spinors of group A set; all others form
  // group B. Collective: the root builds the tables of all determinants and
  // they are held once per node.
  GroupStringTables(Span<Det<W>> dets, const Det<W>& group_a, const Comm& comm)
      : dets_(dets),
        rank_(comm.rank()),
        num_ranks_(comm.size()),
        group_a_(group_a) {
    if (dets.size() > static_cast<size_t>(UINT32_MAX)) {
      throw std::length_error("too many determinants for a 32-bit index");
    }
    std::vector<uint32_t> string_of_det[2];
    std::vector<uint64_t> det_start[2];
    std::vector<uint32_t> det_other[2];
    std::vector<uint32_t> det_index[2];
    std::vector<uint64_t> related_start[2][kNumRelations];
    std::vector<uint32_t> related_list[2][kNumRelations];
    if (comm.is_root()) {
      std::vector<Det<W>> strings[2];
      for (int group = 0; group < 2; ++group) {
        NumberStrings(group, &strings[group], &string_of_det[group]);
      }
      for (int group = 0; group < 2; ++group) {
        GroupDets(strings[group].size(), string_of_det[group],
                  string_of_det[1 - group], &det_start[group],
                  &det_other[group], &det_index[group]);
        FindRelated(strings[group], related_start[group], related_list[group]);
      }
    }
    for (int group = 0; group < 2; ++group) {
      string_of_det_[group] =
          SharedArray<uint32_t>::FromRoot(string_of_det[group], comm);
      det_start_[group] =
          SharedArray<uint64_t>::FromRoot(det_start[group], comm);
      det_other_[group] =
          SharedArray<uint32_t>::FromRoot(det_other[group], comm);
      det_index_[group] =
          SharedArray<uint32_t>::FromRoot(det_index[group], comm);
      for (int r = 0; r < kNumRelations; ++r) {
        related_start_[group][r] =
            SharedArray<uint64_t>::FromRoot(related_start[group][r], comm);
        related_list_[group][r] =
            SharedArray<uint32_t>::FromRoot(related_list[group][r], comm);
      }
    }
    for (int r = 1; r < kNumRelations; ++r) {
      transfers_ = transfers_ || related_list_[0][r].size() > 0;
    }
  }

  size_t num_strings(int group) const { return det_start_[group].size() - 1; }
  // Number of entries in all relation lists.
  size_t num_links() const {
    size_t n = 0;
    for (int group = 0; group < 2; ++group) {
      for (int r = 0; r < kNumRelations; ++r)
        n += related_list_[group][r].size();
    }
    return n;
  }
  // Whether determinants differ in the number of electrons of a group.
  bool has_transfers() const { return transfers_; }

  // Whether this rank is responsible for the row of determinant j.
  bool Owns(size_t j) const {
    return static_cast<long>(j) % num_ranks_ == rank_;
  }

  // Calls visit(i) once for every determinant i < j connected to
  // determinant j or, with `both_sides`, for every determinant connected to
  // it. Adds to *tested_pairs the number of determinants that were compared
  // with j bit by bit.
  template <class Visitor>
  void ForEachConnected(size_t j, const Visitor& visit, uint64_t* tested_pairs,
                        bool both_sides = false) const {
    const uint32_t self = static_cast<uint32_t>(j);
    // Determinants up to this index are reported.
    const uint32_t limit = both_sides ? UINT32_MAX : self;
    uint64_t tested = 0;
    for (int group = 0; group < 2; ++group) {
      const int other = 1 - group;
      const uint32_t mine = string_of_det_[group][j];
      const uint32_t partner = string_of_det_[other][j];
      // The determinants that contain the partner string, by this string.
      const uint32_t* const with_partner =
          det_other_[other].data() + det_start_[other][partner];
      const uint32_t* const with_partner_end =
          det_other_[other].data() + det_start_[other][partner + 1];
      const uint32_t* const with_partner_dets =
          det_index_[other].data() + det_start_[other][partner];

      // A single excitation of this string, the partner unchanged.
      const uint32_t* const singles_end = RelatedEnd(group, kSingle, mine);
      for (const uint32_t* s = RelatedBegin(group, kSingle, mine);
           s != singles_end; ++s) {
        const uint32_t* const found =
            std::lower_bound(with_partner, with_partner_end, *s);
        if (found == with_partner_end || *found != *s) continue;
        const uint32_t i = with_partner_dets[found - with_partner];
        if (i < limit) visit(i);
      }

      // A double excitation of this string: compare with the determinants
      // that share the partner string.
      for (const uint32_t* d = with_partner_dets;
           d != with_partner_dets + (with_partner_end - with_partner); ++d) {
        const uint32_t i = *d;
        if (i >= limit || i == self) continue;
        ++tested;
        if (dets_[i].NumDifferentBits(dets_[j]) == 4) visit(i);
      }
    }

    // Both strings change. Each line of the table at the top of this file,
    // read from A to B and from B to A.
    const uint32_t a = string_of_det_[0][j];
    const uint32_t b = string_of_det_[1][j];
    Combine(limit, a, kSingle, b, kSingle, visit);
    if (!transfers_) {
      *tested_pairs += tested;
      return;
    }
    Combine(limit, a, kMinus1, b, kPlus1, visit);
    Combine(limit, a, kPlus1, b, kMinus1, visit);
    Combine(limit, a, kMinus1, b, kPlus12, visit);
    Combine(limit, a, kPlus12, b, kMinus1, visit);
    Combine(limit, a, kMinus21, b, kPlus1, visit);
    Combine(limit, a, kPlus1, b, kMinus21, visit);
    Combine(limit, a, kMinus2, b, kPlus2, visit);
    Combine(limit, a, kPlus2, b, kMinus2, visit);
    *tested_pairs += tested;
  }

 private:
  // How a string t is related to a string s: the number of electrons that
  // leave and enter s to give t.
  enum Relation : int {
    kSingle = 0,   // 1 out, 1 in
    kMinus1 = 1,   // 1 out
    kPlus1 = 2,    // 1 in
    kMinus21 = 3,  // 2 out, 1 in
    kPlus12 = 4,   // 1 out, 2 in
    kMinus2 = 5,   // 2 out
    kPlus2 = 6,    // 2 in
    kNumRelations = 7
  };

  const uint32_t* RelatedBegin(int group, int relation, uint32_t s) const {
    const SharedArray<uint64_t>& start = related_start_[group][relation];
    if (start.size() == 0) return nullptr;
    return related_list_[group][relation].data() + start[s];
  }
  const uint32_t* RelatedEnd(int group, int relation, uint32_t s) const {
    const SharedArray<uint64_t>& start = related_start_[group][relation];
    if (start.size() == 0) return nullptr;
    return related_list_[group][relation].data() + start[s + 1];
  }

  // Reports the determinants below `limit` whose A string is related to a
  // by relation_a and whose B string to b by relation_b: for every such A
  // string, the determinants that contain it, ordered by their B string,
  // are merged with the list of B strings.
  template <class Visitor>
  void Combine(uint32_t limit, uint32_t a, int relation_a, uint32_t b,
               int relation_b, const Visitor& visit) const {
    const uint32_t* const b_begin = RelatedBegin(1, relation_b, b);
    const uint32_t* const b_end = RelatedEnd(1, relation_b, b);
    if (b_begin == b_end) return;
    const uint32_t* const a_end = RelatedEnd(0, relation_a, a);
    for (const uint32_t* s = RelatedBegin(0, relation_a, a); s != a_end; ++s) {
      const uint32_t* const with_begin =
          det_other_[0].data() + det_start_[0][*s];
      const uint32_t* const with_end =
          det_other_[0].data() + det_start_[0][*s + 1];
      const uint32_t* const with_dets =
          det_index_[0].data() + det_start_[0][*s];
      const uint32_t* with = with_begin;
      const uint32_t* t = b_begin;
      while (with != with_end && t != b_end) {
        if (*with < *t) {
          ++with;
        } else if (*t < *with) {
          ++t;
        } else {
          const uint32_t i = with_dets[with - with_begin];
          if (i < limit) visit(i);
          ++with;
          ++t;
        }
      }
    }
  }

  Det<W> StringOf(const Det<W>& det, int group) const {
    Det<W> out;
    for (int k = 0; k < W; ++k) {
      out.words[k] =
          det.words[k] & (group == 0 ? group_a_.words[k] : ~group_a_.words[k]);
    }
    return out;
  }

  // The distinct strings of one group, sorted, and the number of its string
  // for every determinant.
  void NumberStrings(int group, std::vector<Det<W>>* strings,
                     std::vector<uint32_t>* string_of_det) const {
    const size_t n = dets_.size();
    std::vector<Det<W>> all(n);
    const long count = static_cast<long>(n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < count; ++i) all[i] = StringOf(dets_[i], group);
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

  // A string with `removed` of its electrons taken out, as a key.
  struct Shortened {
    Det<W> key;
    uint32_t string;
    uint8_t removed;
  };

  // Fills the relation lists of one group. Strings s and t are related when
  // they share a key: s with one electron out and t with one out is a
  // single; s with one out and t itself is "1 out"; s with two out and t
  // with one out is "2 out, 1 in" unless t is contained in s; s with two out
  // and t itself is "2 out". Keys with electrons taken out of the longer
  // string are only made when shorter strings exist at all.
  static void FindRelated(const std::vector<Det<W>>& strings,
                          std::vector<uint64_t>* start,
                          std::vector<uint32_t>* list) {
    const size_t num_strings = strings.size();
    int min_count = 64 * W;
    int max_count = 0;
    for (const Det<W>& s : strings) {
      min_count = std::min(min_count, s.Count());
      max_count = std::max(max_count, s.Count());
    }
    const bool transfers = max_count > min_count;

    // The keys, made by all threads and then dealt into parts by their
    // hash, so that equal keys meet in one part and the parts can be sorted
    // and swept side by side.
    std::vector<std::vector<Shortened>> chunks(omp_get_max_threads());
    const long count = static_cast<long>(num_strings);
#pragma omp parallel
    {
      std::vector<Shortened>& mine = chunks[omp_get_thread_num()];
      int occ[64 * W];
#pragma omp for schedule(dynamic, 256) nowait
      for (long s = 0; s < count; ++s) {
        int num_occ = 0;
        for (int k = 0; k < W; ++k) {
          uint64_t word = strings[s].words[k];
          while (word != 0) {
            occ[num_occ++] = 64 * k + __builtin_ctzll(word);
            word &= word - 1;
          }
        }
        const uint32_t id = static_cast<uint32_t>(s);
        if (transfers) mine.push_back({strings[s], id, 0});
        for (int m = 0; m < num_occ; ++m) {
          Shortened one{strings[s], id, 1};
          one.key.Clear(occ[m]);
          mine.push_back(one);
          // Two out only when a shorter string could match.
          if (!transfers || num_occ <= min_count) continue;
          for (int l = 0; l < m; ++l) {
            Shortened two{one.key, id, 2};
            two.key.Clear(occ[l]);
            mine.push_back(two);
          }
        }
      }
    }
    const size_t num_parts = 8 * static_cast<size_t>(omp_get_max_threads());
    Partitioned<Shortened> keys = PartitionChunks(
        &chunks, num_parts,
        [&](const Shortened& item) { return item.key.Hash() % num_parts; });
#pragma omp parallel for schedule(dynamic, 1)
    for (long part = 0; part < static_cast<long>(num_parts); ++part) {
      std::sort(keys.begin(part), keys.end(part),
                [](const Shortened& x, const Shortened& y) {
                  if (x.key != y.key) return x.key < y.key;
                  if (x.removed != y.removed) return x.removed < y.removed;
                  return x.string < y.string;
                });
    }

    const int num_lists = transfers ? kNumRelations : 1;
    for (int r = 0; r < num_lists; ++r) start[r].assign(num_strings + 1, 0);
    // First sweep counts, second fills; several threads may meet at one
    // string.
    std::vector<uint64_t> cursor[kNumRelations];
    for (int fill = 0; fill < 2; ++fill) {
      auto add = [&](int relation, uint32_t from, uint32_t to) {
        if (fill == 0) {
#pragma omp atomic
          ++start[relation][from + 1];
        } else {
          uint64_t slot;
#pragma omp atomic capture
          slot = cursor[relation][from]++;
          list[relation][slot] = to;
        }
      };
#pragma omp parallel for schedule(dynamic, 1)
      for (long part = 0; part < static_cast<long>(num_parts); ++part) {
        const Shortened* run = keys.begin(part);
        const Shortened* const part_end = keys.end(part);
        while (run != part_end) {
          const Shortened* end = run + 1;
          while (end != part_end && end->key == run->key) ++end;
          // The run is ordered by the number of electrons removed: 0 up to
          // first[1], 1 up to first[2], 2 up to first[3].
          const Shortened* first[4] = {run, run, run, end};
          while (first[1] != end && first[1]->removed < 1) ++first[1];
          first[2] = first[1];
          while (first[2] != end && first[2]->removed < 2) ++first[2];
          // Singles: two strings with one electron out each.
          for (const Shortened* x = first[1]; x != first[2]; ++x) {
            for (const Shortened* y = first[1]; y != first[2]; ++y) {
              if (x != y) add(kSingle, x->string, y->string);
            }
          }
          if (transfers) {
            for (const Shortened* t = first[0]; t != first[1]; ++t) {
              const uint32_t shorter = t->string;
              for (const Shortened* x = first[1]; x != first[2]; ++x) {
                add(kMinus1, x->string, shorter);
                add(kPlus1, shorter, x->string);
              }
              for (const Shortened* x = first[2]; x != first[3]; ++x) {
                add(kMinus2, x->string, shorter);
                add(kPlus2, shorter, x->string);
              }
            }
            for (const Shortened* x = first[2]; x != first[3]; ++x) {
              const uint32_t longer = x->string;
              for (const Shortened* t = first[1]; t != first[2]; ++t) {
                const uint32_t shorter = t->string;
                // Contained strings are "1 out", found above.
                if (Contains(strings[longer], strings[shorter])) continue;
                add(kMinus21, longer, shorter);
                add(kPlus12, shorter, longer);
              }
            }
          }
          run = end;
        }
      }
      if (fill == 0) {
        for (int r = 0; r < num_lists; ++r) {
          for (size_t s = 0; s < num_strings; ++s) {
            start[r][s + 1] += start[r][s];
          }
          list[r].resize(start[r][num_strings]);
          cursor[r].assign(start[r].begin(), start[r].end() - 1);
        }
      }
    }
    for (int r = 0; r < num_lists; ++r) {
#pragma omp parallel for schedule(dynamic, 256)
      for (long s = 0; s < count; ++s) {
        std::sort(list[r].begin() + start[r][s],
                  list[r].begin() + start[r][s + 1]);
      }
    }
  }

  static bool Contains(const Det<W>& outer, const Det<W>& inner) {
    for (int k = 0; k < W; ++k) {
      if ((inner.words[k] & ~outer.words[k]) != 0) return false;
    }
    return true;
  }

  Span<Det<W>> dets_;
  long rank_;
  long num_ranks_;
  Det<W> group_a_;
  bool transfers_ = false;
  // Index 0: strings of group A, 1: of group B.
  SharedArray<uint32_t> string_of_det_[2];
  // The determinants that contain string s are det_index_[det_start_[s] ..
  // det_start_[s + 1]), ordered by the number of their other string, which
  // is in det_other_.
  SharedArray<uint64_t> det_start_[2];
  SharedArray<uint32_t> det_other_[2];
  SharedArray<uint32_t> det_index_[2];
  // The strings related to string s by relation r are
  // related_list_[r][related_start_[r][s] .. related_start_[r][s + 1]),
  // increasing. Empty arrays for relations that do not occur.
  SharedArray<uint64_t> related_start_[2][kNumRelations];
  SharedArray<uint32_t> related_list_[2][kNumRelations];
};

}  // namespace internal
}  // namespace zshci

#endif  // ZSHCI_SRC_CONNECT_GROUPS_H_
