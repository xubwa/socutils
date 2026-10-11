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
  // group B. The tables are empty until Extend is called.
  GroupStringTables(const Det<W>& group_a, const Comm& comm)
      : comm_(comm),
        rank_(comm.rank()),
        num_ranks_(comm.size()),
        group_a_(group_a) {}

  // The tables of `dets`. Collective.
  GroupStringTables(Span<Det<W>> dets, const Det<W>& group_a, const Comm& comm)
      : GroupStringTables(group_a, comm) {
    Extend(dets, 0);
  }

  // Makes the tables those of `dets`, of which the determinants before
  // first_new are the ones of the previous call, unchanged. Only what the
  // new determinants bring is worked out: strings not seen before get the
  // next numbers, their keys are merged into the sorted keys kept from
  // before, and the relation lists gain the pairs that involve a new
  // string. Collective: the root does this and the tables are held once per
  // node.
  void Extend(Span<Det<W>> dets, size_t first_new) {
    if (dets.size() > static_cast<size_t>(UINT32_MAX)) {
      throw std::length_error("too many determinants for a 32-bit index");
    }
    dets_ = dets;
    std::vector<uint64_t> det_start[2];
    std::vector<uint32_t> det_other[2];
    std::vector<uint32_t> det_index[2];
    if (comm_.is_root()) {
      if (first_new == 0) root_ = RootState();
      size_t first_new_string[2];
      for (int group = 0; group < 2; ++group) {
        first_new_string[group] = NumberStrings(group, first_new);
      }
      for (int group = 0; group < 2; ++group) {
        GroupDets(root_.strings[group].size(), root_.string_of_det[group],
                  root_.string_of_det[1 - group], &det_start[group],
                  &det_other[group], &det_index[group]);
        UpdateRelated(group, first_new_string[group]);
      }
    }
    transfers_ = false;
    for (int group = 0; group < 2; ++group) {
      string_of_det_[group] =
          SharedArray<uint32_t>::FromRoot(root_.string_of_det[group], comm_);
      det_start_[group] =
          SharedArray<uint64_t>::FromRoot(det_start[group], comm_);
      det_other_[group] =
          SharedArray<uint32_t>::FromRoot(det_other[group], comm_);
      det_index_[group] =
          SharedArray<uint32_t>::FromRoot(det_index[group], comm_);
      for (int r = 0; r < kNumRelations; ++r) {
        related_start_[group][r] = SharedArray<uint64_t>::FromRoot(
            root_.related_start[group][r], comm_);
        related_list_[group][r] = SharedArray<uint32_t>::FromRoot(
            root_.related_list[group][r], comm_);
        if (r > 0 && related_list_[group][r].size() > 0) transfers_ = true;
      }
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
  // by relation_a and whose B string to b by relation_b. For every such A
  // string the determinants that contain it are known with their B strings,
  // and those whose B string is in the list of b are wanted. The list of b
  // is marked once in a bit array of this thread, so that each determinant
  // costs one bit test; only where an A string occurs in very many more
  // determinants than b has related strings, those are looked up in its
  // sorted determinants instead.
  template <class Visitor>
  void Combine(uint32_t limit, uint32_t a, int relation_a, uint32_t b,
               int relation_b, const Visitor& visit) const {
    const uint32_t* const b_begin = RelatedBegin(1, relation_b, b);
    const uint32_t* const b_end = RelatedEnd(1, relation_b, b);
    if (b_begin == b_end) return;
    const uint32_t* const a_begin = RelatedBegin(0, relation_a, a);
    const uint32_t* const a_end = RelatedEnd(0, relation_a, a);
    if (a_begin == a_end) return;
    const size_t num_b = b_end - b_begin;
    std::vector<uint64_t>& marks = Marks();
    bool marked = false;
    for (const uint32_t* s = a_begin; s != a_end; ++s) {
      const uint32_t* const with_begin =
          det_other_[0].data() + det_start_[0][*s];
      const uint32_t* const with_end =
          det_other_[0].data() + det_start_[0][*s + 1];
      const uint32_t* const with_dets =
          det_index_[0].data() + det_start_[0][*s];
      // A bit test in a small array costs far less than a step of a binary
      // search in a long list, hence the large factor.
      if (static_cast<size_t>(with_end - with_begin) > 64 * num_b) {
        for (const uint32_t* t = b_begin; t != b_end; ++t) {
          const uint32_t* const found =
              std::lower_bound(with_begin, with_end, *t);
          if (found == with_end || *found != *t) continue;
          const uint32_t i = with_dets[found - with_begin];
          if (i < limit) visit(i);
        }
        continue;
      }
      if (!marked) {
        for (const uint32_t* t = b_begin; t != b_end; ++t) {
          marks[*t >> 6] |= 1ULL << (*t & 63);
        }
        marked = true;
      }
      for (const uint32_t* with = with_begin; with != with_end; ++with) {
        if (((marks[*with >> 6] >> (*with & 63)) & 1ULL) == 0) continue;
        const uint32_t i = with_dets[with - with_begin];
        if (i < limit) visit(i);
      }
    }
    if (marked) {
      for (const uint32_t* t = b_begin; t != b_end; ++t) marks[*t >> 6] = 0;
    }
  }

  // A bit per B string, all zero between uses; one array per thread.
  std::vector<uint64_t>& Marks() const {
    static thread_local std::vector<uint64_t> marks;
    const size_t needed = num_strings(1) / 64 + 1;
    if (marks.size() < needed) marks.resize(needed, 0);
    return marks;
  }

  Det<W> StringOf(const Det<W>& det, int group) const {
    Det<W> out;
    for (int k = 0; k < W; ++k) {
      out.words[k] =
          det.words[k] & (group == 0 ? group_a_.words[k] : ~group_a_.words[k]);
    }
    return out;
  }

  // Numbers the strings of the determinants from first_new on: known
  // strings keep their number, new ones get the next numbers. Returns the
  // number of strings there were before.
  size_t NumberStrings(int group, size_t first_new) {
    std::vector<Det<W>>& strings = root_.strings[group];
    std::vector<std::pair<Det<W>, uint32_t>>& sorted = root_.sorted[group];
    std::vector<uint32_t>& string_of_det = root_.string_of_det[group];
    const size_t n = dets_.size();
    const size_t num_old = strings.size();
    const long begin = static_cast<long>(first_new);
    const long end = static_cast<long>(n);
    string_of_det.resize(n);
    std::vector<Det<W>> of_det(n - first_new);
    auto by_string = [](const std::pair<Det<W>, uint32_t>& entry,
                        const Det<W>& string) { return entry.first < string; };
#pragma omp parallel for schedule(static)
    for (long i = begin; i < end; ++i) {
      const Det<W> string = StringOf(dets_[i], group);
      of_det[i - begin] = string;
      const auto found =
          std::lower_bound(sorted.begin(), sorted.end(), string, by_string);
      string_of_det[i] = found != sorted.end() && found->first == string
                             ? found->second
                             : UINT32_MAX;
    }
    std::vector<Det<W>> fresh;
    for (long i = begin; i < end; ++i) {
      if (string_of_det[i] == UINT32_MAX) fresh.push_back(of_det[i - begin]);
    }
    std::sort(fresh.begin(), fresh.end());
    fresh.erase(std::unique(fresh.begin(), fresh.end()), fresh.end());
#pragma omp parallel for schedule(static)
    for (long i = begin; i < end; ++i) {
      if (string_of_det[i] != UINT32_MAX) continue;
      string_of_det[i] = static_cast<uint32_t>(
          num_old +
          (std::lower_bound(fresh.begin(), fresh.end(), of_det[i - begin]) -
           fresh.begin()));
    }
    const size_t middle = sorted.size();
    for (size_t k = 0; k < fresh.size(); ++k) {
      strings.push_back(fresh[k]);
      sorted.emplace_back(fresh[k], static_cast<uint32_t>(num_old + k));
    }
    std::inplace_merge(sorted.begin(), sorted.begin() + middle, sorted.end());
    return num_old;
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

  // A pair of related strings found from a shared key.
  struct Link {
    uint32_t from;
    uint32_t to;
    uint8_t relation;
  };

  static bool KeyLess(const Shortened& x, const Shortened& y) {
    if (x.key != y.key) return x.key < y.key;
    if (x.removed != y.removed) return x.removed < y.removed;
    return x.string < y.string;
  }

  // Brings the relation lists of one group up to date with the strings from
  // first_new_string on. Strings s and t are related when they share a key:
  // s with one electron out and t with one out is a single; s with one out
  // and t itself is "1 out"; s with two out and t with one out is "2 out,
  // 1 in" unless t is contained in s; s with two out and t itself is
  // "2 out". Keys with electrons taken out of the longer string are only
  // made when shorter strings exist at all; if new strings change that, the
  // keys of this group are made afresh.
  void UpdateRelated(int group, size_t first_new_string) {
    const std::vector<Det<W>>& strings = root_.strings[group];
    const size_t num_strings = strings.size();
    int min_count = 64 * W;
    int max_count = 0;
    for (const Det<W>& s : strings) {
      min_count = std::min(min_count, s.Count());
      max_count = std::max(max_count, s.Count());
    }
    const bool transfers = max_count > min_count;
    std::vector<std::vector<Shortened>>& kept = root_.keys[group];
    const size_t num_parts = 8 * static_cast<size_t>(omp_get_max_threads());
    if (first_new_string == 0 || kept.size() != num_parts ||
        transfers != root_.transfers[group] ||
        (transfers && min_count != root_.min_count[group])) {
      first_new_string = 0;
      kept.assign(num_parts, std::vector<Shortened>());
      for (int r = 0; r < kNumRelations; ++r) {
        root_.related_start[group][r].clear();
        root_.related_list[group][r].clear();
      }
    }
    root_.transfers[group] = transfers;
    root_.min_count[group] = min_count;

    // The keys of the new strings, made by all threads and dealt into parts
    // by their hash, so that equal keys meet in one part and the parts can
    // be handled side by side.
    std::vector<std::vector<Shortened>> chunks(omp_get_max_threads());
    const long count = static_cast<long>(num_strings);
#pragma omp parallel
    {
      std::vector<Shortened>& mine = chunks[omp_get_thread_num()];
      int occ[64 * W];
#pragma omp for schedule(dynamic, 256) nowait
      for (long s = static_cast<long>(first_new_string); s < count; ++s) {
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
    Partitioned<Shortened> fresh = PartitionChunks(
        &chunks, num_parts,
        [&](const Shortened& item) { return item.key.Hash() % num_parts; });

    // Every run of equal keys that holds a new key gives the pairs in which
    // at least one string is new. The new keys then join the kept ones.
    std::vector<std::vector<Link>> links(omp_get_max_threads());
#pragma omp parallel
    {
      std::vector<Link>& mine = links[omp_get_thread_num()];
      // The members of one run, kept ones first within equal `removed`.
      struct Member {
        uint32_t string;
        uint8_t removed;
        bool is_new;
      };
      std::vector<Member> run;
      auto add = [&](int relation, const Member& from, const Member& to) {
        if (!from.is_new && !to.is_new) return;
        mine.push_back(
            {from.string, to.string, static_cast<uint8_t>(relation)});
      };
#pragma omp for schedule(dynamic, 1)
      for (long part = 0; part < static_cast<long>(num_parts); ++part) {
        std::sort(fresh.begin(part), fresh.end(part), KeyLess);
        std::vector<Shortened>& old = kept[part];
        const Shortened* next = fresh.begin(part);
        const Shortened* const part_end = fresh.end(part);
        while (next != part_end) {
          const Shortened* run_end = next + 1;
          while (run_end != part_end && run_end->key == next->key) ++run_end;
          run.clear();
          const auto old_range =
              std::equal_range(old.begin(), old.end(), *next,
                               [](const Shortened& x, const Shortened& y) {
                                 return x.key < y.key;
                               });
          for (auto it = old_range.first; it != old_range.second; ++it) {
            run.push_back({it->string, it->removed, false});
          }
          for (const Shortened* it = next; it != run_end; ++it) {
            run.push_back({it->string, it->removed, true});
          }
          std::stable_sort(run.begin(), run.end(),
                           [](const Member& x, const Member& y) {
                             return x.removed < y.removed;
                           });
          // Ordered by the number of electrons removed: 0 up to first[1],
          // 1 up to first[2], 2 up to first[3].
          size_t first[4] = {0, 0, 0, run.size()};
          while (first[1] < run.size() && run[first[1]].removed < 1) ++first[1];
          first[2] = first[1];
          while (first[2] < run.size() && run[first[2]].removed < 2) ++first[2];
          // Singles: two strings with one electron out each.
          for (size_t x = first[1]; x < first[2]; ++x) {
            for (size_t y = first[1]; y < first[2]; ++y) {
              if (x != y) add(kSingle, run[x], run[y]);
            }
          }
          if (transfers) {
            for (size_t t = first[0]; t < first[1]; ++t) {
              for (size_t x = first[1]; x < first[2]; ++x) {
                add(kMinus1, run[x], run[t]);
                add(kPlus1, run[t], run[x]);
              }
              for (size_t x = first[2]; x < first[3]; ++x) {
                add(kMinus2, run[x], run[t]);
                add(kPlus2, run[t], run[x]);
              }
            }
            for (size_t x = first[2]; x < first[3]; ++x) {
              for (size_t t = first[1]; t < first[2]; ++t) {
                // Contained strings are "1 out", found above.
                if (Contains(strings[run[x].string], strings[run[t].string])) {
                  continue;
                }
                add(kMinus21, run[x], run[t]);
                add(kPlus12, run[t], run[x]);
              }
            }
          }
          next = run_end;
        }
        const size_t middle = old.size();
        old.insert(old.end(), fresh.begin(part), fresh.end(part));
        std::inplace_merge(old.begin(), old.begin() + middle, old.end(),
                           KeyLess);
      }
    }

    // The lists: what was there, then the new pairs, sorted where a list
    // gained entries. Several threads may meet at one string.
    const int num_lists = transfers ? kNumRelations : 1;
    const long num_chunks = static_cast<long>(links.size());
    size_t num_links = 0;
    for (const std::vector<Link>& chunk : links) num_links += chunk.size();
    std::vector<uint64_t> added[kNumRelations];
    std::vector<uint64_t> cursor[kNumRelations];
    std::vector<uint64_t> new_start[kNumRelations];
    std::vector<uint32_t> new_list[kNumRelations];
    for (int r = 0; r < num_lists; ++r) added[r].assign(num_strings, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < num_chunks; ++c) {
      for (const Link& link : links[c]) {
#pragma omp atomic
        ++added[link.relation][link.from];
      }
    }
    for (int r = 0; r < num_lists; ++r) {
      const std::vector<uint64_t>& start = root_.related_start[group][r];
      const std::vector<uint32_t>& list = root_.related_list[group][r];
      const size_t num_old = start.empty() ? 0 : start.size() - 1;
      new_start[r].assign(num_strings + 1, 0);
      for (size_t s = 0; s < num_strings; ++s) {
        const uint64_t before = s < num_old ? start[s + 1] - start[s] : 0;
        new_start[r][s + 1] = new_start[r][s] + before + added[r][s];
      }
      new_list[r].resize(new_start[r][num_strings]);
      cursor[r].resize(num_strings);
#pragma omp parallel for schedule(static)
      for (long s = 0; s < count; ++s) {
        const uint64_t before =
            static_cast<size_t>(s) < num_old ? start[s + 1] - start[s] : 0;
        if (before > 0) {
          std::copy(list.begin() + start[s], list.begin() + start[s + 1],
                    new_list[r].begin() + new_start[r][s]);
        }
        cursor[r][s] = new_start[r][s] + before;
      }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < num_chunks; ++c) {
      for (const Link& link : links[c]) {
        uint64_t slot;
#pragma omp atomic capture
        slot = cursor[link.relation][link.from]++;
        new_list[link.relation][slot] = link.to;
      }
    }
    for (int r = 0; r < num_lists; ++r) {
      if (num_links > 0) {
#pragma omp parallel for schedule(dynamic, 256)
        for (long s = 0; s < count; ++s) {
          if (added[r][s] == 0) continue;
          std::sort(new_list[r].begin() + new_start[r][s],
                    new_list[r].begin() + new_start[r][s + 1]);
        }
      }
      root_.related_start[group][r] = std::move(new_start[r]);
      root_.related_list[group][r] = std::move(new_list[r]);
    }
  }

  static bool Contains(const Det<W>& outer, const Det<W>& inner) {
    for (int k = 0; k < W; ++k) {
      if ((inner.words[k] & ~outer.words[k]) != 0) return false;
    }
    return true;
  }

  // What the root keeps between calls of Extend.
  struct RootState {
    std::vector<Det<W>> strings[2];                      // by number
    std::vector<std::pair<Det<W>, uint32_t>> sorted[2];  // string, number
    std::vector<uint32_t> string_of_det[2];
    // The keys of all strings, in parts by hash, each part sorted.
    std::vector<std::vector<Shortened>> keys[2];
    // What the keys were made for.
    bool transfers[2] = {false, false};
    int min_count[2] = {0, 0};
    std::vector<uint64_t> related_start[2][kNumRelations];
    std::vector<uint32_t> related_list[2][kNumRelations];
  };

  const Comm& comm_;
  Span<Det<W>> dets_;
  long rank_;
  long num_ranks_;
  Det<W> group_a_;
  bool transfers_ = false;
  RootState root_;
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
