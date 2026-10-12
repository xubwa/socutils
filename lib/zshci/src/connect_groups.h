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
// Sectors. With N electrons in all, an A string of n electrons only ever
// pairs with B strings of N - n electrons: the determinants fall into
// sectors by n. Everything here is organized by them. The strings of a
// group are numbered within their electron count; a relation that changes
// the count by d leads from the strings of count n to those of count n + d,
// and its lists hold numbers of that one count; the determinants of a
// string are listed with the numbers of their other strings, all of one
// count. So from a determinant of sector n every line of the table above
// points to one definite sector - n itself, or n -+ 1, or n -+ 2 - and to
// the tables of that sector alone. The determinants themselves are not
// reordered: they, and a Hamiltonian stored for them, stay as they were
// when determinants are added.
//
// Related strings are found in one of two ways.
//
// By keys: two strings are related when they share a string of one or two
// electrons less, so each string contributes such shortened strings as keys
// and equal keys, brought together by sorting, give the related strings.
// A string of n electrons has n (n + 1) / 2 keys, which the root must keep
// to extend the tables later; with many electrons they outweigh the tables.
//
// By sub-groups (the default): the scheme is applied once more. Each group
// is divided into two sub-groups (its spinors in turn; the two need not be
// of equal size), a string of the group is a string of each, and the
// sub-strings are far fewer and half as long. Electrons leaving and
// entering a string do so in one sub-group or the other, so every relation
// between strings follows from the relations between their sub-strings:
// "2 out, 1 in", say, is "2 out" in one and "1 in" in the other, or "1 out,
// 1 in" in one and "1 out" in the other, and so on. For a new string, the
// related sub-strings are combined and each combination looked up among
// the strings there are. The relations between sub-strings are found by
// keys; those are few and short, are made afresh at every extension, and
// nothing but the numbering of the strings is kept.
//
// Doubles of one string with the other string unchanged are found by
// comparing the determinants that share the other string.

#ifndef ZSHCI_SRC_CONNECT_GROUPS_H_
#define ZSHCI_SRC_CONNECT_GROUPS_H_

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
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
  // group B. `subgroups` chooses how related strings are found (see the top
  // of this file); the tables are the same either way. They are empty until
  // Extend is called.
  GroupStringTables(const Det<W>& group_a, const Comm& comm,
                    bool subgroups = true)
      : comm_(comm),
        rank_(comm.rank()),
        num_ranks_(comm.size()),
        group_a_(group_a),
        subgroups_(subgroups) {}

  // The tables of `dets`. Collective.
  GroupStringTables(Span<Det<W>> dets, const Det<W>& group_a, const Comm& comm,
                    bool subgroups = true)
      : GroupStringTables(group_a, comm, subgroups) {
    Extend(dets, 0);
  }

  // Makes the tables those of `dets`, of which the determinants before
  // first_new are the ones of the previous call, unchanged. Only what the
  // new determinants bring is worked out: strings not seen before get the
  // next numbers of their electron count, their keys are merged into the
  // sorted keys kept from before, and the relation lists gain the pairs
  // that involve a new string. Collective: the root does this and the
  // tables are held once per node.
  void Extend(Span<Det<W>> dets, size_t first_new) {
    if (dets.size() > static_cast<size_t>(UINT32_MAX)) {
      throw std::length_error("too many determinants for a 32-bit index");
    }
    if (first_new > 0 && comm_.is_root() && root_.strings[0].empty()) {
      throw std::logic_error(
          "the string tables can only be extended while "
          "what the root keeps for that is there");
    }
    dets_ = dets;
    num_elec_ = dets.empty() ? 0 : dets[0].Count();
    std::vector<uint64_t> det_start[2];
    std::vector<uint32_t> det_other[2];
    std::vector<uint32_t> det_index[2];
    std::vector<uint64_t> related_start[2][kNumRelations];
    std::vector<uint32_t> related_list[2][kNumRelations];
    if (comm_.is_root()) {
      if (first_new == 0) root_ = RootState();
      std::vector<size_t> num_old[2];
      for (int group = 0; group < 2; ++group) {
        num_old[group] = NumberStrings(group, first_new);
      }
      // Both groups get an entry for every count up to the number of
      // electrons, so that a count and its complement can both be looked up.
      for (int group = 0; group < 2; ++group) {
        root_.strings[group].resize(num_elec_ + 1);
        root_.sorted[group].resize(num_elec_ + 1);
        num_old[group].resize(num_elec_ + 1, 0);
      }
      // Where the strings of each count start among all strings of a group;
      // the lists held so far are laid out by the previous places.
      const std::vector<uint64_t> old_base[2] = {
          first_new == 0 ? std::vector<uint64_t>() : base_[0],
          first_new == 0 ? std::vector<uint64_t>() : base_[1]};
      for (int group = 0; group < 2; ++group) {
        base_[group].assign(1, 0);
        for (const std::vector<Det<W>>& strings : root_.strings[group]) {
          base_[group].push_back(base_[group].back() + strings.size());
        }
      }
      sector_sizes_.assign(num_elec_ + 1, 0);
      for (uint16_t in_a : root_.count_a_of_det) ++sector_sizes_[in_a];
      for (int group = 0; group < 2; ++group) {
        GroupDets(group, &det_start[group], &det_other[group],
                  &det_index[group]);
        UpdateRelated(group, num_old[group], old_base[group],
                      related_start[group], related_list[group]);
      }
    }
    comm_.Bcast(&base_[0]);
    comm_.Bcast(&base_[1]);
    comm_.Bcast(&sector_sizes_);
    count_a_of_det_ =
        SharedArray<uint16_t>::FromRoot(root_.count_a_of_det, comm_);
    transfers_ = false;
    max_b_strings_ = 0;
    for (size_t c = 0; c + 1 < base_[1].size(); ++c) {
      max_b_strings_ =
          std::max<size_t>(max_b_strings_, base_[1][c + 1] - base_[1][c]);
    }
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
        related_start_[group][r] =
            SharedArray<uint64_t>::FromRoot(related_start[group][r], comm_);
        related_list_[group][r] =
            SharedArray<uint32_t>::FromRoot(related_list[group][r], comm_);
        if (r > 0 && related_list_[group][r].size() > 0) transfers_ = true;
      }
    }
  }

  // Releases what the root keeps to extend the tables (the keys above
  // all). The tables stay usable; the next Extend must start afresh, with
  // first_new = 0.
  void DropExtensionState() { root_ = RootState(); }

  // Number of distinct strings of a group, over all electron counts.
  size_t num_strings(int group) const {
    return base_[group].empty() ? 0 : base_[group].back();
  }
  // Number of entries in all relation lists.
  size_t num_links() const {
    size_t n = 0;
    for (int group = 0; group < 2; ++group) {
      for (int r = 0; r < kNumRelations; ++r) {
        n += related_list_[group][r].size();
      }
    }
    return n;
  }
  // Whether determinants differ in the number of electrons of a group.
  bool has_transfers() const { return transfers_; }
  // The sectors: element n is the number of determinants with n electrons in
  // group A (and the rest in group B). Determinants are only connected
  // within a sector and to the sectors one or two electrons away.
  const std::vector<uint64_t>& sector_sizes() const { return sector_sizes_; }

  // Memory of the tables, in bytes. `shared`: what every node holds once
  // and the connection search reads. `kept`: what the root keeps in
  // addition, to extend the tables later, of which `keys` are the sorted
  // keys of all strings. `transient`: the largest amount the root held
  // briefly during a call of Extend (new keys and new pairs).
  struct Memory {
    size_t shared = 0;
    size_t kept = 0;
    size_t keys = 0;
    size_t transient = 0;
  };
  Memory memory() const {
    Memory m;
    m.shared = count_a_of_det_.size() * sizeof(uint16_t);
    for (int group = 0; group < 2; ++group) {
      m.shared += string_of_det_[group].size() * sizeof(uint32_t) +
                  det_start_[group].size() * sizeof(uint64_t) +
                  det_other_[group].size() * sizeof(uint32_t) +
                  det_index_[group].size() * sizeof(uint32_t);
      for (int r = 0; r < kNumRelations; ++r) {
        m.shared += related_start_[group][r].size() * sizeof(uint64_t) +
                    related_list_[group][r].size() * sizeof(uint32_t);
      }
      m.kept += root_.string_of_det[group].size() * sizeof(uint32_t);
      for (const auto& part : root_.strings[group]) {
        m.kept += part.size() * sizeof(Det<W>);
      }
      for (const auto& part : root_.sorted[group]) {
        m.kept += part.size() * sizeof(std::pair<Det<W>, uint32_t>);
      }
      for (const auto& part : root_.keys[group]) {
        m.keys += part.size() * sizeof(Shortened);
      }
    }
    m.kept += root_.count_a_of_det.size() * sizeof(uint16_t) + m.keys;
    m.transient = max_transient_bytes_;
    return m;
  }

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
    // The sector of j: the electrons of its A and of its B string, and the
    // places of the two strings among all strings of their groups.
    const int count[2] = {count_a_of_det_[j], num_elec_ - count_a_of_det_[j]};
    const uint64_t slot[2] = {base_[0][count[0]] + string_of_det_[0][j],
                              base_[1][count[1]] + string_of_det_[1][j]};
    uint64_t tested = 0;
    for (int group = 0; group < 2; ++group) {
      const int other = 1 - group;
      // The determinants that contain the partner string, by the number of
      // their string of this group: all in the sector of j.
      const uint32_t* const with_partner =
          det_other_[other].data() + det_start_[other][slot[other]];
      const uint32_t* const with_partner_end =
          det_other_[other].data() + det_start_[other][slot[other] + 1];
      const uint32_t* const with_partner_dets =
          det_index_[other].data() + det_start_[other][slot[other]];

      // A single excitation of this string, the partner unchanged.
      const uint32_t* const singles_end =
          RelatedEnd(group, kSingle, slot[group]);
      for (const uint32_t* s = RelatedBegin(group, kSingle, slot[group]);
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
    // read from A to B and from B to A; each leads to one sector.
    Combine(limit, count[0], slot, kSingle, kSingle, visit);
    if (transfers_) {
      Combine(limit, count[0], slot, kMinus1, kPlus1, visit);
      Combine(limit, count[0], slot, kPlus1, kMinus1, visit);
      Combine(limit, count[0], slot, kMinus1, kPlus12, visit);
      Combine(limit, count[0], slot, kPlus12, kMinus1, visit);
      Combine(limit, count[0], slot, kMinus21, kPlus1, visit);
      Combine(limit, count[0], slot, kPlus1, kMinus21, visit);
      Combine(limit, count[0], slot, kMinus2, kPlus2, visit);
      Combine(limit, count[0], slot, kPlus2, kMinus2, visit);
    }
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

  // By how much a relation changes the number of electrons of a string.
  static int Shift(int relation) {
    static const int kShift[kNumRelations] = {0, -1, 1, -1, 1, -2, 2};
    return kShift[relation];
  }

  // The strings related to the string at `slot` (its place among all
  // strings of the group), as numbers within the electron count the
  // relation leads to.
  const uint32_t* RelatedBegin(int group, int relation, uint64_t slot) const {
    const SharedArray<uint64_t>& start = related_start_[group][relation];
    if (start.size() == 0) return nullptr;
    return related_list_[group][relation].data() + start[slot];
  }
  const uint32_t* RelatedEnd(int group, int relation, uint64_t slot) const {
    const SharedArray<uint64_t>& start = related_start_[group][relation];
    if (start.size() == 0) return nullptr;
    return related_list_[group][relation].data() + start[slot + 1];
  }

  // Reports the determinants below `limit` whose A string is related to
  // that of the determinant at slot[] by relation_a and whose B string by
  // relation_b. They lie in the sector with count_a + Shift(relation_a)
  // electrons in A. For every related A string the determinants that
  // contain it are known with their B strings, and those whose B string is
  // in the list of b are wanted. That list is marked once in a bit array of
  // this thread, so that each determinant costs one bit test; only where an
  // A string occurs in very many more determinants than b has related
  // strings, those are looked up in its sorted determinants instead.
  template <class Visitor>
  void Combine(uint32_t limit, int count_a, const uint64_t* slot,
               int relation_a, int relation_b, const Visitor& visit) const {
    const uint32_t* const b_begin = RelatedBegin(1, relation_b, slot[1]);
    const uint32_t* const b_end = RelatedEnd(1, relation_b, slot[1]);
    if (b_begin == b_end) return;
    const uint32_t* const a_begin = RelatedBegin(0, relation_a, slot[0]);
    const uint32_t* const a_end = RelatedEnd(0, relation_a, slot[0]);
    if (a_begin == a_end) return;
    // Where the A strings of the target sector start.
    const uint64_t target_base = base_[0][count_a + Shift(relation_a)];
    const size_t num_b = b_end - b_begin;
    std::vector<uint64_t>& marks = Marks();
    bool marked = false;
    for (const uint32_t* s = a_begin; s != a_end; ++s) {
      const uint64_t target = target_base + *s;
      const uint32_t* const with_begin =
          det_other_[0].data() + det_start_[0][target];
      const uint32_t* const with_end =
          det_other_[0].data() + det_start_[0][target + 1];
      const uint32_t* const with_dets =
          det_index_[0].data() + det_start_[0][target];
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

  // A bit per B string of one electron count, all zero between uses; one
  // array per thread.
  std::vector<uint64_t>& Marks() const {
    static thread_local std::vector<uint64_t> marks;
    const size_t needed = max_b_strings_ / 64 + 1;
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

  // Numbers the strings of the determinants from first_new on within their
  // electron count: known strings keep their number, new ones get the next
  // numbers. Returns, per electron count, the number of strings there were
  // before.
  std::vector<size_t> NumberStrings(int group, size_t first_new) {
    std::vector<std::vector<Det<W>>>& strings = root_.strings[group];
    std::vector<std::vector<std::pair<Det<W>, uint32_t>>>& sorted =
        root_.sorted[group];
    std::vector<uint32_t>& string_of_det = root_.string_of_det[group];
    const size_t n = dets_.size();
    const long begin = static_cast<long>(first_new);
    const long end = static_cast<long>(n);
    string_of_det.resize(n);
    if (group == 0) root_.count_a_of_det.resize(n);
    std::vector<Det<W>> of_det(n - first_new);
    std::vector<uint16_t> count_of_det(n - first_new);
    int max_count = static_cast<int>(strings.size()) - 1;
#pragma omp parallel for schedule(static) reduction(max : max_count)
    for (long i = begin; i < end; ++i) {
      of_det[i - begin] = StringOf(dets_[i], group);
      count_of_det[i - begin] =
          static_cast<uint16_t>(of_det[i - begin].Count());
      max_count = std::max<int>(max_count, count_of_det[i - begin]);
    }
    strings.resize(max_count + 1);
    sorted.resize(max_count + 1);
    std::vector<size_t> num_old(max_count + 1);
    for (int c = 0; c <= max_count; ++c) num_old[c] = strings[c].size();

    auto by_string = [](const std::pair<Det<W>, uint32_t>& entry,
                        const Det<W>& string) { return entry.first < string; };
#pragma omp parallel for schedule(static)
    for (long i = begin; i < end; ++i) {
      const Det<W>& string = of_det[i - begin];
      const std::vector<std::pair<Det<W>, uint32_t>>& known =
          sorted[count_of_det[i - begin]];
      const auto found =
          std::lower_bound(known.begin(), known.end(), string, by_string);
      string_of_det[i] = found != known.end() && found->first == string
                             ? found->second
                             : UINT32_MAX;
      if (group == 0) root_.count_a_of_det[i] = count_of_det[i - begin];
    }
    std::vector<std::vector<Det<W>>> fresh(max_count + 1);
    for (long i = begin; i < end; ++i) {
      if (string_of_det[i] == UINT32_MAX) {
        fresh[count_of_det[i - begin]].push_back(of_det[i - begin]);
      }
    }
    for (std::vector<Det<W>>& list : fresh) {
      std::sort(list.begin(), list.end());
      list.erase(std::unique(list.begin(), list.end()), list.end());
    }
#pragma omp parallel for schedule(static)
    for (long i = begin; i < end; ++i) {
      if (string_of_det[i] != UINT32_MAX) continue;
      const int c = count_of_det[i - begin];
      string_of_det[i] = static_cast<uint32_t>(
          num_old[c] + (std::lower_bound(fresh[c].begin(), fresh[c].end(),
                                         of_det[i - begin]) -
                        fresh[c].begin()));
    }
    for (int c = 0; c <= max_count; ++c) {
      const size_t middle = sorted[c].size();
      for (size_t k = 0; k < fresh[c].size(); ++k) {
        strings[c].push_back(fresh[c][k]);
        sorted[c].emplace_back(fresh[c][k],
                               static_cast<uint32_t>(num_old[c] + k));
      }
      std::inplace_merge(sorted[c].begin(), sorted[c].begin() + middle,
                         sorted[c].end());
    }
    return num_old;
  }

  // For every string of a group, at its place among all strings of the
  // group, the determinants that contain it: the number of their other
  // string within its electron count, increasing, and their index.
  void GroupDets(int group, std::vector<uint64_t>* start,
                 std::vector<uint32_t>* other,
                 std::vector<uint32_t>* index) const {
    const std::vector<uint32_t>& mine = root_.string_of_det[group];
    const std::vector<uint32_t>& theirs = root_.string_of_det[1 - group];
    const size_t n = mine.size();
    const size_t num_slots = base_[group].back();
    auto slot_of = [&](size_t i) {
      const int in_a = root_.count_a_of_det[i];
      return base_[group][group == 0 ? in_a : num_elec_ - in_a] + mine[i];
    };
    start->assign(num_slots + 1, 0);
    for (size_t i = 0; i < n; ++i) ++(*start)[slot_of(i) + 1];
    for (size_t s = 0; s < num_slots; ++s) (*start)[s + 1] += (*start)[s];
    std::vector<uint64_t> cursor(start->begin(), start->end() - 1);
    std::vector<std::pair<uint32_t, uint32_t>> items(n);
    for (size_t i = 0; i < n; ++i) {
      items[cursor[slot_of(i)]++] = {theirs[i], static_cast<uint32_t>(i)};
    }
    const long count = static_cast<long>(num_slots);
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

  // A string with `removed` of its electrons taken out, as a key. `string`
  // is the number of the string within its electron count, which is that of
  // the key plus `removed`.
  struct Shortened {
    Det<W> key;
    uint32_t string;
    uint8_t removed;
  };

  // A pair of related strings found from a shared key: `from`, of
  // `from_count` electrons, and `to`, of the count the relation leads to.
  struct Link {
    uint32_t from;
    uint32_t to;
    uint8_t relation;
    uint8_t from_count;
  };

  static bool KeyLess(const Shortened& x, const Shortened& y) {
    if (x.key != y.key) return x.key < y.key;
    if (x.removed != y.removed) return x.removed < y.removed;
    return x.string < y.string;
  }

  // The pairs of related strings in which at least one is new, i.e. has a
  // number from num_old[c] on in its electron count c, found from keys.
  // Strings s and t are related when they share a key: s with one electron
  // out and t with one out is a single; s with one out and t itself is "1
  // out"; s with two out and t with one out is "2 out, 1 in" unless t is
  // contained in s; s with two out and t itself is "2 out". A key of k
  // electrons thus relates strings of k, k + 1 and k + 2 electrons: three
  // neighbouring counts, and nothing else. Keys with electrons taken out of
  // the longer string are only made when shorter strings exist at all
  // (`transfers`, and counts above `min_count`).
  //
  // `kept_keys` holds the sorted keys of the strings before the new ones, in
  // parts by hash (as many as it has entries), and gains those of the new
  // ones. Returns the bytes of the new keys.
  static size_t LinksFromKeys(const std::vector<std::vector<Det<W>>>& strings,
                              const std::vector<size_t>& num_old,
                              bool transfers, int min_count,
                              std::vector<std::vector<Shortened>>* kept_keys,
                              std::vector<std::vector<Link>>* links_out) {
    const int num_counts = static_cast<int>(strings.size());
    std::vector<std::vector<Shortened>>& kept = *kept_keys;
    const size_t num_parts = kept.size();
    std::vector<std::vector<Link>>& links = *links_out;
    // The keys of the new strings, made by all threads and dealt into parts
    // by their hash, so that equal keys meet in one part and the parts can
    // be handled side by side.
    std::vector<std::vector<Shortened>> chunks(omp_get_max_threads());
#pragma omp parallel
    {
      std::vector<Shortened>& mine = chunks[omp_get_thread_num()];
      int occ[64 * W];
      for (int c = 0; c < num_counts; ++c) {
        const long count = static_cast<long>(strings[c].size());
#pragma omp for schedule(dynamic, 256) nowait
        for (long s = static_cast<long>(num_old[c]); s < count; ++s) {
          int num_occ = 0;
          for (int k = 0; k < W; ++k) {
            uint64_t word = strings[c][s].words[k];
            while (word != 0) {
              occ[num_occ++] = 64 * k + __builtin_ctzll(word);
              word &= word - 1;
            }
          }
          const uint32_t id = static_cast<uint32_t>(s);
          if (transfers) mine.push_back({strings[c][s], id, 0});
          for (int m = 0; m < num_occ; ++m) {
            Shortened one{strings[c][s], id, 1};
            one.key.Clear(occ[m]);
            mine.push_back(one);
            // Two out only when a shorter string could match.
            if (!transfers || c <= min_count) continue;
            for (int l = 0; l < m; ++l) {
              Shortened two{one.key, id, 2};
              two.key.Clear(occ[l]);
              mine.push_back(two);
            }
          }
        }
      }
    }
    Partitioned<Shortened> fresh = PartitionChunks(
        &chunks, num_parts,
        [&](const Shortened& item) { return item.key.Hash() % num_parts; });

    // Every run of equal keys that holds a new key gives the pairs in which
    // at least one string is new. The new keys then join the kept ones.
    links.assign(omp_get_max_threads(), std::vector<Link>());
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
          // The electron count of a member is that of the key plus what was
          // removed from it.
          const int key_count = next->key.Count();
          auto add = [&](int relation, const Member& from, const Member& to) {
            if (!from.is_new && !to.is_new) return;
            mine.push_back({from.string, to.string,
                            static_cast<uint8_t>(relation),
                            static_cast<uint8_t>(key_count + from.removed)});
          };
          // Ordered by the number of electrons removed: 0 up to first[1],
          // 1 up to first[2], 2 up to first[3].
          size_t first[4] = {0, 0, 0, run.size()};
          while (first[1] < run.size() && run[first[1]].removed < 1) {
            ++first[1];
          }
          first[2] = first[1];
          while (first[2] < run.size() && run[first[2]].removed < 2) {
            ++first[2];
          }
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
                if (Contains(strings[key_count + 2][run[x].string],
                             strings[key_count + 1][run[t].string])) {
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

    return fresh.items.size() * sizeof(Shortened);
  }

  // Electrons that leave and enter a string under a relation.
  static int Out(int relation) {
    static const int kOut[kNumRelations] = {1, 1, 0, 2, 1, 2, 0};
    return kOut[relation];
  }
  static int In(int relation) {
    static const int kIn[kNumRelations] = {1, 0, 1, 1, 2, 0, 2};
    return kIn[relation];
  }
  // The relation with `out` electrons leaving and `in` entering, for the
  // combinations that have one; kNumRelations stands for "the same string".
  static int RelationOf(int out, int in) {
    static const int kOf[3][3] = {{kNumRelations, kPlus1, kPlus2},
                                  {kMinus1, kSingle, kPlus12},
                                  {kMinus2, kMinus21, -1}};
    return kOf[out][in];
  }
  // The relation of t to s when that of s to t is `relation`.
  static int Inverse(int relation) {
    static const int kInverse[kNumRelations] = {
        kSingle, kPlus1, kMinus1, kPlus12, kMinus21, kPlus2, kMinus2};
    return kInverse[relation];
  }

  // The distinct strings of a sub-group and how they are related.
  struct SubStrings {
    // By electron count, sorted; a string's number is its place there.
    std::vector<std::vector<Det<W>>> strings;
    // base[c]: place of the first string of c electrons among all.
    std::vector<uint64_t> base;
    // The strings related by r to the one at place s are
    // list[r][start[r][s] .. start[r][s + 1]), increasing numbers within
    // the count the relation leads to. No arrays for an empty relation.
    std::vector<uint64_t> start[kNumRelations];
    std::vector<uint32_t> list[kNumRelations];

    uint64_t Place(const Det<W>& string) const {
      const int c = string.Count();
      return base[c] +
             (std::lower_bound(strings[c].begin(), strings[c].end(), string) -
              strings[c].begin());
    }
    size_t bytes() const {
      size_t n = base.size() * sizeof(uint64_t);
      for (const std::vector<Det<W>>& part : strings) {
        n += part.size() * sizeof(Det<W>);
      }
      for (int r = 0; r < kNumRelations; ++r) {
        n += start[r].size() * sizeof(uint64_t) +
             list[r].size() * sizeof(uint32_t);
      }
      return n;
    }
  };

  // The two sub-groups of a group: its spinors in turn, the first, third,
  // ... in one and the others in the other (one more in the first if their
  // number is odd).
  void SubgroupMasks(int group, Det<W>* masks) const {
    masks[0] = Det<W>();
    masks[1] = Det<W>();
    int position = 0;
    for (int p = 0; p < 64 * W; ++p) {
      if (group_a_.Occupied(p) != (group == 0)) continue;
      masks[position % 2].Set(p);
      ++position;
    }
  }

  static Det<W> Masked(const Det<W>& string, const Det<W>& mask) {
    Det<W> out;
    for (int k = 0; k < W; ++k) out.words[k] = string.words[k] & mask.words[k];
    return out;
  }

  // The distinct sub-strings that the strings of a group have in `mask`,
  // with their relation lists. Sets *work_bytes to the bytes of the keys
  // that were made for it.
  static SubStrings MakeSubStrings(
      const std::vector<std::vector<Det<W>>>& strings, const Det<W>& mask,
      size_t* work_bytes) {
    std::vector<Det<W>> all;
    for (const std::vector<Det<W>>& part : strings) {
      for (const Det<W>& string : part) all.push_back(Masked(string, mask));
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    SubStrings sub;
    for (const Det<W>& string : all) {
      const size_t c = string.Count();
      if (sub.strings.size() <= c) sub.strings.resize(c + 1);
      sub.strings[c].push_back(string);
    }
    const int num_counts = static_cast<int>(sub.strings.size());
    int min_count = num_counts;
    int max_count = -1;
    sub.base.assign(1, 0);
    for (int c = 0; c < num_counts; ++c) {
      sub.base.push_back(sub.base.back() + sub.strings[c].size());
      if (sub.strings[c].empty()) continue;
      min_count = std::min(min_count, c);
      max_count = std::max(max_count, c);
    }
    std::vector<std::vector<Shortened>> keys(
        8 * static_cast<size_t>(omp_get_max_threads()));
    std::vector<std::vector<Link>> links;
    *work_bytes =
        LinksFromKeys(sub.strings, std::vector<size_t>(num_counts, 0),
                      max_count > min_count, min_count, &keys, &links);
    const size_t num_places = sub.base.back();
    for (int r = 0; r < kNumRelations; ++r) {
      std::vector<uint64_t>& start = sub.start[r];
      start.assign(num_places + 1, 0);
      for (const std::vector<Link>& chunk : links) {
        for (const Link& link : chunk) {
          if (link.relation == r) {
            ++start[sub.base[link.from_count] + link.from + 1];
          }
        }
      }
      for (size_t s = 0; s < num_places; ++s) start[s + 1] += start[s];
      if (start[num_places] == 0) {
        start.clear();
        continue;
      }
      std::vector<uint32_t>& list = sub.list[r];
      list.resize(start[num_places]);
      std::vector<uint64_t> cursor(start.begin(), start.end() - 1);
      for (const std::vector<Link>& chunk : links) {
        for (const Link& link : chunk) {
          if (link.relation == r) {
            list[cursor[sub.base[link.from_count] + link.from]++] = link.to;
          }
        }
      }
      const long count = static_cast<long>(num_places);
#pragma omp parallel for schedule(dynamic, 256)
      for (long s = 0; s < count; ++s) {
        std::sort(list.begin() + start[s], list.begin() + start[s + 1]);
      }
    }
    return sub;
  }

  // The pairs of related strings of a group in which at least one is new,
  // i.e. has a number from num_old[c] on in its electron count c, found
  // from the sub-strings (see the top of this file). Without `transfers`
  // only the singles. Returns the bytes this took.
  size_t LinksFromSubgroups(int group, const std::vector<size_t>& num_old,
                            bool transfers,
                            std::vector<std::vector<Link>>* links_out) const {
    const std::vector<std::vector<Det<W>>>& strings = root_.strings[group];
    const std::vector<uint64_t>& base = base_[group];
    const int num_counts = static_cast<int>(strings.size());
    Det<W> masks[2];
    SubgroupMasks(group, masks);
    size_t key_bytes[2] = {0, 0};
    const SubStrings sub[2] = {
        MakeSubStrings(strings, masks[0], &key_bytes[0]),
        MakeSubStrings(strings, masks[1], &key_bytes[1])};

    // Every string as a pair of sub-strings, and the strings there are,
    // ordered by the place of their first sub-string and then of their
    // second.
    struct Entry {
      uint64_t first;   // place of the sub-string in sub-group 0
      uint64_t second;  // place of the sub-string in sub-group 1
      uint32_t number;  // of the string within its electron count
      uint16_t count;
    };
    const size_t num_strings = base.back();
    std::vector<Entry> of_place(num_strings);
    for (int c = 0; c < num_counts; ++c) {
      const long count = static_cast<long>(strings[c].size());
#pragma omp parallel for schedule(static)
      for (long s = 0; s < count; ++s) {
        of_place[base[c] + s] = {sub[0].Place(Masked(strings[c][s], masks[0])),
                                 sub[1].Place(Masked(strings[c][s], masks[1])),
                                 static_cast<uint32_t>(s),
                                 static_cast<uint16_t>(c)};
      }
    }
    std::vector<Entry> ordered = of_place;
    std::sort(
        ordered.begin(), ordered.end(), [](const Entry& x, const Entry& y) {
          return x.first != y.first ? x.first < y.first : x.second < y.second;
        });
    std::vector<uint64_t> first_start(sub[0].base.back() + 1, 0);
    for (const Entry& entry : ordered) ++first_start[entry.first + 1];
    for (size_t u = 0; u + 1 < first_start.size(); ++u) {
      first_start[u + 1] += first_start[u];
    }

    // The sub-strings related to the one at `place` by `relation`, as
    // places; for kNumRelations the sub-string itself.
    auto related = [&](int h, int relation, uint64_t place, int count,
                       std::vector<uint64_t>* out) {
      out->clear();
      if (relation == kNumRelations) {
        out->push_back(place);
        return;
      }
      if (sub[h].start[relation].empty()) return;
      const int target = count + Shift(relation);
      for (uint64_t k = sub[h].start[relation][place];
           k < sub[h].start[relation][place + 1]; ++k) {
        out->push_back(sub[h].base[target] + sub[h].list[relation][k]);
      }
    };

    const int num_lists = transfers ? kNumRelations : 1;
    std::vector<std::vector<Link>>& links = *links_out;
    links.assign(omp_get_max_threads(), std::vector<Link>());
#pragma omp parallel
    {
      std::vector<Link>& mine = links[omp_get_thread_num()];
      std::vector<uint64_t> firsts, seconds;
      auto by_second = [](const Entry& entry, uint64_t second) {
        return entry.second < second;
      };
      for (int c = 0; c < num_counts; ++c) {
        const long count = static_cast<long>(strings[c].size());
#pragma omp for schedule(dynamic, 64) nowait
        for (long s = static_cast<long>(num_old[c]); s < count; ++s) {
          const Entry& self = of_place[base[c] + s];
          const int count_first = Masked(strings[c][s], masks[0]).Count();
          const int count_second = c - count_first;
          for (int r = 0; r < num_lists; ++r) {
            // How the electrons that leave and enter divide between the
            // sub-groups.
            for (int out = 0; out <= Out(r); ++out) {
              for (int in = 0; in <= In(r); ++in) {
                related(0, RelationOf(out, in), self.first, count_first,
                        &firsts);
                if (firsts.empty()) continue;
                related(1, RelationOf(Out(r) - out, In(r) - in), self.second,
                        count_second, &seconds);
                if (seconds.empty()) continue;
                for (uint64_t first : firsts) {
                  const Entry* const begin =
                      ordered.data() + first_start[first];
                  const Entry* const end =
                      ordered.data() + first_start[first + 1];
                  const Entry* from = begin;
                  for (uint64_t second : seconds) {
                    from = std::lower_bound(from, end, second, by_second);
                    if (from == end) break;
                    if (from->second != second) continue;
                    mine.push_back({self.number, from->number,
                                    static_cast<uint8_t>(r),
                                    static_cast<uint8_t>(c)});
                    if (from->number < num_old[from->count]) {
                      mine.push_back({from->number, self.number,
                                      static_cast<uint8_t>(Inverse(r)),
                                      static_cast<uint8_t>(from->count)});
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return key_bytes[0] + key_bytes[1] + sub[0].bytes() + sub[1].bytes() +
           (of_place.size() + ordered.size()) * sizeof(Entry) +
           first_start.size() * sizeof(uint64_t);
  }

  // Brings the relation lists of one group up to date with the strings from
  // number num_old[c] on in every electron count c: the pairs in which at
  // least one string is new are found (LinksFromSubgroups or LinksFromKeys)
  // and join the lists.
  //
  // The lists are written to start[r] and list[r] for every relation r, laid
  // out as the ranks hold them: the lists of all electron counts one after
  // the other, a string being found at its place among all strings of the
  // group. What was there is taken from the arrays held so far, which are
  // laid out by `old_base`. A relation without any entry gets no arrays.
  void UpdateRelated(int group, std::vector<size_t> num_old,
                     const std::vector<uint64_t>& old_base,
                     std::vector<uint64_t>* start,
                     std::vector<uint32_t>* list) {
    const std::vector<std::vector<Det<W>>>& strings = root_.strings[group];
    const int num_counts = static_cast<int>(strings.size());
    int min_count = num_counts;
    int max_count = -1;
    for (int c = 0; c < num_counts; ++c) {
      if (strings[c].empty()) continue;
      min_count = std::min(min_count, c);
      max_count = std::max(max_count, c);
    }
    const bool transfers = max_count > min_count;
    std::vector<std::vector<Shortened>>& kept = root_.keys[group];
    const size_t num_parts = 8 * static_cast<size_t>(omp_get_max_threads());
    // The kept keys depend on which strings have keys of two electrons
    // less; if that changes, they are made afresh. With sub-groups nothing
    // of the kind is kept.
    const bool afresh =
        !subgroups_ &&
        (kept.size() != num_parts || transfers != root_.transfers[group] ||
         (transfers && min_count != root_.min_count[group]));
    if (afresh) {
      kept.assign(num_parts, std::vector<Shortened>());
      num_old.assign(num_counts, 0);
    }
    root_.transfers[group] = transfers;
    root_.min_count[group] = min_count;

    std::vector<std::vector<Link>> links;
    size_t work_bytes = 0;
    if (subgroups_) {
      work_bytes = LinksFromSubgroups(group, num_old, transfers, &links);
    } else {
      work_bytes =
          LinksFromKeys(strings, num_old, transfers, min_count, &kept, &links);
    }

    {
      size_t transient = work_bytes;
      for (const std::vector<Link>& chunk : links) {
        transient += chunk.size() * sizeof(Link);
      }
      max_transient_bytes_ = std::max(max_transient_bytes_, transient);
    }

    // The lists: what was there, then the new pairs, sorted where a list
    // gained entries. Several threads may meet at one string.
    const std::vector<uint64_t>& base = base_[group];
    const size_t num_slots = base.back();
    const long num_chunks = static_cast<long>(links.size());
    const int num_lists = transfers ? kNumRelations : 1;
    // added[r][slot]: new entries of the string at `slot` for relation r.
    std::vector<uint64_t> added[kNumRelations];
    std::vector<uint64_t> cursor[kNumRelations];
    for (int r = 0; r < num_lists; ++r) added[r].assign(num_slots, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long k = 0; k < num_chunks; ++k) {
      for (const Link& link : links[k]) {
#pragma omp atomic
        ++added[link.relation][base[link.from_count] + link.from];
      }
    }
    // The place a string had before, or none.
    auto old_slot = [&](int c, size_t s, uint64_t* slot) {
      if (afresh || static_cast<size_t>(c) + 1 >= old_base.size()) return false;
      if (s >= old_base[c + 1] - old_base[c]) return false;
      *slot = old_base[c] + s;
      return true;
    };
    for (int r = 0; r < kNumRelations; ++r) {
      start[r].clear();
      list[r].clear();
      if (r >= num_lists) continue;
      const SharedArray<uint64_t>& old_start = related_start_[group][r];
      const SharedArray<uint32_t>& old_list = related_list_[group][r];
      const bool had_any = old_start.size() > 0;
      start[r].assign(num_slots + 1, 0);
      for (int c = 0; c < num_counts; ++c) {
        for (size_t s = 0; s < strings[c].size(); ++s) {
          const size_t slot = base[c] + s;
          uint64_t before = 0;
          uint64_t old;
          if (had_any && old_slot(c, s, &old)) {
            before = old_start[old + 1] - old_start[old];
          }
          start[r][slot + 1] = start[r][slot] + before + added[r][slot];
        }
      }
      if (start[r][num_slots] == 0 && r != kSingle) {
        start[r].clear();
        continue;
      }
      list[r].resize(start[r][num_slots]);
      cursor[r].assign(num_slots, 0);
#pragma omp parallel for schedule(dynamic, 1)
      for (int c = 0; c < num_counts; ++c) {
        for (size_t s = 0; s < strings[c].size(); ++s) {
          const size_t slot = base[c] + s;
          uint64_t before = 0;
          uint64_t old;
          if (had_any && old_slot(c, s, &old)) {
            before = old_start[old + 1] - old_start[old];
            std::copy(old_list.data() + old_start[old],
                      old_list.data() + old_start[old + 1],
                      list[r].begin() + start[r][slot]);
          }
          cursor[r][slot] = start[r][slot] + before;
        }
      }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (long k = 0; k < num_chunks; ++k) {
      for (const Link& link : links[k]) {
        uint64_t slot;
#pragma omp atomic capture
        slot = cursor[link.relation][base[link.from_count] + link.from]++;
        list[link.relation][slot] = link.to;
      }
    }
#pragma omp parallel
    {
      for (int r = 0; r < num_lists; ++r) {
        if (list[r].empty()) continue;
        const long count = static_cast<long>(num_slots);
#pragma omp for schedule(dynamic, 256) nowait
        for (long slot = 0; slot < count; ++slot) {
          if (added[r][slot] == 0) continue;
          std::sort(list[r].begin() + start[r][slot],
                    list[r].begin() + start[r][slot + 1]);
        }
      }
    }
  }

  static bool Contains(const Det<W>& outer, const Det<W>& inner) {
    for (int k = 0; k < W; ++k) {
      if ((inner.words[k] & ~outer.words[k]) != 0) return false;
    }
    return true;
  }

  // What the root keeps between calls of Extend. Where an index is an
  // electron count, strings are numbered within that count.
  struct RootState {
    std::vector<std::vector<Det<W>>> strings[2];  // by count, number
    // String and number, sorted by string, by count.
    std::vector<std::vector<std::pair<Det<W>, uint32_t>>> sorted[2];
    std::vector<uint32_t> string_of_det[2];
    std::vector<uint16_t> count_a_of_det;
    // The keys of all strings, in parts by hash, each part sorted.
    std::vector<std::vector<Shortened>> keys[2];
    // What the keys were made for.
    bool transfers[2] = {false, false};
    int min_count[2] = {0, 0};
  };

  const Comm& comm_;
  Span<Det<W>> dets_;
  long rank_;
  long num_ranks_;
  Det<W> group_a_;
  bool subgroups_ = true;
  int num_elec_ = 0;
  bool transfers_ = false;
  size_t max_b_strings_ = 0;
  size_t max_transient_bytes_ = 0;
  std::vector<uint64_t> sector_sizes_;
  RootState root_;
  // base_[g][c]: the place of the first string of c electrons among all
  // strings of group g; one more entry for the end.
  std::vector<uint64_t> base_[2];
  // For every determinant, the electrons in its A string and the numbers of
  // its two strings within their counts.
  SharedArray<uint16_t> count_a_of_det_;
  SharedArray<uint32_t> string_of_det_[2];
  // The determinants that contain the string at place s are
  // det_index_[det_start_[s] .. det_start_[s + 1]), ordered by the number
  // of their other string, which is in det_other_.
  SharedArray<uint64_t> det_start_[2];
  SharedArray<uint32_t> det_other_[2];
  SharedArray<uint32_t> det_index_[2];
  // The strings related by relation r to the string at place s are
  // related_list_[r][related_start_[r][s] .. related_start_[r][s + 1]),
  // increasing numbers within the electron count the relation leads to.
  // Empty arrays for relations that do not occur.
  SharedArray<uint64_t> related_start_[2][kNumRelations];
  SharedArray<uint32_t> related_list_[2][kNumRelations];
};

}  // namespace internal
}  // namespace zshci

#endif  // ZSHCI_SRC_CONNECT_GROUPS_H_
