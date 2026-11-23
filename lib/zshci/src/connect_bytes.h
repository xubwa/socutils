// Connected pairs of determinants from strings whose relations are not
// stored but composed, byte by byte, from one small static table.
//
// The scheme is that of connect_groups.h: the spinors are divided into two
// groups, A and B, a determinant is an A string and a B string, and two
// determinants are connected when their strings are related as the table at
// the top of that file says. There the strings related to each string are
// stored, about 800 numbers per string for spinors without a conserved spin
// projection, which is more than a kilobyte per determinant. Here nothing is
// stored per pair of strings.
//
// Bytes. The determinant stays the bit string it is. The spinors of a group
// that are occupied in some determinant, in increasing order, are read as
// the bits of a number of their own (the "compact string": bit k is the
// k-th of these spinors), and that number is read byte by byte, eight
// spinors of the group at a time, the last byte possibly partial with its
// unused bits zero. A byte is the sub-string of a sub-group of eight spinors
// and has 256 possible values, so everything about sub-strings is known
// beforehand:
//
//   changed[o][i][v], o, i = 0, 1, 2: the byte values that v turns into when
//   o of its electrons leave and i enter, as a set of 256 bits.
//
// That table (ByteTable, 72 kB) is made once and does not depend on the
// determinants. Electrons that leave and enter a string do so in one byte or
// in several, so a relation "o out, i in" between strings is: o_k out and i_k
// in at byte k, for every way of writing o and i as sums over the bytes.
//
// Look-up. Of the candidates composed this way only those matter that occur
// in some determinant, a few percent. They are not composed first and looked
// up afterwards: the distinct strings of a group are kept as a tree over
// their bytes, the highest byte at the top, and the composition walks down
// that tree. A node stands for the strings that share the bytes above it and
// holds the set of values the next byte takes among them, 256 bits, with the
// number of its first child; the children are numbered in the order of the
// byte value, so a child is found by counting bits. At a node the walk
// intersects the values that occur with those the byte may turn into - the
// union of changed[o][i][v] over the (o, i) still within what the wanted
// relations allow, given what the bytes above have spent - and goes on below
// each value found; a family of candidates none of which occurs is thus
// given up at its first byte. The leaves are the strings, numbered in
// increasing order of the compact string (their "places"), and the strings
// below a node have consecutive places.
//
// What is stored, once per node of the machine for all ranks on it, per
// group: the tree (40 bytes per node; at most one node per string and byte,
// far fewer near the top); where the determinants of each string start
// (8 bytes per string); and the determinants ordered by the place of their
// string of that group, with their indices (8 W + 4 bytes per determinant).
// The determinants of one string are in increasing order of their index.
//
// The connections of a determinant j with strings a and b, the places of
// which are found by going down the two trees:
//
//   same B string: the determinants of b are compared with j; those that
//     differ in four spinors are double excitations of a.
//   same A string: the determinants of a are compared with j; all are
//     singles or doubles of b or not connected.
//   both differ, or a single of a alone: the walk gives the A strings a'
//     that occur and are related to a by 1 out 1 in, 1 out, 1 in, 2 out 1
//     in, 1 out 2 in, 2 out or 2 in (only the first when no electron moves
//     between the groups); the determinants of each are compared with j and
//     are connected when they differ in at most four spinors. The B strings
//     need no relations at all: the comparison says whether the B string
//     fits.
//
// The rows of a stored Hamiltonian are made for the determinants of an A
// string together (ForEachConnectedOfChunk): one walk for all of them, and
// each related string read once and compared with all of them while it is
// in the cache, instead of once per determinant.
//
// Going string by string for all connections (ForEachConnectedOfTask, for a
// Hamiltonian that is not stored), the walk is also restricted to the A
// strings at lower places, which the order of the tree allows at no cost:
// at the first byte that changes only lower values are taken.
//
// Extension. Nothing here depends on the order in which determinants were
// added, and nothing is kept on the root between calls: when determinants
// are added the strings are sorted again and the trees and the ordered
// copies are made anew, which takes a small fraction of the time the new
// rows of the Hamiltonian take. The determinants themselves and a stored
// Hamiltonian stay as they are.
//
// Ranks. The root makes everything; trees, starts and ordered determinants
// are arrays shared by the ranks of a node and copied to the other nodes.
// A rank searches for the rows, or the tasks, it owns and writes nothing.
//
// Symmetry labels are not used: a string related to a by a change that the
// labels of the B strings cannot compensate simply has no determinant that
// passes the comparison.

#ifndef ZSHCI_SRC_CONNECT_BYTES_H_
#define ZSHCI_SRC_CONNECT_BYTES_H_

#include <omp.h>

#if defined(__BMI2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "comm.h"
#include "det.h"
#include "parallel.h"

// The searches are compiled as functions of their own (see FindClose), and
// apart from the search by stored lists that their callers can take instead.
#define ZSHCI_NOINLINE __attribute__((noinline))

namespace zshci {
namespace internal {

// What a byte, eight spinors of one group, turns into when electrons leave
// and enter it.
struct ByteTable {
  // changed[o][i][v]: bit u of the 256 is set when u has all electrons of v
  // but o of them, and i more.
  uint64_t changed[3][3][256][4];

  ByteTable() : changed() {
    for (int v = 0; v < 256; ++v) {
      for (int u = 0; u < 256; ++u) {
        const int out = __builtin_popcount(v & ~u);
        const int in = __builtin_popcount(u & ~v);
        if (out <= 2 && in <= 2)
          changed[out][in][v][u >> 6] |= 1ULL << (u & 63);
      }
    }
  }

  static const ByteTable& Get() {
    static const ByteTable table;
    return table;
  }
};

template <int W>
class ByteStringTables {
 public:
  // `group_a` has the bits of the spinors of group A set; all others form
  // group B. The tables are empty until Extend is called.
  ByteStringTables(const Det<W>& group_a, const Comm& comm)
      : comm_(comm),
        rank_(comm.rank()),
        num_ranks_(comm.size()),
        group_a_(group_a),
        table_(ByteTable::Get()) {}

  // Makes the tables those of `dets`. Collective: the root does this and
  // the tables are held once per node.
  void Extend(Span<Det<W>> dets) {
    if (dets.size() > static_cast<size_t>(UINT32_MAX)) {
      throw std::length_error("too many determinants for a 32-bit index");
    }
    dets_ = dets;
    const size_t n = dets.size();
    SetGroups();
    for (int group = 0; group < 2; ++group) {
      Tree& tree = trees_[group];
      Partitioned<Item> sorted;
      std::vector<uint64_t> det_start;
      std::vector<uint64_t> nodes;
      if (comm_.is_root()) {
        sorted = SortByString(group);
        det_start = StringStarts(sorted);
        nodes = MakeTree(sorted, det_start, &tree);
        max_transient_bytes_ =
            std::max(max_transient_bytes_, 2 * n * sizeof(Item));
      }
      comm_.Bcast(&tree.level_start);
      det_index_[group] = SharedArray<uint32_t>(n, comm_);
      block_dets_[group] = SharedArray<Det<W>>(n, comm_);
      if (comm_.is_root()) {
        uint32_t* const index = det_index_[group].data();
        Det<W>* const ordered = block_dets_[group].data();
        const long count = static_cast<long>(n);
#pragma omp parallel for schedule(static)
        for (long k = 0; k < count; ++k) {
          index[k] = sorted.items[k].index;
          ordered[k] = dets[index[k]];
        }
      }
      det_index_[group].ShareFromRoot(comm_);
      block_dets_[group].ShareFromRoot(comm_);
      det_start_[group] = SharedArray<uint64_t>::FromRoot(det_start, comm_);
      tree.nodes = SharedArray<uint64_t>::FromRoot(nodes, comm_);
    }
    comm_.Bcast(&sector_sizes_);
    // Electrons move between the groups when the determinants differ in the
    // number they have in group A; only then are relations that change the
    // number of electrons of a string looked for.
    int num_sectors = 0;
    for (uint64_t size : sector_sizes_) num_sectors += size > 0 ? 1 : 0;
    transfers_ = num_sectors > 1;
    wanted_ = 1U << State(1, 1);
    if (transfers_) {
      wanted_ |= 1U << State(1, 0) | 1U << State(0, 1) | 1U << State(2, 1) |
                 1U << State(1, 2) | 1U << State(2, 0) | 1U << State(0, 2);
    }
    // What may have been spent before the last byte: no more than some
    // wanted relation takes.
    unsigned open = 0;
    for (int out = 0; out <= 2; ++out) {
      for (int in = 0; in <= 2; ++in) {
        for (int more_out = out; more_out <= 2; ++more_out) {
          for (int more_in = in; more_in <= 2; ++more_in) {
            if ((wanted_ >> State(more_out, more_in)) & 1U) {
              open |= 1U << State(out, in);
            }
          }
        }
      }
    }
    // For the walk, all changes of a byte that lead on, in one set.
    within_.assign(2 * 9 * 256 * 4, 0);
    for (int last = 0; last < 2; ++last) {
      const unsigned allowed = last ? wanted_ : open;
      for (int out = 0; out <= 2; ++out) {
        for (int in = 0; in <= 2; ++in) {
          for (int o = 0; out + o <= 2; ++o) {
            for (int i = 0; in + i <= 2; ++i) {
              if (((allowed >> State(out + o, in + i)) & 1U) == 0) continue;
              for (int value = 0; value < 256; ++value) {
                uint64_t* const set =
                    within_.data() +
                    4 * (256 * (9 * last + State(out, in)) + value);
                for (int w = 0; w < 4; ++w) {
                  set[w] |= table_.changed[o][i][value][w];
                }
              }
            }
          }
        }
      }
    }
    tasks_.clear();
    if (block_search_) MakeTasks();
  }

  // Number of distinct strings of a group.
  size_t num_strings(int group) const {
    return det_start_[group].size() == 0 ? 0 : det_start_[group].size() - 1;
  }
  // Number of nodes of the tree of a group.
  size_t num_nodes(int group) const {
    return trees_[group].nodes.size() / kNodeWords;
  }
  bool has_transfers() const { return transfers_; }
  // Element n: the number of determinants with n electrons in group A.
  const std::vector<uint64_t>& sector_sizes() const { return sector_sizes_; }

  // Bytes of what every node holds, and the most the root held in addition
  // during a call of Extend (the strings being sorted).
  size_t shared_bytes() const {
    size_t bytes = 0;
    for (int group = 0; group < 2; ++group) {
      bytes += trees_[group].nodes.size() * sizeof(uint64_t) +
               det_start_[group].size() * sizeof(uint64_t) +
               det_index_[group].size() * sizeof(uint32_t) +
               block_dets_[group].size() * sizeof(Det<W>);
    }
    return bytes;
  }
  size_t transient_bytes() const { return max_transient_bytes_; }

  // Whether this rank is responsible for the row of determinant j.
  bool Owns(size_t j) const {
    return static_cast<long>(j) % num_ranks_ == rank_;
  }

  // Calls visit(i) once for every determinant i < j connected to
  // determinant j or, with `both_sides`, for every determinant connected to
  // it. Adds to *rejected the number of determinants that were compared with
  // j and not reported.
  template <class Visitor>
  ZSHCI_NOINLINE void ForEachConnected(size_t j, const Visitor& visit,
                                       uint64_t* rejected,
                                       bool both_sides = false) const {
    // Determinants up to this index are reported.
    const uint32_t limit = both_sides ? UINT32_MAX : static_cast<uint32_t>(j);
    const Det<W> det = dets_[j];
    uint8_t bytes[2][8 * W];
    uint64_t place[2];
    for (int group = 0; group < 2; ++group) {
      Bytes(Compact(det, group), group, bytes[group]);
      place[group] = Find(trees_[group], bytes[group]);
    }
    uint64_t compared = 0;
    uint64_t reported = 0;
    // The determinants in [begin, end) of the order of `group` that differ
    // from j in at most four spinors and at least `least`. Almost none
    // does, so the indices are only read for those.
    auto scan = [&](int group, uint64_t begin, uint64_t end, int least) {
      const uint32_t* const their_index = det_index_[group].data();
      Compare(&det, 1, block_dets_[group].data(), begin, end, least,
              [&](int /*k*/, uint64_t y) {
                const uint32_t i = their_index[y];
                if (i >= limit) return;
                visit(i);
                ++reported;
              });
      compared += end - begin;
    };
    // The same B string: doubles of the A string. (Singles of the A string
    // come with the related A strings below.)
    scan(1, det_start_[1][place[1]], det_start_[1][place[1] + 1], 4);
    // The same A string: singles and doubles of the B string.
    const uint64_t* const start = det_start_[0].data();
    scan(0, start[place[0]], start[place[0] + 1], 2);
    // Related A strings, with whatever B string fits.
    Walk(trees_[0], bytes[0], 0, 0, 0, 0, /*lower=*/false,
         [&](uint64_t target, int /*out*/, int /*in*/) {
           scan(0, start[target], start[target + 1], 0);
         });
    *rejected += compared - reported;
  }

  // --- The rows of the determinants of a string together ---
  //
  // A determinant alone reads the determinants of every string related to
  // its A string once, and so does every other determinant of that A
  // string: the same memory, again and again, which is what limits the
  // speed. Here the determinants of one A string are taken together and
  // each related string is read once for all of them. The determinants in
  // the order of group A are cut into chunks; a chunk is done by one
  // thread, string by string.

  static constexpr uint64_t kChunkSize = 256;
  size_t num_chunks() const {
    return (det_index_[0].size() + kChunkSize - 1) / kChunkSize;
  }
  // Whether this rank does chunk c. A rank takes whole chunks, not the rows
  // it would own one by one, so that the determinants of a string stay
  // together.
  bool OwnsChunk(size_t c) const {
    return static_cast<long>(c) % num_ranks_ == rank_;
  }

  // For the determinants of chunk c that wanted(j) accepts: calls
  // visit(k, j, i) for every determinant i < j connected to j, where k
  // numbers the accepted determinants of one string from 0, and then
  // done(indices, count) for that string with the indices of its `count`
  // accepted determinants, at most kChunkSize. Adds to *rejected the number
  // of pairs that were compared and not reported.
  template <class Wanted, class Visitor, class Done>
  ZSHCI_NOINLINE void ForEachConnectedOfChunk(size_t c, const Wanted& wanted,
                                              const Visitor& visit,
                                              const Done& done,
                                              uint64_t* rejected) const {
    const uint64_t* const start = det_start_[0].data();
    const uint32_t* const index = det_index_[0].data();
    const Det<W>* const mine = block_dets_[0].data();
    const uint64_t chunk_begin = c * kChunkSize;
    const uint64_t chunk_end =
        std::min<uint64_t>(det_index_[0].size(), chunk_begin + kChunkSize);
    // The string of the first determinant of the chunk.
    uint64_t place =
        std::upper_bound(start, start + det_start_[0].size(), chunk_begin) -
        start - 1;
    Det<W> rows[kChunkSize];
    uint32_t row_index[kChunkSize];
    uint64_t compared = 0;
    uint64_t reported = 0;
    uint8_t bytes[8 * W];
    for (uint64_t begin = chunk_begin; begin < chunk_end; ++place) {
      const uint64_t end = std::min(chunk_end, start[place + 1]);
      int count = 0;
      for (uint64_t x = begin; x < end; ++x) {
        if (!wanted(index[x])) continue;
        rows[count] = mine[x];
        row_index[count++] = index[x];
      }
      begin = end;
      if (count == 0) continue;
      // The determinants in [y_begin, y_end) of the order of `group`
      // against rows [k_begin, k_end): reported when they differ in at
      // most four spinors and at least `least` and come before the row.
      auto scan = [&](int group, uint64_t y_begin, uint64_t y_end, int least,
                      int k_begin, int k_end) {
        const uint32_t* const their_index = det_index_[group].data();
        Compare(rows + k_begin, k_end - k_begin, block_dets_[group].data(),
                y_begin, y_end, least, [&](int k, uint64_t y) {
                  const uint32_t i = their_index[y];
                  if (i >= row_index[k_begin + k]) return;
                  visit(k_begin + k, row_index[k_begin + k], i);
                  ++reported;
                });
        compared += (y_end - y_begin) * (k_end - k_begin);
      };
      // The same A string, related A strings, the same B string: as in
      // ForEachConnected.
      scan(0, start[place], start[place + 1], 2, 0, count);
      Bytes(Compact(rows[0], 0), 0, bytes);
      Walk(trees_[0], bytes, 0, 0, 0, 0, /*lower=*/false,
           [&](uint64_t target, int /*out*/, int /*in*/) {
             scan(0, start[target], start[target + 1], 0, 0, count);
           });
      for (int k = 0; k < count; ++k) {
        Bytes(Compact(rows[k], 1), 1, bytes);
        const uint64_t partner = Find(trees_[1], bytes);
        scan(1, det_start_[1][partner], det_start_[1][partner + 1], 4, k,
             k + 1);
      }
      done(row_index, count);
    }
    *rejected += compared - reported;
  }

  // --- Connections string by string (see connect_groups.h) ---

  // Switches the block-wise search on: Extend then also makes the tasks.
  // Call before Extend.
  void set_block_search(bool on) { block_search_ = on; }
  bool has_blocks() const { return block_search_ && !tasks_.empty(); }
  size_t num_tasks() const { return tasks_.size(); }
  // Whether this rank does task t.
  bool OwnsTask(size_t t) const {
    return static_cast<long>(t) % num_ranks_ == rank_;
  }

  // Calls visit(j, i) for the connected pairs of task t. Over all tasks
  // every connected pair of determinants is visited exactly once, in no
  // particular order of j and i. Adds the number of pairs compared to
  // *tested.
  template <class Visitor>
  ZSHCI_NOINLINE void ForEachConnectedOfTask(size_t t, const Visitor& visit,
                                             uint64_t* tested) const {
    const Task& task = tasks_[t];
    const int group = task.group;
    const uint64_t* const start = det_start_[group].data();
    const uint64_t begin = start[task.place];
    const Det<W>* const mine = block_dets_[group].data();
    const uint32_t* const index = det_index_[group].data();
    uint64_t compared = 0;
    // Same string: the determinants of the task with those before them.
    for (uint64_t x = task.begin; x < task.end; ++x) {
      for (uint64_t y = begin; y < x; ++y) {
        if (mine[x].NumDifferentBits(mine[y]) <= 4) visit(index[x], index[y]);
      }
      compared += x - begin;
    }
    if (group == 1) {
      *tested += compared;
      return;
    }
    // Related A strings at lower places.
    uint8_t bytes[8 * W];
    Bytes(Compact(mine[begin], 0), 0, bytes);
    Walk(trees_[0], bytes, 0, 0, 0, 0, /*lower=*/true,
         [&](uint64_t target, int out, int in) {
           const uint64_t t_begin = start[target];
           const uint64_t t_end = start[target + 1];
           // A single of the A string with the same B string belongs to the
           // tasks of that B string; it is the pair that differs in two
           // spinors only.
           const int least = out == 1 && in == 1 ? 4 : 0;
           Compare(mine + task.begin, static_cast<int>(task.end - task.begin),
                   mine, t_begin, t_end, least, [&](int k, uint64_t y) {
                     visit(index[task.begin + k], index[y]);
                   });
           compared += (task.end - task.begin) * (t_end - t_begin);
         });
    *tested += compared;
  }

 private:
  // A node of a tree: the set of values of the next byte, 256 bits, and the
  // number of the first child (a node of the next level or, at the last
  // level, the place of a string).
  static constexpr int kNodeWords = 5;

  // The distinct strings of a group as a tree over their bytes. Level 0
  // decides the highest byte.
  struct Tree {
    int num_bytes = 1;
    // The first node of every level among all nodes; one more for the end.
    std::vector<uint64_t> level_start;
    SharedArray<uint64_t> nodes;
  };

  // A determinant by its string of one group.
  struct Item {
    Det<W> key;  // the compact string
    uint32_t index;

    friend bool operator<(const Item& x, const Item& y) {
      return x.key != y.key ? x.key < y.key : x.index < y.index;
    }
  };

  // A string (group, place) with the range [begin, end) of its determinants
  // in the order of that group.
  struct Task {
    uint64_t place;
    uint64_t begin;
    uint64_t end;
    int group;
  };

  // The number of the pair (electrons out, electrons in), 0 to 8.
  static int State(int out, int in) { return 3 * out + in; }

  static uint64_t Gather(uint64_t word, uint64_t mask) {
#ifdef __BMI2__
    return _pext_u64(word, mask);
#else
    uint64_t out = 0;
    for (int k = 0; mask != 0; ++k) {
      const uint64_t lowest = mask & (~mask + 1);
      if ((word & lowest) != 0) out |= 1ULL << k;
      mask ^= lowest;
    }
    return out;
#endif
  }

  // The string of `det` in a group as a number of its own: bit k is the
  // occupation of the k-th spinor of the group.
  Det<W> Compact(const Det<W>& det, int group) const {
    Det<W> key;
    for (int k = 0; k < W; ++k) {
      const uint64_t bits = Gather(det.words[k], mask_[group].words[k]);
      if (bits == 0) continue;
      const int word = offset_[group][k] >> 6;
      const int shift = offset_[group][k] & 63;
      key.words[word] |= bits << shift;
      if (shift != 0 && word + 1 < W)
        key.words[word + 1] |= bits >> (64 - shift);
    }
    return key;
  }

  // The spinors of the two groups: those that are occupied in some
  // determinant. The others would only make the strings longer. Collective.
  void SetGroups() {
    Det<W> used;
    if (comm_.is_root()) {
      const long count = static_cast<long>(dets_.size());
#pragma omp parallel
      {
        Det<W> local;
#pragma omp for schedule(static) nowait
        for (long i = 0; i < count; ++i) {
          for (int k = 0; k < W; ++k) local.words[k] |= dets_[i].words[k];
        }
#pragma omp critical
        for (int k = 0; k < W; ++k) used.words[k] |= local.words[k];
      }
    }
    comm_.BcastValue(&used);
    for (int group = 0; group < 2; ++group) {
      int num_bits = 0;
      for (int k = 0; k < W; ++k) {
        mask_[group].words[k] =
            used.words[k] &
            (group == 0 ? group_a_.words[k] : ~group_a_.words[k]);
        offset_[group][k] = num_bits;
        num_bits += __builtin_popcountll(mask_[group].words[k]);
      }
      trees_[group].num_bytes = std::max(1, (num_bits + 7) / 8);
    }
  }

  // The bytes of a compact string in the order of the levels of the tree,
  // the highest first.
  void Bytes(const Det<W>& key, int group, uint8_t* bytes) const {
    const int num_bytes = trees_[group].num_bytes;
    for (int level = 0; level < num_bytes; ++level) {
      const int k = num_bytes - 1 - level;
      bytes[level] = static_cast<uint8_t>(key.words[k >> 3] >> (8 * (k & 7)));
    }
  }

  // The determinants as (compact string, index), sorted. Root only.
  Partitioned<Item> SortByString(int group) {
    const size_t n = dets_.size();
    const long count = static_cast<long>(n);
    RawArray<Item> items(n);
    const size_t num_counts = 64 * W + 1;
    std::vector<uint64_t> sizes(num_counts, 0);
#pragma omp parallel
    {
      std::vector<uint64_t> local(num_counts, 0);
#pragma omp for schedule(static) nowait
      for (long i = 0; i < count; ++i) {
        items[i] = {Compact(dets_[i], group), static_cast<uint32_t>(i)};
        ++local[items[i].key.Count()];
      }
#pragma omp critical
      for (size_t c = 0; c < num_counts; ++c) sizes[c] += local[c];
    }
    if (group == 0) {
      while (sizes.size() > 1 && sizes.back() == 0) sizes.pop_back();
      sector_sizes_ = sizes;
    }
    // Parts of about equal size in the order of the items, bounded by items
    // taken at regular steps, are sorted side by side. The index is part of
    // the order, so a string with very many determinants divides too.
    const size_t num_parts =
        n < 4096 ? 1 : 8 * static_cast<size_t>(omp_get_max_threads());
    std::vector<Item> bounds;
    if (num_parts > 1) {
      std::vector<Item> sample;
      const size_t step = std::max<size_t>(1, n / (64 * num_parts));
      for (size_t i = step / 2; i < n; i += step) sample.push_back(items[i]);
      std::sort(sample.begin(), sample.end());
      for (size_t p = 1; p < num_parts; ++p) {
        bounds.push_back(sample[p * sample.size() / num_parts]);
      }
    }
    Partitioned<Item> sorted =
        PartitionArray(items, num_parts, [&](const Item& item) {
          return static_cast<size_t>(
              std::upper_bound(bounds.begin(), bounds.end(), item) -
              bounds.begin());
        });
#pragma omp parallel for schedule(dynamic, 1)
    for (long part = 0; part < static_cast<long>(num_parts); ++part) {
      std::sort(sorted.begin(part), sorted.end(part));
    }
    return sorted;
  }

  // Where the determinants of each distinct string start in the sorted
  // items, with one more entry for the end.
  static std::vector<uint64_t> StringStarts(const Partitioned<Item>& sorted) {
    const Item* const items = sorted.items.data();
    const size_t n = sorted.items.size();
    const long num_parts = static_cast<long>(sorted.num_parts());
    auto starts_string = [&](size_t k) {
      return k == 0 || items[k].key != items[k - 1].key;
    };
    // The strings that start in each part, then their places.
    std::vector<uint64_t> first(num_parts + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long part = 0; part < num_parts; ++part) {
      uint64_t num = 0;
      for (size_t k = sorted.start[part]; k < sorted.start[part + 1]; ++k) {
        num += starts_string(k) ? 1 : 0;
      }
      first[part + 1] = num;
    }
    for (long part = 0; part < num_parts; ++part)
      first[part + 1] += first[part];
    std::vector<uint64_t> start(first[num_parts] + 1);
#pragma omp parallel for schedule(dynamic, 1)
    for (long part = 0; part < num_parts; ++part) {
      uint64_t place = first[part];
      for (size_t k = sorted.start[part]; k < sorted.start[part + 1]; ++k) {
        if (starts_string(k)) start[place++] = k;
      }
    }
    start[first[num_parts]] = n;
    return start;
  }

  // The tree of the distinct strings, which are the keys of the items at
  // `det_start`, in increasing order. Returns the nodes, level after level.
  static std::vector<uint64_t> MakeTree(const Partitioned<Item>& sorted,
                                        const std::vector<uint64_t>& det_start,
                                        Tree* tree) {
    const int num_bytes = tree->num_bytes;
    const size_t num_strings = det_start.size() - 1;
    auto byte_at = [&](const Det<W>& key, int level) {
      const int k = num_bytes - 1 - level;
      return static_cast<int>((key.words[k >> 3] >> (8 * (k & 7))) & 255);
    };
    std::vector<std::vector<uint64_t>> levels(num_bytes);
    // Nodes made so far at every level; the last entry counts the strings.
    std::vector<uint64_t> made(num_bytes + 1, 0);
    for (size_t s = 0; s < num_strings; ++s) {
      const Det<W>& key = sorted.items[det_start[s]].key;
      // The first level at which this string leaves the path of the one
      // before: there the node gains a child, below it new nodes begin.
      int fork = 0;
      if (s > 0) {
        const Det<W>& before = sorted.items[det_start[s - 1]].key;
        while (byte_at(key, fork) == byte_at(before, fork)) ++fork;
      }
      for (int level = s == 0 ? 0 : fork + 1; level < num_bytes; ++level) {
        // The first child is the node, or the string, about to be made.
        levels[level].insert(levels[level].end(),
                             {0, 0, 0, 0, made[level + 1]});
        ++made[level];
      }
      for (int level = fork; level < num_bytes; ++level) {
        const int value = byte_at(key, level);
        uint64_t* const node =
            levels[level].data() + levels[level].size() - kNodeWords;
        node[value >> 6] |= 1ULL << (value & 63);
      }
      ++made[num_bytes];
    }
    // One array, the children of inner nodes numbered among all nodes.
    tree->level_start.assign(1, 0);
    for (int level = 0; level < num_bytes; ++level) {
      tree->level_start.push_back(tree->level_start.back() + made[level]);
    }
    std::vector<uint64_t> nodes;
    nodes.reserve(tree->level_start.back() * kNodeWords);
    for (int level = 0; level < num_bytes; ++level) {
      if (level + 1 < num_bytes) {
        for (size_t k = kNodeWords - 1; k < levels[level].size();
             k += kNodeWords) {
          levels[level][k] += tree->level_start[level + 1];
        }
      }
      nodes.insert(nodes.end(), levels[level].begin(), levels[level].end());
      std::vector<uint64_t>().swap(levels[level]);
    }
    return nodes;
  }

  // A row and another determinant that differ in few spinors.
  struct Close {
    uint64_t other;
    uint32_t row;
  };

  // Calls emit(y) for the determinants list[y], begin <= y < end, that
  // differ from `one` in at most four spinors and at least `least`; hardly
  // any does.
  template <class Emit>
  static inline __attribute__((always_inline)) void FindCloseToOne(
      const Det<W>& one, const Det<W>* list, uint64_t begin, uint64_t end,
      int least, const Emit& emit) {
    // d is wanted when least <= d <= 4: one comparison of unsigned numbers.
    const unsigned span = 4 - least;
    uint64_t y = begin;
#ifdef __AVX2__
    if (W == 1) {
      // Four at a time: the bits of each byte are counted by looking up
      // its two halves, and the bytes of a word are summed.
      const __m256i mine = _mm256_set1_epi64x(one.words[0]);
      const __m256i low_half = _mm256_set1_epi8(0x0f);
      const __m256i counts =
          _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1,
                           1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
      const __m256i most = _mm256_set1_epi64x(4);
      for (; y + 4 <= end; y += 4) {
        const __m256i differ = _mm256_xor_si256(
            mine,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(list + y)));
        const __m256i per_byte = _mm256_add_epi8(
            _mm256_shuffle_epi8(counts, _mm256_and_si256(differ, low_half)),
            _mm256_shuffle_epi8(
                counts,
                _mm256_and_si256(_mm256_srli_epi64(differ, 4), low_half)));
        const __m256i per_word =
            _mm256_sad_epu8(per_byte, _mm256_setzero_si256());
        // All four differ in more than four spinors: nearly always.
        if (__builtin_expect(_mm256_movemask_pd(_mm256_castsi256_pd(
                                 _mm256_cmpgt_epi64(per_word, most))) == 15,
                             1)) {
          continue;
        }
        for (uint64_t z = y; z < y + 4; ++z) {
          const unsigned differing = one.NumDifferentBits(list[z]);
          if (differing - least <= span) emit(z);
        }
      }
    }
#endif
    for (; y < end; ++y) {
      const unsigned differing = one.NumDifferentBits(list[y]);
      if (__builtin_expect(differing - least <= span, 0)) emit(y);
    }
  }

  // Compares rows[k], k < num_rows, with theirs[y], begin <= y < end, and
  // appends to `found` the pairs that differ in at most four spinors and at
  // least `least`.
  //
  // This loop is where the time of every search goes. It is a function of
  // its own so that it is compiled the same way whatever is done with the
  // pairs; merged into its callers it ran up to a quarter slower or faster
  // depending on unrelated details of them. The longer of the two lists is
  // gone through in the inner loop.
  ZSHCI_NOINLINE static void FindClose(const Det<W>* rows, uint64_t num_rows,
                                       const Det<W>* theirs, uint64_t begin,
                                       uint64_t end, int least,
                                       std::vector<Close>* found) {
    if (num_rows < end - begin) {
      for (uint64_t k = 0; k < num_rows; ++k) {
        FindCloseToOne(rows[k], theirs, begin, end, least, [&](uint64_t y) {
          found->push_back({y, static_cast<uint32_t>(k)});
        });
      }
    } else {
      for (uint64_t y = begin; y < end; ++y) {
        FindCloseToOne(theirs[y], rows, 0, num_rows, least, [&](uint64_t k) {
          found->push_back({y, static_cast<uint32_t>(k)});
        });
      }
    }
  }

  // Calls hit(k, y) for every pair of rows[k], k < num_rows, and theirs[y],
  // begin <= y < end, that differ in at most four spinors and at least
  // `least`. The others are taken a limited number at a time, so that they
  // stay in the cache while the rows pass and the pairs found stay few.
  template <class Hit>
  static void Compare(const Det<W>* rows, int num_rows, const Det<W>* theirs,
                      uint64_t begin, uint64_t end, int least, const Hit& hit) {
    const uint64_t kTogether = 1024;
    static thread_local std::vector<Close> found;
    for (uint64_t from = begin; from < end; from += kTogether) {
      found.clear();
      FindClose(rows, num_rows, theirs, from, std::min(end, from + kTogether),
                least, &found);
      for (const Close& pair : found) hit(pair.row, pair.other);
    }
  }

  // The child of `node` for the byte `value`, which must be in its set.
  static uint64_t Child(const uint64_t* node, int value) {
    uint64_t child = node[4];
    for (int w = 0; w < (value >> 6); ++w) {
      child += __builtin_popcountll(node[w]);
    }
    return child + __builtin_popcountll(node[value >> 6] &
                                        ((1ULL << (value & 63)) - 1));
  }

  // The place of a string that occurs, given by its bytes.
  static uint64_t Find(const Tree& tree, const uint8_t* bytes) {
    uint64_t at = 0;
    for (int level = 0; level < tree.num_bytes; ++level) {
      at = Child(tree.nodes.data() + kNodeWords * at, bytes[level]);
    }
    return at;
  }

  // Calls visit(place, out, in) for every string of the tree that the
  // string with the given bytes turns into when `out` electrons leave and
  // `in` enter, for the pairs (out, in) in wanted_. The walk stands at the
  // node `at` of `level`, having spent (out, in) in the bytes above. With
  // `lower` only strings at lower places are wanted and the bytes above are
  // unchanged: a byte that changes must then decrease, after which every
  // string below is at a lower place. What a child spends follows from
  // its value and the own one.
  template <class Visitor>
  void Walk(const Tree& tree, const uint8_t* bytes, int level, uint64_t at,
            int out, int in, bool lower, const Visitor& visit) const {
    const uint64_t* const node = tree.nodes.data() + kNodeWords * at;
    const int own = bytes[level];
    const bool last = level + 1 == tree.num_bytes;
    // The values the byte may take here, all changes together.
    const uint64_t* const within =
        within_.data() +
        4 * (256 * (9 * (last ? 1 : 0) + State(out, in)) + own);
    // Children before those of each word of the set.
    uint64_t before = node[4];
    for (int w = 0; w < 4; ++w) {
      uint64_t found = within[w] & node[w];
      if (lower) {
        // Only the own value and those below it.
        if (w > (own >> 6)) break;
        if (w == (own >> 6)) found &= (2ULL << (own & 63)) - 1;
      }
      while (found != 0) {
        const uint64_t lowest = found & (~found + 1);
        found ^= lowest;
        const int value = 64 * w + __builtin_ctzll(lowest);
        const uint64_t child =
            before + __builtin_popcountll(node[w] & (lowest - 1));
        const int spent_out = out + __builtin_popcount(own & ~value);
        const int spent_in = in + __builtin_popcount(value & ~own);
        if (last) {
          visit(child, spent_out, spent_in);
        } else {
          Walk(tree, bytes, level + 1, child, spent_out, spent_in,
               lower && value == own, visit);
        }
      }
      before += __builtin_popcountll(node[w]);
    }
  }

  // The tasks of the block-wise search: every string with its determinants
  // in ranges of limited length, so that a string with very many does not
  // hold up one thread.
  void MakeTasks() {
    const uint64_t kTaskSize = 256;
    for (int group = 0; group < 2; ++group) {
      const uint64_t num_places = num_strings(group);
      for (uint64_t place = 0; place < num_places; ++place) {
        const uint64_t begin = det_start_[group][place];
        const uint64_t end = det_start_[group][place + 1];
        for (uint64_t x = begin; x < end; x += kTaskSize) {
          tasks_.push_back({place, x, std::min(end, x + kTaskSize), group});
        }
      }
    }
  }

  const Comm& comm_;
  Span<Det<W>> dets_;
  long rank_;
  long num_ranks_;
  Det<W> group_a_;
  const ByteTable& table_;
  // The spinors of each group that are occupied in some determinant, and
  // where those of every word start in the compact string.
  Det<W> mask_[2];
  int offset_[2][W];
  bool transfers_ = false;
  // Bit State(out, in) is set for the relations looked for.
  unsigned wanted_ = 0;
  // For a byte value v, after (out, in) were spent in the bytes above, the
  // values it may take so that a wanted relation is still within reach or,
  // at the last byte, complete: the union of ByteTable::changed over what
  // may be spent here. The four words of v are at
  // 4 * (256 * (9 * last + State(out, in)) + v).
  std::vector<uint64_t> within_;
  size_t max_transient_bytes_ = 0;
  std::vector<uint64_t> sector_sizes_;
  Tree trees_[2];
  // The determinants that contain the string at place s of a group are
  // det_index_[det_start_[s] .. det_start_[s + 1]), increasing, and
  // block_dets_ holds them in that order.
  SharedArray<uint64_t> det_start_[2];
  SharedArray<uint32_t> det_index_[2];
  SharedArray<Det<W>> block_dets_[2];
  bool block_search_ = false;
  std::vector<Task> tasks_;
};

}  // namespace internal
}  // namespace zshci

#undef ZSHCI_NOINLINE

#endif  // ZSHCI_SRC_CONNECT_BYTES_H_
