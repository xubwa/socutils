// Finds the pairs of determinants connected by the Hamiltonian and builds the
// corresponding block of the sparse matrix.
//
// Both methods here work the same way: every determinant emits (key, index)
// entries, the entries are sorted by key, and determinants that share a key -
// a bucket - are tested against each other. No spin or time-reversal
// structure of the orbitals is needed. They differ in what a key stands for.
//
// Strings. Two determinants that differ by a double excitation share exactly
// one string of N-2 electrons, and two that differ by a single excitation
// share exactly one string of N-1 electrons. A determinant emits one entry
// per string it contains: n + n(n-1)/2 of them for n electrons, whether or
// not anything is connected to it through that string. Every pair in a
// bucket is connected, but the number of entries grows with the square of
// the electron number.
//
// Segments. The spinors are cut into K segments. Determinants connected by
// at most a double excitation differ in at most four spinors, hence in at
// most four segments. For each choice of four segments to cover, a
// determinant emits one entry, keyed by its occupation outside the covered
// segments; a bucket then holds determinants that agree outside them, and a
// bit count tells which pairs are connected. One pass per choice, one entry
// per determinant per pass: the memory does not depend on the electron
// number. The price is pairs in a bucket that turn out not to be connected.
//
// A third method divides the spinors into two groups and works from tables
// of the strings of each group instead (connect_groups.h).

#ifndef ZSHCI_SRC_CONNECT_H_
#define ZSHCI_SRC_CONNECT_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "comm.h"
#include "connect_groups.h"
#include "det.h"
#include "hamiltonian.h"
#include "integrals.h"
#include "parallel.h"
#include "sparse_h.h"

namespace zshci {

enum class ConnectMethod { kStrings, kSegments, kGroups };

struct ConnectOptions {
  // Elements with |H(i,j)| <= threshold are not stored. A negative value
  // keeps every connected pair, which the density matrices need.
  double threshold = -1.0;
  ConnectMethod method = ConnectMethod::kStrings;
  // Strings: the keys are processed in this many passes; the transient
  // memory of the entry list is divided by the same factor.
  int num_batches = 1;
  // Segments: the number of segments K.
  int num_segments = 8;
  // Groups: the spinors of group A; empty means those with an even index
  // (the alpha spin orbitals, or one spinor of every interleaved Kramers
  // pair).
  std::vector<int> group_a;
};

// What a connection search did, summed over calls: how many pairs sharing a
// bucket were tested, how many of them were connected, and the largest
// bucket met. Tested pairs that are not connected are the overhead of the
// segment method; the string method has none apart from hash collisions.
struct ConnectStats {
  uint64_t tested_pairs = 0;
  uint64_t connected_pairs = 0;
  uint64_t largest_bucket = 0;
  // Groups: the distinct strings of both groups and the entries of their
  // relation lists, at the last call.
  uint64_t num_strings = 0;
  uint64_t num_links = 0;
  // Seconds spent preparing the search (tables of the group method),
  // counting the elements of every row, filling them in, and sorting the
  // rows.
  double seconds_prepare = 0.0;
  double seconds_count = 0.0;
  double seconds_fill = 0.0;
  double seconds_sort = 0.0;
};

namespace internal {

struct BucketEntry {
  uint64_t key;
  DetIndex index;

  friend bool operator<(const BucketEntry& a, const BucketEntry& b) {
    return a.key != b.key ? a.key < b.key : a.index < b.index;
  }
};

// A finder decides what the keys are. It provides
//   int num_passes() const;
//   void Keys(const Det<W>& det, int pass, const Emit& emit) const;
//       calls emit(key) for every key of `det` in this pass;
//   bool Accept(const Det<W>& a, const Det<W>& b, uint64_t key, int pass) const;
//       whether the pair, found in the bucket `key`, is connected and is to
//       be reported from this bucket.
// Over all passes and all ranks, every connected pair must be accepted
// exactly once.

// Keys are strings of N-1 and N-2 electrons. The key space is split first
// into batches (the passes, to bound memory) and then among the ranks.
template <int W>
class StringFinder {
 public:
  StringFinder(int num_batches, const Comm& comm)
      : num_batches_(std::max(1, num_batches)),
        rank_(comm.rank()),
        num_ranks_(comm.size()) {}

  int num_passes() const { return static_cast<int>(num_batches_); }

  template <class Emit>
  void Keys(const Det<W>& det, int pass, const Emit& emit) const {
    int occ[64 * W];
    int num_occ = 0;
    for (int k = 0; k < W; ++k) {
      uint64_t word = det.words[k];
      while (word != 0) {
        occ[num_occ++] = 64 * k + __builtin_ctzll(word);
        word &= word - 1;
      }
    }
    for (int m = 0; m < num_occ; ++m) {
      Det<W> minus_one = det;
      minus_one.Clear(occ[m]);
      const uint64_t key1 = Key(minus_one, 1);
      if (Mine(key1, pass)) emit(key1);
      for (int n = 0; n < m; ++n) {
        Det<W> minus_two = minus_one;
        minus_two.Clear(occ[n]);
        const uint64_t key2 = Key(minus_two, 2);
        if (Mine(key2, pass)) emit(key2);
      }
    }
  }

  bool Accept(const Det<W>& a, const Det<W>& b, uint64_t key,
              int /*pass*/) const {
    const int num_bits = a.NumDifferentBits(b);
    if (num_bits != 2 && num_bits != 4) return false;
    // The shared string must be the one this bucket stands for; otherwise
    // the pair is here through a hash collision and is reported from its
    // own bucket.
    return Key(a.And(b), num_bits / 2) == key;
  }

 private:
  // Salting with the number of removed electrons keeps N-1 and N-2 strings
  // in separate buckets.
  static uint64_t Key(const Det<W>& string, int num_removed) {
    return Mix64(string.Hash() + static_cast<uint64_t>(num_removed));
  }
  bool Mine(uint64_t key, int pass) const {
    return key % num_batches_ == static_cast<uint64_t>(pass) &&
           (key / num_batches_) % num_ranks_ == rank_;
  }

  uint64_t num_batches_;
  uint64_t rank_;
  uint64_t num_ranks_;
};

// Keys are the occupations outside a cover of four segments. Each pass
// handles one cover; the covers are dealt out to the ranks.
template <int W>
class SegmentFinder {
 public:
  // The covers are enumerated by counting through all subsets of the
  // segments, which bounds their number.
  static constexpr int kMaxSegments = 20;  // also the input's upper bound

  // Cuts the spinors 0 .. num_orbs-1 into consecutive segments that carry
  // about the same amount of information about `dets`. A spinor that is
  // occupied in (almost) all or in (almost) none of the determinants tells
  // them apart poorly, and a segment made of such spinors would give keys
  // shared by nearly everything; so the cuts are placed at equal steps of the
  // summed occupation entropy rather than of the spinor index.
  SegmentFinder(Span<Det<W>> dets, int num_orbs, int num_segments,
                const Comm& comm) {
    const int k =
        std::max(1, std::min(num_segments, std::min(num_orbs, kMaxSegments)));
    const std::vector<int> first = SegmentStarts(dets, num_orbs, k);
    segments_.resize(k);
    for (int s = 0; s < k; ++s) {
      for (int p = first[s]; p < first[s + 1]; ++p) segments_[s].Set(p);
    }
    cover_size_ = std::min(4, k);
    // All covers, in increasing order of their bit pattern; this rank takes
    // every size()-th one.
    int count = 0;
    for (uint64_t bits = 0; bits < (1ULL << k); ++bits) {
      if (__builtin_popcountll(bits) != cover_size_) continue;
      if (count++ % comm.size() != comm.rank()) continue;
      Cover cover;
      cover.bits = static_cast<uint32_t>(bits);
      for (int s = 0; s < k; ++s) {
        if (((bits >> s) & 1U) == 0) {
          for (int w = 0; w < W; ++w) cover.keep.words[w] |= segments_[s].words[w];
        }
      }
      covers_.push_back(cover);
    }
  }

  int num_passes() const { return static_cast<int>(covers_.size()); }

  template <class Emit>
  void Keys(const Det<W>& det, int pass, const Emit& emit) const {
    emit(Mix64(det.And(covers_[pass].keep).Hash() + covers_[pass].bits));
  }

  // A pair is reported from the cover made of the segments in which it
  // differs, filled up with the lowest other segments. A pair that shares
  // this bucket only through a hash collision differs outside the cover and
  // is rejected by the same test.
  bool Accept(const Det<W>& a, const Det<W>& b, uint64_t /*key*/,
              int pass) const {
    const int num_bits = a.NumDifferentBits(b);
    if (num_bits != 2 && num_bits != 4) return false;
    uint32_t differing = 0;
    for (size_t s = 0; s < segments_.size(); ++s) {
      for (int w = 0; w < W; ++w) {
        if (((a.words[w] ^ b.words[w]) & segments_[s].words[w]) != 0) {
          differing |= 1U << s;
          break;
        }
      }
    }
    uint32_t canonical = differing;
    for (int s = 0; __builtin_popcount(canonical) < cover_size_; ++s) {
      canonical |= 1U << s;
    }
    return canonical == covers_[pass].bits;
  }

 private:
  struct Cover {
    uint32_t bits = 0;  // bit s: segment s is covered
    Det<W> keep;        // the spinors outside the cover
  };

  // Returns k + 1 increasing spinor indices: segment s is
  // [first[s], first[s + 1]). Every segment holds at least one spinor.
  static std::vector<int> SegmentStarts(Span<Det<W>> dets, int num_orbs,
                                        int k) {
    // How often each spinor is occupied.
    std::vector<uint64_t> count(num_orbs, 0);
#pragma omp parallel
    {
      std::vector<uint64_t> local(num_orbs, 0);
#pragma omp for schedule(static) nowait
      for (size_t d = 0; d < dets.size(); ++d) {
        for (int w = 0; w < W; ++w) {
          uint64_t word = dets[d].words[w];
          while (word != 0) {
            const int p = 64 * w + __builtin_ctzll(word);
            if (p < num_orbs) ++local[p];
            word &= word - 1;
          }
        }
      }
#pragma omp critical
      for (int p = 0; p < num_orbs; ++p) count[p] += local[p];
    }
    // Entropy of each occupation, plus a small floor so that the cuts stay
    // defined when the spinors carry no information at all.
    std::vector<double> weight(num_orbs);
    double total = 0.0;
    for (int p = 0; p < num_orbs; ++p) {
      const double f =
          dets.empty() ? 0.0 : static_cast<double>(count[p]) / dets.size();
      double entropy = 0.0;
      if (f > 0.0 && f < 1.0) {
        entropy = -f * std::log(f) - (1.0 - f) * std::log(1.0 - f);
      }
      weight[p] = entropy + 1e-6;
      total += weight[p];
    }
    std::vector<int> first(k + 1, num_orbs);
    first[0] = 0;
    double running = 0.0;
    int next = 1;
    for (int p = 0; p < num_orbs && next < k; ++p) {
      running += weight[p];
      // Cut after p when the target is reached, but leave enough spinors for
      // the segments still to come.
      const bool reached = running * k >= total * next;
      const bool must = num_orbs - (p + 1) == k - next;
      if ((reached || must) && p + 1 > first[next - 1]) first[next++] = p + 1;
    }
    return first;
  }

  std::vector<Det<W>> segments_;  // the spinors of each segment
  std::vector<Cover> covers_;
  int cover_size_ = 0;
};

// Appends to `chunks` (one per thread) the entries of dets[begin..end) in
// this pass whose key is accepted by `wanted`.
template <int W, class Finder, class Wanted>
void AppendEntries(Span<Det<W>> dets, size_t begin, size_t end,
                   const Finder& finder, int pass, const Wanted& wanted,
                   std::vector<std::vector<BucketEntry>>* chunks) {
  const size_t first_chunk = chunks->size();
  chunks->resize(first_chunk + omp_get_max_threads());
#pragma omp parallel
  {
    std::vector<BucketEntry>& local =
        (*chunks)[first_chunk + omp_get_thread_num()];
#pragma omp for schedule(static) nowait
    for (size_t d = begin; d < end; ++d) {
      const DetIndex index = static_cast<DetIndex>(d);
      finder.Keys(dets[d], pass, [&](uint64_t key) {
        if (wanted(key)) local.push_back({key, index});
      });
    }
  }
}

// The part, out of `num_parts` equal ranges of the 64-bit key space, that
// holds `key`. Monotonic in the key.
inline size_t KeyRange(uint64_t key, size_t num_parts) {
  return static_cast<size_t>(
      (static_cast<unsigned __int128>(key) * num_parts) >> 64);
}

// Returns the entries of one pass that can connect a determinant to one of
// dets[first_new..], split into ranges of the key and sorted within each
// range. Only buckets holding a new determinant matter, so the old
// determinants contribute only the keys they share with a new one.
template <int W, class Finder>
Partitioned<BucketEntry> MakeEntries(Span<Det<W>> dets, size_t first_new,
                                     const Finder& finder, int pass) {
  // More parts than threads, so that uneven parts even out.
  const size_t num_parts = 8 * static_cast<size_t>(omp_get_max_threads());
  std::vector<std::vector<BucketEntry>> chunks;
  AppendEntries(dets, first_new, dets.size(), finder, pass,
                [](uint64_t) { return true; }, &chunks);

  if (first_new > 0) {
    // The keys of the new determinants, sorted within each key range.
    std::vector<std::vector<uint64_t>> key_chunks(chunks.size());
    const long num_chunks = static_cast<long>(chunks.size());
#pragma omp parallel for schedule(static, 1)
    for (long c = 0; c < num_chunks; ++c) {
      key_chunks[c].reserve(chunks[c].size());
      for (const BucketEntry& entry : chunks[c]) {
        key_chunks[c].push_back(entry.key);
      }
    }
    Partitioned<uint64_t> new_keys = PartitionChunks(
        &key_chunks, num_parts,
        [&](uint64_t key) { return KeyRange(key, num_parts); });
#pragma omp parallel for schedule(dynamic, 1)
    for (long part = 0; part < static_cast<long>(num_parts); ++part) {
      std::sort(new_keys.begin(part), new_keys.end(part));
    }
    AppendEntries(dets, 0, first_new, finder, pass,
                  [&](uint64_t key) {
                    const size_t part = KeyRange(key, num_parts);
                    return std::binary_search(new_keys.begin(part),
                                              new_keys.end(part), key);
                  },
                  &chunks);
  }

  Partitioned<BucketEntry> entries = PartitionChunks(
      &chunks, num_parts,
      [&](const BucketEntry& entry) { return KeyRange(entry.key, num_parts); });
#pragma omp parallel for schedule(dynamic, 1)
  for (long part = 0; part < static_cast<long>(num_parts); ++part) {
    std::sort(entries.begin(part), entries.end(part));
  }
  return entries;
}

// Calls visit(i, j) once for every pair i < j with j >= first_new that the
// finder accepts. `visit` runs concurrently from several threads.
template <int W, class Finder, class Visitor>
void ForEachConnectedPair(Span<Det<W>> dets, size_t first_new,
                          const Partitioned<BucketEntry>& entries,
                          const Finder& finder, int pass, const Visitor& visit,
                          ConnectStats* stats = nullptr) {
  const long num_parts = static_cast<long>(entries.num_parts());
  uint64_t tested = 0;
  uint64_t connected = 0;
  uint64_t largest = 0;
#pragma omp parallel reduction(+ : tested, connected) reduction(max : largest)
  {
    // The determinants of one bucket, each once: a hash collision can put
    // the same determinant twice under one key.
    std::vector<DetIndex> bucket;
#pragma omp for schedule(dynamic, 1)
    for (long part = 0; part < num_parts; ++part) {
      const BucketEntry* const part_end =
          entries.items.data() + entries.start[part + 1];
      const BucketEntry* run = entries.items.data() + entries.start[part];
      while (run != part_end) {
        const uint64_t key = run->key;
        bucket.clear();
        for (; run != part_end && run->key == key; ++run) {
          if (bucket.empty() || bucket.back() != run->index) {
            bucket.push_back(run->index);
          }
        }
        // Indices increase within a bucket, so the last one decides whether
        // the bucket holds a new determinant at all.
        if (bucket.size() < 2 || bucket.back() < first_new) continue;
        largest = std::max<uint64_t>(largest, bucket.size());
        for (size_t a = 0; a < bucket.size(); ++a) {
          const DetIndex i = bucket[a];
          for (size_t b = a + 1; b < bucket.size(); ++b) {
            const DetIndex j = bucket[b];
            if (j < first_new) continue;
            ++tested;
            if (finder.Accept(dets[i], dets[j], key, pass)) {
              ++connected;
              visit(i, j);
            }
          }
        }
      }
    }
  }
  if (stats != nullptr) {
    stats->tested_pairs += tested;
    stats->connected_pairs += connected;
    stats->largest_bucket = std::max(stats->largest_bucket, largest);
  }
}

// Builds the block from the connected pairs. for_each_pair(visit, stats)
// must call visit(i, j) once for every connected pair i < j with
// j >= first_new, possibly from several threads at once; it is called twice,
// the second time with stats == nullptr.
template <int W, typename S, class ForEachPair>
CsrBlock<S> BuildBlockFromPairs(Span<Det<W>> dets, size_t first_new,
                                const Integrals<S>& ints, double threshold,
                                const ForEachPair& for_each_pair,
                                ConnectStats* stats) {
  const size_t n = dets.size();
  if (n > static_cast<size_t>(UINT32_MAX)) {
    throw std::length_error("too many determinants for a 32-bit index");
  }
  const bool keep_all = threshold < 0.0;

  // With a threshold the element has to be evaluated to know whether it is
  // stored, so the counting pass evaluates it too.
  auto stored = [&](DetIndex i, DetIndex j) {
    return keep_all || std::abs(Hij(dets[j], dets[i], ints)) > threshold;
  };

  // Pass 1: number of stored elements per row. This array and the cursors
  // below are the only ones of length n; they are released before returning.
  std::vector<uint64_t> count(n, 0);
  double time = omp_get_wtime();
  for_each_pair(
      [&](DetIndex i, DetIndex j) {
        if (!stored(i, j)) return;
#pragma omp atomic
        ++count[j];
      },
      stats);
  if (stats != nullptr) stats->seconds_count += omp_get_wtime() - time;

  // The rows that hold elements, and where each one starts. `count` turns
  // into the write cursor of every row.
  CsrBlock<S> block;
  block.dim = n;
  block.first_row = first_new;
  size_t num_stored_rows = 0;
  for (size_t i = 0; i < n; ++i) num_stored_rows += count[i] > 0 ? 1 : 0;
  block.row = RawArray<DetIndex>(num_stored_rows);
  block.row_ptr.assign(num_stored_rows + 1, 0);
  {
    size_t r = 0;
    uint64_t offset = 0;
    for (size_t i = 0; i < n; ++i) {
      if (count[i] == 0) continue;
      block.row[r] = static_cast<DetIndex>(i);
      block.row_ptr[r] = offset;
      const uint64_t length = count[i];
      count[i] = offset;
      offset += length;
      ++r;
    }
    block.row_ptr[num_stored_rows] = offset;
  }
  std::vector<uint64_t>& cursor = count;
  block.col = RawArray<DetIndex>(block.row_ptr[num_stored_rows]);
  block.val = RawArray<S>(block.row_ptr[num_stored_rows]);

  // Pass 2: fill. Each row has its own cursor, advanced atomically.
  time = omp_get_wtime();
  for_each_pair(
      [&](DetIndex i, DetIndex j) {
        const S value = Hij(dets[j], dets[i], ints);
        if (!keep_all && std::abs(value) <= threshold) return;
        uint64_t slot;
#pragma omp atomic capture
        slot = cursor[j]++;
        block.col[slot] = i;
        block.val[slot] = value;
      },
      nullptr);
  if (stats != nullptr) stats->seconds_fill += omp_get_wtime() - time;

  // Sort every row by column.
  time = omp_get_wtime();
#pragma omp parallel
  {
    std::vector<std::pair<DetIndex, S>> row;
#pragma omp for schedule(dynamic, 256)
    for (size_t r = 0; r < num_stored_rows; ++r) {
      const uint64_t begin = block.row_ptr[r];
      const uint64_t end = block.row_ptr[r + 1];
      row.clear();
      for (uint64_t k = begin; k < end; ++k) {
        row.emplace_back(block.col[k], block.val[k]);
      }
      std::sort(
          row.begin(), row.end(),
          [](const std::pair<DetIndex, S>& a, const std::pair<DetIndex, S>& b) {
            return a.first < b.first;
          });
      for (uint64_t k = begin; k < end; ++k) {
        block.col[k] = row[k - begin].first;
        block.val[k] = row[k - begin].second;
      }
    }
  }
  if (stats != nullptr) stats->seconds_sort += omp_get_wtime() - time;
  return block;
}

// Builds the rows of the determinants this rank owns from the string tables
// of two groups, in one sweep: each thread finds the connections of a
// determinant, evaluates the elements and appends the finished row to a block
// of its own. Returns the blocks that hold elements.
template <int W, typename S>
std::vector<CsrBlock<S>> BuildGroupBlocks(Span<Det<W>> dets, size_t first_new,
                                          const Integrals<S>& ints,
                                          double threshold,
                                          const GroupStringTables<W>& tables,
                                          ConnectStats* stats) {
  const size_t n = dets.size();
  const bool keep_all = threshold < 0.0;
  const double time = omp_get_wtime();
  std::vector<CsrBlock<S>> blocks(omp_get_max_threads());
  uint64_t tested = 0;
  uint64_t connected = 0;
#pragma omp parallel reduction(+ : tested, connected)
  {
    // The block grows in place as rows are appended; nothing is copied at
    // the end.
    std::vector<DetIndex> rows;
    std::vector<uint64_t> row_ptr(1, 0);
    RawArray<DetIndex> cols;
    RawArray<S> vals;
    size_t num_elements = 0;
    std::vector<std::pair<DetIndex, S>> row;
#pragma omp for schedule(dynamic, 64) nowait
    for (long j = static_cast<long>(first_new); j < static_cast<long>(n); ++j) {
      if (!tables.Owns(j)) continue;
      row.clear();
      tables.ForEachConnected(
          j,
          [&](DetIndex i) {
            ++connected;
            const S value = Hij(dets[j], dets[i], ints);
            if (keep_all || std::abs(value) > threshold) {
              row.emplace_back(i, value);
            }
          },
          &tested);
      if (row.empty()) continue;
      std::sort(
          row.begin(), row.end(),
          [](const std::pair<DetIndex, S>& a, const std::pair<DetIndex, S>& b) {
            return a.first < b.first;
          });
      if (num_elements + row.size() > cols.size()) {
        const size_t capacity =
            std::max(num_elements + row.size(), 2 * cols.size() + 1024);
        cols.Resize(capacity);
        vals.Resize(capacity);
      }
      rows.push_back(static_cast<DetIndex>(j));
      for (const std::pair<DetIndex, S>& element : row) {
        cols[num_elements] = element.first;
        vals[num_elements] = element.second;
        ++num_elements;
      }
      row_ptr.push_back(num_elements);
    }
    // The schedule hands every thread its determinants in increasing order,
    // so the rows are sorted.
    cols.Resize(num_elements);
    vals.Resize(num_elements);
    CsrBlock<S>& block = blocks[omp_get_thread_num()];
    block.dim = n;
    block.first_row = first_new;
    block.row = RawArray<DetIndex>(rows.size());
    std::copy(rows.begin(), rows.end(), block.row.begin());
    block.row_ptr = std::move(row_ptr);
    block.col = std::move(cols);
    block.val = std::move(vals);
  }
  std::vector<CsrBlock<S>> filled;
  for (CsrBlock<S>& block : blocks) {
    if (block.num_elements() > 0) filled.push_back(std::move(block));
  }
  if (stats != nullptr) {
    // Every reported pair counts as tested, as with the other methods.
    stats->tested_pairs += tested + connected;
    stats->connected_pairs += connected;
    stats->seconds_fill += omp_get_wtime() - time;
  }
  return filled;
}

// One block with the rows of several, which must not share a row.
template <typename S>
CsrBlock<S> MergeBlocks(std::vector<CsrBlock<S>> blocks, size_t dim,
                        size_t first_row) {
  struct Source {
    DetIndex row;
    uint32_t block;
    size_t index;
  };
  std::vector<Source> sources;
  size_t num_elements = 0;
  for (size_t b = 0; b < blocks.size(); ++b) {
    num_elements += blocks[b].num_elements();
    for (size_t r = 0; r < blocks[b].num_stored_rows(); ++r) {
      sources.push_back({blocks[b].row[r], static_cast<uint32_t>(b), r});
    }
  }
  std::sort(sources.begin(), sources.end(),
            [](const Source& a, const Source& b) { return a.row < b.row; });
  CsrBlock<S> merged;
  merged.dim = dim;
  merged.first_row = first_row;
  merged.row = RawArray<DetIndex>(sources.size());
  merged.row_ptr.assign(1, 0);
  merged.col = RawArray<DetIndex>(num_elements);
  merged.val = RawArray<S>(num_elements);
  uint64_t offset = 0;
  for (size_t r = 0; r < sources.size(); ++r) {
    const CsrBlock<S>& from = blocks[sources[r].block];
    const uint64_t begin = from.row_ptr[sources[r].index];
    const uint64_t end = from.row_ptr[sources[r].index + 1];
    merged.row[r] = sources[r].row;
    std::copy(from.col.begin() + begin, from.col.begin() + end,
              merged.col.begin() + offset);
    std::copy(from.val.begin() + begin, from.val.begin() + end,
              merged.val.begin() + offset);
    offset += end - begin;
    merged.row_ptr.push_back(offset);
  }
  return merged;
}

// Builds the block from the pairs a bucket finder reports.
template <int W, class Finder, typename S>
CsrBlock<S> BuildBlockWith(Span<Det<W>> dets, size_t first_new,
                           const Integrals<S>& ints, double threshold,
                           const Finder& finder, ConnectStats* stats) {
  const int num_passes = finder.num_passes();
  // A single pass keeps its entries for the second round.
  Partitioned<BucketEntry> cached;
  bool have_cached = false;
  auto for_each_pair = [&](const auto& visit, ConnectStats* pass_stats) {
    for (int pass = 0; pass < num_passes; ++pass) {
      if (num_passes == 1) {
        if (!have_cached) cached = MakeEntries(dets, first_new, finder, pass);
        have_cached = true;
        ForEachConnectedPair(dets, first_new, cached, finder, pass, visit,
                             pass_stats);
      } else {
        const Partitioned<BucketEntry> entries =
            MakeEntries(dets, first_new, finder, pass);
        ForEachConnectedPair(dets, first_new, entries, finder, pass, visit,
                             pass_stats);
      }
    }
  };
  return BuildBlockFromPairs(dets, first_new, ints, threshold, for_each_pair,
                             stats);
}

}  // namespace internal

// Builds the blocks of H(j, i), i < j, with j >= first_new, for the
// determinants `dets`. Determinants before `first_new` are assumed to be
// connected among themselves by earlier blocks. The group method
// returns one block per thread, the others a single one.
//
// With several ranks, each rank handles its share of the keys (strings), of
// the covers (segments) or of the determinants (groups) and returns
// the elements found there. The blocks of all ranks together hold each
// element once; no communication is needed. `stats`, if given, is updated
// with what this rank's search did.
template <int W, typename S>
std::vector<CsrBlock<S>> BuildBlocks(Span<Det<W>> dets, size_t first_new,
                                     const Integrals<S>& ints,
                                     const ConnectOptions& options,
                                     const Comm& comm = Comm(),
                                     ConnectStats* stats = nullptr) {
  if (dets.size() > static_cast<size_t>(UINT32_MAX)) {
    throw std::length_error("too many determinants for a 32-bit index");
  }
  if (options.method == ConnectMethod::kGroups) {
    Det<W> group_a;
    if (options.group_a.empty()) {
      for (int p = 0; p < ints.num_orbs(); p += 2) group_a.Set(p);
    }
    for (int p : options.group_a) {
      if (p < 0 || p >= ints.num_orbs()) {
        throw std::invalid_argument("connect: spinor of group A out of range");
      }
      group_a.Set(p);
    }
    const double time = omp_get_wtime();
    const internal::GroupStringTables<W> tables(dets, group_a, comm);
    if (stats != nullptr) {
      stats->seconds_prepare += omp_get_wtime() - time;
      stats->num_strings = tables.num_strings(0) + tables.num_strings(1);
      stats->num_links = tables.num_links();
    }
    return internal::BuildGroupBlocks(dets, first_new, ints, options.threshold,
                                      tables, stats);
  }
  std::vector<CsrBlock<S>> blocks;
  if (options.method == ConnectMethod::kSegments) {
    const internal::SegmentFinder<W> finder(dets, ints.num_orbs(),
                                            options.num_segments, comm);
    blocks.push_back(internal::BuildBlockWith(
        dets, first_new, ints, options.threshold, finder, stats));
  } else {
    const internal::StringFinder<W> finder(options.num_batches, comm);
    blocks.push_back(internal::BuildBlockWith(
        dets, first_new, ints, options.threshold, finder, stats));
  }
  return blocks;
}

// The same as a single block.
template <int W, typename S>
CsrBlock<S> BuildBlock(Span<Det<W>> dets, size_t first_new,
                       const Integrals<S>& ints, const ConnectOptions& options,
                       const Comm& comm = Comm(),
                       ConnectStats* stats = nullptr) {
  std::vector<CsrBlock<S>> blocks =
      BuildBlocks(dets, first_new, ints, options, comm, stats);
  if (blocks.size() == 1) return std::move(blocks[0]);
  return internal::MergeBlocks(std::move(blocks), dets.size(), first_new);
}

// Reference implementation: tests every pair of determinants. Quadratic; used
// only to check BuildBlock on small cases.
template <int W, typename S>
CsrBlock<S> BuildBlockBruteForce(Span<Det<W>> dets, size_t first_new,
                                 const Integrals<S>& ints,
                                 const ConnectOptions& options) {
  const size_t n = dets.size();
  CsrBlock<S> block;
  block.dim = n;
  block.first_row = first_new;
  std::vector<DetIndex> row;
  std::vector<DetIndex> col;
  std::vector<S> val;
  block.row_ptr.assign(1, 0);
  for (size_t j = first_new; j < n; ++j) {
    for (size_t i = 0; i < j; ++i) {
      const int num_bits = dets[i].NumDifferentBits(dets[j]);
      if (num_bits != 2 && num_bits != 4) continue;
      const S value = Hij(dets[j], dets[i], ints);
      if (options.threshold >= 0.0 && std::abs(value) <= options.threshold) {
        continue;
      }
      col.push_back(static_cast<DetIndex>(i));
      val.push_back(value);
    }
    if (col.size() > block.row_ptr.back()) {
      row.push_back(static_cast<DetIndex>(j));
      block.row_ptr.push_back(col.size());
    }
  }
  block.row = RawArray<DetIndex>(row.size());
  block.col = RawArray<DetIndex>(col.size());
  block.val = RawArray<S>(val.size());
  std::copy(row.begin(), row.end(), block.row.begin());
  std::copy(col.begin(), col.end(), block.col.begin());
  std::copy(val.begin(), val.end(), block.val.begin());
  return block;
}

}  // namespace zshci

#endif  // ZSHCI_SRC_CONNECT_H_
