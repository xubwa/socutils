// Finds the pairs of determinants connected by the Hamiltonian and builds the
// corresponding block of the sparse matrix.
//
// Two determinants that differ by a double excitation share exactly one
// string of N-2 electrons, and two that differ by a single excitation share
// exactly one string of N-1 electrons. Every determinant therefore emits one
// (key, index) entry per string it contains, the key being a 64-bit hash of
// the string; sorting the entries by key brings connected determinants next
// to each other. The work is proportional to the number of connected pairs,
// and no spin or time-reversal structure of the orbitals is needed.

#ifndef ZSHCI_SRC_CONNECT_H_
#define ZSHCI_SRC_CONNECT_H_

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "comm.h"
#include "det.h"
#include "hamiltonian.h"
#include "integrals.h"
#include "parallel.h"
#include "sparse_h.h"

namespace zshci {

struct ConnectOptions {
  // Elements with |H(i,j)| <= threshold are not stored. A negative value
  // keeps every connected pair, which the density matrices need.
  double threshold = -1.0;
  // The keys are processed in this many passes; the transient memory of the
  // entry list is divided by the same factor.
  int num_batches = 1;
};

namespace internal {

struct BucketEntry {
  uint64_t key;
  DetIndex index;

  friend bool operator<(const BucketEntry& a, const BucketEntry& b) {
    return a.key != b.key ? a.key < b.key : a.index < b.index;
  }
};

// Key of a string obtained by removing `num_removed` electrons. Salting with
// the number of removed electrons keeps N-1 and N-2 strings in separate
// buckets.
template <int W>
uint64_t StringKey(const Det<W>& string, int num_removed) {
  return Mix64(string.Hash() + static_cast<uint64_t>(num_removed));
}

// Which strings one call of MakeEntries covers: the keys are split first
// into batches (to bound memory) and then among the ranks.
struct KeySlice {
  uint64_t batch;
  uint64_t num_batches;
  uint64_t rank;
  uint64_t num_ranks;

  bool Contains(uint64_t key) const {
    return key % num_batches == batch &&
           (key / num_batches) % num_ranks == rank;
  }
};

// Appends to `chunks` (one per thread) the entries of the N-1 and N-2
// strings of dets[begin..end) whose key is accepted by `accept`.
template <int W, class Accept>
void AppendEntries(Span<Det<W>> dets, size_t begin, size_t end,
                   const Accept& accept,
                   std::vector<std::vector<BucketEntry>>* chunks) {
  const size_t first_chunk = chunks->size();
  chunks->resize(first_chunk + omp_get_max_threads());
#pragma omp parallel
  {
    std::vector<BucketEntry>& local =
        (*chunks)[first_chunk + omp_get_thread_num()];
    std::vector<int> occ;
#pragma omp for schedule(static) nowait
    for (size_t d = begin; d < end; ++d) {
      occ.clear();
      dets[d].AppendOccupied(&occ);
      const DetIndex index = static_cast<DetIndex>(d);
      for (size_t m = 0; m < occ.size(); ++m) {
        Det<W> minus_one = dets[d];
        minus_one.Clear(occ[m]);
        const uint64_t key1 = StringKey(minus_one, 1);
        if (accept(key1)) local.push_back({key1, index});
        for (size_t n = 0; n < m; ++n) {
          Det<W> minus_two = minus_one;
          minus_two.Clear(occ[n]);
          const uint64_t key2 = StringKey(minus_two, 2);
          if (accept(key2)) local.push_back({key2, index});
        }
      }
    }
  }
}

// The part, out of `num_parts` equal ranges of the 64-bit key space, that
// holds `key`. Monotonic in the key.
inline size_t KeyRange(uint64_t key, size_t num_parts) {
  return static_cast<size_t>(
      (static_cast<unsigned __int128>(key) * num_parts) >> 64);
}

// Returns the entries in `slice` of the strings that can connect a
// determinant to one of dets[first_new..], split into ranges of the key and
// sorted within each range. Only buckets holding a new determinant matter,
// so the old determinants contribute only the strings they share with a new
// one.
template <int W>
Partitioned<BucketEntry> MakeEntries(Span<Det<W>> dets,
                                     size_t first_new, const KeySlice& slice) {
  // More parts than threads, so that uneven parts even out.
  const size_t num_parts = 8 * static_cast<size_t>(omp_get_max_threads());
  std::vector<std::vector<BucketEntry>> chunks;
  AppendEntries(dets, first_new, dets.size(),
                [&](uint64_t key) { return slice.Contains(key); }, &chunks);

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
    AppendEntries(dets, 0, first_new,
                  [&](uint64_t key) {
                    if (!slice.Contains(key)) return false;
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

// Calls visit(i, j) once for every connected pair i < j with j >= first_new.
// `visit` runs concurrently from several threads.
template <int W, class Visitor>
void ForEachConnectedPair(Span<Det<W>> dets, size_t first_new,
                          const Partitioned<BucketEntry>& entries,
                          const Visitor& visit) {
  const long num_parts = static_cast<long>(entries.num_parts());
#pragma omp parallel
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
        for (size_t a = 0; a < bucket.size(); ++a) {
          const DetIndex i = bucket[a];
          for (size_t b = a + 1; b < bucket.size(); ++b) {
            const DetIndex j = bucket[b];
            if (j < first_new) continue;
            const int num_bits = dets[i].NumDifferentBits(dets[j]);
            if (num_bits != 2 && num_bits != 4) continue;
            // The shared string must be the one this bucket stands for;
            // otherwise the pair is here through a hash collision and is
            // visited in its own bucket.
            if (StringKey(dets[i].And(dets[j]), num_bits / 2) != key) continue;
            visit(i, j);
          }
        }
      }
    }
  }
}

}  // namespace internal

// Builds the block of H(i, j), i < j, with j >= first_new, for the
// determinants `dets`. Determinants before `first_new` are assumed to be
// connected among themselves by earlier blocks.
//
// With several ranks, each rank handles the strings whose key it owns and
// returns the elements found there. Every connected pair shares exactly one
// string, so the blocks of all ranks together hold each element once; no
// communication is needed.
template <int W>
CsrBlock BuildBlock(Span<Det<W>> dets, size_t first_new,
                    const Integrals& ints, const ConnectOptions& options,
                    const Comm& comm = Comm()) {
  using internal::BucketEntry;
  const size_t n = dets.size();
  if (n > static_cast<size_t>(UINT32_MAX)) {
    throw std::length_error("too many determinants for a 32-bit index");
  }
  const int num_batches = std::max(1, options.num_batches);
  const bool keep_all = options.threshold < 0.0;

  // With a threshold the element has to be evaluated to know whether it is
  // stored, so the counting pass evaluates it too.
  auto stored = [&](DetIndex i, DetIndex j) {
    return keep_all ||
           std::abs(Hij(dets[i], dets[j], ints)) > options.threshold;
  };

  auto slice = [&](int batch) {
    return internal::KeySlice{static_cast<uint64_t>(batch),
                              static_cast<uint64_t>(num_batches),
                              static_cast<uint64_t>(comm.rank()),
                              static_cast<uint64_t>(comm.size())};
  };

  // Pass 1: number of stored elements per row. This array and the cursors
  // below are the only ones of length n; they are released before returning.
  std::vector<uint64_t> count(n, 0);
  Partitioned<BucketEntry> cached;
  for (int batch = 0; batch < num_batches; ++batch) {
    Partitioned<BucketEntry> entries =
        internal::MakeEntries(dets, first_new, slice(batch));
    internal::ForEachConnectedPair(dets, first_new, entries,
                                   [&](DetIndex i, DetIndex j) {
                                     if (!stored(i, j)) return;
#pragma omp atomic
                                     ++count[i];
                                   });
    if (num_batches == 1) cached = std::move(entries);
  }

  // The rows that hold elements, and where each one starts. `count` turns
  // into the write cursor of every row.
  CsrBlock block;
  block.dim = n;
  block.first_col = first_new;
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
  block.val = RawArray<Complex>(block.row_ptr[num_stored_rows]);

  // Pass 2: fill. Each row has its own cursor, advanced atomically.
  for (int batch = 0; batch < num_batches; ++batch) {
    const Partitioned<BucketEntry> entries =
        num_batches == 1 ? std::move(cached)
                         : internal::MakeEntries(dets, first_new, slice(batch));
    internal::ForEachConnectedPair(
        dets, first_new, entries, [&](DetIndex i, DetIndex j) {
          const Complex value = Hij(dets[i], dets[j], ints);
          if (!keep_all && std::abs(value) <= options.threshold) return;
          uint64_t slot;
#pragma omp atomic capture
          slot = cursor[i]++;
          block.col[slot] = j;
          block.val[slot] = value;
        });
  }

  // Sort every row by column.
#pragma omp parallel
  {
    std::vector<std::pair<DetIndex, Complex>> row;
#pragma omp for schedule(dynamic, 256)
    for (size_t r = 0; r < num_stored_rows; ++r) {
      const uint64_t begin = block.row_ptr[r];
      const uint64_t end = block.row_ptr[r + 1];
      row.clear();
      for (uint64_t k = begin; k < end; ++k) {
        row.emplace_back(block.col[k], block.val[k]);
      }
      std::sort(row.begin(), row.end(),
                [](const std::pair<DetIndex, Complex>& a,
                   const std::pair<DetIndex, Complex>& b) {
                  return a.first < b.first;
                });
      for (uint64_t k = begin; k < end; ++k) {
        block.col[k] = row[k - begin].first;
        block.val[k] = row[k - begin].second;
      }
    }
  }
  return block;
}

// Reference implementation: tests every pair of determinants. Quadratic; used
// only to check BuildBlock on small cases.
template <int W>
CsrBlock BuildBlockBruteForce(Span<Det<W>> dets, size_t first_new,
                              const Integrals& ints,
                              const ConnectOptions& options) {
  const size_t n = dets.size();
  CsrBlock block;
  block.dim = n;
  block.first_col = first_new;
  std::vector<DetIndex> row;
  std::vector<DetIndex> col;
  std::vector<Complex> val;
  block.row_ptr.assign(1, 0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = std::max(i + 1, first_new); j < n; ++j) {
      const int num_bits = dets[i].NumDifferentBits(dets[j]);
      if (num_bits != 2 && num_bits != 4) continue;
      const Complex value = Hij(dets[i], dets[j], ints);
      if (options.threshold >= 0.0 && std::abs(value) <= options.threshold) {
        continue;
      }
      col.push_back(static_cast<DetIndex>(j));
      val.push_back(value);
    }
    if (col.size() > block.row_ptr.back()) {
      row.push_back(static_cast<DetIndex>(i));
      block.row_ptr.push_back(col.size());
    }
  }
  block.row = RawArray<DetIndex>(row.size());
  block.col = RawArray<DetIndex>(col.size());
  block.val = RawArray<Complex>(val.size());
  std::copy(row.begin(), row.end(), block.row.begin());
  std::copy(col.begin(), col.end(), block.col.begin());
  std::copy(val.begin(), val.end(), block.val.begin());
  return block;
}

}  // namespace zshci

#endif  // ZSHCI_SRC_CONNECT_H_
