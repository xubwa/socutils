// Building blocks for the threaded phases: an array that is not touched when
// allocated, and a parallel partition of items into buckets.
//
// Both exist for the same reason. The large arrays of this program (bucket
// entries, perturbative terms, Hamiltonian elements) are filled and then read
// by many threads. If one thread allocates and zeroes such an array, as
// std::vector does, that step is serial and, on a machine with several memory
// domains, places every page next to that one thread.

#ifndef ZSHCI_SRC_PARALLEL_H_
#define ZSHCI_SRC_PARALLEL_H_

#include <omp.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace zshci {

// A fixed-size array of trivially copyable elements whose memory is left
// uninitialized, so that pages are first touched by the threads that fill
// them.
template <class T>
class RawArray {
 public:
  RawArray() = default;
  explicit RawArray(size_t size) : size_(size) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "RawArray holds only flat data");
    if (size_ == 0) return;
    data_ = static_cast<T*>(std::malloc(size_ * sizeof(T)));
    if (data_ == nullptr) throw std::bad_alloc();
  }
  ~RawArray() { std::free(data_); }

  RawArray(RawArray&& other) noexcept { Swap(&other); }
  RawArray& operator=(RawArray&& other) noexcept {
    if (this != &other) {
      RawArray empty;
      Swap(&empty);
      Swap(&other);
    }
    return *this;
  }
  RawArray(const RawArray&) = delete;
  RawArray& operator=(const RawArray&) = delete;

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T* begin() { return data_; }
  T* end() { return data_ + size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  void Swap(RawArray* other) {
    std::swap(data_, other->data_);
    std::swap(size_, other->size_);
  }

  T* data_ = nullptr;
  size_t size_ = 0;
};

// Items grouped by part: part p is items[start[p] .. start[p + 1]).
template <class T>
struct Partitioned {
  RawArray<T> items;
  std::vector<size_t> start;

  size_t num_parts() const { return start.size() - 1; }
  T* begin(size_t part) { return items.data() + start[part]; }
  T* end(size_t part) { return items.data() + start[part + 1]; }
  size_t size(size_t part) const { return start[part + 1] - start[part]; }
};

// A read-only view of consecutive items owned by someone else: a vector, a
// RawArray, or memory shared by the ranks of a node. Cheap to copy; pass by
// value.
template <class T>
class Span {
 public:
  Span() = default;
  Span(const T* data, size_t size) : data_(data), size_(size) {}
  Span(const std::vector<T>& items)  // NOLINT: implicit on purpose
      : data_(items.data()), size_(items.size()) {}

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const T* data() const { return data_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  const T* data_ = nullptr;
  size_t size_ = 0;
};

// Deduces the element type; needed where a function template takes a Span
// and is handed a vector.
template <class T>
Span<T> MakeSpan(const std::vector<T>& items) {
  return Span<T>(items);
}

// Distributes the items of several spans into `num_parts` parts,
// part_of(item) giving the part of an item. Both passes run in parallel over
// the spans.
template <class T, class PartOf>
Partitioned<T> PartitionSpans(const std::vector<Span<T>>& spans,
                              size_t num_parts, const PartOf& part_of) {
  const long num_chunks = static_cast<long>(spans.size());
  // offset[c][p]: where span c writes its items of part p.
  std::vector<std::vector<size_t>> offset(
      num_chunks, std::vector<size_t>(num_parts, 0));
#pragma omp parallel for schedule(dynamic, 1)
  for (long c = 0; c < num_chunks; ++c) {
    for (const T& item : spans[c]) ++offset[c][part_of(item)];
  }

  Partitioned<T> result;
  result.start.assign(num_parts + 1, 0);
  size_t total = 0;
  for (size_t p = 0; p < num_parts; ++p) {
    result.start[p] = total;
    for (long c = 0; c < num_chunks; ++c) {
      const size_t count = offset[c][p];
      offset[c][p] = total;
      total += count;
    }
  }
  result.start[num_parts] = total;
  result.items = RawArray<T>(total);

#pragma omp parallel for schedule(dynamic, 1)
  for (long c = 0; c < num_chunks; ++c) {
    for (const T& item : spans[c]) {
      result.items[offset[c][part_of(item)]++] = item;
    }
  }
  return result;
}

// The same for chunks held in vectors (typically one per generating thread),
// which are emptied.
template <class T, class PartOf>
Partitioned<T> PartitionChunks(std::vector<std::vector<T>>* chunks,
                               size_t num_parts, const PartOf& part_of) {
  std::vector<Span<T>> spans;
  for (const std::vector<T>& chunk : *chunks) spans.emplace_back(chunk);
  Partitioned<T> result = PartitionSpans(spans, num_parts, part_of);
  for (std::vector<T>& chunk : *chunks) std::vector<T>().swap(chunk);
  return result;
}

// The same for one array, cut into one slice per thread.
template <class T, class PartOf>
Partitioned<T> PartitionArray(const RawArray<T>& items, size_t num_parts,
                              const PartOf& part_of) {
  const size_t num_slices = omp_get_max_threads();
  std::vector<Span<T>> spans;
  for (size_t s = 0; s < num_slices; ++s) {
    const size_t begin = items.size() * s / num_slices;
    const size_t end = items.size() * (s + 1) / num_slices;
    spans.emplace_back(items.data() + begin, end - begin);
  }
  return PartitionSpans(spans, num_parts, part_of);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_PARALLEL_H_
