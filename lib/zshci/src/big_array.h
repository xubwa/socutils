// Arrays of many gigabytes for the vectors of the "fci" mode.
//
// The memory is mapped directly (not through malloc), marked for transparent
// huge pages or against them, and touched first by all threads with the static
// schedule that the vector operations use, so that on a machine with several
// memory controllers (sockets) each thread's part lies near it. Nothing is
// zeroed beyond what the mapping gives.

#ifndef ZSHCI_SRC_BIG_ARRAY_H_
#define ZSHCI_SRC_BIG_ARRAY_H_

#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <utility>

namespace zshci {

template <typename T>
class BigArray {
 public:
  BigArray() = default;
  BigArray(size_t size, bool huge_pages) : size_(size) {
    if (size == 0) return;
    bytes_ = size * sizeof(T);
    void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::bad_alloc();
    madvise(p, bytes_, huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    data_ = static_cast<T*>(p);
    const long n = static_cast<long>(size);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) data_[i] = T(0.0);
  }
  ~BigArray() { Release(); }
  BigArray(BigArray&& other) noexcept { *this = std::move(other); }
  BigArray& operator=(BigArray&& other) noexcept {
    if (this != &other) {
      Release();
      data_ = other.data_;
      size_ = other.size_;
      bytes_ = other.bytes_;
      other.data_ = nullptr;
      other.size_ = other.bytes_ = 0;
    }
    return *this;
  }
  BigArray(const BigArray&) = delete;
  BigArray& operator=(const BigArray&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  void Release() {
    if (data_ != nullptr) munmap(data_, bytes_);
    data_ = nullptr;
  }
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t bytes_ = 0;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_BIG_ARRAY_H_
