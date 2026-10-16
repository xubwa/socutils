// Thin wrapper around MPI.
//
// A default-constructed Comm is serial: one rank, and every operation
// degenerates to a local copy. Comm::Init returns the world communicator when
// the program is built with ZSHCI_USE_MPI, and a serial one otherwise, so the
// rest of the code never touches MPI directly and also builds without it.
//
// Only flat arrays of trivially copyable elements travel between ranks; this
// is enforced at compile time. Counts are in elements, not bytes, so single
// messages may exceed 2 GB.
//
// Read-mostly data that every rank needs (integrals, heat-bath tables, the
// determinants, the CI vectors) lives in a SharedArray: one copy per node,
// mapped by all ranks of that node.

#ifndef ZSHCI_SRC_COMM_H_
#define ZSHCI_SRC_COMM_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "parallel.h"

namespace zshci {

class Comm {
 public:
  Comm() = default;

  // Starts MPI (if built with it) and returns the world communicator.
  static Comm Init(int* argc, char*** argv);
  // Shuts MPI down; no Comm may be used afterwards.
  static void Finalize();

  int rank() const { return rank_; }
  int size() const { return size_; }
  bool is_root() const { return rank_ == 0; }
  // Whether this rank is the first of the ranks on its node.
  bool is_node_leader() const { return node_rank_ == 0; }

  // Terminates all ranks with the given exit code.
  [[noreturn]] void Abort(int code) const;

  // Root's content replaces everyone else's.
  template <class T>
  void Bcast(std::vector<T>* data) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only flat data may be broadcast");
    uint64_t count = data->size();
    BcastRaw(&count, 1, sizeof(count));
    data->resize(count);
    BcastRaw(data->data(), count, sizeof(T));
  }
  template <class T>
  void BcastValue(T* value) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only flat data may be broadcast");
    BcastRaw(value, 1, sizeof(T));
  }
  void BcastString(std::string* text) const;
  // `count` elements of `element_size` bytes from rank `root` to all.
  void BcastFrom(void* data, uint64_t count, size_t element_size,
                 int root) const;
  // Sends `send_count` elements to rank `dest` while receiving
  // `receive_count` from rank `source` (dest and source may be this rank
  // only when both counts are zero or the ranks are one).
  void SendReceive(const void* send, uint64_t send_count, int dest,
                   void* receive, uint64_t receive_count, int source,
                   size_t element_size) const;
  void Barrier() const;

  // Element-wise sum over all ranks, in place on every rank.
  void Sum(double* data, size_t count) const;
  // Element-wise sum over all ranks, delivered to the root only; the buffers
  // of the other ranks are left unspecified.
  void SumToRoot(double* data, size_t count) const;

  // Sends outgoing[r] to rank r and returns everything received, ordered by
  // source rank. `outgoing` must have size() entries.
  template <class T>
  std::vector<T> Exchange(const std::vector<std::vector<T>>& outgoing) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only flat data may be exchanged");
    if (size_ == 1) return outgoing[0];
    std::vector<uint64_t> send_counts(size_);
    uint64_t total = 0;
    for (int r = 0; r < size_; ++r) {
      send_counts[r] = outgoing[r].size();
      total += send_counts[r];
    }
    std::vector<T> send_buffer;
    send_buffer.reserve(total);
    for (int r = 0; r < size_; ++r) {
      send_buffer.insert(send_buffer.end(), outgoing[r].begin(),
                         outgoing[r].end());
    }
    std::vector<uint64_t> receive_counts = ExchangeCounts(send_counts);
    uint64_t received = 0;
    for (uint64_t c : receive_counts) received += c;
    std::vector<T> receive_buffer(received);
    ExchangeRaw(send_buffer.data(), send_counts, receive_buffer.data(),
                receive_counts, sizeof(T));
    return receive_buffer;
  }

  // The same for items already grouped by destination rank: part r of
  // `outgoing` goes to rank r. `outgoing` is consumed.
  template <class T>
  RawArray<T> ExchangeParts(Partitioned<T>* outgoing) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only flat data may be exchanged");
    if (size_ == 1) return std::move(outgoing->items);
    std::vector<uint64_t> send_counts(size_);
    for (int r = 0; r < size_; ++r) send_counts[r] = outgoing->size(r);
    const std::vector<uint64_t> receive_counts = ExchangeCounts(send_counts);
    uint64_t received = 0;
    for (uint64_t c : receive_counts) received += c;
    RawArray<T> receive_buffer(received);
    ExchangeRaw(outgoing->items.data(), send_counts, receive_buffer.data(),
                receive_counts, sizeof(T));
    outgoing->items = RawArray<T>();
    return receive_buffer;
  }

  // Returns, on every rank, the concatenation of all ranks' `local` in rank
  // order.
  template <class T>
  std::vector<T> GatherAll(const std::vector<T>& local) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "only flat data may be gathered");
    if (size_ == 1) return local;
    std::vector<uint64_t> counts = GatherCounts(local.size());
    uint64_t total = 0;
    for (uint64_t c : counts) total += c;
    std::vector<T> all(total);
    GatherRaw(local.data(), local.size(), all.data(), counts, sizeof(T));
    return all;
  }

 private:
  template <class T>
  friend class SharedArray;

  // Memory shared by the ranks of a node. Collective. Returns an opaque
  // handle for FreeShared and sets *base to this rank's mapping.
  void* AllocateShared(size_t bytes, void** base) const;
  static void FreeShared(void* handle);
  // Copies the root's shared memory to the other nodes and returns once every
  // rank may read it. Collective.
  void ShareFromRoot(void* base, uint64_t count, size_t element_size) const;

  void BcastRaw(void* data, uint64_t count, size_t element_size) const;
  std::vector<uint64_t> ExchangeCounts(
      const std::vector<uint64_t>& send_counts) const;
  void ExchangeRaw(const void* send, const std::vector<uint64_t>& send_counts,
                   void* receive, const std::vector<uint64_t>& receive_counts,
                   size_t element_size) const;
  std::vector<uint64_t> GatherCounts(uint64_t local_count) const;
  void GatherRaw(const void* local, uint64_t local_count, void* all,
                 const std::vector<uint64_t>& counts,
                 size_t element_size) const;

  int rank_ = 0;
  int size_ = 1;
  int node_rank_ = 0;
};

// A local array that the other ranks read without its owner taking part
// (one-sided communication): Get starts a copy of bytes of rank `owner`'s
// array into this rank's memory, Wait completes the copies from that rank.
// Construction and destruction are collective.
class RemoteArray {
 public:
  RemoteArray(const Comm& comm, const void* local, uint64_t bytes);
  ~RemoteArray();
  RemoteArray(const RemoteArray&) = delete;
  RemoteArray& operator=(const RemoteArray&) = delete;
  void Get(void* destination, int owner, uint64_t offset, uint64_t bytes) const;
  void Wait(int owner) const;

 private:
  const Comm& comm_;
  const void* local_;
  void* window_ = nullptr;  // MPI_Win, allocated
};

// An array of trivially copyable elements held once per node and mapped by
// every rank of that node. Construction and ShareFromRoot are collective:
// all ranks must call them, with the same size. The content is uninitialized
// until a node leader writes it; the usual pattern is that the root fills
// the array and calls ShareFromRoot, after which all ranks may read.
template <class T>
class SharedArray {
 public:
  SharedArray() = default;
  SharedArray(size_t size, const Comm& comm) : size_(size) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "SharedArray holds only flat data");
    void* base = nullptr;
    handle_ = comm.AllocateShared(size * sizeof(T), &base);
    data_ = static_cast<T*>(base);
  }
  ~SharedArray() {
    if (handle_ != nullptr) Comm::FreeShared(handle_);
  }

  SharedArray(SharedArray&& other) noexcept { Swap(&other); }
  SharedArray& operator=(SharedArray&& other) noexcept {
    if (this != &other) {
      SharedArray empty;
      Swap(&empty);
      Swap(&other);
    }
    return *this;
  }
  SharedArray(const SharedArray&) = delete;
  SharedArray& operator=(const SharedArray&) = delete;

  void ShareFromRoot(const Comm& comm) {
    comm.ShareFromRoot(data_, size_, sizeof(T));
  }

  // Collective: an array with the content of the root's `items` on every
  // node. `items` is ignored on the other ranks.
  static SharedArray FromRoot(const std::vector<T>& items, const Comm& comm) {
    uint64_t size = items.size();
    comm.BcastValue(&size);
    SharedArray array(size, comm);
    if (comm.is_root()) std::copy(items.begin(), items.end(), array.data_);
    array.ShareFromRoot(comm);
    return array;
  }

  size_t size() const { return size_; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }
  Span<T> span() const { return Span<T>(data_, size_); }
  // The view of `count` elements starting at `first`.
  Span<T> span(size_t first, size_t count) const {
    return Span<T>(data_ + first, count);
  }

 private:
  void Swap(SharedArray* other) {
    std::swap(data_, other->data_);
    std::swap(size_, other->size_);
    std::swap(handle_, other->handle_);
  }

  T* data_ = nullptr;
  size_t size_ = 0;
  void* handle_ = nullptr;
};

}  // namespace zshci

#endif  // ZSHCI_SRC_COMM_H_
