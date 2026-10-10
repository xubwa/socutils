#include "comm.h"

#include <cstdlib>
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

#ifdef ZSHCI_USE_MPI
#include <mpi.h>
#endif

namespace zshci {

#ifdef ZSHCI_USE_MPI

namespace {

// MPI counts are ints; larger arrays are handled in pieces of this many
// elements.
constexpr uint64_t kChunk = 1ULL << 30;

// The ranks that share this rank's node, and the first rank of every node
// (MPI_COMM_NULL on the other ranks). Set by Comm::Init.
MPI_Comm g_node_comm = MPI_COMM_NULL;
MPI_Comm g_leader_comm = MPI_COMM_NULL;

void Check(int error, const char* what) {
  if (error != MPI_SUCCESS) {
    throw std::runtime_error(std::string("MPI call failed: ") + what);
  }
}

// A datatype of `element_size` bytes, so that counts are in elements.
class ElementType {
 public:
  explicit ElementType(size_t element_size) {
    Check(MPI_Type_contiguous(static_cast<int>(element_size), MPI_BYTE,
                              &type_),
          "MPI_Type_contiguous");
    Check(MPI_Type_commit(&type_), "MPI_Type_commit");
  }
  ~ElementType() { MPI_Type_free(&type_); }
  ElementType(const ElementType&) = delete;
  ElementType& operator=(const ElementType&) = delete;
  MPI_Datatype get() const { return type_; }

 private:
  MPI_Datatype type_;
};

std::vector<int> ToIntCounts(const std::vector<uint64_t>& counts,
                             std::vector<int>* displacements) {
  std::vector<int> out(counts.size());
  displacements->assign(counts.size(), 0);
  uint64_t offset = 0;
  for (size_t r = 0; r < counts.size(); ++r) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        counts[r] > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::length_error(
          "a single exchange exceeds 2^31 elements; use more batches");
    }
    out[r] = static_cast<int>(counts[r]);
    (*displacements)[r] = static_cast<int>(offset);
    offset += counts[r];
  }
  return out;
}

}  // namespace

Comm Comm::Init(int* argc, char*** argv) {
  int provided = 0;
  Check(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided),
        "MPI_Init_thread");
  Comm comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm.rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &comm.size_);
  // Ranks keep their world order inside both communicators, so the world
  // root is the leader of its node and rank 0 among the leaders.
  Check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm.rank_,
                            MPI_INFO_NULL, &g_node_comm),
        "MPI_Comm_split_type");
  MPI_Comm_rank(g_node_comm, &comm.node_rank_);
  Check(MPI_Comm_split(MPI_COMM_WORLD,
                       comm.node_rank_ == 0 ? 0 : MPI_UNDEFINED, comm.rank_,
                       &g_leader_comm),
        "MPI_Comm_split");
  return comm;
}

void Comm::Finalize() {
  if (g_leader_comm != MPI_COMM_NULL) MPI_Comm_free(&g_leader_comm);
  if (g_node_comm != MPI_COMM_NULL) MPI_Comm_free(&g_node_comm);
  MPI_Finalize();
}

namespace {

// What a SharedArray owns: an MPI window, or plain memory when the Comm is a
// serial one (size 1, not obtained from Init).
struct SharedBlock {
  bool is_window = false;
  MPI_Win window = MPI_WIN_NULL;
  void* memory = nullptr;
};

}  // namespace

void* Comm::AllocateShared(size_t bytes, void** base) const {
  SharedBlock* block = new SharedBlock;
  if (size_ == 1) {
    block->memory = std::malloc(bytes == 0 ? 1 : bytes);
    if (block->memory == nullptr) throw std::bad_alloc();
    *base = block->memory;
    return block;
  }
  // The leader provides the memory; the others map it.
  block->is_window = true;
  Check(MPI_Win_allocate_shared(
            is_node_leader() ? static_cast<MPI_Aint>(bytes) : 0, 1,
            MPI_INFO_NULL, g_node_comm, base, &block->window),
        "MPI_Win_allocate_shared");
  if (!is_node_leader()) {
    MPI_Aint size = 0;
    int displacement_unit = 0;
    Check(MPI_Win_shared_query(block->window, 0, &size, &displacement_unit,
                               base),
          "MPI_Win_shared_query");
  }
  return block;
}

void Comm::FreeShared(void* handle) {
  SharedBlock* block = static_cast<SharedBlock*>(handle);
  if (block->is_window) {
    MPI_Win_free(&block->window);
  } else {
    std::free(block->memory);
  }
  delete block;
}

void Comm::ShareFromRoot(void* base, uint64_t count,
                         size_t element_size) const {
  if (size_ == 1) return;
  if (is_node_leader() && count > 0) {
    int num_leaders = 1;
    MPI_Comm_size(g_leader_comm, &num_leaders);
    if (num_leaders > 1) {
      const ElementType type(element_size);
      char* bytes = static_cast<char*>(base);
      for (uint64_t done = 0; done < count; done += kChunk) {
        const int piece = static_cast<int>(std::min(kChunk, count - done));
        Check(MPI_Bcast(bytes + done * element_size, piece, type.get(), 0,
                        g_leader_comm),
              "MPI_Bcast");
      }
    }
  }
  // The other ranks of a node must not read before their leader has the data.
  Check(MPI_Barrier(g_node_comm), "MPI_Barrier");
}

void Comm::Abort(int code) const {
  MPI_Abort(MPI_COMM_WORLD, code);
  std::abort();
}

void Comm::BcastRaw(void* data, uint64_t count, size_t element_size) const {
  if (size_ == 1 || count == 0) return;
  const ElementType type(element_size);
  char* bytes = static_cast<char*>(data);
  for (uint64_t done = 0; done < count; done += kChunk) {
    const int piece = static_cast<int>(std::min(kChunk, count - done));
    Check(MPI_Bcast(bytes + done * element_size, piece, type.get(), 0,
                    MPI_COMM_WORLD),
          "MPI_Bcast");
  }
}

void Comm::BcastString(std::string* text) const {
  std::vector<char> buffer(text->begin(), text->end());
  Bcast(&buffer);
  text->assign(buffer.begin(), buffer.end());
}

void Comm::Sum(double* data, size_t count) const {
  if (size_ == 1) return;
  for (uint64_t done = 0; done < count; done += kChunk) {
    const int piece = static_cast<int>(std::min<uint64_t>(kChunk, count - done));
    Check(MPI_Allreduce(MPI_IN_PLACE, data + done, piece, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD),
          "MPI_Allreduce");
  }
}

void Comm::SumToRoot(double* data, size_t count) const {
  if (size_ == 1) return;
  for (uint64_t done = 0; done < count; done += kChunk) {
    const int piece = static_cast<int>(std::min<uint64_t>(kChunk, count - done));
    if (rank_ == 0) {
      Check(MPI_Reduce(MPI_IN_PLACE, data + done, piece, MPI_DOUBLE, MPI_SUM,
                       0, MPI_COMM_WORLD),
            "MPI_Reduce");
    } else {
      Check(MPI_Reduce(data + done, nullptr, piece, MPI_DOUBLE, MPI_SUM, 0,
                       MPI_COMM_WORLD),
            "MPI_Reduce");
    }
  }
}

std::vector<uint64_t> Comm::ExchangeCounts(
    const std::vector<uint64_t>& send_counts) const {
  std::vector<uint64_t> receive_counts(size_);
  Check(MPI_Alltoall(send_counts.data(), 1, MPI_UINT64_T,
                     receive_counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD),
        "MPI_Alltoall");
  return receive_counts;
}

void Comm::ExchangeRaw(const void* send,
                       const std::vector<uint64_t>& send_counts, void* receive,
                       const std::vector<uint64_t>& receive_counts,
                       size_t element_size) const {
  const ElementType type(element_size);
  std::vector<int> send_displacements, receive_displacements;
  const std::vector<int> sends = ToIntCounts(send_counts, &send_displacements);
  const std::vector<int> receives =
      ToIntCounts(receive_counts, &receive_displacements);
  Check(MPI_Alltoallv(send, sends.data(), send_displacements.data(),
                      type.get(), receive, receives.data(),
                      receive_displacements.data(), type.get(),
                      MPI_COMM_WORLD),
        "MPI_Alltoallv");
}

std::vector<uint64_t> Comm::GatherCounts(uint64_t local_count) const {
  std::vector<uint64_t> counts(size_);
  Check(MPI_Allgather(&local_count, 1, MPI_UINT64_T, counts.data(), 1,
                      MPI_UINT64_T, MPI_COMM_WORLD),
        "MPI_Allgather");
  return counts;
}

void Comm::GatherRaw(const void* local, uint64_t local_count, void* all,
                     const std::vector<uint64_t>& counts,
                     size_t element_size) const {
  const ElementType type(element_size);
  std::vector<int> displacements;
  const std::vector<int> int_counts = ToIntCounts(counts, &displacements);
  Check(MPI_Allgatherv(local, static_cast<int>(local_count), type.get(), all,
                       int_counts.data(), displacements.data(), type.get(),
                       MPI_COMM_WORLD),
        "MPI_Allgatherv");
}

#else  // !ZSHCI_USE_MPI

Comm Comm::Init(int* /*argc*/, char*** /*argv*/) { return Comm(); }
void Comm::Finalize() {}
void* Comm::AllocateShared(size_t bytes, void** base) const {
  *base = std::malloc(bytes == 0 ? 1 : bytes);
  if (*base == nullptr) throw std::bad_alloc();
  return *base;
}
void Comm::FreeShared(void* handle) { std::free(handle); }
void Comm::ShareFromRoot(void*, uint64_t, size_t) const {}
void Comm::Abort(int code) const { std::exit(code); }
void Comm::BcastRaw(void*, uint64_t, size_t) const {}
void Comm::BcastString(std::string*) const {}
void Comm::Sum(double*, size_t) const {}
void Comm::SumToRoot(double*, size_t) const {}
std::vector<uint64_t> Comm::ExchangeCounts(
    const std::vector<uint64_t>& send_counts) const {
  return send_counts;
}
void Comm::ExchangeRaw(const void*, const std::vector<uint64_t>&, void*,
                       const std::vector<uint64_t>&, size_t) const {}
std::vector<uint64_t> Comm::GatherCounts(uint64_t local_count) const {
  return {local_count};
}
void Comm::GatherRaw(const void*, uint64_t, void*,
                     const std::vector<uint64_t>&, size_t) const {}

#endif  // ZSHCI_USE_MPI

}  // namespace zshci
