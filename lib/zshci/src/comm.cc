#include "comm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
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
  // The leader of every rank's node; the nodes are contiguous if each rank
  // either leads its node or has the leader of the rank before it.
  int leader = comm.rank_;
  MPI_Bcast(&leader, 1, MPI_INT, 0, g_node_comm);
  std::vector<int> leaders(comm.size_);
  Check(MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT,
                      MPI_COMM_WORLD),
        "MPI_Allgather");
  comm.node_first_.clear();
  for (int r = 0; r < comm.size_; ++r) {
    if (leaders[r] == r) {
      comm.node_first_.push_back(r);
    } else if (r == 0 || leaders[r] != leaders[r - 1]) {
      comm.nodes_contiguous_ = false;
    }
  }
  comm.node_first_.push_back(comm.size_);
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
    // Freeing a window is collective. While an exception unwinds the stack
    // of this rank the others are not doing the same, and waiting for them
    // would keep the error from ever reaching main(), which aborts all
    // ranks. Leave the window to the abort.
    if (std::uncaught_exceptions() == 0) MPI_Win_free(&block->window);
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

void Comm::BcastFrom(void* data, uint64_t count, size_t element_size,
                     int root) const {
  if (size_ == 1 || count == 0) return;
  const ElementType type(element_size);
  char* bytes = static_cast<char*>(data);
  for (uint64_t done = 0; done < count; done += kChunk) {
    const int piece = static_cast<int>(std::min(kChunk, count - done));
    Check(MPI_Bcast(bytes + done * element_size, piece, type.get(), root,
                    MPI_COMM_WORLD),
          "MPI_Bcast");
  }
}

void Comm::SendReceive(const void* send, uint64_t send_count, int dest,
                       void* receive, uint64_t receive_count, int source,
                       size_t element_size) const {
  const ElementType type(element_size);
  const char* out = static_cast<const char*>(send);
  char* in = static_cast<char*>(receive);
  // Both sides know both counts, so they agree on the pieces.
  uint64_t sent = 0, received = 0;
  while (sent < send_count || received < receive_count) {
    const int s = static_cast<int>(std::min(kChunk, send_count - sent));
    const int r = static_cast<int>(std::min(kChunk, receive_count - received));
    Check(MPI_Sendrecv(out + sent * element_size, s, type.get(),
                       s > 0 ? dest : MPI_PROC_NULL, 7,
                       in + received * element_size, r, type.get(),
                       r > 0 ? source : MPI_PROC_NULL, 7, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE),
          "MPI_Sendrecv");
    sent += s;
    received += r;
  }
}

void Comm::Barrier() const {
  if (size_ > 1) Check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
}

void Comm::Progress() {
  int flag = 0;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag,
             MPI_STATUS_IGNORE);
}

RemoteArray::RemoteArray(const Comm& comm, const void* local, uint64_t bytes)
    : comm_(comm), local_(local) {
  MPI_Win* window = new MPI_Win;
  Check(MPI_Win_create(const_cast<void*>(local), static_cast<MPI_Aint>(bytes),
                       1, MPI_INFO_NULL, MPI_COMM_WORLD, window),
        "MPI_Win_create");
  Check(MPI_Win_lock_all(MPI_MODE_NOCHECK, *window), "MPI_Win_lock_all");
  window_ = window;
}

RemoteArray::~RemoteArray() {
  MPI_Win* window = static_cast<MPI_Win*>(window_);
  MPI_Win_unlock_all(*window);
  MPI_Win_free(window);
  delete window;
}

void RemoteArray::Get(void* destination, int owner, uint64_t offset,
                      uint64_t bytes) const {
  if (owner == comm_.rank()) {
    std::memcpy(destination, static_cast<const char*>(local_) + offset, bytes);
    return;
  }
  const MPI_Win window = *static_cast<MPI_Win*>(window_);
  char* to = static_cast<char*>(destination);
  constexpr uint64_t kPiece = 1ULL << 30;
  for (uint64_t done = 0; done < bytes; done += kPiece) {
    const int piece = static_cast<int>(std::min(kPiece, bytes - done));
    Check(
        MPI_Get(to + done, piece, MPI_BYTE, owner,
                static_cast<MPI_Aint>(offset + done), piece, MPI_BYTE, window),
        "MPI_Get");
  }
}

void RemoteArray::Wait(int owner) const {
  if (owner == comm_.rank()) return;
  Check(MPI_Win_flush(owner, *static_cast<MPI_Win*>(window_)), "MPI_Win_flush");
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
  // In pieces of a few MB, which the library pipelines: on four nodes a
  // sum of 2.9e7 numbers took 215 ms in one call and 71 ms in pieces of
  // 2^19 (Intel MPI 2021.6).
  constexpr uint64_t kPiece = 1ULL << 19;
  for (uint64_t done = 0; done < count; done += kPiece) {
    const int piece = static_cast<int>(std::min<uint64_t>(kPiece, count - done));
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

namespace {

int CheckedInt(uint64_t value) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("a segment exceeds 2^31 elements");
  }
  return static_cast<int>(value);
}

// Counts and displacements of the segments, in elements.
void SegmentCounts(const std::vector<uint64_t>& begin, std::vector<int>* counts,
                   std::vector<int>* displacements) {
  const size_t num = begin.size() - 1;
  counts->resize(num);
  displacements->resize(num);
  for (size_t r = 0; r < num; ++r) {
    (*displacements)[r] = CheckedInt(begin[r]);
    (*counts)[r] = CheckedInt(begin[r + 1] - begin[r]);
  }
}

}  // namespace

void Comm::GatherShared(void* base, const std::vector<uint64_t>& begin,
                        size_t element_size) const {
  if (size_ == 1) return;
  if (!nodes_contiguous_) {
    throw std::logic_error("GatherShared needs the ranks of a node together");
  }
  // Every rank of the node has written its segment before the leaders send
  // the node's segments, and reads only after they have received the rest.
  Check(MPI_Barrier(g_node_comm), "MPI_Barrier");
  const int num_nodes = static_cast<int>(node_first_.size()) - 1;
  if (is_node_leader() && num_nodes > 1) {
    const ElementType type(element_size);
    std::vector<uint64_t> node_begin(num_nodes + 1);
    for (int k = 0; k <= num_nodes; ++k) node_begin[k] = begin[node_first_[k]];
    std::vector<int> counts, displacements;
    SegmentCounts(node_begin, &counts, &displacements);
    Check(MPI_Allgatherv(MPI_IN_PLACE, 0, type.get(), base, counts.data(),
                         displacements.data(), type.get(), g_leader_comm),
          "MPI_Allgatherv");
  }
  Check(MPI_Barrier(g_node_comm), "MPI_Barrier");
}

void Comm::SumScatter(double* data, double* mine,
                      const std::vector<uint64_t>& begin) const {
  if (size_ == 1) {
    std::copy(data + begin[0], data + begin[1], mine);
    return;
  }
  // The segment of every rank summed onto it in pieces, as SumToRoot.
  constexpr uint64_t kPiece = 1ULL << 19;
  for (int owner = 0; owner < size_; ++owner) {
    for (uint64_t done = begin[owner]; done < begin[owner + 1];
         done += kPiece) {
      const int piece =
          static_cast<int>(std::min(kPiece, begin[owner + 1] - done));
      Check(MPI_Reduce(data + done,
                       rank_ == owner ? mine + (done - begin[owner]) : nullptr,
                       piece, MPI_DOUBLE, MPI_SUM, owner, MPI_COMM_WORLD),
            "MPI_Reduce");
    }
  }
}

void Comm::Scatter(const void* data, void* mine,
                   const std::vector<uint64_t>& begin,
                   size_t element_size) const {
  std::vector<int> counts, displacements;
  SegmentCounts(begin, &counts, &displacements);
  if (size_ == 1) {
    std::memcpy(mine, data, counts[0] * element_size);
    return;
  }
  const ElementType type(element_size);
  Check(MPI_Scatterv(data, counts.data(), displacements.data(), type.get(),
                     mine, counts[rank_], type.get(), 0, MPI_COMM_WORLD),
        "MPI_Scatterv");
}

void Comm::Gather(const void* mine, void* data,
                  const std::vector<uint64_t>& begin,
                  size_t element_size) const {
  std::vector<int> counts, displacements;
  SegmentCounts(begin, &counts, &displacements);
  if (size_ == 1) {
    std::memcpy(data, mine, counts[0] * element_size);
    return;
  }
  const ElementType type(element_size);
  Check(MPI_Gatherv(mine, counts[rank_], type.get(), data, counts.data(),
                    displacements.data(), type.get(), 0, MPI_COMM_WORLD),
        "MPI_Gatherv");
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
void Comm::BcastFrom(void*, uint64_t, size_t, int) const {}
void Comm::SendReceive(const void* send, uint64_t send_count, int,
                       void* receive, uint64_t, int,
                       size_t element_size) const {
  if (send_count > 0 && send != receive) {
    std::memcpy(receive, send, send_count * element_size);
  }
}
void Comm::Barrier() const {}
void Comm::Progress() {}
RemoteArray::RemoteArray(const Comm& comm, const void* local, uint64_t)
    : comm_(comm), local_(local) {}
RemoteArray::~RemoteArray() {}
void RemoteArray::Get(void* destination, int, uint64_t offset,
                      uint64_t bytes) const {
  std::memcpy(destination, static_cast<const char*>(local_) + offset, bytes);
}
void RemoteArray::Wait(int) const {}
void Comm::Sum(double*, size_t) const {}
void Comm::SumToRoot(double*, size_t) const {}
void Comm::GatherShared(void*, const std::vector<uint64_t>&, size_t) const {}
void Comm::SumScatter(double* data, double* mine,
                      const std::vector<uint64_t>& begin) const {
  std::copy(data + begin[0], data + begin[1], mine);
}
void Comm::Scatter(const void* data, void* mine,
                   const std::vector<uint64_t>& begin,
                   size_t element_size) const {
  std::memcpy(mine, data, (begin[1] - begin[0]) * element_size);
}
void Comm::Gather(const void* mine, void* data,
                  const std::vector<uint64_t>& begin,
                  size_t element_size) const {
  std::memcpy(data, mine, (begin[1] - begin[0]) * element_size);
}
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
