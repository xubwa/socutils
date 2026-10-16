#include "fci_davidson.h"

#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace zshci {
namespace {

template <typename S>
using Matrix = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;

// Largest basis this solver works with.
constexpr int kMaxSpace = 64;

// Checkpoint files: a header, the energies, and from kDataOffset on the
// vectors one after another in the global order.
constexpr char kMagic[8] = {'Z', 'S', 'H', 'C', 'I', 'C', 'I', '1'};
constexpr int64_t kDataOffset = 4096;
struct CheckpointHeader {
  char magic[8];
  int64_t dim;
  int32_t num_roots;
  int32_t is_complex;
  uint64_t fingerprint;
  int64_t iteration;
};

double Norm2(double x) { return x * x; }
double Norm2(Complex x) { return std::norm(x); }

// A pseudo-random number in [-1, 1) from an index.
double Noise(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x ^= x >> 31;
  return static_cast<double>(x >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

// A diagonal element and its place, for the start vectors.
struct Candidate {
  double value;
  int64_t index;
  bool operator<(const Candidate& o) const {
    return value < o.value || (value == o.value && index < o.index);
  }
};

template <typename S>
class Solver {
 public:
  Solver(const CiOperator<S>& h, const double* diagonal,
         const VectorLayout& layout, const FciDavidsonOptions& options,
         const Comm& comm)
      : h_(h),
        diagonal_(diagonal),
        layout_(layout),
        options_(options),
        comm_(comm),
        n_(layout.local_size()),
        k_(options.num_roots) {}

  FciDavidsonResult<S> Run();

 private:
  // <x_l|y> for l < count, x the basis.
  std::vector<S> Dots(int count, const S* y) const {
    std::vector<S> total(count, S(0.0));
#pragma omp parallel
    {
      std::vector<S> part(count, S(0.0));
#pragma omp for schedule(static)
      for (long i = 0; i < n_; ++i) {
        const S yi = y[i];
        for (int l = 0; l < count; ++l) part[l] += Conj(x_[l][i]) * yi;
      }
#pragma omp critical
      for (int l = 0; l < count; ++l) total[l] += part[l];
    }
    comm_.Sum(reinterpret_cast<double*>(total.data()),
              count * kDoublesPerScalar<S>);
    return total;
  }

  double Norm(const S* y) const {
    double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total)
    for (long i = 0; i < n_; ++i) total += Norm2(y[i]);
    comm_.Sum(&total, 1);
    return std::sqrt(total);
  }

  // y -= sum_l c[l] x_l, then y *= scale.
  void Subtract(const std::vector<S>& c, S* y, double scale = 1.0) const {
    const int count = static_cast<int>(c.size());
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n_; ++i) {
      S value = y[i];
      for (int l = 0; l < count; ++l) value -= c[l] * x_[l][i];
      y[i] = value * scale;
    }
  }

  // Makes x_[slot] orthonormal to x_[0 .. slot); false if nothing is left.
  bool Orthonormalize(int slot) {
    S* y = x_[slot].data();
    const double initial = Norm(y);
    if (initial == 0.0) return false;
    for (int pass = 0; pass < 2; ++pass) {
      if (slot > 0) Subtract(Dots(slot, y), y);
    }
    const double norm = Norm(y);
    if (norm < 1e-10 * initial || norm < 1e-14) return false;
    const double scale = 1.0 / norm;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n_; ++i) y[i] *= scale;
    return true;
  }

  // v_c <- sum_l v_l t(l, c) for c < t.cols(), in place, for the basis and
  // (with_sigma) the products.
  void Transform(const Matrix<S>& t, bool with_sigma) {
    const int rows = static_cast<int>(t.rows());
    const int cols = static_cast<int>(t.cols());
    for (int which = 0; which < (with_sigma ? 2 : 1); ++which) {
      std::vector<BigArray<S>>& v = which == 0 ? x_ : hx_;
      S* p[kMaxSpace];
      for (int l = 0; l < rows; ++l) p[l] = v[l].data();
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n_; ++i) {
        S old[kMaxSpace];
        for (int l = 0; l < rows; ++l) old[l] = p[l][i];
        for (int c = 0; c < cols; ++c) {
          S value(0.0);
          for (int l = 0; l < rows; ++l) value += old[l] * t(l, c);
          p[c][i] = value;
        }
      }
    }
  }

  // Norms of the residuals sum_l c(l, j) (hx_l - theta_j x_l), and with
  // `slots` the preconditioned residuals of the roots `which` written into
  // x_[slots[m]].
  std::vector<double> Residuals(const Matrix<S>& c,
                                const std::vector<double>& theta,
                                const std::vector<int>& which,
                                const std::vector<int>& slots) {
    const int size = static_cast<int>(c.rows());
    const int roots = static_cast<int>(c.cols());
    std::vector<double> total(roots, 0.0);
    const double shift = options_.level_shift;
#pragma omp parallel
    {
      std::vector<double> part(roots, 0.0);
      S r[kMaxSpace];
#pragma omp for schedule(static)
      for (long i = 0; i < n_; ++i) {
        S xs[kMaxSpace], hs[kMaxSpace];
        for (int l = 0; l < size; ++l) {
          xs[l] = x_[l][i];
          hs[l] = hx_[l][i];
        }
        for (int j = 0; j < roots; ++j) {
          S value(0.0);
          for (int l = 0; l < size; ++l) {
            value += c(l, j) * (hs[l] - theta[j] * xs[l]);
          }
          r[j] = value;
          part[j] += Norm2(value);
        }
        for (size_t m = 0; m < slots.size(); ++m) {
          const int j = which[m];
          double denominator = diagonal_[i] - theta[j] + shift;
          if (std::abs(denominator) < 1e-8) {
            denominator = denominator < 0.0 ? -1e-8 : 1e-8;
          }
          x_[slots[m]][i] = r[j] / denominator;
        }
      }
#pragma omp critical
      for (int j = 0; j < roots; ++j) total[j] += part[j];
    }
    comm_.Sum(total.data(), roots);
    for (double& t : total) t = std::sqrt(t);
    return total;
  }

  void ApplyH(int slot) {
    const double start = omp_get_wtime();
    h_.Apply(x_[slot].data(), hx_[slot].data());
    result_.seconds_products += omp_get_wtime() - start;
    ++result_.num_products;
  }

  // A(l, slot) = <x_l|H x_slot> for l <= slot.
  void ExtendProjected(int slot, Matrix<S>* a) const {
    const std::vector<S> column = Dots(slot + 1, hx_[slot].data());
    for (int l = 0; l <= slot; ++l) {
      (*a)(l, slot) = column[l];
      (*a)(slot, l) = Conj(column[l]);
    }
    (*a)(slot, slot) = Real(column[slot]);
  }

  void Guess(int first);
  void WriteCheckpoint(const Matrix<S>& c, const std::vector<double>& theta,
                       int iteration) const;
  FciDavidsonResult<S> Dense();

  const CiOperator<S>& h_;
  const double* diagonal_;
  const VectorLayout& layout_;
  const FciDavidsonOptions& options_;
  const Comm& comm_;
  const long n_;
  const int k_;
  std::vector<BigArray<S>> x_, hx_;
  FciDavidsonResult<S> result_;
};

// Unit vectors on the lowest diagonal elements (not taken yet) with a little
// noise, for the roots from `first` on.
template <typename S>
void Solver<S>::Guess(int first) {
  const int need = k_ - first;
  if (need <= 0) return;
  // The `want` lowest (value, global index), a few more in case some are
  // dependent on the vectors read back.
  const int want = need + first;
  std::vector<Candidate> best;
#pragma omp parallel
  {
    std::vector<Candidate> mine;
    long base = 0;
    for (const VectorLayout::Segment& seg : layout_.segments) {
      const long len = seg.end - seg.begin;
#pragma omp for schedule(static) nowait
      for (long i = 0; i < len; ++i) {
        const Candidate item{diagonal_[base + i], seg.begin + i};
        if (static_cast<int>(mine.size()) < want || item < mine.back()) {
          mine.insert(std::upper_bound(mine.begin(), mine.end(), item), item);
          if (static_cast<int>(mine.size()) > want) mine.pop_back();
        }
      }
      base += len;
    }
#pragma omp critical
    best.insert(best.end(), mine.begin(), mine.end());
  }
  std::sort(best.begin(), best.end());
  if (static_cast<int>(best.size()) > want) best.resize(want);
  best = comm_.GatherAll(best);
  std::sort(best.begin(), best.end());
  int slot = first;
  for (size_t b = 0; b < best.size() && slot < k_; ++b) {
    S* y = x_[slot].data();
    long base = 0;
    for (const VectorLayout::Segment& seg : layout_.segments) {
      const long len = seg.end - seg.begin;
#pragma omp parallel for schedule(static)
      for (long i = 0; i < len; ++i) {
        const uint64_t global = seg.begin + i;
        y[base + i] =
            S(1e-6 * Noise(global * 64 + slot)) +
            (static_cast<int64_t>(global) == best[b].index ? S(1.0) : S(0.0));
      }
      base += len;
    }
    if (Orthonormalize(slot)) ++slot;
  }
  if (slot < k_) throw std::runtime_error("fci: no independent start vectors");
}

template <typename S>
void Solver<S>::WriteCheckpoint(const Matrix<S>& c,
                                const std::vector<double>& theta,
                                int iteration) const {
  const std::string tmp = options_.checkpoint + ".tmp";
  // The root makes the file, then every rank writes its elements.
  if (comm_.is_root()) {
    const int made = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (made >= 0) close(made);
  }
  comm_.Barrier();
  const int fd = open(tmp.c_str(), O_WRONLY);
  if (fd < 0) {
    throw std::runtime_error("fci: cannot write checkpoint " + tmp + ": " +
                             std::strerror(errno));
  }
  auto write_at = [&](const void* data, size_t bytes, int64_t offset) {
    const char* p = static_cast<const char*>(data);
    while (bytes > 0) {
      const ssize_t done = pwrite(fd, p, bytes, offset);
      if (done <= 0) {
        close(fd);
        throw std::runtime_error("fci: writing checkpoint " + tmp + ": " +
                                 std::strerror(errno));
      }
      p += done;
      bytes -= done;
      offset += done;
    }
  };
  CheckpointHeader header;
  std::memcpy(header.magic, kMagic, 8);
  header.dim = layout_.global_dim;
  header.num_roots = k_;
  header.is_complex = ScalarTraits<S>::kIsComplex ? 1 : 0;
  header.fingerprint = options_.fingerprint;
  header.iteration = iteration;
  if (comm_.is_root()) {
    write_at(&header, sizeof(header), 0);
    write_at(theta.data(), sizeof(double) * k_, sizeof(header));
  }
  // The Ritz vectors, piece by piece.
  const int size = static_cast<int>(c.rows());
  const long chunk = 1 << 21;
  std::vector<S> buffer(chunk);
  for (int j = 0; j < k_; ++j) {
    long base = 0;
    for (const VectorLayout::Segment& seg : layout_.segments) {
      for (long b = 0; b < seg.end - seg.begin; b += chunk) {
        const long len = std::min(chunk, seg.end - seg.begin - b);
#pragma omp parallel for schedule(static)
        for (long i = 0; i < len; ++i) {
          S value(0.0);
          for (int l = 0; l < size; ++l) value += c(l, j) * x_[l][base + b + i];
          buffer[i] = value;
        }
        write_at(buffer.data(), sizeof(S) * len,
                 kDataOffset + static_cast<int64_t>(sizeof(S)) *
                                   (j * layout_.global_dim + seg.begin + b));
      }
      base += seg.end - seg.begin;
    }
  }
  if (fsync(fd) != 0 || close(fd) != 0) {
    throw std::runtime_error("fci: closing checkpoint " + tmp);
  }
  comm_.Barrier();
  if (comm_.is_root() &&
      std::rename(tmp.c_str(), options_.checkpoint.c_str()) != 0) {
    throw std::runtime_error("fci: cannot rename " + tmp);
  }
  comm_.Barrier();
}

// The whole matrix, from products with unit vectors.
template <typename S>
FciDavidsonResult<S> Solver<S>::Dense() {
  const long n = n_;
  Matrix<S> dense(n, n);
  BigArray<S> unit(n, false), column(n, false);
  for (long j = 0; j < n; ++j) {
    unit[j] = 1.0;
    h_.Apply(unit.data(), column.data());
    unit[j] = 0.0;
    for (long i = 0; i < n; ++i) dense(i, j) = column[i];
  }
  // Hermitian to rounding.
  Matrix<S> herm = 0.5 * (dense + dense.adjoint());
  Eigen::SelfAdjointEigenSolver<Matrix<S>> solver(herm);
  FciDavidsonResult<S> result;
  for (int j = 0; j < k_; ++j) {
    result.energies.push_back(solver.eigenvalues()[j]);
    result.residual_norms.push_back(0.0);
    BigArray<S> v(n, false);
    for (long i = 0; i < n; ++i) v[i] = solver.eigenvectors()(i, j);
    result.vectors.push_back(std::move(v));
  }
  result.converged = true;
  result.num_products = static_cast<int>(n);
  return result;
}

template <typename S>
FciDavidsonResult<S> Solver<S>::Run() {
  const double start = omp_get_wtime();
  if (k_ < 1 || k_ > layout_.global_dim) {
    throw std::runtime_error("fci: bad number of roots");
  }
  if (comm_.size() == 1 && n_ == layout_.global_dim &&
      layout_.global_dim <= options_.dense_dim) {
    return Dense();
  }
  const int m = options_.max_space;
  if (m < 2 * k_ || m > kMaxSpace) {
    throw std::runtime_error(
        "fci: max_space must be at least 2 * nroots and "
        "at most " +
        std::to_string(kMaxSpace));
  }
  for (int l = 0; l < m; ++l) {
    x_.emplace_back(n_, options_.huge_pages);
    hx_.emplace_back(n_, options_.huge_pages);
  }

  // Start vectors: the checkpoint, then the lowest diagonal elements.
  int read = 0;
  if (options_.restart && !options_.checkpoint.empty()) {
    std::vector<BigArray<S>> back;
    std::vector<double> energies;
    read =
        ReadCiCheckpoint<S>(options_.checkpoint, layout_, options_.fingerprint,
                            k_, options_.huge_pages, &back, &energies);
    int kept = 0;
    for (int j = 0; j < read; ++j) {
      std::swap(x_[kept], back[j]);
      if (Orthonormalize(kept)) ++kept;
    }
    read = kept;
    result_.restarted = read > 0;
    if (read > 0 && options_.print) {
      std::printf("fci: %d vector(s) read from %s (energies", read,
                  options_.checkpoint.c_str());
      for (int j = 0; j < read; ++j) std::printf(" %.10f", energies[j]);
      std::printf(")\n");
    } else if (options_.print) {
      std::printf("fci: no checkpoint %s; starting afresh\n",
                  options_.checkpoint.c_str());
    }
  }
  Guess(read);

  int size = k_;
  Matrix<S> a = Matrix<S>::Zero(m, m);
  for (int l = 0; l < size; ++l) {
    ApplyH(l);
    ExtendProjected(l, &a);
  }
  Matrix<S> previous;  // Ritz coefficients of the last iteration
  const bool print = options_.print;
  if (print) {
    std::printf("  %5s %5s %22s %12s %10s %10s\n", "iter", "root", "energy",
                "residual", "sigma/s", "total/s");
  }
  for (int iter = 1;; ++iter) {
    const double iter_start = omp_get_wtime();
    const double products_before = result_.seconds_products;
    Eigen::SelfAdjointEigenSolver<Matrix<S>> eig(a.topLeftCorner(size, size));
    Matrix<S> c = eig.eigenvectors().leftCols(k_);
    std::vector<double> theta(k_);
    for (int j = 0; j < k_; ++j) theta[j] = eig.eigenvalues()[j];

    std::vector<double> norms = Residuals(c, theta, {}, {});
    std::vector<int> open;
    for (int j = 0; j < k_; ++j) {
      if (!(norms[j] < options_.tol)) open.push_back(j);
    }
    result_.num_iterations = iter;
    result_.energies = theta;
    result_.residual_norms = norms;
    const bool done = open.empty() || iter >= options_.max_iterations;
    if (!options_.checkpoint.empty() &&
        (done || iter % std::max(1, options_.checkpoint_every) == 0)) {
      const double t = omp_get_wtime();
      WriteCheckpoint(c, theta, iter);
      if (print)
        std::printf("  checkpoint written, %.1f s\n", omp_get_wtime() - t);
    }
    if (done) {
      Transform(c, false);
      result_.converged = open.empty();
      for (int j = 0; j < k_ && print; ++j) {
        std::printf("  %5d %5d %22.14f %12.3e\n", iter, j, theta[j], norms[j]);
      }
      break;
    }

    // Collapse onto the Ritz vectors and, as far as there is room, those of
    // the last iteration.
    const int num_add = static_cast<int>(open.size());
    if (size + num_add > m) {
      const int room = m - k_ - num_add;
      Matrix<S> t = c;
      if (room > 0 && previous.cols() > 0) {
        Matrix<S> p = Matrix<S>::Zero(size, previous.cols());
        p.topRows(previous.rows()) = previous;
        for (int col = 0; col < p.cols() && t.cols() < k_ + room; ++col) {
          Eigen::Matrix<S, Eigen::Dynamic, 1> u = p.col(col);
          for (int pass = 0; pass < 2; ++pass) {
            u -= t * (t.adjoint() * u);
          }
          const double norm = u.norm();
          if (norm < 1e-8) continue;
          t.conservativeResize(Eigen::NoChange, t.cols() + 1);
          t.col(t.cols() - 1) = u / norm;
        }
      }
      Transform(t, true);
      const int new_size = static_cast<int>(t.cols());
      Matrix<S> small = t.adjoint() * a.topLeftCorner(size, size) * t;
      a.setZero();
      a.topLeftCorner(new_size, new_size) = 0.5 * (small + small.adjoint());
      size = new_size;
      c = Matrix<S>::Identity(size, k_);
    }
    previous = c;

    // Preconditioned residuals of the open roots become new basis vectors.
    std::vector<int> slots;
    for (int j = 0; j < num_add; ++j) slots.push_back(size + j);
    Residuals(c, theta, open, slots);
    int added = 0;
    for (int j = 0; j < num_add; ++j) {
      if (size + added != slots[j]) {
        std::swap(x_[size + added], x_[slots[j]]);
      }
      if (Orthonormalize(size + added)) ++added;
    }
    if (added == 0) {
      std::printf("fci: the subspace cannot be extended\n");
      Transform(c, false);
      break;
    }
    for (int j = 0; j < added; ++j) {
      ApplyH(size + j);
      ExtendProjected(size + j, &a);
    }
    size += added;
    for (int j = 0; j < k_ && print; ++j) {
      std::printf("  %5d %5d %22.14f %12.3e", iter, j, theta[j], norms[j]);
      if (j == 0) {
        std::printf(" %10.1f %10.1f",
                    result_.seconds_products - products_before,
                    omp_get_wtime() - iter_start);
      }
      std::printf("\n");
    }
  }
  for (int j = 0; j < k_; ++j) result_.vectors.push_back(std::move(x_[j]));
  x_.clear();
  hx_.clear();
  result_.seconds_total = omp_get_wtime() - start;
  return std::move(result_);
}

}  // namespace

template <typename S>
FciDavidsonResult<S> FciDavidson(const CiOperator<S>& h, const double* diagonal,
                                 const VectorLayout& layout,
                                 const FciDavidsonOptions& options,
                                 const Comm& comm) {
  Solver<S> solver(h, diagonal, layout, options, comm);
  return solver.Run();
}

template <typename S>
int ReadCiCheckpoint(const std::string& path, const VectorLayout& layout,
                     uint64_t fingerprint, int num_roots, bool huge_pages,
                     std::vector<BigArray<S>>* vectors,
                     std::vector<double>* energies) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return 0;
  auto read_at = [&](void* data, size_t bytes, int64_t offset) {
    char* p = static_cast<char*>(data);
    while (bytes > 0) {
      const ssize_t done = pread(fd, p, bytes, offset);
      if (done <= 0) {
        close(fd);
        throw std::runtime_error("fci: checkpoint " + path + " is truncated");
      }
      p += done;
      bytes -= done;
      offset += done;
    }
  };
  CheckpointHeader header;
  read_at(&header, sizeof(header), 0);
  if (std::memcmp(header.magic, kMagic, 8) != 0 ||
      header.dim != layout.global_dim ||
      header.is_complex != (ScalarTraits<S>::kIsComplex ? 1 : 0) ||
      header.fingerprint != fingerprint) {
    close(fd);
    throw std::runtime_error("fci: checkpoint " + path +
                             " belongs to another space or number type");
  }
  const int count = std::min<int>(num_roots, header.num_roots);
  std::vector<double> e(header.num_roots);
  read_at(e.data(), sizeof(double) * header.num_roots, sizeof(header));
  vectors->clear();
  for (int j = 0; j < count; ++j) {
    BigArray<S> v(layout.local_size(), huge_pages);
    long base = 0;
    for (const VectorLayout::Segment& seg : layout.segments) {
      read_at(v.data() + base, sizeof(S) * (seg.end - seg.begin),
              kDataOffset + static_cast<int64_t>(sizeof(S)) *
                                (j * header.dim + seg.begin));
      base += seg.end - seg.begin;
    }
    vectors->push_back(std::move(v));
  }
  close(fd);
  if (energies != nullptr) energies->assign(e.begin(), e.begin() + count);
  return count;
}

template FciDavidsonResult<double> FciDavidson(const CiOperator<double>&,
                                               const double*,
                                               const VectorLayout&,
                                               const FciDavidsonOptions&,
                                               const Comm&);
template FciDavidsonResult<Complex> FciDavidson(const CiOperator<Complex>&,
                                                const double*,
                                                const VectorLayout&,
                                                const FciDavidsonOptions&,
                                                const Comm&);
template int ReadCiCheckpoint(const std::string&, const VectorLayout&, uint64_t,
                              int, bool, std::vector<BigArray<double>>*,
                              std::vector<double>*);
template int ReadCiCheckpoint(const std::string&, const VectorLayout&, uint64_t,
                              int, bool, std::vector<BigArray<Complex>>*,
                              std::vector<double>*);

}  // namespace zshci
