#include "davidson.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace zshci {
namespace {

using Vector = std::vector<Complex>;

// <a|b>
Complex Dot(const Vector& a, const Vector& b) {
  double re = 0.0;
  double im = 0.0;
  const long n = static_cast<long>(a.size());
#pragma omp parallel for reduction(+ : re, im) schedule(static)
  for (long i = 0; i < n; ++i) {
    const Complex z = std::conj(a[i]) * b[i];
    re += z.real();
    im += z.imag();
  }
  return Complex(re, im);
}

double Norm(const Vector& a) { return std::sqrt(Dot(a, a).real()); }

// y += alpha x
void Axpy(Complex alpha, const Vector& x, Vector* y) {
  const long n = static_cast<long>(x.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) (*y)[i] += alpha * x[i];
}

void Scale(double alpha, Vector* x) {
  const long n = static_cast<long>(x->size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) (*x)[i] *= alpha;
}

// Orthogonalizes v against `basis` (twice, for numerical safety) and
// normalizes it. Returns false if nothing is left of v.
bool Orthonormalize(const std::vector<Vector>& basis, Vector* v) {
  const double initial_norm = Norm(*v);
  if (initial_norm == 0.0) return false;
  for (int pass = 0; pass < 2; ++pass) {
    for (const Vector& b : basis) Axpy(-Dot(b, *v), b, v);
  }
  const double norm = Norm(*v);
  if (norm < 1e-10 * initial_norm || norm < 1e-14) return false;
  Scale(1.0 / norm, v);
  return true;
}

// out[k] = sum_m basis[m] * coeff(m, k), for the first num_out columns.
std::vector<Vector> Combine(const std::vector<Vector>& basis,
                            const Eigen::MatrixXcd& coeff, int num_out) {
  const size_t n = basis.empty() ? 0 : basis[0].size();
  std::vector<Vector> out(num_out, Vector(n, Complex(0.0)));
  for (int k = 0; k < num_out; ++k) {
    for (size_t m = 0; m < basis.size(); ++m) {
      Axpy(coeff(m, k), basis[m], &out[k]);
    }
  }
  return out;
}

DavidsonResult SolveDense(const LinearOperator& h,
                          std::vector<Vector>* vectors) {
  const int n = static_cast<int>(h.dim());
  const int num_roots = static_cast<int>(vectors->size());
  Eigen::MatrixXcd dense(n, n);
  Vector unit(n, Complex(0.0));
  Vector column(n);
  for (int j = 0; j < n; ++j) {
    unit[j] = 1.0;
    h.Apply(unit.data(), column.data());
    unit[j] = 0.0;
    for (int i = 0; i < n; ++i) dense(i, j) = column[i];
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(dense);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("dense diagonalization failed");
  }
  DavidsonResult result;
  for (int k = 0; k < num_roots; ++k) {
    result.energies.push_back(solver.eigenvalues()[k]);
    result.residual_norms.push_back(0.0);
    for (int i = 0; i < n; ++i) (*vectors)[k][i] = solver.eigenvectors()(i, k);
  }
  result.converged = true;
  return result;
}

}  // namespace

DavidsonResult Davidson(const LinearOperator& h, const DavidsonOptions& options,
                        std::vector<std::vector<Complex>>* vectors) {
  const size_t n = h.dim();
  const int num_roots = static_cast<int>(vectors->size());
  if (num_roots == 0) throw std::invalid_argument("Davidson: no roots asked");
  if (static_cast<size_t>(num_roots) > n) {
    throw std::invalid_argument(
        "Davidson: more roots than determinants in the space");
  }
  for (const Vector& v : *vectors) {
    if (v.size() != n) throw std::invalid_argument("Davidson: bad guess size");
  }
  if (n <= static_cast<size_t>(options.dense_dim)) {
    return SolveDense(h, vectors);
  }

  const int max_space = std::max(2, options.max_space_per_root) * num_roots;
  const std::vector<double>& diagonal = h.diagonal();

  std::vector<Vector> basis;
  std::vector<Vector> sigma;
  for (Vector& guess : *vectors) {
    if (Orthonormalize(basis, &guess)) basis.push_back(std::move(guess));
  }
  if (static_cast<int>(basis.size()) < num_roots) {
    throw std::runtime_error("Davidson: starting guesses are dependent");
  }

  DavidsonResult result;
  Eigen::MatrixXcd projected(0, 0);
  if (options.print) {
    std::printf("  %5s %5s %20s %12s\n", "iter", "root", "energy", "residual");
  }

  for (int iter = 0; iter < options.max_iterations; ++iter) {
    result.num_iterations = iter + 1;
    // Apply H to the new basis vectors and extend the projected matrix.
    const int old_size = static_cast<int>(sigma.size());
    const int size = static_cast<int>(basis.size());
    for (int m = old_size; m < size; ++m) {
      sigma.emplace_back(n);
      h.Apply(basis[m].data(), sigma[m].data());
    }
    Eigen::MatrixXcd grown = Eigen::MatrixXcd::Zero(size, size);
    grown.topLeftCorner(old_size, old_size) = projected;
    for (int m = old_size; m < size; ++m) {
      for (int l = 0; l <= m; ++l) {
        grown(l, m) = Dot(basis[l], sigma[m]);
        grown(m, l) = std::conj(grown(l, m));
      }
    }
    projected = grown;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(projected);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Davidson: subspace diagonalization failed");
    }
    const Eigen::MatrixXcd& coeff = solver.eigenvectors();
    std::vector<Vector> ritz = Combine(basis, coeff, num_roots);
    std::vector<Vector> residual = Combine(sigma, coeff, num_roots);

    result.energies.assign(num_roots, 0.0);
    result.residual_norms.assign(num_roots, 0.0);
    bool all_converged = true;
    for (int k = 0; k < num_roots; ++k) {
      const double theta = solver.eigenvalues()[k];
      Axpy(-theta, ritz[k], &residual[k]);
      result.energies[k] = theta;
      result.residual_norms[k] = Norm(residual[k]);
      all_converged = all_converged && result.residual_norms[k] < options.tol;
      if (options.print) {
        std::printf("  %5d %5d %20.12f %12.3e\n", iter, k, theta,
                    result.residual_norms[k]);
      }
    }
    if (all_converged) {
      *vectors = std::move(ritz);
      result.converged = true;
      return result;
    }

    // Collapse the subspace onto the Ritz vectors when it is full.
    if (size + num_roots > max_space) {
      sigma = Combine(sigma, coeff, num_roots);
      basis = std::move(ritz);
      projected = Eigen::MatrixXcd::Zero(num_roots, num_roots);
      for (int k = 0; k < num_roots; ++k) {
        projected(k, k) = solver.eigenvalues()[k];
      }
    }

    // Diagonally preconditioned residuals of the unconverged roots.
    int num_added = 0;
    for (int k = 0; k < num_roots; ++k) {
      if (result.residual_norms[k] < options.tol) continue;
      const double theta = result.energies[k];
      Vector& correction = residual[k];
      const long length = static_cast<long>(n);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < length; ++i) {
        double denominator = theta - diagonal[i];
        if (std::abs(denominator) < 1e-8) {
          denominator = denominator < 0.0 ? -1e-8 : 1e-8;
        }
        correction[i] /= denominator;
      }
      if (Orthonormalize(basis, &correction)) {
        basis.push_back(std::move(correction));
        ++num_added;
      }
    }
    if (num_added == 0) break;  // the subspace cannot be extended
  }

  // Not converged: hand back the best vectors and let the caller decide.
  const int size = static_cast<int>(sigma.size());
  basis.resize(size);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(
      projected.topLeftCorner(size, size));
  *vectors = Combine(basis, solver.eigenvectors(), num_roots);
  result.converged = false;
  return result;
}

}  // namespace zshci
