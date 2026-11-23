// The number type of integrals, Hamiltonian elements and CI coefficients.
//
// Everything that holds such numbers takes the type as a template parameter
// S, which is double or Complex. The helpers below let one piece of code
// serve both: for double the conjugations are no-ops and the imaginary parts
// vanish at compile time.

#ifndef ZSHCI_SRC_SCALAR_H_
#define ZSHCI_SRC_SCALAR_H_

#include <complex>
#include <cstddef>

namespace zshci {

using Complex = std::complex<double>;

inline double Conj(double x) { return x; }
inline Complex Conj(Complex x) { return std::conj(x); }

inline double Real(double x) { return x; }
inline double Real(Complex x) { return x.real(); }

inline double Imag(double /*x*/) { return 0.0; }
inline double Imag(Complex x) { return x.imag(); }

// Number of doubles in one S; an array of S is handed to MPI as doubles.
template <typename S>
constexpr size_t kDoublesPerScalar = sizeof(S) / sizeof(double);

template <typename S>
struct ScalarTraits;

template <>
struct ScalarTraits<double> {
  static constexpr bool kIsComplex = false;
  static constexpr const char* kName = "real";
  // `im` is dropped.
  static double Make(double re, double /*im*/) { return re; }
};

template <>
struct ScalarTraits<Complex> {
  static constexpr bool kIsComplex = true;
  static constexpr const char* kName = "complex";
  static Complex Make(double re, double im) { return Complex(re, im); }
};

// Adds `value` to a shared accumulator; several threads may hit one element.
inline void AtomicAdd(double* target, double value) {
#pragma omp atomic
  *target += value;
}
inline void AtomicAdd(Complex* target, Complex value) {
  double* parts = reinterpret_cast<double*>(target);
  const double re = value.real();
  const double im = value.imag();
#pragma omp atomic
  parts[0] += re;
#pragma omp atomic
  parts[1] += im;
}

}  // namespace zshci

#endif  // ZSHCI_SRC_SCALAR_H_
