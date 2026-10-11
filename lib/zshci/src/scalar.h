// The number type of integrals, Hamiltonian elements and CI coefficients.
//
// Everything that holds such numbers takes the type as a template parameter
// S, which is double or Complex. The helpers below let one piece of code
// serve both: for double the conjugations are no-ops and the imaginary parts
// vanish at compile time.

#ifndef ZSHCI_SRC_SCALAR_H_
#define ZSHCI_SRC_SCALAR_H_

#include <cmath>
#include <complex>
#include <cstddef>

namespace zshci {

// A complex number: two doubles, laid out like std::complex<double> (and
// like the complex types of Fortran and HDF5), with the arithmetic written
// out. std::complex is not used for the numbers of a calculation because
// its product follows C99 Annex G, which makes the compiler call a library
// routine for every multiplication - or at best test every result for NaN -
// unless it is told otherwise with an option; the speed of the program then
// hangs on how it was compiled. Where a library wants std::complex (Eigen,
// HDF5), ScalarTraits converts.
struct Complex {
  double re;
  double im;

  // As with std::complex, a real number converts.
  constexpr Complex(double real = 0.0, double imag = 0.0)  // NOLINT
      : re(real), im(imag) {}

  constexpr double real() const { return re; }
  constexpr double imag() const { return im; }

  Complex& operator+=(Complex b) {
    re += b.re;
    im += b.im;
    return *this;
  }
  Complex& operator-=(Complex b) {
    re -= b.re;
    im -= b.im;
    return *this;
  }
  Complex& operator*=(Complex b) {
    const double real = re * b.re - im * b.im;
    im = re * b.im + im * b.re;
    re = real;
    return *this;
  }
  Complex& operator*=(double b) {
    re *= b;
    im *= b;
    return *this;
  }
  Complex& operator/=(double b) {
    re /= b;
    im /= b;
    return *this;
  }
};

constexpr Complex operator-(Complex a) { return {-a.re, -a.im}; }
constexpr Complex operator+(Complex a, Complex b) {
  return {a.re + b.re, a.im + b.im};
}
constexpr Complex operator-(Complex a, Complex b) {
  return {a.re - b.re, a.im - b.im};
}
constexpr Complex operator*(Complex a, Complex b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
constexpr Complex operator*(double a, Complex b) {
  return {a * b.re, a * b.im};
}
constexpr Complex operator*(Complex a, double b) {
  return {a.re * b, a.im * b};
}
constexpr Complex operator/(Complex a, double b) {
  return {a.re / b, a.im / b};
}
// The plain formula; fine for the ordinary magnitudes that occur here.
constexpr Complex operator/(Complex a, Complex b) {
  return Complex{a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im} /
         (b.re * b.re + b.im * b.im);
}
constexpr bool operator==(Complex a, Complex b) {
  return a.re == b.re && a.im == b.im;
}
constexpr bool operator!=(Complex a, Complex b) { return !(a == b); }

inline double Conj(double x) { return x; }
constexpr Complex Conj(Complex x) { return {x.re, -x.im}; }

inline double Real(double x) { return x; }
constexpr double Real(Complex x) { return x.re; }

inline double Imag(double /*x*/) { return 0.0; }
constexpr double Imag(Complex x) { return x.im; }

// Modulus and squared modulus.
inline double Abs(double x) { return std::fabs(x); }
inline double Abs(Complex x) { return std::sqrt(x.re * x.re + x.im * x.im); }
inline double Norm(double x) { return x * x; }
constexpr double Norm(Complex x) { return x.re * x.re + x.im * x.im; }

// Number of doubles in one S; an array of S is handed to MPI as doubles.
template <typename S>
constexpr size_t kDoublesPerScalar = sizeof(S) / sizeof(double);

template <typename S>
struct ScalarTraits;

template <>
struct ScalarTraits<double> {
  static constexpr bool kIsComplex = false;
  static constexpr const char* kName = "real";
  // The type libraries know this number by, and the conversions.
  using Std = double;
  static double ToStd(double x) { return x; }
  static double FromStd(double x) { return x; }
  // `im` is dropped.
  static double Make(double re, double /*im*/) { return re; }
};

template <>
struct ScalarTraits<Complex> {
  static constexpr bool kIsComplex = true;
  static constexpr const char* kName = "complex";
  using Std = std::complex<double>;
  static Std ToStd(Complex x) { return Std(x.re, x.im); }
  static Complex FromStd(Std x) { return Complex(x.real(), x.imag()); }
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
