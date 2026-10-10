// Slater-Condon rules for spinor determinants.
//
// The Hamiltonian is
//   H = sum_pq h(p,q) p^+ q + 1/4 sum_pqrs <pq||rs> p^+ q^+ s r + E_core,
// with the antisymmetrized integrals of integrals.h. No spin or time-reversal
// structure is assumed. The number
// type S of the integrals is double or Complex (scalar.h).

#ifndef ZSHCI_SRC_HAMILTONIAN_H_
#define ZSHCI_SRC_HAMILTONIAN_H_

#include <vector>

#include "det.h"
#include "integrals.h"

namespace zshci {

// The spinors in which a bra differs from a ket. `holes` are occupied in the
// ket only, `particles` in the bra only; both are in increasing order.
// `rank` is 0, 1 or 2; any higher excitation is reported as 3.
struct Excitation {
  int rank = 0;
  int holes[2] = {-1, -1};
  int particles[2] = {-1, -1};
};

template <int W>
Excitation FindExcitation(const Det<W>& bra, const Det<W>& ket) {
  Excitation exc;
  int num_holes = 0;
  int num_particles = 0;
  for (int k = 0; k < W; ++k) {
    const uint64_t diff = bra.words[k] ^ ket.words[k];
    uint64_t hole_bits = diff & ket.words[k];
    uint64_t particle_bits = diff & bra.words[k];
    while (hole_bits != 0) {
      if (num_holes == 2) {
        exc.rank = 3;
        return exc;
      }
      exc.holes[num_holes++] = 64 * k + __builtin_ctzll(hole_bits);
      hole_bits &= hole_bits - 1;
    }
    while (particle_bits != 0) {
      if (num_particles == 2) {
        exc.rank = 3;
        return exc;
      }
      exc.particles[num_particles++] = 64 * k + __builtin_ctzll(particle_bits);
      particle_bits &= particle_bits - 1;
    }
  }
  exc.rank = num_holes;
  return exc;
}

// <bra| a^+ i |ket> for a single excitation i -> a out of `ket`.
template <int W>
double SingleSign(const Det<W>& ket, int i, int a) {
  const int lo = i < a ? i : a;
  const int hi = i < a ? a : i;
  return (ket.CountBetween(lo, hi) & 1) ? -1.0 : 1.0;
}

// <bra| a^+ b^+ j i |ket> for the double excitation (i, j) -> (a, b) out of
// `ket`, with i < j and a < b.
template <int W>
double DoubleSign(const Det<W>& ket, int i, int j, int a, int b) {
  double sign = SingleSign(ket, i, a) * SingleSign(ket, j, b);
  if (a > j || b < i) sign = -sign;
  return sign;
}

// <det|H|det> without the core energy. Called once per candidate determinant
// in perturbation theory, so it must not allocate: memory allocation is a
// point of contention between threads.
template <int W, typename S>
double DiagonalEnergy(const Det<W>& det, const Integrals<S>& ints) {
  int occ[64 * W];
  int num_occ = 0;
  for (int k = 0; k < W; ++k) {
    uint64_t word = det.words[k];
    while (word != 0) {
      occ[num_occ++] = 64 * k + __builtin_ctzll(word);
      word &= word - 1;
    }
  }
  double energy = 0.0;
  for (int m = 0; m < num_occ; ++m) {
    const int p = occ[m];
    energy += Real(ints.OneBody(p, p));
    for (int n = 0; n < m; ++n) {
      energy += ints.PairEnergy(p, occ[n]);
    }
  }
  return energy;
}

// <bra|H|ket> for a bra that differs from the ket by `exc` (rank 1 or 2).
template <int W, typename S>
S OffDiagonalElement(const Det<W>& ket, const Excitation& exc,
                     const Integrals<S>& ints) {
  if (exc.rank == 1) {
    const int i = exc.holes[0];
    const int a = exc.particles[0];
    S value = ints.OneBody(a, i);
    for (int k = 0; k < W; ++k) {
      uint64_t word = ket.words[k];
      while (word != 0) {
        const int j = 64 * k + __builtin_ctzll(word);
        value += ints.Antisym(a, j, i, j);
        word &= word - 1;
      }
    }
    return SingleSign(ket, i, a) * value;
  }
  if (exc.rank == 2) {
    const int i = exc.holes[0];
    const int j = exc.holes[1];
    const int a = exc.particles[0];
    const int b = exc.particles[1];
    return DoubleSign(ket, i, j, a, b) * ints.AntisymOrdered(a, b, i, j);
  }
  return S(0.0);
}

// <bra|H|ket> for two different determinants; zero beyond double excitations.
template <int W, typename S>
S Hij(const Det<W>& bra, const Det<W>& ket, const Integrals<S>& ints) {
  return OffDiagonalElement(ket, FindExcitation(bra, ket), ints);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_HAMILTONIAN_H_
