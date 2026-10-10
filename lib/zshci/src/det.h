// Fixed-width occupation bit string for a Slater determinant of spinors.
//
// Bit p is set when spinor p is occupied. The width W (number of 64-bit words)
// is a template parameter so that every instantiation is a trivially copyable
// value type: it can be stored in flat arrays, sorted, hashed, sent through
// MPI as bytes and written to HDF5 as an N x W table. The driver picks W from
// the number of spinors when the calculation starts.

#ifndef ZSHCI_SRC_DET_H_
#define ZSHCI_SRC_DET_H_

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace zshci {

// Finalizer of splitmix64; a cheap, well mixing 64-bit permutation.
inline uint64_t Mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

template <int W>
struct Det {
  std::array<uint64_t, W> words{};

  bool Occupied(int p) const { return (words[p >> 6] >> (p & 63)) & 1ULL; }
  void Set(int p) { words[p >> 6] |= 1ULL << (p & 63); }
  void Clear(int p) { words[p >> 6] &= ~(1ULL << (p & 63)); }

  int Count() const {
    int n = 0;
    for (int k = 0; k < W; ++k) n += __builtin_popcountll(words[k]);
    return n;
  }

  // Number of set bits in which the two determinants differ; twice the
  // excitation rank when both hold the same number of electrons.
  int NumDifferentBits(const Det& other) const {
    int n = 0;
    for (int k = 0; k < W; ++k) {
      n += __builtin_popcountll(words[k] ^ other.words[k]);
    }
    return n;
  }

  // Number of occupied spinors strictly between p and q (p < q).
  int CountBetween(int p, int q) const {
    int n = 0;
    for (int k = (p + 1) >> 6; k <= (q - 1) >> 6 && p + 1 < q; ++k) {
      uint64_t word = words[k];
      const int lo = (k == ((p + 1) >> 6)) ? ((p + 1) & 63) : 0;
      const int hi = (k == ((q - 1) >> 6)) ? ((q - 1) & 63) : 63;
      word >>= lo;
      const int width = hi - lo + 1;
      if (width < 64) word &= (1ULL << width) - 1;
      n += __builtin_popcountll(word);
    }
    return n;
  }

  Det And(const Det& other) const {
    Det out;
    for (int k = 0; k < W; ++k) out.words[k] = words[k] & other.words[k];
    return out;
  }

  uint64_t Hash() const {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int k = 0; k < W; ++k) h = Mix64(h ^ words[k]);
    return h;
  }

  // Appends the indices of the occupied spinors, in increasing order.
  void AppendOccupied(std::vector<int>* occ) const {
    for (int k = 0; k < W; ++k) {
      uint64_t word = words[k];
      while (word != 0) {
        occ->push_back(64 * k + __builtin_ctzll(word));
        word &= word - 1;
      }
    }
  }

  std::string ToString(int num_orbs) const {
    std::string s(num_orbs, '0');
    for (int p = 0; p < num_orbs; ++p) {
      if (Occupied(p)) s[p] = '1';
    }
    return s;
  }

  friend bool operator==(const Det& a, const Det& b) {
    return a.words == b.words;
  }
  friend bool operator!=(const Det& a, const Det& b) { return !(a == b); }

  // Orders by the highest word first, i.e. as one big integer.
  friend bool operator<(const Det& a, const Det& b) {
    for (int k = W - 1; k >= 0; --k) {
      if (a.words[k] != b.words[k]) return a.words[k] < b.words[k];
    }
    return false;
  }
};

static_assert(std::is_trivially_copyable<Det<1>>::value, "Det must be POD");
static_assert(sizeof(Det<2>) == 16, "Det must hold nothing but its words");

// Number of 64-bit words needed for num_orbs spinors.
inline int WordsForOrbitals(int num_orbs) { return (num_orbs + 63) / 64; }

// Largest width instantiated by the driver.
constexpr int kMaxDetWords = 4;

}  // namespace zshci

#endif  // ZSHCI_SRC_DET_H_
