// One- and two-electron integrals over spinors, and the FCIDUMP reader.
//
// The two-electron integrals are held antisymmetrized,
//   <pq||rs> = (pr|qs) - (ps|qr),
// with (pr|qs) in chemists' notation, which is all a determinant program
// needs: the element of a double excitation (i, j) -> (a, b) is <ab||ij>.
// They are antisymmetric in p, q and in r, s, and <pq||rs> = conj(<rs||pq>),
// so only p < q, r < s with the pair (p, q) not below the pair (r, s) is
// stored: about n^4 / 8 numbers for n spinors.

#ifndef ZSHCI_SRC_INTEGRALS_H_
#define ZSHCI_SRC_INTEGRALS_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "scalar.h"

namespace zshci {

// S is double or Complex; both are instantiated in integrals.cc.
template <typename S>
class Integrals {
 public:
  Integrals() = default;

  // Allocates storage for num_orbs spinors, zeroed. With several ranks this
  // is collective, and the two-electron integrals are held once per node.
  explicit Integrals(int num_orbs, const Comm& comm = Comm());

  int num_orbs() const { return num_orbs_; }
  int num_elec() const { return num_elec_; }
  double core_energy() const { return core_energy_; }
  const std::vector<int>& labels() const { return labels_; }

  void set_num_elec(int n) { num_elec_ = n; }
  void set_core_energy(double e) { core_energy_ = e; }
  void set_labels(std::vector<int> labels) { labels_ = std::move(labels); }

  // h(p, q) = <p|h|q>.
  S OneBody(int p, int q) const { return one_body_[p * num_orbs_ + q]; }

  // <pq||rs> for p < q and r < s.
  S AntisymOrdered(int p, int q, int r, int s) const {
    const size_t pq = PairIndex(p, q);
    const size_t rs = PairIndex(r, s);
    return pq >= rs ? two_body_[pq * (pq + 1) / 2 + rs]
                    : Conj(two_body_[rs * (rs + 1) / 2 + pq]);
  }

  // <pq||rs> for any indices.
  S Antisym(int p, int q, int r, int s) const {
    if (p == q || r == s) return S(0.0);
    bool negative = false;
    if (p > q) {
      std::swap(p, q);
      negative = !negative;
    }
    if (r > s) {
      std::swap(r, s);
      negative = !negative;
    }
    const S value = AntisymOrdered(p, q, r, s);
    return negative ? -value : value;
  }

  // <pq||pq> = (pp|qq) - (pq|qp), the interaction energy of two occupied
  // spinors, from a table of its own.
  double PairEnergy(int p, int q) const {
    return pair_energy_[p * num_orbs_ + q];
  }

  void SetOneBody(int p, int q, S value) {
    one_body_[p * num_orbs_ + q] = value;
  }
  // Sets <pq||rs> and, with it, <rs||pq>; p < q and r < s.
  void SetAntisymOrdered(int p, int q, int r, int s, S value) {
    const size_t pq = PairIndex(p, q);
    const size_t rs = PairIndex(r, s);
    if (pq >= rs) {
      two_body_[pq * (pq + 1) / 2 + rs] = value;
    } else {
      two_body_[rs * (rs + 1) / 2 + pq] = Conj(value);
    }
  }

  // Adds what the integral (pq|rs) = value, together with its Hermitian
  // partner (qp|sr) = conj(value), contributes to the antisymmetrized
  // integrals. Call once for every such pair of integrals; (rs|pq) is the
  // same integral as (pq|rs) and must not be added again.
  void AddChemist(int p, int q, int r, int s, S value);

  // Fills the PairEnergy table; call once after all integrals are set.
  void Finalize();

  // Copies the root's integrals to every rank. Collective.
  void Share(const Comm& comm);

 private:
  // Index of the pair p < q.
  static size_t PairIndex(int p, int q) {
    return static_cast<size_t>(q) * (q - 1) / 2 + p;
  }

  int num_orbs_ = 0;
  int num_elec_ = 0;
  double core_energy_ = 0.0;
  std::vector<int> labels_;
  std::vector<S> one_body_;
  SharedArray<S> two_body_;
  std::vector<double> pair_energy_;
};

// FCIDUMP files: a namelist header with NORB, NELEC and (optionally) ORBSYM,
// then one integral per line with 1-based indices, either
//   "(re,im) p q r s"   complex format, or
//   "value p q r s"     real format.
// "p q r s" is the integral (pq|rs) in chemists' notation; "p q 0 0" is a
// one-body element, whose Hermitian partner is filled in as well; "0 0 0 0"
// is the core energy; "p 0 0 0" (orbital energy) is skipped.
//
// Of the four integrals (pq|rs) = (rs|pq) = conj((qp|sr)) = conj((sr|qp)) the
// first one met in the file is used and the others, if listed, are ignored,
// so a file may list all of them or any one of each set.

// Whether the file is in the complex format, judged from its first integral
// line. Collective: the root looks and every rank returns the answer.
bool FcidumpIsComplex(const std::string& path, const Comm& comm = Comm());

// Reads a file in either format. With S = double an imaginary part other
// than zero is an error. Throws std::runtime_error on a malformed file.
//
// Collective: the root reads the file and every rank returns the integrals.
// Instantiated for double and Complex.
template <typename S>
Integrals<S> ReadFcidump(const std::string& path, const Comm& comm = Comm());

}  // namespace zshci

#endif  // ZSHCI_SRC_INTEGRALS_H_
