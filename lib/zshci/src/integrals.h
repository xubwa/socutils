// One- and two-electron integrals over spinors, and the FCIDUMP reader.
//
// The two-electron integrals are held antisymmetrized,
//   <pq||rs> = (pr|qs) - (ps|qr),
// with (pr|qs) in chemists' notation, which is all a determinant program
// needs: the element of a double excitation (i, j) -> (a, b) is <ab||ij>.
// They are antisymmetric in p, q and in r, s, and <pq||rs> = conj(<rs||pq>),
// so only p < q, r < s with the pair (p, q) not below the pair (r, s) is
// stored: about n^4 / 8 numbers for n spinors.
//
// With symmetry labels (labels.h) the integrals are held in blocks. Every
// pair p < q carries the sum of the labels of p and q, and <pq||rs> vanishes
// unless both pairs carry the same sums; only those are stored. Spin
// orbitals labelled by their spin projection, for example, cost 3/8 of the
// storage without labels.

#ifndef ZSHCI_SRC_INTEGRALS_H_
#define ZSHCI_SRC_INTEGRALS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "labels.h"
#include "scalar.h"

namespace zshci {

// S is double or Complex; both are instantiated in integrals.cc.
template <typename S>
class Integrals {
 public:
  Integrals() = default;

  // Allocates storage for num_orbs spinors, zeroed, in the blocks of the
  // label sets `symmetry` (each with one label per spinor; none: one block).
  // With several ranks this is collective, and the two-electron integrals
  // are held once per node.
  Integrals(int num_orbs, std::vector<LabelSet> symmetry,
            const Comm& comm = Comm());
  explicit Integrals(int num_orbs, const Comm& comm = Comm())
      : Integrals(num_orbs, std::vector<LabelSet>(), comm) {}

  int num_orbs() const { return num_orbs_; }
  int num_elec() const { return num_elec_; }
  double core_energy() const { return core_energy_; }
  // The label sets the integrals obey.
  const std::vector<LabelSet>& symmetry() const { return symmetry_; }
  size_t num_stored() const { return two_body_.size(); }
  // Largest magnitude of an integral that was offered to SetOneBody,
  // AddChemist or SetAntisymOrdered but would change a label. Such integrals
  // are not stored.
  double max_forbidden() const { return max_forbidden_; }

  void set_num_elec(int n) { num_elec_ = n; }
  void set_core_energy(double e) { core_energy_ = e; }

  // h(p, q) = <p|h|q>.
  S OneBody(int p, int q) const { return one_body_[p * num_orbs_ + q]; }

  // <pq||rs> for p < q and r < s.
  S AntisymOrdered(int p, int q, int r, int s) const {
    const PairSlot bra = pair_slot_[PairIndex(p, q)];
    const PairSlot ket = pair_slot_[PairIndex(r, s)];
    if (bra.block != ket.block) return S(0.0);
    const size_t offset = block_offset_[bra.block];
    const size_t i = bra.position;
    const size_t j = ket.position;
    return i >= j ? two_body_[offset + i * (i + 1) / 2 + j]
                  : Conj(two_body_[offset + j * (j + 1) / 2 + i]);
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

  // The pairs r < s for which <pq||rs> can be nonzero, in increasing order
  // of s and then r; p < q.
  struct OrbitalPair {
    uint16_t first;
    uint16_t second;
  };
  const OrbitalPair* PartnersBegin(int p, int q) const {
    return block_pairs_.data() +
           block_start_[pair_slot_[PairIndex(p, q)].block];
  }
  const OrbitalPair* PartnersEnd(int p, int q) const {
    return block_pairs_.data() +
           block_start_[pair_slot_[PairIndex(p, q)].block + 1];
  }

  void SetOneBody(int p, int q, S value);
  // Sets <pq||rs> and, with it, <rs||pq>; p < q and r < s.
  void SetAntisymOrdered(int p, int q, int r, int s, S value);

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

  // Where a pair sits: its block, and its position among the pairs of that
  // block.
  struct PairSlot {
    uint32_t block;
    uint32_t position;
  };

  // The stored number that holds <pair||other pair> or its conjugate, and
  // whether it is the conjugate; nullptr if the pairs are in different
  // blocks.
  S* Stored(size_t bra, size_t ket, bool* conjugated);
  void NoteForbidden(S value);

  int num_orbs_ = 0;
  int num_elec_ = 0;
  double core_energy_ = 0.0;
  double max_forbidden_ = 0.0;
  std::vector<LabelSet> symmetry_;
  std::vector<PairSlot> pair_slot_;       // by PairIndex
  std::vector<size_t> block_offset_;      // into two_body_, by block
  std::vector<size_t> block_start_;       // into block_pairs_, by block
  std::vector<OrbitalPair> block_pairs_;  // the pairs of every block
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
// `symmetry` are the label sets the integrals are expected to obey; a set
// without labels takes ORBSYM of the file. An integral that would change a
// label is an error if its magnitude exceeds `symmetry_tolerance` and is
// dropped otherwise (Integrals::max_forbidden tells the largest).
//
// Collective: the root reads the file and every rank returns the integrals.
// Instantiated for double and Complex.
template <typename S>
Integrals<S> ReadFcidump(const std::string& path,
                         const std::vector<LabelSet>& symmetry,
                         double symmetry_tolerance, const Comm& comm = Comm());

// The same without symmetry labels.
template <typename S>
Integrals<S> ReadFcidump(const std::string& path, const Comm& comm = Comm()) {
  return ReadFcidump<S>(path, std::vector<LabelSet>(), 0.0, comm);
}

}  // namespace zshci

#endif  // ZSHCI_SRC_INTEGRALS_H_
