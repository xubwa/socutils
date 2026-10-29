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
//
// Spatial orbitals. When all electrons see the same real spatial orbitals
// (a restricted reference), the integrals (PQ|RS) over n spatial orbitals
// have eightfold symmetry and take n^4 / 8 numbers. SpatialIntegrals holds
// them that way and presents the 2n spin orbitals p = 2P (alpha), 2P + 1
// (beta) through the same functions, <pq||rs> being put together from two
// spatial integrals on each call. The spin projection is then the first
// label set.

#ifndef ZSHCI_SRC_INTEGRALS_H_
#define ZSHCI_SRC_INTEGRALS_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "comm.h"
#include "labels.h"
#include "scalar.h"

namespace zshci {

// The kinds of integrals are classes of their own (INTEGRALS_DESIGN.md):
// SpinorIntegrals (label blocks of <pq||rs>) and SpatialIntegrals ((PQ|RS)
// over restricted spatial orbitals). They share one inline interface -
// Scalar, num_orbs, OneBody, AntisymOrdered, Antisym, PairEnergy,
// SingleBound - on which the kernels are templated, and the data that is
// alike (IntegralTables); what is called once goes through IntegralsBase.

// What every kind offers to code that is not templated on it.
class IntegralsBase {
 public:
  virtual ~IntegralsBase() = default;
  // The integrals as the program describes them at the start.
  virtual std::string Describe() const = 0;
  // Bytes of the stored two-electron integrals.
  virtual size_t MemoryBytes() const = 0;
};

// The data all kinds have: sizes, energies, labels, the one-body integrals,
// and the tables made from the two-body ones (Finalize). S is double or
// Complex.
template <typename S>
class IntegralTables : public IntegralsBase {
 public:
  using Scalar = S;

  int num_orbs() const { return num_orbs_; }
  int num_elec() const { return num_elec_; }
  double core_energy() const { return core_energy_; }
  // The label sets the integrals obey.
  const std::vector<LabelSet>& symmetry() const { return symmetry_; }
  // Whether the orbitals are spin orbitals 2P, 2P + 1 with the spin
  // projection as the first label set.
  bool has_spin_orbitals() const { return spin_orbitals_; }
  // Largest magnitude of an integral that was offered but would change a
  // label. Such integrals are not stored.
  double max_forbidden() const { return max_forbidden_; }

  void set_num_elec(int n) { num_elec_ = n; }
  void set_core_energy(double e) { core_energy_ = e; }

  // h(p, q) = <p|h|q>.
  S OneBody(int p, int q) const { return one_body_[p * num_orbs_ + q]; }
  void SetOneBody(int p, int q, S value);

  // An upper bound of |<D'|H|D>| for the single excitation p -> q of any
  // determinant: |h(q,p)| plus the num_elec - 1 largest |<qk||pk>|. Lets the
  // excitation generator skip singles without evaluating them.
  double SingleBound(int p, int q) const {
    // Before Finalize there is no bound: nothing is skipped.
    return single_bound_.empty() ? 1e300 : single_bound_[p * num_orbs_ + q];
  }
  // <pq||pq> = (pp|qq) - (pq|qp), the interaction energy of two occupied
  // spinors, from a table of its own.
  double PairEnergy(int p, int q) const {
    return pair_energy_[p * num_orbs_ + q];
  }

 protected:
  IntegralTables() = default;

  // <pq||rs> for any indices from AntisymOrdered of the kind.
  template <class Kind>
  static S AntisymOf(const Kind& kind, int p, int q, int r, int s) {
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
    const S value = kind.AntisymOrdered(p, q, r, s);
    return negative ? -value : value;
  }
  // The pair energies and the singles bound, from the kind's Antisym.
  template <class Kind>
  void FillTables(const Kind& kind);
  // Copies what is in these tables from the root. Collective.
  void ShareTables(const Comm& comm);
  void NoteForbidden(S value) {
    max_forbidden_ = std::max(max_forbidden_, std::abs(value));
  }

  int num_orbs_ = 0;
  bool spin_orbitals_ = false;
  int num_elec_ = 0;
  double core_energy_ = 0.0;
  double max_forbidden_ = 0.0;
  std::vector<LabelSet> symmetry_;
  std::vector<S> one_body_;
  std::vector<double> pair_energy_;
  std::vector<double> single_bound_;
};

// Antisymmetrized integrals over spinors, in the blocks of the labels.
template <typename S>
class SpinorIntegrals : public IntegralTables<S> {
 public:
  static constexpr bool kSpatial = false;

  SpinorIntegrals() = default;
  // Allocates storage for num_orbs spinors, zeroed, in the blocks of the
  // label sets `symmetry` (each with one label per spinor; none: one block).
  // With several ranks this is collective, and the two-electron integrals
  // are held once per node.
  SpinorIntegrals(int num_orbs, std::vector<LabelSet> symmetry,
                  const Comm& comm = Comm());
  explicit SpinorIntegrals(int num_orbs, const Comm& comm = Comm())
      : SpinorIntegrals(num_orbs, std::vector<LabelSet>(), comm) {}

  // Storage over 2 * num_spatial spin orbitals, 2P (alpha) and 2P + 1 (beta),
  // whose alpha and beta spatial parts may differ (an unrestricted
  // reference): antisymmetrized integrals in the blocks of the spin
  // projection and of `spatial_symmetry`. Collective.
  static SpinorIntegrals SpinOrbitals(
      int num_spatial, const std::vector<LabelSet>& spatial_symmetry,
      const Comm& comm = Comm());

  bool is_spatial() const { return false; }
  size_t num_stored() const { return two_body_.size(); }

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
    return this->AntisymOf(*this, p, q, r, s);
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

  // Sets <pq||rs> and, with it, <rs||pq>; p < q and r < s.
  void SetAntisymOrdered(int p, int q, int r, int s, S value);
  // Adds what the integral (pq|rs) = value, together with its Hermitian
  // partner (qp|sr) = conj(value), contributes to the antisymmetrized
  // integrals. Call once for every such pair of integrals; (rs|pq) is the
  // same integral as (pq|rs) and must not be added again.
  void AddChemist(int p, int q, int r, int s, S value);

  // Fills the pair energies and the singles bound; call once after all
  // integrals are set.
  void Finalize() { this->FillTables(*this); }
  // Copies the root's integrals to every rank. Collective.
  void Share(const Comm& comm);

  std::string Describe() const override;
  size_t MemoryBytes() const override { return two_body_.size() * sizeof(S); }

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

  std::vector<PairSlot> pair_slot_;       // by PairIndex
  std::vector<size_t> block_offset_;      // into two_body_, by block
  std::vector<size_t> block_start_;       // into block_pairs_, by block
  std::vector<OrbitalPair> block_pairs_;  // the pairs of every block
  SharedArray<S> two_body_;
};

// Integrals (PQ|RS) over n restricted spatial orbitals, eightfold
// symmetric, n^4 / 8 numbers, real only (instantiated for S = double),
// presented as the spin orbitals p = 2P
// (alpha), 2P + 1 (beta): <pq||rs> is put together from two spatial
// integrals on each call. The spin projection is the first label set.
template <typename S>
class SpatialIntegrals : public IntegralTables<S> {
 public:
  static constexpr bool kSpatial = true;

  SpatialIntegrals() = default;
  // Storage for num_spatial spatial orbitals, zeroed. `spatial_symmetry`
  // holds one label per spatial orbital in each set; symmetry() then lists
  // the spin projection (labels +1 for alpha, -1 for beta) followed by these
  // sets, per spin orbital. Collective.
  SpatialIntegrals(int num_spatial,
                   const std::vector<LabelSet>& spatial_symmetry,
                   const Comm& comm = Comm());

  bool is_spatial() const { return true; }
  int num_spatial() const { return num_spatial_; }
  size_t num_stored() const { return two_body_.size(); }

  // <pq||rs> = (pr|qs) - (ps|qr) over spin orbitals, each term present when
  // the spins of its two charge distributions match; p < q, r < s.
  S AntisymOrdered(int p, int q, int r, int s) const {
    const int sp = p & 1, sq = q & 1, sr = r & 1, ss = s & 1;
    S value(0.0);
    if (sp == sr && sq == ss) {
      value += two_body_[SpatialIndex(p >> 1, r >> 1, q >> 1, s >> 1)];
    }
    if (sp == ss && sq == sr) {
      value -= two_body_[SpatialIndex(p >> 1, s >> 1, q >> 1, r >> 1)];
    }
    return value;
  }
  S Antisym(int p, int q, int r, int s) const {
    return this->AntisymOf(*this, p, q, r, s);
  }
  // (PQ|RS) over spatial orbitals.
  S SpatialTwoBody(int p, int q, int r, int s) const {
    return two_body_[SpatialIndex(p, q, r, s)];
  }

  // h(P, Q) for both spins, and (PQ|RS) with all its eight permutations.
  void SetSpatialOneBody(int p, int q, S value);
  void SetSpatialTwoBody(int p, int q, int r, int s, S value);

  void Finalize() { this->FillTables(*this); }
  void Share(const Comm& comm);

  std::string Describe() const override;
  size_t MemoryBytes() const override { return two_body_.size() * sizeof(S); }

 private:
  static size_t Triangle(size_t a, size_t b) {
    return a >= b ? a * (a + 1) / 2 + b : b * (b + 1) / 2 + a;
  }
  size_t SpatialIndex(int p, int q, int r, int s) const {
    return Triangle(Triangle(p, q), Triangle(r, s));
  }

  int num_spatial_ = 0;
  SharedArray<S> two_body_;
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

// A file may instead be an HDF5 file with the two-electron integrals already
// antisymmetrized, which is read in one go (written by
// socutils.tools.fcidump_rel.from_integrals_h5). Spinor pairs p < q are
// numbered P = q (q - 1) / 2 + p. Attributes: format = "antisym-1", norb,
// nelec, ecore, complex (0 or 1). Datasets:
//   h1        (norb, norb, 2)  real and imaginary part of h(p, q)
//   pair_bra  (m)              P(p, q) of each stored integral
//   pair_ket  (m)              P(r, s), not above pair_bra
//   eri       (m, 2)           real and imaginary part of <pq||rs>
//   orbsym    (norb)           optional; taken as ORBSYM is
// with <rs||pq> = conj(<pq||rs>) implied and integrals not listed zero.

// Whether the file is in the complex format, judged from its first integral
// line (HDF5: from the attribute). Collective: the root looks and every rank
// returns the answer.
bool FcidumpIsComplex(const std::string& path, const Comm& comm = Comm());

// Reads a file in either format. With S = double an imaginary part other
// than zero is an error. Throws std::runtime_error on a malformed file.
//
// `symmetry` are the label sets the integrals are expected to obey; a set
// without labels takes ORBSYM of the file. An integral that would change a
// label is an error if its magnitude exceeds `symmetry_tolerance` and is
// dropped otherwise (max_forbidden tells the largest).
//
// Collective: the root reads the file and every rank returns the integrals.
// Instantiated for double and Complex.
template <typename S>
SpinorIntegrals<S> ReadFcidump(const std::string& path,
                               const std::vector<LabelSet>& symmetry,
                               double symmetry_tolerance,
                               const Comm& comm = Comm());

// The same without symmetry labels.
template <typename S>
SpinorIntegrals<S> ReadFcidump(const std::string& path,
                               const Comm& comm = Comm()) {
  return ReadFcidump<S>(path, std::vector<LabelSet>(), 0.0, comm);
}

// Reads integrals over real spatial orbitals from a FCIDUMP file in the real
// format: "value i j k l" is (ij|kl), with any of its eight permutations
// standing for all of them. The result presents spin orbitals
// (SpatialIntegrals).
//
// `symmetry` holds label sets per spatial orbital. A set without labels
// takes ORBSYM of the file as irreducible representations of D2h or a
// subgroup numbered as in Molpro, whose products are the exclusive or of the
// numbers minus one: it becomes up to three sets of labels modulo 2.
//
// Collective. Real integrals only (S = double).
template <typename S>
SpatialIntegrals<S> ReadSpatialFcidump(const std::string& path,
                                       const std::vector<LabelSet>& symmetry,
                                       double symmetry_tolerance,
                                       const Comm& comm = Comm());

// Reads the integrals of an unrestricted reference from an HDF5 file with
// the real datasets, for n spatial orbitals of each spin,
//   h1_a, h1_b                 (n, n)
//   eri_aa, eri_bb, eri_ab     (n, n, n, n), (pq|rs) with p, q of the first
//                              and r, s of the second spin named
// and the optional attributes ecore (double) and nelec (int). The result
// holds spin orbitals (SpinorIntegrals::SpinOrbitals); `symmetry` is per
// spatial orbital, and a set without labels is not allowed.
//
// Collective. Instantiated for double and Complex.
template <typename S>
SpinorIntegrals<S> ReadUnrestrictedIntegrals(
    const std::string& path, const std::vector<LabelSet>& symmetry,
    double symmetry_tolerance, const Comm& comm = Comm());

}  // namespace zshci

#endif  // ZSHCI_SRC_INTEGRALS_H_
