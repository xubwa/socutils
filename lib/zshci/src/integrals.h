// One- and two-electron integrals over spinors, and the FCIDUMP reader.

#ifndef ZSHCI_SRC_INTEGRALS_H_
#define ZSHCI_SRC_INTEGRALS_H_

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include "comm.h"

namespace zshci {

using Complex = std::complex<double>;

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
  Complex OneBody(int p, int q) const { return one_body_[p * num_orbs_ + q]; }

  // (pq|rs) in chemists' notation: p, q belong to electron 1.
  Complex TwoBody(int p, int q, int r, int s) const {
    return two_body_[PairIndex(p, q, r, s)];
  }

  // (pp|qq) and (pq|qp); real for a Hermitian two-electron operator.
  double Direct(int p, int q) const { return direct_[p * num_orbs_ + q]; }
  double Exchange(int p, int q) const { return exchange_[p * num_orbs_ + q]; }

  void SetOneBody(int p, int q, Complex value) {
    one_body_[p * num_orbs_ + q] = value;
  }
  void SetTwoBody(int p, int q, int r, int s, Complex value) {
    two_body_[PairIndex(p, q, r, s)] = value;
  }

  // Fills the Direct/Exchange tables; call once after all integrals are set.
  void Finalize();

  // Copies the root's integrals to every rank. Collective.
  void Share(const Comm& comm);

 private:
  // Only the particle-exchange symmetry (pq|rs) = (rs|pq) is used.
  size_t PairIndex(int p, int q, int r, int s) const {
    const size_t pq = static_cast<size_t>(p) * num_orbs_ + q;
    const size_t rs = static_cast<size_t>(r) * num_orbs_ + s;
    const size_t hi = pq > rs ? pq : rs;
    const size_t lo = pq > rs ? rs : pq;
    return hi * (hi + 1) / 2 + lo;
  }

  int num_orbs_ = 0;
  int num_elec_ = 0;
  double core_energy_ = 0.0;
  std::vector<int> labels_;
  std::vector<Complex> one_body_;
  SharedArray<Complex> two_body_;
  std::vector<double> direct_;
  std::vector<double> exchange_;
};

// Reads a complex FCIDUMP file: a namelist header with NORB, NELEC and
// (optionally) ORBSYM, then lines "(re,im) p q r s" with 1-based indices.
// "p q 0 0" is a one-body element, whose Hermitian partner is filled in as
// well; "0 0 0 0" is the core energy; "p 0 0 0" (orbital energy) is skipped.
// Throws std::runtime_error on a malformed file.
//
// Collective: the root reads the file and every rank returns the integrals.
Integrals ReadFcidump(const std::string& path, const Comm& comm = Comm());

}  // namespace zshci

#endif  // ZSHCI_SRC_INTEGRALS_H_
