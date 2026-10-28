// Result files, all in the directory `prefix`.
//
//   shci.e          nroots native doubles: the total energies, including the
//                   perturbative correction if one was computed.
//   wfn.h5          determinants, CI vectors, energies, the resolved input.
//   rdm_<i>_<j>.h5  density matrices between states i and j, as arrays of
//                   the number type of the calculation (real or complex) in
//                   the convention of PySCF:
//                     dm1[p,q]     = <i| p^+ q |j>              shape (n,n)
//                     dm2[p,q,r,s] = <i| p^+ r^+ s q |i>        shape (n,n,n,n)
//                   over spinors or, with "orbitals": "spatial", summed over
//                   the spins and over the n spatial orbitals (PySCF's
//                   spin-traced dm1 and dm2).
//                   dm2 is present only for i = j and when requested. A
//                   reader gets the arrays as they are (h5py: f['dm1'][()]);
//                   no conjugation or reordering is needed.

#ifndef ZSHCI_SRC_OUTPUT_H_
#define ZSHCI_SRC_OUTPUT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "rdm.h"
#include "scalar.h"

namespace zshci {

void WriteEnergies(const std::string& prefix,
                   const std::vector<double>& energies);

// The functions below that depend on the number type S are instantiated for
// double and Complex.

// Writes rdm_<bra>_<ket>.h5 with dm1 and, if dm.two is not empty, dm2.
template <typename S>
void WriteDensityMatrices(const std::string& prefix, int bra, int ket,
                          const DensityMatrices<S>& dm);

// The wave function: `det_words` is the N x W table of occupation words.
template <typename S>
struct WaveFunction {
  int num_orbs = 0;
  int num_elec = 0;
  int det_width = 0;
  std::vector<uint64_t> det_words;                // N * W
  std::vector<std::vector<S>> coefficients;       // nroots x N
  std::vector<double> energies;                    // variational, total
  std::vector<double> pt_corrections;              // empty without PT
  std::vector<double> pt_errors;
  std::vector<int> labels;      // label sets x spinors, set by set
  std::vector<int> label_mods;  // one modulus per label set
  std::string input;  // resolved input, JSON text
  // Checkpoints of the select mode: the iteration and whether its
  // eigenvectors are converged (1) or those of the iteration before,
  // padded with zeros (0). Not written if negative.
  int select_iteration = -1;
  int select_solved = 0;
};

// The coefficients go to "ci_real" and, for a complex calculation, "ci_imag".
template <typename S>
void WriteWaveFunction(const std::string& path, const WaveFunction<S>& wfn);

// The select-mode checkpoint state of a file written by WriteWaveFunction;
// false if it has none.
bool ReadSelectState(const std::string& path, int* iteration, int* solved);

// The epsilon1 of the heat-bath selection that made the determinants of a
// wfn.h5 or select.h5 file (of the iteration it was written in, or of the
// last entry of the schedule); 0 if the file does not say.
double ReadSelectionThreshold(const std::string& path);

// Reads the determinant table of a wfn.h5 file; sets *det_width.
std::vector<uint64_t> ReadDetWords(const std::string& path, int* det_width);

// Reads the CI vectors of a wfn.h5 file into `vectors`, one per root, if the
// file holds `num_roots` vectors over `num_dets` determinants; otherwise
// returns false and leaves `vectors` alone. The imaginary parts are dropped
// when S is double. Instantiated for double and Complex.
template <typename S>
bool ReadCoefficients(const std::string& path, size_t num_dets,
                      size_t num_roots, std::vector<std::vector<S>>* vectors);

}  // namespace zshci

#endif  // ZSHCI_SRC_OUTPUT_H_
