// Result files.
//
//   <prefix>/shci.e              nroots native doubles: the total energies,
//                                including the PT correction if one was run.
//   <prefix>/Dice_i_j.rdm1.h5    datasets rdm1_real, rdm1_imag (n^2 values).
//   <prefix>/Dice_i_i.rdm2.h5    datasets rdm2_real, rdm2_imag (n^4 values).
//   <prefix>/wfn.h5              determinants, CI vectors, energies, input.
//
// The first three keep the names and the element order written by the
// ZSHCI/DetCI programs of this repository, so existing readers (socutils
// hci/shci.py) work unchanged:
//   rdm1 flat index q * n + p               holds <i| p^+ q |j>,
//   rdm2 flat index ((p*n + q)*n + r)*n + s holds conj(<i| p^+ q^+ s r |i>).

#ifndef ZSHCI_SRC_OUTPUT_H_
#define ZSHCI_SRC_OUTPUT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "integrals.h"
#include "rdm.h"

namespace zshci {

void WriteEnergies(const std::string& prefix,
                   const std::vector<double>& energies);

void WriteOneRdm(const std::string& prefix, int bra, int ket,
                 const DensityMatrices& dm);

void WriteTwoRdm(const std::string& prefix, int root,
                 const DensityMatrices& dm);

// The wave function: `det_words` is the N x W table of occupation words.
struct WaveFunction {
  int num_orbs = 0;
  int num_elec = 0;
  int det_width = 0;
  std::vector<uint64_t> det_words;                // N * W
  std::vector<std::vector<Complex>> coefficients;  // nroots x N
  std::vector<double> energies;                    // variational, total
  std::vector<double> pt_corrections;              // empty without PT
  std::vector<double> pt_errors;
  std::vector<int> labels;
  std::string input;  // resolved input, JSON text
};

void WriteWaveFunction(const std::string& path, const WaveFunction& wfn);

// Reads the determinant table of a wfn.h5 file; sets *det_width.
std::vector<uint64_t> ReadDetWords(const std::string& path, int* det_width);

}  // namespace zshci

#endif  // ZSHCI_SRC_OUTPUT_H_
