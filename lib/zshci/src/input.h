// Calculation input, read from a JSON file.
//
// Example:
//   {
//     "mode": "diag",
//     "integrals": "FCIDUMP",
//     "scalar": "auto",
//     "nroots": 2,
//     "prefix": "./scratch",
//     "dets": [[0, 1, 2, 3], [0, 1, 2, 5]],
//     "symmetry": [{"labels": [1, -1, 3, -3], "mod": 0},
//                  {"labels": [0, 0, 1, 1], "mod": 2}],
//     "restrictions": [{"orbitals": [0, 1, 2, 3], "min": 2, "max": 4}],
//     "select": {"schedule": [[0, 1e-3], [3, 1e-4]], "max_iter": 12},
//     "davidson": {"tol": 1e-8},
//     "pt": {"method": "semistochastic", "epsilon2": 1e-8,
//            "epsilon2_large": 1e-5, "target_error": 1e-5},
//     "rdm": {"one": true, "two": true, "transition": false}
//   }
// Comments (// and /* */) are allowed. Unknown keys are errors.

#ifndef ZSHCI_SRC_INPUT_H_
#define ZSHCI_SRC_INPUT_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "davidson.h"
#include "labels.h"

namespace zshci {

// Largest number of segments of the segment connection search.
constexpr int kMaxConnectSegments = 20;

enum class Mode { kSelect, kDiag };

// The number type of the calculation. kAuto: real if the integral file is in
// the real format, complex if it is in the complex format. kReal on a file in
// the complex format requires every imaginary part to be exactly zero; kComplex
// accepts both formats.
enum class ScalarChoice { kAuto, kReal, kComplex };

// A bound on the number of electrons in a set of spinors (0-based indices).
struct RestrictionInput {
  std::vector<int> orbitals;
  int min_elec = 0;
  int max_elec = 0;
};

struct SelectInput {
  // (first iteration, epsilon1) pairs: from that iteration on, a determinant
  // D_a is added when |H(a,i) c_i| > epsilon1 for some D_i in the space.
  std::vector<std::pair<int, double>> schedule;
  // Maximum number of iterations. If not given: six beyond the start of the
  // last schedule entry.
  int max_iter = 0;
  // Once the last epsilon1 is reached, the iterations stop when the mean
  // absolute change of the energies falls below this.
  double energy_tol = 1e-8;
  // Davidson threshold during the iterations; davidson.tol is used for the
  // final solution.
  double davidson_tol_loose = 5e-5;
  // Double-excitation integrals below this are left out of the heat-bath
  // tables.
  double integral_threshold = 1e-12;

  // epsilon1 of iteration `iter`.
  double Epsilon(int iter) const {
    double epsilon = schedule.front().second;
    for (const std::pair<int, double>& entry : schedule) {
      if (entry.first <= iter) epsilon = entry.second;
    }
    return epsilon;
  }
};

enum class PtMethod { kNone, kDeterministic, kStochastic, kSemistochastic };

struct PtInput {
  PtMethod method = PtMethod::kNone;
  // Terms with |H(a,i) c_i| <= epsilon2 are neglected.
  double epsilon2 = 1e-8;
  // Semistochastic only: the part with |H(a,i) c_i| > epsilon2_large is
  // evaluated deterministically.
  double epsilon2_large = 0.0;
  // Stochastic methods run one walker per core (thread of a rank). In every
  // iteration each walker draws num_samples determinants and produces one
  // estimate; the iterations stop when the standard error of all estimates
  // so far falls below target_error.
  int num_samples = 200;
  int min_iterations = 5;
  int max_iterations = 1000;
  double target_error = 1e-4;
  // Seed of the random streams; runs with equal seeds, rank and thread
  // counts repeat exactly.
  uint64_t seed = 1;
  // Number of passes over the external space (trades time for memory); for
  // the stochastic methods, per walker.
  int num_batches = 1;
};

struct RdmInput {
  bool one = false;
  bool two = false;
  // <0| p^+ q |j> for every root j > 0.
  bool transition = false;
};

struct Input {
  Mode mode = Mode::kSelect;
  std::string integrals = "FCIDUMP";
  ScalarChoice scalar = ScalarChoice::kAuto;
  std::string prefix = ".";
  int nroots = 1;

  // The determinants, each as the list of its occupied spinors (0-based):
  // the starting space in select mode, the whole space in diag mode.
  std::vector<std::vector<int>> dets;
  // Alternatively, an HDF5 file with a dataset "dets" as written to wfn.h5.
  std::string detfile;

  // The conserved label sets (labels.h); "symmetry" is an array of them, or
  // a single one written without the brackets. A set without labels takes
  // ORBSYM of the integral file ("labels": "fcidump"). Default: that, added
  // without a modulus.
  std::vector<LabelSet> symmetry = {LabelSet()};
  // The integrals must obey the labels: the largest integral that would
  // change a label may not exceed this. Such integrals are not stored.
  double symmetry_tolerance = 1e-8;
  // Occupation restrictions. Selection and perturbation theory only generate
  // determinants that satisfy all of them, and the starting determinants
  // must. Not applied in diag mode, where the list is taken as given.
  std::vector<RestrictionInput> restrictions;
  SelectInput select;
  DavidsonOptions davidson;
  PtInput pt;
  RdmInput rdm;

  // |H(i,j)| below this is not stored. Negative keeps every connected pair.
  // If not given: -1 when density matrices are requested, 0 otherwise.
  double hij_threshold = 0.0;
  // How connected determinants are found (see connect.h): "strings" (N-1 and
  // N-2 electron strings; no wasted tests, memory grows with the square of
  // the electron number) or "segments" (covers of four out of
  // `connect_segments` orbital segments; memory independent of the electron
  // number, at the price of testing unconnected pairs).
  bool connect_by_segments = false;
  int connect_segments = 8;
  // Strings only: number of passes (trades time for memory).
  int connect_batches = 1;
  // Compare the Hamiltonian and its product against slow reference
  // implementations; for testing on small cases.
  bool check = false;
  int output_level = 1;

  // The input with all defaults filled in, as JSON text.
  std::string resolved;
};

// Parses and validates JSON text. Throws std::runtime_error naming the
// offending key on any problem.
Input ParseInput(const std::string& text);

// Returns the content of a text file.
std::string ReadFile(const std::string& path);

}  // namespace zshci

#endif  // ZSHCI_SRC_INPUT_H_
