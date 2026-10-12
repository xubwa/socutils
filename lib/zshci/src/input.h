// Calculation input, read from a JSON file.
//
// Example:
//   {
//     "mode": "diag",
//     "integrals": "FCIDUMP",
//     "scalar": "auto",
//     "orbitals": "spinor",
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

// The kind of selected CI (select.h).
//   heat_bath  add D_a when |H(a,i) c_i| > epsilon1 for some D_i ("schedule")
// The others score every determinant D_a outside the space from its
// first-order interaction with the wave function, V_a = sum_i H(a,i) c_i:
//   cipsi  by its second-order energy |V_a|^2 / (E_a - E); the best ones are
//          added so that the space grows by the factor "growth" per
//          iteration up to "target_dets"
//   asci   by its first-order amplitude |V_a| / (E_a - E), V_a summed over
//          the "core_dets" determinants of largest coefficient only; growth
//          and target as for cipsi
//   aci    by its second-order energy; those left out are the smallest ones,
//          whose energies sum to at most "sigma"
// None of them removes determinants from the space, unless "prune" is set:
//   asci   the determinants of the space are then ranked together with the
//          candidates, by the same amplitude, and only the target_dets (or,
//          while growing, growth times the present number) highest of both
//          stay in the space
//   aci    the determinants of largest coefficients that hold all but
//          "gamma" * sigma of the squared norm generate the candidates and
//          stay; every other determinant is scored from them, those of the
//          space included, and the lowest ones, whose energies sum to at
//          most sigma, are left out
// With any method, "prune_coefficient": c takes out, after every
// diagonalization, the determinants whose coefficients (root-sum-square
// over the roots) are below c; the states are then those of what remains.
// A determinant taken out of the space stays in the list of the program,
// with coefficient zero, so that the stored Hamiltonian is neither reordered
// nor rebuilt, and is treated like any determinant outside the space; it
// comes back without cost if it is selected again. The list is cleared of
// such determinants at the end and, if "compact_above" is set, whenever
// their share of the list exceeds it; the Hamiltonian is then made afresh.
enum class SelectMethod { kHeatBath, kCipsi, kAsci, kAci };

struct SelectInput {
  SelectMethod method = SelectMethod::kHeatBath;
  // cipsi, asci: size of the space at which the selection stops, and the
  // factor by which it grows per iteration.
  size_t target_dets = 0;
  double growth = 2.0;
  // asci: number of determinants that generate candidates; 0 means all.
  size_t core_dets = 0;
  // asci, aci: whether determinants leave the space (see above).
  bool prune = false;
  // aci with prune: see above.
  double gamma = 1.0;
  // Determinants with coefficients below this leave the space; 0: none.
  double prune_coefficient = 0.0;
  // Share of determinants out of the space above which the list is
  // compacted during the iterations; 0: only at the end.
  double compact_above = 0.0;
  // aci: the energy left out, in Hartree.
  double sigma = 0.0;
  // cipsi, asci, aci: only terms with |H(a,i) c_i| above this enter V_a
  // (0: all single and double excitations).
  double epsilon_generate = 0.0;
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
  // "orbitals": "unrestricted" reads the integrals of an unrestricted
  // reference, different spatial orbitals for the two spins, from an HDF5
  // file (ReadUnrestrictedIntegrals); determinants and labels as for
  // "spatial", density matrices over the spin orbitals.
  bool unrestricted_orbitals = false;
  // "orbitals": "spatial" reads integrals over real spatial orbitals, which
  // all electrons share (a restricted reference); the determinants are then
  // over the spin orbitals 2P (alpha) and 2P + 1 (beta) of spatial orbital P,
  // the spin projection is conserved, and the labels of "symmetry" are per
  // spatial orbital. The default, "spinor", takes the orbitals of the file as
  // they are.
  bool spatial_orbitals = false;
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
  // How connected determinants are found (see connect.h). "groups", the
  // default: tables of the strings of two groups of spinors
  // (connect_groups.h); "group_a" lists the spinors of the first group
  // (default: those with an even index, which for spin orbitals are the
  // alpha ones); "relations" says how related strings are found there:
  // "subgroups" (default), each group divided once more, or "keys", which
  // keeps a number of keys per string that grows with the square of its
  // electrons. "strings": N-1 and N-2 electron strings of the whole
  // determinant; no wasted tests, memory grows with the square of the
  // electron number. "segments": covers of four out of `connect_segments`
  // orbital segments; least memory, at the price of testing unconnected
  // pairs.
  bool connect_by_segments = false;
  bool connect_by_groups = true;
  std::vector<int> connect_group_a;
  bool connect_subgroups = true;
  int connect_segments = 8;
  // Strings only: number of passes (trades time for memory).
  int connect_batches = 1;
  // "hamiltonian": "direct" does not store the matrix: every product with a
  // vector works the connections out again from the tables of the group
  // method (which it implies) and evaluates the elements on the way. Slow
  // per product, but the memory of the matrix is saved. Default "stored".
  bool direct_hamiltonian = false;
  // "incremental": true (default) extends the stored Hamiltonian and the
  // string tables by what the determinants added in an iteration bring, for
  // which the root keeps the sorted keys of all strings. false makes both
  // afresh in every iteration and releases the keys each time: slower, less
  // memory.
  bool incremental = true;
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
