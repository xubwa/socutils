// zshci: selected configuration interaction for spinor Hamiltonians.
//
// Usage: zshci input.json
//        mpirun -np N zshci input.json     (when built with MPI=1)

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "comm.h"
#include "det.h"
#include "diag.h"
#include "input.h"
#include "integrals.h"
#include "select.h"

namespace zshci {
namespace {

template <int W, typename S>
void Run(const Input& input, const Integrals<S>& ints, const Comm& comm) {
  switch (input.mode) {
    case Mode::kDiag:
      RunDiag<W, S>(input, ints, comm);
      return;
    case Mode::kSelect:
      RunSelect<W, S>(input, ints, comm);
      return;
  }
}

template <typename S>
void RunWithWidth(const Input& input, const Integrals<S>& ints,
                  const Comm& comm) {
  switch (WordsForOrbitals(ints.num_orbs())) {
    case 1:
      return Run<1, S>(input, ints, comm);
    case 2:
      return Run<2, S>(input, ints, comm);
    case 3:
      return Run<3, S>(input, ints, comm);
    case 4:
      return Run<4, S>(input, ints, comm);
    default:
      throw std::runtime_error(
          "NORB = " + std::to_string(ints.num_orbs()) + " exceeds the " +
          std::to_string(64 * kMaxDetWords) + " spinors this build supports");
  }
}

// Reads the integrals as numbers of type S and runs the calculation.
template <typename S>
void RunWithScalar(const Input& input, const Comm& comm) {
  const double time = omp_get_wtime();
  const Integrals<S> ints =
      input.spatial_orbitals
          ? ReadSpatialFcidump<S>(input.integrals, input.symmetry,
                                  input.symmetry_tolerance, comm)
          : ReadFcidump<S>(input.integrals, input.symmetry,
                           input.symmetry_tolerance, comm);
  std::printf("integrals: %s, %s, %d %s, core energy %.12f, %.2f s\n",
              input.integrals.c_str(), ScalarTraits<S>::kName,
              ints.is_spatial() ? ints.num_spatial() : ints.num_orbs(),
              ints.is_spatial() ? "spatial orbitals" : "spinors",
              ints.core_energy(), omp_get_wtime() - time);
  const size_t num_pairs =
      static_cast<size_t>(ints.num_orbs()) * (ints.num_orbs() - 1) / 2;
  std::printf(
      "two-electron integrals: %zu stored, %.1f%% of what spinors without "
      "symmetry would need; largest integral forbidden by the labels: %.1e\n",
      ints.num_stored(),
      100.0 * ints.num_stored() / (num_pairs * (num_pairs + 1) / 2.0),
      ints.max_forbidden());
  RunWithWidth(input, ints, comm);
}

void Main(const char* input_path, const Comm& comm) {
  // The root reads the files; the other ranks receive their content. Only
  // the root prints, line by line so that progress shows up in a log file.
  std::string input_text;
  if (comm.is_root()) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    input_text = ReadFile(input_path);
  } else if (std::freopen("/dev/null", "w", stdout) == nullptr) {
    throw std::runtime_error("cannot silence the output of a non-root rank");
  }
  comm.BcastString(&input_text);
  const Input input = ParseInput(input_text);
  std::printf("zshci, %d rank(s) x %d thread(s)\n", comm.size(),
              omp_get_max_threads());
  std::printf("input:\n%s\n", input.resolved.c_str());

  bool is_complex = input.scalar == ScalarChoice::kComplex;
  if (input.scalar == ScalarChoice::kAuto) {
    is_complex = FcidumpIsComplex(input.integrals, comm);
  }
  if (input.spatial_orbitals) is_complex = false;
  if (is_complex) {
    RunWithScalar<Complex>(input, comm);
  } else {
    RunWithScalar<double>(input, comm);
  }
  std::printf("zshci finished normally\n");
}

}  // namespace
}  // namespace zshci

int main(int argc, char** argv) {
  const zshci::Comm comm = zshci::Comm::Init(&argc, &argv);
  if (argc != 2) {
    if (comm.is_root()) std::fprintf(stderr, "usage: %s input.json\n", argv[0]);
    zshci::Comm::Finalize();
    return EXIT_FAILURE;
  }
  try {
    zshci::Main(argv[1], comm);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "zshci: error on rank %d: %s\n", comm.rank(),
                 e.what());
    // The other ranks may be waiting for this one; take them all down.
    if (comm.size() > 1) comm.Abort(EXIT_FAILURE);
    zshci::Comm::Finalize();
    return EXIT_FAILURE;
  }
  zshci::Comm::Finalize();
  return EXIT_SUCCESS;
}
