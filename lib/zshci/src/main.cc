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
#include <type_traits>

#ifdef __GLIBC__
#include <malloc.h>
#endif

#include "comm.h"
#include "det.h"
#include "diag.h"
#include "fci.h"
#include "input.h"
#include "integrals.h"
#include "select.h"

namespace zshci {
namespace {

template <int W, class Ints>
void Run(const Input& input, const Ints& ints, const Comm& comm) {
  switch (input.mode) {
    case Mode::kDiag:
      RunDiag<W, Ints>(input, ints, comm);
      return;
    case Mode::kSelect:
      RunSelect<W, Ints>(input, ints, comm);
      return;
    case Mode::kFci:
      RunFci(input, ints, comm);
      return;
  }
}

// Runs with the determinant width W for `words` words, trying W = 1, 2, ...
// up to kMaxDetWords; the widths are instantiated at compile time.
template <int W, class Ints>
void RunWithWidth(int words, const Input& input, const Ints& ints,
                  const Comm& comm) {
  if constexpr (W <= kMaxDetWords) {
    if (words == W) return Run<W, Ints>(input, ints, comm);
    return RunWithWidth<W + 1, Ints>(words, input, ints, comm);
  } else {
    throw std::runtime_error(
        "NORB = " + std::to_string(ints.num_orbs()) + " spinors need " +
        std::to_string(words) +
        " 64-bit words per determinant; this build "
        "supports " +
        std::to_string(kMaxDetWords) + " (" +
        std::to_string(64 * kMaxDetWords) +
        " spinors): rebuild with make MAX_DET_WORDS=" + std::to_string(words));
  }
}

// Prints what was read and runs with the width it needs.
template <class Ints>
void RunWithIntegrals(const Input& input, const Ints& ints, double seconds,
                      const Comm& comm) {
  std::printf("integrals: %s, %s, core energy %.12f, %.2f s\n",
              input.integrals.c_str(), ints.Describe().c_str(),
              ints.core_energy(), seconds);
  std::printf("largest integral forbidden by the labels: %.1e\n",
              ints.max_forbidden());
  RunWithWidth<1, Ints>(WordsForOrbitals(ints.num_orbs()), input, ints, comm);
}

// Reads the integrals of the kind the input asks for, as numbers of type S,
// and runs the calculation: the one place the kind is decided.
template <typename S>
void RunWithScalar(const Input& input, const Comm& comm) {
  const double time = omp_get_wtime();
  if (input.spatial_orbitals) {
    // Restricted spatial orbitals are real.
    if constexpr (std::is_same<S, double>::value) {
      const SpatialIntegrals<S> ints = ReadSpatialFcidump<S>(
          input.integrals, input.symmetry, input.symmetry_tolerance, comm);
      return RunWithIntegrals(input, ints, omp_get_wtime() - time, comm);
    } else {
      throw std::logic_error("spatial orbitals with complex numbers");
    }
  }
  if (input.unrestricted_orbitals) {
    // Unrestricted spatial orbitals are real.
    if constexpr (std::is_same<S, double>::value) {
      const UnrestrictedIntegrals ints = ReadUnrestrictedIntegrals(
          input.integrals, input.symmetry, input.symmetry_tolerance, comm);
      return RunWithIntegrals(input, ints, omp_get_wtime() - time, comm);
    } else {
      throw std::logic_error("unrestricted orbitals with complex numbers");
    }
  }
  const SpinorIntegrals<S> ints = ReadFcidump<S>(
      input.integrals, input.symmetry, input.symmetry_tolerance, comm);
  RunWithIntegrals(input, ints, omp_get_wtime() - time, comm);
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
  if (input.spatial_orbitals && FcidumpIsComplex(input.integrals, comm)) {
    throw std::runtime_error(
        input.integrals +
        ": \"orbitals\": \"spatial\" needs real integrals, but the file is "
        "in the complex format; give it as real numbers or use "
        "\"orbitals\": \"spinor\"");
  }
  if (input.spatial_orbitals || input.unrestricted_orbitals) {
    is_complex = false;
  } else if (input.scalar == ScalarChoice::kAuto) {
    is_complex = FcidumpIsComplex(input.integrals, comm);
  }
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
#ifdef __GLIBC__
  // Arrays from a megabyte on are mapped on their own and go back to the
  // system when they are released. By default the limit rises to 32 MB as
  // the program runs, and what is released below it stays with the process:
  // too much where many arrays of that size are filled and then copied into
  // one (the rows of the Hamiltonian, connect.h).
  mallopt(M_MMAP_THRESHOLD, 1 << 20);
#endif
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
