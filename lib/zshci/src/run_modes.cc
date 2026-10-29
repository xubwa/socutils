// One driver for one width and one kind of integrals (run_modes.h), chosen
// by the Makefile through
//
//   -DZSHCI_RUN_MODE_diag or -DZSHCI_RUN_MODE_select,
//   -DZSHCI_RUN_KIND_spatial, _unrestricted, _realspinor or _complexspinor,
//   -DZSHCI_RUN_WIDTH=W.

#include "run_modes.h"

#include "integrals.h"
#include "scalar.h"

#if defined(ZSHCI_RUN_MODE_diag)
#include "diag.h"
#define ZSHCI_RUN_DRIVER RunDiag
#elif defined(ZSHCI_RUN_MODE_select)
#include "select.h"
#define ZSHCI_RUN_DRIVER RunSelect
#else
#error "run_modes.cc needs ZSHCI_RUN_MODE_diag or ZSHCI_RUN_MODE_select"
#endif

#ifndef ZSHCI_RUN_WIDTH
#error "run_modes.cc needs ZSHCI_RUN_WIDTH"
#endif

namespace zshci {

#if defined(ZSHCI_RUN_KIND_spatial)
using RunInts = SpatialIntegrals<double>;
#elif defined(ZSHCI_RUN_KIND_unrestricted)
using RunInts = UnrestrictedIntegrals;
#elif defined(ZSHCI_RUN_KIND_realspinor)
using RunInts = SpinorIntegrals<double>;
#elif defined(ZSHCI_RUN_KIND_complexspinor)
using RunInts = SpinorIntegrals<Complex>;
#else
#error "run_modes.cc needs ZSHCI_RUN_KIND_<kind>"
#endif

template void ZSHCI_RUN_DRIVER<ZSHCI_RUN_WIDTH, RunInts, RunInts::Scalar>(
    const Input&, const RunInts&, const Comm&);

}  // namespace zshci
