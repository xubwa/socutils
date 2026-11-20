// The drivers of the "diag" and "select" modes, declared without their
// definitions, so that main.cc does not compile them. Each one, for one
// determinant width W and one kind of integrals, is compiled in a
// translation unit of its own from run_modes.cc; the Makefile builds one for
// every width up to kMaxDetWords, and these compile in parallel.

#ifndef ZSHCI_SRC_RUN_MODES_H_
#define ZSHCI_SRC_RUN_MODES_H_

#include "comm.h"
#include "input.h"

namespace zshci {

// Defined in diag.h and select.h, where S defaults to Ints::Scalar.
template <int W, class Ints, typename S>
void RunDiag(const Input& input, const Ints& ints, const Comm& comm);
template <int W, class Ints, typename S>
void RunSelect(const Input& given, const Ints& ints, const Comm& comm);

}  // namespace zshci

#endif  // ZSHCI_SRC_RUN_MODES_H_
