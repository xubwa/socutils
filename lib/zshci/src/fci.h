// The "fci" mode: full CI or GAS CI over the spinors of the integral file
// (gas_space.h), diagonalized by the Davidson of fci_davidson.h. One
// process, threads; FCI.md describes the input and the files.

#ifndef ZSHCI_SRC_FCI_H_
#define ZSHCI_SRC_FCI_H_

#include "comm.h"
#include "input.h"
#include "integrals.h"

namespace zshci {

// Instantiated for double and Complex.
template <typename S>
void RunFci(const Input& input, const Integrals<S>& ints, const Comm& comm);

}  // namespace zshci

#endif  // ZSHCI_SRC_FCI_H_
