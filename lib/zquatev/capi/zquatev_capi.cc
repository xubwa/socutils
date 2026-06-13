/*
 * zquatev_capi.cc -- plain-C ABI shim around Toru Shiozaki's zquatev
 * quaternionic eigensolver, so the solver can be loaded through ctypes
 * (no pybind11 dependency).
 *
 * This shim is part of socutils and is licensed Apache-2.0.  The solver it
 * calls (csrc/*.cc, csrc/*.h) is a verbatim copy of zquatev and remains under
 * its own BSD-2-Clause license (see ../LICENSE and the per-file headers).
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <complex>
#include "zquatev.h"

extern "C" {

/*
 * Diagonalize a Hermitian quaternionic matrix
 *
 *     ( A  -B* )
 *     ( B   A* )
 *
 * stored column-major in `mat` (leading dimension `n`, full dimension n = 2m),
 * referencing only the left half.  On success `mat` is overwritten with the
 * symmetry-adapted eigenvectors and `eig` (length n) is filled with the
 * eigenvalues (each Kramers pair duplicated: eig[m+i] = eig[i]).
 *
 * `mat` must point at interleaved (re, im) doubles, i.e. the buffer of a
 * numpy complex128 array.  Returns 0 on success, non-zero on failure.
 */
int zquatev_eigh(int n, double* mat, double* eig)
{
    std::complex<double>* D = reinterpret_cast<std::complex<double>*>(mat);
    const int info = ts::zquatev(n, D, n, eig);
    if (info == 0) {
        const int m = n / 2;
        for (int i = 0; i != m; ++i)
            eig[m + i] = eig[i];
    }
    return info;
}

} // extern "C"
