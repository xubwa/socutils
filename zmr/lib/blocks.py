'''
Block structure of Hermitian matrices from their pattern of exact zeros.

A matrix whose non-zero elements connect the indices only within groups
(the connected components of its graph) is block diagonal up to a
permutation, and its eigenproblem splits into independent problems.  The
sector Hamiltonians of a symmetry-adapted active space (zmr.active.symmetry)
have this structure exactly, because the charge-violating integrals are
removed before the matrices are built; without symmetry there is a single
component and the dense routines are used unchanged.
'''

import numpy

from socutils.zmr.lib import kernels
from socutils.zmr.lib import linalg as la


def components(a):
    '''Connected components of the graph of the non-zero elements of the
    square matrix a (either triangle may be empty).

    Returns (labels, ncomp): labels (n,) in order of first appearance.'''
    return kernels.components(a)


def groups(labels, ncomp):
    '''Index arrays (ascending) of the components.'''
    order = numpy.argsort(labels, kind='stable')
    bounds = numpy.searchsorted(labels[order], numpy.arange(ncomp + 1))
    return [order[bounds[b]:bounds[b + 1]] for b in range(ncomp)]


def eigh(a, labels=None, ncomp=None):
    '''Eigenvalues (ascending) and eigenvectors (columns) of a Hermitian
    matrix (lower triangle read), solved component by component.  Every
    eigenvector is supported on one component.'''
    if labels is None:
        labels, ncomp = components(a)
    if ncomp <= 1:
        return la.eigh(a)
    n = a.shape[0]
    parts = []
    for idx in groups(labels, ncomp):
        parts.append((idx,) + la.eigh(a[numpy.ix_(idx, idx)]))
    w = numpy.concatenate([wb for _, wb, _ in parts])
    perm = numpy.argsort(w, kind='stable')
    # column of every eigenvector in the sorted spectrum
    position = numpy.empty(n, dtype=numpy.int64)
    position[perm] = numpy.arange(n)
    v = numpy.zeros((n, n), dtype=numpy.result_type(a.dtype, numpy.float64))
    col = 0
    for idx, wb, vb in parts:
        v[numpy.ix_(idx, position[col:col + len(idx)])] = vb
        col += len(idx)
    return w[perm], v


def function_groups(rows, labels, ncomp):
    '''Independent groups of a set of vectors given as the rows of rows
    (M, D) over an index space with components (labels, ncomp): components
    are merged whenever one row is supported on both.

    Returns a list of (indices, row numbers), both ascending.  Vanishing
    rows belong to no group and components that no row touches are left
    out.  None is returned if nothing can be split (a single group that
    covers every index).'''
    import scipy.sparse
    from scipy.sparse.csgraph import connected_components
    touch = kernels.block_support(rows, labels, ncomp)
    t = scipy.sparse.csr_matrix(touch, dtype=numpy.int32)
    nmerged, merged = connected_components(t.T @ t, directed=False)
    used = touch.any(axis=0)
    comps = groups(labels, ncomp)
    live = touch.any(axis=1)
    row_group = numpy.where(live, merged[touch.argmax(axis=1)], -1)
    out = []
    for g in numpy.unique(merged[used]):
        members = numpy.nonzero((merged == g) & used)[0]
        idx = numpy.sort(numpy.concatenate([comps[c] for c in members]))
        out.append((idx, numpy.nonzero(row_group == g)[0]))
    if len(out) == 1 and len(out[0][0]) == rows.shape[1]:
        return None
    return out
