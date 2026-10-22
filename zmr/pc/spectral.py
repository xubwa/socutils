'''
PC-NEVPT2 first-order spaces from explicit CI vectors.

For one perturber class the PC functions of external label e are
E_e O_X |Phi_c Psi_I>; their active parts O_X |Psi_I> are CI vectors of the
target sector.  The span of the chosen functions (all model states for the
'union' contraction, the ket state only for 'state') is orthonormalized,
discarding metric eigenvalues below lindep, and H_act is diagonalized inside
it (the Koopmans-matrix problem of PC-NEVPT2, solved once per class).  The
resulting orthonormal functions b_j with energies mu_j replace the sector
eigenstates in the uncontracted contraction engines,

    T'^I[j, X] = <b_j|O_X|Psi_I>.

contracted_space_det works in the determinant basis and needs the sector
spectrum only when the functions span the whole sector (caaa / aaav with
many states, where PC = UC); otherwise the cost is one product H_act Q and
an eigenproblem of the size of the span:

    functions M < sector dimension D   eigenvectors of the M x M metric
                                       (Gram matrix of the functions),
                                       q = Phi^T conj(y) / sqrt(s), followed
                                       by one Cholesky-QR step; K = Q^+ (H Q)
    M >= D, full rank (Cholesky test)  the sector eigenstates themselves
    M >= D, rank deficient             eigenvectors of the D x D Gram matrix
                                       of the determinant coefficients, then K

Both Gram matrices have the same non-zero eigenvalues, the metric eigenvalues
that lindep refers to.

If the sector Hamiltonian is block diagonal (symmetry-adapted active
orbitals, zmr.active.symmetry) and every function lies in one block, the
whole construction is done block by block (zmr.lib.blocks); blocks that are
connected by a function are merged first, so model states that mix the
symmetry blocks only cost efficiency.

The metric and the active-space zeroth-order matrix of the contracted
functions,

    S^{IJ}[X, Y] = <Psi_I|O_X^+ O_Y|Psi_J>          = sum_k conj(T^I[k,X]) T^J[k,Y]
    K^{IJ}[X, Y] = <Psi_I|O_X^+ H_act O_Y|Psi_J>    = sum_k conj(T^I[k,X]) lam_k T^J[k,Y]

(metric_matrices) are the quantities an RDM-based implementation has to
reproduce.
'''

import numpy

from socutils.zmr.lib import blocks as lb
from socutils.zmr.lib import linalg as la


def orthonormal_span(rows, lindep=1e-10):
    '''Orthonormal basis of the span of the functions given as the rows of
    rows (M, D); metric eigenvalues <= lindep are removed.

    Returns (q, full): q (D, m) with orthonormal columns, or (None, True) if
    the functions span the whole D-dimensional space.'''
    m, d = rows.shape
    if m < d:
        # rows rows^+ is the complex conjugate of the metric <f_X|f_Y>
        s, y = la.eigh(la.herk(rows, 'N', fill=False))
        keep = s > lindep
        if not keep.any():
            return numpy.zeros((d, 0), dtype=rows.dtype), False
        q = la.gemm(rows, y[:, keep], 'C', 'N')
        numpy.conjugate(q, out=q)
        q /= numpy.sqrt(s[keep])
        # the columns are orthonormal only to eps * cond(metric)
        return la.orthonormalize(q), False
    # Gram matrix of the columns rows^T: G = rows^T conj(rows) = conj(rows^+ rows)
    g = la.herk(rows, 'C', fill=False)
    numpy.conjugate(g, out=g)
    if la.is_positive_definite(g - lindep * numpy.eye(d)):
        return None, True
    s, q = la.eigh(g)
    return numpy.ascontiguousarray(q[:, s > lindep]), False


def contracted_space(rows, apply_h, sector, lindep=1e-10):
    '''(B, mu): orthonormal basis of the span of the functions rows (M, D)
    that diagonalizes H_act, and its energies.

    apply_h: function q (D, m) -> H_act q
    sector: function returning the eigen-decomposition of H_act in the whole
        space (object with vecs, lam), or None if it is not available;
        called only if the functions span the whole space.'''
    q, full = orthonormal_span(rows, lindep)
    if full:
        sec = sector()
        if sec is not None:
            return sec.vecs, sec.lam
        q = numpy.eye(rows.shape[1], dtype=rows.dtype)
    if q.shape[1] == 0:
        return q, numpy.zeros(0)
    mu, w = la.eigh(la.gemm(q, apply_h(q), 'C', 'N'))
    return la.gemm(q, w), mu


class _Eigen:
    def __init__(self, h):
        self.lam, self.vecs = la.eigh(h)


def contracted_space_det(rows, hmat, sector, lindep=1e-10, blocks=None):
    '''Orthonormal, H_act-diagonal basis of the span of the functions rows
    (M, D) in the determinant basis.

    hmat: callable returning the dense sector Hamiltonian (D, D)
    sector: callable returning the sector eigen-decomposition (used only if
        the functions span the whole sector)
    blocks: (labels, nblocks) of the block-diagonal structure of the sector
        Hamiltonian (ActiveSpace.blocks).  If the functions respect it (every
        function lies in one block, as for symmetry-pure model states), the
        problem is solved block by block.

    Returns (B, mu): B (D, m) orthonormal columns, mu (m,).'''
    parts = None
    if blocks is not None and blocks[1] > 1:
        parts = lb.function_groups(rows, *blocks)
    if parts is None:
        return contracted_space(rows, lambda q: la.gemm(hmat(), q), sector, lindep)
    h = hmat()
    pieces = []
    for idx, funcs in parts:
        hb = numpy.ascontiguousarray(h[numpy.ix_(idx, idx)])
        sub = numpy.ascontiguousarray(rows[numpy.ix_(funcs, idx)])
        b, mu = contracted_space(sub, lambda q: la.gemm(hb, q), lambda: _Eigen(hb), lindep)
        if len(mu):
            pieces.append((idx, b, mu))
    m = sum(len(mu) for _, _, mu in pieces)
    basis = numpy.zeros((rows.shape[1], m), dtype=rows.dtype)
    energies = numpy.empty(m)
    col = 0
    for idx, b, mu in pieces:
        cols = numpy.arange(col, col + len(mu))
        basis[numpy.ix_(idx, cols)] = b
        energies[cols] = mu
        col += len(mu)
    return basis, energies


def metric_matrices(t, lam):
    '''S[I, J, X, Y] and K[I, J, X, Y] of the PC functions O_X |Psi_I>.'''
    s = numpy.einsum('ikx,jky->ijxy', t.conj(), t, optimize=True)
    k = numpy.einsum('ikx,k,jky->ijxy', t.conj(), lam, t, optimize=True)
    return s, k
