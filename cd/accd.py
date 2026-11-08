#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Atomic compact Cholesky (acCD) auxiliary basis sets.

For every atom (basis) of the molecule:

1. two-step block pivoted Cholesky decomposition of the one-centre ERIs with
   threshold ``tau`` (:func:`socutils.cd.cd.pivoted_cholesky`); the pivot
   shell pairs (a, b) define the product functions chi_a chi_b that the
   auxiliary basis has to represent;
2. candidate primitives r^L exp(-z r^2) for L = |la-lb|, |la-lb|+2, ...,
   la+lb, where z is alpha_i + beta_j (the primitives of a and b) mapped to
   the effective exponent of the component r^(la+lb) Y_L (same <r>, Lehtola
   2021, Appendix II);
3. per L, pivoted Cholesky of the Coulomb metric of the candidates
   (absolute threshold SELECT_THRESH) removes the linearly dependent ones;
4. per L, the product functions are fitted on the selected primitives in the
   Coulomb metric and the fitted radial functions compressed by an SVD in that
   metric (squared singular values >= FIT_THRESH); the result is one generally
   contracted shell.  The contraction is used only when it saves functions,
   nctr <= CONTRACT_RATIO * nprim; otherwise, and for atoms whose orbital basis
   is uncontracted, the selected primitives are used uncontracted.

The ERI error of the auxiliary basis is bounded from below by the two-centre
products, which no one-centre basis represents exactly: a tau much below 1e-4
adds functions without improving the integrals much.

Unlike the atom-centred rules of autoaux (Stoychev, Auer, Neese, J. Chem.
Theory Comput. 13, 554 (2017)), which limit L to l_max + l_inc, the angular
momenta come from the pivots and the products of polarization functions are
kept; they enter the virtual-virtual integrals (e.g. the particle-particle
ladder of CCSD).

The procedure follows S. Lehtola, J. Chem. Theory Comput. 17, 6886 (2021)
and J. Chem. Theory Comput. 19, 6242 (2023) (also in basis_set_exchange.auxgen)
and F. Aquilante, L. Gagliardi, T. B. Pedersen, R. Lindh, J. Chem. Phys. 130,
154107 (2009); here the integrals come from libcint and the pivots from the
contracted orbital basis.
'''

from math import gamma
import numpy
from pyscf import gto, lib
from pyscf.df import incore
from pyscf.lib import logger
from socutils.cd import cd

# pivoted Cholesky threshold of the Coulomb metric of the candidates (step 3);
# 1e-6 (or 1e-6 on the unit-diagonal metric) loses tight functions of Ag:
# energy errors of 1e-4 at small integral errors
SELECT_THRESH = 1e-8
# squared singular values kept by the compression (step 4)
FIT_THRESH = 1e-6
# general contraction only if it saves enough functions
CONTRACT_RATIO = .75
# libcint limits of one shell (cint.h)
ANG_MAX = 15
NPRIM_MAX = 64
NCTR_MAX = 64


def _select_exponents(L, exps, thresh=SELECT_THRESH):
    '''Pivoted Cholesky of the Coulomb metric of the (overlap-normalized)
    Gaussians r^L exp(-z r^2), absolute threshold; returns the exponents of
    the pivots, largest first.  The self-repulsion of a tight Gaussian grows
    as sqrt(z): unlike a threshold on the unit-diagonal metric this keeps the
    tight functions the energy is sensitive to and drops redundant diffuse
    ones.'''
    tmol = gto.M(atom=[('X', (0, 0, 0))], basis={'X': [[L, [float(e), 1.]] for e in exps]},
                 verbose=0)
    nl = 2 * L + 1
    S = tmol.intor('int2c2e')[::nl, ::nl]         # one m component; the same for all
    r = S.diagonal().copy()
    V = numpy.zeros((0, len(exps)))
    piv = []
    while len(piv) < len(exps):
        k = int(numpy.argmax(r))
        if r[k] < thresh:
            break
        v = (S[:, k] - V.T.dot(V[:, k])) / numpy.sqrt(r[k])
        V = numpy.vstack([V, v])
        r = r - v**2
        r[k] = 0
        piv.append(k)
    return numpy.sort(numpy.asarray(exps)[piv])[::-1]


def _effective_exponent_factor(L, n):
    '''alpha_eff / alpha of the Gaussian r^L exp(-alpha_eff r^2) with the same
    <r> as r^n exp(-alpha r^2) (Lehtola 2021, Appendix II eq 16); 1 for n = L.
    A product of shells la, lb has the components r^(la+lb) Y_L, L < la+lb,
    which pure Gaussians of the bare exponent alpha_i+beta_j fit poorly.'''
    s = (gamma(L + 2) * gamma(n + 1.5)) / (gamma(L + 1.5) * gamma(n + 2))
    return s * s


def _is_uncontracted(amol):
    return all(amol.bas_nprim(i) == 1 for i in range(amol.nbas))


def _atom_shells(amol, tau, max_memory, log):
    _, pivots = cd.pivoted_cholesky(amol, tau, 1e-2, max_memory, 0, active_set=True)
    pairs = sorted({b for grp, U in pivots for b in grp})
    ao_loc = amol.ao_loc_nr()

    cand = {}
    prod_idx = []
    for si, sj in pairs:
        la, lb = amol.bas_angular(si), amol.bas_angular(sj)
        z = (amol.bas_exp(si)[:, None] + amol.bas_exp(sj)[None, :]).ravel()
        for L in range(abs(la - lb), la + lb + 1, 2):
            cand.setdefault(L, set()).update((z * _effective_exponent_factor(L, la + lb)).tolist())
        prod_idx.append(cd._shell_pair_columns(ao_loc, si, sj)[0])
    prod_idx = numpy.unique(numpy.hstack(prod_idx))

    if max(cand) > ANG_MAX:
        raise NotImplementedError('acCD: auxiliary L = %d exceeds the libcint limit %d'
                                  % (max(cand), ANG_MAX))
    exps = {L: _select_exponents(L, sorted(cand[L], reverse=True)) for L in sorted(cand)}
    if _is_uncontracted(amol):
        return [[L, [float(e), 1.]] for L in exps for e in exps[L]]

    auxmol = gto.M(atom=[('X', (0, 0, 0))], verbose=0,
                   basis={'X': [[L, [float(e), 1.]] for L in exps for e in exps[L]]})
    S = auxmol.intor('int2c2e')
    V = incore.aux_e2(amol, auxmol, 'int3c2e', aosym='s2ij')[prod_idx].T   # (naux, nprod)
    aux_loc = auxmol.ao_loc_nr()
    shells = []
    for L in exps:
        first = numpy.array([aux_loc[i] for i in range(auxmol.nbas) if auxmol.bas_angular(i) == L])
        SL = S[numpy.ix_(first, first)]
        w, v = numpy.linalg.eigh(SL)
        keep = w > 1e-13 * w.max()
        w, v = w[keep], v[:, keep]
        # fitted radial functions SL^-1 V of all product functions and m
        # components, in the orthonormal basis X = v w^-1/2 of the metric:
        # G = X^T V
        X = v / numpy.sqrt(w)
        G = numpy.hstack([X.T.dot(V[first + m]) for m in range(2 * L + 1)])
        u, sig = numpy.linalg.svd(G, full_matrices=False)[:2]
        nctr = int((sig**2 >= FIT_THRESH).sum())
        nprim = len(first)
        if nctr == 0:
            continue
        if nctr <= CONTRACT_RATIO * nprim and nprim <= NPRIM_MAX and nctr <= NCTR_MAX:
            C = X.dot(u[:, :nctr])          # metric-orthonormal contracted functions
            shells.append([L] + [[float(e)] + C[k].tolist() for k, e in enumerate(exps[L])])
            log.debug1('    L = %d: %d primitives, %d contracted', L, nprim, nctr)
        else:
            shells.extend([L, [float(e), 1.]] for e in exps[L])
            log.debug1('    L = %d: %d primitives (%d directions), uncontracted', L, nprim, nctr)
    return shells


def accd_basis(mol, tau=1e-4, max_memory=None, verbose=None):
    '''acCD auxiliary basis of mol, as a basis dict for df.DF(mol, auxbasis=...).

    Args:
        tau : threshold of the atomic Cholesky decompositions.
        max_memory : MB the atomic decompositions may use in addition to the
            memory in use (default mol.max_memory).

    Returns:
        dict {atom label: shells}, one entry per distinct atom label.
    '''
    if mol.cart:
        raise NotImplementedError('acCD auxiliary basis for cartesian orbital basis')
    log = logger.new_logger(mol, verbose)
    if max_memory is None:
        max_memory = mol.max_memory
    out = {}
    for label, _ in mol._atom:
        if label in out:
            continue
        amol = gto.M(atom=[(label, (0, 0, 0))], basis={label: mol._basis[label]},
                     spin=None, verbose=0)
        out[label] = _atom_shells(amol, tau, lib.current_memory()[0] + max_memory, log)
        log.info('acCD auxiliary basis for %s: %d functions', label,
                 sum((2 * s[0] + 1) * (len(s[1]) - 1) for s in out[label]))
    return out
