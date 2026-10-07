"""Relativistic recontraction of a primitive set with the X2CAMF atomic SCF.

The recipe (basisGenerator of X2CAMF, xx2cbasis of CFOUR): an atomic
X2C-1e Hartree-Fock in the uncontracted primitives; the occupied atomic orbitals become the
contracted functions, and selected primitives are added as uncontracted functions.
  spin-free sets ("-X2C", "-SF"):  SF-X2C-1e HF, one contraction per occupied (n, l).
  j-adapted sets ("-SO", "-SOG", "-SOB"): X2C-1e HF whose one-electron Hamiltonian contains
      the X2CAMF spin-orbit integrals of a 4c DHF (DC, +Gaunt, +Breit); for l >= 1 the
      contractions of j = l-1/2 come first, then those of j = l+1/2.
"""
import numpy as np

from . import engine, genbas

SUFFIX_HAM = {'sf': 'sfx2c1e', 'so': 'x2camf', 'sog': 'x2camf-g', 'sob': 'x2camf-b'}


def pattern_from(entry, tol=1e-10):
    """Contraction pattern of an existing general-contracted GENBAS entry:
    {l: (ncontracted, [indices of primitives kept uncontracted])}; a column with a single
    non-zero coefficient counts as an uncontracted primitive."""
    pat = {}
    for l, (exps, coef) in entry.shells.items():
        nz = [np.flatnonzero(abs(coef[:, k]) > tol) for k in range(coef.shape[1])]
        unc = [int(i[0]) for i in nz if len(i) == 1 and len(exps) > 1]
        if len(exps) == 1:
            unc = [0]
        n = coef.shape[1] - len(unc)
        if l in entry.so_info():        # j-adapted: both kappas are listed
            assert n == 2 * entry.so_info()[l], (l, n, entry.so_info())
            n //= 2
        pat[l] = (n, sorted(unc))
    return pat


def contract(element, prims, pattern, kind='sf', aoc=True, gau_nuc=False, name=None,
             zero_uncontracted=False, **atom_kwargs):
    """Contracted basis as a genbas.Entry.

    prims   : {l: exponents}
    pattern : {l: (n, [primitive indices kept uncontracted])}; n = contracted functions per
              l (spin-free) or per kappa (j-adapted, l >= 1); l absent from pattern or with
              n = 0 stay fully uncontracted.  Indices refer to exponents in descending order.
    kind    : 'sf', 'so', 'sog' or 'sob'.
    aoc     : average-of-configuration SCF for open shells (as the generators do).
    zero_uncontracted : zero the rows of the uncontracted primitives in the contracted
              functions (the stand-alone basisGenerator does this, xx2cbasis does not).
    """
    result = engine.atom(element, prims, SUFFIX_HAM[kind], aoc=aoc, gau_nuc=gau_nuc,
                         **atom_kwargs)
    if not result.converged:
        raise RuntimeError('atomic SCF did not converge')
    shells, so_info = {}, []
    for l in sorted(result.prims):
        exps = np.array(result.prims[l])
        n, unc = pattern.get(l, (0, list(range(len(exps)))))
        if n == 0:
            unc = list(range(len(exps)))
        kappas = [2 * l + 1] if (kind == 'sf' or l == 0) else [2 * l - 1, 2 * l + 1]
        cols = []
        for two_j in kappas:
            c = result.block(l, two_j)['c'][:len(exps), :n].copy()
            # fix the phase: positive largest coefficient
            c *= np.sign(c[np.argmax(abs(c), axis=0), np.arange(n)])
            if zero_uncontracted:
                c[unc, :] = 0.0
            cols.append(c)
        for i in unc:
            e = np.zeros((len(exps), 1))
            e[i] = 1.0
            cols.append(e)
        shells[l] = (exps, np.hstack(cols))
        if kind != 'sf' and l >= 1 and n > 0:
            so_info.append((l, n))
    comment = 'X2CAMF atomic %s, %s' % (SUFFIX_HAM[kind], 'AOC' if aoc else 'fractional occ.')
    if kind != 'sf' and so_info:
        comment = '%d  ' % len(so_info) + ' '.join('%d %d' % x for x in so_info)
    el = engine.ELEMENTS[engine.charge(element)].upper()
    entry = genbas.Entry('%s:%s' % (el, name or 'NEW-' + kind.upper()), comment, shells)
    entry.scf = result
    return entry


def compare(a, b):
    """Largest deviation between two entries: exponents (relative) and the spaces spanned by
    the contracted functions of each l (largest principal-angle sine), plus max |dc| of the
    columns after normalising the sign."""
    out = {}
    for l in sorted(a.shells):
        ea, ca = a.shells[l]
        eb, cb = b.shells[l]
        if ca.shape != cb.shape:
            out[l] = 'shape %s vs %s' % (ca.shape, cb.shape)
            continue
        s = engine.radial_power_matrix(l, ea, 0)
        dc = abs(abs(ca) - abs(cb)).max()
        # overlap of the two contracted spaces
        na = ca / np.sqrt(np.einsum('ik,ij,jk->k', ca, s, ca))
        la = np.linalg.cholesky(na.T @ s @ na)
        lb = np.linalg.cholesky(cb.T @ s @ cb)
        m = np.linalg.solve(la, na.T @ s @ cb) @ np.linalg.inv(lb).T
        sv = np.linalg.svd(m, compute_uv=False)
        out[l] = 'max|dexp/exp| %.1e  max|dc| %.2e  1-min(sv) %.1e' % (
            abs(ea / eb - 1).max(), dc, 1 - sv.min())
    return out
