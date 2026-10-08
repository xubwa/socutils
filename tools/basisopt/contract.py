"""Relativistic recontraction of a primitive set with the X2CAMF atomic SCF.

The recipe (basisGenerator of X2CAMF, xx2cbasis of CFOUR): an atomic
X2C-1e Hartree-Fock in the uncontracted primitives; the occupied atomic orbitals become the
contracted functions, and selected primitives are added as uncontracted functions.
  spin-free sets ("-X2C", "-SF"):  SF-X2C-1e HF, one contraction per occupied (n, l).
  j-adapted sets ("-SO", "-SOG", "-SOB"): X2C-1e HF whose one-electron Hamiltonian contains
      the X2CAMF spin-orbit integrals of a 4c DHF (DC, +Gaunt, +Breit); for l >= 1 the
      contractions of j = l-1/2 come first, then those of j = l+1/2.

Command line (appends a GENBAS entry; genbas.to_pyscf() converts an entry to a PySCF basis):

  # recontract the primitives of a PySCF set with its own contraction pattern
  python -m socutils.tools.basisopt.contract O --basis aug-cc-pvtz --kind sf \
         --name APVTZ-SF --out GENBAS
  # the same with an added tight p, j-adapted, Gaunt in the spin-orbit integrals
  python -m socutils.tools.basisopt.contract O --basis aug-cc-pvtz --add p:225.713 \
         --kind sog --name APVTZ+P1-SOG --out GENBAS
  # primitives and pattern of an existing GENBAS entry, and a comparison with it
  python -m socutils.tools.basisopt.contract Xe --genbas GENBAS --entry XE:PWCVQZ-SO \
         --kind so --name PWCVQZ-SO-NEW --out GENBAS.new --compare
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


def pattern_from_pyscf(element, basis):
    """(prims, pattern) of a PySCF basis-set name: the primitive exponents {l: [...]} and
    the contraction pattern {l: (ncontracted, [uncontracted primitive indices])}; a shell
    with a single primitive counts as an uncontracted primitive."""
    from pyscf import gto
    cols = {}
    for sh in gto.basis.load(basis, element):
        l, rows = sh[0], [r for r in sh[1:] if not isinstance(r, int)]
        for k in range(1, len(rows[0])):
            cols.setdefault(l, []).append([r[0] for r in rows if abs(r[k]) > 1e-12])
    prims, pattern = {}, {}
    for l, cl in cols.items():
        exps = sorted({e for c in cl for e in c}, reverse=True)
        unc = sorted({exps.index(c[0]) for c in cl if len(c) == 1})
        ncont = sum(len(c) > 1 for c in cl)
        if len(exps) == 1:
            unc, ncont = [0], 0
        prims[l], pattern[l] = exps, (ncont, unc)
    return prims, pattern


def pattern_occupied(element, prims, keep=None, gau_nuc=False):
    """Contraction pattern for an uncontracted set: every occupied atomic orbital contracted,
    and the primitives `keep` = {l: [k, ...]} left uncontracted, counted FROM THE SMALLEST
    exponent (1 = most diffuse), which is how Dyall's papers prescribe contracted sets
    ("add the first and third s primitives ...").  Angular momenta without occupied
    orbitals stay uncontracted."""
    nocc = {}
    for n_, l_, two_j, f, e in engine.atom(element, prims, 'sfx2c1e', gau_nuc=gau_nuc).occupied():
        nocc[l_] = max(nocc.get(l_, 0), n_ - l_)
    pattern = {}
    for l, exps in prims.items():
        n = len(exps)
        if l not in nocc:
            pattern[l] = (0, list(range(n)))
            continue
        ks = sorted(set((keep or {}).get(l, [])))
        if ks and (ks[0] < 1 or ks[-1] > n):
            raise ValueError('l = %d: primitive numbers must be in 1..%d' % (l, n))
        pattern[l] = (nocc[l], sorted(n - k for k in ks))
    return pattern


def add_primitives(prims, pattern, added, uncontract_added=False):
    """New (prims, pattern) with the exponents `added` = {l: [...]} merged in.  The added
    primitives enter the contracted functions only, unless uncontract_added."""
    prims = {l: list(v) for l, v in prims.items()}
    pattern = dict(pattern)
    for l, new in added.items():
        old = sorted(prims.get(l, []), reverse=True)
        n, unc = pattern.get(l, (0, list(range(len(old)))))
        keep = [old[i] for i in unc] + (list(new) if uncontract_added or n == 0 else [])
        merged = sorted(old + list(new), reverse=True)
        if min([a / b for a, b in zip(merged, merged[1:])] or [9]) < 1.2:
            raise ValueError('added l=%d exponents too close to existing ones' % l)
        prims[l] = merged
        pattern[l] = (n, sorted(merged.index(e) for e in keep))
    return prims, pattern


def contract(element, prims, pattern, kind='sog', aoc=True, gau_nuc=False, name=None,
             zero_uncontracted=False, **atom_kwargs):
    """Contracted basis as a genbas.Entry.

    prims   : {l: exponents}
    pattern : {l: (n, [primitive indices kept uncontracted])}; n = contracted functions per
              l (spin-free) or per kappa (j-adapted, l >= 1); l absent from pattern or with
              n = 0 stay fully uncontracted.  Indices refer to exponents in descending order.
    kind    : 'sog' (default), 'sob', 'so' or 'sf'.  'sog' also serves Dirac-Coulomb-Breit
              calculations (the gauge term changes the contraction by < 1e-6 in the valence
              splittings, rare gases Ne-Og); 'so' (no Gaunt) differs in the spinor energies
              of heavy atoms by mEh and is only for calculations without the Gaunt term.
    aoc     : average-of-configuration SCF for open shells (as the generators do).
    zero_uncontracted : zero the rows of the uncontracted primitives in the contracted
              functions (the stand-alone basisGenerator does this, xx2cbasis does not).
    """
    result = engine.atom(element, prims, SUFFIX_HAM[kind], aoc=aoc, gau_nuc=gau_nuc,
                         **atom_kwargs)
    if not result.converged:
        raise RuntimeError('atomic SCF did not converge')
    nocc = {}
    for n_, l_, two_j, f, e in result.occupied():
        nocc[l_] = max(nocc.get(l_, 0), n_ - l_)
    for l, (n, unc) in pattern.items():
        if n > nocc.get(l, 0):
            raise ValueError('l = %d: %d contracted functions requested but only %d occupied '
                             'shells; the recipe contracts occupied atomic orbitals only'
                             % (l, n, nocc.get(l, 0)))
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


def contraction_error(entry, element, hamiltonian='x2camf-g', gau_nuc=False, result=None):
    """Spinor energies of the occupied shells in the contracted space of `entry` against the
    uncontracted primitives: the converged (fractional-occupation) two-component Fock matrix
    of the uncontracted atom is diagonalised in the span of the contracted functions of each
    kappa (j-adapted entries: each kappa with its own contractions).  A non-self-consistent
    estimate; the contraction error enters in second order.

    Returns [(n, l, two_j, eps_uncontracted, eps_contracted)]."""
    import scipy.linalg
    prims = {l: list(e) for l, (e, c) in entry.shells.items()}
    # `result`: an engine.atom() result in the same primitives, to avoid recomputing it
    r = result or engine.atom(element, prims, hamiltonian, gau_nuc=gau_nuc)
    so = entry.so_info()
    out = []
    for b in r.blocks:
        l, two_j = b['l'], b['two_j']
        coef = entry.shells[l][1]
        if l in so:      # columns: n of j=l-1/2, n of j=l+1/2, then the shared primitives
            n = so[l]
            own = range(0, n) if two_j == 2 * l - 1 else range(n, 2 * n)
            coef = coef[:, list(own) + list(range(2 * n, coef.shape[1]))]
        f, s = coef.T @ b['fock'] @ coef, coef.T @ b['ovlp'] @ coef
        e = scipy.linalg.eigh(f, s, eigvals_only=True)
        for i, occ in enumerate(b['occ']):
            if abs(occ) > 1e-10:
                out.append((i + l + 1, l, two_j, b['e'][i], e[i]))
    return out


def print_contraction_error(entry, element, hamiltonian='x2camf-g', gau_nuc=False):
    if engine.charge(element) == 1:
        # one electron: the fractional-occupation Fock matrix of the check contains a
        # self-interaction that the configuration-average SCF of the contraction does not
        print('  contraction check: not meaningful for a one-electron atom, skipped')
        return
    rows = contraction_error(entry, element, hamiltonian, gau_nuc)
    print('  contraction check (%s, spinor energies in Eh):' % hamiltonian)
    eps = {}
    for n, l, two_j, e0, e1 in rows:
        eps[(n, l, two_j)] = (e0, e1)
        print('    %d%s%d/2  uncontracted %14.8f  contracted %14.8f  diff %9.2e' % (
            n, 'spdfghi'[l], two_j, e0, e1, e1 - e0))
    for (n, l, two_j), (e0, e1) in eps.items():
        if two_j == 2 * l + 1 and (n, l, 2 * l - 1) in eps:
            d0 = e0 - eps[(n, l, 2 * l - 1)][0]
            d1 = e1 - eps[(n, l, 2 * l - 1)][1]
            print('    splitting %d%s: uncontracted %.4f cm-1, contracted %.4f cm-1, ratio %.5f'
                  % (n, 'spdfghi'[l], d0 * 219474.6313705, d1 * 219474.6313705, d1 / d0))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('element')
    src = ap.add_argument_group('source of the primitives and of the contraction pattern')
    src.add_argument('--basis', help='PySCF basis-set name')
    src.add_argument('--genbas', help='GENBAS file (with --entry)')
    src.add_argument('--entry', help="entry of --genbas, e.g. 'O:APVTZ-X2C'")
    ap.add_argument('--occupied', action='store_true',
                    help='for an uncontracted source: contract every occupied atomic orbital '
                    '(instead of taking the contraction pattern of the source)')
    ap.add_argument('--keep', default='', help='with --occupied: primitives left uncontracted, '
                    'counted from the smallest exponent as in Dyall\'s prescriptions, '
                    'e.g. "s:1,3;p:1,3;d:1,2;f:1,3"')
    ap.add_argument('--add', default='', help='added primitives, e.g. p:225.713 or '
                    'p:1223.0/182.873,d:50.1; they enter the contracted functions only')
    ap.add_argument('--uncontract-added', action='store_true',
                    help='also keep the added primitives as uncontracted functions')
    ap.add_argument('--kind', default='sog', choices=sorted(SUFFIX_HAM),
                    help='sog (default) / sob / so: j-adapted with the X2CAMF spin-orbit '
                    'integrals of a 4c DC-Gaunt / DC-Breit / DC DHF; sf: SF-X2C-1e')
    ap.add_argument('--name', required=True, help='name of the new set (without the element)')
    ap.add_argument('--out', default='GENBAS', help='GENBAS file to append to')
    ap.add_argument('--gau-nuc', action='store_true', help='Gaussian nucleus (default: point)')
    ap.add_argument('--fractional', action='store_true',
                    help='fractional-occupation SCF instead of the configuration average')
    ap.add_argument('--check', action='store_true',
                    help='print the contraction error of the occupied spinor energies and '
                    'spin-orbit splittings (X2CAMF with Gaunt, uncontracted Fock matrix)')
    ap.add_argument('--compare', action='store_true',
                    help='compare with the source entry (--genbas/--entry, no --add)')
    args = ap.parse_args()

    if args.genbas:
        ref = genbas.read(args.genbas, args.entry)
        prims = {l: list(e) for l, (e, c) in ref.shells.items()}
        pattern = pattern_from(ref)
    elif args.basis:
        ref = None
        prims, pattern = pattern_from_pyscf(args.element, args.basis)
    else:
        ap.error('give --basis or --genbas/--entry')
    if args.occupied:
        keep = {'spdfghi'.index(item.split(':')[0]): [int(x) for x in item.split(':')[1].split(',')]
                for item in args.keep.split(';') if item}
        pattern = pattern_occupied(args.element, prims, keep, gau_nuc=args.gau_nuc)
    if args.add:
        added = {'spdfghi'.index(item.split(':')[0]): [float(x) for x in item.split(':')[1].split('/')]
                 for item in args.add.split(',')}
        prims, pattern = add_primitives(prims, pattern, added, args.uncontract_added)
    entry = contract(args.element, prims, pattern, args.kind, aoc=not args.fractional,
                     gau_nuc=args.gau_nuc, name=args.name)
    genbas.append(args.out, entry)
    print('%s appended to %s  (atomic SCF energy %.8f Eh)' % (entry.name, args.out, entry.scf.e_scf))
    for l in sorted(entry.shells):
        exps, coef = entry.shells[l]
        n, unc = pattern.get(l, (0, []))
        print('  %s: %2d primitives -> %2d functions (%s)' % (
            'spdfghi'[l], len(exps), coef.shape[1],
            'uncontracted' if n == 0 else '%d contracted%s + %d primitives' % (
                n, ' per kappa' if args.kind != 'sf' and l > 0 else '', len(unc))))
    if args.check:
        print_contraction_error(entry, args.element, gau_nuc=args.gau_nuc)
    if args.compare and ref is not None and not args.add:
        for l, v in compare(entry, ref).items():
            print('  compare with %s, l=%d: %s' % (ref.name, l, v))


if __name__ == '__main__':
    main()
