"""Checks of LN:DYVTZ-{SF,SOG}-T07 for Ln3+ 4f^n (configuration average, Gaussian nucleus, X2CAMF DC-Gaunt AMF):
 1. truncation: full dyall.v3z vs primitives >= 0.07 (both uncontracted): total energy, 4f/5s/5p spinor energies,
    4f spin-orbit splitting (zeta_4f = 2/7 [eps(4f7/2)-eps(4f5/2)]), <r^-3>_4f, Slater integrals F2/F4/F6 of the 4f
    (large component), which together fix the free-ion 4f^n multiplet structure;
 2. contraction: spinor energies and splittings of the SOG set in its contracted space vs its primitives
    (contract.contraction_error); the contracted 4f equals the SCF 4f of the reference state, so items 1 give
    also the contracted-set values."""
import sys, math, numpy as np
from socutils.tools.basisopt import engine, contract, genbas
sys.argv += [] ; import importlib
mk = importlib.import_module('make_ln')
HA2CM = 219474.63
def radial(b, i, r):
    l = b['l']; ex = np.array(b['exp']); nb = len(ex)
    c = b['c'][:nb, i]
    norm = np.array([1 / math.sqrt(0.5 * math.gamma(l + 1.5) / (2 * x) ** (l + 1.5)) for x in ex])
    R = (c * norm) @ (r[None, :] ** l * np.exp(-np.outer(ex, r ** 2)))
    return R / math.sqrt(np.trapezoid(R ** 2 * r ** 2, r))
def slater(R, r, k):
    rho = R ** 2 * r ** 2
    rl, rg = np.minimum.outer(r, r), np.maximum.outer(r, r)
    return np.trapezoid(np.trapezoid(np.outer(rho, rho) * rl ** k / rg ** (k + 1), r, axis=1), r)
r = np.concatenate([np.geomspace(1e-4, 0.05, 300), np.linspace(0.05, 12, 2500)[1:]])
for el in sys.argv[1:]:
    full, prims, pattern = mk.build(el)
    occ = mk.occ_ion(mk.NF[el])
    res = {}
    for lab, p in (('full v3z', full), ('>= 0.07', prims)):
        a = engine.atom(el, p, 'x2camf-g', gau_nuc=True, aoc=True, occ=occ, max_iter=1000, conv=1e-9)
        assert a.converged, (el, lab)
        atm = a
        b5, b7 = a.block(3, 5), a.block(3, 7)
        R7 = radial(b7, 0, r); R5 = radial(b5, 0, r)
        F = [slater(R7, r, k) for k in (2, 4, 6)]
        res[lab] = dict(E=a.e_scf, e5=a.eps(4, 3, 5), e7=a.eps(4, 3, 7), zeta=2 / 7 * (a.eps(4, 3, 7) - a.eps(4, 3, 5)),
                        r3=a.r_power(4, 3, 7, -3), r3b=a.r_power(4, 3, 5, -3), F=F, e5s=a.eps(5, 0, 1), e5p=a.eps(5, 1, 3))
    print('== %s3+ 4f%d   (X2CAMF DC-Gaunt AMF, configuration average)' % (el, mk.NF[el]))
    f, t = res['full v3z'], res['>= 0.07']
    print('   E_SCF            full %.8f  truncated %.8f  diff %+.3f mEh' % (f['E'], t['E'], (t['E'] - f['E']) * 1e3))
    for k, lab, u in (('e5', 'eps 4f5/2 (Eh)', 1), ('e7', 'eps 4f7/2 (Eh)', 1), ('e5s', 'eps 5s (Eh)', 1), ('e5p', 'eps 5p3/2 (Eh)', 1)):
        print('   %-16s full %12.6f  truncated %12.6f  diff %+.1e' % (lab, f[k], t[k], t[k] - f[k]))
    print('   zeta_4f (cm-1)   full %10.2f  truncated %10.2f  ratio %.5f' % (f['zeta'] * HA2CM, t['zeta'] * HA2CM, t['zeta'] / f['zeta']))
    print('   <r-3> 4f7/2 (au) full %10.5f  truncated %10.5f  ratio %.5f   (4f5/2: %.5f / %.5f)' % (f['r3'], t['r3'], t['r3'] / f['r3'], f['r3b'], t['r3b']))
    for k, a_, b_ in zip((2, 4, 6), f['F'], t['F']):
        print('   F%d(4f,4f) (cm-1) full %10.1f  truncated %10.1f  ratio %.5f' % (k, a_ * HA2CM, b_ * HA2CM, b_ / a_))
    e = genbas.read('GENBAS.ln', '%s:DYVTZ-SOG-T07' % el.upper())
    print('   contraction error of %s (x2camf-g, Fock of the Ln3+ configuration average in the truncated primitives):' % e.name)
    for n, l, tj, x0, x1 in contract.contraction_error(e, el, 'x2camf-g', gau_nuc=True, result=atm):
        if n >= 4:
            print('     %d%s%d/2  %12.6f  diff %+.1e' % (n, 'spdf'[l], tj, x0, x1 - x0))
    sys.stdout.flush()
