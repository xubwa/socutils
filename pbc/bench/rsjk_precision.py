'''Accuracy/cost of pyscf RSJK J vs cell.precision and omega (Gamma point).

    python rsjk_precision.py quartz|y2o3A [prec1,prec2,...]

The reference is the tightest precision at the default omega; rerunning it at
two other omega values estimates the reference error (the result must not
depend on omega).
'''
import sys
import time
import numpy as np
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import rsjk
from socutils.pbc.bench.cells import make_cell


def run(name, precs=(1e-6, 1e-8, 1e-10, 1e-12), omegas=(0.5, 0.9)):
    cell = make_cell(name, 1e-8)
    dm = pscf.RHF(cell).get_init_guess(key='minao')
    print('%s: natm %d nao %d nelec %d' % (name, cell.natm, cell.nao, cell.nelectron), flush=True)
    res, tim = {}, {}

    def rs(prec, omega=None):
        c = make_cell(name, prec)
        t = time.time()
        jk = rsjk.RangeSeparatedJKBuilder(c)
        jk.build(omega=omega)
        tb = time.time() - t
        t = time.time()
        vj = jk.get_jk(dm[None], hermi=1, kpts=np.zeros((1, 3)), with_k=False)[0][0]
        return vj, jk.omega, tb, time.time() - t

    for p in precs:
        vj, om, tb, tj = rs(p)
        res['prec=%g' % p], tim['prec=%g' % p] = vj, tb + tj
        print('RSJK prec=%-6g omega=%.3f  E_J=%.10f  %.0fs' % (p, om, .5 * np.einsum('ij,ji', dm, vj), tb + tj), flush=True)
    pref = min(precs)
    for om in omegas:
        vj, _, tb, tj = rs(pref, om)
        res['prec=%g omega=%.1f' % (pref, om)], tim['prec=%g omega=%.1f' % (pref, om)] = vj, tb + tj
    ref = res['prec=%g' % pref]
    print('--- deviation from prec=%g (default omega)' % pref)
    for k, v in res.items():
        print('%-24s max|dJ| %.1e  dE_J %9.1e  %6.0fs' % (k, abs(v - ref).max(), .5 * np.einsum('ij,ji', dm, v - ref), tim[k]))


if __name__ == '__main__':
    precs = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else (1e-6, 1e-8, 1e-10, 1e-12)
    run(sys.argv[1], precs)
