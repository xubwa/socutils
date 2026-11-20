'''GDF J and K accuracy/cost vs auxiliary basis, against RSJK (prec 1e-10).

    python gdf_aux.py quartz|y2o3A [etb|auxgen|j2c]

etb    : ETB beta = 2.0, 1.7, 1.5, 1.3 and AutoAux
auxgen : Lehtola pivoted-Cholesky sets (needs BSE master, see auxgen_util)
j2c    : dense sets with the metric forced through eigendecomposition

The RSJK reference J and K are cached in refjk_<system>.npz in the working
directory (the Y2O3 reference takes ~10 min on 4 cores).
'''
import os
import sys
import time
import numpy as np
from pyscf.df import addons as dfaddons
from pyscf.df.autoaux import autoaux as make_autoaux
from pyscf.pbc import scf as pscf, df as pdf
from pyscf.pbc.scf import rsjk
from pyscf.pbc.df import rsdf_builder
from socutils.pbc.bench.cells import make_cell


def reference(name, cell, dm):
    fn = 'refjk_%s.npz' % name
    if os.path.exists(fn):
        z = np.load(fn)
        return z['vj'], z['vk']
    t = time.time()
    jk = rsjk.RangeSeparatedJKBuilder(cell)
    jk.build()
    vj, vk = jk.get_jk(dm[None], hermi=1, kpts=np.zeros((1, 3)), with_k=True, exxdiv=None)
    np.savez(fn, vj=vj[0], vk=vk[0])
    print('reference RSJK J+K prec=1e-10: %.0fs' % (time.time() - t), flush=True)
    return vj[0], vk[0]


def gdf_errors(cell, dm, refj, refk, aux, linear_dep_threshold=None):
    g = pdf.GDF(cell)
    g.auxbasis = aux
    if linear_dep_threshold is not None:
        g.linear_dep_threshold = linear_dep_threshold
    t = time.time()
    g.build()
    tb = time.time() - t
    vj = g.get_jk(dm, with_k=False)[0]
    t = time.time()
    vk = g.get_jk(dm, with_j=False, exxdiv=None)[1]
    tk = time.time() - t
    nprim = max(len(sh) - 1 for el in aux for sh in aux[el]) if isinstance(aux, dict) else 0
    return ('naux %5d maxprim %2d | J: max %.1e dE %9.1e | K: max %.1e dE %9.1e | build %5.0fs K %.1fs'
            % (g.get_naoaux(), nprim, abs(vj - refj).max(), .5 * np.einsum('ij,ji', dm, vj - refj),
               abs(vk - refk).max(), -.25 * np.einsum('ij,ji', dm, vk - refk), tb, tk))


def main(name, which):
    cell = make_cell(name, 1e-10)
    dm = pscf.RHF(cell).get_init_guess(key='minao')
    refj, refk = reference(name, cell, dm)
    print('%s: nao %d nelec %d' % (name, cell.nao, cell.nelectron), flush=True)
    if which == 'etb':
        cases = [('ETB beta=%.1f' % b, None, (lambda b=b: dfaddons.aug_etb(cell, beta=b))) for b in (2.0, 1.7, 1.5, 1.3)]
        cases.append(('AutoAux', None, lambda: make_autoaux(cell)))
    elif which == 'auxgen':
        from socutils.pbc.bench.auxgen_util import auxgen
        cases = [('auxgen %s (SVD-contracted)' % s, None, (lambda s=s: auxgen(cell, size=s))) for s in ('large', 'verylarge')]
        cases += [('auxgen uncontracted tau=%g' % t, None, (lambda t=t: auxgen(cell, contract=False, prune_lmax=False, threshold=t)))
                  for t in (1e-7, 1e-8, 1e-9)]
    elif which == 'j2c':
        from socutils.pbc.bench.auxgen_util import auxgen
        cases = []
        for tag, mk in [('auxgen uncontracted tau=1e-8', lambda: auxgen(cell, contract=False, prune_lmax=False, threshold=1e-8)),
                        ('ETB beta=1.3', lambda: dfaddons.aug_etb(cell, beta=1.3))]:
            cases += [(tag + ' cholesky', ('chol', None), mk)]
            cases += [(tag + ' eigh thr=%g' % thr, ('eig', thr), mk) for thr in (1e-10, 1e-8, 1e-6)]
    else:
        raise ValueError(which)
    for tag, metric, mkaux in cases:
        try:
            if metric is not None:
                rsdf_builder._RSGDFBuilder.j2c_eig_always = metric[0] == 'eig'
            line = gdf_errors(cell, dm, refj, refk, mkaux(), None if metric is None else metric[1])
            print('%-36s %s' % (tag, line), flush=True)
        except Exception as e:
            print('%-36s failed: %s' % (tag, str(e)[:160]), flush=True)
        finally:
            rsdf_builder._RSGDFBuilder.j2c_eig_always = False


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'etb')
