'''socutils RSDF vs pyscf GDF with the same auxiliary basis.

    python rsdf_vs_gdf.py quartz|y2o3A [omega] [precision]

Errors are against the cached RSJK reference (gdf_aux.reference, refjk_*.npz).
'''
import sys
import time
import numpy as np
from pyscf.pbc import scf as pscf, df as pdf
from socutils.pbc.bench.cells import make_cell
from socutils.pbc.bench.auxgen_util import auxgen
from socutils.pbc.bench.gdf_aux import reference
from socutils.pbc.rsdf import RSDF


def errors(dm, vj, vk, refj, refk):
    return ('J: max %.1e dE %9.1e | K: max %.1e dE %9.1e'
            % (abs(vj - refj).max(), .5 * np.einsum('ij,ji', dm, vj - refj),
               abs(vk - refk).max(), -.25 * np.einsum('ij,ji', dm, vk - refk)))


def main(name, omega=0.5, precision=1e-10):
    cell = make_cell(name, 1e-10)
    dm = pscf.RHF(cell).get_init_guess(key='minao')
    refj, refk = reference(name, cell, dm)
    aux = auxgen(cell, contract=False, prune_lmax=False, threshold=1e-8)
    print('%s: nao %d, auxgen uncontracted tau=1e-8' % (name, cell.nao), flush=True)

    t = time.time()
    r = RSDF(cell, aux, omega=omega, precision=precision).build()
    tb = time.time() - t
    t = time.time()
    vj, vk = r.get_jk(dm)
    print('socutils RSDF (omega %.2f, prec %.0e): naux %d  build %6.1fs  JK %.2fs | %s | metric min eig %.1e'
          % (omega, precision, r.naux_kept, tb, time.time() - t, errors(dm, vj, vk, refj, refk), r.metric_eig[0]),
          flush=True)

    t = time.time()
    g = pdf.GDF(cell)
    g.auxbasis = aux
    g.build()
    tb = time.time() - t
    t = time.time()
    vj = g.get_jk(dm, with_k=False)[0]
    vk = g.get_jk(dm, with_j=False, exxdiv=None)[1]
    print('pyscf GDF (cell.precision 1e-10):       naux %d  build %6.1fs  JK %.2fs | %s'
          % (g.get_naoaux(), tb, time.time() - t, errors(dm, vj, vk, refj, refk)), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], *(float(x) for x in args[1:]))
