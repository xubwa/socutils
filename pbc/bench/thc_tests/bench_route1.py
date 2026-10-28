'''Route-1 THCJK on a production cell: build time, peak memory, storage, one J+K (minao density), vs nothing else.

usage: bench_route1.py CIF NX,NY,NZ BASIS_PY [ACTIVE_LABEL]   (env RCUT, NRAD, NMU, STREAM_GB, PIV_FILE)'''
import os, sys, time, resource, runpy, numpy as np
from pyscf import gto
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf
from socutils.pbc.thc import THCJK
cif, sc, basis_py = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
lab = sys.argv[4] if len(sys.argv) > 4 else None
sysm = cells.from_cif(cif)
if sc != [1, 1, 1]:
    sysm = cells.supercell(sysm, sc)
if lab:
    k, at = 0, []
    for e, x in sysm[1]:
        if e == 'Eu':
            e = lab if k == 0 else e
            k += 1
        at.append((e, x))
    sysm = (sysm[0], at)
B = dict(runpy.run_path(basis_py)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=380000)
aux = {}
for l_ in sorted(set(cell.atom_symbol(i) for i in range(cell.natm))):
    el = gto.mole._std_symbol(l_)
    aux[l_] = auxgen_from_basis({el: B[l_]}, contract=False, prune_lmax=False, threshold=1e-8)[el]
mem = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print('%s %s: %d atoms, nao %d' % (os.path.basename(cif), sc, cell.natm, cell.nao), flush=True)
t = time.time()
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=float(os.environ.get('NMU', 16)), nuc_zeta=spinor_scf.nuclear_zeta(cell),
           local_rcut=float(os.environ.get('RCUT', 6)))
jk.gapw_opts = {'nrad': int(os.environ.get('NRAD', 600))}
jk.stream = True                      # V, A, Xi in mu blocks, no full xi
jk.stream_gb = float(os.environ.get('STREAM_GB', 30))
jk.piv_file = os.environ.get('PIV_FILE')
jk.build()
print('build %.0f s, peak memory %.1f GB; %s' % (time.time() - t, mem(), ', '.join('%s %.0f' % kv for kv in jk.th.timings.items())),
      flush=True)
print('storage (doubles): %s' % jk.th.storage(), flush=True)
dm = pscf.RHF(cell).get_init_guess(key='minao')
t = time.time(); vj, vk = jk.get_jk(dm); tjk = time.time() - t
print('one J+K (real dm): %.0f s, peak memory %.1f GB; timings %s' % (tjk, mem(), {k: round(v) for k, v in jk.th.timings.items()}),
      flush=True)
