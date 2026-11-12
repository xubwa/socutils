'''Wall time per spinor SCF cycle (each get_veff call) on a production cell: RSDF (cderi in memory) or THC route 1
(a built THCJK loaded from a pickle).  usage: scf_cycle_time.py rsdf|thc CIF NX,NY,NZ BASIS_PY [PICKLE]'''
import sys, time, runpy, pickle, numpy as np
from pyscf import gto
from socutils.pbc.bench import cells
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf, split_direct
from socutils.pbc.rsdf import RSDF
from socutils.scf import spinor_hf
mode, cif, sc, basis_py = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')], sys.argv[4]
B = dict(runpy.run_path(basis_py)['BASIS'])
sysm = cells.from_cif(cif)
if sc != [1, 1, 1]:
    sysm = cells.supercell(sysm, sc)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=380000)
t = time.time()
if mode == 'rsdf':
    els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
    aux = {e: auxgen_from_basis({gto.mole._std_symbol(e): B[e]}, contract=False, prune_lmax=False,
                                threshold=1e-8)[gto.mole._std_symbol(e)] for e in els}
    r = RSDF(cell, aux, omega=0.5, precision=1e-10, nuc_zeta=spinor_scf.nuclear_zeta(cell), lr='gspace')
    r.cderi_in_memory = True
    r.build()
    print('RSDF build %.0f s, cderi %.1f GB' % (time.time() - t, r.cderi.nbytes / 1e9), flush=True)
    mf = spinor_scf.SpinorRSDF(cell, r)
else:
    with open(sys.argv[5], 'rb') as f:
        jk = pickle.load(f)
    print('THC loaded %.0f s' % (time.time() - t), flush=True)
    mf = split_direct.periodic(spinor_hf.SpinorSCF(cell), jk)
times = []
gv = mf.get_veff
def timed(*a, **k):
    t0 = time.time(); v = gv(*a, **k); times.append(time.time() - t0)
    print('  get_veff %d: %.0f s' % (len(times), times[-1]), flush=True)
    return v
mf.get_veff = timed
mf.max_cycle = 3
mf.verbose = 4
t = time.time(); mf.kernel(); print('%s: %d get_veff calls %s s; SCF wall %.0f s' % (mode, len(times), [round(x) for x in times], time.time() - t))
