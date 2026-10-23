'''Checks of socutils.pbc.split_direct.SplitDirectJK on a small cell.

  1. SR part: K(SplitDirectJK) - K(LR rows) against an exact SR K in the cell's basis (hybrid_k cache, minao dm);
  2. J/K against RSDF lr='gspace' (global SR fit): the DF error;
  3. timings of one J+K for a real dm and for the four complex spin blocks of a spinor density;
  4. (SCF=1) spinor HF energy with SplitDirectJK vs RSDF.

usage: split_direct_check.py SYSTEM CUTS [EXACT_NPY]     CUTS e.g. 8 or 0.5,2,8,32'''
import os, sys, time, runpy, numpy as np
from pyscf import lib
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.rsdf import RSDF
from socutils.pbc import split_direct, spinor_scf

name, cuts = sys.argv[1], [float(x) for x in sys.argv[2].split(',')]
cuts = cuts if len(cuts) > 1 else cuts[0]
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell.verbose = 4
nz = spinor_scf.nuclear_zeta(cell)
dm = pscf.RHF(cell).get_init_guess(key='minao')
print('%s: nao %d, %d threads' % (name, cell.nao, lib.num_threads()), flush=True)

t = time.time()
jk = split_direct.SplitDirectJK(cell, cuts=cuts, nuc_zeta=nz).build()
print('SplitDirectJK build %.0f s: split nao %d, supmol %d shells' % (time.time() - t, jk.split_cell.nao, jk.rsjk.supmol_sr.nbas),
      flush=True)
t = time.time()
vj, vk = jk.get_jk(dm, hermi=1)
print('J+K (real dm): %.1f s' % (time.time() - t), flush=True)
vk_lr = jk.rsdf.get_jk(dm, hermi=1, with_j=False)[1]
if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
    K0 = np.load(sys.argv[3])
    d = vk - vk_lr - K0
    print('1. SR K vs exact (%s): max %.2e, d(E_x) %.2e Eh' % (sys.argv[3], abs(d).max(), -.25 * np.einsum('ij,ji', dm, d)),
          flush=True)

els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
t = time.time()
r = RSDF(cell, aux, omega=0.5, precision=1e-10, nuc_zeta=nz, lr='gspace')
r.cderi_in_memory = True
r.build()
print('RSDF build %.0f s' % (time.time() - t), flush=True)
vj1, vk1 = r.get_jk(dm, hermi=1)
print('2. vs RSDF: J max %.2e, E_J %.2e Eh; K max %.2e, E_x %.2e Eh; same nuclei: %.1e, c_g0 %.1e'
      % (abs(vj - vj1).max(), .5 * np.einsum('ij,ji', dm, vj - vj1), abs(vk - vk1).max(),
         -.25 * np.einsum('ij,ji', dm, vk - vk1), abs(jk.get_nuc() - r.get_nuc()).max(), jk.c_g0 - r.c_g0), flush=True)

# four complex spin blocks of a spinor-like density
rng = np.random.default_rng(1)
n = cell.nao
X = 1e-2 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
dms = np.array([dm / 2 + 0j, dm / 2 + 0j, X, X.conj().T])
t = time.time()
jk.get_jk(dms, hermi=0)
print('3. J+K, 4 complex spin blocks: %.1f s' % (time.time() - t), flush=True)

if os.environ.get('SCF'):
    from socutils.scf import spinor_hf
    res = {}
    for lab, obj in (('RSDF', r), ('SplitDirect', jk)):
        mf = (spinor_scf.SpinorRSDF(cell, obj) if lab == 'RSDF' else split_direct.SpinorSplitDirect(cell, obj))
        mf.conv_tol = 1e-8
        mf.max_cycle = 60
        t = time.time()
        res[lab] = mf.kernel()
        print('4. spinor HF with %s: E = %.10f (%d cycles? converged %s, %.0f s)'
              % (lab, res[lab], getattr(mf, 'cycles', -1), mf.converged, time.time() - t), flush=True)
    print('   E(SplitDirect) - E(RSDF) = %.2e Eh' % (res['SplitDirect'] - res['RSDF']), flush=True)
