'''Route-1 K: LR of all blocks through the full LR rows (k_lr 'dense', default) vs the old CX-SS cross term
(k_lr 'cross'); timings of the K pieces and of J.

usage: k_lr_check.py [CIF BASIS_PY [NX,NY,NZ]]   (default quartz)'''
import os, sys, time, runpy, numpy as np
from pyscf import gto
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import spinor_scf
from socutils.pbc.thc import THCJK
if len(sys.argv) > 2:
    B = dict(runpy.run_path(sys.argv[2])['BASIS'])
    sysm = cells.from_cif(sys.argv[1])
    if len(sys.argv) > 3:
        sysm = cells.supercell(sysm, [int(x) for x in sys.argv[3].split(',')])
else:
    B, sysm = dict(runpy.run_path(RULES)['BASIS']), cells.SYSTEMS['quartz']
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({gto.mole._std_symbol(e): B[e]}, contract=False, prune_lmax=False,
                            threshold=1e-8)[gto.mole._std_symbol(e)] for e in els}
t = time.time()
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
jk.stream = True
jk.stream_gb = float(os.environ.get('STREAM_GB', 30))
jk.build()
th = jk.th
print('%d atoms, nao %d, N_mu %d, naux %d, nCX %d, nLR %d; build %.0f s (%s)'
      % (cell.natm, cell.nao, len(th.pts), th.auxmol.nao, len(th.lr_cols), th.Llr_cx.shape[0], time.time() - t,
         {k: round(v) for k, v in th.timings.items()}), flush=True)
dm = pscf.RHF(cell).get_init_guess(key='minao')
Ccx = th.Cs[:, th.lr_cols].tocsc()
t = time.time(); ref = np.asarray(Ccx.T @ th.A); ts = time.time() - t
t = time.time(); blocks = th._cx_atom_blocks(Ccx); tb0 = time.time() - t
t = time.time()
Bp = np.zeros_like(ref)
for rows, cols, Cd in blocks:
    Bp[cols] += Cd.T @ th.A[rows]
print('C^T A: sparse %.1f s, atom blocks %.1f s (+%.1f s setup), max diff %.2e' % (ts, time.time() - t, tb0, abs(Bp - ref).max()),
      flush=True)
ref = Bp = None
t = time.time(); vj = jk.get_jk(dm, with_k=False)[0]; print('J %.0f s' % (time.time() - t), flush=True)
import os
Ks = {}
t = time.time(); vj = jk.get_jk(dm, with_k=False)[0]; print('J (second call) %.0f s' % (time.time() - t), flush=True)
for mode in (('dense',) if os.environ.get('SKIP_OLD') else ('dense', 'cross')):
    th.k_lr = mode
    t = time.time(); Ks[mode] = jk.get_jk(dm, with_j=False)[1]
    print('K (%s) %.0f s: %s' % (mode, time.time() - t, {k: round(v, 1) for k, v in th.timings.items() if k.startswith('K')}),
          flush=True)
if os.environ.get('SKIP_OLD'):
    t = time.time(); K2 = jk.get_jk(dm, with_j=False)[1]
    print('K again %.0f s: %s, |K - K2| %.1e' % (time.time() - t, {k: round(v, 1) for k, v in th.timings.items() if k.startswith('K')},
          abs(K2 - Ks['dense']).max()), flush=True)
    X = 1e-2 * (np.random.default_rng(1).standard_normal(dm.shape) + 1j * np.random.default_rng(2).standard_normal(dm.shape))
    a = jk.get_jk(X, with_j=False)[1]
    b = jk.get_jk(X.real, with_j=False)[1] + 1j * jk.get_jk(X.imag, with_j=False)[1]
    print('complex dm: |K(X) - K(Re X) - i K(Im X)| %.1e' % abs(a - b).max(), flush=True)
    sys.exit()
th.k_lr, th.k_cc = 'dense', 'loop'
t = time.time(); Ks['loop'] = jk.get_jk(dm, with_j=False)[1]
print('K (CX-CX per-aux loop) %.0f s: %s' % (time.time() - t, {k: round(v, 1) for k, v in th.timings.items() if k.startswith('K')}),
      flush=True)
th.k_cc = 'blocks'
d = Ks['dense'] - Ks['loop']
print('CX-CX blocks vs loop: K max %.2e, E_x %.2e' % (abs(d).max(), -.25 * np.einsum('ij,ji', dm, d)), flush=True)
d = Ks['dense'] - Ks['cross']
print('dense vs cross: K max %.2e, E_x %.2e' % (abs(d).max(), -.25 * np.einsum('ij,ji', dm, d)), flush=True)
X = 1e-2 * (np.random.default_rng(1).standard_normal(dm.shape) + 1j * np.random.default_rng(2).standard_normal(dm.shape))
th.k_lr = 'dense'; a = jk.get_jk(X, with_j=False)[1]
th.k_lr, th.k_cc = 'cross', 'loop'; b = jk.get_jk(X, with_j=False)[1]
print('complex dm: dense vs cross max %.2e' % abs(a - b).max(), flush=True)
