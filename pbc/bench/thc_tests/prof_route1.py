import os, sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf, gto as pgto
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.thc import THCExchange
name = sys.argv[1]
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
ac = 4.
full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
els = sorted(set(full.atom_symbol(i) for i in range(full.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm0 = pscf.RHF(full).get_init_guess(key='minao')
T = np.linalg.lstsq(cell.pbc_intor('int1e_ovlp'), pgto.intor_cross('int1e_ovlp', cell, full), rcond=1e-12)[0]
dm = T @ dm0 @ T.T
t = time.time()
th = THCExchange(cell, aux, ac, selector='coarse', local_rcut=6.)
th.gapw_opts = {'nrad': int(os.environ.get('NRAD', 300))}
th.build_common()
th.build_thc(16 * len(th.smooth))
print('build %.0f s: %s' % (time.time() - t, ', '.join('%s %.0f' % kv for kv in th.timings.items())), flush=True)
t = time.time(); J = th.get_j(dm); tj1 = time.time() - t
t = time.time(); J2 = th.get_j(dm); tj2 = time.time() - t
print('GAPW J: first call %.0f s (setup included), second %.0f s' % (tj1, tj2), flush=True)
t = time.time(); Kd = th.get_k(dm); tkd = time.time() - t
t = time.time(); Ks = th.get_k_sparse(dm); tks = time.time() - t
print('K dense (prototype) %.0f s, K sparse DM %.0f s; |Ks - Kd| %.2e (max %.2e)' % (tkd, tks, abs(Ks - Kd).max(), abs(Kd).max()), flush=True)
rng = np.random.default_rng(0)
Z = 1e-2 * (rng.standard_normal(dm.shape) + 1j * rng.standard_normal(dm.shape))
print('complex non-Hermitian D: |Ks - Kd| %.2e' % abs(th.get_k_sparse(Z) - th.get_k(Z)).max(), flush=True)
print('timings', {k: round(v, 1) for k, v in th.timings.items()})
