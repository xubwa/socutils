'''Route-1 batched K (get_k_multi) vs the single-density reference (get_k_sparse_v1): spinor-like spin blocks of random
complex occupied coefficients (aa, bb Hermitian, ab general; ba = ab^H); timings of one batched call vs one real K.

usage: multi_check.py [CIF BASIS_PY [NX,NY,NZ]]   (default quartz; env STREAM_GB)'''
import os, sys, time, runpy, numpy as np
from pyscf import gto
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
import pickle
t = time.time()
if os.environ.get('LOAD') and os.path.exists(os.environ['LOAD']):     # a built THCJK (pickle), to skip the build
    with open(os.environ['LOAD'], 'rb') as f:
        jk = pickle.load(f)
    print('loaded %s in %.0f s' % (os.environ['LOAD'], time.time() - t), flush=True)
else:
    jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=spinor_scf.nuclear_zeta(cell), local_rcut=6.)
    jk.stream = True
    jk.stream_gb = float(os.environ.get('STREAM_GB', 30))
    jk.build()
    if os.environ.get('SAVE'):
        ts = time.time()
        with open(os.environ['SAVE'], 'wb') as f:
            pickle.dump(jk, f, protocol=5)
        print('saved %s (%.1f GB) in %.0f s' % (os.environ['SAVE'], os.path.getsize(os.environ['SAVE']) / 1e9,
                                               time.time() - ts), flush=True)
th = jk.th
print('%d atoms, nao %d (split %d), N_mu %d, naux %d, nCX %d, nLR %d; build %.0f s (%s)'
      % (cell.natm, cell.nao, th.nao, len(th.pts), th.auxmol.nao, len(th.lr_cols), th.Llr_cx.shape[0],
         time.time() - t, {k: round(v) for k, v in th.timings.items()}), flush=True)
rng = np.random.default_rng(7)
nocc = cell.nelectron // 2
ca = 0.1 * (rng.standard_normal((th.nao, nocc)) + 1j * rng.standard_normal((th.nao, nocc))) / np.sqrt(th.nao)
cb = 0.1 * (rng.standard_normal((th.nao, nocc)) + 1j * rng.standard_normal((th.nao, nocc))) / np.sqrt(th.nao)
Daa, Dbb, Dab = ca @ ca.conj().T, cb @ cb.conj().T, ca @ cb.conj().T
dm0 = __import__('pyscf.pbc.scf', fromlist=['x']).RHF(cell).get_init_guess(key='minao')
for i in range(2):
    t = time.time(); jk.get_jk(dm0, with_k=False); print('J (call %d) %.0f s' % (i + 1, time.time() - t), flush=True)
t = time.time(); K1 = th.get_k_multi([Daa.real]); t1 = time.time() - t
print('one real density: %.0f s %s' % (t1, {k: round(v, 1) for k, v in th.timings.items() if k.startswith('K')}), flush=True)
t = time.time()
Kr = th.get_k_multi([Daa.real, Daa.imag, Dbb.real, Dbb.imag, Dab.real, Dab.imag], sym=[1, -1, 1, -1, 0, 0])
t6 = time.time() - t
print('spin blocks batched (6 real densities): %.0f s %s' % (t6, {k: round(v, 1) for k, v in th.timings.items() if k.startswith('K')}),
      flush=True)
Kaa, Kbb, Kab = Kr[0] + 1j * Kr[1], Kr[2] + 1j * Kr[3], Kr[4] + 1j * Kr[5]
if not os.environ.get('SKIP_REF'):
    t = time.time()
    ref = {k: th.get_k_sparse_v1(D) for k, D in (('aa', Daa), ('bb', Dbb), ('ab', Dab), ('ba', Dab.conj().T))}
    print('reference (v1, 4 complex blocks): %.0f s' % (time.time() - t), flush=True)
    for k, Kn in (('aa', Kaa), ('bb', Kbb), ('ab', Kab), ('ba', Kab.conj().T)):
        print('  %s: max |K_multi - K_v1| %.2e (|K| %.2e)' % (k, abs(Kn - ref[k]).max(), abs(ref[k]).max()), flush=True)
    print('  real single: %.2e' % abs(K1[0] - th.get_k_sparse_v1(Daa.real)).max(), flush=True)
