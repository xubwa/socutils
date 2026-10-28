'''THCJK (pbc/thc.py) in the periodic spinor SCF: J/K against RSDF on a minao density (real, and four complex spin
blocks against SplitDirectJK's exact-SR reference), then spinor HF energies with THCJK and RSDF.

usage: thc_scf_check.py SYSTEM ALPHA_C NMU_FACTOR [SELECTOR]     (env NO_SCF=1 skips the SCF; REF_E: a reference E)'''
import os, sys, time, runpy, numpy as np
from pyscf.pbc import scf as pscf
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.rsdf import RSDF
from socutils.pbc import split_direct, spinor_scf
from socutils.pbc.thc import THCJK

name, ac, fac = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
sel = sys.argv[4] if len(sys.argv) > 4 else 'coarse'
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
nz = spinor_scf.nuclear_zeta(cell)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
dm = pscf.RHF(cell).get_init_guess(key='minao')

t = time.time()
jk = THCJK(cell, aux, alpha_c=ac, nmu_factor=fac, selector=sel, nuc_zeta=nz,
           local_rcut=float(os.environ['LOCAL']) if os.environ.get('LOCAL') else None)
if os.environ.get('NRAD'):
    jk.gapw_opts = {'nrad': int(os.environ['NRAD'])}
jk.build()
print('THCJK storage (doubles): %s' % jk.th.storage()) if hasattr(jk.th, 'storage') else None
print('THCJK build %.0f s (N_mu %d, smooth %d; timings %s)' % (time.time() - t, len(jk.th.pts), len(jk.th.smooth),
      ', '.join('%s %.0f' % kv for kv in jk.th.timings.items())), flush=True)
r = RSDF(cell, aux, omega=0.5, precision=1e-10, nuc_zeta=nz, lr='gspace')
r.cderi_in_memory = True
r.build()
t = time.time()
vj, vk = jk.get_jk(dm)
print('THCJK J+K (real dm) %.1f s' % (time.time() - t), flush=True)
vj1, vk1 = r.get_jk(dm, hermi=1)
print('vs RSDF: J max %.2e E_J %.2e; K max %.2e E_x %.2e Eh; nuclei %.1e; E_nuc %.1e'
      % (abs(vj - vj1).max(), .5 * np.einsum('ij,ji', dm, vj - vj1), abs(vk - vk1).max(),
         -.25 * np.einsum('ij,ji', dm, vk - vk1), abs(jk.get_nuc() - r.get_nuc()).max(), jk.energy_nuc() - r.energy_nuc()),
      flush=True)
if os.environ.get('EXACT_J'):                                       # exact SR + LR rows (SplitDirectJK)
    sd = split_direct.SplitDirectJK(cell, cuts=8., nuc_zeta=nz).build()
    vj0 = sd.get_jk(dm, hermi=1, with_k=False)[0]
    print('J vs exact: THCJK E_J %.2e (max %.2e), RSDF E_J %.2e Eh'
          % (.5 * np.einsum('ij,ji', dm, vj - vj0), abs(vj - vj0).max(), .5 * np.einsum('ij,ji', dm, vj1 - vj0)), flush=True)
# complex spin blocks: linearity check (K of a complex D = K(Re D) + i K(Im D))
rng = np.random.default_rng(1)
X = 1e-2 * (rng.standard_normal(dm.shape) + 1j * rng.standard_normal(dm.shape))
vjc, vkc = jk.get_jk(np.array([X]))
vjr, vkr = jk.get_jk(np.array([X.real]))
vji, vki = jk.get_jk(np.array([X.imag]))
print('complex dm: |K - (K_re + i K_im)| %.1e, |J - (J_re + i J_im)| %.1e'
      % (abs(vkc - vkr - 1j * vki).max(), abs(vjc - vjr - 1j * vji).max()), flush=True)

if not os.environ.get('NO_SCF'):
    res = {}
    for lab, obj in (('RSDF', r), ('THC', jk)):
        mf = spinor_scf.SpinorRSDF(cell, obj) if lab == 'RSDF' else split_direct.periodic(
            __import__('socutils.scf.spinor_hf', fromlist=['x']).SpinorSCF(cell), obj)
        mf.conv_tol = 1e-8
        t = time.time()
        res[lab] = mf.kernel()
        print('spinor HF with %s: E = %.10f, converged %s, %.0f s' % (lab, res[lab], mf.converged, time.time() - t),
              flush=True)
    print('E(THC) - E(RSDF) = %.2e Eh' % (res['THC'] - res['RSDF']), flush=True)
    if os.environ.get('REF_E'):
        print('E(THC) - E(ref %s) = %.2e Eh, E(RSDF) - E(ref) = %.2e Eh'
              % (os.environ['REF_E'], res['THC'] - float(os.environ['REF_E']), res['RSDF'] - float(os.environ['REF_E'])))
