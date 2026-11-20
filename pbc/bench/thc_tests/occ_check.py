'''occ-RI spinor J/K (THCJK.get_jk_mo) against the density-matrix path (ghf spin blocks), and timings.'''
import sys, time, runpy, numpy as np
from pyscf import lib
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc import split_direct, spinor_scf
from socutils.pbc.thc import THCJK
from socutils.scf import spinor_hf
name = sys.argv[1]
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B, max_memory=150000)
nz = spinor_scf.nuclear_zeta(cell)
els = sorted(set(cell.atom_symbol(i) for i in range(cell.natm)))
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in els}
jk = THCJK(cell, aux, alpha_c=4., nmu_factor=16., nuc_zeta=nz).build()
mf = split_direct.periodic(spinor_hf.SpinorSCF(cell), jk)
n2c = cell.nao_2c()
rng = np.random.default_rng(0)
S = mf.get_ovlp()
e, v = np.linalg.eigh(S)
X = v / np.sqrt(e)
nocc = cell.nelectron
Q, _ = np.linalg.qr(rng.standard_normal((n2c, nocc)) + 1j * rng.standard_normal((n2c, nocc)))
mo = X @ Q                                                   # S-orthonormal spinor MOs
occ = np.linspace(1., .3, nocc)
dm = (mo * occ) @ mo.conj().T
t = time.time(); vj1, vk1 = mf.get_jk(cell, lib.tag_array(dm, mo_coeff=mo, mo_occ=occ)); t1 = time.time() - t
t = time.time(); vj0, vk0 = mf.get_jk(cell, dm); t0 = time.time() - t
print('occ-RI vs DM path: |dJ| %.1e |dK| %.1e (max |K| %.1e); time occ-RI %.0f s, DM path %.0f s'
      % (abs(vj1 - vj0).max(), abs(vk1 - vk0).max(), abs(vk0).max(), t1, t0), flush=True)
print('timings', jk.th.timings)
