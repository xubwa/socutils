'''Eu-doped cell: periodic RSDF -> KRHF + one-centre X2CAMF with the Eu 4f shell
fractionally occupied (chosen by character) -> CASCI(6e, 14) on the 4f spinors.

    python eu_casci.py eu_y2o3A|eu_yso [omega] [k_rebuild_cycle]

eu_y2o3A (5 atoms, 4 cores): RSDF 272 s, SCF 160 s, CASCI 24 s; 7F0 ground,
7F1 split 1+2 and 7F2 split 2+1+2 (C3v site).  Workflow example, not physics
(approximate structure, x2c-SVPall trimmed, no dynamic correlation).
'''
import numpy as np, time, sys
from pyscf import lib
from socutils.pbc.bench.cells import make_cell
from socutils.pbc.bench.auxgen_util import auxgen
from socutils.pbc.rsdf import RSDF
from socutils.pbc import spinor_scf
from socutils.scf import spinor_hf, addons
from socutils.mcscf import zcasci
name = sys.argv[1]
omega = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
kcyc = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cell = make_cell(name, 1e-10, nucmod='G', max_memory=10000)
ieu = [i for i in range(cell.natm) if cell.atom_symbol(i) == 'Eu'][0]
print('%s: %d atoms, nao %d, n2c %d, nelectron %d' % (name, cell.natm, cell.nao, cell.nao_2c(), cell.nelectron), flush=True)
aux = auxgen(cell, contract=False, prune_lmax=False, threshold=1e-8)
t = time.time()
r = RSDF(cell, aux, omega=omega, nuc_zeta=spinor_scf.nuclear_zeta(cell)); r.verbose = 4; r.stdout = sys.stdout; r.build()
print('RSDF build %.0f s, naux kept %d, min eig %.1e' % (time.time() - t, r.naux_kept, r.metric_eig[0]), flush=True)
mf = spinor_scf.periodic(spinor_hf.KRHF(cell), r).x2camf()
mf.k_rebuild_cycle = kcyc
mf.max_cycle = 80; mf.conv_tol = 1e-8; mf.verbose = 4; mf.stdout = sys.stdout
idx = addons.spinor_ao_indices(cell, ieu, l=3)
addons.projected_frac_occ(mf, idx, 14, 6)
t = time.time(); e = mf.kernel(); 
print('SCF %s E = %.10f  %.0f s, %d cycles, %d exact K builds' % (mf.converged, e, time.time() - t, mf.cycles, mf.k_builds), flush=True)
w = addons.subspace_weights(mf.get_ovlp(), mf.mo_coeff, idx)
o = mf.open_shell_mo
print('open-shell 4f weights', np.round(w[o], 3), '\nenergies', np.round(mf.mo_energy[o], 4), flush=True)
mo, ncore = addons.cas_order(mf)
mc = zcasci.CASCI(mf, 14, 6, ncore=ncore); mc.verbose = 4; mc.stdout = sys.stdout
mc.fcisolver.nroots = 13
t = time.time(); mc.kernel(mo)
e = np.atleast_1d(mc.e_tot)
print('CASCI(6e,14) %.0f s; E0 = %.10f; excitations (cm-1):' % (time.time() - t, e.min()), np.round((np.sort(e) - e.min()) * 219474.63, 1), flush=True)
