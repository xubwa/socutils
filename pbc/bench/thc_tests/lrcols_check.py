'''RSDF(sr_fit=False, lr_columns=CX columns, one s aux per atom) against the full RSDF: LR rows of the CX columns,
nuclear attraction, E_nuc, c0; timings.'''
import runpy, time, numpy as np
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis
from socutils.pbc.rsdf import RSDF
from socutils.pbc import spinor_scf
B = dict(runpy.run_path(RULES)['BASIS'])
ac = 4.
cell = cells.make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
nz = spinor_scf.nuclear_zeta(cell)
aux = {e: auxgen_from_basis({e: B[e]}, contract=False, prune_lmax=False, threshold=1e-8)[e] for e in ('O', 'Si')}
t = time.time()
r = RSDF(cell, aux, omega=0.5, precision=1e-10, nuc_zeta=nz, lr='gspace'); r.cderi_in_memory = True; r.verbose = 0; r.build()
t_full = time.time() - t
nao = cell.nao
compact = np.repeat([cell.bas_exp(i).min() >= ac for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
ti, tj = np.tril_indices(nao)
cols = np.where(compact[ti] | compact[tj])[0]
t = time.time()
r2 = RSDF(cell, {l: [[0, [1., 1.]]] for l in cell._basis}, omega=0.5, precision=1e-10, nuc_zeta=nz, lr='gspace')
r2.sr_fit = False; r2.lr_columns = cols; r2.verbose = 0; r2.build()
t_lr = time.time() - t
nk = r.cderi.shape[0] - r.nlr
print('LR rows (CX cols) max diff %.2e (max %.2e); vnuc %.2e; E_nuc %.2e; c0 %.1e; time full RSDF %.0f s, nuclei + CX LR rows %.0f s; rows %s'
      % (abs(r2.cderi - r.cderi[nk:][:, cols]).max(), abs(r2.cderi).max(), abs(r2.get_nuc() - r.get_nuc()).max(),
         r2.energy_nuc() - r.energy_nuc(), r2.c_g0 - r.c_g0, t_full, t_lr, r2.cderi.shape))
