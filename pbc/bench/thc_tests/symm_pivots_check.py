import runpy, time, numpy as np
from socutils.pbc.bench.cells import make_cell, from_cif
from socutils.pbc.bench.sr_locality import split_basis
from socutils.pbc.thc import space_group_ops, symmetric_mesh, grid_orbits, pivots_orbits, pivots_blocked, _subcell
from pyscf.pbc.gto.cell import estimate_ke_cutoff
P = '/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/'
B = runpy.run_path(P + '../basis_rules.py')['BASIS']
cell = make_cell(from_cif('/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/struct/oxyhalides/COD_1542170.cif'), 1e-10,
                 nucmod='G', basis=split_basis(B, 4.))
ops = space_group_ops(cell)
aohard = np.repeat([cell.bas_exp(i).min() >= 4. for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
sm = np.where(~aohard)[0]
m0 = cell.cutoff_to_mesh(estimate_ke_cutoff(_subcell(cell, sm), 1e-10))
m = symmetric_mesh(m0, ops[:2], 3); mc = m // 3
print(ops[2], len(ops[0]), 'mesh', m0, '->', m, 'coarse', mc, flush=True)
cc = cell.get_uniform_grids(mc)
phic = cell.pbc_eval_gto('GTOval_sph', cc)[:, sm]
t = time.time(); orb, mem = grid_orbits(mc, ops[:2]); print('orbits %d of %d points (%.1f s)' % (len(mem), len(cc), time.time() - t))
t = time.time(); pv, L = pivots_orbits(phic, 16 * len(sm), orb, mem); t1 = time.time() - t
t = time.time(); pb, Lb = pivots_blocked(phic, 16 * len(sm)); t2 = time.time() - t
S = set(pv.tolist())
closed = all(set(mem[o].tolist()) <= S for o in set(orb[pv].tolist()))
print('orbit pivots %d (%.0f s), closed under the group: %s; plain blocked %d (%.0f s)' % (len(pv), t1, closed, len(pb), t2))
print('residual trace after: orbit %.3e  blocked %.3e' % ((np.einsum('ri,ri->r', phic, phic) ** 2).sum() - (L ** 2).sum(),
      (np.einsum('ri,ri->r', phic, phic) ** 2).sum() - (Lb ** 2).sum()))
