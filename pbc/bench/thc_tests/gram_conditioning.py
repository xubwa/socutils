'''Gram spectrum of the fine-grid interpolation points: blocked vs orbit selection (quartz, ke precision 1e-12)'''
import runpy, numpy as np, scipy.linalg
from socutils.pbc.bench import cells
from socutils.pbc.bench.cells import make_cell
from socutils.pbc.bench.sr_locality import RULES, split_basis
from socutils.pbc.thc import space_group_ops, symmetric_mesh, grid_orbits, pivots_orbits, pivots_blocked, _subcell
from pyscf.pbc.gto.cell import estimate_ke_cutoff
B = dict(runpy.run_path(RULES)['BASIS'])
cell = make_cell(cells.SYSTEMS['quartz'], 1e-10, nucmod='G', basis=split_basis(B, 4.))
aohard = np.repeat([cell.bas_exp(i).min() >= 4. for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
sm = np.where(~aohard)[0]; ns = len(sm)
m0 = cell.cutoff_to_mesh(estimate_ke_cutoff(_subcell(cell, sm), 1e-12))
ops = space_group_ops(cell)
a = cell.lattice_vectors()
for label, m, orbits, dep in (('blocked', m0, False, 0), ('orbits', symmetric_mesh(m0, ops[:2], 3), True, 1e-2), ('orbits', symmetric_mesh(m0, ops[:2], 3), True, 1e-1)):
    m = np.asarray(m); mc = m // 3
    cc = cell.get_uniform_grids(mc)
    phic = cell.pbc_eval_gto('GTOval_sph', cc)[:, sm]
    if orbits:
        orb, mem = grid_orbits(mc, ops[:2]); pv, L = pivots_orbits(phic, 16 * ns, orb, mem, orbit_dep=dep)
    else:
        pv, L = pivots_blocked(phic, 16 * ns)
    PP = phic[pv]
    e = scipy.linalg.eigvalsh((PP @ PP.T) ** 2)
    e = e[::-1] / e.max()
    # nearest-neighbour distance among the points (minimum image)
    x = cc[pv]; f = x @ np.linalg.inv(a)
    dmin = []
    for i in range(0, len(f), 500):
        df = f[i:i + 500, None] - f[None]; df -= np.round(df)
        d = np.linalg.norm(df @ a, axis=2); d[d < 1e-9] = np.inf
        dmin.append(d.min(axis=1))
    dmin = np.concatenate(dmin)
    h = np.linalg.norm(a, axis=1) / mc
    print('dep %.0e ' % dep + '%-8s mesh %s, %d pts: eig rel < 1e-10: %d, < 1e-12: %d, smallest %.1e; nearest-neighbour distance min %.2f bohr, '
          '#pairs closer than 1.5 coarse spacings %d (coarse spacing %.2f)' % (label, list(m), len(pv), (e < 1e-10).sum(),
          (e < 1e-12).sum(), e[-1], dmin.min(), (dmin < 1.5 * h.min()).sum(), h.min()), flush=True)
