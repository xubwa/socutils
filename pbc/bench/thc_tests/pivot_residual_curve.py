'''relative residual trace of the pivoted Cholesky (coarse grid) vs points per smooth function: blocked vs orbits'''
import runpy, numpy as np
from socutils.pbc.bench import cells
from socutils.pbc.bench.cells import make_cell, from_cif
from socutils.pbc.bench.sr_locality import RULES, split_basis
from socutils.pbc.thc import space_group_ops, symmetric_mesh, grid_orbits, pivots_orbits, pivots_blocked, _subcell
from pyscf.pbc.gto.cell import estimate_ke_cutoff
P = '/scratch4/lcheng24/xwang405/zmcscf/Eu/periodic/euocl/'
systems = [('quartz', cells.SYSTEMS['quartz'], dict(runpy.run_path(RULES)['BASIS'])),
           ('euocl', from_cif('/scratch4/lcheng24/xwang405/zmcscf/Eu/aimp/struct/oxyhalides/COD_1542170.cif'),
            runpy.run_path(P + '../basis_rules.py')['BASIS'])]
for name, sysm, B in systems:
    cell = make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, 4.))
    aohard = np.repeat([cell.bas_exp(i).min() >= 4. for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
    sm = np.where(~aohard)[0]; ns = len(sm)
    m0 = cell.cutoff_to_mesh(estimate_ke_cutoff(_subcell(cell, sm), 1e-10))
    ops = space_group_ops(cell)
    for label, m, orbits in (('blocked', m0, False), ('orbits', symmetric_mesh(m0, ops[:2], 3), True)):
        mc = np.asarray(m) // 3
        phic = cell.pbc_eval_gto('GTOval_sph', cell.get_uniform_grids(mc))[:, sm]
        tr0 = ((np.einsum('ri,ri->r', phic, phic)) ** 2).sum()
        kmax = 40 * ns
        if orbits:
            orb, mem = grid_orbits(mc, ops[:2])
            pv, L = pivots_orbits(phic, kmax, orb, mem)
        else:
            pv, L = pivots_blocked(phic, kmax)
        rel = 1 - np.cumsum((L ** 2).sum(axis=1)) / tr0
        out = ['f%d %.2e' % (f, rel[min(f * ns, len(rel)) - 1]) for f in (8, 12, 16, 20, 24, 32, 40)]
        print('%-7s %-8s %s ns %d, %d pivots max: rel residual %s' % (name, ops[2], label, ns, len(pv), ', '.join(out)), flush=True)
