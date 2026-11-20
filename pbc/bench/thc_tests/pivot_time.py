'''Point-selection time alone on a production cell (no RSDF / local fit): split cell at alpha_c 4, smooth functions,
THC mesh, coarse grid, pivots for N_mu = 16 x nsmooth; column-by-column (coarsen 2, timed on a few hundred pivots and
extrapolated) vs blocked (coarsen 3 and 4, full), and overlap of the selected point sets.

usage: pivot_time.py CIF NX,NY,NZ BASIS_PY [ACTIVE_LABEL]'''
import sys, time, runpy, numpy as np
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis
from socutils.pbc.thc import pivots, pivots_blocked, _subcell
from pyscf.pbc.gto.cell import estimate_ke_cutoff
cif, sc, basis_py = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
lab = sys.argv[4] if len(sys.argv) > 4 else None
sysm = cells.from_cif(cif)
if sc != [1, 1, 1]:
    sysm = cells.supercell(sysm, sc)
if lab:
    k, at = 0, []
    for e, x in sysm[1]:
        if e == 'Eu':
            e = lab if k == 0 else e
            k += 1
        at.append((e, x))
    sysm = (sysm[0], at)
B = dict(runpy.run_path(basis_py)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, 4.), max_memory=380000)
aohard = np.repeat([cell.bas_exp(i).min() >= 4. for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
smooth = np.where(~aohard)[0]
mesh = cell.cutoff_to_mesh(estimate_ke_cutoff(_subcell(cell, smooth), 1e-10))
kmax = 16 * len(smooth)
print('%d atoms, nao %d, smooth %d, mesh %s, N_mu %d' % (cell.natm, cell.nao, len(smooth), mesh, kmax), flush=True)
for co, blocked in ((3, True), (4, True), (2, False)):
    mc = np.maximum(np.asarray(mesh) // co, 1)
    t = time.time()
    phic = cell.pbc_eval_gto('GTOval_sph', cell.get_uniform_grids(mc))[:, smooth]
    tg = time.time() - t
    t = time.time()
    if blocked:
        p, _ = pivots_blocked(phic, min(len(phic), kmax))
        print('coarsen %d blocked: %d candidates, grid %.0f s, pivots %.0f s (%d points)'
              % (co, len(phic), tg, time.time() - t, len(p)), flush=True)
    else:
        n = 300
        p, _ = pivots(phic, n)
        tp = time.time() - t                                   # cost ~ k^2 ng: extrapolate from n pivots
        print('coarsen %d column-by-column: %d candidates, grid %.0f s, %d pivots %.0f s -> %d pivots ~%.0f s (est.)'
              % (co, len(phic), tg, n, tp, kmax, tp * (kmax / n) ** 2), flush=True)
    phic = p = None
