'''ISDF rank of the smooth-smooth pair densities (HANDOFF Open item 2, the grid half of the compact/smooth split).

The basis is split at alpha_c (sr_locality.split_basis); the smooth part (exponents < alpha_c) is evaluated at the
Gamma point on the uniform grid that pyscf's ke-cutoff estimate gives for it.  ISDF interpolates the pair densities
rho_mn(r) = phi_m(r) phi_n(r) from their values at N_mu interpolation points; the points are the pivots of a
pivoted Cholesky of the grid Gram matrix G(r, r') = sum_mn rho_mn(r) rho_mn(r') = (Phi Phi^T)(r, r')^2, built
column by column (never the full ngrid^2 matrix).  The residual trace of G after k pivots is the squared
Frobenius error of the interpolated pair densities; reported is N_mu / nao_smooth at relative errors 1e-3..1e-8.

usage: isdf_rank.py SYSTEM ALPHA_C [ALPHA_C ...]   (SYSTEM: cells.SYSTEMS name or CIF; bases from the production rules)'''
import sys, time, runpy, numpy as np
from pyscf.pbc import gto as pgto
from pyscf.pbc.gto.cell import estimate_ke_cutoff
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES

name = sys.argv[1]
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
for ac in [float(x) for x in sys.argv[2:]]:
    t = time.time()
    full = cells.make_cell(sysm, 1e-10, nucmod='G', basis=B)
    cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac, keep='smooth'))
    ke = estimate_ke_cutoff(cell, 1e-10)
    mesh = cell.cutoff_to_mesh(ke)
    coords = cell.get_uniform_grids(mesh)
    phi = cell.pbc_eval_gto('GTOval_sph', coords)                    # (ngrid, nao_s)
    ng, nao = phi.shape
    w = cell.vol / ng
    diag = np.einsum('ri,ri->r', phi, phi) ** 2
    tr0 = diag.sum()
    print('\n%s alpha_c %g: nao %d (full basis %d), smooth max exponent %.3g, ke_cutoff %.0f Eh, mesh %s = %d points (%.0f s)'
          % (name, ac, nao, full.nao, max(np.hstack(cell.bas_exps()).max(), 0), ke, mesh, ng, time.time() - t), flush=True)
    kmax = min(ng, 40 * nao, int(float(__import__('os').environ.get('ISDF_MEM_GB', '100')) * 1e9 / 8 / ng))
    L = np.zeros((kmax, ng))
    res = diag.copy()
    hit = {}
    t = time.time()
    for k in range(kmax):
        p = int(np.argmax(res))
        rel = res.sum() / tr0
        for tol in TOLS:
            if tol not in hit and rel <= tol:
                hit[tol] = k
        if len(hit) == len(TOLS) or res[p] <= 0:
            break
        col = (phi @ phi[p]) ** 2                                      # G(:, p)
        col -= L[:k].T @ L[:k, p]
        L[k] = col / np.sqrt(res[p])
        res -= L[k] ** 2
        np.maximum(res, 0, out=res)
    print('   ISDF points N_mu (N_mu/nao_smooth) at relative pair-density error: ' + '  '.join(
        '%.0e: %s' % (tol, '%d (%.1f)' % (hit[tol], hit[tol] / nao) if tol in hit else '>%d' % kmax) for tol in TOLS)
        + '  (%.0f s)' % (time.time() - t), flush=True)
    nm = hit.get(1e-6, kmax)
    print('   storage at 1e-6 (doubles): THC V_PQ + phi(r_P) N_mu^2 + N_mu nao %.2e  vs  three-index with naux = N_mu, '
          'N_mu nao(nao+1)/2 %.2e' % (nm * nm + nm * nao, nm * nao * (nao + 1) / 2), flush=True)
    L = phi = None
