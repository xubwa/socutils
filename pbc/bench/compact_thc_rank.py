'''THC rank of the pairs with a compact function on an atom-centred grid (HANDOFF Open item 2, a possible level 4:
THC for every pair, uniform grid for the smooth part and atomic grids for the compact part).

The basis is split at alpha_c (sr_locality.split_basis).  For each symmetry-distinct atom A: an atomic grid at A
(pyscf Treutler radial x Lebedev angular), the compact functions c of A and all functions x of the crystal
(Gamma-point lattice sums) on it, the pair densities rho_cx(r) = phi_c(r) phi_x(r) with the quadrature weights,
and a pivoted Cholesky of their Gram matrix G(r, r') = sqrt(w w') (Phi_c Phi_c^T)(r, r') (Phi_x Phi_x^T)(r, r'),
built column by column.  Reported: points N_mu,A at relative errors 1e-4..1e-8 (weighted L2 of the interpolated
pair densities), against the number of compact functions on A and of aux functions on A.

usage: compact_thc_rank.py SYSTEM ALPHA_C [NRAD NANG]'''
import sys, time, runpy, numpy as np, scipy.linalg
from pyscf import gto, dft
from socutils.pbc.bench import cells
from socutils.pbc.bench.sr_locality import split_basis, RULES
from socutils.pbc.bench.auxgen_util import auxgen_from_basis

name, ac = sys.argv[1], float(sys.argv[2])
nrad, nang = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (120, 434)
sysm = cells.from_cif(name) if name.endswith('.cif') else cells.SYSTEMS[name]
B = dict(runpy.run_path(RULES)['BASIS'])
cell = cells.make_cell(sysm, 1e-10, nucmod='G', basis=split_basis(B, ac), max_memory=150000)
nao = cell.nao
aohard = np.repeat([cell.bas_exp(i).min() >= ac for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
aoat = np.repeat([cell.bas_atom(i) for i in range(cell.nbas)], np.diff(cell.ao_loc_nr()))
print('%s alpha_c %g: split nao %d, compact %d; atomic grid %d x %d' % (name, ac, nao, aohard.sum(), nrad, nang), flush=True)
TOLS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
done = set()
for A in range(cell.natm):
    el = cell.atom_symbol(A)
    if el in done:
        continue
    done.add(el)
    c_idx = np.where(aohard & (aoat == A))[0]
    if len(c_idx) == 0:
        continue
    t = time.time()
    m = gto.M(atom=[(el, (0., 0., 0.))], basis={el: B[el]}, unit='B', verbose=0, spin=None)
    g = dft.gen_grid.Grids(m)
    g.atom_grid = {el: (nrad, nang)}
    g.prune = None
    g.build()
    coords = g.coords + cell.atom_coord(A)
    w = np.sqrt(np.abs(g.weights))
    phi = cell.pbc_eval_gto('GTOval_sph', coords)                     # (npt, nao), Gamma lattice sums
    pc = phi[:, c_idx] * w[:, None] ** .5
    px = phi * w[:, None] ** .5                                         # sqrt(w) split over the two factors
    npt = len(coords)
    # equilibrate: the core pairs at the nucleus dominate the Gram diagonal by ~1e15 and exhaust double precision
    # before the valence-scale pairs are resolved; scale each function to a comparable size on this grid (the
    # relative per-pair error does not depend on it, only the pivot order does)
    xn = np.linalg.norm(px, axis=0)
    px = px / np.maximum(xn, 1e-3 * xn.max())
    pc = pc / np.linalg.norm(pc, axis=0)
    diag = np.einsum('ri,ri->r', pc, pc) * np.einsum('ri,ri->r', px, px)
    tr0 = diag.sum()
    # pivots by pivoted Cholesky of the Gram matrix; the error is measured per pair: ISDF interpolation
    # xi = G[:, P] G[P, P]^-1 and, for every compact-compact pair of A and 2000 random (c, x) pairs,
    # max over pairs of |rho - xi rho(P)|_w / |rho|_w (the total trace is dominated by the core pairs)
    nc = len(c_idx)
    kmax = min(npt, 40 * nc, int(40e9 / 8 / npt))
    L = np.zeros((kmax, npt))
    res = diag.copy()
    piv = []
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a in range(nc) for b in c_idx[: a + 1]] + \
        [(int(rng.integers(nc)), int(rng.integers(nao))) for _ in range(2000)]
    Z = np.array([pc[:, a] * px[:, b] for a, b in pairs]).T          # (npt, npairs), weighted
    zn = np.linalg.norm(Z, axis=0)
    # keep the pairs significant for their compact function (>= 1e-3 of its largest pair on this grid)
    ca = np.array([a for a, _ in pairs])
    cmax = np.array([np.linalg.norm(pc[:, a:a + 1] * px, axis=0).max() for a in range(nc)])
    sig = zn >= 1e-3 * cmax[ca]
    Z, zn = Z[:, sig], zn[sig]
    checks = sorted(set(int(nc * f) for f in (0.5, 1, 2, 4, 8, 16, 32) if 1 <= int(nc * f) < kmax))
    hit = {}
    dmax = diag.max()
    for k in range(kmax + 1):
        p = int(np.argmax(res)) if k < kmax else -1
        if k == kmax or res[p] <= 1e-15 * dmax or p in piv:      # numerical rank reached
            if k and k not in hit:
                P = np.array(piv)
                LP = L[:k][:, P]
                xi = scipy.linalg.solve_triangular(LP, L[:k], lower=False).T
                err = np.linalg.norm(Z - xi @ Z[P], axis=0) / zn
                hit[k] = (err.max(), np.median(err))
            print('     numerical rank %d (%s)' % (k, 'roundoff' if k < kmax else 'kmax'), flush=True)
            break
        col = (pc @ pc[p]) * (px @ px[p])
        piv.append(p)
        col = col - L[:k].T @ L[:k, p]
        L[k] = col / np.sqrt(res[p])
        res -= L[k] ** 2
        np.maximum(res, 0, out=res)
        if k + 1 in checks:
            # xi = G[:, P] G[P, P]^-1 = L^T L_P^-T with the Cholesky factor (stable; G[P, P] = L_P^T L_P)
            P = np.array(piv)
            LP = L[:k + 1][:, P]                     # L[j, p_i] = 0 for j > i: upper triangular (rows j, columns i)
            xi = scipy.linalg.solve_triangular(LP, L[:k + 1], lower=False).T   # xi^T = L_P^-1 L, (npt, k+1)
            err = np.linalg.norm(Z - xi @ Z[P], axis=0) / zn
            hit[k + 1] = (err.max(), np.median(err))
    print('  %s: %d compact functions x %d (all), %d grid points, %d test pairs; per-pair ISDF error (max / median) at '
          'N_mu = %s' % (el, nc, nao, npt, Z.shape[1], '  '.join('%d (%.1fx): %.1e/%.1e' % (k, k / nc, a, b)
                                                               for k, (a, b) in sorted(hit.items()))), flush=True)
    naux_A = gto.M(atom=[(el, (0., 0., 0.))], basis=auxgen_from_basis({el: B[el]}, contract=False, prune_lmax=False,
                                                                       threshold=1e-8), unit='B', verbose=0, spin=None).nao
    print('     aux functions on the atom: %d (%.0f s)' % (naux_A, time.time() - t), flush=True)
    L = phi = px = pc = None
