#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Local fit of the pairs with a compact function (CX pairs) for socutils.pbc.thc, built without a global
# three-index tensor (HANDOFF Open item 2, route 1, step 1).
#
# For every atom A of the cell that carries compact functions (split basis, sr_locality.split_basis) and every
# atom B + L (L a lattice translation) within r_cut of A, the pair densities phi_c(r - R_A) phi_x(r - R_B - L)
# (c compact on A, x any function on B) are fitted with the auxiliary functions on the two atoms only, in the SR
# (erfc) metric, from molecular integrals over the two-atom cluster {A, B + L} (pair-atom PARI):
#     C^{AB,L} = M_DD^-1 (P|c x_L)_SR,    D = aux(A) u aux(B + L).
# The Gamma-point pair (c, x) is the sum over L of the image pairs, so its fit coefficients on the cell's aux
# functions (Bloch sums) are the sums over L of the image-pair coefficients:
#     C^Gamma_{P, (c, x)} = sum_L C^{AB,L}_{P', (c, x_L)},   P' the image of P in D.
# Pairs with two compact functions are fitted once, from the atom of the first (packed) index.
#
# The non-robust CX blocks of THCExchange need only these coefficients and the global SR metric of the aux
# functions (pbc/bench/thc_tests/nonrobust.py: within 1e-7 Eh of the robust form).
#

import time
import numpy as np
from pyscf import gto, lib
from pyscf.df import incore as df_incore
from socutils.pbc.bench.hybrid_k import _inv_metric


def _translations(cell, r_cut):
    A = np.asarray(cell.lattice_vectors())
    n = [int(np.ceil(r_cut / np.linalg.norm(A[i]))) + 1 for i in range(3)]
    T = np.array([[i, j, k] for i in range(-n[0], n[0] + 1) for j in range(-n[1], n[1] + 1)
                  for k in range(-n[2], n[2] + 1)]) @ A
    return T


def _atom_mol(cell, atoms, coords, basis):
    '''Molecule of the given atoms (cell atom indices) at the given positions, basis per atom label.'''
    syms = [cell.atom_symbol(i) for i in atoms]
    m = gto.Mole()
    m.atom = [(s, tuple(x)) for s, x in zip(syms, coords)]
    m.basis = {s: basis[s] for s in set(syms)}
    m.unit = 'B'
    m.spin = None
    m.verbose = 0
    m.cart = cell.cart
    m.build(False, False)
    return m


def local_cx_fit(cell, auxbasis, compact, omega=0.5, r_cut=6.0, r_dom=0., lindep=1e-12, verbose=True, sparse=False,
                 far=False, far_grid=(300, 302)):
    '''Gamma-point local fit coefficients of the CX pairs (dense (naux, npair) packed, for validation).

    Args:
        cell : the split-basis cell
        auxbasis : aux basis dict (per atom label, as for RSDF)
        compact : bool (nao,), compact functions of the cell
        r_cut : Angstrom, partner atoms (with images) within r_cut of the compact atom
        r_dom : Angstrom, the fit domain also takes the aux functions of the atoms (with images) within r_dom of
            the compact atom A (0: the pair's two atoms only)
        far : also fit the images beyond the explicit (r_cut) clusters, on A's aux functions only (far_cx_fit):
            c (compact, on A) times the lattice sum of the remaining images of x, which is smooth where c lives
    Returns C (naux, nao (nao+1) / 2) (dense, or scipy.sparse CSR with sparse=True), stats dict.'''
    from pyscf.df import addons as df_addons
    t0 = time.time()
    BOHR = 0.52917721092
    rc = r_cut / BOHR
    nao = cell.nao
    auxcell = df_addons.make_auxmol(cell, auxbasis)
    naux = auxcell.nao
    ao_loc = cell.aoslice_by_atom()[:, 2:4]
    aux_loc = auxcell.aoslice_by_atom()[:, 2:4]
    basis = cell._basis
    abasis = auxcell._basis
    X = cell.atom_coords()
    T = _translations(cell, rc)
    ti, tj = np.tril_indices(nao)
    pidx = np.full((nao, nao), -1, dtype=np.int64)
    pidx[ti, tj] = np.arange(len(ti))
    pidx[tj, ti] = np.arange(len(ti))
    C = None if sparse else np.zeros((naux, len(ti)))
    coo = ([], [], [])                                                 # rows, cols, values (sparse)
    Cacc = None                                                        # CSR sum of the folded triplets
    ncluster = nimgpair = 0
    emin = np.array([min(cell.bas_exp(i).min() for i in range(cell.nbas) if cell.bas_atom(i) == at)
                     for at in range(cell.natm)])
    alpha_c = min(cell.bas_exp(i).min() for i in range(cell.nbas)
                  if compact[cell.ao_loc_nr()[i]])
    metric_cache = {}
    nskip = 0
    near = {A: [] for A in range(cell.natm)}                          # explicit (B, translation) per compact atom
    for A in range(cell.natm):
        a0, a1 = ao_loc[A]
        cA = np.where(compact[a0:a1])[0] + a0
        if len(cA) == 0:
            continue
        for B in range(cell.natm):
            b0, b1 = ao_loc[B]
            xs_all = np.arange(b0, b1)
            # pairs fitted from this side: c compact on A; for two compact functions only the packed
            # (larger index first) orientation, i.e. skip (c, x) with x compact and x > c
            keep = ~(compact[xs_all][None, :] & (xs_all[None, :] > cA[:, None]))
            if not keep.any():
                continue
            d = np.linalg.norm(X[B] + T - X[A], axis=1)
            for L in np.where(d <= rc)[0]:
                # the product of a compact function (>= alpha_c) and the most diffuse function of B is negligible
                ab = alpha_c * emin[B] / (alpha_c + emin[B])
                if ab * d[L] ** 2 > 36.:
                    nskip += 1
                    continue
                near[A].append((B, T[L]))
                same = B == A and d[L] < 1e-8
                atoms = [A] if same else [A, B]
                coords = [X[A]] if same else [X[A], X[B] + T[L]]
                mol = _atom_mol(cell, atoms, coords, basis)
                nbA = sum(1 for i in range(mol.nbas) if mol.bas_atom(i) == 0)
                # fit domain in a canonical order (the metric of (A, B + L) and (B, A - L) is computed once)
                datoms, dcoords = list(atoms), list(coords)
                for E in range(cell.natm):
                    for t_ in T[np.linalg.norm(X[E] + T - X[A], axis=1) <= r_dom / BOHR]:
                        xe = X[E] + t_
                        if all(np.linalg.norm(xe - c_) > 1e-6 for c_ in dcoords):
                            datoms.append(E)
                            dcoords.append(xe)
                order = sorted(range(len(datoms)), key=lambda k: (datoms[k], tuple(np.round(dcoords[k] - dcoords[0], 6))))
                rel = [np.round(dcoords[k] - dcoords[order[0]], 6) for k in order]
                key = tuple((datoms[k], tuple(r_)) for k, r_ in zip(order, rel))
                if len(order) == 2 and datoms[order[0]] == datoms[order[1]] and tuple(rel[1]) < (0., 0., 0.):
                    order = order[::-1]                                # images of one atom: one orientation
                    key = tuple((datoms[k], tuple(np.round(dcoords[k] - dcoords[order[0]], 6))) for k in order)
                datoms = [datoms[k] for k in order]
                dcoords = [dcoords[k] for k in order]
                amol = _atom_mol(auxcell, datoms, dcoords, abasis)
                with mol.with_range_coulomb(-omega), amol.with_range_coulomb(-omega):
                    jsl = (0, nbA) if same else (nbA, mol.nbas)
                    j3 = df_incore.aux_e2(mol, amol, intor='int3c2e', aosym='s1',
                                          shls_slice=(0, nbA) + jsl + (0, amol.nbas))
                    if key not in metric_cache:
                        m2 = amol.intor('int2c2e')
                        metric_cache[key] = _inv_metric(.5 * (m2 + m2.T), lindep)
                v, e = metric_cache[key]
                blk = j3[cA - a0]                                  # (n_c, n_x, naux_m)
                coef = ((v / e) @ (v.T @ blk.reshape(-1, blk.shape[2]).T)).T.reshape(blk.shape)
                pa = np.concatenate([np.arange(*aux_loc[at]) for at in datoms])
                cols = pidx[cA[:, None], xs_all[None, :]]
                kk = np.where(keep)
                if sparse:
                    vals = coef[kk[0], kk[1]].T                         # (naux_m, nk)
                    coo[0].append(np.repeat(pa, vals.shape[1]))
                    coo[1].append(np.tile(cols[kk], len(pa)))
                    coo[2].append(vals.ravel())
                else:
                    Cblk = np.zeros((naux, len(kk[0])))
                    np.add.at(Cblk, pa, coef[kk[0], kk[1]].T)
                    C[:, cols[kk]] += Cblk
                ncluster += 1
                nimgpair += len(kk[0])
        metric_cache.clear() if len(metric_cache) > 2000 else None
        if sparse and coo[0]:                                          # fold this atom's triplets into CSR (memory)
            Cacc = _coo_add(Cacc, coo, naux, len(ti))
    if far:
        tf = time.time()
        rows_f, cols_f, vals_f = far_cx_fit(cell, auxcell, compact, omega, near, pidx, lindep, far_grid)
        if sparse:
            coo[0].append(rows_f); coo[1].append(cols_f); coo[2].append(vals_f)
            Cacc = _coo_add(Cacc, coo, naux, len(ti))
        else:
            np.add.at(C, (rows_f, cols_f), vals_f)
        if verbose:
            print('far CX fit (beyond the clusters, one-centre aux): %d coefficients (%.0f s)'
                  % (len(vals_f), time.time() - tf), flush=True)
    stats = {'clusters': ncluster, 'image pairs': nimgpair, 'skipped': nskip, 'time': time.time() - t0}
    if verbose:
        print('local CX fit: %d two-atom clusters (%d negligible skipped), %d image pairs, r_cut %.1f A (%.0f s)'
              % (ncluster, nskip, nimgpair, r_cut, stats['time']), flush=True)
    if sparse:
        if coo[0]:
            Cacc = _coo_add(Cacc, coo, naux, len(ti))
        C = Cacc
        C.eliminate_zeros()
        stats['nnz'] = C.nnz
    return C, stats


def _coo_add(Cacc, coo, nrow, ncol):
    '''Cacc + the triplets in coo (duplicates summed), as CSR; empties coo.'''
    import scipy.sparse
    C = scipy.sparse.coo_matrix((np.concatenate(coo[2]), (np.concatenate(coo[0]), np.concatenate(coo[1]))),
                                shape=(nrow, ncol)).tocsr()
    for lst in coo:
        lst.clear()
    return C if Cacc is None else Cacc + C


def _lebedev(nang):
    '''Lebedev unit vectors and weights (summing to 4 pi; pyscf's sum to 1).'''
    from pyscf.dft import gen_grid
    ang = gen_grid.MakeAngularGrid(nang)
    return ang[:, :3], 4 * np.pi * ang[:, 3]


def _sr_potential_at(amol, pts, omega):
    '''u_Q(g) = int Q(r) erfc(omega |r - g|) / |r - g| dr for the functions Q of amol at the points g, (npts, nQ):
    int1e_grids of the pairs (Q, s0) with s0 an s function of negligible exponent (constant over Q), divided by its
    value.'''
    m = amol.copy(deep=True)
    eps = 1e-14
    xyz = m.atom_coord(0)
    s0 = gto.M(atom=[('H', tuple(xyz))], basis={'H': [[0, [eps, 1.]]]}, unit='B', spin=None, verbose=0)
    mm = gto.conc_mol(m, s0)
    nQ = m.nbas
    v0 = s0.eval_gto('GTOval_sph', xyz[None])[0, 0]
    with mm.with_range_coulomb(-omega):
        u = mm.intor('int1e_grids', grids=pts, shls_slice=(0, nQ, nQ, nQ + 1))      # (npts, nQ_ao, 1)
    return u[:, :, 0] / v0


def far_cx_fit(cell, auxcell, compact, omega, near, pidx, lindep=1e-12, grid=(300, 302), r_thr=36.):
    '''Fit of the CX image pairs beyond the explicit clusters (near[A]: the (B, translation) fitted explicitly):
    for c compact on A,  x_far(r) = sum over the remaining images of x (the lattice sum minus the explicit images)
    is smooth where c lives, and the product c x_far is fitted with A's aux functions only (SR metric):
        (Q | c x_far)_SR = int c(r) x_far(r) u_Q(r) dr   on an atomic grid of A (u_Q: SR potential of Q).
    Returns the COO triplets (aux rows of the cell aux, packed pair columns, values).'''
    from pyscf.dft import radi
    X = cell.atom_coords()
    ao_loc = cell.aoslice_by_atom()[:, 2:4]
    aux_loc = auxcell.aoslice_by_atom()[:, 2:4]
    nrad, nang = grid
    alpha_c = min(cell.bas_exp(i).min() for i in range(cell.nbas) if compact[cell.ao_loc_nr()[i]])
    r_max = np.sqrt(r_thr / alpha_c)                                   # where the compact functions live
    xyz_ang, w_ang = _lebedev(nang)
    xs_ok = ~compact                                                     # far partners: the non-compact functions
    rows, cols, vals = [], [], []
    for A in range(cell.natm):
        a0, a1 = ao_loc[A]
        cA = np.where(compact[a0:a1])[0] + a0
        if len(cA) == 0:
            continue
        r, dr = radi.mura_knowles(nrad, cell.atom_charge(A))
        sel = r < r_max
        r, wr = r[sel], (dr * r ** 2)[sel]
        pts = (r[:, None, None] * xyz_ang[None]).reshape(-1, 3) + X[A]
        w = (wr[:, None] * w_ang[None]).ravel()
        # c on A (home cell only)
        molA = _atom_mol(cell, [A], [X[A]], cell._basis)
        cv = molA.eval_gto('GTOval_sph', pts)[:, cA - a0]               # (npts, n_c)
        # x_far: lattice sum minus the explicit images
        xf = cell.pbc_eval_gto('GTOval_sph', pts)                       # (npts, nao)
        for B, t in near[A]:
            mB = _atom_mol(cell, [B], [X[B] + t], cell._basis)
            xf[:, ao_loc[B][0]:ao_loc[B][1]] -= mB.eval_gto('GTOval_sph', pts)
        xf[:, ~xs_ok] = 0.
        xcols = np.where(xs_ok)[0]
        xf = np.ascontiguousarray(xf[:, xcols])
        # SR potentials of A's aux functions and the one-centre SR metric
        amol = _atom_mol(auxcell, [A], [X[A]], auxcell._basis)
        u = _sr_potential_at(amol, pts, omega)                           # (npts, nQ)
        with amol.with_range_coulomb(-omega):
            m2 = amol.intor('int2c2e')
        v, e = _inv_metric(.5 * (m2 + m2.T), lindep)
        Minv = (v / e) @ v.T
        uw = u * w[:, None]
        pa = np.arange(*aux_loc[A])
        for k, c in enumerate(cA):
            J = (uw * cv[:, k:k + 1]).T @ xf                             # (nQ, n_x) = (Q | c x_far)_SR
            coef = Minv @ J
            big = np.abs(coef) > 1e-14
            qq, xx = np.nonzero(big)
            rows.append(pa[qq])
            cols.append(pidx[c, xcols[xx]])
            vals.append(coef[qq, xx])
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
