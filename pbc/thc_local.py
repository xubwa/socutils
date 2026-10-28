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


def local_cx_fit(cell, auxbasis, compact, omega=0.5, r_cut=6.0, r_dom=0., lindep=1e-12, verbose=True, sparse=False):
    '''Gamma-point local fit coefficients of the CX pairs (dense (naux, npair) packed, for validation).

    Args:
        cell : the split-basis cell
        auxbasis : aux basis dict (per atom label, as for RSDF)
        compact : bool (nao,), compact functions of the cell
        r_cut : Angstrom, partner atoms (with images) within r_cut of the compact atom
        r_dom : Angstrom, the fit domain also takes the aux functions of the atoms (with images) within r_dom of
            the compact atom A (0: the pair's two atoms only)
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
    ncluster = nimgpair = 0
    emin = np.array([min(cell.bas_exp(i).min() for i in range(cell.nbas) if cell.bas_atom(i) == at)
                     for at in range(cell.natm)])
    alpha_c = min(cell.bas_exp(i).min() for i in range(cell.nbas)
                  if compact[cell.ao_loc_nr()[i]])
    metric_cache = {}
    nskip = 0
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
    stats = {'clusters': ncluster, 'image pairs': nimgpair, 'skipped': nskip, 'time': time.time() - t0}
    if verbose:
        print('local CX fit: %d two-atom clusters (%d negligible skipped), %d image pairs, r_cut %.1f A (%.0f s)'
              % (ncluster, nskip, nimgpair, r_cut, stats['time']), flush=True)
    if sparse:
        import scipy.sparse
        C = scipy.sparse.coo_matrix((np.concatenate(coo[2]), (np.concatenate(coo[0]), np.concatenate(coo[1]))),
                                    shape=(naux, len(ti))).tocsr()      # duplicates (images, repeated aux) summed
        C.eliminate_zeros()
        stats['nnz'] = C.nnz
    return C, stats
