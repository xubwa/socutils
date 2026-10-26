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


def local_cx_fit(cell, auxbasis, compact, omega=0.5, r_cut=6.0, r_dom=0., lindep=1e-12, verbose=True):
    '''Gamma-point local fit coefficients of the CX pairs (dense (naux, npair) packed, for validation).

    Args:
        cell : the split-basis cell
        auxbasis : aux basis dict (per atom label, as for RSDF)
        compact : bool (nao,), compact functions of the cell
        r_cut : Angstrom, partner atoms (with images) within r_cut of the compact atom
        r_dom : Angstrom, the fit domain also takes the aux functions of the atoms (with images) within r_dom of
            the compact atom A (0: the pair's two atoms only)
    Returns C (naux, nao (nao+1) / 2), stats dict.'''
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
    C = np.zeros((naux, len(ti)))
    ncluster = nimgpair = 0
    for A in range(cell.natm):
        a0, a1 = ao_loc[A]
        cA = np.where(compact[a0:a1])[0] + a0
        if len(cA) == 0:
            continue
        for B in range(cell.natm):
            b0, b1 = ao_loc[B]
            d = np.linalg.norm(X[B] + T - X[A], axis=1)
            for L in np.where(d <= rc)[0]:
                same = B == A and d[L] < 1e-8
                atoms = [A] if same else [A, B]
                coords = [X[A]] if same else [X[A], X[B] + T[L]]
                mol = _atom_mol(cell, atoms, coords, basis)
                # fit domain: the pair's atoms + the neighbours of A within r_dom
                datoms, dcoords = list(atoms), list(coords)
                for E in range(cell.natm):
                    for t_ in T[np.linalg.norm(X[E] + T - X[A], axis=1) <= r_dom / BOHR]:
                        xe = X[E] + t_
                        if all(np.linalg.norm(xe - c_) > 1e-6 for c_ in dcoords):
                            datoms.append(E)
                            dcoords.append(xe)
                amol = _atom_mol(auxcell, datoms, dcoords, abasis)
                with mol.with_range_coulomb(-omega), amol.with_range_coulomb(-omega):
                    j3 = df_incore.aux_e2(mol, amol, intor='int3c2e', aosym='s1')   # (nao_m, nao_m, naux_m)
                    m2 = amol.intor('int2c2e')
                nA = a1 - a0
                ic = cA - a0                                      # compact functions of A in the cluster
                jx = np.arange(nA) if same else nA + np.arange(b1 - b0)   # functions of B (+ L)
                xs = np.arange(a0, a1) if same else np.arange(b0, b1)
                blk = j3[np.ix_(ic, jx)]                         # (n_c, n_x, naux_m)
                # pairs fitted from this side: c compact on A; for two compact functions only the packed
                # (larger index first) orientation, i.e. skip (c, x) with x compact and x > c
                keep = np.ones((len(cA), len(xs)), bool)
                both = compact[xs][None, :] & np.ones((len(cA), 1), bool)
                keep &= ~(both & (xs[None, :] > cA[:, None]))
                if not keep.any():
                    continue
                v, e = _inv_metric(.5 * (m2 + m2.T), lindep)
                coef = ((v / e) @ (v.T @ blk.reshape(-1, blk.shape[2]).T)).T.reshape(blk.shape)
                # aux of the cluster -> cell aux index
                pa = np.concatenate([np.arange(*aux_loc[at]) for at in datoms])
                cols = pidx[cA[:, None], xs[None, :]]
                kk = np.where(keep)
                Cblk = np.zeros((naux, len(kk[0])))
                np.add.at(Cblk, pa, coef[kk[0], kk[1]].T)
                C[:, cols[kk]] += Cblk
                ncluster += 1
                nimgpair += len(kk[0])
    stats = {'clusters': ncluster, 'image pairs': nimgpair, 'time': time.time() - t0}
    if verbose:
        print('local CX fit: %d two-atom clusters, %d image pairs, r_cut %.1f A (%.0f s)'
              % (ncluster, nimgpair, r_cut, stats['time']), flush=True)
    return C, stats
