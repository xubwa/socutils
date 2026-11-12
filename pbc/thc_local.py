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
                 far=False, far_grid=(300, 302), comm=None, rows_to_owner=False):
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
        comm : MPI (mpi4py, sparse only): the compact atoms split over the ranks (greedy by size), the partial
            CSR matrices summed on every rank, or with rows_to_owner only the rows of the aux atoms a rank owns
            (owner: greedy by the global nonzeros of an aux atom's rows; stats['row_owner'], per aux atom) summed
            on that rank -- the other rows empty, C distributed
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
    # triplets (16 bytes; ~3x that transiently in the COO -> CSR fold) held at most to ~10% of the budget
    from pyscf import lib as _lib
    ncoo, coo_max = 0, int(max(2e7, 0.1 * max(0., cell.max_memory - _lib.current_memory()[0]) * 1e6 / 48))
    ncluster = nimgpair = 0
    emin = np.array([min(cell.bas_exp(i).min() for i in range(cell.nbas) if cell.bas_atom(i) == at)
                     for at in range(cell.natm)])
    alpha_c = min(cell.bas_exp(i).min() for i in range(cell.nbas)
                  if compact[cell.ao_loc_nr()[i]])
    # inverse SR metrics of the fit domains, by domain geometry: an LRU cache bounded in bytes (15% of the budget;
    # an Eu-Eu domain's eigenvectors are ~0.26 GB, and an entry-count bound let EuOCl grow it to ~20 GB)
    from collections import OrderedDict
    metric_cache = OrderedDict()
    mc_bytes, mc_max = [0], 0.15 * max(0., cell.max_memory - _lib.current_memory()[0]) * 1e6
    nskip = 0
    near = {A: [] for A in range(cell.natm)}                          # explicit (B, translation) per compact atom
    my_atoms = _my_compact_atoms(cell, compact, comm)
    for A in range(cell.natm):
        if A not in my_atoms:
            continue
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
                jsl = (0, nbA) if same else (nbA, mol.nbas)
                if key in metric_cache:
                    metric_cache.move_to_end(key)
                else:
                    with amol.with_range_coulomb(-omega):
                        m2 = amol.intor('int2c2e')
                    metric_cache[key] = _inv_metric(.5 * (m2 + m2.T), lindep)
                    mc_bytes[0] += metric_cache[key][0].nbytes + metric_cache[key][1].nbytes
                    while mc_bytes[0] > mc_max and len(metric_cache) > 1:     # least recently used out
                        _, (v_, e_) = metric_cache.popitem(last=False)
                        mc_bytes[0] -= v_.nbytes + e_.nbytes
                v, e = metric_cache[key]
                ve = v / e
                pa = np.concatenate([np.arange(*aux_loc[at]) for at in datoms])
                # (c x | P)_SR only for the compact shells of A (all of A's shells gave (297, 297, 4818) = 3.4 GB
                # per Eu cluster), in batches of rows within ~15% of the budget (j3, its fit and the kept values)
                naux_m, nx_ = amol.nao, jsl[1] - jsl[0]
                nx_ = mol.ao_loc_nr()[jsl[1]] - mol.ao_loc_nr()[jsl[0]]
                avail_ = max(0., cell.max_memory - _lib.current_memory()[0]) * 1e6
                rows_max = int(max(1, 0.15 * avail_ / (24. * nx_ * naux_m)))
                mloc = mol.ao_loc_nr()
                cshells = [i for i in range(nbA) if compact[a0 + mloc[i]]]
                batches, cur, nrow = [], [], 0
                for i in cshells:                                      # runs of consecutive compact shells
                    di = mloc[i + 1] - mloc[i]
                    if cur and (i != cur[-1] + 1 or nrow + di > rows_max):
                        batches.append(cur); cur, nrow = [], 0
                    cur.append(i); nrow += di
                if cur:
                    batches.append(cur)
                nk_cluster = 0
                for sh in batches:
                    s0, s1 = sh[0], sh[-1] + 1
                    with mol.with_range_coulomb(-omega), amol.with_range_coulomb(-omega):
                        j3 = df_incore.aux_e2(mol, amol, intor='int3c2e', aosym='s1',
                                              shls_slice=(s0, s1) + jsl + (0, amol.nbas))   # (n_cb, n_x, naux_m)
                    crow = a0 + np.arange(mloc[s0], mloc[s1])                 # these rows' AOs (compact, on A)
                    ic = np.searchsorted(cA, crow)                             # their positions in cA
                    coef = (ve @ (v.T @ j3.reshape(-1, naux_m).T)).T.reshape(j3.shape)
                    j3 = None
                    kp = keep[ic]
                    kk = np.where(kp)
                    cols = pidx[crow[:, None], xs_all[None, :]]
                    if sparse:
                        vals = coef[kk[0], kk[1]].T                     # (naux_m, nk)
                        coo[0].append(np.repeat(pa.astype(np.int32), vals.shape[1]))
                        coo[1].append(np.tile(cols[kk].astype(np.int32), len(pa)))
                        coo[2].append(vals.ravel())
                        ncoo += vals.size
                        if ncoo > coo_max:                             # fold into CSR within the memory budget
                            Cacc = _coo_add(Cacc, coo, naux, len(ti))
                            ncoo = 0
                    else:
                        Cblk = np.zeros((naux, len(kk[0])))
                        np.add.at(Cblk, pa, coef[kk[0], kk[1]].T)
                        C[:, cols[kk]] += Cblk
                    coef = vals = None
                    nk_cluster += len(kk[0])
                kk = (np.zeros(nk_cluster),)
                ncluster += 1
                nimgpair += len(kk[0])
        if sparse and coo[0]:                                          # fold this atom's triplets into CSR (memory)
            Cacc = _coo_add(Cacc, coo, naux, len(ti))
            ncoo = 0
    if far:
        tf = time.time()
        rows_f, cols_f, vals_f = far_cx_fit(cell, auxcell, compact, omega, near, pidx, lindep, far_grid,
                                            atoms=my_atoms)
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
        if Cacc is None:
            import scipy.sparse
            Cacc = scipy.sparse.csr_matrix((naux, len(ti)))
        if comm is not None and rows_to_owner:                        # rows summed on their owner only
            tg = time.time()
            Cacc, stats['row_owner'] = _reduce_rows_to_owner(comm, Cacc, aux_loc)
            if verbose:
                print('local CX fit: partial C of %d ranks summed by aux-atom owner (%.0f s)'
                      % (comm.Get_size(), time.time() - tg), flush=True)
        elif comm is not None:                                        # sum of the ranks' partial matrices
            tg = time.time()
            Cacc = _allsum_csr(comm, Cacc)
            if verbose:
                print('local CX fit: partial C of %d ranks summed (%.0f s)' % (comm.Get_size(), time.time() - tg),
                      flush=True)
        C = Cacc
        C.eliminate_zeros()
        stats['nnz'] = C.nnz
    return C, stats


def _my_compact_atoms(cell, compact, comm):
    '''The compact atoms of this rank: all without comm, else a greedy split by the number of compact functions.'''
    loc = cell.aoslice_by_atom()[:, 2:4]
    w = [int(compact[a0:a1].sum()) for a0, a1 in loc]
    atoms = [A for A in range(cell.natm) if w[A] > 0]
    if comm is None or comm.Get_size() == 1:
        return set(atoms)
    size, rank = comm.Get_size(), comm.Get_rank()
    load = np.zeros(size)
    own = {}
    for A in sorted(atoms, key=lambda a: -w[a]):
        k = int(np.argmin(load))
        own[A] = k
        load[k] += w[A]
    return {A for A, k in own.items() if k == rank}


def send_array(comm, a, dest, chunk=1 << 28):
    '''Send of a numpy array of any size (chunks below the 2^31-element MPI limit).'''
    comm.send((a.shape, a.dtype.str), dest=dest, tag=71)
    flat = np.ascontiguousarray(a).reshape(-1)
    for i in range(0, flat.size, chunk):
        comm.Send(flat[i:i + chunk], dest=dest, tag=72)


def recv_array(comm, source, chunk=1 << 28):
    shape, dt = comm.recv(source=source, tag=71)
    a = np.empty(shape, dtype=np.dtype(dt))
    flat = a.reshape(-1)
    for i in range(0, flat.size, chunk):
        comm.Recv(flat[i:i + chunk], source=source, tag=72)
    return a


def _reduce_rows_to_owner(comm, C, aux_loc):
    '''Partial CSR matrices (naux, ncol) of the ranks -> on every rank the sum of the rows of the aux atoms it
    owns (others empty), and the owner of every aux atom (greedy by its rows' global nonzeros).  Pieces go to
    their owner one sender at a time: the peak is the owned sum plus one piece.'''
    import scipy.sparse
    from mpi4py import MPI
    rank, size = comm.Get_rank(), comm.Get_size()
    C = C.tocsr()
    nat = len(aux_loc)
    w = np.array([float(C.indptr[a1] - C.indptr[a0]) for a0, a1 in aux_loc])
    comm.Allreduce(MPI.IN_PLACE, w, op=MPI.SUM)
    load = np.zeros(size)
    owner = [0] * nat
    for a in np.argsort(-w, kind='stable'):
        k = int(np.argmin(load))
        owner[a] = k
        load[k] += w[a]
    out = None
    for r in range(size):
        rows_r = np.concatenate([np.arange(*aux_loc[a]) for a in range(nat) if owner[a] == r] or
                                [np.zeros(0, dtype=int)])
        piece = C[rows_r]
        if rank == r:
            acc = piece
            for p in range(size):
                if p != r:
                    d, ix, ip = recv_array(comm, p), recv_array(comm, p), recv_array(comm, p)
                    acc = acc + scipy.sparse.csr_matrix((d, ix, ip), shape=piece.shape)
            acc = acc.tocsr()
            acc.sort_indices()
            counts = np.zeros(C.shape[0], dtype=np.int64)
            counts[rows_r] = np.diff(acc.indptr)
            out = scipy.sparse.csr_matrix((acc.data, acc.indices, np.concatenate([[0], np.cumsum(counts)])),
                                          shape=C.shape)
        else:
            send_array(comm, piece.data, r)
            send_array(comm, piece.indices, r)
            send_array(comm, piece.indptr, r)
        piece = None
    return out, owner


def bcast_array(comm, a, root, chunk=1 << 28):
    '''Bcast of a numpy array of any size (chunks below the 2^31-element MPI limit); a is None off the root.'''
    meta = comm.bcast((a.shape, a.dtype.str) if comm.Get_rank() == root else None, root=root)
    if comm.Get_rank() != root:
        a = np.empty(meta[0], dtype=np.dtype(meta[1]))
    flat = a.reshape(-1)
    for i in range(0, flat.size, chunk):
        comm.Bcast(flat[i:i + chunk], root=root)
    return a


def _allsum_csr(comm, C):
    '''Sum over the ranks of CSR matrices of one shape, the result on every rank (one rank's arrays broadcast at a
    time, so the peak is the sum plus one partial matrix).'''
    import scipy.sparse
    shape = C.shape
    out = None
    for r in range(comm.Get_size()):
        mine = comm.Get_rank() == r
        data = bcast_array(comm, C.data if mine else None, r)
        indices = bcast_array(comm, C.indices if mine else None, r)
        indptr = bcast_array(comm, C.indptr if mine else None, r)
        Cr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
        out = Cr if out is None else out + Cr
    return out


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


def far_cx_fit(cell, auxcell, compact, omega, near, pidx, lindep=1e-12, grid=(300, 302), r_thr=36., atoms=None):
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
    pa_of = lambda A: np.arange(*aux_loc[A])
    from pyscf import lib
    for A in range(cell.natm):
        if atoms is not None and A not in atoms:
            continue
        a0, a1 = ao_loc[A]
        cA = np.where(compact[a0:a1])[0] + a0
        if len(cA) == 0:
            continue
        r, dr = radi.mura_knowles(nrad, cell.atom_charge(A))
        sel = r < r_max
        r, wr = r[sel], (dr * r ** 2)[sel]
        pts_all = (r[:, None, None] * xyz_ang[None]).reshape(-1, 3) + X[A]
        w_all = (wr[:, None] * w_ang[None]).ravel()
        molA = _atom_mol(cell, [A], [X[A]], cell._basis)
        nearA = [(_atom_mol(cell, [B], [X[B] + t], cell._basis), B) for B, t in near[A]]
        xcols = np.where(xs_ok)[0]
        amol = _atom_mol(auxcell, [A], [X[A]], auxcell._basis)
        with amol.with_range_coulomb(-omega):
            m2 = amol.intor('int2c2e')
        v, e = _inv_metric(.5 * (m2 + m2.T), lindep)
        Minv = (v / e) @ v.T
        nQ, nx, nc = amol.nao, len(xcols), len(cA)
        # within the memory budget (cell.max_memory): J accumulators (n_c batch x nQ x n_x) and per point chunk
        # c, x_far (all AOs while the explicit images are subtracted), u, u w (~ 8 (2 nQ + nao + n_c) bytes a
        # point, x2 for the evaluation buffers)
        avail = max(0., cell.max_memory - lib.current_memory()[0]) * 1e6
        kb = int(max(1, min(nc, 0.3 * avail / (8 * nQ * nx))))
        npc = int(max(500, min(len(pts_all), 0.3 * avail / (16 * (2 * nQ + cell.nao + nc)))))
        for k0 in range(0, nc, kb):
            k1 = min(k0 + kb, nc)
            J = np.zeros((k1 - k0, nQ, nx))
            for p0 in range(0, len(pts_all), npc):
                p1 = min(p0 + npc, len(pts_all))
                pts, w = pts_all[p0:p1], w_all[p0:p1]
                cv = molA.eval_gto('GTOval_sph', pts)[:, cA[k0:k1] - a0]          # c on A (home cell only)
                xf = cell.pbc_eval_gto('GTOval_sph', pts)                         # x_far: lattice sum minus the
                for mB, B in nearA:                                               # explicit images
                    xf[:, ao_loc[B][0]:ao_loc[B][1]] -= mB.eval_gto('GTOval_sph', pts)
                xf = np.ascontiguousarray(xf[:, xcols])
                uw = _sr_potential_at(amol, pts, omega) * w[:, None]              # SR potentials of A's aux
                for k in range(k1 - k0):
                    J[k] += (uw * cv[:, k:k + 1]).T @ xf                          # (Q | c x_far)_SR
                cv = xf = uw = None
            for k in range(k1 - k0):
                coef = Minv @ J[k]
                big = np.abs(coef) > 1e-14
                qq, xx = np.nonzero(big)
                rows.append(pa_of(A)[qq])
                cols.append(pidx[cA[k0 + k], xcols[xx]])
                vals.append(coef[qq, xx])
            J = None
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
