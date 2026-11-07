#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Cholesky decomposition (CD) of the electron repulsion integrals,

    (mu nu|la si) ~= sum_J L^J_{mu nu} L^J_{la si},

with all residual diagonal elements below ``tau``, so that every integral is
reproduced to ``tau`` (|error of (p|q)| <= sqrt(d_p d_q)).

:class:`CD` is a ``df.DF`` subclass: after build(), ``_cderi`` holds the
vectors in the (naux, nao_pair) lower-triangular layout of density fitting,
so ``loop()``, ``get_jk()`` and every method that takes its integrals from
``mf.with_df`` work unchanged.

Two algorithms (Folkestad, Kjønstad, Koch, J. Chem. Phys. 150, 194112 (2019)):

* ``'onestep'``: pivoted CD, the vectors built along with the pivots.
* ``'twostep'``: the same pivot selection, then the vectors rebuilt from the
  exact integrals with the pivots as auxiliary functions (RI with the pivot
  metric J = K K^T); keeps ``pivots`` and ``metric_chol`` for gradients.

Both use the same pivots: whole shell pairs (all components), taken from the
eigen-directions of the residual block of a shell pair (or of a group of
symmetry-equivalent shell pairs), largest first, with the span factor
``sigma`` (Koch et al., J. Chem. Phys. 118, 9481 (2003)).  The decomposition
is invariant under rotations of the molecule and keeps its point-group
symmetry.  The integral columns of a batch of candidate shell pairs are
computed once and their residual is formed with one matrix product.

No pair is screened out of the vectors: dropping the pairs with small
diagonals keeps every single integral within tau, but the dropped
contributions all lower the Coulomb repulsion, and in a large molecule they
add up (C60/cc-pVDZ: 3.7 Eh too low with D_p * max(D) < tau**2 dropped).
'''

import numpy
import scipy.linalg
from pyscf import df, lib
from pyscf.gto import moleintor
from pyscf.lib import logger


def cholesky(mf, tau=1e-4, sigma=1e-2, method='onestep',
             cderi=None, cderi_to_save=None, with_df=None, **kwargs):
    '''Attach Cholesky-decomposed ERIs to an SCF object -- the CD analogue of
    ``mf.density_fit()``.

    Builds a :class:`CD` object and routes it through the mean field's own
    ``density_fit(with_df=...)``, so it works for spinor, GHF and plain pyscf
    mean fields alike::

        mf = spinor_hf.SCF(mol).x2camf().cholesky()

    The on-disk cache uses the pyscf DF interface (``_cderi_to_save`` /
    ``_cderi``), exposed here as keywords::

        mf = ....cholesky(cderi_to_save='cderi.h5')   # compute and save
        mf = ....cholesky(cderi='cderi.h5')           # reuse

    Extra keyword arguments (e.g. ``only_dfj``) go to ``density_fit``.
    '''
    kwargs.pop('compress', None)   # accepted for backward compatibility
    if with_df is None:
        with_df = CD(mf.mol, tau=tau, sigma=sigma, method=method)
    if cderi is not None:
        with_df._cderi = cderi
    if cderi_to_save is not None:
        with_df._cderi_to_save = cderi_to_save
    return mf.density_fit(with_df=with_df, **kwargs)


def get_cderi_ao(mf):
    '''AO Cholesky (or density-fitting) factors of ``mf.with_df``, unpacked to
    shape (naux, nao, nao).'''
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError('%s has no with_df; attach the factorised integrals with '
                         'mf.cholesky() or mf.density_fit() first' % mf.__class__.__name__)
    return numpy.vstack([lib.unpack_tril(blk) for blk in with_df.loop()])


class _Int2e:
    '''Integrals (pairs | shell pair) and pair diagonals, sharing one cintopt.'''
    def __init__(self, mol):
        self.mol = mol
        self.intor = mol._add_suffix('int2e')
        self.cintopt = moleintor.make_cintopt(mol._atm, mol._bas, mol._env, self.intor)
        self.ao_loc = mol.ao_loc_nr()

    def __call__(self, shls_slice, aosym='s1'):
        mol = self.mol
        return moleintor.getints(self.intor, mol._atm, mol._bas, mol._env,
                                 shls_slice, aosym=aosym, ao_loc=self.ao_loc,
                                 cintopt=self.cintopt)

    def column(self, si, sj):
        '''(mu nu | i j) for all mu >= nu and i in shell si, j in shell sj:
        array (nao_pair, di*dj).'''
        nbas = self.mol.nbas
        buf = self((0, nbas, 0, nbas, si, si+1, sj, sj+1), aosym='s2ij')
        return buf.reshape(buf.shape[0], -1)

    def diagonal(self):
        '''(mu nu|mu nu) for mu >= nu, in the pack_tril order.'''
        ao_loc = self.ao_loc
        nao = int(ao_loc[-1])
        diag = numpy.empty(nao*(nao+1)//2)
        for si in range(self.mol.nbas):
            i0, i1 = ao_loc[si], ao_loc[si+1]
            for sj in range(si+1):
                j0, j1 = ao_loc[sj], ao_loc[sj+1]
                buf = self((si, si+1, sj, sj+1, si, si+1, sj, sj+1))
                d = numpy.einsum('ijij->ij', buf.reshape(i1-i0, j1-j0, i1-i0, j1-j0))
                for i in range(i0, i1):
                    jend = min(j1, i+1)
                    p0 = i*(i+1)//2
                    diag[p0+j0:p0+jend] = d[i-i0, :jend-j0]
        return diag


def _shell_pair_columns(ao_loc, si, sj):
    '''pack_tril pair indices of the block (si, sj), si >= sj, and their
    positions in the (di*dj) columns of _Int2e.column.'''
    i0, i1 = ao_loc[si], ao_loc[si+1]
    j0, j1 = ao_loc[sj], ao_loc[sj+1]
    i, j = numpy.meshgrid(numpy.arange(i0, i1), numpy.arange(j0, j1), indexing='ij')
    mask = i >= j
    pairs = (i*(i+1)//2 + j)[mask]
    local = ((i-i0)*(j1-j0) + (j-j0))[mask]
    return pairs, local


# candidate columns of one macro iteration (bounded by the memory as well)
MAX_CANDIDATES = 128
# vectors made between two refreshes of the candidate residual block
REFRESH = 32
# relative difference of the block measure regarded as a tie (equivalent pairs)
TIE = 1e-8
# active set (two-step): compact once the done pairs exceed this fraction
ACTIVE_COMPACT = .2


def _pair_weights(ao_loc, blocks, bpairs, npair):
    """sqrt(2) for chi_mu chi_nu (mu > nu) in a block of a shell with itself:
    the orthonormal basis of the symmetric products.  With these weights a
    rotation acts on every block by an orthogonal transformation."""
    cw = numpy.ones(npair)
    for (si, sj), p in zip(blocks, bpairs):
        if si == sj:
            i = ((numpy.sqrt(8*p+1) - 1) // 2).astype(int)
            cw[p[i*(i+1)//2 + i != p]] = numpy.sqrt(2.)
    return cw


def pivoted_cholesky(mol, tau=1e-4, sigma=1e-2, max_memory=4000, verbose=None,
                     active_set=False):
    """Shell-pair block pivoted Cholesky decomposition of the ERIs.

    The pivots are whole shell pairs.  For a group of shell pairs (one pair,
    or the symmetry-equivalent pairs with the same measure) with residual
    block S (in the orthonormal basis of the pair functions, see
    _pair_weights) the eigen-directions with eigenvalue >= cut give the new
    vectors,

        L_new = Lambda^-1/2 U^T R_(group, all pairs),

    and the group is done once all its eigenvalues are below tau.  The blocks
    are taken largest mean residual (trace / size) first; cut = max(tau,
    sigma * largest measure).  Eigenvalues, traces and the groups are invariant
    under rotations and permutations of equivalent atoms, so is the
    decomposition.

    With active_set (for the two-step method, which needs the pivots only)
    the vectors are kept on the pairs of the groups that are not done: a done
    group never becomes a candidate again, and the residual of its pairs is
    not needed any more.  The arrays over the pairs are compacted once the
    done pairs exceed ACTIVE_COMPACT of them.  The pivots are the same.

    Returns:
        L : (nchol, nao_pair) array, the vectors (None with active_set)
        pivots : list of (shell pairs, U), U (components x k) the directions
    """
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    ints = _Int2e(mol)
    ao_loc = ints.ao_loc
    nbas = mol.nbas
    nao = int(ao_loc[-1])
    npair = nao*(nao+1)//2

    blocks = [(si, sj) for si in range(nbas) for sj in range(si+1)]
    bpairs, blocal = [], []
    bid = numpy.empty(npair, dtype=int)
    for ib, (si, sj) in enumerate(blocks):
        p, loc = _shell_pair_columns(ao_loc, si, sj)
        bpairs.append(p)
        blocal.append(loc)
        bid[p] = ib
    bsize = numpy.array([p.size for p in bpairs])
    nblk = len(blocks)
    cw = _pair_weights(ao_loc, blocks, bpairs, npair)
    c2 = cw * cw

    d = ints.diagonal()
    done = numpy.zeros(nblk, dtype=bool)
    t1 = log.timer_debug1('CD diagonal', *t0)

    # active pairs: the arrays over pairs (d, the columns of L and Bt) are
    # indexed by position in act; pos[p] is the position of pair p
    act = numpy.arange(npair)
    pos = numpy.arange(npair)
    bid_a, c2_a, cw_a = bid, c2, cw
    nact = npair

    def bp(ib):
        return pos[bpairs[ib]]

    def traces():
        t = numpy.zeros(nblk)
        numpy.add.at(t, bid_a, c2_a * d)
        t[done] = 0
        return t

    def avail_mb():
        return max_memory - lib.current_memory()[0]

    def grow(L, n, k):
        cap = L.shape[0] + max(nao, k, L.shape[0] // 2)
        need = (cap + L.shape[0]) * nact * 8 / 1e6       # new buffer + copy
        if need > avail_mb():
            raise MemoryError(
                'CD needs %.0f MB for %d+ vectors (tau = %g, max residual %.2e) '
                'but only %.0f MB of max_memory are left; raise max_memory or tau.'
                % (need, n + k, tau, d.max(), avail_mb()))
        L1 = numpy.empty((cap, nact))
        L1[:n] = L[:n]
        return L1

    # numpy.empty reserves the buffer; pages are allocated when first written
    # (by the gemm that makes the vectors), so a generous size costs nothing
    cap0 = max(64, min(npair, int(avail_mb() * 1e6 / 8 / npair / 2)))
    L = numpy.empty((cap0, npair))
    n = 0
    pivots = []
    nmacro = 0
    lam_seen = None
    while True:
        # a block with residual trace below tau has every eigenvalue below tau:
        # done (the trace only decreases)
        done |= traces() < tau
        if active_set:
            alive = ~done[bid_a]
            if (~alive).sum() > ACTIVE_COMPACT * nact:
                keep_pos = numpy.where(alive)[0]
                L1 = numpy.empty((L.shape[0], keep_pos.size))
                lib.take_2d(L[:n], numpy.arange(n), keep_pos, out=L1[:n])   # threaded
                L = L1
                del L1
                act = act[alive]
                d = d[alive]
                bid_a, c2_a, cw_a = bid[act], c2[act], cw[act]
                pos = numpy.full(npair, -1)
                pos[act] = numpy.arange(act.size)
                log.debug1('CD active set: %d -> %d pairs', nact, act.size)
                nact = act.size
        t = traces()
        tmax = t.max()
        if tmax < tau:
            break
        # the trace bounds lambda_max from above; after an iteration that took
        # nothing the cut is lowered to the largest eigenvalue seen
        cut = max(tau, sigma * (tmax if lam_seen is None else lam_seen))
        lam_seen = None
        nmacro += 1
        cand = numpy.where(t >= cut)[0]
        mean = t / numpy.maximum(bsize, 1)
        cand = cand[numpy.argsort(-mean[cand])]
        # whole groups of equal measure, never split between iterations
        kmax = max(1, min(MAX_CANDIDATES, int(avail_mb() * 1e6 / 8 / nact / 4)))
        sel = []
        ncol = 0
        i = 0
        while i < cand.size:
            j = i + 1
            while j < cand.size and mean[cand[i]] - mean[cand[j]] <= TIE * mean[cand[i]]:
                j += 1
            k = bsize[cand[i:j]].sum()
            if ncol and ncol + k > kmax:
                break
            sel.extend(cand[i:j])
            ncol += k
            i = j

        # residual of the candidates, stored transposed: Bt[c, :] = R[:, c]
        Bt = numpy.empty((ncol, nact))
        off = {}
        c0 = 0
        for ib in sel:
            si, sj = blocks[ib]
            k = bsize[ib]
            # (all pairs | shell pair): libcint parallelizes over the pairs;
            # transposed into the rows by the threaded lib.transpose
            col = ints.column(si, sj)
            if nact < npair or si == sj:
                rows_ = act if nact < npair else numpy.arange(npair)
                cols_ = blocal[ib] if si == sj else numpy.arange(col.shape[1])
                col = lib.take_2d(col, rows_, cols_)      # threaded, contiguous
            rows = lib.transpose(col, out=Bt[c0:c0+k])
            if not numpy.shares_memory(rows, Bt):
                Bt[c0:c0+k] = rows
            off[ib] = (c0, c0 + k)
            c0 += k
        if n:
            cp_all = numpy.hstack([bp(ib) for ib in sel])
            lib.dot(L[:n, cp_all].T, L[:n], -1, Bt, 1)

        m0 = n
        n_start = n
        lam_max_seen = 0.
        live = list(sel)
        while live:
            tl = numpy.array([(c2_a[bp(ib)] * d[bp(ib)]).sum() for ib in live])
            ml = tl / bsize[live]
            j = int(numpy.argmax(ml))
            if tl[j] < cut:
                break
            grp = [live[i] for i in numpy.where(abs(ml - ml[j]) <= TIE * ml[j])[0]]
            for ib in grp:
                live.remove(ib)
            p = numpy.hstack([bp(ib) for ib in grp])
            if len(grp) == 1:
                Bb = Bt[off[grp[0]][0]:off[grp[0]][1]]
            else:
                Bb = Bt[numpy.hstack([numpy.arange(*off[ib]) for ib in grp])]
            if n > m0:
                Bb = lib.dot(L[m0:n, p].T, L[m0:n], -1, Bb.copy(), 1)
            cp = cw_a[p]
            S = Bb[:, p] * cp[:, None] * cp
            S = (S + S.T) * .5
            w, W = numpy.linalg.eigh(S)
            keep = w >= cut
            rest = w[~keep]
            if rest.size == 0 or rest.max() < tau:
                done[grp] = True
            elif not keep.any():
                lam_max_seen = max(lam_max_seen, rest.max())
            if not keep.any():
                continue
            U = W[:, keep] * cp[:, None]
            k = U.shape[1]
            if n + k > L.shape[0]:
                L = grow(L, n, k)
            Lnew = L[n:n+k]
            lib.dot(numpy.ascontiguousarray((U / numpy.sqrt(w[keep])).T), Bb, 1, Lnew, 0)
            n += k
            pivots.append(([blocks[ib] for ib in grp], U))
            d -= numpy.einsum('kp,kp->p', Lnew, Lnew)
            numpy.maximum(d, 0, out=d)
            if n - m0 >= REFRESH and live:
                idx = numpy.hstack([numpy.arange(*off[b]) for b in live])
                newoff = {}
                c0 = 0
                for b in live:
                    newoff[b] = (c0, c0 + bsize[b])
                    c0 += bsize[b]
                Bt = Bt[idx]
                off = newoff
                lib.dot(L[m0:n, numpy.hstack([bp(b) for b in live])].T, L[m0:n], -1, Bt, 1)
                m0 = n
        if n == n_start:
            lam_seen = lam_max_seen
        log.debug1('CD macro iteration %d: %d candidate columns, nchol = %d',
                   nmacro, ncol, n)

    log.info('CD: %d vectors (%.1f nao) from %d pivot groups in %d macro iterations, tau = %g',
             n, n / max(nao, 1), len(pivots), nmacro, tau)
    log.timer('CD pivoted decomposition', *t1)
    if active_set:
        return None, pivots
    return L[:n], pivots


class CD(df.DF):
    '''Cholesky decomposition of the ERIs as a DF object.

    Attributes:
        tau : float
            Decomposition threshold: all residual diagonals below tau.
        sigma : float
            Span factor of the in-block pivot selection.
        method : str
            'onestep' (default) or 'twostep' ('threeloop' is accepted as an
            old name of 'onestep').
        pivots : list of (shell pairs, U) or None
            After build(): the pivot groups and their directions U
            (components x k) in the pair functions of the group.
        metric_chol : ndarray or None
            'twostep': Cholesky factor K of the pivot metric J = K K^T.
    '''

    _keys = {'tau', 'sigma', 'method', 'pivots', 'metric_chol'}

    def __init__(self, mol, tau=1e-4, sigma=1e-2, method='onestep', **kwargs):
        df.DF.__init__(self, mol)
        kwargs.pop('compress', None)   # accepted for backward compatibility
        if kwargs:
            raise TypeError('unexpected arguments %s' % list(kwargs))
        self.tau = tau
        self.sigma = sigma
        self.method = method
        self.pivots = None
        self.metric_chol = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('Cholesky decomposition: method = %s, tau = %g, sigma = %g',
                 self.method, self.tau, self.sigma)
        log.info('max_memory = %s', self.max_memory)
        return self

    @staticmethod
    def cderi_shape(cderi, dataname='j3c'):
        '''(naux, nao_pair) of an on-disk cderi, for either the single-dataset
        or the blocked-group layout.'''
        import h5py
        with h5py.File(cderi, 'r') as f:
            obj = f[dataname]
            if isinstance(obj, h5py.Group):
                blocks = [obj[k] for k in sorted(obj.keys(), key=int)]
                return blocks[0].shape[0], sum(b.shape[1] for b in blocks)
            return tuple(obj.shape)

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        # build first: with _cderi unset, the J-only path of df_jk would make
        # its own even-tempered auxiliary basis instead of using the CD vectors
        if self._cderi is None:
            self.build()
        return df.DF.get_jk(self, dm, hermi, with_j, with_k,
                            direct_scf_tol, omega)

    def build(self):
        log = logger.new_logger(self)
        if self._cderi is not None:
            log.info('CD: using preset _cderi (%s); skipping the decomposition',
                     self._cderi if isinstance(self._cderi, str) else 'in-core array')
            return self
        method = {'threeloop': 'onestep'}.get(self.method, self.method)
        if method not in ('onestep', 'twostep'):
            raise ValueError("CD.method must be 'onestep' or 'twostep', got %r" % self.method)
        self.dump_flags()

        mol = self.mol
        L, pivots = pivoted_cholesky(mol, self.tau, self.sigma, self.max_memory, self.verbose,
                                     active_set=(method == 'twostep'))
        self.pivots = pivots
        if method == 'onestep':
            cderi = L
        else:
            cderi = self._rebuild(mol, pivots, log)

        if isinstance(self._cderi_to_save, str):
            import h5py
            dataname = getattr(self, '_dataname', 'j3c')
            with h5py.File(self._cderi_to_save, 'w') as f:
                f[dataname] = cderi
            self._cderi = self._cderi_to_save
            log.info('CD: saved %d vectors to %s', cderi.shape[0], self._cderi_to_save)
        else:
            self._cderi = cderi
        return self

    def _rebuild(self, mol, pivots, log):
        """Two-step CD, step 2: L = K^-1 T with T = U^T (pivot groups | pairs)
        and J = T restricted to the pivot directions = K K^T."""
        t0 = (logger.process_clock(), logger.perf_counter())
        ints = _Int2e(mol)
        ao_loc = ints.ao_loc
        nao = int(ao_loc[-1])
        npair = nao*(nao+1)//2
        npiv = sum(U.shape[1] for _, U in pivots)
        need = 2 * npiv * npair * 8 / 1e6
        avail = self.max_memory - lib.current_memory()[0]
        if need > avail:
            raise MemoryError('two-step CD needs %.0f MB for %d pivots, %.0f MB left'
                              % (need, npiv, avail))
        T = numpy.empty((npiv, npair))
        gpairs = []
        r0 = 0
        for grp, U in pivots:
            # integrals of the group, transposed (components x pairs) by the
            # threaded lib.transpose / lib.take_2d, then T rows = U^T Ht by gemm
            ps = []
            locs = []
            for si, sj in grp:
                p, loc = _shell_pair_columns(ao_loc, si, sj)
                ps.append(p)
                locs.append(loc)
            Ht = numpy.empty((sum(p.size for p in ps), npair))
            c0 = 0
            for (si, sj), p, loc in zip(grp, ps, locs):
                col = ints.column(si, sj)
                if si == sj:
                    col = lib.take_2d(col, numpy.arange(npair), loc)
                rows = lib.transpose(col, out=Ht[c0:c0+p.size])
                if not numpy.shares_memory(rows, Ht):
                    Ht[c0:c0+p.size] = rows
                c0 += p.size
            k = U.shape[1]
            lib.dot(numpy.ascontiguousarray(U.T), Ht, 1, T[r0:r0+k], 0)
            gpairs.append(numpy.hstack(ps))
            r0 += k
        J = numpy.empty((npiv, npiv))
        r0 = 0
        for (grp, U), p in zip(pivots, gpairs):
            k = U.shape[1]
            J[:, r0:r0+k] = T[:, p] @ U
            r0 += k
        J = (J + J.T) * .5
        try:
            K = scipy.linalg.cholesky(J, lower=True)
        except numpy.linalg.LinAlgError:
            raise RuntimeError('two-step CD: the pivot metric is not positive definite')
        self.metric_chol = K
        cderi = scipy.linalg.solve_triangular(K, T, lower=True, overwrite_b=True)
        log.timer('CD two-step reconstruction', *t0)
        return cderi
