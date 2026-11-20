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

Two-step algorithm (Folkestad, Kjønstad, Koch, J. Chem. Phys. 150, 194112
(2019)): the pivots are selected first, then the vectors are built from the
exact integrals with the pivots as auxiliary functions (RI with the pivot
metric J = K K^T); ``pivots`` and ``metric_chol`` are kept.

The pivots are whole shell pairs (all components), taken from the
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

Usage, as any density-fitting object, or with the shortcut cholesky()::

    mf = scf.RHF(mol).density_fit(with_df=CD(mol, tau=1e-5))
    mf = spinor_hf.SCF(mol).x2camf().cholesky(tau=1e-5)
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.lib import logger
from pyscf.df.df import DF
from pyscf import __config__


def cholesky(mf, tau=1e-4, sigma=1e-2, method=None,
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
    ``method`` is accepted for old scripts: only the two-step algorithm is
    left ('twostep' or None).
    '''
    kwargs.pop('compress', None)   # accepted for backward compatibility
    if method not in (None, 'twostep'):
        raise ValueError("CD method %r: only the two-step algorithm is available "
                         "(the one-step one gave the same vectors, slower)" % method)
    if with_df is None:
        with_df = CD(mf.mol, tau=tau, sigma=sigma)
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
# active set: compact once the done pairs exceed this fraction
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


def _compact_columns(buf, n, ncol, keep, blksize=None):
    '''In place: the rows of buf viewed as (cap, ncol) keep only the columns
    keep; returns buf viewed as (cap', len(keep)).  Row r moves to the front
    (its new start r*len(keep) is not after its old start r*ncol), in blocks
    of rows read into a temporary before they are written.'''
    nkeep = keep.size
    if nkeep == 0:
        return buf[:0].reshape(0, 0)
    old = buf[:n*ncol].reshape(n, ncol)
    if blksize is None:
        blksize = max(1, min(n, int(64e6 / 8 / max(nkeep, 1))))
    rows = numpy.arange(blksize)
    for r0 in range(0, n, blksize):
        r1 = min(n, r0 + blksize)
        tmp = lib.take_2d(old[r0:r1], rows[:r1-r0], keep)     # threaded copy
        buf[r0*nkeep:r1*nkeep] = tmp.ravel()
    cap = buf.size // nkeep
    return buf[:cap*nkeep].reshape(cap, nkeep)


def estimate_nvec(nao, tau):
    '''Number of CD vectors, an estimate on the high side: the vectors per AO
    grow by about 1.5 per decade of tau (3.4-4.3 at 1e-3, 8.4-8.9 at 1e-6, 16
    at 1e-10 for TZ basis sets).'''
    return int(min(nao*(nao+1)//2, (1.5 * -numpy.log10(tau) + 1) * nao))


def select_pivots(mol, tau=1e-4, sigma=1e-2, max_memory=4000, verbose=None,
                  force=False):
    """Pivots of the shell-pair block pivoted Cholesky decomposition of the
    ERIs.

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

    The vectors made on the way are kept only on the pairs of the groups
    that are not done (the active set): a done group never becomes a
    candidate again, and the residual of its pairs is not needed any more.
    The arrays over the pairs are compacted once the done pairs exceed
    ACTIVE_COMPACT of them.

    Before the decomposition the memory of the vectors on the active pairs
    is estimated (estimate_nvec, on the high side); MemoryError if it exceeds
    max_memory, unless force.  The memory is checked again whenever vectors
    are added.

    Returns:
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

    def out_of_memory(n, k):
        return MemoryError(
            'CD needs memory for %d+ vectors (tau = %g, max residual %.2e) but '
            'only %.0f MB of max_memory are left; raise max_memory or tau.'
            % (n + k, tau, d.max(), avail_mb()))

    # blocks with residual trace below tau are done from the start (pairs of
    # distant atoms): the active pairs bound the memory of the vectors
    done |= traces() < tau
    nact0 = int((~done[bid]).sum())
    est = estimate_nvec(nao, tau) * nact0 * 8 / 1e6
    log.info('CD: %d of %d pairs active; estimated memory of the vectors %.0f MB, '
             '%.0f MB of max_memory left', nact0, npair, est, avail_mb())
    if est > avail_mb() and not force:
        raise MemoryError(
            'CD: the vectors need an estimated %.0f MB (tau = %g, %d active pairs; the '
            'estimate is on the high side), %.0f MB of max_memory are left.  Raise '
            'max_memory or tau, or set force = True to run anyway.'
            % (est, tau, nact0, avail_mb()))

    # One buffer for the vectors, never reallocated: numpy.empty reserves the
    # address space, the pages are allocated when first written (by the gemm
    # that makes the vectors), so the largest size max_memory allows costs
    # nothing up front.  The vectors live on the active pairs, L[:n, :nact],
    # rows of length nact in the flat buffer.
    cap = int(min(npair, max(1, avail_mb()) * 1e6 / 8 / npair))
    buf = numpy.empty(cap * npair)
    L = buf.reshape(cap, npair)
    n = 0
    pivots = []
    nmacro = 0
    lam_seen = None
    while True:
        # a block with residual trace below tau has every eigenvalue below tau:
        # done (the trace only decreases)
        done |= traces() < tau
        alive = ~done[bid_a]
        if (~alive).sum() > ACTIVE_COMPACT * nact:
            keep_pos = numpy.where(alive)[0]
            L = _compact_columns(buf, n, nact, keep_pos)
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
            if n + k > L.shape[0] or k * nact * 8 / 1e6 > avail_mb():
                raise out_of_memory(n, k)
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

    log.info('CD: %d pivots (%.1f nao) in %d groups from %d macro iterations, tau = %g',
             n, n / max(nao, 1), len(pivots), nmacro, tau)
    log.timer('CD pivot selection', *t1)
    return pivots


_NO_AUXMOL = ('CD has no atom-centred auxiliary basis (its auxiliary functions are '
              'the pivot products); use df.DF for methods that need auxmol or auxbasis')


class CD(DF):
    '''Cholesky decomposition of the ERIs as a DF object.

    Attributes:
        tau : float
            Decomposition threshold: all residual diagonals below tau.
        sigma : float
            Span factor of the in-block pivot selection.
        pivots : list of (shell pairs, U) or None
            After build(): the pivot groups and their directions U
            (components x k) in the pair functions of the group.
        metric_chol : ndarray or None
            After build(): Cholesky factor K of the pivot metric J = K K^T.
        force : bool
            The pivot search first estimates the memory of its vectors (on
            the high side, see select_pivots) and raises MemoryError if it
            exceeds max_memory.  force = True skips this check; the memory is
            still checked while the vectors are made.  The final vectors are
            written to _cderi_to_save in blocks (as df.DF) when they do not
            fit in memory or _cderi_to_save is set.
    '''

    _keys = {'tau', 'sigma', 'pivots', 'metric_chol', 'force'}

    def __init__(self, mol, tau=1e-4, sigma=1e-2):
        DF.__init__(self, mol)
        self.tau = tau
        self.sigma = sigma
        self.force = False
        self.pivots = None
        self.metric_chol = None

    # CD has no atom-centred auxiliary basis (its auxiliary functions are the
    # pivot products, see pivots and metric_chol).  Methods that need auxmol
    # or set an auxbasis would otherwise build a DF object of their own.
    @property
    def auxmol(self):
        raise NotImplementedError(_NO_AUXMOL)

    @auxmol.setter
    def auxmol(self, x):
        if x is not None:
            raise NotImplementedError(_NO_AUXMOL)

    @property
    def auxbasis(self):
        raise NotImplementedError(_NO_AUXMOL)

    @auxbasis.setter
    def auxbasis(self, x):
        if x is not None:
            raise NotImplementedError(_NO_AUXMOL)

    def range_coulomb(self, omega):
        raise NotImplementedError('range-separated integrals with CD')

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('Cholesky decomposition: tau = %g, sigma = %g', self.tau, self.sigma)
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
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is not None:
            raise NotImplementedError('range-separated integrals with CD')
        # build first: with _cderi unset, the J-only path of df_jk would make
        # its own even-tempered auxiliary basis instead of using the CD vectors
        if self._cderi is None:
            self.build()
        return DF.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)

    def build(self):
        log = logger.new_logger(self)
        if self._cderi is not None:
            log.info('CD: using preset _cderi (%s); skipping the decomposition',
                     self._cderi if isinstance(self._cderi, str) else 'in-core array')
            return self
        self.dump_flags()

        mol = self.mol
        pivots = select_pivots(mol, self.tau, self.sigma, self.max_memory, self.verbose,
                               self.force)
        self.pivots = pivots

        # step 2 as df.DF.build: the pivots are the auxiliary functions,
        # T = U^T (pivot | pairs) the 3-index integrals, J = U^T (pivot |
        # pivot) U the metric.  In memory if the vectors fit and no file is
        # requested, else in blocks of pairs in _cderi_to_save.
        nao = mol.nao_nr()
        npiv = sum(U.shape[1] for _, U in pivots)
        is_custom_storage = isinstance(self._cderi_to_save, str)
        max_memory = self.max_memory - lib.current_memory()[0]
        if npiv * nao*(nao+1)//2 * 8 / 1e6 < .9 * max_memory and not is_custom_storage:
            self._cderi = self._rebuild(mol, pivots, log)
        else:
            if self._cderi_to_save is None:
                self._cderi_to_save = lib.NamedTemporaryFile(dir=lib.param.TMPDIR)
            cderi = self._cderi_to_save
            log.info('_cderi_to_save = %s', getattr(cderi, 'name', cderi))
            self._rebuild_outcore(mol, pivots, cderi, max_memory, log)
            self._cderi = cderi
        return self

    def _t_rows(self, ints, pivots, T, shell_range=None):
        """T = U^T (pivot groups | pairs): rows for all pairs (shell_range
        None) or for the pairs of the bra shells a0..a1 (all ket shells up to
        a1).  The gemm takes the integral columns (pairs x components, as
        libcint makes them) transposed, without copies.  For a shell with
        itself the rows of U are spread over all di*dj components (zero for
        mu < nu)."""
        ao_loc = ints.ao_loc
        if shell_range is not None:
            a0, a1 = shell_range
            i0, i1 = ao_loc[a0], ao_loc[a1]
            # rows (i, j <= i) of the (i in block, j < i1) integrals, in the
            # pack_tril order p = i*(i+1)/2 + j
            rows = numpy.hstack([(i - i0) * i1 + numpy.arange(i + 1) for i in range(i0, i1)])
        r0 = 0
        for grp, U in pivots:
            k = U.shape[1]
            u0 = 0
            for ig, (si, sj) in enumerate(grp):
                p, loc = _shell_pair_columns(ao_loc, si, sj)
                if shell_range is None:
                    col = ints.column(si, sj)
                else:
                    buf = ints((a0, a1, 0, a1, si, si+1, sj, sj+1))
                    col = buf.reshape(-1, buf.shape[2] * buf.shape[3])[rows]
                Ub = U[u0:u0+p.size]
                if si == sj:
                    Uf = numpy.zeros((col.shape[1], k))
                    Uf[loc] = Ub
                    Ub = Uf
                lib.dot(numpy.ascontiguousarray(Ub.T), col.T, 1, T[r0:r0+k], 1 if ig else 0)
                u0 += p.size
            r0 += k
        return T

    @staticmethod
    def _solve(K, T):
        """L = K^-1 T in place: T (C order) is T^T in Fortran order, so solve
        X K^T = T^T with dtrsm (scipy.linalg.solve_triangular would first copy
        T into Fortran order)."""
        from scipy.linalg import blas
        L = blas.dtrsm(1., K, T.T, side=1, lower=1, trans_a=1, overwrite_b=1).T
        if not numpy.shares_memory(L, T):
            raise RuntimeError('dtrsm did not work in place')
        return L

    def _metric_chol(self, J):
        """Cholesky factor K of the pivot metric, J = K K^T."""
        J = (J + J.T) * .5
        try:
            K = scipy.linalg.cholesky(J, lower=True)
        except numpy.linalg.LinAlgError:
            raise RuntimeError('CD: the pivot metric is not positive definite')
        self.metric_chol = K
        return K

    def _rebuild(self, mol, pivots, log):
        """Step 2 in memory (as df.incore.cholesky_eri): T = U^T (pivot |
        pairs), the metric J from the pivot rows of T, L = K^-1 T in place."""
        t0 = (logger.process_clock(), logger.perf_counter())
        ints = _Int2e(mol)
        ao_loc = ints.ao_loc
        nao = int(ao_loc[-1])
        npiv = sum(U.shape[1] for _, U in pivots)
        T = self._t_rows(ints, pivots, numpy.empty((npiv, nao*(nao+1)//2)))
        J = numpy.empty((npiv, npiv))
        r0 = 0
        for grp, U in pivots:
            k = U.shape[1]
            p = numpy.hstack([_shell_pair_columns(ao_loc, si, sj)[0] for si, sj in grp])
            J[:, r0:r0+k] = T[:, p] @ U
            r0 += k
        cderi = self._solve(self._metric_chol(J), T)
        log.timer('CD vectors', *t0)
        return cderi

    def _rebuild_outcore(self, mol, pivots, cderi, max_memory, log, buflen=None):
        """Step 2 in blocks of pairs (as df.outcore.cholesky_eri_b): the bra
        shells are split into ranges of at most buflen pairs (default from
        max_memory, as df.outcore); the metric J is collected from the pivot
        rows of T over the ranges that hold pivot pairs, then for every range
        L = K^-1 T is written to dataname/istep."""
        from pyscf.df.outcore import _guess_shell_ranges, _create_h5file
        t0 = (logger.process_clock(), logger.perf_counter())
        ints = _Int2e(mol)
        ao_loc = ints.ao_loc
        nao = int(ao_loc[-1])
        npair = nao*(nao+1)//2
        npiv = sum(U.shape[1] for _, U in pivots)
        if buflen is None:
            buflen = min(max(int(max_memory*.24e6/8/npiv), 1), npair)
        shranges = _guess_shell_ranges(mol, buflen, 's2ij')
        log.debug('cderi %.8g MB, IO buf size %.8g MB', npiv*npair*8/1e6, buflen*npiv*8/1e6)
        npair_at = ao_loc * (ao_loc + 1) // 2           # first pair of shell a
        gpairs = [numpy.hstack([_shell_pair_columns(ao_loc, si, sj)[0] for si, sj in grp])
                  for grp, U in pivots]
        allp = numpy.hstack(gpairs)

        # the metric: pivot rows of T, over the ranges that hold pivot pairs
        J = numpy.zeros((npiv, npiv))
        for a0, a1, nrow in shranges:
            p0, p1 = npair_at[a0], npair_at[a1]
            if not numpy.any((allp >= p0) & (allp < p1)):
                continue
            Tb = self._t_rows(ints, pivots, numpy.empty((npiv, p1 - p0)), (a0, a1))
            r0 = 0
            for (grp, U), p in zip(pivots, gpairs):
                k = U.shape[1]
                m = (p >= p0) & (p < p1)
                if m.any():
                    J[:, r0:r0+k] += Tb[:, p[m] - p0] @ U[m]
                r0 += k
            Tb = None
        K = self._metric_chol(J)
        t1 = log.timer('CD pivot metric', *t0)

        feri = _create_h5file(cderi, self._dataname)
        for istep, (a0, a1, nrow) in enumerate(shranges):
            Tb = self._t_rows(ints, pivots, numpy.empty((npiv, npair_at[a1] - npair_at[a0])),
                              (a0, a1))
            feri['%s/%d' % (self._dataname, istep)] = self._solve(K, Tb)
            Tb = None
            t1 = log.timer_debug1('CD vectors [%d/%d], AO shells [%d:%d]'
                                  % (istep+1, len(shranges), a0, a1), *t1)
        feri.close()
        log.timer('CD vectors (%d blocks of pairs on disk)' % len(shranges), *t0)
