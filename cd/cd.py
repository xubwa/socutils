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

Both work on shell-pair blocks: the integral column block (all pairs | shell
pair) is computed once, its residual with respect to all previous vectors is
formed with one matrix product, and the pivots inside the block are taken
from it with a small in-block Cholesky (span factor ``sigma``, Koch et al.,
J. Chem. Phys. 118, 9481 (2003)).

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
        nao = ao_loc[-1]
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


def pivoted_cholesky(mol, tau=1e-4, sigma=1e-2, max_memory=4000, verbose=None):
    '''Shell-pair blocked pivoted Cholesky decomposition of the ERIs.

    Returns:
        L : (nchol, nao_pair) array, the vectors
        kept : (nao_pair,) pack_tril indices of the pairs (all of them)
        pivots : (nchol,) pack_tril indices of the pivots, in order
    '''
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    ints = _Int2e(mol)
    ao_loc = ints.ao_loc
    nao = ao_loc[-1]
    ao2shl = numpy.repeat(numpy.arange(mol.nbas), numpy.diff(ao_loc))

    diag = ints.diagonal()
    # all pairs are kept (see the module docstring)
    kept = numpy.arange(diag.size)
    nk = kept.size
    pos = numpy.full(diag.size, -1)
    pos[kept] = numpy.arange(nk)
    d = diag[kept].copy()
    t1 = log.timer_debug1('CD diagonal', *t0)

    def avail_mb():
        return max_memory - lib.current_memory()[0]

    def grow(L, n):
        cap = L.shape[0] + max(nao, L.shape[0] // 2)
        need = (cap + L.shape[0]) * nk * 8 / 1e6      # new buffer + copy
        if need > avail_mb():
            raise MemoryError(
                'CD needs %.0f MB for %d+ vectors (tau = %g, max residual %.2e) '
                'but only %.0f MB of max_memory are left; raise max_memory or tau.'
                % (need, n, tau, d.max(), avail_mb()))
        L1 = numpy.empty((cap, nk))
        L1[:n] = L[:n]
        return L1

    # shell pair of every kept pair
    i_of = ((numpy.sqrt(8*kept+1) - 1) // 2).astype(int)
    j_of = kept - i_of*(i_of+1)//2
    nbas = mol.nbas
    sp_of = ao2shl[i_of] * nbas + ao2shl[j_of]

    # vector buffer: room for ~6 nao vectors if memory allows, never above
    # half of what is left (the rest is for the candidate block and growth)
    cap0 = max(1, min(6 * nao, int(avail_mb() * 1e6 / 8 / nk / 2)))
    L = numpy.empty((cap0, nk))
    n = 0
    pivots = []
    nmacro = 0
    NB = 32          # in-block vectors between two refreshes of the residual block
    while True:
        dmax = d.max()
        if dmax < tau:
            break
        cut = max(tau, sigma * dmax)
        nmacro += 1

        # all shell pairs with a residual above cut, largest first; the number
        # of candidate columns is capped by the memory of the residual block
        spmax = numpy.zeros(nbas * nbas)
        numpy.maximum.at(spmax, sp_of, d)
        sps = numpy.where(spmax >= cut)[0]
        sps = sps[numpy.argsort(-spmax[sps])]
        kmax = max(1, min(MAX_CANDIDATES, int(avail_mb() * 1e6 / 8 / nk / 4)))
        cols = []
        cpos = []
        ncol = 0
        for sp in sps:
            si, sj = divmod(int(sp), nbas)
            pairs, local = _shell_pair_columns(ao_loc, si, sj)
            cp = pos[pairs]
            sel = cp >= 0
            local, cp = local[sel], cp[sel]
            sel = d[cp] >= cut
            if not sel.any():
                continue
            if ncol and ncol + sel.sum() > kmax:
                break
            cols.append(ints.column(si, sj)[kept][:, local[sel]])
            cpos.append(cp[sel])
            ncol += sel.sum()
        cpos = numpy.hstack(cpos)
        # residual of the candidate columns w.r.t. all previous vectors (one
        # pass over the vectors per macro iteration)
        B = numpy.asfortranarray(numpy.hstack(cols))
        del cols
        if n:
            B -= lib.dot(L[:n].T, L[:n, cpos])

        # pivoted Cholesky of the candidates: the chosen column is completed
        # with the vectors made since the last refresh (m0..n); every NB
        # vectors the whole residual block is refreshed with one gemm
        m0 = n
        while True:
            dc = d[cpos]
            k = numpy.argmax(dc)
            if dc[k] < cut:
                break
            if n == L.shape[0]:
                L = grow(L, n)
            b = B[:, k]
            if n > m0:
                b = b - L[m0:n].T @ L[m0:n, cpos[k]]
            v = b * (1. / numpy.sqrt(dc[k]))
            L[n] = v
            n += 1
            pivots.append(kept[cpos[k]])
            d -= v * v
            d[cpos[k]] = 0
            if n - m0 == NB:
                live = d[cpos] >= cut
                B = numpy.asfortranarray(B[:, live])
                cpos = cpos[live]
                if cpos.size == 0:
                    break
                B -= lib.dot(L[m0:n].T, L[m0:n, cpos])
                m0 = n
        numpy.maximum(d, 0, out=d)
        log.debug1('CD macro iteration %d: %d candidate columns, nchol = %d, '
                   'max residual = %.2e', nmacro, ncol, n, d.max())

    log.info('CD: %d vectors (%.1f nao) in %d macro iterations, tau = %g',
             n, n / max(nao, 1), nmacro, tau)
    log.timer('CD pivoted decomposition', *t1)
    return L[:n], kept, numpy.asarray(pivots, dtype=int)


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
        pivots : list of (shell_i, shell_j, func_i, func_j) or None
            The pivot pairs after build().
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
        L, kept, piv = pivoted_cholesky(mol, self.tau, self.sigma, self.max_memory, self.verbose)
        self.pivots = self._pivot_labels(piv)
        if method == 'onestep':
            nao = mol.nao_nr()
            if kept.size == nao*(nao+1)//2:
                cderi = L
            else:
                cderi = numpy.zeros((L.shape[0], nao*(nao+1)//2))
                cderi[:, kept] = L
            del L
        else:
            del L
            cderi = self._rebuild(mol, piv, log)

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

    def _pivot_labels(self, piv):
        '''(shell_i, shell_j, func_i_in_shell, func_j_in_shell) of the pivots.'''
        ao_loc = self.mol.ao_loc_nr()
        ao2shl = numpy.repeat(numpy.arange(self.mol.nbas), numpy.diff(ao_loc))
        i = ((numpy.sqrt(8*piv+1) - 1) // 2).astype(int)
        j = piv - i*(i+1)//2
        return [(ao2shl[a], ao2shl[b], a - ao_loc[ao2shl[a]], b - ao_loc[ao2shl[b]])
                for a, b in zip(i, j)]

    def _rebuild(self, mol, piv, log):
        '''Two-step CD, step 2: L = K^-1 (pivots | pairs), J = K K^T.'''
        t0 = (logger.process_clock(), logger.perf_counter())
        ints = _Int2e(mol)
        ao_loc = ints.ao_loc
        npiv = piv.size
        nao = ao_loc[-1]
        npair = nao*(nao+1)//2
        need = 2 * npiv * npair * 8 / 1e6
        avail = self.max_memory - lib.current_memory()[0]
        if need > avail:
            raise MemoryError('two-step CD needs %.0f MB for %d pivots, %.0f MB left'
                              % (need, npiv, avail))
        ao2shl = numpy.repeat(numpy.arange(mol.nbas), numpy.diff(ao_loc))
        i = ((numpy.sqrt(8*piv+1) - 1) // 2).astype(int)
        j = piv - i*(i+1)//2
        blocks = {}
        for k, (a, b) in enumerate(zip(i, j)):
            blocks.setdefault((ao2shl[a], ao2shl[b]), []).append(k)
        T = numpy.empty((npiv, npair))
        for (si, sj), ks in blocks.items():
            pairs, local = _shell_pair_columns(ao_loc, si, sj)
            col = ints.column(si, sj)
            where = {p: c for p, c in zip(pairs, local)}
            T[ks] = col[:, [where[piv[k]] for k in ks]].T
        J = T[:, piv]
        J = (J + J.T) * .5
        try:
            K = scipy.linalg.cholesky(J, lower=True)
        except numpy.linalg.LinAlgError:
            raise RuntimeError('two-step CD: the pivot metric is not positive definite')
        self.metric_chol = K
        cderi = scipy.linalg.solve_triangular(K, T, lower=True, overwrite_b=True)
        log.timer('CD two-step reconstruction', *t0)
        return cderi
