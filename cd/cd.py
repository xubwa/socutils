#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Cholesky decomposition of electron repulsion integrals.
# Compatible with PySCF DF interface — inherits from df.DF so that
# _cderi / loop() / get_naoaux() work transparently for downstream code.
#

import os
import time
import numpy as np
import scipy.linalg
from pyscf import df, lib, gto
from pyscf.lib import logger
from socutils.tools import timer


def cholesky(mf, tau=1e-4, sigma=1e-2, method='threeloop',
             cderi=None, cderi_to_save=None, with_df=None, compress=True, **kwargs):
    """Attach Cholesky-decomposed ERIs to an SCF object -- the CD analogue of
    ``mf.density_fit()``.

    Builds a :class:`CD` object (a ``df.DF`` subclass producing ``_cderi`` in
    the same compressed-triangular layout as density fitting) and routes it
    through the mean field's own ``density_fit(with_df=...)``, so it works for
    spinor, GHF and plain pyscf mean fields alike:

        mf = spinor_hf.SCF(mol).x2camf().cholesky()

    The on-disk cache uses exactly the same interface as pyscf DF -- the
    ``_cderi_to_save`` / ``_cderi`` attributes -- exposed here as keywords:

        # save the decomposition (first run computes, writes 'cderi.h5'):
        mf = spinor_hf.SCF(mol).x2camf().cholesky(cderi_to_save='cderi.h5')
        # reuse it (skip the decomposition entirely):
        mf = spinor_hf.SCF(mol).x2camf().cholesky(cderi='cderi.h5')

    equivalently, set ``mf.with_df._cderi_to_save`` / ``mf.with_df._cderi``
    directly, just like a density-fitted object.  With neither, the vectors
    stay in memory and plug into the DF machinery like ``.density_fit()``.

    Extra keyword arguments (e.g. ``only_dfj``) are forwarded to the underlying
    ``density_fit``.
    """
    if with_df is None:
        with_df = CD(mf.mol, tau=tau, sigma=sigma, method=method, compress=compress)
    if cderi is not None:
        with_df._cderi = cderi
    if cderi_to_save is not None:
        with_df._cderi_to_save = cderi_to_save
    return mf.density_fit(with_df=with_df, **kwargs)



def get_cderi_ao(mf):
    '''AO Cholesky (or density-fitting) factors of ``mf.with_df``, unpacked to
    shape (naux, nao, nao).  Correlated methods take their integrals from the
    mean field: attach them with ``mf.cholesky()`` or ``mf.density_fit()``.'''
    with_df = getattr(mf, 'with_df', None)
    if with_df is None:
        raise ValueError('%s has no with_df; attach the factorised integrals with '
                         'mf.cholesky() or mf.density_fit() first' % mf.__class__.__name__)
    return np.vstack([lib.unpack_tril(blk) for blk in with_df.loop()])


def chunked_cholesky_threeloop(mol, max_error=1e-5, verbose=True, cmax=15):
    """Modified cholesky decomposition from pyscf eris.

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    start = time.time()
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
        #print(l, nc)
    dims = mol.ao_loc_nr()
    timer_ao = timer.Timer() 
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        #print(shls)
        timer_ao.start()
        buf = mol.intor('int2e', shls_slice=shls)
        #timer_ao.accumulate()
        #print(timer_ao.cpu_time, timer_ao.wall_time, flush=True)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs.")
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
    while abs(delta_max) > max_error:
        # Update cholesky vector
        iter_start = time.time()
        # M'_ii = L_i^x L_i^x
        # D_ii = M_ii - M'_ii
        # find nu through max delta
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        timer_ao.start()
        eri_col = mol.intor('int2e', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
        timer_ao.accumulate()
        # Select correct ERI chunk from shell.
        ang_j = mol.bas_angular(sj)
        ang_l = mol.bas_angular(sl)
        sub_delta_max = delta_max
        while (sub_delta_max) > max(max_error, delta_max*0.01):
            sub_delta = delta.reshape(nao,nao)[dims[sj]:dims[sj+1], dims[sl]:dims[sl+1]]
            #print(sub_delta)
            sub_nu = np.argmax(np.abs(sub_delta))
            cj0, cl0 = max(j - dims[sj], 0), max(l - dims[sl], 0)
            cj = sub_nu // (dims[sl+1]-dims[sl])
            cl = sub_nu % (dims[sl+1]-dims[sl])
            #print(dims, dims[sj], dims[sl])
            #print(sj, sl, cj0, cl0, cj, cl)
            i_ang_j = cj // (2*ang_j+1)
            i_ang_l = cl // (2*ang_l+1)
            j_start = i_ang_j * (2 * ang_j + 1)
            j_end = j_start + 2 * ang_j + 1
            l_start = i_ang_l * (2 * ang_l + 1)
            l_end = l_start + 2 * ang_l + 1
            Munu0 = eri_col[:, :, j_start:j_end, l_start:l_end]
            for i_j in range(j_start, j_end):
                for i_l in range(l_start, l_end):
                    nu = (dims[sj]+i_j) * nao + (dims[sl]+i_l)
                    delta_nu = delta[nu]
                    if delta_nu < min(max_error, 1e-8):
                        continue
                    #print(delta[nu], nu, (dims[sj]+cj)*nao+dims[sl]+cl)
                    # Grow the buffer if cmax*nao was not enough, so cmax is
                    # only an initial guess and never a hard cap.  Grow once a
                    # full nao of headroom is left (not at the very last slot),
                    # so a shell-pair block adding several vectors always fits;
                    # nchol ends up at a few times nao, so this costs only a
                    # handful of reallocations.
                    if nchol + nao >= nchol_max:
                        chol_vecs = np.concatenate(
                            [chol_vecs, np.zeros((nao, nao * nao))], axis=0)
                        nchol_max += nao
                    # Updated residual = \sum_x L_i^x L_nu^x
                    R = np.dot(chol_vecs[:nchol + 1, nu], chol_vecs[:nchol + 1, :])
                    munu0 = eri_col[:,:,i_j,i_l].reshape(nao*nao)
                    chol_vecs[nchol + 1] = (munu0 - R) / (delta_nu)**0.5
                    nchol += 1
                    Mapprox += chol_vecs[nchol]*chol_vecs[nchol] 
                    delta = diag-Mapprox
            sub_delta_max = max(sub_delta.reshape(-1))
        if verbose:
            step_time = time.time() - iter_start
            total_time = time.time() - start
            info = (nchol, delta_max, step_time, total_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e total_time = %13.8e" % info, flush=True)
    return chol_vecs[:nchol]


class CD(df.DF):
    """Cholesky decomposition of ERIs.

    After build(), self._cderi contains the Cholesky vectors in the same
    (naux, nao_pair) compressed-triangular format as density fitting,
    so all downstream code (get_jk, loop, etc.) works unchanged.

    Attributes:
        tau : float
            Decomposition threshold.  Controls the accuracy of the
            approximation (mu nu|la ga) ≈ sum_J L^J_{mu nu} L^J_{la ga}.
        pivots : list of tuples or None
            After build(), records the selected pivot basis pairs as
            [(shell_i, shell_j, func_i_in_shell, func_j_in_shell), ...].
            Used by Step 2 (RI construction) and for analytic gradients.
        metric_chol : ndarray or None
            Cholesky factor K of the pivot metric J = KK^T.
            Stored for gradient calculations.
    """

    _keys = {'tau', 'sigma', 'method', 'pivots', 'metric_chol', 'compress'}

    def __init__(self, mol, tau=1e-4, sigma=1e-2, method='threeloop', compress=True):
        df.DF.__init__(self, mol)
        self.tau = tau
        self.sigma = sigma  # span factor (only used by the 'twostep' method)
        # compress: in 'twostep' Step 1, drop converged (Diag < tau) pairs from
        # the active candidate set so the dominant projection reads only the
        # shrinking active block.  EXACT -- Step 2 rebuilds all pairs' vectors
        # from exact integrals, so compression only affects pivot selection.
        self.compress = compress
        # Decomposition algorithm:
        #   'threeloop' (default) -- chunked_cholesky_threeloop below, the
        #                            direct pivoted Cholesky used elsewhere in
        #                            socutils.  Its vector buffer grows
        #                            dynamically, so there is no cap to tune.
        #   'twostep'             -- pivot selection + RI reconstruction
        #                            (_determine_pivots + _construct_vectors);
        #                            stores metric_chol for analytic gradients.
        self.method = method
        self.pivots = None
        self.metric_chol = None

    @staticmethod
    def cderi_shape(cderi, dataname='j3c'):
        '''(naux, nao_pair) of an on-disk cderi, read layout-agnostically.

        A large cderi is stored as an HDF5 *group* of AO-pair column blocks
        (``j3c/0``, ``j3c/1``, ... sharing the aux/row axis), not a single
        ``j3c`` dataset.  loop()/get_naoaux() already read either layout; this
        adds the nao_pair (column) count for a pre-build basis<->cderi sanity
        check, so callers never hand-open the file -- ``f[dataname].shape``
        raises ``'Group' object has no attribute 'shape'`` on the group layout.
        '''
        import h5py
        with h5py.File(cderi, 'r') as f:
            obj = f[dataname]
            if isinstance(obj, h5py.Group):
                blocks = [obj[k] for k in sorted(obj.keys(), key=int)]
                return blocks[0].shape[0], sum(b.shape[1] for b in blocks)
            return tuple(obj.shape)

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        # Make sure the Cholesky vectors exist before delegating, so that BOTH
        # J and K are built from THEM.  Otherwise pyscf's J-only fast path
        # (df_jk.get_j, taken when ``_cderi is None``) would build a separate
        # even-tempered DF auxbasis and compute J from that instead of the CD
        # decomposition -- inconsistent with the CD-based K, and the source of
        # the spurious "ETB ... DF auxbasis" output.
        if self._cderi is None:
            self.build()
        return df.DF.get_jk(self, dm, hermi, with_j, with_k,
                            direct_scf_tol, omega)

    def build(self):
        log = logger.new_logger(self)
        mol = self.mol

        # Same on-disk cache interface as pyscf DF: a pre-set _cderi (file path
        # or in-core array) is used as-is and the decomposition is skipped.
        if self._cderi is not None:
            log.info('CD: using preset _cderi (%s); skipping decomposition',
                     self._cderi if isinstance(self._cderi, str) else 'in-core array')
            return self

        if self.method == 'threeloop':
            cderi = self._build_threeloop(mol, log)
        elif self.method == 'twostep':
            cderi = self._build_twostep(mol, log)
        else:
            raise ValueError("CD.method must be 'threeloop' or 'twostep', "
                             "got %r" % self.method)

        # Honour _cderi_to_save exactly like DF: a string selects an on-disk
        # HDF5 cache (dataset = self._dataname, default 'j3c'); _cderi is then
        # the saved path, so the DF machinery reads from the file.
        if isinstance(self._cderi_to_save, str):
            import h5py
            dataname = getattr(self, '_dataname', 'j3c')
            with h5py.File(self._cderi_to_save, 'w') as f:
                f[dataname] = cderi
            self._cderi = self._cderi_to_save
            log.info('CD: saved %d vectors to %s', cderi.shape[0],
                     self._cderi_to_save)
        else:
            self._cderi = cderi
        return self

    def _build_threeloop(self, mol, log):
        """Default: :func:`chunked_cholesky_threeloop`.

        Returns the Cholesky vectors packed to (naux, nao_pair), the DF layout.
        """
        t0 = (logger.process_clock(), logger.perf_counter())
        nao = mol.nao_nr()
        # The buffer inside chunked_cholesky_threeloop grows dynamically, so the
        # cmax there is only an initial allocation -- nothing to tune here.
        chol = chunked_cholesky_threeloop(mol, max_error=self.tau,
                                          verbose=(self.verbose >= logger.INFO))
        # (nchol, nao*nao) -> (nchol, nao_pair); L_{mu nu} is symmetric in mu,nu.
        cderi = lib.pack_tril(np.asarray(chol).reshape(-1, nao, nao))
        log.info('CD(threeloop): %d vectors, max_error = %g',
                 cderi.shape[0], self.tau)
        log.timer('CD threeloop decomposition', *t0)
        return cderi

    def _build_twostep(self, mol, log):
        """Pivot selection (CD) + RI reconstruction; stores metric_chol."""
        t0 = (logger.process_clock(), logger.perf_counter())
        if getattr(self, 'compress', True):
            pivots, pivot_indices = self._determine_pivots_compressed(mol, self.tau, log)
        else:
            pivots, pivot_indices = self._determine_pivots(mol, self.tau, log)
        self.pivots = pivots
        log.info('CD: %d pivots selected with tau = %g',
                 len(pivot_indices), self.tau)
        t1 = log.timer('CD Step 1: pivot determination', *t0)
        cderi = self._construct_vectors(mol, pivots, pivot_indices, log)
        log.timer('CD Step 2: RI construction', *t1)
        return cderi

    def _determine_pivots(self, mol, tau, log):
        """Step 1: Determine the Cholesky pivot set B.

        Uses the conventional CD algorithm but only records which basis
        function pairs are selected as pivots, without storing the full
        Cholesky vectors.

        Returns:
            pivots : list of (shell_i, shell_j, func_i_local, func_j_local)
            pivot_indices : list of (global_i, global_j) AO function pairs
        """
        import time
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        nbas = mol.nbas

        # Compute diagonal elements (mu mu|mu mu) block by block
        diag = np.zeros(nao * nao)
        ndiag = 0
        t_start = time.perf_counter()
        for i in range(nbas):
            shls = (i, i + 1, 0, nbas, i, i + 1, 0, nbas)
            buf = mol.intor('int2e_sph', shls_slice=shls)
            di = ao_loc[i + 1] - ao_loc[i]
            diag[ndiag:ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
            ndiag += di * nao
        log.info('CD: diagonal computation %.2f sec', time.perf_counter() - t_start)

        # Cholesky vectors stored for diagonal update
        # Preallocate generously so the grow-realloc below (which transiently
        # holds old+new copies, ~2x) is never triggered for typical pivot counts
        # (one-step CD at tau=1e-7 needs ~12*nao here).  At nao~1081 this single
        # ~141 GB buffer fits a 180 GB node, whereas the realloc doubling did not.
        nchol_max = 14 * nao
        chol_vecs = np.zeros((nchol_max, nao * nao))
        nchol = 0
        Mapprox = np.zeros(nao * nao)

        pivots = []
        pivot_indices = []

        sigma = self.sigma
        selected_shell_pairs = set()
        # Only track lower triangle (i >= j)
        tril_idx = np.tril_indices(nao)
        delta = diag.copy()
        while True:
            delta_max = np.max(np.abs(delta.reshape(nao, nao)[tril_idx]))
            if delta_max < tau:
                break

            threshold = max(sigma * delta_max, tau)

            # Keep selecting shell pairs until none exceed span factor threshold
            found_any = False
            while True:
                # Find the shell pair with largest delta among unselected
                delta_2d = np.abs(delta.reshape(nao, nao))
                best_sp = None
                best_val = 0
                for si in range(nbas):
                    for sj in range(si + 1):  # only lower triangle si >= sj
                        if (si, sj) in selected_shell_pairs:
                            continue
                        block = delta_2d[ao_loc[si]:ao_loc[si + 1],
                                         ao_loc[sj]:ao_loc[sj + 1]]
                        val = np.max(block)
                        if val > best_val:
                            best_val = val
                            best_sp = (si, sj)

                if best_sp is None or best_val < threshold:
                    break

                found_any = True
                sj, sl = best_sp
                selected_shell_pairs.add((sj, sl))

                # Compute ERI column for this shell pair
                eri_col = mol.intor('int2e_sph', shls_slice=(
                    0, nbas, 0, nbas, sj, sj + 1, sl, sl + 1))

                # Collect function pairs in this shell pair (lower triangle only: gj >= gl)
                dj = ao_loc[sj + 1] - ao_loc[sj]
                dl = ao_loc[sl + 1] - ao_loc[sl]
                pairs = []
                for i_j in range(dj):
                    for i_l in range(dl):
                        gj = ao_loc[sj] + i_j
                        gl = ao_loc[sl] + i_l
                        if gj < gl:
                            continue  # skip upper triangle
                        idx = gj * nao + gl
                        pairs.append((idx, i_j, i_l, gj, gl))

                # Process by descending delta for numerical stability
                while pairs:
                    pairs.sort(key=lambda p: -np.abs(delta[p[0]]))
                    idx, i_j, i_l, gj, gl = pairs.pop(0)
                    delta_nu = delta[idx]

                    if delta_nu < 1e-14:
                        break

                    # Record pivot
                    pivots.append((sj, sl, i_j, i_l))
                    pivot_indices.append((gj, gl))

                    # Construct Cholesky vector (needed for diagonal update)
                    R = np.dot(chol_vecs[:nchol, idx], chol_vecs[:nchol, :]) if nchol > 0 else 0
                    munu = eri_col[:, :, i_j, i_l].reshape(nao * nao)
                    chol_vecs[nchol] = (munu - R) / np.sqrt(delta_nu)
                    Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
                    nchol += 1
                    delta = diag - Mapprox

                    if nchol >= nchol_max:
                        new_vecs = np.zeros((nchol_max + nao, nao * nao))
                        new_vecs[:nchol_max] = chol_vecs
                        chol_vecs = new_vecs
                        nchol_max += nao

                log.info('  shell pair (%d,%d): nchol = %4d, %.2f sec',
                         sj, sl, nchol, time.perf_counter() - t_start)

            if not found_any:
                break

            t_iter = time.perf_counter() - t_start
            log.info('CD pivot: nchol = %4d, delta_max = %.2e, wall time = %.2f sec',
                     nchol, delta_max, t_iter)

        log.info('CD Step 1 total: %d chol vecs, %d shell pairs, %.2f sec',
                 nchol, len(selected_shell_pairs),
                 time.perf_counter() - t_start)
        return pivots, pivot_indices

    def _determine_pivots_compressed(self, mol, tau, log, compact_every=64):
        """Step 1 with ACTIVE-SET COMPRESSION (exact in the two-step framework).

        Identical structure to _determine_pivots -- span-factor shell-pair
        selection, eri_col computed ONCE per shell pair and reused -- but the
        working Cholesky-vector columns are kept active-compacted: pairs whose
        diagonal residual falls below tau are dropped, so the dominant
        projection  R = chol[:nchol] . column  reads only the shrinking active
        block instead of the full nao**2 pair space.

        Keeping the shell-pair loop is essential: a per-pair global argmax would
        scatter consecutive pivots across shell pairs and thrash the eri_col
        cache (re-running mol.intor), which more than cancels the projection
        saving.  Here intor is unchanged from the baseline; only R=dot shrinks.

        EXACT -- Step 2 rebuilds the vectors for ALL mu nu from exact integrals,
        so a compressed-out (converged) pair still gets its exact vector;
        compression only removes it from future pivot consideration (Schwarz
        screening to tau).  pos2pair / pair2pos keep the active-prefix
        permutation internal; pivots are recorded in original coordinates.
        """
        import time
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        nbas = mol.nbas
        n = nao * nao
        t_start = time.perf_counter()

        diag = np.zeros(n)
        nd = 0
        for i in range(nbas):
            buf = mol.intor('int2e_sph', shls_slice=(i, i + 1, 0, nbas, i, i + 1, 0, nbas))
            di = ao_loc[i + 1] - ao_loc[i]
            diag[nd:nd + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
            nd += di * nao
        log.info('CD: diagonal computation %.2f sec', time.perf_counter() - t_start)

        delta = diag.copy()                  # FULL nao**2 -- for shell-pair selection
        pos2pair = np.arange(n)              # active position -> original pair index
        pair2pos = np.arange(n)              # original pair  -> active position
        chol = np.zeros((min(14 * nao, n), n))
        na = n
        nchol = 0
        pivots = []
        pivot_indices = []
        sigma = self.sigma
        selected_shell_pairs = set()
        tril_idx = np.tril_indices(nao)

        while True:
            delta_max = np.max(np.abs(delta.reshape(nao, nao)[tril_idx]))
            if delta_max < tau:
                break
            threshold = max(sigma * delta_max, tau)
            found_any = False
            while True:
                delta_2d = np.abs(delta.reshape(nao, nao))
                best_sp = None
                best_val = 0
                for si in range(nbas):
                    for sj in range(si + 1):
                        if (si, sj) in selected_shell_pairs:
                            continue
                        v = np.max(delta_2d[ao_loc[si]:ao_loc[si + 1],
                                            ao_loc[sj]:ao_loc[sj + 1]])
                        if v > best_val:
                            best_val = v
                            best_sp = (si, sj)
                if best_sp is None or best_val < threshold:
                    break
                found_any = True
                sj, sl = best_sp
                selected_shell_pairs.add((sj, sl))
                eri_col = mol.intor('int2e_sph', shls_slice=(
                    0, nbas, 0, nbas, sj, sj + 1, sl, sl + 1))
                dj = ao_loc[sj + 1] - ao_loc[sj]
                dl = ao_loc[sl + 1] - ao_loc[sl]
                pairs = []
                for i_j in range(dj):
                    for i_l in range(dl):
                        gj = ao_loc[sj] + i_j
                        gl = ao_loc[sl] + i_l
                        if gj < gl:
                            continue
                        pairs.append((gj * nao + gl, i_j, i_l, gj, gl))

                while pairs:
                    pairs.sort(key=lambda p: -np.abs(delta[p[0]]))
                    idx, i_j, i_l, gj, gl = pairs.pop(0)
                    dnu = delta[idx]
                    # stop at tau (not 1e-14): a sub-tau pair may already be
                    # compacted out of the active set, so picking it would use a
                    # stale pair2pos.  Step 2 reconstructs everything to tau anyway.
                    if dnu < tau:
                        break
                    pivots.append((sj, sl, i_j, i_l))
                    pivot_indices.append((gj, gl))
                    if nchol >= chol.shape[0]:
                        chol = np.vstack([chol, np.zeros((nao, n))])
                    pos = pair2pos[idx]                              # pivot's active column
                    munu = eri_col[:, :, i_j, i_l].reshape(n)[pos2pair[:na]]
                    R = chol[:nchol, :na].T @ chol[:nchol, pos] if nchol else 0.0
                    L = (munu - R) / np.sqrt(dnu)
                    chol[nchol, :na] = L
                    # incremental active-diagonal update, scattered into full delta
                    delta[pos2pair[:na]] = np.clip(delta[pos2pair[:na]] - L * L, 0, None)
                    nchol += 1

                    if nchol % compact_every == 0:                  # drop converged pairs
                        keep = delta[pos2pair[:na]] > tau
                        nk = int(keep.sum())
                        if nk < na:
                            order = np.concatenate([np.where(keep)[0], np.where(~keep)[0]])
                            chol[:nchol, :na] = chol[:nchol, order]
                            pos2pair[:na] = pos2pair[order]
                            pair2pos[pos2pair[:nk]] = np.arange(nk)
                            na = nk

                log.info('  shell pair (%d,%d): nchol = %4d, active = %d, %.2f sec',
                         sj, sl, nchol, na, time.perf_counter() - t_start)
            if not found_any:
                break

        log.info('CD Step 1 (compressed): %d pivots, final active %d / %d, %.2f sec',
                 nchol, na, n, time.perf_counter() - t_start)
        return pivots, pivot_indices

    def _construct_vectors(self, mol, pivots, pivot_indices, log):
        """Step 2: Construct Cholesky vectors using the RI formula.

        L^p_{mu nu} = sum_{p'} (mu nu|p') K^{-T}_{p'p}

        where J_{pp'} = (p|p') is the metric matrix among pivots,
        and J = KK^T (Cholesky decomposition of J).

        Returns:
            cderi : ndarray, shape (npiv, nao_pair)
                Cholesky vectors in compressed triangular format.
        """
        nao = mol.nao_nr()
        nao_pair = nao * (nao + 1) // 2
        ao_loc = mol.ao_loc_nr()
        nbas = mol.nbas
        npiv = len(pivot_indices)

        if npiv == 0:
            return np.zeros((0, nao_pair))

        # Collect unique pivot shell pairs
        shell_pairs = {}  # (si, sj) -> list of (pivot_idx, func_i, func_j)
        for ipiv, (si, sj, fi, fj) in enumerate(pivots):
            key = (si, sj)
            if key not in shell_pairs:
                shell_pairs[key] = []
            shell_pairs[key].append((ipiv, fi, fj))

        # Three-index integrals (mu nu|p), stored COMPRESSED as (npiv, nao_pair).
        # (mu nu|p) is symmetric in mu<->nu, so only the lower triangle is kept;
        # this halves the buffer vs a dense (npiv, nao, nao) array (and avoids a
        # second dense cderi_full), so the build fits in ~2*npiv*nao_pair*8 bytes
        # instead of ~3*npiv*nao**2*8 -- e.g. ~114 GB rather than ~290 GB at
        # nao ~ 1081, npiv ~ 12000, which lets it run on a 180 GB node.
        def _tril(gj, gl):                       # pack_tril index; requires gj >= gl
            return gj * (gj + 1) // 2 + gl
        eri_3idx = np.zeros((npiv, nao_pair))
        for (si, sj), piv_list in shell_pairs.items():
            eri_block = mol.intor('int2e_sph', shls_slice=(
                0, nbas, 0, nbas, si, si + 1, sj, sj + 1))
            for ipiv, fi, fj in piv_list:
                eri_3idx[ipiv] = lib.pack_tril(
                    np.ascontiguousarray(eri_block[:, :, fi, fj]))

        # Extract metric J from the three-index integrals
        # J_{pp'} = (p|p') = (gj gl|p') for pivot p = (gj, gl), gj >= gl
        J = np.zeros((npiv, npiv))
        for ipiv, (gj, gl) in enumerate(pivot_indices):
            J[:, ipiv] = eri_3idx[:, _tril(gj, gl)]

        # Decompose metric: try Cholesky, fallback to eig if not positive definite.
        # solve_triangular acts column-by-column over the nao_pair axis, so solving
        # only the lower-triangle columns is identical to solving the full (mu,nu)
        # set then packing -- cderi comes out already in (naux, nao_pair) form.
        J = (J + J.T) * 0.5  # symmetrize
        try:
            K = scipy.linalg.cholesky(J, lower=True)
            self.metric_chol = K
            # L^p_{mu nu} = sum_{p'} (mu nu|p') K^{-T}_{p'p}
            cderi = scipy.linalg.solve_triangular(K, eri_3idx, lower=True)
            naux = npiv
            log.info('CD metric: Cholesky succeeded, %d vectors', naux)
        except scipy.linalg.LinAlgError:
            log.info('CD metric: Cholesky failed, using eigendecomposition')
            lindep = 1e-12
            w, v = scipy.linalg.eigh(J)
            mask = w > lindep
            naux = mask.sum()
            log.info('CD metric: %d / %d eigenvalues above lindep = %g',
                     naux, npiv, lindep)
            v = v[:, mask] / np.sqrt(w[mask])
            self.metric_chol = None
            cderi = lib.dot(v.T, eri_3idx)

        log.info('CD: %d vectors from %d pivots', naux, npiv)
        return cderi
