#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
# Cholesky decomposition of electron repulsion integrals.
# Compatible with PySCF DF interface — inherits from df.DF so that
# _cderi / loop() / get_naoaux() work transparently for downstream code.
#

import os
import numpy as np
import scipy.linalg
from pyscf import df, lib, gto
from pyscf.lib import logger


def cholesky(mf, tau=1e-4, sigma=1e-2, method='threeloop',
             cderi=None, cderi_to_save=None, with_df=None, **kwargs):
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
        with_df = CD(mf.mol, tau=tau, sigma=sigma, method=method)
    if cderi is not None:
        with_df._cderi = cderi
    if cderi_to_save is not None:
        with_df._cderi_to_save = cderi_to_save
    return mf.density_fit(with_df=with_df, **kwargs)


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

    _keys = {'tau', 'sigma', 'method', 'pivots', 'metric_chol'}

    def __init__(self, mol, tau=1e-4, sigma=1e-2, method='threeloop'):
        df.DF.__init__(self, mol)
        self.tau = tau
        self.sigma = sigma  # span factor (only used by the 'twostep' method)
        # Decomposition algorithm:
        #   'threeloop' (default) -- zmc_ao2mo.chunked_cholesky_threeloop, the
        #                            direct pivoted Cholesky used elsewhere in
        #                            socutils.  Its vector buffer grows
        #                            dynamically, so there is no cap to tune.
        #   'twostep'             -- pivot selection + RI reconstruction
        #                            (_determine_pivots + _construct_vectors);
        #                            stores metric_chol for analytic gradients.
        self.method = method
        self.pivots = None
        self.metric_chol = None

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
        """Default: chunked_cholesky_threeloop from socutils.mcscf.zmc_ao2mo.

        Returns the Cholesky vectors packed to (naux, nao_pair), the DF layout.
        """
        from socutils.mcscf.zmc_ao2mo import chunked_cholesky_threeloop
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
        nchol_max = 8 * nao
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

        # Compute three-index integrals (mu nu|p) for all AO pairs and pivots
        # Stored as (npiv, nao, nao) then compressed
        eri_3idx = np.zeros((npiv, nao, nao))
        for (si, sj), piv_list in shell_pairs.items():
            eri_block = mol.intor('int2e_sph', shls_slice=(
                0, nbas, 0, nbas, si, si + 1, sj, sj + 1))
            for ipiv, fi, fj in piv_list:
                eri_3idx[ipiv] = eri_block[:, :, fi, fj]

        # Extract metric J from the three-index integrals
        # J_{pp'} = (p|p') = eri_3idx[p', gj, gl] where (gj, gl) is pivot p
        J = np.zeros((npiv, npiv))
        for ipiv, (gj, gl) in enumerate(pivot_indices):
            J[:, ipiv] = eri_3idx[:, gj, gl]

        # Decompose metric: try Cholesky, fallback to eig if not positive definite
        J = (J + J.T) * 0.5  # symmetrize
        try:
            K = scipy.linalg.cholesky(J, lower=True)
            self.metric_chol = K
            # L^p_{mu nu} = sum_{p'} (mu nu|p') K^{-T}_{p'p}
            cderi_full = scipy.linalg.solve_triangular(
                K, eri_3idx.reshape(npiv, -1), lower=True)
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
            cderi_full = lib.dot(v.T, eri_3idx.reshape(npiv, -1))

        # Compress to triangular format (naux, nao_pair)
        L = cderi_full.reshape(naux, nao, nao)
        cderi = np.zeros((naux, nao_pair))
        for q in range(naux):
            cderi[q] = lib.pack_tril(L[q])

        log.info('CD: %d vectors from %d pivots', naux, npiv)
        return cderi
