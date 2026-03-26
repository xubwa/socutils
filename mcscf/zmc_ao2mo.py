import sys, tempfile, ctypes, time, numpy, h5py
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import mc_ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import outcore
from pyscf.ao2mo import r_outcore
from pyscf.ao2mo import nrr_outcore
from pyscf import ao2mo
from socutils.tools import timer
import numpy
import numpy as np
from scipy import linalg as la

def chunked_cholesky(mol, max_error=1e-5, verbose=True, cmax=8):
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
        print(l, nc)
    dims = mol.ao_loc_nr()
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        print(shls)
        buf = mol.intor('int2e_sph', shls_slice=shls)
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
    eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
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
        eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
        # Select correct ERI chunk from shell.
        ang_j = mol.bas_angular(sj)
        ang_l = mol.bas_angular(sl)
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
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
                print(delta[nu], nu, (dims[sj]+cj)*nao+dims[sl]+cl)
                # Updated residual = \sum_x L_i^x L_nu^x
                R = np.dot(chol_vecs[:nchol + 1, nu], chol_vecs[:nchol + 1, :])
                munu0 = eri_col[:,:,i_j,i_l].reshape(nao*nao)
                chol_vecs[nchol + 1] = (munu0 - R) / (delta_nu)**0.5
                nchol += 1
                Mapprox += chol_vecs[nchol]*chol_vecs[nchol] 
                delta = diag-Mapprox
        if verbose:
            step_time = time.time() - iter_start
            total_time = time.time() - start
            info = (nchol, delta_max, step_time, total_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e total_time = %13.8e" % info)

    return chol_vecs[:nchol]

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

def chunked_cholesky_twoloop(mol, max_error=1e-5, verbose=True, cmax=15):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

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
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
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
    eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
    while abs(delta_max) > max_error:
        # Update cholesky vector
        iter_start = time.time()
        # M'_ii = L_i^x L_i^x
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
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
        eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
        # Select correct ERI chunk from shell.
        sub_delta_max = delta_max
        while (sub_delta_max) > max(max_error, delta_max*0.01):
            sub_delta = delta.reshape(nao,nao)[dims[sj]:dims[sj+1], dims[sl]:dims[sl+1]]
            sub_nu = np.argmax(np.abs(sub_delta))
            cj0, cl0 = max(j - dims[sj], 0), max(l - dims[sl], 0)
            cj = sub_nu // (dims[sl+1]-dims[sl])
            cl = sub_nu % (dims[sl+1]-dims[sl])
            print(sub_delta_max, max_error, delta_max, sub_nu, cj, cl, cj0, cl0)
            Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
            nu = (dims[sj]+cj) * nao + (dims[sl]+cl)
            delta_nu = delta[nu]
            # Updated residual = \sum_x L_i^x L_nu^x
            R = np.dot(chol_vecs[:nchol + 1, nu], chol_vecs[:nchol + 1, :])
            chol_vecs[nchol + 1] = (Munu0 - R) / (delta_nu)**0.5
            nchol += 1
            Mapprox += chol_vecs[nchol]*chol_vecs[nchol] 
            delta = diag-Mapprox
            sub_delta = delta.reshape(nao,nao)[dims[sj]:dims[sj+1], dims[sl]:dims[sl+1]]
            sub_delta_max = max(sub_delta.reshape(-1))
        if verbose:
            step_time = time.time() - iter_start
            total_time = time.time() - start
            info = (nchol, delta_max, step_time, total_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e total_time = %13.8e" % info, flush=True)

    return chol_vecs[:nchol]

def chunked_cholesky0(mol, max_error=1e-5, verbose=True, cmax=15):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

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
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
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
    eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        iter_start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
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
        eri_col = mol.intor('int2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol + 1, nu], chol_vecs[:nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - iter_start
            total_time = time.time() - start
            info = (nchol, delta_max, step_time, total_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e total_time = %13.8e" % info, flush=True)

    return chol_vecs[:nchol]

# level = 1: aaaa
# level = 2: paaa
class _CDERIS(lib.StreamObject):
    def __init__(self, zcasscf, mo, level=2, cderi=None, df=False, mode='j-spinor'):
        import gc
        gc.collect()
        mol = zcasscf.mol
        nao, nmo = mo.shape
        ncore = zcasscf.ncore
        ncas = zcasscf.ncas
        nocc = ncore+ncas
        nao_nr = mol.nao_nr()

        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(sys.stdout, zcasscf.verbose)

        # Determine DF integral source
        with_df = getattr(zcasscf._scf, 'with_df', None)
        if with_df is None and cderi is not None:
            # Legacy path: cderi passed as numpy array
            # Convert to compressed triangular for uniform handling
            ncd = cderi.shape[0]
            cderi = cderi.reshape((ncd, mol.nao_nr(), mol.nao_nr()))

        # Project MOs to sph GHF basis
        c2 = numpy.vstack(mol.sph2spinor_coeff())
        if mode == 'ghf':
            c2 = numpy.eye(c2.shape[0], dtype=complex)
        moa_sph = numpy.dot(c2, mo[:, ncore:nocc])  # (2*nao_nr, ncas)
        mop_sph = numpy.dot(c2, mo)                  # (2*nao_nr, nmo)

        if level is 2:
            cd_pa = self._build_cd_pa(
                with_df, cderi, mol, nao_nr, nmo, ncas,
                moa_sph, mop_sph, log)
            self.cd_pa = cd_pa
            self.cd_aa = cd_pa[:, ncore:nocc, :].copy()
        elif level > 2:
            raise NotImplementedError('level > 2 with new CDERIS not yet supported')

        self.aaaa = lib.einsum('ptu,pvw->tuvw', self.cd_aa, self.cd_aa)
        self.paaa = lib.einsum('ptu,pvw->tuvw', self.cd_pa, self.cd_aa)
        self._scf = zcasscf._scf
        self.mo = mo
        self.ncore = ncore
        self.ncas = ncas
        self.nocc = nocc
        log.timer('CD integral transformation', *t0)

    def _build_cd_pa(self, with_df, cderi, mol, nao_nr, nmo, ncas,
                     moa_sph, mop_sph, log):
        """Build cd_pa using C library half-transform + second-step contraction.

        Step 1: L_{P,μ,t} = Σ_ν eri(P,μ,ν) C_{ν,t}  (C library, real/imag split)
        Step 2: cd_pa[P,p,t] = Σ_μ C*_{p,μ} L_{P,μ,t}  (numpy dot)
        """
        import ctypes
        from pyscf.ao2mo import _ao2mo

        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
        fmmm   = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
        fdrv   = _ao2mo.libao2mo.AO2MOnr_e2_drv
        null   = lib.c_null_ptr()

        # Split active MOs (sph GHF basis) into alpha/beta, real/imag
        C_aR = numpy.asfortranarray(moa_sph[:nao_nr].real)
        C_aI = numpy.asfortranarray(moa_sph[:nao_nr].imag)
        C_bR = numpy.asfortranarray(moa_sph[nao_nr:].real)
        C_bI = numpy.asfortranarray(moa_sph[nao_nr:].imag)

        mop_sph_H = mop_sph.T.conj()  # (nmo, 2*nao_nr)

        def _half(eri1, C_real, buf):
            naux_ = eri1.shape[0]
            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 C_real.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_), ctypes.c_int(nao_nr),
                 (ctypes.c_int * 4)(0, ncas, 0, nao_nr),
                 null, ctypes.c_int(0))

        if with_df is not None:
            # Read from with_df.loop() — compressed triangular format
            naux = with_df.get_naoaux()
            cd_pa = numpy.zeros((naux, nmo, ncas), dtype=complex)
            blksize = with_df.blockdim
            buf = numpy.empty((blksize * ncas, nao_nr))

            b0 = 0
            for eri1 in with_df.loop(blksize):
                naux_blk = eri1.shape[0]
                bufslice = buf[:naux_blk * ncas]

                # Half-transform alpha real/imag, beta real/imag
                _half(eri1, C_aR, bufslice)
                LaR = bufslice.reshape(naux_blk, ncas, nao_nr).copy()
                _half(eri1, C_aI, bufslice)
                LaI = bufslice.reshape(naux_blk, ncas, nao_nr).copy()
                _half(eri1, C_bR, bufslice)
                LbR = bufslice.reshape(naux_blk, ncas, nao_nr).copy()
                _half(eri1, C_bI, bufslice)
                LbI = bufslice.reshape(naux_blk, ncas, nao_nr).copy()

                # Combine into complex L_{P,μ,t} in sph GHF basis (2*nao_nr, ncas)
                # La = LaR + i*LaI, shape (naux_blk, ncas, nao_nr)
                # Lb = LbR + i*LbI, shape (naux_blk, ncas, nao_nr)
                # L_{P,μ,t}: stack alpha and beta → (naux_blk, ncas, 2*nao_nr)
                # then transpose to (naux_blk, 2*nao_nr, ncas)
                La = LaR + 1j * LaI  # (naux_blk, ncas, nao_nr)
                Lb = LbR + 1j * LbI
                L_half = numpy.concatenate([La, Lb], axis=2)  # (naux_blk, ncas, 2*nao_nr)
                L_half = L_half.transpose(0, 2, 1)  # (naux_blk, 2*nao_nr, ncas)

                # Step 2: cd_pa[P,p,t] = Σ_μ C*_{p,μ} L_{P,μ,t}
                for i in range(naux_blk):
                    cd_pa[b0 + i] = numpy.dot(mop_sph_H, L_half[i])
                b0 += naux_blk

            log.info('CDERIS: %d aux functions from with_df', naux)
        else:
            # Legacy path: cderi as (ncd, nao_nr, nao_nr) numpy array
            ncd = cderi.shape[0]
            cd_pa = numpy.zeros((ncd, nmo, ncas), dtype=complex)
            for i in range(ncd):
                chol_i = cderi[i]
                tmp = numpy.vstack((
                    numpy.dot(chol_i, moa_sph[:nao_nr]),
                    numpy.dot(chol_i, moa_sph[nao_nr:])))
                cd_pa[i] = numpy.dot(mop_sph_H, tmp)

        return cd_pa

    def get_jk(self, dm, mo_coeff=None, mo_occ=None):
        """Compute J and K matrices.

        If mo_coeff and mo_occ are provided, tags them onto dm for MO path.
        Otherwise falls back to DM path.
        """
        if mo_coeff is not None and mo_occ is not None:
            dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        return self._scf.get_jk(self._scf.mol, dm)

    def get_jk_active_mo(self, casdm1):
        """Compute active J and K matrices in MO basis.

        K_a from cd_pa and casdm1 directly.
        J_a via _scf.get_jk (DM path, J only), then transformed to MO basis.

        Returns:
            vj_a_mo, vk_a_mo: J and K in MO basis (nmo, nmo)
        """
        mo = self.mo
        nmo = mo.shape[1]
        ncore = self.ncore
        nocc = self.nocc

        # K_a in MO basis from cd_pa
        tmp = lib.einsum('Ppt,tu->Ppu', self.cd_pa, casdm1)
        vk_a_mo = lib.einsum('Ppu,Pqu->pq', tmp, self.cd_pa.conj())

        # J_a via DM path (J only), then transform to MO basis
        dm_active = numpy.zeros((nmo, nmo), dtype=complex)
        dm_active[ncore:nocc, ncore:nocc] = casdm1
        dm_active_ao = reduce(numpy.dot, (mo, dm_active, mo.T.conj()))
        vj_a, _ = self._scf.get_jk(self._scf.mol, dm_active_ao,
                                     with_j=True, with_k=False)
        vj_a_mo = reduce(numpy.dot, (mo.T.conj(), vj_a, mo))

        return vj_a_mo, vk_a_mo

class _ERIS(object):
    def __init__(self, zcasscf, mo, method='outcore', level=1):
        mol = zcasscf.mol
        nao, nmo = mo.shape
        ncore = zcasscf.ncore
        ncas = zcasscf.ncas
        nocc = ncore+ncas

        mem_incore, mem_outcore, mem_basic = mc_ao2mo._mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        eri = zcasscf._scf._eri
        moc, moa, moo = mo[:,:ncore], mo[:,ncore:nocc], mo[:,:nocc]
        if (method == 'incore' or mol.incore_anyway):
            raise NotImplementedError

        else:
            import gc
            gc.collect()
            log = logger.Logger(zcasscf.stdout, zcasscf.verbose)
            self.feri = lib.H5TmpFile()
            max_memory = max(3000, zcasscf.max_memory*.9-mem_now)
            if max_memory < mem_basic:
                log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                         (mem_basic+mem_now)/.9, zcasscf.max_memory)
            if level == 1:
                #r_outcore.general(mol, (moa, moa, moa, moa), self.feri, dataname='aaaa', intor="int2e_spinor")
                nrr_outcore.general(mol, (moa, moa, moa, moa), self.feri, dataname='aaaa', motype='ghf', verbose=5)
                self.aaaa = \
                        self.feri['aaaa'][:,:].reshape((ncas, ncas, ncas, ncas))
            elif level == 2:   
                nrr_outcore.general(mol, (moa, moa, mo, moa), self.feri, dataname='aapa', motype='j-spinor', verbose=5)
                #r_outcore.general(mol, (moa, moa, mo, moa), self.feri, dataname='aapa', intor="int2e_spinor", verbose=5)
                self.paaa = self.feri['aapa'][:,:].T.reshape((nmo, ncas, ncas, ncas))
                #r_outcore.general(mol, (mo, moa, moa, moa), self.feri, dataname='paaa', intor="int2e_spinor", verbose=5)
                #self.paaa = self.feri['paaa'][:,:].reshape((nmo, ncas, ncas, ncas))
            else:
                #r_outcore.general(mol, (mo, mo, mo, mo), self.feri, dataname='pppp', intor="int2e_spinor", verbose=5)
                nrr_outcore.general(mol, (mo, mo, mo, mo), self.feri, dataname='pppp', motype='j-spinor', verbose=5)
                self.pppp = self.feri['pppp'][:,:].reshape((nmo, nmo, nmo, nmo))
            
        if (level == 1):
            self.aaaa.shape = (ncas, ncas, ncas, ncas)
        elif (level == 2):
            self.paaa.shape = (nmo, ncas, ncas, ncas)
            self.aaaa = self.paaa[ncore:nocc]
        else:
            self.paaa = self.pppp[:,ncore:nocc,ncore:nocc,ncore:nocc]
            self.aaaa = self.pppp[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc]
