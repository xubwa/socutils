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
        self.cd_up = None
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
        moo_sph = numpy.dot(c2, mo[:, :nocc])        # (2*nao_nr, nocc)
        mop_sph = numpy.dot(c2, mo)                   # (2*nao_nr, nmo)

        if level is 2:
            if with_df is not None:
                self.vj_core = None
                self.jtu_tril = None
                cd_pa, vk_core = self._build_cd_pa_and_vk(
                    with_df, mol, nao_nr, nmo, ncore, ncas, nocc,
                    moo_sph, mop_sph, log, cache_owner=zcasscf)
                self.cd_pa = cd_pa
                self.cd_aa = cd_pa[:, ncore:nocc, :].copy()
                self.vk_core = vk_core  # spinor AO basis
            elif cderi is not None:
                # Legacy path
                moa_sph = numpy.dot(c2, mo[:, ncore:nocc])
                cd_pa = self._build_cd_pa(
                    None, cderi, mol, nao_nr, nmo, ncas,
                    moa_sph, mop_sph, log)
                self.cd_pa = cd_pa
                self.cd_aa = cd_pa[:, ncore:nocc, :].copy()
                self.vk_core = None
                self.vj_core = None
                self.jtu_tril = None
            else:
                raise ValueError('Either with_df or cderi must be provided')
        elif level > 2:
            raise NotImplementedError('level > 2 with new CDERIS not yet supported')

        self.aaaa = lib.einsum('ptu,pvw->tuvw', self.cd_aa, self.cd_aa)
        self.paaa = lib.einsum('ptu,pvw->tuvw', self.cd_pa, self.cd_aa)

        # Memory report
        def _mb(arr):
            return arr.nbytes / 1e6 if arr is not None else 0
        log.info('CDERIS memory: cd_pa %.1f MB, cd_aa %.1f MB, aaaa %.1f MB, paaa %.1f MB, vk_core %.1f MB, total %.1f MB',
                 _mb(self.cd_pa), _mb(self.cd_aa), _mb(self.aaaa), _mb(self.paaa),
                 _mb(getattr(self, 'vk_core', None)),
                 _mb(self.cd_pa) + _mb(self.cd_aa) + _mb(self.aaaa) + _mb(self.paaa) + _mb(getattr(self, 'vk_core', None)))
        self._scf = zcasscf._scf
        self.mo = mo
        self.ncore = ncore
        self.ncas = ncas
        self.nocc = nocc
        log.timer('CD integral transformation', *t0)

    @staticmethod
    def _half_transform(with_df, mo_sph, nao_nr, log):
        """Half-transform DF integrals with complex MO coefficients in sph GHF basis.

        L_{P,μ,i} = Σ_ν eri(P,μ,ν) C_{ν,i}

        Uses PySCF C library with real/imag splitting for complex MOs.

        Args:
            with_df: DF object with loop() method
            mo_sph: complex MO coefficients in sph GHF basis, shape (2*nao_nr, nmo_subset)
            nao_nr: number of real spherical AO basis functions
            log: logger

        Returns:
            L_half: (naux, 2*nao_nr, nmo_subset) complex half-transformed integrals
        """
        import ctypes
        from pyscf.ao2mo import _ao2mo

        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
        fmmm   = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
        fdrv   = _ao2mo.libao2mo.AO2MOnr_e2_drv
        null   = lib.c_null_ptr()

        nmo_sub = mo_sph.shape[1]

        # Split MOs into alpha/beta, real/imag
        C_aR = numpy.asfortranarray(mo_sph[:nao_nr].real)
        C_aI = numpy.asfortranarray(mo_sph[:nao_nr].imag)
        C_bR = numpy.asfortranarray(mo_sph[nao_nr:].real)
        C_bI = numpy.asfortranarray(mo_sph[nao_nr:].imag)

        def _half(eri1, C_real, buf):
            naux_ = eri1.shape[0]
            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 C_real.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_), ctypes.c_int(nao_nr),
                 (ctypes.c_int * 4)(0, nmo_sub, 0, nao_nr),
                 null, ctypes.c_int(0))

        naux = with_df.get_naoaux()
        L_half = numpy.zeros((naux, 2 * nao_nr, nmo_sub), dtype=complex)
        blksize = with_df.blockdim
        buf = numpy.empty((blksize * nmo_sub, nao_nr))

        b0 = 0
        for eri1 in with_df.loop(blksize):
            naux_blk = eri1.shape[0]
            bufslice = buf[:naux_blk * nmo_sub]

            _half(eri1, C_aR, bufslice)
            LaR = bufslice.reshape(naux_blk, nmo_sub, nao_nr).copy()
            _half(eri1, C_aI, bufslice)
            LaI = bufslice.reshape(naux_blk, nmo_sub, nao_nr).copy()
            _half(eri1, C_bR, bufslice)
            LbR = bufslice.reshape(naux_blk, nmo_sub, nao_nr).copy()
            _half(eri1, C_bI, bufslice)
            LbI = bufslice.reshape(naux_blk, nmo_sub, nao_nr).copy()

            # Combine: L_{P,μ,i} with μ in sph GHF basis (2*nao_nr)
            # La/Lb shape: (naux_blk, nmo_sub, nao_nr)
            # → (naux_blk, nmo_sub, 2*nao_nr) → transpose to (naux_blk, 2*nao_nr, nmo_sub)
            La = LaR + 1j * LaI
            Lb = LbR + 1j * LbI
            L_chunk = numpy.concatenate([La, Lb], axis=2).transpose(0, 2, 1)
            L_half[b0:b0 + naux_blk] = L_chunk
            b0 += naux_blk

        return L_half

    @staticmethod
    def _cderi_blocks(with_df, blksize, work_mb, cache_owner, log):
        """AO Cholesky/DF vectors block by block, as with_df.loop does.

        Every macro iteration reads all the vectors again.  When they are on
        disk and fit into the memory left under cache_owner.max_memory (after
        work_mb for the buffers of the pass), they are read once and kept on
        cache_owner; otherwise they are read from disk every time, as before.
        """
        src = with_df._cderi
        if cache_owner is None or isinstance(src, numpy.ndarray):
            return with_df.loop(blksize)
        cached = getattr(cache_owner, '_cderi_incore', None)
        if cached is not None and cached[0] == src:
            arr = cached[1]
        else:
            cache_owner._cderi_incore = None
            naux = with_df.get_naoaux()
            nao = with_df.mol.nao_nr()
            size_mb = naux * (nao * (nao + 1) // 2) * 8 / 1e6
            avail_mb = cache_owner.max_memory * .9 - lib.current_memory()[0] - work_mb
            if size_mb > avail_mb:
                log.info('CDERIS: AO vectors (%.0f MB) are read from disk in every macro iteration: '
                         '%.0f MB left under max_memory = %.0f MB', size_mb, max(avail_mb, 0), cache_owner.max_memory)
                return with_df.loop(blksize)
            arr = None
            b0 = 0
            for eri1 in with_df.loop(blksize):
                if eri1.ndim != 2:
                    return with_df.loop(blksize)
                if arr is None:
                    arr = numpy.empty((naux, eri1.shape[1]))
                arr[b0:b0 + eri1.shape[0]] = eri1
                b0 += eri1.shape[0]
            cache_owner._cderi_incore = (src, arr)
            log.info('CDERIS: AO vectors (%.0f MB) kept in memory for the following macro iterations', size_mb)
        return (arr[b0:b0 + blksize] for b0 in range(0, arr.shape[0], blksize))

    def _build_cd_pa_and_vk(self, with_df, mol, nao_nr, nmo, ncore, ncas, nocc,
                             moo_sph, mop_sph, log, cache_owner=None):
        """One-pass: half-transform occ MOs, produce cd_pa and vk_core simultaneously.

        Mirrors scf.spinor_hf._DFJHF._get_k_mo's real/imag-split K build: never
        materialise a complex (naux_blk, nocc, nao_nr) tensor (ncore is usually
        >> ncas, and core rows are only ever needed for K) -- the old code
        complex-materialised the full nocc block and made an extra .copy() of
        each half-transform buffer, which dominated both memory and wall time.
        blksize is with_df.blockdim, same as before this rewrite.

        For each DF chunk:
        1. Half-transform with occ MOs -> real L_{P,i,mu} (alpha/beta x real/imag)
        2. Core rows (i < ncore): K assembly directly from the real/imag split,
           via real GEMMs only -- accumulate vk_sph.
        3. Active rows (i >= ncore): complex-materialise just this small slice,
           then one batched matmul -> cd_pa.
        """
        import ctypes
        from pyscf.ao2mo import _ao2mo
        from socutils.scf.spinor_hf import sph2spinor

        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
        fmmm   = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
        fdrv   = _ao2mo.libao2mo.AO2MOnr_e2_drv
        null   = lib.c_null_ptr()

        # Split occ MOs (sph GHF) into alpha/beta, real/imag
        C_aR = numpy.asfortranarray(moo_sph[:nao_nr].real)
        C_aI = numpy.asfortranarray(moo_sph[:nao_nr].imag)
        C_bR = numpy.asfortranarray(moo_sph[nao_nr:].real)
        C_bI = numpy.asfortranarray(moo_sph[nao_nr:].imag)

        def _half(eri1, C_real, buf, i0, i1):
            naux_ = eri1.shape[0]
            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 C_real.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux_), ctypes.c_int(nao_nr),
                 (ctypes.c_int * 4)(i0, i1, 0, nao_nr),
                 null, ctypes.c_int(0))

        naux = with_df.get_naoaux()
        cd_pa = numpy.zeros((naux, nmo, ncas), dtype=complex)
        # the same vectors with the active index in front, (P, u, p): the layout
        # the second transformation step produces, and the one in which the
        # active exchange matrix is a single matrix product
        cd_up = numpy.zeros((naux, ncas, nmo), dtype=complex)
        mop_sph_conj = numpy.ascontiguousarray(mop_sph.conj())
        vk_sph = numpy.zeros((2 * nao_nr, 2 * nao_nr), dtype=complex)
        # core Coulomb matrix in the same pass: J = sum_P rho_P L_P with
        # rho_P = sum_i L_P,ii, accumulated in the packed (s2) AO layout of eri1
        npair = nao_nr * (nao_nr + 1) // 2
        vj_tril = numpy.zeros(npair)
        CcaR = numpy.ascontiguousarray(C_aR[:, :ncore].T).ravel()      # (ncore*nao_nr,)
        CcaI = numpy.ascontiguousarray(C_aI[:, :ncore].T).ravel()
        CcbR = numpy.ascontiguousarray(C_bR[:, :ncore].T).ravel()
        CcbI = numpy.ascontiguousarray(C_bI[:, :ncore].T).ravel()
        # J^{tu}_{mu nu} = sum_P L_P(mu nu) L_P,tu for every active pair, also in
        # this pass: the Coulomb matrix of any active density is then
        # sum_tu J^{tu} D_ut without reading the AO vectors again
        jtu_tril = numpy.zeros((npair, ncas * ncas), dtype=complex)

        blksize = with_df.blockdim
        # core and active orbitals are half-transformed into separate buffers:
        # slicing the core rows out of a (P, nocc, mu) buffer is not contiguous
        # and the copy it forces costs more than the matrix products of K
        nbuf = min(blksize, naux)
        cbuf = [numpy.empty((nbuf * ncore, nao_nr)) for _ in range(4)]
        abuf = [numpy.empty((nbuf * ncas, nao_nr)) for _ in range(4)]

        log.info('CDERIS before loop: current memory %.1f MB, blksize=%d',
                 lib.current_memory()[0], blksize)
        # Per-phase cumulative (cpu, wall) seconds, tallied across all blocks and
        # reported once after the loop -- splits the "CD integral transformation"
        # total into half-transform (C kernel, untouched) vs core-K assembly vs
        # cd_pa construction (the two this rewrite changed), so a slow run shows
        # which phase to look at instead of one opaque number.
        # (three separate accumulators: a chained assignment made them one array
        # and the three phases always printed the same number)
        t_half, t_corek, t_cdpa = numpy.zeros(2), numpy.zeros(2), numpy.zeros(2)
        b0 = 0
        # half-transform buffers, the complex active block and the two copies of cd_pa
        work_mb = (4 * nbuf * nocc * nao_nr * 8 + nbuf * ncas * 2 * nao_nr * 16
                   + 2 * naux * nmo * ncas * 16 + npair * ncas * ncas * 16) / 1e6
        for eri1 in self._cderi_blocks(with_df, blksize, work_mb, cache_owner, log):
            tb0 = (logger.process_clock(), logger.perf_counter())
            naux_blk = eri1.shape[0]
            # Half-transform: alpha/beta x real/imag, real arrays only
            LaR_c, LaI_c, LbR_c, LbI_c = [x[:naux_blk * ncore] for x in cbuf]
            LaR_a, LaI_a, LbR_a, LbI_a = [x[:naux_blk * ncas] for x in abuf]
            for C_real, core, act in ((C_aR, LaR_c, LaR_a), (C_aI, LaI_c, LaI_a),
                                      (C_bR, LbR_c, LbR_a), (C_bI, LbI_c, LbI_a)):
                if ncore > 0:
                    _half(eri1, C_real, core, 0, ncore)
                _half(eri1, C_real, act, ncore, nocc)
            tb1 = (logger.process_clock(), logger.perf_counter())
            t_half += (tb1[0] - tb0[0], tb1[1] - tb0[1])

            # --- Core J: rho_P = sum_i C_i^dagger L_P C_i (real), J += rho_P L_P ---
            if eri1.ndim == 2 and vj_tril is not None:
                if ncore > 0:
                    rho = (lib.dot(LaR_c.reshape(naux_blk, -1), CcaR[:, None])
                           + lib.dot(LaI_c.reshape(naux_blk, -1), CcaI[:, None])
                           + lib.dot(LbR_c.reshape(naux_blk, -1), CcbR[:, None])
                           + lib.dot(LbI_c.reshape(naux_blk, -1), CcbI[:, None]))
                    vj_tril += lib.dot(rho.T, numpy.asarray(eri1)).ravel()
            else:
                vj_tril = None

            # --- Core K assembly (i < ncore), real GEMMs only ---
            # K[mu,nu] = sum_{Pi} L[Pi,mu] L[Pi,nu]* with L = LR + i*LI splits
            # into a real symmetric part (LR.T@LR + LI.T@LI) and an imaginary
            # antisymmetric part (LI.T@LR - (LI.T@LR).T); this never forms the
            # complex L itself for the (naux_blk, ncore, nao_nr) core block.
            if ncore > 0:
                IRa = lib.dot(LaI_c.T, LaR_c)
                vk_sph[:nao_nr, :nao_nr] += (lib.dot(LaR_c.T, LaR_c) + lib.dot(LaI_c.T, LaI_c)
                                              + 1j * (IRa - IRa.T))
                IRb = lib.dot(LbI_c.T, LbR_c)
                vk_sph[nao_nr:, nao_nr:] += (lib.dot(LbR_c.T, LbR_c) + lib.dot(LbI_c.T, LbI_c)
                                              + 1j * (IRb - IRb.T))
                K_ab = (lib.dot(LaR_c.T, LbR_c) + lib.dot(LaI_c.T, LbI_c)
                        + 1j * (lib.dot(LaI_c.T, LbR_c) - lib.dot(LaR_c.T, LbI_c)))
                vk_sph[:nao_nr, nao_nr:] += K_ab
                vk_sph[nao_nr:, :nao_nr] += K_ab.conj().T
            tb2 = (logger.process_clock(), logger.perf_counter())
            t_corek += (tb2[0] - tb1[0], tb2[1] - tb1[1])

            # --- cd_pa: active rows only, complex-materialised (ncas << ncore) ---
            # one product over all (P, u) rows: X[(P,u), p] = sum_mu L[P,u,mu] C*[mu,p]
            L_act = numpy.empty((naux_blk * ncas, 2 * nao_nr), dtype=complex)
            L_act[:, :nao_nr].real = LaR_a
            L_act[:, :nao_nr].imag = LaI_a
            L_act[:, nao_nr:].real = LbR_a
            L_act[:, nao_nr:].imag = LbI_a
            x_up = lib.dot(L_act, mop_sph_conj).reshape(naux_blk, ncas, nmo)
            cd_up[b0:b0 + naux_blk] = x_up
            cd_pa[b0:b0 + naux_blk] = x_up.transpose(0, 2, 1)
            if vj_tril is not None:
                laa = numpy.ascontiguousarray(x_up[:, :, ncore:nocc].transpose(0, 2, 1)).reshape(naux_blk, ncas * ncas)
                e1 = numpy.asarray(eri1)
                jtu_tril += lib.dot(e1.T, laa.real) + 1j * lib.dot(e1.T, laa.imag)
            tb3 = (logger.process_clock(), logger.perf_counter())
            t_cdpa += (tb3[0] - tb2[0], tb3[1] - tb2[1])

            b0 += naux_blk
            del L_act, x_up
            if b0 % (blksize * 5) == 0 or b0 >= naux - blksize:
                log.info('  chunk b0=%d, memory %.1f MB', b0, lib.current_memory()[0])

        import gc
        gc.collect()
        vk_core = sph2spinor(mol, vk_sph)
        if vj_tril is not None:
            vj_ao = lib.unpack_tril(vj_tril)
            vj_sph = numpy.zeros((2 * nao_nr, 2 * nao_nr), dtype=complex)
            vj_sph[:nao_nr, :nao_nr] = vj_sph[nao_nr:, nao_nr:] = vj_ao
            self.vj_core = sph2spinor(mol, vj_sph)
            self.jtu_tril = jtu_tril
        self.cd_up = cd_up
        log.info('CDERIS after loop: current memory %.1f MB', lib.current_memory()[0])
        log.info('CDERIS: %d aux functions, cd_pa and vk_core built in one pass', naux)
        log.info('CDERIS phase breakdown: half-transform CPU %.1f wall %.1f sec, '
                 'core-K CPU %.1f wall %.1f sec, cd_pa CPU %.1f wall %.1f sec',
                 t_half[0], t_half[1], t_corek[0], t_corek[1], t_cdpa[0], t_cdpa[1])
        return cd_pa, vk_core

    def _build_cd_pa(self, with_df, cderi, mol, nao_nr, nmo, ncas,
                     moa_sph, mop_sph, log):
        """Build cd_pa using half-transform + second-step contraction.

        Step 1: L_{P,μ,t} = Σ_ν eri(P,μ,ν) C_{ν,t}  (C library)
        Step 2: cd_pa[P,p,t] = Σ_μ C*_{p,μ} L_{P,μ,t}  (numpy dot)
        """
        mop_sph_H = mop_sph.T.conj()  # (nmo, 2*nao_nr)

        if with_df is not None:
            # C library half-transform
            L_half = self._half_transform(with_df, moa_sph, nao_nr, log)
            naux = L_half.shape[0]
            log.info('CDERIS: %d aux functions from with_df', naux)

            # Step 2: full transform
            cd_pa = numpy.zeros((naux, nmo, ncas), dtype=complex)
            for i in range(naux):
                cd_pa[i] = numpy.dot(mop_sph_H, L_half[i])
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

        If vk_core is cached (from _build_cd_pa_and_vk), uses it directly.
        J always goes through _scf.get_jk (DM path).
        Otherwise falls back entirely to _scf.get_jk.
        """
        if self.vk_core is not None and mo_coeff is not None and mo_occ is not None:
            # J and K of the core density cached from the half-transform pass
            # (built during __init__); only valid for the core density of self.mo
            if getattr(self, 'vj_core', None) is not None:
                return self.vj_core, self.vk_core
            vj, _ = self._scf.get_jk(self._scf.mol, dm, with_j=True, with_k=False)
            return vj, self.vk_core
        else:
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
        # K_pq = sum_P sum_tu L_P,pt D_tu L_P,qu^*
        if getattr(self, 'cd_up', None) is not None:
            # (P, u, p) layout: one matrix product over the (P, u) rows
            tmp = numpy.matmul(numpy.asarray(casdm1).T, self.cd_up).reshape(-1, nmo)
            vk_a_mo = lib.dot(tmp.T, self.cd_up.reshape(-1, nmo).conj())
        else:
            tmp = lib.einsum('Ppt,tu->Ppu', self.cd_pa, casdm1)
            vk_a_mo = lib.einsum('Ppu,Pqu->pq', tmp, self.cd_pa.conj())

        if getattr(self, 'jtu_tril', None) is not None:
            # J_a = sum_tu J^{tu} D_ut with the pair operators accumulated during
            # the half-transform pass (real: L_P and D are Hermitian)
            from socutils.scf.spinor_hf import sph2spinor
            mol = self._scf.mol
            nao_nr = mol.nao_nr()
            ncas = nocc - ncore
            vj_tril = lib.dot(self.jtu_tril, numpy.asarray(casdm1).T.reshape(ncas * ncas, 1)).ravel().real
            vj_sph = numpy.zeros((2 * nao_nr, 2 * nao_nr), dtype=complex)
            vj_sph[:nao_nr, :nao_nr] = vj_sph[nao_nr:, nao_nr:] = lib.unpack_tril(vj_tril)
            vj_a = sph2spinor(mol, vj_sph)
        else:
            # J_a via DM path (J only)
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
