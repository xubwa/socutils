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
import numpy
import numpy as np
from scipy import linalg as la

def chunked_cholesky(mol, max_error=1e-6, verbose=True, cmax=100):
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
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
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
        start = time.time()
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
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]

# level = 1: aaaa
# level = 2: paaa
class _CDERIS(lib.StreamObject):
    def __init__(self, zcasscf, mo, level=2, cderi=None):
        import gc
        gc.collect()
        mol = zcasscf.mol
        nao, nmo = mo.shape
        ncore = zcasscf.ncore
        ncas = zcasscf.ncas
        nocc = ncore+ncas
        
        if cderi is None:
            print('cd from init')
            cderi = chunked_cholesky(mol)
        ncd = cderi.shape[0]
        cderi=cderi.reshape((ncd, mol.nao, mol.nao))
        moa = mo[:, ncore:nocc]
        c2 = numpy.vstack(mol.sph2spinor_coeff())

        moa_sph = numpy.dot(c2, moa)
        mop_sph = numpy.dot(c2, mo)
        if level is 2:
            cd_aa = np.zeros((ncd, ncas, ncas), dtype=complex)
            cd_pa = np.zeros((ncd, nmo,  ncas), dtype=complex)
            for i in range(ncd):
                chol_i = cderi[i].reshape(mol.nao, mol.nao)
                chol_i = la.block_diag(chol_i, chol_i)
                cd_pa[i] = reduce(np.dot, (mop_sph.T.conj(), chol_i, moa_sph))
            self.cd_pa = cd_pa
            self.cd_aa = cd_pa[:,ncore:nocc,:].copy()
        elif level > 2:
            cd_aa = np.zeros((ncd, ncas, ncas), dtype=complex)
            cd_pa = np.zeros((ncd, nmo,  ncas), dtype=complex)
            cd_pp = np.zeros((ncd, nmo,  nmo), dtype=complex)
            for i in range(ncd):
                print(i)
                chol_i = cderi[i].reshape(mol.nao, mol.nao)
                chol_i = la.block_diag(chol_i, chol_i)
                cd_pp[i] = reduce(np.dot, (mop_sph.T.conj(), chol_i, mop_sph))
            self.cd_cc = cd_pp[:,:ncore,:ncore].copy()
            self.cd_ca = cd_pp[:,:ncore,ncore:nocc].copy()
            self.cd_cv = cd_pp[:,:ncore,nocc:].copy()
            self.cd_ac = cd_pp[:,ncore:nocc,:ncore].copy()
            self.cd_aa = cd_pp[:,ncore:nocc,ncore:nocc].copy()
            self.cd_av = cd_pp[:,ncore:nocc,nocc:].copy()
            self.cd_vc = cd_pp[:,nocc:,:ncore].copy()
            self.cd_va = cd_pp[:,nocc:,ncore:nocc].copy()
            self.cd_vv = cd_pp[:,nocc:,nocc:].copy()
            self.cd_pa = cd_pp[:,:,ncore:nocc]
            print('cd integrals saved')
        self.aaaa = lib.einsum('ptu,pvw->tuvw', self.cd_aa, self.cd_aa) 
        #print('aaaa done')
        #self.paaa = lib.einsum('pqt,pvw->qtvw', cd_pa, self.cd_aa)
        #self.aaaa = self.paaa[ncore:nocc,:,:,:]

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
