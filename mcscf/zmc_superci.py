# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2023/12/4
import sys
import numpy
import numpy as np
import time
from pyscf import lib, scf, gto, mcscf
from pyscf.lib import logger
from functools import reduce
#from scipy.linalg import expm as expmat
from scipy.sparse.linalg import gmres

#from .hf_superci import GMRES
from socutils.mcscf import zcahf, zcasci, zmcscf, zmc_ao2mo
from scipy.sparse.linalg import LinearOperator
import scipy
from numpy.linalg import norm
from socutils.mcscf.hf_superci import precondition_grad, postprocess_x

def expmat(x):
    expm = np.eye(x.shape[0], dtype=complex)
    xx = np.eye(x.shape[0], dtype=complex)
    for i in range(10):
        print('xx', i, np.linalg.norm(xx))
        xx = np.dot(xx, x)/(i+1)
        expm += xx
    return expm 

def form_kramers(mo_coeff):
    nao = mo_coeff.shape[0]//2
    # a, b for alpha and beta atomic orbitals
    # u, b for unbarred and barred spinors
    mo_au = mo_coeff[::2, ::2] #U
    mo_bu = mo_coeff[1::2, ::2] #-V^*
    mo_coeff[::2,1::2] = mo_bu.conj() #V
    mo_coeff[1::2,1::2] = -mo_au.conj() #U^*
    return mo_coeff

def ensure_kramers(mat):
    a  = mat[ ::2,  ::2].copy()
    at = mat[1::2, 1::2].copy()
    b  = mat[ ::2, 1::2].copy()
    bt = mat[1::2,  ::2].copy()
    out = numpy.zeros_like(mat)
    aa = (a + at.conj()) * 0.5
    out[ ::2, ::2] = aa
    out[1::2, 1::2] = aa.conj()
    bb = (b - bt.conj()) * 0.5
    print(norm(a), norm(at), norm(b), norm(bt))
    print(norm(a-at.T), norm(b+bt.T))
    out[ ::2, 1::2] = bb
    out[1::2,  ::2] = -bb.conj()
    print('ensure kramers', numpy.linalg.norm(mat - out), numpy.linalg.norm(mat))
    return out

def compute_lambda_(mat1, mat2, x_):
    nlast = mat1.shape[0]
    assert (nlast > 1)
    lambda_test = 1.0
    lambda_lasttest = 0.0
    stepsize_lasttest = 0.0
    stepsize = 0.0
    maxstepsize = 1.0
    iok = 0
    for i in range(10):
        scr = mat1 + mat2 * (1.0 / lambda_test)
        e, c = np.linalg.eigh(scr)

        ivec = -1
        for j in range(nlast):
            if abs(c[0, j]) <= 1.1 and abs(c[0, j]) > 0.1:
                ivec = j
                break
        if ivec < 0:
            raise Exception('logical error in AugHess')
        c[:, ivec] = c[:, ivec] / (c[0, ivec])
        step = np.dot(x_[1:, :nlast], c[:nlast, ivec])
        stepsize = np.linalg.norm(step[1:]) / abs(lambda_test)
        #print(ivec, e, stepsize, lambda_test)

        if i == 0:
            if stepsize <= maxstepsize:
                break
            lambda_lasttest = lambda_test
            lambda_test = stepsize / maxstepsize
        else:
            if abs(stepsize - maxstepsize) / maxstepsize < 0.01:
                break
            if (stepsize > maxstepsize):
                lambda_lasttest = lambda_test
                lambda_test = lambda_test * (stepsize / maxstepsize)
            else:
                if (iok > 2):
                    break
                iok += 1
                d1 = maxstepsize - stepsize
                d2 = stepsize_lasttest - maxstepsize
                if (d1 == 0.0 or d1 == -d2):
                    break
                lambda_lasttest_ = lambda_lasttest
                lambda_lasttest = lambda_test
                lambda_test = d1 / (d1 + d2) * lambda_lasttest_ + d2 / (d1 + d2) * lambda_test
            if lambda_test < 1.0:
                lambda_test = 1.0
            stepsize_lasttest = stepsize
    return lambda_test, stepsize


def davidson(hop, g, hdiag, tol=5e-6, neig=1, mmax=20):
    # Setup the subspace trial vectors
    print('No. of start vectors:', 1)
    neig = neig
    print('No. of desired Eigenvalues:', neig)
    n = g.shape[0] + 1
    x = np.zeros((n, mmax + 1), dtype=complex)  # holder for trial vectors as iterations progress
    sigma = np.zeros((n, mmax + 1), dtype=complex)
    ritz = np.zeros((n, n), dtype=complex)
    #-------------------------------------------------------------------------------
    # Begin iterations
    #-------------------------------------------------------------------------------
    #initial extra (1,0) vector
    start = time.time()
    x[0, 0] = 1.0
    sigma[1:, 0] = g

    # "first" guess vector

    for i in range(1, n):
        if hdiag[i - 1] + 0.001 > 1.0e-8:
            x[i, 1] = -g[i - 1] / (hdiag[i - 1] + 0.001)
        else:
            x[i, 1] = -g[i - 1] / (hdiag[i - 1] + 0.001 - 1.0e-8)

    x[:, 1] = x[:, 1] / np.linalg.norm(x[:, 1])
    for m in range(1, mmax + 1):
        sigma[1:, m] = hop(x[1:, m])
        sigma[0, m] = np.dot(g.conj(), x[1:, m])
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        T = np.linalg.multi_dot((x[:, :m + 1].T.conj(), sigma[:, :m + 1]))
        #print(T)
        mat1 = numpy.zeros((m + 1, m + 1), dtype=complex)
        mat2 = numpy.zeros((m + 1, m + 1), dtype=complex)
        mat2[1:, 1:] = T[1:, 1:]
        mat1 = T - mat2
        lambda_, stepsize = compute_lambda_(mat1, mat2, x)
        T = mat1 + mat2 * (1.0 / lambda_)
        e, vects = np.linalg.eigh(T[:m + 1, :m + 1])
        for j in range(m + 1):
            if abs(vects[0, j]) <= 1.1 and abs(vects[0, j]) > 0.1:
                ivec = j
                break
        if ivec < 0:
            raise Exception('logical error in AugHess')
        elif ivec > 0:
            print('... the vector found in AugHess was not the lowest eigenvector ...')

        ritz[1:, 0] = np.dot(sigma[1:, :m + 1] - lambda_ * e[ivec] * x[1:, :m + 1], vects[:, ivec])
        err = np.linalg.norm(ritz[:, 0])
        print(f'iter {m}, ivec {ivec}, c[ivec], {vects[0,ivec]:6.2f}, eps={e[ivec]:6.2f}, res={err:8.6e},  lambda={lambda_:6.2f}, step={stepsize:8.4f}')
        if err < min(1e-3 * stepsize, tol) and m > 4:
            print('Davidson converged at iteration no.:', m - 1)
            end = time.time()
            print('Davidson time:', end - start)
            x_ = np.dot(x[:, :m + 1], vects[:, ivec])
            return x_[1:] / x_[0], e[ivec]
        elif m is mmax:
            print('Max iteration reached')
            x_ = np.dot(x[:, :m + 1], vects[:, ivec])
            return x_[1:] / x_[0], e[ivec]
        else:
            for idx in range(hdiag.shape[0]):
                denom = hdiag[idx] - e[ivec] * lambda_
                if abs(denom) > 1e-8:
                    ritz[idx + 1, 0] = -ritz[idx + 1, 0] / denom
                else:
                    ritz[idx + 1, 0] = -ritz[idx + 1, 0] / 1e-8
            # orthonormalize ritz vector
            for idx in range(m + 1):
                ritz[:, 0] = ritz[:, 0] - (np.dot(x[:, idx], ritz[:, 0]) * x[:, idx])  #/np.linalg.norm(x[:,idx])
            ritz[:, 0] = ritz[:, 0] / (np.linalg.norm(ritz[:, 0]))
            x[:, m + 1] = ritz[:, 0]

# note: the ncas, nelecas, ncore should all be counted as the number of spin orbitals
def gen_g_hop(casscf, mo, casdm1, casdm2, eris):
    if casscf.mo_coeff is None:
        casscf.mo_coeff = mo
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    ################# gradient #################
    dm_core = np.zeros((nmo, nmo), dtype=complex)
    dm_active = np.zeros((nmo, nmo), dtype=complex)
    idx = np.arange(ncore)
    dm_core[idx, idx] = 1
    dm_active[ncore:nocc, ncore:nocc] = casdm1
    dm1 = dm_core + dm_active
    h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
    vj_c, vk_c = casscf.get_jk(casscf.mol,
                                   reduce(np.dot, (mo, dm_core, mo.T.conj())))
    vj_a, vk_a = casscf.get_jk(casscf.mol,
                                   reduce(np.dot, (mo, dm_active, mo.T.conj())))
    vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
    vhf_a = reduce(np.dot, (mo.T.conj(), vj_a - vk_a, mo))
    vhf_ca = vhf_c + vhf_a
    g = np.zeros((nmo, nmo), dtype=complex)
    g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    g[:, ncore:nocc] = np.dot(
        h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)

    g_new = np.zeros((nmo, nmo), dtype=complex)
    #g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    #g[:, ncore:nocc] = np.dot(
    #h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)
    g_new[ncore:, :ncore] = h1e_mo[ncore:, :ncore] + vhf_ca[ncore:, :ncore]
    g_new[ncore:, ncore:nocc] = np.dot(
        h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)[ncore:,:]
    g_new[nocc:,ncore:nocc] = np.dot(h1e_mo[nocc:,ncore:nocc] + vhf_c[nocc:,ncore:nocc], casdm1)
    g_new[ncore:nocc,:ncore] = np.dot(casdm1, h1e_mo[ncore:nocc, :ncore] + vhf_c[ncore:nocc,:ncore])
    if isinstance(eris, zmc_ao2mo._ERIS):
        paaa = eris.paaa
        g_dm2 = lib.einsum('puvw,tuvw->pt', paaa, casdm2)
    elif isinstance(eris, zmc_ao2mo._CDERIS):
        cd_pa = eris.cd_pa
        cd_aa = eris.cd_aa
        tmp = lib.einsum('Lvw,tuvw->Ltu', cd_aa, casdm2)
        g_dm2 = lib.einsum('Lpu,Ltu->pt', cd_pa, tmp)
        del tmp
    else:
        raise('eris type not recognized')
    g[:, ncore:nocc] += g_dm2
    g_new[ncore:nocc,:ncore] += g_dm2.T[:,:ncore]
    g_new[nocc:,ncore:nocc] += g_dm2[nocc:,:]
    g_orb = casscf.pack_uniq_var(g - g.T.conj())
    #g_orb = casscf.pack_uniq_var(g_new-g_new.T.conj())

    fock_eff = h1e_mo + vhf_ca
    if (norm(g_orb)<0.00001):
        print('level shift start')
        print(fock_eff[ncore:nocc,ncore:nocc])
        # compute a dynamic level shift
        target = (fock_eff[nocc,nocc] + fock_eff[ncore-1,ncore-1])/2.0
        active_mean = np.trace(fock_eff[ncore:nocc, ncore:nocc]) / (nocc-ncore)
        print(active_mean, target, fock_eff[nocc,nocc], fock_eff[ncore-1,ncore-1])
        g[ncore:nocc, ncore:nocc] += np.eye(nocc-ncore) * (target-active_mean)
        fock_eff[ncore:nocc,ncore:nocc] += np.eye(nocc-ncore) * (target-active_mean)
        print(fock_eff[ncore:nocc,ncore:nocc])
    #g_orb = casscf.pack_uniq_var(g - g.T.conj())
    #g = g - g.T.conj()

    # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
    f_oo = fock_eff[:ncore, :ncore]
    f_vv = fock_eff[nocc:, nocc:]
    f_aa = fock_eff[ncore:nocc, ncore:nocc]
    # intermediate for hessian calculation
    # g = np.zeros((nmo, nmo), dtype=complex)
    # g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    # g[:, ncore:nocc] = np.dot(
    #     h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)
    # paaa = eris.paaa
    # g_dm2 = lib.einsum('puvw,tuvw->pt', paaa, casdm2)
    # g_tu = d_tu,vw F_vw - F_vw,D_vw,D_tu
    g_tu = lib.einsum('tuvw,vw->tu', casdm2, g[ncore:nocc,ncore:nocc]) - lib.einsum('tu,vw,vw->tu', casdm1, casdm1, g[ncore:nocc,ncore:nocc])

    y = lib.einsum('pu,qu->pq', (h1e_mo + vhf_c)[ncore:nocc, ncore:nocc], casdm1)
    h_diag = np.ones((nmo, nmo), dtype=complex)
    for v_idx in range(nocc,nmo):
        for i_idx in range(ncore):
            h_diag[v_idx, i_idx]=fock_eff[v_idx,v_idx]-fock_eff[i_idx, i_idx]

    for a_idx in range(ncore, nocc):
        for i_idx in range(ncore):
            d_tt = dm1[a_idx, a_idx]
            h_diag[a_idx, i_idx] = fock_eff[a_idx, a_idx] - (1.-d_tt)*fock_eff[i_idx, i_idx]\
                - y[a_idx-ncore, a_idx-ncore] - g_dm2[a_idx, a_idx-ncore]

    for v_idx in range(nocc, nmo):
        for a_idx in range(ncore, nocc):
            h_diag[v_idx, a_idx] = fock_eff[v_idx, v_idx] * dm1[a_idx, a_idx] - y[a_idx-ncore, a_idx-ncore] - g_dm2[a_idx, a_idx-ncore]

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x)
        # super-ci hessian
        sigma = np.zeros_like(x1)
        f_oo = fock_eff[:ncore, :ncore]
        f_vv = fock_eff[nocc:, nocc:]
        f_aa = fock_eff[ncore:nocc, ncore:nocc]

        f_ov = fock_eff[:ncore, nocc:]
        dm1_aa = dm1[ncore:nocc, ncore:nocc]

        n_tt = dm1_aa.diagonal()
        m_tt = 1. - dm1_aa.diagonal()
        n_tt_sqrt = np.sqrt(dm1_aa.diagonal())
        m_tt_sqrt = np.sqrt(1. - dm1_aa.diagonal())

        one = np.ones((nocc-ncore, nocc-ncore))
        scale = one - np.einsum('ij,j->ij', one, n_tt) - np.einsum('ij,i->ij', one, n_tt)

        # core-virtual block
        #sigma = x1 * (h_diag+h_diag.T)
        # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
        sigma[nocc:, :ncore] += \
            (lib.einsum('ab,bi->ai', f_vv, x1[nocc:, :ncore])\
            - lib.einsum('ji,aj->ai', f_oo, x1[nocc:, :ncore]))

        # term 2 h_ai,bu = -delta_ab*f_vi*D_vu
        sigma[nocc:, :ncore] -= lib.einsum(
            'vi,vu,au->ai', g[ncore:nocc, :ncore], dm1_aa,
            x1[nocc:, ncore:nocc])

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        sigma[nocc:, :ncore] += \
             lib.einsum('au,ui->ai', g[nocc:, ncore:nocc], x1[ncore:nocc, :ncore])\
            -lib.einsum('av,uv,ui->ai', g[nocc:, ncore:nocc], dm1_aa, x1[ncore:nocc, :ncore])

        # core-active block
        # term5 h_ti,uj =
        # f_ji * (D_ut - delta_tu)
        sigma[ncore:nocc, :ncore] += \
            lib.einsum('ut,ji,uj->ti', dm1_aa, f_oo, x1[ncore:nocc, :ncore])\
            - lib.einsum('ji,tj->ti', f_oo, x1[ncore:nocc, :ncore])
        # h_ti,uj += f_tu delta_ij - f_tv,d_uv,delta_ij
        sigma[ncore:nocc, :ncore] += \
            lib.einsum('tu,ui->ti', g[ncore:nocc,ncore:nocc] - g_tu, x1[ncore:nocc, :ncore])\
            -lib.einsum('tv,uv,ui->ti', g[ncore:nocc,ncore:nocc], dm1_aa, x1[ncore:nocc, :ncore])\
            -lib.einsum('tv,uv,ui->ti', dm1_aa, g[ncore:nocc,ncore:nocc], x1[ncore:nocc, :ncore])
        # term5 continued
        # g_tu = d_tu,vw F_vw - F_vw,D_vw,D_tu
        # + delta_ij(f_tu-(d_tu,vw-d_ut*d_vw)*f_vw-f_tv*d_uv
        # the last two term differs from molpro's expression since molpro
        # suppose a symmetrized form of 2rdm while we don't.

        #sigma[ncore:nocc, :ncore] +=\
        #    lib.einsum('t,u,tu,ui->ti', m_tt_sqrt, m_tt_sqrt, f_aa_scale-g_tu, x1[ncore:nocc,:ncore])\
        #    -lib.einsum('ij,tj->ti', f_oo, x1[ncore:nocc, :ncore])
        #sigma[ncore:nocc, :ncore] +=\
        #    (lib.einsum('tu,ui->ti',f_aa, x1[ncore:nocc, :ncore])
        #    - lib.einsum('tu,ui->ti', g_tu, x1[ncore:nocc, :ncore])
        #    - lib.einsum('ij,ti->tj',f_oo, x1[ncore:nocc, :ncore]))

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        # adjoint of term 3 h_ti,bj x_bj->sigma_ti
        sigma[ncore:nocc,:ncore] += \
            lib.einsum('au,ai->ui', g[nocc:, ncore:nocc], x1[nocc:,:ncore]) \
           -lib.einsum('av,uv,ai->ui', g[nocc:, ncore:nocc], dm1_aa, x1[nocc:,:ncore])

        # virtual-active block
        # adjoint of term2
        # h_bu,ai = -delta_ab * f_vi*D_vu
        sigma[nocc:,ncore:nocc] -= lib.einsum('vi,vu,ai->au', g[ncore:nocc, :ncore], dm1_aa, x1[nocc:,:ncore])
        # term4 h_ti,bu = 0

        #term 6 h_at,bu=delta_ab(d_tu,vw-d_tu*d_vw)f_vw+d_tu*f_ab
        sigma[nocc:,ncore:nocc] += x1[nocc:,ncore:nocc] * h_diag[nocc:,ncore:nocc]
        
        #sigma[nocc:, ncore:nocc] += lib.einsum('tu,ab,bu->at', dm1_aa, f_vv, x1[nocc:, ncore:nocc]) + lib.einsum('tu,au->at', g_tu, x1[nocc:, ncore:nocc])
        
        sigma_pack = casscf.pack_uniq_var(sigma)
        return sigma_pack

    n_uniq_var = g_orb.shape[0]
    hop = LinearOperator((n_uniq_var,n_uniq_var), matvec=h_op)
    def h_diag_inv(x):
        return x/(casscf.pack_uniq_var(h_diag+h_diag.T))
    precond = LinearOperator((n_uniq_var, n_uniq_var), h_diag_inv)
    print(g_orb)
    return g_orb, casscf.pack_uniq_var((h_diag+h_diag.T).real), hop, precond


def precondition_grad0(grad, xs, ys, rhos, bfgs_space=10):
    assert len(ys) <= bfgs_space, 'size of xs greater than bfgs space size'
    gbar = grad.copy()
    niter = len(xs) if bfgs_space > len(xs) else bfgs_space
    a = np.zeros(niter)
    for ii in range(niter):
        i = niter-ii-1
        a[i] = np.dot(xs[i].conj(), gbar).real / rhos[i]
        gbar = gbar - ys[i] * a[i]
        #print(f'precond_grad, {ii}, {i}, {np.linalg.norm(gbar):.4e}, {np.linalg.norm(xs[i]):.4f} {np.linalg.norm(ys[i]):.4f}, {rhos[i]:.4f}, {a[i]:.4e}, {np.dot(xs[i].conj(), gbar).real:.4e} ')
    print(f'{len(ys)} {len(xs)} {np.linalg.norm(gbar):.4e} {np.linalg.norm(gbar-grad):.4e} bfgs precond')
    return gbar, a


def postprocess_x0(xbar, xs, ys, rhos, a, bfgs_space=10):
    assert len(xs) <= bfgs_space, 'size of xs greater than bfgs space size'
    niter = len(xs) if bfgs_space > len(xs) else bfgs_space
    xorig = xbar.copy()
    for i in range(niter):
        b = np.dot(ys[i].conj(), xbar).real / rhos[i]
        #print(f'postprocess {a[i]:.4e}, {b:.4e}, {np.linalg.norm(xs[i]):.4f}')
        xbar = xbar - xs[i] * (a[i] - b)
    print(f'bfgs post {np.linalg.norm(xorig-xbar):.4e}, {np.linalg.norm(xorig):.4e}')
    return 0.5*xbar


def mcscf_superci(mc, mo_coeff, max_stepsize=0.5, conv_tol=None,
                  conv_tol_grad=None, verbose=5, cderi=None):
    bfgs = False
    log = logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol is None:
        conv_tol = mc.conv_tol
    mol = mc.mol
    #if mc.irrep is None:
    #    mo = form_kramers(mo_coeff)
    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas

    mci = mc.view(zcasci.CASCI)
    if cderi is None:
        cderi = zmc_ao2mo.chunked_cholesky(mol)
    #eris = zmc_ao2mo._ERIS(mc, mo, level=2)
    eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    print('first CI calculation')
    e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
    mc.e_tot, mc.e_cas = e_tot, e_cas
    #mc._finalize()
    #e_tot, e_cas, fcivec = mc.casci(mo, ci0=None, eris=eris)
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mc, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot

    t1m = log.timer('Initializing Super-CI based MCSCF', *cput0)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)

    norm_rot = 0.0
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)

    imacro = 0
    xs = []
    ys = []
    rhos = []
    g_prev = None
    x_prev = None
    rejected = False
    trust_radii = 1.0
    e_last = e_tot
    dr = None
    while not conv and imacro < mc.max_cycle_macro:
        # compute natural orbital and transform ci to natural orbtial basis
        # no transform function available now so re do a ci calculation
        # do it in gen_g_hop

        moa = mo[:, ncore:nocc]
        natocc, c = mc._eig(casdm1)
        #print(casdm1)
        #print(natocc)
        moa_new = np.dot(moa, c)
        mo[:, ncore:nocc] = moa_new

        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi)
        t2m = log.timer('update eris', *t2m)
        mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
        print('update eris', verbose)
        e_tot, e_cas, fcivec = mci.kernel(mo, ci0=None, verbose=verbose)
        casdm1, casdm2 = mci.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)

        g, h_diag, hop, precond = gen_g_hop(mc, mo, casdm1, casdm2, eris)
        norm_gorb = norm(g)
        print(f'Iter {imacro:3d}: E = {e_tot:20.15f}  dE = {de:12.10f}' +
              f'  norm(grad) = {norm_gorb:8.6f} step_size = {norm_rot:8.6f}')
        t2m = log.timer('Compute gradient', *t2m)
        norm_gorb = np.linalg.norm(g)
        g_unpack = mc.unpack_uniq_var(g)

        g_unpack = mc.unpack_uniq_var(g)
        row, col = np.unravel_index(np.argmax(g_unpack), g_unpack.shape)
        for i in range(nmo):
            for j in range(i):
                if abs(g_unpack[i,j])>1e-3:
                    print(f'{i}, {j}, {g_unpack[i,j]}')
        if mc.irrep is not None:
            print(
                f'{row}, {col}, {g_unpack[row,col].real:8.6f} {mc.irrep[row]}, {mc.irrep[col]}'
            )
        else:
            print(
                f'{row}, {col}, {g_unpack[row,col].real:8.6f}'
            )
        if abs(de) < conv_tol or norm_gorb < conv_tol_grad:
            conv = True
        if conv:
            break

        print(f'scale gradient by {trust_radii:.4f}')
        gbar = g

        class gmres_counter(object):

            def __init__(self, disp=True):
                self._disp = disp
                self.niter = 0
                self.callbacks = []
                self.timer = (logger.process_clock(), logger.perf_counter())
                self.log = logger.Logger(sys.stdout, verbose)

            def __call__(self, rk=None):
                self.callbacks.append(str(rk))
                self.niter += 1
                # if self._disp:
                self.timer = self.log.timer(
                    f'GMRES iteration #{self.niter}, residual : {rk:.4e}',
                    *self.timer)

        #if verbose >= logger.INFO:
        #    counter = gmres_counter()
        #else:
        #    counter = None
        counter = gmres_counter()
        print(counter)
        t_gmres = (logger.process_clock(), logger.perf_counter())
        BFGS_SUBSPACE = 6
        if imacro > 0:
            bfgs_on = 0.001
            if not rejected and norm_gorb < bfgs_on:
                ys.append(g - g_prev)
                xs.append(x_prev)
                rhos.append(np.dot(ys[-1].conj(), xs[-1]).real)
            if len(ys) > BFGS_SUBSPACE:
                ys.pop(0)
                xs.pop(0)
                rhos.pop(0)
            if (np.linalg.norm(g_prev) < norm_gorb):
                print('gradeint norm', np.linalg.norm(g_prev), norm_gorb)
                xs = []
                ys = []
                rhos = []
            if bfgs is True and norm_gorb < bfgs_on:
                gbar, a = precondition_grad(g, xs, ys, rhos, bfgs_space=BFGS_SUBSPACE)

            residuals = []
            def callback(rk):
                residuals.append(rk)
            #x, _ = gmres(hop, -gbar, M=precond, maxiter=50, callback=callback)
            x, e = davidson(hop, gbar, h_diag)
            #print('residuals', _, residuals)
            if bfgs is True and norm_gorb < bfgs_on:
                x = 0.5*postprocess_x(x, xs, ys, rhos, a, bfgs_space=BFGS_SUBSPACE)
        else:
            residuals = []
            def callback(rk):
                residuals.append(rk)
            #x, _ = gmres(hop, -trust_radii*gbar, M=precond, maxiter=1000, callback=callback)
            x, e = davidson(hop, gbar, h_diag)
            print('residuals', residuals)
        t2m = log.timer('Solving Super-CI equation', *t_gmres)

        dr = mc.unpack_uniq_var(x)
        step_control = max_stepsize
        #if norm(dr) > step_control:
        #    print('step rescaled')
        #    dr = dr * (step_control / norm(dr))
        #    print(norm(dr))
        #    exit()
        rotation = expmat(dr)

        norm_rot = np.linalg.norm(rotation-np.eye(nmo, dtype=complex))
        g_unpack = mc.unpack_uniq_var(g)

        mo_new = np.dot(mo, rotation)
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo_new, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi)
        t2m = log.timer('update eris', *t2m)
        mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        print('update eris2')
        e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)

        # trus radius control
        #g_new, h_diag, hop, precondition = gen_g_hop(mc, mo_new, casdm1, casdm2, eris)
        #dg = norm(g_new) - norm(g)
        de = e_tot - e_last  #+ dg * .1
        e2 = 0.5 * np.dot(x.T.conj(), g)
        r = de / e2

        print(f'Energy change {de:.4e}, predicted change {e2:.4e}')
        #while(True):
        #    if de < 0.0:
        #        print('energy lowered, exit iteration')
        #        break
        #    elif (abs(r) < 2.0):
        #        print('normal step')
        #        break
        #    dr = 0.5*dr
        #    rotation = expmat(dr)
        #    mo_new = np.dot(mo, rotation)
        #    eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi)
        #    mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        #    e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)
        #    de = e_tot - e_last
        #    e2 = 0.5 * np.dot(x.T.conj(), g)
        #    r = de / e2
        #    print(f'Energy change {de:.4e}, predicted change {e2:.4e}')
        if False: #r < -10 and de > 0.0:
            trust_radii *= 0.7
            print(dr)
            print(f'step rejected, shrink trust radius to {trust_radii:.4f}')
            rejected=True
            new_rot = expmat(0.01*dr)
            mo_new = np.dot(mo, new_rot)
            eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi)
            t2m = log.timer('update eris', *t2m)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
            print('step size', np.linalg.norm(dr), np.linalg.norm(rotation))
            #print(rotation[np.where(abs(rotation) > 1e-4)])
            #for dri in dr:
            #    print(dri)
            #for roti in rotation:
            #    print(roti)
            print('update eris2')
            e_tot, e_cas, fcivec = mci.kernel(mo, ci0=None, verbose=verbose)
            casdm1, casdm2 = mci.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
            de = e_tot-e_last
            e_last = e_tot
            #continue
        elif de > 0.0:
            trust_radii *= 0.7
            print(f'Energy rises by {de:.4e}, accept step but shrik trust radius to {trust_radii:.4f}')
        elif r < 0.25: #and de < 0.0:
            trust_radii *= 0.7
            print(f'r value too small, shrink the trust radius {trust_radii:.4f}')
        elif r > 0.75 and de < 0.0:
            trust_radii = min(1.4*trust_radii, 1.0)
            print(f'r value pretty large, we can uplift the trust radius {trust_radii:.4f}')
        else:
            print(f'normal step, trust radius now is {trust_radii:.4f}')
        if trust_radii < 1e-2*max_stepsize:
            trust_radii = 1e-2*max_stepsize
            rejected=False
        #'''
        #print(trust_radii)
        #dr[::2,::2] = dr[1::2,1::2]
        #dr[::2,1::2] = dr[1::2,::2]

        rotation = expmat(dr)
        norm_rot = np.linalg.norm(dr)
        nvar = rotation.shape[0]
        for i in range(nvar):
            if abs(rotation[i, i]) > 1.01 or abs(rotation[i, i]) < 0.99:
                print(rotation[i, i] > 1.01, rotation[i, i] < 0.99, i, j,
                      rotation[i, i])
            for j in range(i):
                if abs(rotation[i, j]) > 0.01:
                    continue

        rejected = False
        mo = mo_new
        e_last = e_tot
        x_prev = x
        g_prev = g
        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        nact = casdm1.shape[0]
        #for i in range(nact):
        #    for j in range(i):
        #        if abs(casdm1[i, j]) > 1e-5:
        #            print(f'{i}, {j}, {casdm1[i,j]}')
        imacro += 1
        t1m = log.timer(f'macro iter {imacro}', *t1m)
        if verbose >= logger.INFO:
            mc.e_tot = e_tot
            mc.e_cas = e_cas
            mc._finalize()
    mc.mo_coeff = mo
    return conv, e_tot, e_cas, fcivec, mo, None


if __name__ == '__main__':

    mol = gto.M(atom='''
C -0.600  0.000  0.000
C  0.600  0.000  0.000
H   -1.4523499293        0.8996235720         .0000000000
H   -1.4523499293       -0.8996235720         .0000000000
H    1.4523499293        0.8996235720         .0000000000
H    1.4523499293       -0.8996235720         .0000000000
''', basis='ccpvdz', verbose=4, charge=0, max_memory=40000, nucmod='G')

    from socutils.scf import spinor_hf, x2camf_hf
    from pyscf.x2c import x2c
    mf = x2c.RHF(mol)

    #mf.with_x2c = x2camf_hf.SpinorX2CAMFHelper(mol)
    mf.max_cycle=50
    mf.kernel()
    print(mf.mo_coeff[:,0], mf.mo_coeff[:,1])
    for ene, occ in zip(mf.mo_energy, mf.mo_occ):
        print(f'{ene:20.15f} {occ:8.4g}')

    mf.mol.charge=0
    mf.mol.build()
    mc = zmcscf.CASSCF(mf, ncas=6, nelecas=4)
    #mc = mc.state_average_(numpy.ones(9)/9.)
    mc.superci()
    print(mc.e_tot)
