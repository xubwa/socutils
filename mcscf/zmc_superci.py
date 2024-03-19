# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2023/12/4

import numpy
import numpy as np
import time
from pyscf import lib, scf, gto, mcscf
from pyscf.lib import logger
from functools import reduce
from scipy.linalg import expm as expmat
from .hf_superci import GMRES
from . import zcahf, zcasci, zmcscf, zmc_ao2mo

from numpy.linalg import norm

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
        if err < max(0.001 * stepsize, tol) and m > 4:
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

#
#    # part5
#    jkcaa = numpy.empty((nocc, ncas))
#    # part2, part3
#    vhf_a = numpy.empty((nmo, nmo))
#    # part1 ~ (J + 2K)
#    dm2tmp = casdm2.transpose(1, 2, 0, 3) + casdm2.transpose(0, 2, 1, 3)
#    dm2tmp = dm2tmp.reshape(ncas**2, -1)
#    hdm2 = numpy.empty((nmo, ncas, nmo, ncas))
#    g_dm2 = numpy.empty((nmo, ncas))
#    for i in range(nmo):
#        jbuf = eris.ppaa[i]
#        kbuf = eris.papa[i]
#        if i < nocc:
#            jkcaa[i] = numpy.einsum('ik,ik->i', 6 * kbuf[:, i] - 2 * jbuf[i],
#                                    casdm1)
#        vhf_a[i] = (numpy.einsum('quv,uv->q', jbuf, casdm1) -
#                    numpy.einsum('uqv,uv->q', kbuf, casdm1) * .5)
#        jtmp = lib.dot(jbuf.reshape(nmo, -1), casdm2.reshape(ncas * ncas, -1))
#        jtmp = jtmp.reshape(nmo, ncas, ncas)
#        ktmp = lib.dot(kbuf.transpose(1, 0, 2).reshape(nmo, -1), dm2tmp)
#        hdm2[i] = (ktmp.reshape(nmo, ncas, ncas) + jtmp).transpose(1, 0, 2)
#        g_dm2[i] = numpy.einsum('uuv->v', jtmp[ncore:nocc])
#    jbuf = kbuf = jtmp = ktmp = dm2tmp = None
#    vhf_ca = eris.vhf_c + vhf_a
#    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))

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
    
    if isinstance(eris, zmc_ao2mo._ERIS):
        paaa = eris.paaa
        g_dm2 = lib.einsum('puvw,tuvw->pt', paaa, casdm2)
    elif isinstance(eris, zmc_ao2mo._CDERIS):
        cd_pa = eris.cd_pa
        cd_aa = eris.cd_aa
        tmp = lib.einsum('Lvw,tuvw->Ltu', cd_aa, casdm2)
        g_dm2 = lib.einsum('Lpt,Ltu->pt', cd_pa, tmp)
        del tmp
    else:
        raise('eris type not recognized')
    g[:, ncore:nocc] += g_dm2
    g_orb = casscf.pack_uniq_var(g - g.T.conj())

    fock_eff = h1e_mo + vhf_ca

    # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
    f_oo = fock_eff[:ncore, :ncore]
    f_vv = fock_eff[nocc:, nocc:]
    f_aa = fock_eff[ncore:nocc, ncore:nocc]
    
    '''
    h_diag = np.ones((nmo, nmo), dtype=complex)
    for v_idx in range(nocc,nmo):
        for i_idx in range(ncore):
            h_diag[v_idx, i_idx]=fock_eff[v_idx,v_idx]-fock_eff[i_idx, i_idx]
    #h_diag[ncore:nocc,:]=1.0
    # term5 h_ti,uj =
    # f_ji * (D_ut - delta_tu)
    inter1 = np.einsum('tuvw,vw->tu',casdm2,f_aa)-casdm1*np.einsum('tu,vw,vw->tu',casdm1,casdm1,f_aa)
    inter2 = np.einsum('tv,vu->tu', f_aa,casdm1)
    for a_idx in range(ncore,nocc):
        for i_idx in range(ncore):
            h_diag[a_idx,i_idx] = fock_eff[a_idx, a_idx] - fock_eff[i_idx,i_idx]
            #fock_eff[i_idx,i_idx]*(dm1[a_idx,a_idx]-1.)\
            #        + fock_eff[a_idx, a_idx] - (inter1+inter2)[a_idx-ncore,a_idx-ncore]
    # term5 continued
    # + delta_ij(f_tu-(d_tu,vw-d_ut*d_vw)*f_vw-f_tv*d_vu)
    # the last two term differs from molpro's expression since molpro
    # suppose a symmetrized form of 2rdm while we don't.
    # term 6 h_at,bu=delta_ab(d_tu,vw-d_tu*d_vw)f_vw+d_tu*f_ab
    #term_6 = np.einsum
    for v_idx in range(nocc,nmo):
        for a_idx in range(ncore,nocc):
            h_diag[v_idx, a_idx]=fock_eff[v_idx,v_idx]-fock_eff[a_idx,a_idx]#dm1[a_idx,a_idx]*fock_eff[v_idx,v_idx]+inter1[a_idx-ncore,a_idx-ncore]
    #print('part 1', h_diag[nocc:,:ncore])
    # print('part 2', h_diag[nocc:,ncore:nocc])
    #h_diag[nocc:, ncore:nocc] = 1.0
    #print('part 3', h_diag[ncore:nocc,:ncore])
    h_diag = casscf.pack_uniq_var(h_diag)
    h_diag[h_diag < 1.0] = 1.0
    '''

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

    h_diag_compact = 0.5 * casscf.pack_uniq_var(h_diag+h_diag.T)

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x)
        # super-ci hessian
        sigma = np.zeros_like(x1)
        f_oo = fock_eff[:ncore, :ncore]
        f_vv = fock_eff[nocc:, nocc:]
        f_aa = fock_eff[ncore:nocc, ncore:nocc]
        f_ov = fock_eff[:ncore, nocc:]
        dm1_aa = dm1[ncore:nocc, ncore:nocc]

        sigma = x1 * (h_diag+h_diag.T)
        # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
        sigma[nocc:, :ncore] = \
            (lib.einsum('ab,bi->ai', f_vv, x1[nocc:, :ncore])\
            - lib.einsum('ji,aj->ai', f_oo, x1[nocc:, :ncore]))
        # term 2 h_ai,bu = -delta_ab*f_vi*D_vu
        sigma[nocc:, :ncore] += lib.einsum(
            'vi,vu,au->ai', fock_eff[ncore:nocc, :ncore], dm1_aa,
            x1[nocc:, ncore:nocc])

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        sigma[nocc:, :ncore] += \
            lib.einsum('au,ui->ai', fock_eff[nocc:, ncore:nocc], x1[ncore:nocc, :ncore])\
            -lib.einsum('av,uv,ui->ai', fock_eff[nocc:, ncore:nocc], dm1_aa, x1[ncore:nocc, :ncore])
        # term4 h_ti,bu = 0
        # term5 h_ti,uj =
        # f_ji * (D_ut - delta_tu)
        #sigma[ncore:nocc,:ncore] = x1[ncore:nocc, :ncore] * h_diag[ncore:nocc, :ncore]
        #sigma[ncore:nocc, :ncore] += \
        #    lib.einsum('ut,ji,uj->ti', dm1_aa, f_oo, x1[ncore:nocc, :ncore])\
        #    - lib.einsum('ji,tj->ti', f_oo, x1[ncore:nocc, :ncore])
        ## term5 continued
        # + delta_ij(f_tu-(d_tu,vw-d_ut*d_vw)*f_vw-f_tv*d_uv
        # the last two term differs from molpro's expression since molpro
        # suppose a symmetrized form of 2rdm while we don't.
        #sigma[ncore:nocc, :ncore] +=\
        #    (lib.einsum('tu,ui->ti',fock_eff[ncore:nocc, ncore:nocc], x1[ncore:nocc, :ncore])
        #    - lib.einsum('tuvw,vw,ui->ti', casdm2, f_aa, x1[ncore:nocc, :ncore])
        #    + lib.einsum('ut,vw,vw,ui->ti', dm1_aa, dm1_aa, f_aa, x1[ncore:nocc, :ncore])
        #    - lib.einsum('tv,uv,ui->ti', f_aa, dm1_aa, x1[ncore:nocc, :ncore]))
        ##term 6 h_at,bu=delta_ab(d_tu,vw-d_tu*d_vw)f_vw+d_tu*f_ab
        sigma[nocc:, ncore:nocc] = lib.einsum('tuvw,vw,au->at', casdm2, f_aa, x1[nocc:, ncore:nocc])\
            -lib.einsum('tu,vw,vw,au->at', dm1_aa, dm1_aa, f_aa, x1[nocc:, ncore:nocc])\
            +lib.einsum('tu,ab,bu->at', dm1_aa, f_vv, x1[nocc:, ncore:nocc])
        return casscf.pack_uniq_var(sigma)
    from scipy.sparse.linalg import LinearOperator
    n_uniq_var = g_orb.shape[0]
    hop = LinearOperator((n_uniq_var,n_uniq_var), matvec=h_op)
    def h_diag_inv(x):
        return x/(casscf.pack_uniq_var(h_diag+h_diag.T))        
    precond = LinearOperator((n_uniq_var, n_uniq_var), h_diag_inv) 
    return g_orb, casscf.pack_uniq_var(h_diag+h_diag.T), hop, precond


def precondition_grad(grad, xs, ys, rhos, bfgs_space=10):
    assert len(ys) <= bfgs_space, 'size of xs greater than bfgs space size'
    gbar = grad
    niter = len(xs) if bfgs_space > len(xs) else bfgs_space
    a = np.zeros(niter)
    for i in range(niter, -1):
        a[i] = rhos[i] * np.dot(xs[i].conj(), gbar)
        gbar = gbar - ys[i] * a[i]

    return gbar, a


def postprocess_x(xbar, xs, ys, rhos, a, bfgs_space=10):
    assert len(xs) <= bfgs_space, 'size of xs greater than bfgs space size'
    niter = len(xs) if bfgs_space > len(xs) else bfgs_space
    for i in range(niter):
        b = rhos[i] * np.dot(ys[i].conj(), xbar)
        xbar = xbar + xs[i] * (-a[i] - b)
    return xbar


def mcscf_superci(mc, mo_coeff, max_stepsize=0.2, conv_tol=None, conv_tol_grad=None,
                  verbose=logger.INFO, cderi=None):
    log = logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol is None:
        conv_tol = mc.conv_tol
    mol = mc.mol
    #mo = form_kramers(mo_coeff)
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
    e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
    mc.e_tot, mc.e_cas = e_tot, e_cas
    mc._finalize()
    #e_tot, e_cas, fcivec = mc.casci(mo, ci0=None, eris=eris)
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mc, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot

    t1m = log.timer('Initializing Super-CI based MCSCF', *cput0)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
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
    trust_radii = 0.7
    e_last = e_tot
    dr = None
    while not conv and imacro < mc.max_cycle_macro:
        #eris = mc.ao2mo(mo)
        g, h_diag, hop, precond = gen_g_hop(mc, mo, casdm1, casdm2, eris)
        norm_gorb = np.linalg.norm(g)
        print(
            f'Iter {imacro:3d}: E = {e_tot:20.15f}  dE = {de:12.10f}' +
            f'  norm(grad) = {norm_gorb:8.6f} '
        )
        #for h in h_diag:
        #    print(h)
        #h_diag=None
        if abs(de) < conv_tol and norm_gorb < conv_tol_grad:
            conv = True
        if conv:
            break

        gbar = g
        x0 = -gbar/h_diag
        from scipy.sparse.linalg import gmres, lgmres
        class gmres_counter(object):
            def __init__(self, disp=True):
                self._disp = disp
                self.niter = 0
                self.callbacks = []
            def __call__(self, rk=None):
                self.callbacks.append(str(rk))
                self.niter += 1
                if self._disp:
                    print('%s' %(str(rk)))
        #counter = gmres_counter()
        #x, _ = gmres(hop, -gbar, maxiter=20, callback=counter, M=precond)
        #x, _ = gmres(hop, -trust_radii*gbar, maxiter=20, M=precond)
        if imacro > 0:
            if not rejected:
                ys.append(g - g_prev)
                xs.append(x_prev)
                rhos.append(np.dot(ys[-1].conj(), xs[-1]))
            if len(ys) > 10:
                ys.pop(0)
                xs.pop(0)
                rhos.pop(0)
            print(rejected)
            print(lib.fp(ys[-1]), lib.fp(xs[-1]))
            gbar, a = precondition_grad(gbar, xs, ys, rhos)
            x, _ = gmres(hop, -trust_radii*gbar, M=precond, maxiter=20)
            #x, _ = davidson(hop, gbar, h_diag, max_iter=100)
            x = postprocess_x(x, xs, ys, rhos, a)
        else:
            x, _ = gmres(hop, -trust_radii*gbar, M=precond, maxiter=20)
        #print(_)
        #x, _ = GMRES(hop, -gbar, x0=x0)#h_diag)
        #x, _ = davidson(hop, gbar, h_diag)

        dr = mc.unpack_uniq_var(x)
        step_control=max_stepsize
        if norm(dr) > step_control:
            print('step rescaled')
            dr = dr*(step_control/norm(dr))
        rotation = expmat(dr)
        #rotation = ensure_kramers(rotation)
        #print(np.where(abs(rotation-1.0)>0.01), rotation[abs(rotation-1.0)>0.01])
        mo_new = np.dot(mo, rotation)
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo_new, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi)
        mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)

        # trus radius control
        de = e_tot - e_last
        e2 = 0.5 * np.dot(x.T.conj(), g)
        r = de/e2
        print(de, e2)
        if de > 1e-3:
            trust_radii *= 0.7
            print(f'step rejected, shrink trust radius to {trust_radii:.4f}')
            rejected=True
            continue
        if r < 0.25:
            trust_radii *= 0.8
            print(f'r value too small, shrink the trust radius')
        elif r > 0.75:
            if trust_radii < 1.0:
                trust_radii *= 1.25
            print(f'r value pretty large, we can uplift the trust radius')
        else:
            print(f'normal step')
        #if trust_radii < 1e-1:
        #    trust_radii = 1e-1
        #print(trust_radii)
        rotation = expmat(dr)
        nvar = rotation.shape[0]
        for i in range(nvar):
            if abs(rotation[i,i])>1.01 or abs(rotation[i,i]) < 0.99:
                print(rotation[i,i]>1.01, rotation[i,i]<0.99,i,j,rotation[i,i])
            for j in range(i):
                if abs(rotation[i,j])>0.01:
                    print(rotation[i,j], i,j)
        rejected=False
        mo = mo_new
        e_last = e_tot        
        x_prev = x
        g_prev = g

        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        '''
        step_control = 0.1#min(0.1, np.linalg.norm(g))
        if np.linalg.norm(dr) > step_control:
            #scale_step_size = 0.1 / np.linalg.norm(dr)
            #dr[abs(dr)>0.05] = 0.005
            print(f'Step size rescaled from {np.linalg.norm(dr)}')
            #dr[abs(dr)>1e-3]=1e-3# *= 0.1/np.linalg.norm(dr)
            dr *= step_control/np.linalg.norm(dr)
        #dr[abs(dr)<1e-9]=0.0
        mo = np.dot(mo, expmat(dr))
        e_last = e_tot
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi)
        mci = _fake_h_for_fast_casci(mc, mo, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo, ci0=None)
        de = e_tot - e_last
        '''
        #casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas, verbose=verbose)
        imacro += 1
        if verbose >= logger.INFO:
            mc.e_tot = e_tot
            mc.e_cas = e_cas
            mc._finalize()
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

    from pyscf.socutils import spinor_hf, x2camf_hf
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
