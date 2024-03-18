# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2023/12/28
# CASSCF just uses diagonal hessian

import numpy
import numpy as np
import time
from pyscf import lib, scf, gto, mcscf
from pyscf.lib import logger
import zcahf
from functools import reduce
from scipy.linalg import expm as expmat
import zcasci
import zmcscf
import zmc_ao2mo


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
    
    paaa = eris.paaa
    g_dm2 = np.einsum('puvw,tuvw->pt', paaa, casdm2)
    g[:, ncore:nocc] += g_dm2
    g_orb = casscf.pack_uniq_var(g - g.T.conj())

    fock_eff = h1e_mo + vhf_ca

    #for i in range(ncore-10,nocc+10):
    #    print(fock_eff[i,i])
    # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
    f_oo = fock_eff[:ncore, :ncore]
    f_vv = fock_eff[nocc:, nocc:]
    f_aa = fock_eff[ncore:nocc, ncore:nocc]

    y = np.einsum('pu,qu->pq', (h1e_mo + vhf_c)[ncore:nocc, ncore:nocc], casdm1)
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

    h_diag = 0.5 * casscf.pack_uniq_var(h_diag+h_diag.T)

    return g_orb, h_diag


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


def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = casscf.view(gzcasci.CASCI)
    mc.mo_coeff = mo

    if eris is None:
        return mc

    mc.get_h2eff = lambda *args: eris.aaaa
    return mc

def mcscf_diaghess(mc, mo_coeff, conv_tol=1e-10, conv_tol_grad=None,
                  verbose=logger.INFO, cderi=None):
    log = logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start Spinor MCSCF Using Diagonal Hessian approximation')
    mol = mc.mol
    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    
    mci = mc.view(gzcasci.CASCI)
    if cderi is None:
        cderi = zmc_ao2mo.chunked_cholesky(mol)
    #eris = zmc_ao2mo._ERIS(mc, mo, level=2)
    eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi)
    mci = _fake_h_for_fast_casci(mc, mo, eris)
    e_tot, e_cas, fcivec = mci.kernel(mo)
    #e_tot, e_cas, fcivec = mc.casci(mo, ci0=None, eris=eris)
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mc, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
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

    trust_radii = 0.4
    e_last = e_tot

    while not conv and imacro < mc.max_cycle_macro:
        g, h_diag = gen_g_hop(mc, mo, casdm1, casdm2, eris)
        norm_gorb = np.linalg.norm(g)
        if abs(de) < conv_tol and norm_gorb < conv_tol_grad:
            conv = True
        if conv:
            break

        gbar = g

        if imacro > 0:
            ys.append(g - g_prev)
            xs.append(x_prev)
            rhos.append(np.dot(ys[-1].conj(), xs[-1]))
            bfgs_space = 10
            if len(ys) > bfgs_space:
                ys.pop(0)
                xs.pop(0)
                rhos.pop(0)
            gbar, a = precondition_grad(gbar, xs, ys, rhos, bfgs_space)

            x = -trust_radii*gbar / h_diag
            x = postprocess_x(x, xs, ys, rhos, a, bfgs_space)
        
        else:
            x = -gbar / h_diag


        dr = mc.unpack_uniq_var(x)

        #if np.linalg.norm(dr) > trust_radii:
        #    print(f'Step size rescaled from {np.linalg.norm(dr):.4f}')
        #    dr *= trust_radii/np.linalg.norm(dr)

        mo_new = np.dot(mo, expmat(dr))
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi)
        mci = _fake_h_for_fast_casci(mc, mo_new, eris)
        e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None)

        # trus radius control
        de = e_tot - e_last
        e2 = 0.5 * np.dot(x.T.conj(), g)
        r = de/e2
        print(de, e2)
        #if de > 0.0 and trust_radii > 0.01:
        #    trust_radii *= 0.7
        #    print(f'step rejected, shrink trust radius to {trust_radii:.4f}')
        #    continue
        if r < 0.25:
            trust_radii *= 0.7
            print(f'r value too small, shrink the trust radius')
        elif r > 0.75:
            trust_radii *= 1.2
            print(f'r value pretty large, we can uplift the trust radius')

        print('step accepted')
        mo = mo_new
        e_last = e_tot        
        x_prev = x
        g_prev = g

        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        print(
            f'Iter {imacro:3d}: E = {e_tot:20.15f}  dE = {de:12.10f}' +
            f'  norm(grad) = {norm_gorb:8.6f} norm(dr) = {np.linalg.norm(dr):8.6f}'
        )
        imacro += 1

    mc.mo_coeff = mo
    return conv, e_tot


if __name__ == '__main__':
    
    mol = gto.M(atom='Nd 0 0 0', basis='ccpvdzdk', verbose=4, charge=6, max_memory=40000)

    from pyscf.socutils import spinor_hf, x2camf_hf
    from pyscf.x2c import x2c
    mf = spinor_hf.SpinorSCF(mol)

    #mf.with_x2c = x2camf_hf.SpinorX2CAMFHelper(mol)
    mf.with_x2c = x2c.SpinorX2CHelper(mol)
    #mf.max_cycle=
    mf.chkfile='nd.chk'
    mf.max_cycle=100
    lib.num_threads(12)
    mf.kernel()
    for ene, occ in zip(mf.mo_energy, mf.mo_occ):
        print(f'{ene:20.15f} {occ:8.4g}')
    #mf.analyze()
    mol.charge=3
    mol.spin=1
    mol.build()
    mc = zmcscf.CASSCF(mf, 14,3)
    mc.fcisolver = zcahf.CAHF(mol)
    mcscf_diaghess(mc, mf.mo_coeff, verbose=5)
