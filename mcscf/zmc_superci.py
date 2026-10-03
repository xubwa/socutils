# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2023/12/4
import sys
import numpy
import numpy as np
import time
from pyscf import lib, scf, gto, mcscf
from pyscf.lib import logger
from functools import reduce
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

from scipy.linalg import expm as expmat

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
    out[ ::2, 1::2] = bb
    out[1::2,  ::2] = -bb.conj()
    print('ensure kramers: removed non-KR component %.3e of %.3e'
          % (numpy.linalg.norm(mat - out), numpy.linalg.norm(mat)))
    return out

def compute_lambda_(mat1, mat2, x_, S=None, maxstepsize=1.0, lowest=False):
    nlast = mat1.shape[0]
    assert (nlast > 1)
    lambda_test = 1.0
    lambda_lasttest = 0.0
    stepsize_lasttest = 0.0
    stepsize = 0.0
    iok = 0
    for i in range(10):
        scr = mat1 + mat2 * (1.0 / lambda_test)
        # the trial vectors are orthonormal in the Euclidean metric, not the
        # sop metric, so the subspace problem is generalized: keep the same
        # overlap S as the main davidson loop or lambda is solved for the
        # wrong equation (this inconsistency is why the call was disabled)
        if S is None:
            e, c = np.linalg.eigh(scr)
        else:
            e, c = scipy.linalg.eigh(scr, S)

        ivec = -1
        if lowest:
            # restricted-step AH: always the lowest root, lambda shortens it
            if abs(c[0, 0]) > 1e-10:
                ivec = 0
        else:
            for j in range(nlast):
                if abs(c[0, j]) <= 1.1 and abs(c[0, j]) > 0.1:
                    ivec = j
                    break
        if ivec < 0:
            raise Exception('logical error in AugHess')
        c[:, ivec] = c[:, ivec] / (c[0, ivec])
        step = np.dot(x_[1:, :nlast], c[:nlast, ivec])
        stepsize = np.linalg.norm(step) / abs(lambda_test)
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


def davidson(hop, g, hdiag, sop=None, max_stepsize=1.0, tol=5e-6, neig=1, mmax=10,
             use_lambda=False, root='yanai', microstep_tol=1e-4, precond=None, micro_red=0.1):
    # use_lambda: restricted-step augmented Hessian (as BAGEL AugHess / ORZ):
    # lambda (>= 1) scales the Hessian block so that |step| <= max_stepsize,
    # and the micro iterations stop at |r|/lambda < max(tol, microstep_tol*step).
    # root: 'yanai' = first root with 0.1 < |c0| <= 1.1 (ORZ/BAGEL rule),
    #       'lowest' = always the lowest root.
    lowest = use_lambda and root == "lowest"
    # Setup the subspace trial vectors
    print('No. of start vectors:', 1)
    neig = neig
    print('No. of desired Eigenvalues:', neig)
    n = g.shape[0] + 1
    x = np.zeros((n, mmax + 1), dtype=complex)  # holder for trial vectors as iterations progress
    sigma = np.zeros((n, mmax + 1), dtype=complex)
    sdotx = np.zeros((n, mmax + 1), dtype=complex)
    ritz = np.zeros((n, mmax), dtype=complex)
    #-------------------------------------------------------------------------------
    # Begin iterations
    #-------------------------------------------------------------------------------
    #initial extra (1,0) vector
    start = time.time()
    x[0, 0] = 1.0
    sigma[1:, 0] = g
    sdotx[0, 0] = 1.0
    if sop is None:
        print('No sop parsed')
        exit()

    # "first" guess vector

    if precond is None:
        den = hdiag.real + 0.001
        den = np.where(abs(den) > PRECOND_DEN_FLOOR, den, np.where(den < 0, -PRECOND_DEN_FLOOR, PRECOND_DEN_FLOOR))
        x[1:, 1] = -g / den
    else:
        # precond(r, shift) = (h_diag - shift*s_diag)^-1 r
        x[1:, 1] = -precond(g, -0.001)

    x[:, 1] = x[:, 1] / np.linalg.norm(x[:, 1])
    for m in range(1, mmax + 1):
        sigma[1:, m] = hop(x[1:, m])
        sigma[0, m] = np.dot(g.conj(), x[1:, m])
        sdotx[1:, m] = sop(x[1:, m])
        sdotx[0, m] = 0.0j
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        T = np.linalg.multi_dot((x[:, :m + 1].T.conj(), sigma[:, :m + 1]))
        S = np.linalg.multi_dot((x[:, :m + 1].T.conj(), sdotx[:, :m + 1]))
        mat1 = numpy.zeros((m + 1, m + 1), dtype=complex)
        mat2 = numpy.zeros((m + 1, m + 1), dtype=complex)
        mat2[1:, 1:] = T[1:, 1:]
        mat1 = T - mat2
        if use_lambda:
            try:
                lambda_, stepsize = compute_lambda_(mat1, mat2, x, S=S,
                                                    maxstepsize=max_stepsize, lowest=lowest)
            except Exception as exc:
                print(f'compute_lambda_ failed ({exc}), falling back to lambda=1')
                lambda_ = 1.0
                stepsize = 0.0
        else:
            lambda_ = 1.0
            stepsize = 0.0
        T = mat1 + mat2 * (1.0 / lambda_)
        try:
            e, vects = scipy.linalg.eigh(T[:m + 1, :m + 1], S[:m+1,:m+1])
        except np.linalg.LinAlgError:
            # near-linearly-dependent trial vectors made S numerically
            # singular; solve in the well-conditioned subspace of S instead
            w, V = scipy.linalg.eigh(S[:m + 1, :m + 1])
            keep = w > 1e-10 * w.max()
            Vk = V[:, keep] / np.sqrt(w[keep])
            e, u = scipy.linalg.eigh(Vk.T.conj() @ T[:m + 1, :m + 1] @ Vk)
            vects = Vk @ u
        ivec = -1
        if lowest and abs(vects[0, 0]) > 1e-10:
            ivec = 0
        else:
            for j in range(m + 1):
                if abs(vects[0, j]) <= 1.1 and abs(vects[0, j]) > 0.1:
                    ivec = j
                    break
        if ivec < 0:
            raise Exception('logical error in AugHess')
        elif ivec > 0:
            print('... the vector found in AugHess was not the lowest eigenvector ...')

        # residual of (g c0 + H x / lambda - eps S x) = 0, the equation the
        # subspace matrix mat1 + mat2/lambda stands for.  sigma[1:, 0] is g (the
        # (1,0) basis vector), sigma[1:, k>=1] = H x_k.  The former expression
        # sigma - lambda*eps*S x scaled the H and S terms by lambda but not the
        # gradient term, i.e. it was lambda*r - (lambda-1) g c0: identical for
        # lambda = 1, never zero for lambda != 1 (BAGEL AugHess has the form used
        # here, written with the vector divided by lambda*c0).
        v = vects[:, ivec]
        ritz[1:, 0] = g * v[0] + np.dot(sigma[1:, 1:m + 1], v[1:]) / lambda_ \
            - e[ivec] * np.dot(sdotx[1:, :m + 1], v)
        ritz[0, 0] = 0.0
        err = np.linalg.norm(ritz[:, 0])
        print(f'iter {m}, ivec {ivec}, c[{ivec}], {vects[0,ivec]:8.4f}, eps={e[ivec]:12.8f}, res={err:8.6e},  lambda={lambda_:6.2f}, step={stepsize:8.4f}')
        # NOTE stepsize is hardwired to 0.0 above (compute_lambda_ disabled),
        # so the old criterion min(1e-3*stepsize, tol) was always 0 and the
        # solver never converged -- it silently ran to mmax every time
        # BAGEL criterion with lambda on; the old absolute one otherwise
        # with lambda on: relative to the gradient (an absolute tol is met by
        # the first trial vector once |g| is small) and at least 3 micro iterations
        if use_lambda:
            converged = m >= 3 and err < max(microstep_tol * stepsize, micro_red * np.linalg.norm(g) * abs(v[0]))
        else:
            converged = err < tol and m > 4
        if converged:
            print('Davidson converged at iteration no.:', m - 1)
            end = time.time()
            print('Davidson time:', end - start)
            x_ = np.dot(x[:, :m + 1], vects[:, ivec])
            print(f'{x_[0]:.6f}, x_[0]')
            # restricted-step AH: the level-shifted step is scaled by 1/lambda
            return x_[1:] / x_[0] / lambda_, e[ivec]
        elif m is mmax:
            print('Max iteration reached')
            x_ = np.dot(x[:, :m + 1], vects[:, ivec])
            print(f'{x_[0]:.6f}, x_[0]')
            return x_[1:] / x_[0] / lambda_, e[ivec]
        else:
            if precond is None:
                den = hdiag.real - e[ivec] * lambda_
                den = np.where(abs(den) > PRECOND_DEN_FLOOR, den, np.where(den < 0, -PRECOND_DEN_FLOOR, PRECOND_DEN_FLOOR))
                ritz[1:, 0] = -ritz[1:, 0] / den
            else:
                ritz[1:, 0] = -precond(ritz[1:, 0], e[ivec] * lambda_)
            # orthonormalize ritz vector (complex inner product needs the
            # conjugate; two passes of Gram-Schmidt for numerical stability)
            for _ in range(2):
                for idx in range(m + 1):
                    ritz[:, 0] = ritz[:, 0] - (np.dot(x[:, idx].conj(), ritz[:, 0]) * x[:, idx])
            nrm = np.linalg.norm(ritz[:, 0])
            if nrm < 1e-10:
                print('Davidson: new trial vector linearly dependent, returning current solution')
                x_ = np.dot(x[:, :m + 1], vects[:, ivec])
                return x_[1:] / x_[0] / lambda_, e[ivec]
            x[:, m + 1] = ritz[:, 0] / nrm

# term 5 of the super-CI operator contracts (g - G)[t,u] x[u,i]; gauge covariance
# needs G = g_tu2 (= g_tu^T).  False keeps the historical g_tu, which differs
# already for real rdms whenever g_tu is not symmetric.
TERM5_GTU_COVARIANT = True

# smallest |h_diag - shift*s_diag| used by the natural-orbital preconditioner
PRECOND_DEN_FLOOR = 0.05

# note: the ncas, nelecas, ncore should all be counted as the number of spin orbitals
def gen_g_hop(casscf, mo, casdm1, casdm2, eris):
    if casscf.mo_coeff is None:
        casscf.mo_coeff = mo
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    # The CI solvers (pyscf fci_dhf_slow and the Dice reader alike) return
    # casdm1[t,u] = <t^dagger u>.  Everything below uses casdm1 as an MO-basis
    # density matrix, rho = C D C^dagger, which needs D = casdm1^T.  The two
    # coincide for a real rdm (natural orbitals), so only complex rdms (frozen
    # basis, natorb=False) were affected: there the orbital gradient was wrong
    # in size and sign (FD-checked on Nd(H2O)8 SA-52).  casdm2 is consistent
    # with the g_dm2 contraction as is.
    # casdm1_ci keeps the solver convention for the few places that need it.
    # Every contraction below is written gauge covariant: under orbital phases
    # phi_p -> ph_p phi_p all operator matrices (F, g, kappa, sigma, casdm1)
    # transform as O_pq -> conj(ph_p) ph_q O_pq, casdm1_ci and the index pairs
    # of casdm2 as the transpose of that (checked numerically, see
    # mcscf/test/test_orbital_grad_fd.py).
    casdm1_ci = casdm1
    casdm1 = casdm1.T

    ################# gradient #################
    dm_core = np.zeros((nmo, nmo), dtype=complex)
    dm_active = np.zeros((nmo, nmo), dtype=complex)
    idx = np.arange(ncore)
    dm_core[idx, idx] = 1
    dm_active[ncore:nocc, ncore:nocc] = casdm1
    dm1 = dm_core + dm_active
    h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
    core_occ = np.zeros(nmo)
    core_occ[:ncore] = 1
    dm_core_ao = reduce(np.dot, (mo, dm_core, mo.T.conj()))
    vj_c, vk_c = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
    eris.vj_c = vj_c
    eris.vk_c = vk_c
    vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
    vj_a_mo, vk_a_mo = eris.get_jk_active_mo(casdm1)
    vhf_a = vj_a_mo - vk_a_mo
    vhf_ca = vhf_c + vhf_a
    
    c_core = np.eye(ncore)
    c_vir = np.eye(nmo - nocc)
    # canonicalization begins here
    if casscf.canonicalize_:
        fock_eff = h1e_mo + vhf_ca
        fock_core = fock_eff[:ncore, :ncore]
        fock_vir = fock_eff[nocc:, nocc:]
        e_core, c_core = casscf._scf.eig(fock_core, mo=mo[:, :ncore])
        e_vir, c_vir = casscf._scf.eig(fock_vir, mo=mo[:, nocc:])
        mo[:,:ncore] = np.dot(mo[:,:ncore], c_core)
        mo[:,nocc:] = np.dot(mo[:,nocc:], c_vir)
        h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
        vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
        # Rotate vhf_a to new MO basis using canonicalization rotation
        c = np.eye(nmo, dtype=complex)
        c[:ncore, :ncore] = c_core
        c[nocc:, nocc:] = c_vir
        vhf_a = reduce(np.dot, (c.T.conj(), vhf_a, c))
        vhf_ca = vhf_c + vhf_a
        print('ecore', e_core)
        print('evir', e_vir)
        print('Active-active block of effective Fock matrix:\n', fock_eff[ncore:nocc, ncore:nocc])
        print('canonicalization done')
    # canonicalization ends
    fock_eff = h1e_mo + vhf_ca
    fock_eff_core = fock_eff[:ncore,:ncore]
    fock_eff_vir = fock_eff[nocc:,nocc:] 
    fock_offdiag_core = np.linalg.norm(fock_eff_core - np.diag(fock_eff_core.diagonal()))
    fock_offdiag_vir = np.linalg.norm(fock_eff_vir - np.diag(fock_eff_vir.diagonal()))
    print(f'off diagonal elements of effective fock matrix, core-core block {fock_offdiag_core:20.12g}, vir-vir block {fock_offdiag_vir:20.12g}')
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
    
    # transform g_dm2 to canonical basis
    c = np.eye(nmo, dtype=complex)
    c[:ncore, :ncore] = c_core
    c[nocc:, nocc:] = c_vir
    g_dm2 = np.dot(c.T.conj(), g_dm2)
    # transformation done

    g[:, ncore:nocc] += g_dm2
    g_new[ncore:nocc,:ncore] += g_dm2.T[:,:ncore]
    g_new[nocc:,ncore:nocc] += g_dm2[nocc:,:]
    g_orb = casscf.pack_uniq_var(g - g.T.conj())
    #g_orb = casscf.pack_uniq_var(g_new-g_new.T.conj())

    fock_eff = h1e_mo + vhf_ca
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
    g_tu = lib.einsum('tuvw,vw->tu', casdm2, g[ncore:nocc,ncore:nocc]) - lib.einsum('tu,vw,vw->tu', casdm1_ci, casdm1_ci, g[ncore:nocc,ncore:nocc])
    # g_tu2 = d_vw,ut F_vw - F_vw,D_vw,D_tu
    g_tu2 = lib.einsum('vwut,vw->tu', casdm2, g[ncore:nocc,ncore:nocc]) - lib.einsum('ut,vw,vw->tu', casdm1_ci, casdm1_ci, g[ncore:nocc,ncore:nocc])

    y = lib.einsum('pu,qu->pq', (h1e_mo + vhf_c)[ncore:nocc, ncore:nocc], casdm1_ci)
    # zeros, not ones: only the lower-triangle blocks are filled and the packed
    # diagonal is taken from h_diag + h_diag.T, so a ones-initialised upper
    # triangle silently added 1 Eh to every element handed to the Davidson
    h_diag = np.zeros((nmo, nmo), dtype=complex)
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
    h_diag = h_diag.real*(1.+0.j)

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
        # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
        sigma[nocc:, :ncore] += \
            (lib.einsum('ab,bi->ai', f_vv, x1[nocc:, :ncore])\
            - lib.einsum('ji,aj->ai', f_oo, x1[nocc:, :ncore]))

        # term 2 h_ai,bu = -delta_ab*f_vi*D_vu
        sigma[nocc:, :ncore] -= lib.einsum(
            'vi,uv,au->ai', g[ncore:nocc, :ncore], dm1_aa,
            x1[nocc:, ncore:nocc])

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        sigma[nocc:, :ncore] += \
             lib.einsum('au,ui->ai', g[nocc:, ncore:nocc], x1[ncore:nocc, :ncore])\
            -lib.einsum('av,vu,ui->ai', g[nocc:, ncore:nocc], dm1_aa, x1[ncore:nocc, :ncore])

        # core-active block
        # term5 h_ti,uj =
        # f_ji * (D_ut - delta_tu)
        sigma[ncore:nocc, :ncore] += \
            lib.einsum('tu,ji,uj->ti', dm1_aa, f_oo, x1[ncore:nocc, :ncore])\
            - lib.einsum('ji,tj->ti', f_oo, x1[ncore:nocc, :ncore])
        # h_ti,uj += f_tu delta_ij - f_tv,d_uv,delta_ij
        sigma[ncore:nocc, :ncore] += \
            lib.einsum('tu,ui->ti', g[ncore:nocc,ncore:nocc] - (g_tu2 if TERM5_GTU_COVARIANT else g_tu), x1[ncore:nocc, :ncore])\
            -lib.einsum('tv,vu,ui->ti', g[ncore:nocc,ncore:nocc], dm1_aa, x1[ncore:nocc, :ncore])\
            -lib.einsum('tv,uv,ui->ti', dm1_aa, g[ncore:nocc,ncore:nocc].conj(), x1[ncore:nocc, :ncore])
        # term5 continued
        # g_tu = d_tu,vw F_vw - F_vw,D_vw,D_tu
        # + delta_ij(f_tu-(d_tu,vw-d_ut*d_vw)*f_vw-f_tv*d_uv
        # the last two term differs from molpro's expression since molpro
        # suppose a symmetrized form of 2rdm while we don't.

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        # adjoint of term 3 h_ti,bj x_bj->sigma_ti
        sigma[ncore:nocc,:ncore] += \
            lib.einsum('au,ai->ui', g[nocc:, ncore:nocc].conj(), x1[nocc:,:ncore]) \
           -lib.einsum('av,uv,ai->ui', g[nocc:, ncore:nocc].conj(), dm1_aa, x1[nocc:,:ncore])
        #sigma[ncore:nocc,:ncore] += x1[ncore:nocc,:ncore] * h_diag[ncore:nocc,:ncore]
        # virtual-active block
        # adjoint of term2
        # h_bu,ai = -delta_ab * f_vi*D_vu
        sigma[nocc:,ncore:nocc] -= lib.einsum('vi,vu,ai->au', g[ncore:nocc, :ncore].conj(), dm1_aa, x1[nocc:,:ncore])
        # term4 h_ti,bu = 0

        #term 6 h_at,bu=delta_ab(d_tu,vw-d_tu*d_vw)f_vw+d_tu*f_ab
        #sigma[nocc:,ncore:nocc] += x1[nocc:,ncore:nocc] * h_diag[nocc:,ncore:nocc]
        
        sigma[nocc:, ncore:nocc] += lib.einsum('ut,ab,bu->at', dm1_aa, f_vv, x1[nocc:, ncore:nocc]) + lib.einsum('tu,au->at', g_tu, x1[nocc:, ncore:nocc])
        
        sigma_pack = casscf.pack_uniq_var(sigma)
        return sigma_pack

    n_uniq_var = g_orb.shape[0]
    hop = LinearOperator((n_uniq_var,n_uniq_var), matvec=h_op)
    def s_op(x):
        # overlap of the super-CI singly excited functions, with the full
        # active 1-rdm (the diagonal-only form is this metric in natural
        # orbitals; without natorb casdm1 is not diagonal):
        #   <ai|bj> = delta_ab delta_ij
        #   <at|bu> = delta_ab D_tu            -> (S x)_at = x_au D_ut
        #   <ti|uj> = delta_ij (delta_tu-D_tu) -> (S x)_ti = x_ti - D_tu x_ui
        # casdm1 here is the MO density matrix (operator-like), as in h_op.
        x1 = casscf.unpack_uniq_var(x)
        out = np.zeros_like(x1)
        out[nocc:, :ncore] = x1[nocc:, :ncore]
        out[nocc:, ncore:nocc] = np.dot(x1[nocc:, ncore:nocc], casdm1)
        out[ncore:nocc, :ncore] = x1[ncore:nocc, :ncore] - np.dot(casdm1, x1[ncore:nocc, :ncore])
        return casscf.pack_uniq_var(out)
    sop = LinearOperator((n_uniq_var, n_uniq_var), matvec=s_op)
    def h_diag_inv(x):
        den = casscf.pack_uniq_var(h_diag+h_diag.T).real
        den = np.where(abs(den) > PRECOND_DEN_FLOOR, den, np.where(den < 0, -PRECOND_DEN_FLOOR, PRECOND_DEN_FLOOR))
        return x/den
    precond = LinearOperator((n_uniq_var, n_uniq_var), h_diag_inv)

    # Preconditioner for the generalized problem (H - eps*S) x = r, applied in
    # the natural-orbital representation of the active space: there S is
    # diagonal (1, n_t, 1-n_t) and the diagonal h formulas above are the ones
    # they were derived for.  Rotating the active index of the residual with the
    # eigenvectors of casdm1 makes the Davidson iteration independent of the
    # active-orbital basis (frozen basis == natural orbitals), while the orbital
    # rotation itself and the L-BFGS history stay in the frozen basis.
    occ_no, u_no = np.linalg.eigh(casdm1)          # casdm1 = u n u^dagger
    f_no = reduce(np.dot, (u_no.T.conj(), fock_eff[ncore:nocc, ncore:nocc], u_no)).diagonal().real
    y_no = reduce(np.dot, (u_no.T.conj(), y, u_no)).diagonal().real
    g2_no = reduce(np.dot, (u_no.T.conj(), g_dm2[ncore:nocc], u_no)).diagonal().real
    f_c = fock_eff.diagonal().real[:ncore]
    f_v = fock_eff.diagonal().real[nocc:]
    hd_vc = f_v[:, None] - f_c[None, :]
    hd_tc = (f_no - y_no - g2_no)[:, None] - (1. - occ_no)[:, None] * f_c[None, :]
    hd_vt = f_v[:, None] * occ_no[None, :] - (y_no + g2_no)[None, :]
    sd_tc = np.repeat((1. - occ_no)[:, None], ncore, axis=1)
    sd_vt = np.repeat(occ_no[None, :], nmo - nocc, axis=0)

    def _safe_div(r, den):
        # nearly redundant rotations (active occupation -> 0 or 1) have s_diag
        # and h_diag -> 0 together; an unprotected denominator amplifies them by
        # orders of magnitude and the Davidson diverges (OH CAS(7e,8), SA-4).
        # The historical preconditioner was protected by an accidental +1 Eh.
        den = np.where(abs(den) > PRECOND_DEN_FLOOR, den, np.where(den < 0, -PRECOND_DEN_FLOOR, PRECOND_DEN_FLOOR))
        return r / den

    def precond_shift(x, shift=0.0):
        '''(h_diag - shift*s_diag)^-1 x in the active natural-orbital basis.'''
        x1 = casscf.unpack_uniq_var(x)
        out = np.zeros_like(x1)
        out[nocc:, :ncore] = _safe_div(x1[nocc:, :ncore], hd_vc - shift)
        tc = np.dot(u_no.T.conj(), x1[ncore:nocc, :ncore])
        out[ncore:nocc, :ncore] = np.dot(u_no, _safe_div(tc, hd_tc - shift * sd_tc))
        vt = np.dot(x1[nocc:, ncore:nocc], u_no)
        out[nocc:, ncore:nocc] = np.dot(_safe_div(vt, hd_vt - shift * sd_vt), u_no.T.conj())
        return casscf.pack_uniq_var(out)
    hop.precond_shift = precond_shift
    return g_orb, casscf.pack_uniq_var((h_diag+h_diag.T).real), hop, sop, precond, mo


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


def lbfgs_precondition(grad, xs, ys, dens):
    '''First loop of the L-BFGS two-loop recursion (Kreplin, Knowles, Werner,
    JCP 152, 074102 (2020), Fig. 1).  Vectors are packed complex rotations; the
    inner product is Re(u^dagger v), in which the secant condition reads
    h x_i = y_i with y_i = g_{i+1} - g_i the difference of the packed gradients
    and h the operator of the super-CI equation (h x = -g).'''
    gbar = grad.copy()
    a = np.zeros(len(xs))
    for i in reversed(range(len(xs))):
        a[i] = np.dot(xs[i].conj(), gbar).real / dens[i]
        gbar = gbar - ys[i] * a[i]
    return gbar, a


def lbfgs_postprocess(x, xs, ys, dens, a):
    '''Second loop.  x is the STEP returned by the approximate-Hessian solver
    for the preconditioned gradient (x = -h0^-1 gbar), hence the plus sign.'''
    for i in range(len(xs)):
        b = np.dot(ys[i].conj(), x).real / dens[i]
        x = x - xs[i] * (a[i] + b)
    return x


def mcscf_superci(mc, mo_coeff, max_stepsize=0.2, uphill_step_tol=1e-3, conv_tol=None,
                  conv_tol_grad=None, verbose=5, cderi=None, bfgs=False, solver='davidson',  # cderi kept for backward compat
                  davidson_maxiter=10, ah_lambda=False, ah_root='yanai', no_basis_precond=True, descent_guard=True,
                  step_control='legacy', trust_max=2.0, grad_scale=1.0, reject_uphill=True,
                  sd_hdiag_floor=0.05, kramers_check=True):
    bfgs = bfgs
    davidson_mmax = davidson_maxiter
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

    # every step is projected with ensure_kramers, which is only meaningful for
    # orbitals laid out as interleaved, phase +1 Kramers pairs.  Orbitals that
    # lost this (plain-eigh canonicalisation, sort_mo, an unrestricted run)
    # would silently have most of each step projected away.
    if kramers_check:
        from socutils.tools import kramers
        dev = kramers.pairing_deviation(mol, mo)
        log.info('Kramers pairing of the starting orbitals: max deviation %.2e', dev)
        if dev > 1e-6:
            raise RuntimeError(
                'starting orbitals are not interleaved, phase +1 Kramers pairs '
                '(max deviation %.2e); phase-clean them (e.g. a CAHF pass with '
                'trace_orbital=False) or use socutils.tools.kramers.symmetrize, '
                'or pass kramers_check=False' % dev)

    mci = mc.view(zcasci.CASCI)
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
    trust_radii = 0.5
    # step_control='trust': restricted-step augmented Hessian with an adaptive
    # trust radius delta (Frobenius norm of the rotation matrix, the same norm
    # max_stepsize has always been compared with).  The AH step is the
    # level-shifted step y = -(H - mu S)^-1 g with mu chosen (through lambda)
    # so that |y| <= delta; delta follows rho = dE_actual / Q(y) with the
    # second-order model Q(y) = 2 Re(g.y) + Re(y.Hy):  rho < 0.25 halve,
    # rho > 0.75 at the boundary double (up to trust_max), dE > 0 reject.
    # max_stepsize is the initial delta.  'legacy' keeps the old behaviour
    # (unconstrained step divided by its norm, fixed cap max_stepsize).
    tr_mode = step_control == 'trust'
    use_lam = ah_lambda or tr_mode
    delta = max_stepsize
    e_last = e_tot
    dr = None
    while not conv and imacro < mc.max_cycle_macro:
        # compute natural orbital and transform ci to natural orbtial basis
        # no transform function available now so re do a ci calculation
        # do it in gen_g_hop
        if mc.natorb is True: 
            moa = mo[:, ncore:nocc]
            natocc, c = mc._scf.eig(-casdm1, mo=moa)
            moa_new = np.dot(moa, c)
            mo[:, ncore:nocc] = moa_new

            eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
            t2m = log.timer('update eris', *t2m)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
            print('update eris', verbose)
            e_nat_tot, e_cas, fcivec = mci.kernel(mo, ci0=None, verbose=verbose)
            print(e_tot, e_nat_tot)
            casdm1, casdm2 = mci.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
            print('natural occupation number in cas space', casdm1.diagonal())

        g, h_diag, hop, sop, precond, mo = gen_g_hop(mc, mo, casdm1, casdm2, eris)
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
                if abs(g_unpack[i,j])>1e-2:
                    print(f'{i}, {j}, {g_unpack[i,j]}')
        if mc.irrep is not None:
            print(
                f'{row}, {col}, {g_unpack[row,col].real:8.6f} {mc.irrep[row]}, {mc.irrep[col]}'
            )
        else:
            print(
                f'{row}, {col}, {g_unpack[row,col].real:8.6f}'
            )
        if abs(de) < conv_tol and norm_gorb < conv_tol_grad:
            conv = True
        if conv:
            break

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
            bfgs_on = 1.0
            if not rejected and norm_gorb < bfgs_on:
                # x_prev is the step actually taken; keep the pair only if
                # it has positive curvature (KKW II eq. 40)
                den = np.dot((g - g_prev).conj(), x_prev).real
                if den > 0:
                    ys.append(g - g_prev)
                    xs.append(x_prev)
                    rhos.append(den)
                else:
                    print(f'bfgs: pair skipped, y.x = {den:.4e} <= 0')
            if len(ys) > BFGS_SUBSPACE:
                ys.pop(0)
                xs.pop(0)
                rhos.pop(0)
            #if np.linalg.norm(g_prev) < norm_gorb or de > 0.0:
            if de > 0.0:
                print('gradeint norm', np.linalg.norm(g_prev), norm_gorb)
                xs = []
                ys = []
                rhos = []
            # never precondition with BFGS while retrying a rejected step: the
            # history produced the rejected direction, the retry must not
            # inherit it (the de>0 wipe above makes this mostly redundant,
            # but the intent should not depend on that coincidence)
            use_bfgs = bfgs is True and norm_gorb < bfgs_on and not rejected and len(xs) > 0
            # anti-stagnation floor for the gradient-scaling mode only: with
            # ah_lambda the trust radius is the step-length bound and must be
            # allowed to stay small after a rejection (the floor re-inflated it
            # after every accepted step and produced reject/accept limit cycles)
            if not rejected and not use_lam:
                trust_radii = max(trust_radii, 0.02)
            # the trust radius scales the gradient handed to the solver, so
            # the two-loop recursion has to act on the same scaled gradient
            # without ah_lambda the trust radius scales the gradient handed to
            # the solver (the two-loop then acts on that scaled gradient); with
            # ah_lambda the step length is limited by lambda inside the AH
            gbar = g if use_lam else trust_radii * g
            # grad_scale < 1 damps the super-CI step uniformly (equivalent to
            # scaling its Hessian by 1/grad_scale near convergence): the
            # super-CI matrix underestimates the curvature of the slow modes
            gbar = grad_scale * gbar
            if use_bfgs:
                gbar, a = lbfgs_precondition(gbar, xs, ys, rhos)

            residuals = []
            def callback(rk):
                residuals.append(rk)
            if solver == 'gmres':
                x, _ = gmres(hop, -gbar, M=precond, maxiter=50, callback=callback)
            elif solver == 'davidson':
                # the 0.02 floor on trust_radii above keeps the step from
                # vanishing (anti-stagnation near local minima)
                x, e = davidson(hop, gbar, h_diag, sop=sop,
                                max_stepsize=delta / np.sqrt(2.) if tr_mode else min(trust_radii, max_stepsize),
                                mmax=davidson_mmax, use_lambda=use_lam, root=ah_root,
                                precond=getattr(hop, 'precond_shift', None) if no_basis_precond else None)
                print('residuals', residuals)
            if use_bfgs:
                x = lbfgs_postprocess(x, xs, ys, rhos, a)
        else:
            residuals = []
            def callback(rk):
                residuals.append(rk)
            if solver == 'gmres':
                x, _ = gmres(hop, -trust_radii*gbar, M=precond, maxiter=50, callback=callback)
                #x = precond(-trust_radii*gbar) 
            elif solver == 'davidson':
                x, e = davidson(hop, grad_scale * (gbar if use_lam else trust_radii*gbar), h_diag, sop=sop, mmax=10,
                                max_stepsize=delta / np.sqrt(2.) if tr_mode else min(trust_radii, max_stepsize),
                                use_lambda=use_lam, root=ah_root,
                                precond=getattr(hop, 'precond_shift', None) if no_basis_precond else None)
                print('residuals', residuals)
        t2m = log.timer('Solving Super-CI equation', *t_gmres)

        def finalize_step(x):
            dr = mc.unpack_uniq_var(x)
            # project the step onto the Kramers-symmetric subspace: numerical
            # T-breaking in dr is amplified x3-10 per macro iteration by the soft
            # modes (seen even with exact CI), so kill it at the source.  Assumes
            # interleaved +1-phase Kramers pairs (phase-clean CAHF start + frozen
            # basis keep this true throughout).
            dr = ensure_kramers(dr)
            # with ah_lambda the trust radius bounds the step actually taken,
            # including the L-BFGS post-processing (KKW II: the post-processed
            # rotation is rescaled to the trust radius)
            if tr_mode:
                step_cap = delta
            else:
                step_cap = min(trust_radii, max_stepsize) if ah_lambda else max_stepsize
            if norm(dr) > step_cap:
                print('step rescaled')
                dr = dr * (step_cap / norm(dr))
                print(norm(dr))
            # from here on x is the step actually taken (KR-projected, rescaled):
            # the energy model and the BFGS history must both see this one
            return dr, mc.pack_uniq_var(dr)

        dr, x = finalize_step(x)
        # descent guard: the super-CI AH step is not guaranteed to go downhill
        # (unconverged Davidson / root picked by |c0| can carry eps > 0).  If the
        # step taken is not a descent direction of the first-order model, or we
        # are retrying after an uphill step, fall back to the Fock-diagonal
        # preconditioned gradient step instead of re-solving the same AH.
        slope = np.dot(g.conj(), x).real
        if descent_guard and (slope >= 0 or (rejected and not tr_mode)):
            why = 'retry after uphill step' if rejected else f'AH step not downhill, Re(g.x) = {slope:.3e}'
            x = -(1.0 if tr_mode else trust_radii) * g / np.maximum(h_diag.real, sd_hdiag_floor)
            dr, x = finalize_step(x)
            print(f'descent guard ({why}): preconditioned gradient step, '
                  f'|dr| = {norm(dr):.4e}, Re(g.x) = {np.dot(g.conj(), x).real:.3e}')
        for i in range(nmo):
            for j in range(i):
                if abs(dr[i,j])>1e-2:
                    print(f'{i}, {j}, dr {dr[i,j]}')
        rotation = expmat(dr)

        norm_rot = np.linalg.norm(rotation-np.eye(nmo, dtype=complex))
        g_unpack = mc.unpack_uniq_var(g)

        mo_new = np.dot(mo, rotation)
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        #eris = zmc_ao2mo._ERIS(mc, mo_new, level=2)
        eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi, level=2)
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
        if tr_mode:
            q_model = 2. * np.vdot(g, x).real + np.vdot(x, hop(x)).real
            rho = de / q_model if q_model < 0 else -1.0
            print(f'trust region: |dr| = {norm(dr):.4e}, delta = {delta:.4e}, model Q = {q_model:.4e}, rho = {rho:.3f}')
            # an energy rise below conv_tol is noise at the level we converge to:
            # never reject on it.  reject_uphill=False accepts every step (as the
            # original driver and KKW's L-BFGS do): the trust radius still
            # shrinks and the L-BFGS history is discarded at the next iteration.
            if de > conv_tol and reject_uphill:
                delta = 0.5 * norm(dr)
                print(f'step rejected (dE = {de:.4e} > 0), trust radius -> {delta:.4e}')
                eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
                rejected = True
                imacro += 1
                t1m = log.timer(f'macro iter {imacro}', *t1m)
                continue
            if de > conv_tol:
                delta = 0.5 * norm(dr)
                print(f'uphill step accepted (dE = {de:.4e} > 0), trust radius -> {delta:.4e}')
            elif rho < 0.25:
                delta = 0.5 * norm(dr)
                print(f'rho < 0.25: trust radius -> {delta:.4e}')
            elif rho > 0.75 and norm(dr) > 0.9 * delta:
                delta = min(2. * delta, trust_max)
                print(f'rho > 0.75 at the boundary: trust radius -> {delta:.4e}')
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
        #    eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi, level=2)
        #    mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        #    e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)
        #    de = e_tot - e_last
        #    e2 = 0.5 * np.dot(x.T.conj(), g)
        #    r = de / e2
        #    print(f'Energy change {de:.4e}, predicted change {e2:.4e}')
        if tr_mode:
            pass
        elif False: #r < -10 and de > 0.0:
            trust_radii *= 0.7
            print(dr)
            print(f'step rejected, shrink trust radius to {trust_radii:.4f}')
            rejected=True
            new_rot = expmat(0.01*dr)
            mo_new = np.dot(mo, new_rot)
            eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi, level=2)
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
            trust_radii *= 0.5
            print(f'Energy rises by {de:.4e}, accept step but shrik trust radius to {trust_radii:.4f}')
        elif r < 0.25: #and de < 0.0:
            trust_radii *= 0.5
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
        #for i in range(nvar):
        #    if abs(rotation[i, i]) > 1.01 or abs(rotation[i, i]) < 0.99:
        #        print(rotation[i, i] > 1.01, rotation[i, i] < 0.99, i, i,
        #              rotation[i, i])
        #    for j in range(i):
        #        if abs(rotation[i, j]) > 0.01:
        #            continue

        trust_floor = 1e-2 * max_stepsize
        if reject_uphill and not tr_mode and de > 0.0 and norm(dr) > uphill_step_tol and imacro > 0 \
                and trust_radii > trust_floor * 1.001:
            # substantial uphill step: reject AND shrink the trust radius --
            # without the shrink the retry re-solves the same problem at the
            # same point and proposes the same step forever (deadlock).  Once
            # the trust radius reaches its floor the step is force-accepted
            # instead (the model cannot be trusted to do better at this
            # resolution and the outer loop keeps re-evaluating).
            trust_radii = max(trust_radii * 0.5, trust_floor)
            print(f'step rejected (dE = {de:.4e} > 0, |dr| = {norm(dr):.4e} > {uphill_step_tol:.1e}), '
                  f'shrink trust radius to {trust_radii:.4f}')
            eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
            rejected = True
            imacro += 1
            t1m = log.timer(f'macro iter {imacro}', *t1m)
            continue
        elif reject_uphill and not tr_mode and de > 0.0 and norm(dr) > uphill_step_tol and imacro > 0:
            print(f'trust radius at floor {trust_floor:.4f}: force-accepting '
                  f'uphill step (dE = {de:.4e}, |dr| = {norm(dr):.4e})')

        rejected = False
        mo = mo_new
        e_last = e_tot
        x_prev = x
        g_prev = g
        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        #from socutils.tools import analyze
        #analyze.analyze(mol, mo[:, ncore:nocc], casdm1.diagonal())
        nact = casdm1.shape[0]
        #for i in range(nact):
        #    for j in range(i):
        #        if abs(casdm1[i, j]) > 1e-5:
        #            print(f'{i}, {j}, {casdm1[i,j]}')
        imacro += 1
        t1m = log.timer(f'macro iter {imacro}', *t1m)
        lib.chkfile.save(mc.chkfile, f'mo_coeff_iter_{imacro}', mo)
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
