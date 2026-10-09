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

def compute_lambda_(mat1, mat2, x_, S=None, maxstepsize=1.0, lowest=False, gram=None):
    # gram: Euclidean overlap of the rotation parts of the trial vectors,
    # gram[i, j] = <x_i[1:], x_j[1:]>.  With it the step length of every trial
    # lambda is c^H gram c and the step itself is never rebuilt (x_ is unused).
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
        if gram is not None:
            cv = c[:nlast, ivec]
            stepsize = np.sqrt(abs(np.vdot(cv, gram[:nlast, :nlast] @ cv))) / abs(lambda_test)
        else:
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
             use_lambda=False, root='yanai', microstep_tol=1e-4, precond=None, micro_red=0.1,
             with_hx=False):
    # with_hx: also return H x for the step x returned (from the stored images
    # of the trial vectors, no extra hop call)
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
    # trial vectors and their images are stored as rows: every vector is then
    # contiguous, and the subspace matrices are updated by one row and one
    # column per micro iteration instead of being rebuilt
    x = np.zeros((mmax + 1, n), dtype=complex)
    sigma = np.zeros((mmax + 1, n), dtype=complex)
    sdotx = np.zeros((mmax + 1, n), dtype=complex)
    Tfull = np.zeros((mmax + 1, mmax + 1), dtype=complex)   # x_i^H sigma_j
    Sfull = np.zeros((mmax + 1, mmax + 1), dtype=complex)   # x_i^H (S x_j)
    Gfull = np.zeros((mmax + 1, mmax + 1), dtype=complex)   # <x_i[1:], x_j[1:]>
    #-------------------------------------------------------------------------------
    # Begin iterations
    #-------------------------------------------------------------------------------
    #initial extra (1,0) vector
    start = time.time()
    x[0, 0] = 1.0
    sigma[0, 1:] = g
    sdotx[0, 0] = 1.0
    Sfull[0, 0] = 1.0
    if sop is None:
        print('No sop parsed')
        exit()

    # "first" guess vector

    if precond is None:
        den = hdiag.real + 0.001
        den = np.where(abs(den) > PRECOND_DEN_FLOOR, den, np.where(den < 0, -PRECOND_DEN_FLOOR, PRECOND_DEN_FLOOR))
        x[1, 1:] = -g / den
    else:
        # precond(r, shift) = (h_diag - shift*s_diag)^-1 r
        x[1, 1:] = -precond(g, -0.001)

    x[1] = x[1] / np.linalg.norm(x[1])
    for m in range(1, mmax + 1):
        sigma[m, 1:] = hop(x[m, 1:])
        sigma[m, 0] = np.dot(g.conj(), x[m, 1:])
        sdotx[m, 1:] = sop(x[m, 1:])
        sdotx[m, 0] = 0.0j
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        xc = x[:m + 1].conj()
        Tfull[:m + 1, m] = xc @ sigma[m]
        Tfull[m, :m] = sigma[:m] @ xc[m]
        Sfull[:m + 1, m] = xc @ sdotx[m]
        Sfull[m, :m] = sdotx[:m] @ xc[m]
        Gfull[:m + 1, m] = xc[:, 1:] @ x[m, 1:]
        Gfull[m, :m] = Gfull[:m, m].conj()
        T = Tfull[:m + 1, :m + 1].copy()
        S = Sfull[:m + 1, :m + 1].copy()
        mat2 = numpy.zeros((m + 1, m + 1), dtype=complex)
        mat2[1:, 1:] = T[1:, 1:]
        mat1 = T - mat2
        if use_lambda:
            try:
                lambda_, stepsize = compute_lambda_(mat1, mat2, None, S=S, maxstepsize=max_stepsize,
                                                    lowest=lowest, gram=Gfull[:m + 1, :m + 1])
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
        # subspace matrix mat1 + mat2/lambda stands for.  sigma[0, 1:] is g (the
        # (1,0) basis vector), sigma[k>=1, 1:] = H x_k.  The former expression
        # sigma - lambda*eps*S x scaled the H and S terms by lambda but not the
        # gradient term, i.e. it was lambda*r - (lambda-1) g c0: identical for
        # lambda = 1, never zero for lambda != 1 (BAGEL AugHess has the form used
        # here, written with the vector divided by lambda*c0).
        v = vects[:, ivec]
        ritz = np.zeros(n, dtype=complex)
        ritz[1:] = g * v[0] + (v[1:] @ sigma[1:m + 1, 1:]) / lambda_ \
            - e[ivec] * (v @ sdotx[:m + 1, 1:])
        err = np.linalg.norm(ritz)
        print(f'iter {m}, ivec {ivec}, c[{ivec}], {vects[0,ivec]:8.4f}, eps={e[ivec]:12.8f}, res={err:8.6e},  lambda={lambda_:6.2f}, step={stepsize:8.4f}')
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
            x_ = v @ x[:m + 1]
            print(f'{x_[0]:.6f}, x_[0]')
            # restricted-step AH: the level-shifted step is scaled by 1/lambda
            if with_hx:
                return x_[1:] / x_[0] / lambda_, e[ivec], (v[1:] @ sigma[1:m + 1, 1:]) / x_[0] / lambda_
            return x_[1:] / x_[0] / lambda_, e[ivec]
        elif m is mmax:
            print('Max iteration reached')
            x_ = v @ x[:m + 1]
            print(f'{x_[0]:.6f}, x_[0]')
            if with_hx:
                return x_[1:] / x_[0] / lambda_, e[ivec], (v[1:] @ sigma[1:m + 1, 1:]) / x_[0] / lambda_
            return x_[1:] / x_[0] / lambda_, e[ivec]
        else:
            if precond is None:
                den = hdiag.real - e[ivec] * lambda_
                den = np.where(abs(den) > PRECOND_DEN_FLOOR, den, np.where(den < 0, -PRECOND_DEN_FLOOR, PRECOND_DEN_FLOOR))
                ritz[1:] = -ritz[1:] / den
            else:
                ritz[1:] = -precond(ritz[1:], e[ivec] * lambda_)
            # orthonormalize ritz vector (complex inner product needs the
            # conjugate; two passes of Gram-Schmidt for numerical stability)
            for _ in range(2):
                ritz -= (xc @ ritz) @ x[:m + 1]
            nrm = np.linalg.norm(ritz)
            if nrm < 1e-10:
                print('Davidson: new trial vector linearly dependent, returning current solution')
                x_ = v @ x[:m + 1]
                if with_hx:
                    return x_[1:] / x_[0] / lambda_, e[ivec], (v[1:] @ sigma[1:m + 1, 1:]) / x_[0] / lambda_
                return x_[1:] / x_[0] / lambda_, e[ivec]
            x[m + 1] = ritz / nrm

# wall-time breakdown, accumulated over a run: name -> [seconds, calls]
TIMINGS = {}


def _tick(name, t0):
    '''Add the wall time since t0 to TIMINGS[name]; returns the current time.'''
    t1 = time.perf_counter()
    rec = TIMINGS.setdefault(name, [0.0, 0])
    rec[0] += t1 - t0
    rec[1] += 1
    return t1


def print_timings(header='wall-time breakdown', wall=None):
    '''Percentages are of `wall` (the whole run).  Nested entries overlap: the
    "grad:" lines are inside gen_g_hop, "super-CI H x" inside the AH solve.'''
    if wall is None:
        wall = sum(v[0] for v in TIMINGS.values())
    print(f'{header}: wall time {wall:.1f} s')
    for name, (sec, n) in sorted(TIMINGS.items(), key=lambda kv: -kv[1][0]):
        print(f'  TIMING {name:36s} {sec:10.1f} s  {n:5d} calls  {sec/max(n,1):8.2f} s/call  {100*sec/max(wall,1e-30):5.1f} %')


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
    _t = time.perf_counter()
    h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
    _t = _tick('grad: hcore to MO', _t)
    core_occ = np.zeros(nmo)
    core_occ[:ncore] = 1
    dm_core_ao = reduce(np.dot, (mo, dm_core, mo.T.conj()))
    vj_c, vk_c = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
    eris.vj_c = vj_c
    eris.vk_c = vk_c
    vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
    _t = _tick('grad: core J-K', _t)
    vj_a_mo, vk_a_mo = eris.get_jk_active_mo(casdm1)
    _t = _tick('grad: active J-K', _t)
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
        _t = _tick('grad: 2-rdm contraction', _t)
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

    _t = _tick('grad: assembly, g_tu, h_diag', _t)

    def h_op(x):
        _th = time.perf_counter()
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
        _tick('super-CI H x', _th)
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
    # the natural occupations are (at least) pairwise degenerate and eigh
    # returns an arbitrary basis of each degenerate subspace; diagonalising
    # the Fock matrix inside every such subspace makes the representation
    # unique, so the diagonals below do not depend on that arbitrariness
    f_act = fock_eff[ncore:nocc, ncore:nocc]
    f_act = 0.5 * (f_act + f_act.T.conj())
    i0 = 0
    while i0 < ncas:
        i1 = i0 + 1
        while i1 < ncas and abs(occ_no[i1] - occ_no[i0]) < 1e-7:
            i1 += 1
        if i1 - i0 > 1:
            sub = u_no[:, i0:i1]
            u_no[:, i0:i1] = np.dot(sub, np.linalg.eigh(reduce(np.dot, (sub.T.conj(), f_act, sub)))[1])
        i0 = i1
    f_no = reduce(np.dot, (u_no.T.conj(), fock_eff[ncore:nocc, ncore:nocc], u_no)).diagonal().real
    # y and the active block of g_dm2 are not Hermitian away from convergence;
    # only the diagonal of a Hermitian, time-reversal symmetric matrix is
    # independent of how the degenerate natural-orbital Kramers pairs are chosen
    herm = lambda m: 0.5 * (m + m.T.conj())
    y_no = reduce(np.dot, (u_no.T.conj(), herm(y), u_no)).diagonal().real
    g2_no = reduce(np.dot, (u_no.T.conj(), herm(g_dm2[ncore:nocc]), u_no)).diagonal().real
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
    _tick('grad: NO preconditioner setup', _t)
    return g_orb, casscf.pack_uniq_var((h_diag+h_diag.T).real), hop, sop, precond, mo


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


def mcscf_superci(mc, mo_coeff, max_stepsize=None, conv_tol=None, conv_tol_grad=None, verbose=5,
                  cderi=None, bfgs=False, solver='davidson', davidson_maxiter=10, ah_root='yanai',
                  trust_radius=None, trust_radius_max=2.0, reject_uphill=True, descent_guard=True,
                  sd_hdiag_floor=0.05, kramers_check=True, kramers_project=True, gen_g_hop_fn=None,
                  bfgs_start_grad=1.0, damp_overshoot=False, bfgs_keep_history=False,
                  reject_noise=1e-3, reject_ratio=0.5, trust_radius_min=0.01, stall_limit=5, **deprecated):
    '''Super-CI orbital optimisation with an adaptive trust region.

    Every macro iteration solves the super-CI augmented-Hessian problem for the
    level-shifted step y = -(H - mu S)^-1 g, with mu chosen (through lambda in
    the Davidson solver) so that |y| <= trust radius; |y| is the Frobenius norm
    of the rotation matrix.  With bfgs=True the gradient is preconditioned and
    the step post-processed with the L-BFGS two-loop recursion (Kreplin,
    Knowles, Werner, JCP 152, 074102 (2020)) and then scaled back into the
    trust region.  The trust radius follows rho = dE / Q, Q = 2 Re(g.y) +
    Re(y.Hy): halved for rho < 0.25 or an energy rise, doubled for rho > 0.75
    when the step reached the boundary, never above trust_radius_max.

    Q is never bought with an extra Hessian-vector product.  For the Davidson
    solution it follows from the images the solver stored; for a step that
    L-BFGS has post-processed, from the equation the step solves,
    B x = -g + mu S x (B the L-BFGS Hessian, mu the level shift), i.e.
    Q = Re(g.x) + mu x.Sx.  For any other step (descent guard, a Kramers
    projection that changed it) there is no Q and the trust radius only reacts
    to an energy rise.

    trust_radius      initial trust radius (default 0.2).  max_stepsize is the
                      former name of this argument and is used when
                      trust_radius is not given.
    reject_uphill     True: a step that raises the energy by more than
                      reject_noise (1e-3 Eh) and by more than reject_ratio (0.5)
                      times the last decrease that was achieved is not taken:
                      back to the last accepted orbitals, half the trust radius,
                      L-BFGS history kept, and the radius may not grow straight
                      back to the one that failed.  A smaller rise is accepted
                      with half the radius and a discarded L-BFGS history.
                      False: every step is accepted in that way.
    trust_radius_min  the trust radius never falls below min(trust_radius_min, |g|).
    stall_limit       stop, not converged, after this many consecutive rejected
                      steps at the minimum radius.
                      (Step control settled on the non-relativistic benchmarks of
                      2026-10: Cu2O2, Al4O2, FeCl3, a Co2 complex, Fe-porphine, 21
                      aromatics.  Rejecting every rise stalls in flat regions and
                      at the noise level of the CI energy; discarding the history
                      on a rejection brings back a two-step cycle.)
    kramers_check     require interleaved, phase +1 Kramers pairs at entry.
    kramers_project   project every rotation onto its time-reversal even part.
    descent_guard     replace a step that is not a descent direction by the
                      Fock-diagonal preconditioned gradient step.
    bfgs_start_grad   with bfgs=True, the L-BFGS pairs are collected and used only
                      once the gradient norm is below this value.
    damp_overshoot    (experimental, off) halve the step when the last step showed
                      a curvature above 1.5 times the model's.  With a step s
                      times the model's Newton step, rho = (2 - c s) / (2 - s)
                      gives the curvature ratio c along it.
    bfgs_keep_history (experimental, off) keep the L-BFGS pairs after a step that
                      raised the energy instead of discarding them.
    gen_g_hop_fn      a replacement for gen_g_hop with the same return values
                      (zmc_sosci uses it to run this macro iteration with the
                      SO-SCI Hessian).

    The orbital basis is not rotated between iterations (mc.canonicalize_ and
    mc.natorb are not needed: metric, Hessian and preconditioner are basis
    covariant), which keeps the L-BFGS history and the Kramers pairing intact.
    '''
    for key in deprecated:
        if key in ('uphill_step_tol', 'ah_lambda', 'step_control', 'no_basis_precond', 'grad_scale'):
            print(f'mcscf_superci: argument {key} is no longer used and is ignored')
        else:
            raise TypeError(f'mcscf_superci got an unexpected keyword argument {key!r}')
    if trust_radius is None:
        trust_radius = 0.2 if max_stepsize is None else max_stepsize
    davidson_mmax = davidson_maxiter
    log = logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol is None:
        conv_tol = mc.conv_tol
    mol = mc.mol
    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas

    # the Kramers projection of the step is only meaningful for orbitals laid
    # out as interleaved, phase +1 Kramers pairs.  Orbitals that lost this
    # (plain-eigh canonicalisation, sort_mo, an unrestricted run) would
    # silently have most of each step projected away.
    if kramers_check:
        from socutils.tools import kramers
        s_ovlp = mol.intor('int1e_ovlp_spinor')
        dev = kramers.pairing_deviation(mol, mo, s_ovlp)
        log.info('Kramers pairing of the starting orbitals: max deviation %.2e', dev)
        if dev > 1e-6:
            raise RuntimeError(
                'starting orbitals are not interleaved, phase +1 Kramers pairs '
                '(max deviation %.2e); phase-clean them (e.g. a CAHF pass with '
                'trace_orbital=False) or use socutils.tools.kramers.symmetrize, '
                'or pass kramers_check=False' % dev)

    # Rotating the orbital basis between iterations is not a harmless option:
    # natorb costs a second integral half-transform and a second CI solve per
    # macro iteration; both rotations invalidate the L-BFGS history; and their
    # diagonalisations fall back to a plain eigh when the Kramers check fails,
    # which scrambles the pairing the Kramers projection relies on.  Neither is
    # needed: metric, Hessian and preconditioner are basis covariant.
    if getattr(mc, 'natorb', False) or getattr(mc, 'canonicalize_', False):
        log.warn('mc.natorb = %s, mc.canonicalize_ = %s: the orbital basis is rotated every macro '
                 'iteration.  This is not needed by this driver; it discards the L-BFGS history each '
                 'time, natorb doubles the integral transformation and CI work, and a non-Kramers '
                 'diagonalisation can scramble the Kramers pairing.  Leave both False unless you '
                 'know you need them.', mc.natorb, mc.canonicalize_)
    TIMINGS.clear()
    _t = time.perf_counter()
    eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
    _t = _tick('integral half-transform (_CDERIS)', _t)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    _t = _tick('core J-K for the CASCI', _t)
    print('first CI calculation')
    e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
    _t = _tick('CI solver', _t)
    mc.e_tot, mc.e_cas = e_tot, e_cas
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mc, 'Set conv_tol_grad to %g', conv_tol_grad)

    conv = False
    norm_gorb = -1
    de = e_tot
    t1m = log.timer('Initializing Super-CI based MCSCF', *cput0)
    _t = time.perf_counter()
    casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
    _tick('make_rdm12', _t)
    norm_rot = 0.0
    t2m = log.timer('CAS DM', *t1m)

    BFGS_SUBSPACE = 6
    BFGS_GRAD_MAX = bfgs_start_grad      # no L-BFGS while |g| is above this
    imacro = 0
    xs, ys, dens = [], [], []
    g_prev = x_prev = None
    rejected = False
    curv_last = 1.0          # curvature along the last step in units of the model's
    e_last = e_tot
    last_drop = None         # the last energy decrease that was achieved
    de_accepted = de
    radius_cap = trust_radius_max
    nstall = 0
    stalled = False

    def finalize_step(x):
        '''The step actually taken: Kramers projected, inside the trust region.'''
        dr = mc.unpack_uniq_var(x)
        if kramers_project:
            # assumes interleaved, phase +1 Kramers pairs (checked at entry;
            # exp(dr) with a time-reversal even dr keeps the pairing)
            dr = ensure_kramers(dr)
        if norm(dr) > trust_radius:
            print('step rescaled')
            dr = dr * (trust_radius / norm(dr))
            print(norm(dr))
        return dr, mc.pack_uniq_var(dr)

    while not conv and imacro < mc.max_cycle_macro:
        if mc.natorb is True:
            # optional: natural active orbitals.  casdm1[t,u] = <t^dagger u>
            # transforms as c^T D c^* under mo -> mo c, so the natural orbitals
            # are the eigenvectors of its transpose.
            moa = mo[:, ncore:nocc]
            natocc, c = mc._scf.eig(-casdm1.T, mo=moa)
            mo[:, ncore:nocc] = np.dot(moa, c)
            eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
            t2m = log.timer('update eris', *t2m)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
            e_nat_tot, e_cas, fcivec = mci.kernel(mo, ci0=None, verbose=verbose)
            print(e_tot, e_nat_tot)
            casdm1, casdm2 = mci.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
            print('natural occupation number in cas space', casdm1.diagonal())
            xs, ys, dens = [], [], []     # the history is in the old basis
            g_prev = x_prev = None

        _t = time.perf_counter()
        g, h_diag, hop, sop, precond, mo = (gen_g_hop_fn or gen_g_hop)(mc, mo, casdm1, casdm2, eris)
        _tick('gen_g_hop (total)', _t)
        if getattr(mc, 'canonicalize_', False):
            xs, ys, dens = [], [], []     # gen_g_hop rotated the core / virtual orbitals
            g_prev = x_prev = None
        norm_gorb = norm(g)
        print(f'Iter {imacro:3d}: E = {e_tot:20.15f}  dE = {de:12.10f}' +
              f'  norm(grad) = {norm_gorb:8.6f} step_size = {norm_rot:8.6f}')
        t2m = log.timer('Compute gradient', *t2m)
        g_unpack = mc.unpack_uniq_var(g)
        row, col = np.unravel_index(np.argmax(g_unpack), g_unpack.shape)
        for i in range(nmo):
            for j in range(i):
                if abs(g_unpack[i,j])>1e-2:
                    print(f'{i}, {j}, {g_unpack[i,j]}')
        if mc.irrep is not None:
            print(f'{row}, {col}, {g_unpack[row,col].real:8.6f} {mc.irrep[row]}, {mc.irrep[col]}')
        else:
            print(f'{row}, {col}, {g_unpack[row,col].real:8.6f}')
        if abs(de) < conv_tol and norm_gorb < conv_tol_grad:
            conv = True
            break

        t_solve = (logger.process_clock(), logger.perf_counter())
        # L-BFGS history: (step taken, gradient difference) of accepted steps
        if imacro > 0 and not rejected and norm_gorb < BFGS_GRAD_MAX and g_prev is not None:
            den = np.dot((g - g_prev).conj(), x_prev).real
            if den > 0:
                ys.append(g - g_prev)
                xs.append(x_prev)
                dens.append(den)
            else:
                print(f'bfgs: pair skipped, y.x = {den:.4e} <= 0')
            if len(ys) > BFGS_SUBSPACE:
                ys.pop(0); xs.pop(0); dens.pop(0)
        if imacro > 0 and de > 0.0 and not bfgs_keep_history:
            # the energy rose: discard the history (as KKW do)
            xs, ys, dens = [], [], []
        use_bfgs = bfgs is True and norm_gorb < BFGS_GRAD_MAX and len(xs) > 0

        gbar = g
        if use_bfgs:
            gbar, a = lbfgs_precondition(g, xs, ys, dens)
        x_ah = hx_ah = None
        if solver == 'gmres':
            x, _ = gmres(hop, -gbar, M=precond, maxiter=50)
        elif solver == 'davidson':
            # restricted-step AH; the packed vector has norm |dr| / sqrt(2)
            x, e, hx_ah = davidson(hop, gbar, h_diag, sop=sop, max_stepsize=trust_radius / np.sqrt(2.),
                                   mmax=davidson_mmax, use_lambda=True, root=ah_root,
                                   precond=getattr(hop, 'precond_shift', None), with_hx=True)
            x_ah = x
        else:
            raise ValueError(f'unknown solver {solver}')
        x_qn = mu_qn = None
        if use_bfgs:
            x = lbfgs_postprocess(x, xs, ys, dens, a)
            if x_ah is not None:
                # level shift of the AH solution: H x_ah = -gbar + mu S x_ah
                mu_qn = (np.vdot(x_ah, hx_ah + gbar).real
                         / max(np.vdot(x_ah, sop(x_ah)).real, 1e-300))
                x_qn = x
        if damp_overshoot and curv_last > 1.5:
            print(f'damping: the last step saw {curv_last:.2f} times the model curvature, step halved')
            x = 0.5 * x
        t2m = log.timer('Solving Super-CI equation', *t_solve)
        _tick('AH Davidson + L-BFGS (total)', t_solve[1])

        dr, x = finalize_step(x)
        # descent guard: the AH step is not guaranteed to go downhill once it
        # has been post-processed.  If the step taken is not a descent
        # direction of the first-order model, fall back to the Fock-diagonal
        # preconditioned gradient step.
        slope = np.dot(g.conj(), x).real
        if descent_guard and slope >= 0:
            x = -g / np.maximum(h_diag.real, sd_hdiag_floor)
            dr, x = finalize_step(x)
            print(f'descent guard (step not downhill, Re(g.x) = {slope:.3e}): preconditioned '
                  f'gradient step, |dr| = {norm(dr):.4e}, Re(g.x) = {np.dot(g.conj(), x).real:.3e}')
        for i in range(nmo):
            for j in range(i):
                if abs(dr[i,j])>1e-2:
                    print(f'{i}, {j}, dr {dr[i,j]}')

        _t = time.perf_counter()
        mo_new = np.dot(mo, expmat(dr))
        _t = _tick('expm of the rotation', _t)
        eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi, level=2)
        _t = _tick('integral half-transform (_CDERIS)', _t)
        t2m = log.timer('update eris', *t2m)
        mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        _t = _tick('core J-K for the CASCI', _t)
        e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)
        _t = _tick('CI solver', _t)

        # trust region: compare with the second-order model of the step taken
        de = e_tot - e_last
        _t = time.perf_counter()
        # <x, H x> of the step taken, only from what is already there: the
        # images stored by the Davidson solver when the step is its solution up
        # to a real factor, the L-BFGS model when the step was post-processed
        # (the Kramers projection removes roundoff of order 1e-13 in absolute
        # terms, which is a sizeable fraction of a step near convergence: the
        # match is tested to 1e-6, far below anything that matters for Q)
        STEP_MATCH_TOL = 1e-6
        xhx = None
        fac = 1.0
        if x_qn is not None:
            fac = np.vdot(x_qn, x).real / max(np.vdot(x_qn, x_qn).real, 1e-300)
            if norm(x - fac * x_qn) <= STEP_MATCH_TOL * norm(x):
                xhx = fac**2 * (-np.vdot(g, x_qn).real + mu_qn * np.vdot(x_qn, sop(x_qn)).real)
        elif x_ah is not None:
            fac = np.vdot(x_ah, x).real / max(np.vdot(x_ah, x_ah).real, 1e-300)
            if norm(x - fac * x_ah) <= STEP_MATCH_TOL * norm(x):
                xhx = fac**2 * np.vdot(x_ah, hx_ah).real
        q_model = None if xhx is None else 2. * np.vdot(g, x).real + xhx
        norm_rot = norm(dr)
        if q_model is None:
            rho = None
            curv_last = 1.0
            print(f'Energy change {de:.4e}, no model energy for this step')
            print(f'trust region: |dr| = {norm_rot:.4e}, trust radius = {trust_radius:.4e}')
        else:
            rho = de / q_model if q_model < 0 else -1.0
            curv_last = min(max((2. - rho * (2. - fac)) / fac, 1.0), 4.0) if (q_model < 0 and 0 < fac < 1.999) else 1.0
            print(f'Energy change {de:.4e}, predicted change {q_model:.4e}')
            print(f'trust region: |dr| = {norm_rot:.4e}, trust radius = {trust_radius:.4e}, rho = {rho:.3f}')
        # a rise is large when it exceeds the noise level and a fraction of the last
        # decrease (before the first decrease: the gain the model promised)
        if last_drop is not None:
            big_rise = de > max(reject_noise, reject_ratio * last_drop)
        else:
            big_rise = de > max(reject_noise, abs(q_model) if (q_model is not None and q_model < 0) else 0.0)
        r_min = min(trust_radius_min, norm_gorb)
        if big_rise and reject_uphill:
            nstall = nstall + 1 if norm_rot <= 1.001 * r_min else 0
            trust_radius = max(0.5 * norm_rot, r_min)
            radius_cap = trust_radius      # do not grow straight back to the radius that failed
            print(f'step rejected (dE = {de:.4e}), trust radius -> {trust_radius:.4e}')
            if nstall >= stall_limit:
                print(f'stopped: {nstall} consecutive steps at the minimum trust radius {r_min:.3e} raised the energy')
                stalled = True
                e_tot = e_last
                break
            eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
            rejected = True
            e_tot = e_last
            de = de_accepted               # the L-BFGS history is kept
            imacro += 1
            t1m = log.timer(f'macro iter {imacro}', *t1m)
            continue
        nstall = 0
        if de > conv_tol:
            trust_radius = 0.5 * norm_rot
            print(f'uphill step accepted (dE = {de:.4e} > 0), trust radius -> {trust_radius:.4e}')
        elif q_model is None or abs(q_model) < conv_tol:
            pass      # no model, or changes below conv_tol that say nothing about it
        elif rho < 0.25:
            trust_radius = 0.5 * norm_rot
            print(f'rho < 0.25: trust radius -> {trust_radius:.4e}')
        elif rho > 0.75 and norm_rot > 0.9 * trust_radius:
            radius_cap = min(1.25 * radius_cap, trust_radius_max)
            trust_radius = min(2. * trust_radius, trust_radius_max, radius_cap)
            print(f'rho > 0.75 at the boundary: trust radius -> {trust_radius:.4e}')
        trust_radius = max(trust_radius, r_min)
        if de < 0:
            last_drop = -de
        de_accepted = de

        rejected = False
        mo = mo_new
        e_last = e_tot
        x_prev = x
        g_prev = g
        _t = time.perf_counter()
        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        _t = _tick('make_rdm12', _t)
        imacro += 1
        t1m = log.timer(f'macro iter {imacro}', *t1m)
        lib.chkfile.save(mc.chkfile, f'mo_coeff_iter_{imacro}', mo)
        _t = _tick('chkfile save of the orbitals', _t)
        if kramers_check:
            print('Kramers pairing deviation of the orbitals after macro iter %d: %.3e'
                  % (imacro, kramers.pairing_deviation(mol, mo, s_ovlp)))
            _t = _tick('Kramers pairing check', _t)
        if verbose >= logger.INFO:
            mc.e_tot = e_tot
            mc.e_cas = e_cas
            mc._finalize()
    mc.mo_coeff = mo
    print_timings('mcscf_superci', wall=time.perf_counter() - cput0[1])
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
