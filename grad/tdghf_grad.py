#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Analytic nuclear gradient of a TDA excited state on a GHF reference (complex,
two-component spin orbitals; X2C spin-orbit hcore welcome).

With amplitudes z[i,a] (pyscf.tdscf.ghf convention, normalized), the
transition density X = C_v z^T C_o^H and the unrelaxed difference density
T = C_v z^T z* C_v^H - C_o z* z^T C_o^H, the excitation energy is

    omega = Tr[F T] + Tr[G[X] X^H],      G[D] = J[D] - K[D].

z is variational, so d omega/dR is taken at fixed z:

    d omega = Tr[F^(R) T]                    F^(R): AO derivative at fixed D0
            + Tr[G^(R)[X] X^H]               integral derivative
            + Tr[G[dD0] T]                   reference-density response
            + Tr[F dT] + 2 Re Tr[G[dX] X^H]  orbital response of T and X

with the orbital response C^(R) = C U from coupled-perturbed GHF (U_vo), the
orthonormality fixing U_oo = U_vv = -S/2 and U_ov = -U_vo^H - S_ov.

This first version solves the coupled-perturbed equations for all 3N nuclear
displacements (reusing socutils.hessian.ghf) instead of a single Z-vector
equation: simple and exact, O(N) more response solves.
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from socutils.hessian import ghf as ghf_hess
import socutils.grad.ghf_grad  # noqa: F401


def _orbital_response(hessobj, mo_energy, mo_coeff, mo_occ, atmlst, verbose):
    '''Full U^(R) (nmo x nmo, per atom: (3, nmo, nmo)) and the first-order
    Fock matrices h1ao at fixed density.'''
    mf = hessobj.base
    mol = hessobj.mol
    nao = mol.nao_nr()
    occ = mo_occ > 0
    vir = ~occ
    C = mo_coeff
    h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, verbose)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    mocc = C[:, occ]
    fx = ghf_hess.gen_vind(mf, mo_coeff, mo_occ)
    h1vo = numpy.vstack([[C.conj().T.dot(x).dot(mocc) for x in h1ao[ia]] for ia in atmlst])
    s1mo = {ia: numpy.asarray([C.conj().T.dot(x).dot(C) for x in ghf_hess._s1ao(mol, ia, s1a, nao)])
            for ia in atmlst}
    s1vo = numpy.vstack([s1mo[ia][:, :, occ] for ia in atmlst])
    mo1, e1 = ghf_hess.cphf_solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                                  max_cycle=hessobj.max_cycle, tol=mf.conv_tol_cpscf,
                                  verbose=verbose)
    mo1 = mo1.reshape(len(atmlst), 3, C.shape[1], occ.sum())
    oi = numpy.where(occ)[0]
    vi = numpy.where(vir)[0]
    U = {}
    for k, ia in enumerate(atmlst):
        s1 = s1mo[ia]
        u = numpy.zeros(s1.shape, dtype=numpy.complex128)
        u[:, :, oi] = mo1[k]                                   # U_vo and U_oo = -S_oo/2
        u[:, vi[:, None], vi] = -.5 * s1[:, vi[:, None], vi]
        u[:, oi[:, None], vi] = (-mo1[k][:, vi, :].conj().transpose(0, 2, 1)
                                 - s1[:, oi[:, None], vi])
        U[ia] = u
    return U, h1ao


def grad_elec(td_grad, z, atmlst=None, verbose=logger.INFO):
    '''d omega / dR of the TDA state with amplitudes z (nocc, nvir), with a
    single Z-vector equation.

    The orbital response enters omega linearly:  Delta(U) = Re Tr[Lambda U],
        Lambda = 2 (t F + P_o V_T + x V + x^H V^H)            (MO basis)
    with t, x the MO difference / transition densities, V = G_td[X^H] and
    V_T = G_0[T] (+ the kxc potential of the kernel term), using
    Tr[G[A] B^H] = Tr[G[B^H] A].  With U = kappa - S/2 (kappa anti-Hermitian,
    vir-occ) and the coupled-perturbed equations A U_vo = -B'(R),
        Re Tr[Lambda U] = -Re <Z, B'> + Re <L, S_vo>/2 - Re Tr[Lambda S]/2,
        A Z = L,   L_ai = Lambda_ia^* - Lambda_ai,
    B'_vo = (F^R - S^R e_o)_vo + v[-C_o S^R_oo C_o^H]_vo.  A (the orbital
    Hessian, real-linear in U) is symmetric for Re <.,.>.'''
    log = logger.new_logger(td_grad, verbose)
    td = td_grad.base
    mf = td._scf
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    nao = mol.nao_nr()
    C = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occ = mo_occ > 0
    vir = ~occ
    oi = numpy.where(occ)[0]
    vi = numpy.where(vir)[0]
    nmo = C.shape[1]
    Co = C[:, occ]
    z = numpy.asarray(z).reshape(oi.size, vi.size)

    t = numpy.zeros((nmo, nmo), dtype=numpy.complex128)
    t[vi[:, None], vi] = z.T.dot(z.conj())
    t[oi[:, None], oi] = -z.conj().dot(z.T)
    x = numpy.zeros((nmo, nmo), dtype=numpy.complex128)
    x[vi[:, None], oi] = z.T
    X = C.dot(x).dot(C.conj().T)
    T = C.dot(t).dot(C.conj().T)
    Fmo = C.conj().T.dot(mf.get_fock()).dot(C)

    vresp0 = mf.gen_response(C, mo_occ, hermi=1)
    vresp_td = td.gen_response(hermi=0)
    # Tr[G[A] B^H] = Tr[G[B^H] A]  (checked numerically for the GHF kernels)
    V = vresp_td(X.conj().T[None])[0]
    VT = vresp0(T[None])[0]                    # T is Hermitian
    is_ks = isinstance(mf, hf.KohnShamDFT)
    D0 = mf.make_rdm1()
    if is_ks:
        VT = VT + kxc_potential(td_grad, X, D0)
    Vmo = C.conj().T.dot(V).dot(C)
    VTmo = C.conj().T.dot(VT).dot(C)
    Po = numpy.diag(occ.astype(float))
    Lam = 2 * (t.dot(Fmo) + Po.dot(VTmo) + x.dot(Vmo) + x.conj().T.dot(Vmo.conj().T))
    L = Lam[oi[:, None], vi].conj().T - Lam[vi[:, None], oi]

    # Z-vector: A Z = L  (cphf_solve solves A u = -h1)
    fx = ghf_hess.gen_vind(mf, C, mo_occ)
    h1 = numpy.zeros((1, nmo, oi.size), dtype=numpy.complex128)
    h1[0, vi] = -L
    Z = ghf_hess.cphf_solve(fx, mo_energy, mo_occ, h1, numpy.zeros_like(h1),
                            max_cycle=50, tol=mf.conv_tol_cpscf, verbose=verbose)[0][0][vi]
    log.timer_debug1('Z-vector')

    hessobj = mf.Hessian()
    if td_grad.grid_response and hasattr(hessobj, 'grid_response'):
        hessobj.grid_response = True
        hessobj.grid_response_method = td_grad.grid_response_method
    h1ao = hessobj.make_h1(C, mo_occ, None, atmlst, verbose)
    v1x = td_grad.veff_deriv1(X, atmlst)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    e_o = mo_energy[occ]

    # occupied-occupied part of the reference-density response, all at once
    s1mo = {ia: numpy.asarray([C.conj().T.dot(s).dot(C) for s in ghf_hess._s1ao(mol, ia, s1a, nao)])
            for ia in atmlst}
    delta = numpy.asarray([-Co.dot(s1mo[ia][xx][oi[:, None], oi]).dot(Co.conj().T)
                           for ia in atmlst for xx in range(3)])
    gdelta = vresp0(delta).reshape(len(atmlst), 3, 2 * nao, 2 * nao)

    de = numpy.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        for xx in range(3):
            Fr = C.conj().T.dot(h1ao[ia][xx]).dot(C)
            Sr = s1mo[ia][xx]
            Bp = (Fr[vi[:, None], oi] - Sr[vi[:, None], oi] * e_o
                  + C[:, vi].conj().T.dot(gdelta[k, xx]).dot(Co))
            de[k, xx] = (numpy.einsum('pq,qp', h1ao[ia][xx], T).real
                         + numpy.einsum('pq,qp', v1x[ia][xx], X.conj().T).real
                         - numpy.vdot(Z, Bp).real
                         + .5 * numpy.vdot(L, Sr[vi[:, None], oi]).real
                         - .5 * numpy.einsum('pq,qp', Lam, Sr).real)
    if is_ks:
        de += xc_kernel_terms(td_grad, X, D0, None, atmlst)
    log.timer('TDA-GHF excitation-energy gradient (Z-vector)')
    return de


def kxc_potential(td_grad, X, D0):
    '''AO potential matrix M[Y] of Y_w = sum_uv k_uvw rho_X^u* rho_X^v (charge
    channel), so that int Y rho[dD] = Tr[M[Y] dD] for the reference-density
    response through the kernel term.'''
    from pyscf.dft import numint2c
    from socutils.hessian import xc_grid_response as xgr
    td = td_grad.base
    mf = td._scf
    mol = mf.mol
    ni = mf._numint
    xc = td_grad.kernel_xc()
    xctype = ni._xc_type(xc)
    nao = mol.nao_nr()
    m = numpy.zeros((nao, nao))
    if xctype == 'HF':
        return numpy.zeros((2 * nao, 2 * nao))
    spin_channels = [2] if td_grad.response_collinear()[0] == 'c' else [0, 1, 2]
    nrow = 1 if xctype == 'LDA' else 4
    for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, 0 if xctype == 'LDA' else 1):
        a = ao if xctype == 'LDA' else ao[:4]
        rows = a.reshape(nrow, -1, nao)
        rhoX = numint2c.eval_rho(mol, a, X, mask, xctype, 0, False).reshape(4, nrow, -1)
        rho0 = numint2c.eval_rho(mol, a, D0, mask, xctype, 1, False).reshape(4, nrow, -1)[0].real
        Y = _kernel_contractions(ni, xc, xctype, rhoX, rho0, spin_channels, weight, nrow)[1]
        m += xgr._vmat(rows, Y, xctype)
    out = numpy.zeros((2 * nao, 2 * nao))
    out[:nao, :nao] = m
    out[nao:, nao:] = m
    return out


def grad_elec_cphf(td_grad, z, atmlst=None, verbose=logger.INFO):
    '''d omega / dR of the TDA state with amplitudes z (nocc, nvir), solving
    the coupled-perturbed equations for all 3N displacements (reference
    implementation for grad_elec).'''
    log = logger.new_logger(td_grad, verbose)
    td = td_grad.base
    mf = td._scf
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    C = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    occ = mo_occ > 0
    vir = ~occ
    Co = C[:, occ]
    Cv = C[:, vir]
    z = numpy.asarray(z).reshape(occ.sum(), vir.sum())

    X = Cv.dot(z.T).dot(Co.conj().T)
    T = Cv.dot(z.T).dot(z.conj()).dot(Cv.conj().T) - Co.dot(z.conj()).dot(z.T).dot(Co.conj().T)
    fock = mf.get_fock()

    hessobj = mf.Hessian()
    if td_grad.grid_response and hasattr(hessobj, 'grid_response'):
        hessobj.grid_response = True
        hessobj.grid_response_method = td_grad.grid_response_method
    U, h1ao = _orbital_response(hessobj, mo_energy, C, mo_occ, atmlst, verbose)
    log.timer_debug1('orbital response')

    # integral derivative of Tr[G[X] X^H]
    v1x = td_grad.veff_deriv1(X, atmlst)

    vresp0 = mf.gen_response(C, mo_occ, hermi=1)       # reference-density response
    vresp_td = td.gen_response(hermi=0)                # TDA kernel (G[X])

    de = numpy.zeros((len(atmlst), 3))
    dD0s = {}
    for k, ia in enumerate(atmlst):
        dC = numpy.einsum('pq,xqr->xpr', C, U[ia])
        dCo = dC[:, :, occ]
        dCv = dC[:, :, vir]
        dD0 = numpy.einsum('xpi,qi->xpq', dCo, Co.conj())
        dD0 = dD0 + dD0.conj().transpose(0, 2, 1)
        zt = z.T
        zz_v = zt.dot(z.conj())                         # (nvir, nvir)
        zz_o = z.conj().dot(zt)                         # (nocc, nocc)
        dT = (numpy.einsum('xpa,ab,qb->xpq', dCv, zz_v, Cv.conj())
              + numpy.einsum('pa,ab,xqb->xpq', Cv, zz_v, dCv.conj())
              - numpy.einsum('xpi,ij,qj->xpq', dCo, zz_o, Co.conj())
              - numpy.einsum('pi,ij,xqj->xpq', Co, zz_o, dCo.conj()))
        dX = (numpy.einsum('xpa,ai,qi->xpq', dCv, zt, Co.conj())
              + numpy.einsum('pa,ai,xqi->xpq', Cv, zt, dCo.conj()))
        g_d0 = vresp0(dD0)
        g_dx = vresp_td(dX)
        dD0s[ia] = dD0
        for x in range(3):
            de[k, x] = (numpy.einsum('pq,qp', h1ao[ia][x], T)
                        + numpy.einsum('pq,qp', v1x[ia][x], X.conj().T)
                        + numpy.einsum('pq,qp', g_d0[x], T)
                        + numpy.einsum('pq,qp', fock, dT[x])
                        + 2 * numpy.einsum('pq,qp', g_dx[x], X.conj().T).real).real
    if isinstance(mf, hf.KohnShamDFT):
        de += xc_kernel_terms(td_grad, X, mf.make_rdm1(), dD0s, atmlst)
    log.timer('TDA-GHF excitation-energy gradient')
    return de


# rows of the pyscf AO array (deriv=2): d/dx_i of row r
_D1 = {(0, 0): 1, (0, 1): 2, (0, 2): 3,
       (1, 0): 4, (1, 1): 5, (1, 2): 6,
       (2, 0): 5, (2, 1): 7, (2, 2): 8,
       (3, 0): 6, (3, 1): 8, (3, 2): 9}


def xc_kernel_terms(td_grad, X, D0, dD0s, atmlst):
    '''Exchange-correlation-kernel part of d omega/dR not carried by the
    response functions:  E_f = int sum_uv rho_X^u* f_uv(rho0) rho_X^v  with
    u, v in (rho, mx, my, mz) x (value, gradient rows), evaluated at fixed AO
    matrices X, D0 (AO motion: rho_X and rho0) plus the reference-density
    response rho[dD0] through the third derivative.

    Kramers closed shell: m0 = 0, so the kernel is block diagonal --
    f_tt (charge) and f_ss for each spin component of the response kernel
    (x, y, z for ncol/mcol, z only for a collinear kernel) -- and the third
    derivatives needed are k_ttt and k_sst = d f_ss / d rho, all collinear
    derivatives at s = 0.  AO-motion derivatives of the densities use the
    exact polarization identity for a quadratic form,
        d rho / dR = [rho(ao + dao) - rho(ao - dao)] / 2.
    No grid response.'''
    from pyscf.dft import numint2c
    from socutils.dft import noncollinear
    td = td_grad.base
    mf = td._scf
    mol = mf.mol
    ni = mf._numint
    xc = td_grad.kernel_xc()
    xctype = ni._xc_type(xc)
    natm = len(atmlst)
    if xctype == 'HF':
        return numpy.zeros((natm, 3))
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError('TDA-GKS gradient for %s functionals' % xctype)
    spin_channels = [2] if td_grad.response_collinear()[0] == 'c' else [0, 1, 2]
    nao = mol.nao_nr()
    aoslices = mol.aoslice_by_atom()
    ao_deriv = 1 if xctype == 'LDA' else 2
    nrow = 1 if xctype == 'LDA' else 4
    eval_rho = numint2c.eval_rho
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])

    def rho_of(ao_rows, dm, mask, hermi):
        a = ao_rows[0] if xctype == 'LDA' else ao_rows
        r = eval_rho(mol, a, dm, mask, xctype, hermi, False)
        return r.reshape(4, nrow, -1)

    grid_response = td_grad.grid_response
    de = numpy.zeros((natm, 3))
    for ao, mask, weight, coords in ni.block_loop(mol, mf.grids, nao, ao_deriv, max_memory):
        a0 = ao[:nrow]
        rhoX = rho_of(a0, X, mask, 0)
        rho0 = rho_of(a0, D0, mask, 1)[0].real
        W, Y = _kernel_contractions(ni, xc, xctype, rhoX, rho0, spin_channels, weight, nrow)[:2]
        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia][2:]
            for x in range(3):
                if dD0s is not None:
                    # reference-density response (the Z-vector code carries it in V_T)
                    rho_resp = rho_of(a0, dD0s[ia][x], mask, 1)[0].real
                    de[i0, x] += numpy.einsum('xg,xg', Y, rho_resp)
                if grid_response:
                    continue            # AO and grid motion: finite difference of E_f below
                dao = numpy.zeros_like(a0)
                for r in range(nrow):
                    dao[r][:, p0:p1] = -ao[_D1[r, x]][:, p0:p1]
                dX = .5 * (rho_of(a0 + dao, X, mask, 0) - rho_of(a0 - dao, X, mask, 0))
                drho0 = .5 * (rho_of(a0 + dao, D0, mask, 1) - rho_of(a0 - dao, D0, mask, 1))[0].real
                de[i0, x] += 2 * numpy.einsum('uxg,uxg', W, dX).real
                de[i0, x] += numpy.einsum('xg,xg', Y, drho0)
    if grid_response:
        if td_grad.grid_response_method == 'fd':
            de += _ef_geometric_fd(td_grad, X, D0, atmlst, spin_channels, xc, xctype)
        else:
            de += _ef_geometric_analytic(td_grad, X, D0, atmlst, spin_channels, xc, xctype)
    return de


def _kernel_contractions(ni, xc, xctype, rhoX, rho0, spin_channels, weight, nrow):
    '''W^u = sum_v f_uv rho_X^v*, Y_w = sum_uv k_uvw rho_X^u* rho_X^v (weighted)
    and the energy density of E_f, for the block-diagonal kernel at m0 = 0.'''
    from socutils.dft import noncollinear
    rho_ts = numpy.stack([rho0, numpy.zeros_like(rho0)])
    f, k = noncollinear.eval_xc_ts(ni, xc, rho_ts, 3, xctype)[2:4]
    f = f.reshape(2, nrow, 2, nrow, -1)
    k = k.reshape(2, nrow, 2, nrow, 2, nrow, -1)
    f_tt, f_ss = f[0, :, 0], f[1, :, 1]
    k_ttt, k_sst = k[0, :, 0, :, 0], k[1, :, 1, :, 0]
    cX = rhoX.conj()
    W = numpy.zeros_like(rhoX)
    W[0] = numpy.einsum('xyg,yg->xg', f_tt, cX[0])
    Y = numpy.einsum('xywg,xg,yg->wg', k_ttt, cX[0], rhoX[0]).real
    for c in spin_channels:
        W[1 + c] = numpy.einsum('xyg,yg->xg', f_ss, cX[1 + c])
        Y += numpy.einsum('xywg,xg,yg->wg', k_sst, cX[1 + c], rhoX[1 + c]).real
    e_f = numpy.einsum('uxg,uxg->g', W, rhoX).real
    return W * weight, Y * weight, numpy.dot(e_f, weight)


def ef_energy(td_grad, mol, grids, X, D0, spin_channels, xc, xctype):
    '''E_f = int sum_uv rho_X^u* f_uv(rho0) rho_X^v on the given grid.'''
    from pyscf.dft import numint2c
    mf = td_grad.base._scf
    ni = mf._numint
    nao = mol.nao_nr()
    nrow = 1 if xctype == 'LDA' else 4
    e = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 0 if xctype == 'LDA' else 1):
        a = ao if xctype == 'LDA' else ao[:4]
        rhoX = numint2c.eval_rho(mol, a, X, mask, xctype, 0, False).reshape(4, nrow, -1)
        rho0 = numint2c.eval_rho(mol, a, D0, mask, xctype, 1, False).reshape(4, nrow, -1)[0].real
        e += _kernel_contractions(ni, xc, xctype, rhoX, rho0, spin_channels, weight, nrow)[2]
    return e


def _ef_geometric_analytic(td_grad, X, D0, atmlst, spin_channels, xc, xctype):
    '''d E_f / dR at fixed AO matrices X, D0 with the full grid response:
    Becke weight derivatives and the motion of the points through the AO
    coefficients c_A = delta_A,owner - delta_A,atom(mu)
    (socutils.hessian.xc_grid_response).'''
    from pyscf.dft import numint2c
    from socutils.hessian import xc_grid_response as xgr
    from socutils.grad import becke_deriv
    mf = td_grad.base._scf
    mol = mf.mol
    ni = mf._numint
    nrow = 1 if xctype == 'LDA' else 4
    rows = xgr._rows(xctype)
    ao_deriv = 1 if xctype == 'LDA' else 2

    def rho_of(ao_rows, dm, hermi):
        a = ao_rows[0] if xctype == 'LDA' else ao_rows
        return numint2c.eval_rho(mol, a, dm, None, xctype, hermi, False).reshape(4, nrow, -1)

    natm = mol.natm
    de = numpy.zeros((natm, 3))
    for owner, (coords, w, w1, w2) in enumerate(becke_deriv.becke_weight_derivs(mf.grids)):
        c = xgr._coef(mol, owner)
        for p0, p1 in lib.prange(0, w.size, 2000):
            ao = ni._to_numint1c().eval_ao(mol, coords[p0:p1], deriv=ao_deriv)
            ao0 = numpy.asarray([xgr._d(ao, r) for r in rows])
            rhoX = rho_of(ao0, X, 0)
            rho0 = rho_of(ao0, D0, 1)[0].real
            W, Y, _ = _kernel_contractions(ni, xc, xctype, rhoX, rho0, spin_channels,
                                           numpy.ones(p1 - p0), nrow)
            cX = rhoX.conj()
            e_f = numpy.einsum('uxg,uxg->g', W, rhoX).real
            for A in range(natm):
                for x in range(3):
                    ap = xgr._displaced_ao(ao, rows, c[A], x, 1.)
                    am = xgr._displaced_ao(ao, rows, c[A], x, -1.)
                    dX = .5 * (rho_of(ap, X, 0) - rho_of(am, X, 0))
                    d0 = .5 * (rho_of(ap, D0, 1) - rho_of(am, D0, 1))[0].real
                    dens = 2 * numpy.einsum('uxg,uxg->g', W, dX).real + numpy.einsum('xg,xg->g', Y, d0)
                    de[A, x] += numpy.dot(w1[A, x, p0:p1], e_f) + numpy.dot(w[p0:p1], dens)
            del cX
    return de[atmlst]


def _ef_geometric_fd(td_grad, X, D0, atmlst, spin_channels, xc, xctype, step=1e-3):
    '''d E_f / dR at fixed AO matrices X, D0, with the grid moving with the
    atoms: five-point difference (the grid response of the kernel term).'''
    import copy
    mf = td_grad.base._scf
    mol = mf.mol
    coords = mol.atom_coords()
    de = numpy.zeros((len(atmlst), 3))
    for i0, ia in enumerate(atmlst):
        for x in range(3):
            for n, c in ((2, -1 / 12.), (1, 8 / 12.), (-1, -8 / 12.), (-2, 1 / 12.)):
                xyz = coords.copy()
                xyz[ia, x] += n * step
                mold = mol.set_geom_(xyz, unit='Bohr', inplace=False)
                g = copy.copy(mf.grids)
                g.reset(mold)
                g.build(with_non0tab=True)
                de[i0, x] += c * ef_energy(td_grad, mold, g, X, D0, spin_channels, xc, xctype) / step
    return de


class Gradients(lib.StreamObject):
    '''Excited-state gradient (ground-state gradient + d omega/dR)'''

    # response of the integration grid in the XC terms (semi-numerical: the
    # first-order Fock matrices of socutils.hessian.gks with grid_response and
    # a finite difference of the kernel term E_f at fixed AO matrices)
    grid_response = False
    grid_response_method = 'analytic'       # or 'fd'
    # 'zvector' (one response solve) or 'cphf' (3N solves; reference)
    method = 'zvector'

    def __init__(self, td):
        self.base = td
        self.mol = td.mol
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.state = 1                  # 1-based, as in pyscf
        self.de = None

    def kernel_xc(self):
        '''Functional of the response kernel (socutils TDA: xc_kernel).'''
        mf = self.base._scf
        return getattr(self.base, 'xc_kernel', None) or getattr(mf, 'xc', 'HF')

    def response_collinear(self):
        '''Spin scheme of the response kernel used by the TDA solve.'''
        td = self.base
        if hasattr(td, 'response_collinear'):
            col = td.response_collinear()
            if col is not None:
                return col
        ni = getattr(td._scf, '_numint', None)
        return getattr(ni, 'collinear', 'col')

    def veff_deriv1(self, X, atmlst):
        '''Integral derivative of (J - c_x K)[X] for the kernel functional.'''
        mf = self.base._scf
        mol = self.mol
        if not isinstance(mf, hf.KohnShamDFT):
            return ghf_hess.veff_deriv1(mol, X, atmlst, hermi=0)
        ni = mf._numint
        xc = self.kernel_xc()
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(xc)
        v1 = ghf_hess.veff_deriv1(mol, X, atmlst, with_k=hybrid, k_scale=hyb, hermi=0)
        if hybrid and omega != 0:
            with mol.with_range_coulomb(omega):
                vlr = ghf_hess.veff_deriv1(mol, X, atmlst, with_k=True, k_scale=alpha - hyb, hermi=0)
                vj = ghf_hess.veff_deriv1(mol, X, atmlst, with_k=False, hermi=0)
            for ia in atmlst:
                v1[ia] += vlr[ia] - vj[ia]
        return v1

    def _response_env(self):
        '''The response settings of the TDA solve (socutils TDA: xc_kernel,
        non-collinear kernel) must hold for the gradient as well.'''
        import contextlib
        td = self.base
        if hasattr(td, '_response_env'):
            return td._response_env()
        return contextlib.nullcontext()

    def grad_omega(self, state=None, atmlst=None):
        if state is None:
            state = self.state
        z = self.base.xy[state - 1][0]
        fn = grad_elec_cphf if self.method == 'cphf' else grad_elec
        with self._response_env():
            return fn(self, z, atmlst, self.verbose)

    def kernel(self, state=None, atmlst=None):
        mf = self.base._scf
        g0 = mf.Gradients().kernel(atmlst=atmlst)
        self.de = g0 + self.grad_omega(state, atmlst)
        return self.de
