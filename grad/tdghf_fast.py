#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Fast Z-vector TDA excited-state gradient (socutils.grad.tdghf_grad.grad_elec) of a two-component reference:
the same equations, every term evaluated where the ground-state fast gradient evaluates it.

* Responses (V = G_td[X^H], V_T = G_0[T], the Z-vector orbital Hessian, G_0 of the occupied-overlap term) come
  from the SCF the TDA ran on: for a spinor reference (spinor_ghf.to_ghf_tda) the spinor SCF's own response,
  i.e. its RIJ / COSX J / K and fused two-component f_xc, inside the spinor TDA's response settings.  MO-basis
  quantities are the same in both representations; AO matrices go between the spinor and the GHF AO basis by
  the unitary U = spinor2ghf_coeff (M_ghf = U M_sp U^H).
* The occupied-overlap response Re <Z, C_v^H G_0[delta_A] C_o>, delta_A = -C_o S^A_oo C_o^H (3 natm response
  densities in grad_elec) is Tr[S^A_oo C_o^H G_0[N_h] C_o] with one response of N_h = Herm(C_o Z^H C_v^H)
  (Tr[G[A] B] = Tr[G[B] A]); with the other overlap terms it is one AO contraction with S^A.
* The hcore term Re Tr[h^A W] of all atoms is one X2C adjoint evaluation (with_x2c.nuc_grad_hcore) of
  W_h = Herm(W).
* The two-electron terms Re Tr[G'[D0] W] and Re Tr[G'[X] X^H] are written as quadratic forms
  Q[H] = Tr[G[H] H] of Hermitian matrices (polarisation for the first, X = H1 + i H2 for the second), and Q of
  a Hermitian GHF matrix as E_J / E_K of real symmetric and anti-symmetric scalar matrices (_quad_pieces); their
  derivatives come from the scalar fast kernels as in socutils.grad.kr_fast (exact: the summed unique-quartet
  kernels; RIJ: hessian.rhf_fast.ri_j_grad on the spinor SCF's RIJ proxy; COSX: COSX.grad on the response grid).
* The XC terms (d V_xc / dR contracted with W, and the AO-motion derivative of the kernel term
  E_f = int rho_X^* f rho_X) are 'potential x derivative density' contractions int u d rho_M / dR on one grid
  pass (_xc_terms), with the C helpers of grad.rhf_fast.xc_grad.

Grid response (td_grad.grid_response, 'auto' by default): the XC terms where the partition supports it
(socutils.grad.grid_response), COSX on request (td_grad.cosx_grid_response).

grad_elec returns None (the caller keeps tdghf_grad.grad_elec) when a piece is not available: an XC grid
response on a partition socutils.grad.grid_response lacks, range-separated or NLC functionals, density-fitted SCF, Cartesian basis, missing libraries.
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf

ENABLE = True
PIECE_TOL = 1e-14       # relative size below which a scalar piece of Q[H] is dropped


# ---------------------------------------------------------------- quadratic forms
def _add_piece(pieces, M, jw, kw, tol):
    '''add jw E_J[M] + kw E_K[M]; a matrix proportional to one already listed (M = l M') merges (E[l M'] =
    l^2 E[M'])'''
    amax = abs(M).max()
    if amax <= tol:
        return
    for p in pieces:
        Mp = p[0]
        i = numpy.unravel_index(numpy.argmax(abs(Mp)), Mp.shape)
        lam = M[i] / Mp[i]
        if abs(M - lam * Mp).max() <= 1e-12 * amax:
            p[1] += jw * lam * lam
            p[2] += kw * lam * lam
            return
    pieces.append([numpy.ascontiguousarray(M), jw, kw])


def _quad_pieces(pieces, H, nao, cj, ck, sign=1., tol=None):
    '''append to pieces the scalar terms of sign * Q[H], Q[H] = Tr[(cj J - ck K)[H] H] of a Hermitian GHF
    matrix H (2nao, 2nao), with E_J[M] = 1/2 (ab|cd) M_ab M_cd and E_K[M] = 1/4 sum (ab|cd) M_ad M_bc:

        Q[H] = 2 cj E_J[rho] - 4 ck sum_{st} (E_K[Re H_st] + E_K[Im H_st]),  rho = sym Re (H_aa + H_bb),

    each real block split into its symmetric and anti-symmetric parts (their cross terms vanish), the ba block
    equal to the ab one (H_ba = H_ab^H, E_K[M^T] = E_K[M])'''
    H = numpy.asarray(H)
    scale = abs(H).max()
    if tol is None:
        tol = PIECE_TOL * max(scale, 1e-300)
    aa, ab, bb = H[:nao, :nao], H[:nao, nao:], H[nao:, nao:]
    if cj != 0:
        r = (aa + bb).real
        _add_piece(pieces, (r + r.T) * .5, 2 * cj * sign, 0., tol)
    if ck != 0:
        for blk, mult in ((aa, 1.), (bb, 1.), (ab, 2.)):
            for m in (blk.real, blk.imag):
                s = (m + m.T) * .5
                a = (m - m.T) * .5
                _add_piece(pieces, s, 0., -4 * ck * mult * sign, tol)
                _add_piece(pieces, a, 0., -4 * ck * mult * sign, tol)
    return pieces


def _herm(m):
    return (m + m.conj().T) * .5


def _jk_grad(td_grad, mol, pieces, cosx_gr=False):
    '''sum_m d/dR (jw_m E_J[M_m] + kw_m E_K[M_m]) (natm, 3) by the scalar fast engines in the J / K mode of the
    SCF (kr_fast routes), None if not available'''
    from socutils.grad import kr_fast, rhf_fast
    from socutils.hessian import rhf_fast as hess_fast
    from socutils.scf import spinor_jk
    mf = td_grad.base._scf
    m = kr_fast._jk_mode(mf)
    tol, prim_tol = rhf_fast.GRAD2E_TOL, rhf_fast.GRAD2E_PRIM_TOL
    jlist = [(p[0], p[1]) for p in pieces if p[1] != 0]
    klist = [(p[0], p[2]) for p in pieces if p[2] != 0]
    de = numpy.zeros((mol.natm, 3))
    t0 = (logger.process_clock(), logger.perf_counter())
    exact_j = exact_k = True
    if m.startswith('rij') and jlist:
        src = kr_fast._source(mf)
        p = getattr(src, '_spinor_rij_proxy', None)
        if p is None or p.mol is not mol:
            return None
        cache = getattr(p, '_rij_cache', None) or {}
        key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
        cderi = cache.get('B') if cache.get('key') == key else None
        engine = cache.get('direct') if cache.get('key') == key else None
        auxmol = engine.auxmol if engine is not None else p.with_df.auxmol
        for M, w in jlist:
            r = hess_fast.ri_j_grad(mol, auxmol, M, tol=tol, cderi=cderi, engine=engine,
                                    max_memory=max(1000, mf.max_memory - lib.current_memory()[0]))
            if r is None:
                return None
            de += w * r
        exact_j = False
        t0 = logger.timer(td_grad, 'TDA gradient: RIJ J (%d densities)' % len(jlist), *t0)
    if m.endswith('cosx') and klist:
        from socutils.scf import cosx
        src = kr_fast._source(mf)
        st = getattr(src, '_spinor_cosx', None)
        k = None
        if st is not None and st.get('mol') is mol and cosx.available(mol):
            # the grid of the response K (the excitation energy's and the SCF orbitals')
            k = spinor_jk._kobj(src, st, spinor_jk.RESPONSE_GRID)
        if k is None:
            return None
        # COSX.grad(M) = d/dR <M, P K_plain[M]> = 4 dE_K[M]/dR (kr_fast)
        for M, w in klist:
            de += .25 * w * k.grad(M, grid_response=cosx_gr)
        exact_k = False
        t0 = logger.timer(td_grad, 'TDA gradient: COSX K (%d densities, %d points%s)' % (
            len(klist), k._weights.size, ', grid response' if cosx_gr else ''), *t0)
    mats = []
    if exact_j:
        mats += [(M, w, 0.) for M, w in jlist]
    if exact_k:
        mats += [(M, 0., w) for M, w in klist]
    if exact_j and exact_k:
        # J and K of the same matrix in one entry
        mats = [(p[0], p[1], p[2]) for p in pieces]
    if mats:
        summed = getattr(hess_fast, 'grad2e_jk_cart_summed', None)
        for i0 in range(0, len(mats), 8):
            chunk = mats[i0:i0 + 8]
            g = None
            if summed is not None:
                g = summed(mol, [c[0] for c in chunk], (1., 1.), jscale=[c[1] for c in chunk],
                           kscale=[c[2] for c in chunk], tol=tol, prim_tol=prim_tol)
            if g is None:
                g = numpy.zeros((mol.natm, 3))
                for M, jw, kw in chunk:
                    r = hess_fast.grad2e_jk_cart(mol, M, [(jw, kw)], tol=tol, prim_tol=prim_tol)
                    if r is None:
                        return None
                    g += r[0]
            de += g
        logger.timer(td_grad, 'TDA gradient: exact %s (%d densities)' % (
            'J + K' if exact_j and exact_k else ('J' if exact_j else 'K'), len(mats)), *t0)
    return de


# ---------------------------------------------------------------- XC
def _pot_grad(libdft, aoT, cs, wv, D, xctype, out, pb=None):
    '''out (3, nao) += the AO sums of int wv d rho_D / dR (the integrand of grad.rhf_fast.xc_grad, wv the
    weighted potential rows with its GGA / tau factors, D symmetric); pb (3, ng) += its per-point AO sums (the
    point-motion part of the grid response, as grad.rhf_fast.xc_grad)'''
    nao, ng = aoT.shape[1], aoT.shape[2]
    nw = 1 if xctype == 'LDA' else 4
    itype = {'LDA': 0, 'GGA': 1, 'MGGA': 2}[xctype]
    null = lib.c_null_ptr()
    wv = numpy.ascontiguousarray(wv)
    aow = numpy.empty((nao, ng))
    libdft.SOCDFTxc_grad_aow(aow.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p),
                             wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nw), ctypes.c_int(ng),
                             ctypes.c_int(nao), ctypes.c_long(cs))
    U = lib.dot(D, aow)
    V = lib.dot(D, aoT[0]) if xctype != 'LDA' else None
    T = None
    if xctype == 'MGGA':
        T = numpy.empty((3, nao, ng))
        for k in range(3):
            lib.dot(D, aoT[1 + k] * wv[4], c=T[k])
    libdft.SOCDFTrks_xc_grad(out.ctypes.data_as(ctypes.c_void_p), aoT.ctypes.data_as(ctypes.c_void_p),
                             U.ctypes.data_as(ctypes.c_void_p),
                             null if V is None else V.ctypes.data_as(ctypes.c_void_p),
                             wv.ctypes.data_as(ctypes.c_void_p),
                             null if T is None else T.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(itype), ctypes.c_int(ng), ctypes.c_int(nao), ctypes.c_long(cs))
    if pb is not None:
        from socutils.hessian import rhf_fast as hess_fast
        for x in range(3):
            pb[x] += numpy.einsum('ig,ig->g', aoT[1 + x], U)
            if xctype != 'LDA':
                pb[x] += numpy.einsum('ig,ig->g', aoT[1 + x], V) * wv[0]
                for k in range(3):
                    pb[x] += numpy.einsum('ig,ig->g', aoT[hess_fast._D2[x][k]], V) * wv[1 + k]
            if xctype == 'MGGA':
                for k in range(3):
                    pb[x] += numpy.einsum('ig,ig->g', aoT[hess_fast._D2[x][k]], T[k])


def _scale_pot(wv, xctype):
    '''the factors grad.rhf_fast.xc_grad applies to the potential rows (rho row of GGA / MGGA, tau row)'''
    wv = numpy.array(wv, dtype=numpy.double, copy=True)
    if xctype != 'LDA':
        wv[0] *= .5
    if xctype == 'MGGA':
        wv[4] *= .5
    return wv


def _xc_terms(td_grad, mol, D0, Wh, X, max_memory=2000, grid_response=False):
    '''(natm, 3): Re Tr[d V_xc / dR (fixed D0) W_h] (SCF functional) + the AO-motion derivative of
    E_f = int rho_X^* f rho_X at fixed X, D0 (response kernel), on the SCF grid without grid response:

        int v[rho0] d rho_W + int (f[rho0] rho_W + Y) d rho0 + 2 Re sum_u int W^u d rho_X^u,

    W^u = sum_v f_uv rho_X^v*, Y = sum k rho_X^* rho_X (tdghf_grad._kernel_contractions), rho_W the charge
    density of W_h (a Kramers closed-shell reference: V_xc and d rho0 / dR are spin free).  grid_response:
    the grid moving with the atoms (socutils.grad.grid_response: the owner-atom point term of the per-point
    sums and the partition-weight derivative of the energy density v rho_W + rho_X^* f rho_X).  None if not
    available.'''
    from pyscf.dft import numint
    from socutils.dft import noncollinear
    from socutils.grad import tdghf_grad
    from socutils.hessian import rhf_fast as hess_fast
    from socutils.hessian import xc_grid_response as xgr
    td = td_grad.base
    mf = td._scf
    libdft = hess_fast._load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTrks_xc_grad'):
        return None
    ni2 = mf._numint
    xc0 = mf.xc
    xck = td_grad.kernel_xc()
    ni = numint.NumInt()
    xctype = ni._xc_type(xc0)
    if xctype not in ('LDA', 'GGA', 'MGGA') or ni._xc_type(xck) != xctype:
        return None
    nao = mol.nao_nr()
    nvar = xgr._nvar(xctype)
    spin_channels = [2] if td_grad.response_collinear()[0] == 'c' else [0, 1, 2]
    d0c = numpy.ascontiguousarray((D0[:nao, :nao] + D0[nao:, nao:]).real)
    d0c = (d0c + d0c.T) * .5
    wc = (Wh[:nao, :nao] + Wh[nao:, nao:]).real
    wc = numpy.ascontiguousarray((wc + wc.T) * .5)
    ds = noncollinear._rho_mag_dms(X)              # 4 components x (Re, Im), symmetric
    dsn = [abs(d).max() > PIECE_TOL * max(abs(X).max(), 1e-300) for d in ds]
    grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if grid_response:
        from socutils.grad import grid_response as gresp
        if not gresp.available(grids):
            return None
        ngrid = grids.weights.size
        pgrid = numpy.zeros((3, ngrid))
        egrid = numpy.zeros(ngrid)
    p0 = 0
    ao_deriv = 1 if xctype == 'LDA' else 2
    out = numpy.zeros((3, nao))
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ng = weight.size
        pb = pgrid[:, p0:p0 + ng] if grid_response else None
        a0 = ao[0] if xctype == 'LDA' else ao[:4]
        rho0 = numint.eval_rho(mol, a0, d0c, mask, xctype, hermi=1, with_lapl=False).reshape(nvar, -1)
        rhow = numint.eval_rho(mol, a0, wc, mask, xctype, hermi=1, with_lapl=False).reshape(nvar, -1)
        vxc, fxc = ni.eval_xc_eff(xc0, rho0, 2, xctype=xctype)[1:3]
        vxc = vxc.reshape(nvar, -1)
        fxc = fxc.reshape(nvar, nvar, -1)
        rhoX = noncollinear.eval_rho_2c(mol, a0, X, mask, xctype, 0, False).reshape(4, nvar, -1)
        W1, Y1 = tdghf_grad._kernel_contractions(ni2, xck, xctype, rhoX, rho0, spin_channels,
                                                 numpy.ones(ng), nvar)[:2]
        W, Y = W1 * weight, Y1 * weight
        aoT, cs = hess_fast._ao_stack(ao)
        _pot_grad(libdft, aoT, cs, _scale_pot(vxc * weight, xctype), wc, xctype, out, pb)
        u0 = numpy.einsum('xyg,yg->xg', fxc, rhow) * weight + Y
        _pot_grad(libdft, aoT, cs, _scale_pot(u0, xctype), d0c, xctype, out, pb)
        for u in range(4):
            if dsn[2 * u]:
                _pot_grad(libdft, aoT, cs, _scale_pot(2 * W[u].real, xctype), ds[2 * u], xctype, out, pb)
            if dsn[2 * u + 1]:
                _pot_grad(libdft, aoT, cs, _scale_pot(-2 * W[u].imag, xctype), ds[2 * u + 1], xctype, out, pb)
        if grid_response:
            egrid[p0:p0 + ng] = (numpy.einsum('xg,xg->g', vxc, rhow)
                                 + numpy.einsum('uxg,uxg->g', W1, rhoX).real)
        p0 += ng
    de = numpy.zeros((mol.natm, 3))
    for ia, (a0, a1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = -2 * out[:, a0:a1].sum(axis=1)
    if grid_response:
        de += 2 * gresp.owner_sum(grids, pgrid, mol.natm) + gresp.partition_grad(grids, egrid, mol)
    return de


# ---------------------------------------------------------------- one-electron terms
def _ovlp_grad(mol, M):
    '''(natm, 3): Re Tr[S^A M] of a GHF AO matrix M (S^A the spin-diagonal overlap derivative)'''
    nao = mol.nao_nr()
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    mc = M[:nao, :nao] + M[nao:, nao:]
    msym = (mc + mc.T).real
    de = numpy.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = numpy.einsum('xij,ji->x', s1a[:, p0:p1], msym[:, p0:p1])
    return de


def _hcore_grad(td_grad, mol, Wh):
    '''(natm, 3): Tr[h^A W_h], the X2C hcore term by one adjoint evaluation (as ghf_grad), else per atom'''
    from socutils.grad import ghf_grad
    mf = td_grad.base._scf
    g = mf.Gradients()
    dh = ghf_grad._x2c_hcore_grad(g, mol, Wh)
    if dh is not None:
        return numpy.asarray(dh).real
    from socutils.hessian import ghf as ghf_hess
    hcore_deriv = g.hcore_generator(mol)
    nao = mol.nao_nr()
    de = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        h1 = hcore_deriv(ia)
        if h1.shape[-1] == nao:
            h1 = ghf_hess._block_diag(h1, nao)
        de[ia] = numpy.einsum('xpq,qp->x', h1, Wh).real
    return de


# ---------------------------------------------------------------- driver
def _grid_response_mode(td_grad, mf):
    '''(xc, cosx) grid-response switches: td_grad.grid_response True / False both on / off; 'auto' the XC
    response where the partition supports it (socutils.grad.grid_response), the COSX one off;
    td_grad.cosx_grid_response (True / False) overrides the COSX part'''
    g = td_grad.grid_response
    cgr = getattr(td_grad, 'cosx_grid_response', None)
    if isinstance(g, str) and g == 'auto':
        xc = False
        if isinstance(mf, hf.KohnShamDFT):
            from socutils.grad import grid_response as gresp
            grids = mf.grids
            xc = grids.coords is not None and gresp.available(grids)
        return xc, bool(cgr)
    return bool(g), bool(g) if cgr is None else bool(cgr)



def _source(td_grad):
    '''(the TDA the response comes from, its SCF, U): the spinor TDA / SCF of a spinor reference and
    U = spinor2ghf_coeff, else the GHF TDA itself and U = None'''
    td = td_grad.base
    tds = getattr(td, '_spinor_td', None)
    if tds is not None:
        from socutils.scf import spinor_ghf
        mfs = tds._scf
        U = spinor_ghf.spinor2ghf_coeff(mfs.mol, numpy.eye(mfs.mo_coeff.shape[0]))
        return tds, mfs, U
    return td, td._scf, None


def applicable(td_grad):
    if not ENABLE:
        return False
    td = td_grad.base
    mf = td._scf
    mol = mf.mol
    if mol.cart or getattr(mf, 'with_df', None) is not None:
        return False
    tds, mfs, U = _source(td_grad)
    if getattr(mfs, 'with_df', None) is not None:
        return False
    if isinstance(mf, hf.KohnShamDFT):
        if mf.do_nlc():
            return False
        ni = mf._numint
        for xc in {mf.xc, td_grad.kernel_xc()}:
            omega = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)[0]
            if omega != 0:
                return False
    return True


def grad_elec(td_grad, z, atmlst=None, verbose=logger.INFO):
    '''d omega / dR of the TDA state with amplitudes z (nocc, nvir): tdghf_grad.grad_elec on the fast engines
    (module doc); None if not applicable'''
    from socutils.hessian import ghf as ghf_hess
    if not applicable(td_grad):
        return None
    log = logger.new_logger(td_grad, verbose)
    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    td = td_grad.base
    mf = td._scf
    mol = mf.mol
    nao = mol.nao_nr()
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    tds, mfs, U = _source(td_grad)
    is_ks = isinstance(mf, hf.KohnShamDFT)

    def to_ghf(m):
        return m if U is None else U.dot(m).dot(U.conj().T)

    def to_src(m):
        return m if U is None else U.conj().T.dot(m).dot(U)

    C = mfs.mo_coeff
    mo_energy = mfs.mo_energy
    mo_occ = mfs.mo_occ
    occ = mo_occ > 0
    vir = ~occ
    oi = numpy.where(occ)[0]
    vi = numpy.where(vir)[0]
    nmo = C.shape[1]
    Co, Cv = C[:, occ], C[:, vir]
    z = numpy.asarray(z).reshape(oi.size, vi.size)

    t = numpy.zeros((nmo, nmo), dtype=numpy.complex128)
    t[vi[:, None], vi] = z.T.dot(z.conj())
    t[oi[:, None], oi] = -z.conj().dot(z.T)
    x = numpy.zeros((nmo, nmo), dtype=numpy.complex128)
    x[vi[:, None], oi] = z.T
    X = C.dot(x).dot(C.conj().T)
    T = C.dot(t).dot(C.conj().T)
    D0 = mfs.make_rdm1(C, mo_occ)
    Fmo = numpy.diag(mo_energy).astype(numpy.complex128)     # canonical SCF orbitals
    X_g, D0_g = to_ghf(X), to_ghf(D0)

    import contextlib
    from socutils.scf import spinor_jk
    with contextlib.ExitStack() as env:
        env.enter_context(tds._response_env() if U is not None else td_grad._response_env())
        # single Hermitian densities below are response densities (spinor_jk: response grid, no SCF increments)
        env.enter_context(spinor_jk.response_densities(mfs))
        vresp0 = mfs.gen_response(C, mo_occ, hermi=1)
        vresp_td = tds.gen_response(hermi=0)
        V = vresp_td(X.conj().T[None])[0]
        VT = vresp0(T[None])[0]
        if is_ks:
            from socutils.grad.tdghf_grad import kxc_potential
            VT = VT + to_src(kxc_potential(td_grad, X_g, D0_g))
        t1 = log.timer('TDA gradient: responses V, V_T', *t1)
        Vmo = C.conj().T.dot(V).dot(C)
        VTmo = C.conj().T.dot(VT).dot(C)
        Po = numpy.diag(occ.astype(float))
        Lam = 2 * (t.dot(Fmo) + Po.dot(VTmo) + x.dot(Vmo) + x.conj().T.dot(Vmo.conj().T))
        L = Lam[oi[:, None], vi].conj().T - Lam[vi[:, None], oi]

        fx = ghf_hess.gen_vind(mfs, C, mo_occ)
        h1 = numpy.zeros((1, nmo, oi.size), dtype=numpy.complex128)
        h1[0, vi] = -L
        Z = ghf_hess.cphf_solve(fx, mo_energy, mo_occ, h1, numpy.zeros_like(h1),
                                max_cycle=50, tol=mf.conv_tol_cpscf, verbose=log)[0][0][vi]
        t1 = log.timer('TDA gradient: Z-vector', *t1)
        N = Co.dot(Z.conj().T).dot(Cv.conj().T)
        GN = vresp0(_herm(N)[None])[0]
        t1 = log.timer('TDA gradient: occupied-overlap response', *t1)

    # overlap terms: Re Tr[A^H S^mo] = Re Tr[S C A^H C^H] summed over
    #   -Re <Z, -S_vo e_o>, -Re <Z, C_v^H G_0[delta] C_o>, +1/2 Re <L, S_vo>, -1/2 Re Tr[Lam S]
    e_o = mo_energy[occ]
    A = -.5 * Lam.conj().T
    A[vi[:, None], oi] += Z * e_o + .5 * L
    A[oi[:, None], oi] += Co.conj().T.dot(GN).dot(Co)
    Mao = to_ghf(C.dot(A.conj().T).dot(C.conj().T))
    de = _ovlp_grad(mol, Mao)

    W = T - Co.dot(Z.conj().T).dot(Cv.conj().T)
    Wh_g = to_ghf(_herm(W))
    de += _hcore_grad(td_grad, mol, Wh_g)
    t1 = log.timer('TDA gradient: hcore and overlap terms', *t1)

    # two-electron terms: Re Tr[G'[D0] W] (SCF functional), Re Tr[G'[X] X^H] (response kernel)
    ni = getattr(mf, '_numint', None)

    def jk_coef(xc):
        if not is_ks:
            return 1.
        return ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)[2] if ni.libxc.is_hybrid_xc(xc) else 0.

    c0 = jk_coef(mf.xc) if is_ks else 1.
    ck = jk_coef(td_grad.kernel_xc()) if is_ks else 1.
    pieces = []
    _quad_pieces(pieces, (D0_g + Wh_g) * .5, nao, 1., c0, 1.)
    _quad_pieces(pieces, (D0_g - Wh_g) * .5, nao, 1., c0, -1.)
    _quad_pieces(pieces, _herm(X_g), nao, 1., ck)
    _quad_pieces(pieces, (X_g - X_g.conj().T) * (-.5j), nao, 1., ck)
    xc_gr, cosx_gr = _grid_response_mode(td_grad, mf)
    g2 = _jk_grad(td_grad, mol, pieces, cosx_gr)
    if g2 is None:
        logger.info(td_grad, 'TDA gradient: no fast two-electron route, GHF path')
        return None
    de += g2
    t1 = log.timer('TDA gradient: two-electron terms (%d scalar pieces)' % len(pieces), *t1)

    if is_ks:
        dx = _xc_terms(td_grad, mol, D0_g, Wh_g, X_g,
                       max_memory=max(2000, mf.max_memory * .9 - lib.current_memory()[0]), grid_response=xc_gr)
        if dx is None:
            logger.info(td_grad, 'TDA gradient: no fast XC route, GHF path')
            return None
        de += dx
        t1 = log.timer('TDA gradient: XC terms%s' % (' (grid response)' if xc_gr else ''), *t1)
    log.timer('TDA excitation-energy gradient (fast)', *t0)
    return de[atmlst]
