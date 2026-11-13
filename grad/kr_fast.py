#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Two-electron and XC parts of the nuclear gradient of a Kramers-restricted
(closed-shell, time-reversal-even) two-component SCF / KS from the scalar
fast engines of socutils.grad.rhf_fast.

The molecular two-electron integrals of X2C / X2CAMF are spin free.  For a
Hermitian time-reversal-even GHF density (D_bb = D_aa^*, D_ba = -D_ab^*),
D_aa = R + iA, D_ab = P + iQ with R symmetric and A, P, Q anti-symmetric,

    E_J = 1/2 (ab|cd) rho_ab rho_cd,             rho = 2R,
    E_x = -1/2 tr(D K[D]) = -sum_{X in R, A, P, Q} E_K[2X],
    E_K[X] = 1/4 sum (ab|cd) X_ad X_bc

(the cross terms between a symmetric and an anti-symmetric matrix vanish,
and for an anti-symmetric X the eight permutations of a quartet give the
same weight X_ad X_bc + X_ac X_bd as for a symmetric one, so the scalar
gradient kernels take the anti-symmetric pieces as they are).  The XC
energy depends on rho = 2R only (the spin density of a Kramers pair is zero
point by point), so it is the closed-shell RKS functional of rho.

    exact J        hessian.rhf_fast.md_j_grad / grad2e_jk_cart (rho)
    exact K        hessian.rhf_fast.grad2e_jk_cart, the four pieces in one
                   integral pass (one density per group) where libsocvhf
                   has it, else one pass per piece
    RIJ            hessian.rhf_fast.ri_j_grad on the spinor SCF's RIJ proxy
                   (its auxiliary basis, resident factor / direct engine)
    COSX           scf.cosx.COSX.grad of the four pieces on the spinor
                   SCF's final COSX grid (spinor_jk.FINAL_GRID)
    XC             grad.rhf_fast.xc_grad (the fused LDA / GGA / MGGA pass,
                   grid response as rhf_fast: 'auto')

terms() returns None (the caller keeps the GHF path) for densities that are
not Kramers even, range-separated functionals, NLC, density-fitted SCF, or
when a library is missing.
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf

ENABLE = True       # ghf_grad / gks_grad: the KR fast path (else the 8-density GHF J/K derivative)
KRAMERS_TOL = 1e-9
# exact K: 'kernel' (the unique-quartet kernels, the pieces as groups of one pass: their cost grows ~ linearly
# with the groups, the contraction dominates), 'deriv' (pyscf's int2e_ip1 K derivative matrices of the four
# pieces in one libcvhf pass, contracted per atom), 'auto': 'kernel' where the generated kernels take every
# quartet (l <= 2, <= 4 contractions), else 'deriv'.  (HgCl2)2 x2c-SVPall (nao 210, 16 cores): kernel 103 s,
# deriv (4 pieces, J + K) 40 s, the 8-density GHF J/K 28-61 s.
EXACT_K = 'auto'


def pieces(dm, nao, tol=KRAMERS_TOL):
    '''(R, A, P, Q) of a Hermitian time-reversal-even GHF density (2nao, 2nao), or None'''
    dm = numpy.asarray(dm)
    if dm.ndim != 2 or dm.shape != (2 * nao, 2 * nao):
        return None
    aa, ab = dm[:nao, :nao], dm[:nao, nao:]
    ba, bb = dm[nao:, :nao], dm[nao:, nao:]
    scale = max(abs(aa).max(), abs(ab).max(), 1e-300)
    if (abs(bb - aa.conj()).max() > tol * scale or abs(ba + ab.conj()).max() > tol * scale
            or abs(aa - aa.conj().T).max() > tol * scale or abs(ab + ab.T).max() > tol * scale):
        return None
    R = numpy.ascontiguousarray((aa.real + aa.real.T) * .5)
    A = numpy.ascontiguousarray((aa.imag - aa.imag.T) * .5)
    P = numpy.ascontiguousarray((ab.real - ab.real.T) * .5)
    Q = numpy.ascontiguousarray((ab.imag - ab.imag.T) * .5)
    return R, A, P, Q


def _source(mf):
    '''the spinor SCF a GHF object was converted from (spinor_ghf.to_ghf), else mf itself'''
    return getattr(mf, '_spinor_src', None) or mf


def _jk_mode(mf):
    '''the J / K mode of the SCF the density came from: '', 'rij', 'cosx', 'rijcosx' '''
    src = _source(mf)
    if not hasattr(src, 'rijcosx'):
        return ''
    from socutils.scf import spinor_jk
    return spinor_jk.mode(src)


def _scalar_ks(mf, grad):
    '''a scalar RKS proxy (pyscf NumInt) of the 2c KS mf on its grid, and its rhf_fast gradient object'''
    from pyscf import dft
    from socutils.grad import rhf_fast
    p = dft.RKS(mf.mol, xc=mf.xc)
    p.grids = mf.grids
    p.verbose = mf.verbose
    p.stdout = mf.stdout
    p.max_memory = mf.max_memory
    pg = rhf_fast.RKSGradients(p)
    pg.grids = getattr(grad, 'grids', None)
    pg.grid_response = grad.grid_response
    pg.verbose = grad.verbose
    pg.stdout = grad.stdout
    pg.max_memory = grad.max_memory
    return p, pg


def _anti_k_grad(mol, mats, hyb, tol, prim_tol, j_first=True):
    '''sum_i grad of [jf_i E_J - hyb E_K](mats[i]) with jf = 1 for the first matrix (if j_first), else 0,
    by the unique-quartet kernels; mats symmetric or anti-symmetric (natm, 3) or None'''
    from socutils.hessian import rhf_fast as hess_fast
    groups = [((1. if (i == 0 and j_first) else 0.), -hyb) for i in range(len(mats))]
    multi = getattr(hess_fast, 'grad2e_jk_cart_multi', None)
    if multi is not None and len(mats) > 1:
        g = multi(mol, mats, groups, tol=tol, prim_tol=prim_tol)
        if g is not None:
            return g.sum(axis=0)
    de = numpy.zeros((mol.natm, 3))
    for m, grp in zip(mats, groups):
        g = hess_fast.grad2e_jk_cart(mol, m, [grp], tol=tol, prim_tol=prim_tol)
        if g is None:
            return None
        de += g[0]
    return de


def _exact_k_route(mol):
    if EXACT_K != 'auto':
        return EXACT_K
    return 'kernel' if mol._bas[:, 1].max() <= 2 and mol._bas[:, 3].max() <= 4 else 'deriv'


def _deriv_k_grad(mol, mats, hyb):
    '''-hyb sum_X grad E_K[X] (natm, 3) from pyscf's K derivative matrices of the (symmetric or
    anti-symmetric) mats = 2X in one integral pass: d E_K[2X] / dR_A = 4 sum_{i in A} sum_j vk[X]_ij X_ij,
    vk[X] = ((-nabla i) j | k l) X_jk (pyscf.grad.rhf.get_k)'''
    from pyscf.grad import rhf as rhf_grad
    x = numpy.ascontiguousarray(numpy.asarray(mats) * .5)
    vk = rhf_grad.get_k(mol, x)
    de = numpy.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        for n in range(len(mats)):
            de[ia] -= hyb * 4 * numpy.einsum('xij,ij->x', vk[n][:, p0:p1], x[n][p0:p1])
    return de


def _cosx_obj(mf, mol):
    '''the COSX object of the spinor SCF's final grid (spinor_jk.FINAL_GRID, else its SCF grid), None if the
    SCF did not run COSX or no grid represents the basis'''
    from socutils.scf import spinor_jk
    src = _source(mf)
    st = getattr(src, '_spinor_cosx', None)
    if st is None or st.get('mol') is not mol:
        return None
    level = spinor_jk.FINAL_GRID or 2
    return spinor_jk._kobj(src, st, level)


def terms(grad, mol, dm):
    '''(natm, 3): the 2e (J, K) and XC parts of the gradient at the KR GHF density dm; None (the caller's
    GHF path) when not applicable'''
    if not ENABLE or mol.cart:
        return None
    mf = grad.base
    nao = mol.nao_nr()
    pcs = pieces(dm, nao)
    if pcs is None:
        return None
    if getattr(mf, 'with_df', None) is not None:
        return None
    from socutils.grad import rhf_fast
    from socutils.hessian import rhf_fast as hess_fast
    R, A, P, Q = pcs
    rho = 2 * R
    is_ks = isinstance(mf, hf.KohnShamDFT)
    if is_ks:
        if mf.do_nlc():
            return None
        from pyscf.dft import numint as nr_numint
        ni = nr_numint.NumInt()
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=0)
        if omega != 0:
            return None
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
        kp, kgrad = _scalar_ks(mf, grad)
        xc_gr, cosx_gr = rhf_fast._grid_response_mode(kgrad)
        if grad.grid_response is True:
            from socutils.grad import grid_response as gresp
            grids = grad.grids if getattr(grad, 'grids', None) is not None else mf.grids
            if grids.coords is None:
                grids.build(with_non0tab=True)
            if not gresp.available(grids):
                return None
    else:
        hyb, hybrid = 1., True
        g = getattr(grad, 'grid_response', False)
        cosx_gr = bool(g) if not isinstance(g, str) else False
    m = _jk_mode(mf)
    t0 = (logger.process_clock(), logger.perf_counter())
    tol, prim_tol = rhf_fast.GRAD2E_TOL, rhf_fast.GRAD2E_PRIM_TOL
    de = numpy.zeros((mol.natm, 3))
    jdone = False
    kdone = not hybrid
    anti = [2 * X for X in (A, P, Q) if abs(X).max() > 0]
    if m.startswith('rij'):
        src = _source(mf)
        p = getattr(src, '_spinor_rij_proxy', None)
        if p is None or p.mol is not mol:
            return None
        cache = getattr(p, '_rij_cache', None) or {}
        key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
        cderi = cache.get('B') if cache.get('key') == key else None
        engine = cache.get('direct') if cache.get('key') == key else None
        auxmol = engine.auxmol if engine is not None else p.with_df.auxmol
        r = hess_fast.ri_j_grad(mol, auxmol, rho, tol=tol, cderi=cderi, engine=engine,
                                max_memory=max(1000, mf.max_memory - lib.current_memory()[0]))
        if r is None:
            return None
        de += r
        jdone = True
        t0 = logger.timer(grad, 'KR gradient: RIJ J (rho = 2 Re D_aa)', *t0)
    if not kdone and m.endswith('cosx'):
        from socutils.scf import cosx
        k = _cosx_obj(mf, mol) if cosx.available(mol) else None
        if k is not None:
            # E_x = -hyb / 4 sum_X <X, P K_plain[X]> (Frobenius; COSX.grad differentiates that form, which is
            # tr(X P K_plain[X]) for the symmetric 2R and -tr for the anti-symmetric 2A, 2P, 2Q)
            g = k.grad(rho, grid_response=cosx_gr)
            for X in anti:
                g += k.grad(X, grid_response=cosx_gr)
            de += -.25 * hyb * g
            kdone = True
            t0 = logger.timer(grad, 'KR gradient: COSX K (%d pieces, %d points%s)' % (
                1 + len(anti), k._weights.size, ', grid response' if cosx_gr else ''), *t0)
        else:
            logger.info(grad, 'KR gradient: no COSX grid of the SCF, exact K')
    kroute = _exact_k_route(mol)
    if not kdone and not jdone and kroute == 'kernel' and rhf_fast._gen_grad_usable(mol):
        r = _anti_k_grad(mol, [rho] + anti, hyb, tol, prim_tol, j_first=True)
        if r is None:
            return None
        de += r
        jdone = kdone = True
        t0 = logger.timer(grad, 'KR gradient: exact J + K (%d pieces)' % (1 + len(anti)), *t0)
    if not jdone:
        r = hess_fast.md_j_grad(mol, rho)
        if r is None:
            r = hess_fast.grad2e_jk_cart(mol, rho, [(1., 0.)], tol=tol, prim_tol=prim_tol)
            if r is None:
                return None
            de += r[0]
        else:
            de += r[0]
        jdone = True
        t0 = logger.timer(grad, 'KR gradient: exact J', *t0)
    if not kdone:
        if kroute == 'deriv':
            de += _deriv_k_grad(mol, [rho] + anti, hyb)
        else:
            r = _anti_k_grad(mol, [rho] + anti, hyb, tol, prim_tol, j_first=False)
            if r is None:
                return None
            de += r
        t0 = logger.timer(grad, 'KR gradient: exact K (%d pieces)' % (1 + len(anti)), *t0)
    if is_ks:
        max_memory = max(2000, grad.max_memory * .9 - lib.current_memory()[0])
        dx = rhf_fast.xc_grad(kgrad, mol, rho, max_memory, grid_response=xc_gr)
        if dx is None:
            return None
        de += dx
        logger.timer(grad, 'KR gradient: XC (fused, rho%s)' % (', grid response' if xc_gr else ''), *t0)
    return de
