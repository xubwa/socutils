#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
pyscf RHF / RKS Hessians with the two-electron second derivatives over the
unique shell quartets (lib/vhf/hess2e.c through
socutils.hessian.ghf.deriv2_energies_direct) instead of pyscf's one pass per
derivative type.  Drop-in classes:

    from socutils.hessian import rhf_fast
    h = rhf_fast.RHFHessian(mf).kernel()        # mf: pyscf scf.RHF
    h = rhf_fast.RKSHessian(mf).kernel()        # mf: pyscf dft.RKS

The 2e parts of the first-order Fock matrices come from one unique-quartet pass
(lib/vhf/fock1.c) and the CPHF response J / K from socutils.scf.jk_multi;
the 1e terms, XC and the CPHF solver are pyscf's.
'''

import ctypes
import sys
import numpy
import scipy.linalg
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess
from socutils.hessian import ghf as ghf_hess


def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True, hyb=None):
    '''pyscf.hessian.rhf._partial_hess_ejk with ej, ek from the unique-quartet
    driver: E_J = 1/2 (ab|cd) D_ba D_dc, E_K = 1/4 (ab|cd) D_bc D_da.  With hyb
    (and with_k) the exact-integral path returns ej - hyb ek as ej and ek = 0
    (one density group: the weights and contractions once); the other paths
    ej, ek as pyscf (the caller's ej - hyb ek holds either way).'''
    log = logger.new_logger(hessobj, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)
    atmlst = list(atmlst)
    dfm = _df_mode(mf)
    if (atmlst != list(range(mol.natm)) or ghf_hess.deriv2_energies_direct(mol, []) is None
            or (dfm == 'jk' and with_k)):
        if dfm is not None:
            from pyscf.df.hessian import rhf as df_rhf_hess
            return df_rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                                 max_memory, verbose, with_k)
        return rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                          max_memory, verbose, with_k)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ > 0]) * 2
    natm = len(atmlst)
    ej = None
    combine = False
    # the K second derivatives by COSX (with the first-order K of make_h1 from the same pass, cached)
    cx = _hess_cosx(mf, HESS_COSX_EXPLICIT) if with_k else None
    ek_cx = None
    if cx is not None:
        t0 = (logger.process_clock(), logger.perf_counter())
        want_f1 = HESS_COSX_FOCK1 is not False and getattr(hessobj, '_h1_occ', H1_OCC)
        r = cx.hess(mocc, with_fock1=bool(want_f1))
        if want_f1:
            ek_cx, f1 = r
            hessobj._cosx_fock1 = (mocc.tobytes(), f1)
        else:
            ek_cx = r
        ek_cx = ek_cx * .25
        with_k = False
        log.timer('K second derivatives (COSX)', *t0)
    if dfm is not None:
        # density-fitted J (exact derivatives of the DF energy), exact K below
        eng = _rij_engine(mf)
        r = ri_j_deriv(mol, eng.auxmol if eng is not None else _df_auxmol(mf), dm0, with_vj1=False,
                       max_memory=max(1000, mf.max_memory - lib.current_memory()[0]), cderi=_rij_cderi(mf),
                       engine=eng)
        if r is None:
            from pyscf.df.hessian import rhf as df_rhf_hess
            return df_rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                                 max_memory, verbose, with_k)
        ej = r[0]
        groups = [(0, 1)] if with_k else []
    else:
        # J and K from the same integral pass (groups 0 and 1), or J - hyb K as one group
        combine = with_k and hyb is not None and COMBINE_JK
        groups = [(1, -hyb)] if combine else [(1, 0), (0, 1)] if with_k else [(1, 0)]
    out = []
    if groups:
        # the combined Cartesian kernel up to g functions, else the
        # libcint-based driver
        out = hess2e_jk_cart(mol, dm0, groups)
        if out is None:
            terms = [] if ej is not None else [(dm0, 'ba', dm0, 'dc', .5, 0)]
            if with_k:
                terms.append((dm0, 'bc', dm0, 'da', -.25 * hyb if combine else .25, 0 if combine else len(terms)))
            out = ghf_hess.deriv2_energies_direct(mol, terms)
    if ej is None:
        ej = out[0]
        ek = out[1] if with_k and not combine else numpy.zeros_like(ej)
    else:
        ek = out[0] if with_k else numpy.zeros_like(ej)
    if ek_cx is not None:
        ek = ek_cx
    log.timer_debug1('two-electron second derivatives (unique quartets)', *time0)

    eh = _hcore_e1(hessobj, dm0)
    hcore_deriv = hessobj.hcore_generator(mol) if eh is None else None
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    e1 = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        e1[i0, i0] -= numpy.einsum('xypq,pq->xy', s1aa[:, :, p0:p1], dme0[p0:p1]) * 2
        for j0, ja in enumerate(atmlst[:i0 + 1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0, j0] -= numpy.einsum('xypq,pq->xy', s1ab[:, :, p0:p1, q0:q1], dme0[p0:p1, q0:q1]) * 2
            if eh is None:
                e1[i0, j0] += numpy.einsum('xypq,pq->xy', hcore_deriv(ia, ja), dm0)
            else:
                e1[i0, j0] += eh[ia, ja]
        for j0 in range(i0):
            e1[j0, i0] = e1[i0, j0].T
    log.timer('RHF partial hessian (unique quartets)', *time0)
    return e1, ej, ek


COMBINE_JK = True           # the explicit 2e part as one group J - hyb K (else J and K separately)
HESS2E_CART_LMAX = 4        # lib/vhf/hess2e_cart.c LMAX, MAXPRIM
HESS2E_CART_MAXPRIM = 64


def hess2e_jk_cart(mol, dm, groups, mode=0, tol=None, prim_tol=None):
    '''Second derivatives of E_g = jfac E_J + kfac E_K at the symmetric
    density dm (E_J = 1/2 (ab|cd) D_ab D_cd, E_K = 1/4 (ab|cd) D_ad D_bc), one
    Rys quadrature per primitive quartet for all derivative components,
    contracted with the density in the Cartesian basis (lib/vhf/hess2e_cart.c).
    groups: [(jfac, kfac), ...].  Screening: shell quartets below tol
    (default ghf.HESS2E_TOL), primitive quartets (prefactor x largest
    primitive density weight) below prim_tol (default tol * 1e-2, i.e. 1e-15:
    ~5e-11 in the Hessian of (H2O)_4; prim_tol=0 keeps every primitive).
    Returns (ngroup, natm, natm, 3, 3), or the energies (ngroup,) for mode=1;
    None if the library is unavailable.'''
    return _deriv2e_cart(mol, dm, groups, 2, mode, tol, prim_tol)


def grad2e_jk_cart(mol, dm, groups, mode=0, tol=None, prim_tol=None):
    '''First derivatives of E_g = jfac E_J + kfac E_K (conventions of
    hess2e_jk_cart, e.g. [(1, -1)] for the RHF 2e energy), the same kernel at
    derivative order 1.  Returns (ngroup, natm, 3), or the energies (ngroup,)
    for mode=1; None if the library is unavailable.'''
    return _deriv2e_cart(mol, dm, groups, 1, mode, tol, prim_tol)


MD_J_LMAX = 4               # lib/vhf/md_j.c LMAX


def md_j_grad(mol, dm, tol=None):
    '''Gradient of the exact Coulomb energy E_J = 1/2 (ab|cd) D_ab D_cd from the
    McMurchie-Davidson J engine (lib/vhf/md_j.c): the basis decontracted to
    Cartesian primitives, the density in Hermite form once per primitive pair,
    no Cartesian integrals.  Returns (grad (natm, 3), E_J), or None if the
    library is unavailable or the basis has l > MD_J_LMAX.'''
    import ctypes
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFmd_j_grad') or mol._bas[:, gto.ANG_OF].max() > MD_J_LMAX:
        return None
    if tol is None:
        tol = ghf_hess.HESS2E_TOL
    pmol, c = mol.decontract_basis(to_cart=True, aggregate=True)
    dmp = numpy.ascontiguousarray(c.dot(numpy.asarray(dm, dtype=numpy.double)).dot(c.T))
    grad = numpy.zeros((mol.natm, 3))
    e = numpy.zeros(1)
    ao_loc = pmol.ao_loc_nr().astype(numpy.int32)
    libv.SOCVHFmd_j_grad(
        grad.ctypes.data_as(ctypes.c_void_p), e.ctypes.data_as(ctypes.c_void_p),
        dmp.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol),
        ao_loc.ctypes.data_as(ctypes.c_void_p), pmol._atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(pmol.natm), pmol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(pmol.nbas),
        pmol._env.ctypes.data_as(ctypes.c_void_p))
    return grad, e[0]


SPLIT_GENERAL = True
GRAD2E_HGP = True           # order 1: the HGP kernel (lib/vhf/hgp.c) for the low-l quartets
HGP_LMAX = 1                # quartets with l_i + l_j + l_k + l_l <= HGP_LMAX to HGP, the others to Rys
GRAD2E_GEN = True           # order 1: the generated kernels (lib/vhf/jk_gen.c SOCVHFgen_grad2e), l <= 2
GEN_GRAD_LMAX = 5           # quartets with total l <= GEN_GRAD_LMAX to them (<= gen_grad_vrr.py LSUM_MAX), the others to Rys;
                            # at total l 6 the Rys kernel is the faster (cc-pVDZ, 4 threads: pterin 2.44 -> 2.37 s, C16H34 9.52 -> 9.31 s)
HESS2E_GEN = True           # order 2: the generated kernels (jk_gen.c SOCVHFgen_hess2e), l <= 2, total l <=
                            # gen_hess_vrr.py LSUM_MAX; the others to Rys


def _split_general(mol):
    '''mol with every generally contracted shell (nctr > 1) replaced by its
    primitive shells, in place (the AO blocks keep their order), and C
    (nao_new, nao) with phi_old = sum_new C phi_new.  Segmented shells are kept.'''
    import scipy.linalg
    from pyscf.gto.mole import gto_norm
    bas = []
    env = [mol._env]
    blocks = []
    ptr = mol._env.size
    for ib in range(mol.nbas):
        b = mol._bas[ib]
        l = b[gto.ANG_OF]
        nf = (l + 1) * (l + 2) // 2 if mol.cart else 2 * l + 1
        if b[gto.NCTR_OF] == 1:
            bas.append(b)
            blocks.append(numpy.eye(nf))
            continue
        es = mol.bas_exp(ib)
        cs = mol._libcint_ctr_coeff(ib)          # (nprim, nctr), radial normalisation included
        norm = gto_norm(l, es)
        nprim = es.size
        for p in range(nprim):
            r = b.copy()
            r[gto.NPRIM_OF] = r[gto.NCTR_OF] = 1
            r[gto.PTR_EXP] = ptr + p
            r[gto.PTR_COEFF] = ptr + nprim + p
            bas.append(r)
        env.append(es)
        env.append(norm)
        ptr += 2 * nprim
        blocks.append(numpy.einsum('pi,p,xm->pxim', cs, 1. / norm, numpy.eye(nf)).reshape(nprim * nf, -1))
    pmol = mol.copy(deep=False)
    pmol._bas = numpy.asarray(bas, dtype=numpy.int32)
    pmol._env = numpy.hstack(env)
    return pmol, scipy.linalg.block_diag(*blocks)


def _deriv2e_cart(mol, dm, groups, order, mode, tol, prim_tol):
    import ctypes
    libv = ghf_hess._load_socvhf()
    fname = 'SOCVHFhess2e_jk_cart' if order == 2 else 'SOCVHFgrad2e_jk_cart'
    if not libv or not hasattr(libv, fname):
        return None
    # the generated kernels take the quartets of l <= 2, <= 4 contractions, total l <= GEN_GRAD_LMAX;
    # the Rys kernel the others (with SOCVHFcart_set_skip_gen, also when the basis has f shells)
    skip_gen = hasattr(libv, 'SOCVHFcart_set_skip_gen')
    gen = (order == 1 and mode == 0 and GRAD2E_GEN and hasattr(libv, 'SOCVHFgen_grad2e')
           and (skip_gen or (mol._bas[:, gto.ANG_OF].max() <= 2 and mol._bas[:, gto.NCTR_OF].max() <= 4)))
    hgp = (not gen and order == 1 and GRAD2E_HGP and hasattr(libv, 'SOCVHFhgp_grad2e')
           and mol._bas[:, gto.NCTR_OF].max() <= 4)       # hgp.c MAXCTR
    if (mol._bas[:, gto.ANG_OF].max() > HESS2E_CART_LMAX
            or mol._bas[:, gto.NPRIM_OF].max() > HESS2E_CART_MAXPRIM):
        return None
    if tol is None:
        tol = ghf_hess.HESS2E_TOL
    if prim_tol is None:
        prim_tol = tol * 1e-2
    ngroup = len(groups)
    assert 0 < ngroup <= 4
    if mol.cart:
        molc, dmc = mol, numpy.asarray(dm, dtype=numpy.double)
    else:
        molc = mol.copy(deep=False)
        molc.cart = True
        c = mol.cart2sph_coeff()
        dmc = c.dot(dm).dot(c.T)
    if not hgp and not gen and SPLIT_GENERAL and order == 1 and (molc._bas[:, gto.NCTR_OF] > 1).any():
        # generally contracted shells as primitive shells: every quartet on the
        # single-contraction path, no primitive is evaluated twice
        molc, cg = _split_general(molc)
        dmc = cg.dot(dmc).dot(cg.T)
    dmc = numpy.ascontiguousarray(dmc)
    # integral estimate per shell pair, bounding every derivative type
    intors = ('int2e_ipip1', 'int2e_ipvip1', 'int2e_ip1ip2') if order == 2 else ('int2e_ip1',)
    q = numpy.max([ghf_hess._q_cond(molc, intor)[k] for intor in intors for k in (0, 1)], axis=0)
    q = numpy.ascontiguousarray(numpy.maximum(q, q.T))
    dcond = ghf_hess._dm_cond(molc, [dmc])
    jfac = numpy.asarray([g[0] for g in groups], dtype=numpy.double)
    kfac = numpy.asarray([g[1] for g in groups], dtype=numpy.double)
    natm = mol.natm
    if mode == 1:
        out = numpy.zeros(ngroup)
    elif order == 2:
        out = numpy.zeros((ngroup, natm, natm, 3, 3))
    else:
        out = numpy.zeros((ngroup, natm, 3))
    ao_loc = molc.ao_loc_nr().astype(numpy.int32)
    atm, bas, env = molc._atm, molc._bas, molc._env
    args = (ctypes.c_int(mode), ctypes.c_int(ngroup),
            jfac.ctypes.data_as(ctypes.c_void_p), kfac.ctypes.data_as(ctypes.c_void_p),
            dmc.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
            dcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol), ctypes.c_double(prim_tol),
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(molc.nbas),
            env.ctypes.data_as(ctypes.c_void_p))
    if (order == 2 and mode == 0 and HESS2E_GEN and skip_gen and hasattr(libv, 'SOCVHFgen_hess2e')):
        # the generated second-derivative kernels for l <= 2, the rest (larger l or total l) by Rys
        lsum = libv.SOCVHFgen_hess_lsum_max()
        out2 = numpy.zeros_like(out)
        try:
            libv.SOCVHFgen_hess2e(out.ctypes.data_as(ctypes.c_void_p), *args)
            libv.SOCVHFcart_set_skip_gen(ctypes.c_int(lsum))
            getattr(libv, fname)(out2.ctypes.data_as(ctypes.c_void_p), *args)
        finally:
            libv.SOCVHFcart_set_skip_gen(ctypes.c_int(-1))
        return out + out2
    if gen:
        # the generated kernels (native general contraction), the rest by Rys
        out2 = numpy.zeros_like(out)
        try:
            libv.SOCVHFgen_grad_set_lrange(ctypes.c_int(0), ctypes.c_int(GEN_GRAD_LMAX))
            libv.SOCVHFgen_grad2e(out.ctypes.data_as(ctypes.c_void_p), *args)
            if skip_gen:
                libv.SOCVHFcart_set_skip_gen(ctypes.c_int(GEN_GRAD_LMAX))
                getattr(libv, fname)(out2.ctypes.data_as(ctypes.c_void_p), *args)
            elif GEN_GRAD_LMAX < 8:
                libv.SOCVHFcart_set_lrange(ctypes.c_int(GEN_GRAD_LMAX + 1), ctypes.c_int(1000))
                getattr(libv, fname)(out2.ctypes.data_as(ctypes.c_void_p), *args)
        finally:
            libv.SOCVHFgen_grad_set_lrange(ctypes.c_int(0), ctypes.c_int(1000))
            libv.SOCVHFcart_set_lrange(ctypes.c_int(0), ctypes.c_int(1000))
            if skip_gen:
                libv.SOCVHFcart_set_skip_gen(ctypes.c_int(-1))
        return out + out2
    if not hgp:
        getattr(libv, fname)(out.ctypes.data_as(ctypes.c_void_p), *args)
        return out
    # low total l by HGP (contraction before the horizontal recursion), the rest by Rys
    out2 = numpy.zeros_like(out)
    try:
        libv.SOCVHFhgp_set_lrange(ctypes.c_int(0), ctypes.c_int(HGP_LMAX))
        libv.SOCVHFcart_set_lrange(ctypes.c_int(HGP_LMAX + 1), ctypes.c_int(1000))
        libv.SOCVHFhgp_grad2e(out.ctypes.data_as(ctypes.c_void_p), *args)
        getattr(libv, fname)(out2.ctypes.data_as(ctypes.c_void_p), *args)
    finally:
        libv.SOCVHFhgp_set_lrange(ctypes.c_int(0), ctypes.c_int(1000))
        libv.SOCVHFcart_set_lrange(ctypes.c_int(0), ctypes.c_int(1000))
    return out + out2


def _hcore_e1(hessobj, dm0):
    """sum_pq d^2 h_pq / dR_A dR_B (dm0)_pq for all atom pairs, (natm, natm, 3, 3),
    the hcore part of pyscf's _partial_hess_ejk: the second-derivative rinv
    integrals once per nucleus (2 natm intor calls, not ~4 per atom pair as
    hcore_generator) and contracted with the symmetric dm0 row by row.  None
    (use pyscf's generator) with ECP or X2C."""
    mol = hessobj.mol
    if mol.has_ecp() or getattr(hessobj.base, 'with_x2c', None) or mol._pseudo:
        return None
    natm = mol.natm
    nao = mol.nao_nr()
    aoslices = mol.aoslice_by_atom()
    h1aa, h1ab = hessobj.get_hcore(mol)                  # (3, 3, nao, nao)
    D = numpy.asarray(dm0)
    S = numpy.zeros((nao, natm))
    for ia in range(natm):
        S[aoslices[ia][2]:aoslices[ia][3], ia] = 1
    e1 = numpy.zeros((natm, natm, 3, 3))
    # kinetic + nuclear-attraction second derivatives on the basis centres:
    # h1aa (same centre) on the diagonal, h1ab (bra on A, ket on B) per block
    raa = numpy.einsum('xypq,pq->xyp', h1aa, D)
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        e1[ia, ia] += raa[:, :, p0:p1].sum(axis=2)
    hab = h1ab * D                                       # (3, 3, nao, nao)
    # S^T hab[x, y] S as GEMMs (numpy's einsum without optimize loops over all of x y p q A B)
    for x in range(3):
        for y in range(3):
            e1[:, :, x, y] += lib.dot(S.T, lib.dot(hab[x, y], S))
    # operator (nucleus C) derivatives
    for ic in range(natm):
        zc = mol.atom_charge(ic)
        c0, c1 = aoslices[ic][2:]
        with mol.with_rinv_at_nucleus(ic):
            aa = mol.intor('int1e_ipiprinv', comp=9).reshape(3, 3, nao, nao) * zc
            ab = mol.intor('int1e_iprinvip', comp=9).reshape(3, 3, nao, nao) * zc
        xa = numpy.einsum('xypq,pq->xyp', aa, D)         # row sums
        xb = numpy.einsum('xypq,pq->xyp', ab, D)
        yb = numpy.einsum('xypq,pq->xyq', ab, D)         # column sums
        per_atom_a = xa.dot(S)                           # (3, 3, natm): rows of atom B
        per_atom_b = xb.dot(S)
        per_atom_bT = numpy.einsum('xypq,pq->yxp', ab, D).dot(S)
        # i == j == C: -aa - ab overall, + the C-row / C-column blocks
        e1[ic, ic] += -xa.sum(axis=2) - xb.sum(axis=2)
        e1[ic, ic] += xa[:, :, c0:c1].sum(axis=2) + xb[:, :, c0:c1].sum(axis=2)
        e1[ic, ic] += xa[:, :, c0:c1].sum(axis=2) + yb[:, :, c0:c1].sum(axis=2)
        # i = C, j = B != C: rows of B, aa + ab^T(x<->y);  i = B, j = C: rows of B, aa + ab
        for ib in range(natm):
            if ib == ic:
                continue
            e1[ic, ib] += per_atom_a[:, :, ib] + per_atom_bT[:, :, ib]
            e1[ib, ic] += per_atom_a[:, :, ib] + per_atom_b[:, :, ib]
    # hcore + hcore^T contracted with the symmetric dm0
    return e1 * 2


H1_OCC = True     # hess_elec: the RKS first-order Fock matrices as h1 C_occ (rks_make_h1 / xc_deriv1_batched)


def _is_h1_occ(h1, mo_coeff, mo_occ):
    nocc = int((mo_occ > 0).sum())
    return h1.ndim == 4 and h1.shape[-1] == nocc and h1.shape[-2] == mo_coeff.shape[0] != nocc


def _solve_mo1_occ(mf, mo_energy, mo_coeff, mo_occ, h1c, fx, atmlst=None, max_memory=4000,
                   max_cycle=50, level_shift=0):
    '''pyscf.hessian.rhf.solve_mo1 for h1 given as h1 C_occ (natm, 3, nao, nocc)'''
    from pyscf.scf import cphf
    mol = mf.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = mocc.shape[1]
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory * .9 - mem_now)
    blksize = max(2, int(max_memory * 1e6 / 8 / (nmo * nocc * 3 * 6)))
    mo1s = [None] * mol.natm
    e1s = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1vo = []
        h1vo = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            p0, p1 = aoslices[ia][2:]
            # s1 C_occ: rows of A (s1a C) plus (s1a^T restricted to A's columns)
            s1c = numpy.zeros((3, nao, nocc))
            s1c[:, p0:p1] += lib.dot(s1a[:, p0:p1].reshape(-1, nao), mocc).reshape(3, p1 - p0, nocc)
            for x in range(3):
                s1c[x] += lib.dot(s1a[x, p0:p1].T, mocc[p0:p1])
            s1vo.append([lib.dot(mo_coeff.T, s1c[x]) for x in range(3)])
            h1vo.append([lib.dot(mo_coeff.T, h1c[ia, x]) for x in range(3)])
        h1vo = numpy.vstack(h1vo)
        s1vo = numpy.vstack(s1vo)
        tol = mf.conv_tol_cpscf * (ia1 - ia0)
        mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo, max_cycle=max_cycle, level_shift=level_shift,
                             tol=tol)
        mo1 = numpy.einsum('pq,xqi->xpi', mo_coeff, mo1).reshape(-1, 3, nao, nocc)
        e1 = e1.reshape(-1, 3, nocc, nocc)
        for k in range(ia1 - ia0):
            ia = atmlst[k + ia0]
            mo1s[ia] = mo1[k]
            e1s[ia] = e1[k].reshape(3, nocc, nocc)
        mo1 = e1 = None
    return mo1s, e1s


def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None, mo1=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    """pyscf.hessian.rhf.hess_elec with the response terms of all atom pairs as
    three GEMMs over (atom, x) (pyscf: einsums per atom pair, rebuilding the
    first-order density of atom B for every A).  Same lower-triangle-and-mirror
    convention as pyscf."""
    log = logger.new_logger(hessobj, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if atmlst is not None and list(atmlst) != list(range(mol.natm)):
        return rhf_hess.hess_elec(hessobj, mo_energy, mo_coeff, mo_occ, mo1, mo_e1, h1ao, atmlst,
                                  max_memory, verbose)
    natm = mol.natm
    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, None, max_memory, log)
    if h1ao is None:
        # inside the Hessian only h1 C_occ is needed (the CP-SCF right-hand sides, h1 . dm1)
        with lib.temporary_env(hessobj, _h1_occ=H1_OCC):
            h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, None, log)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, None, None, max_memory, log)
    nao = mo_coeff.shape[0]
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = mocc.shape[1]
    e_o = mo_energy[mo_occ > 0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    # everything in the occupied form (natm 3, nao, nocc): s1 C_occ from s1a without the
    # (natm, 3, nao, nao) s1ao, and no dm1 / dme1 (h1 . dm1 = sum mo1 (h1 C), s1 . dme1 = sum mo1 (s1 C e))
    S1C = lib.dot(s1a.reshape(3 * nao, nao), mocc).reshape(3, nao, nocc)
    s1c = numpy.zeros((natm, 3, nao, nocc))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        s1c[ia, :, p0:p1] += S1C[:, p0:p1]
        for x in range(3):
            s1c[ia, x] += lib.dot(s1a[x, p0:p1].T, mocc[p0:p1])
    S1C = None
    mo1 = numpy.asarray(mo1).reshape(natm * 3, nao * nocc)
    h1 = numpy.asarray(h1ao)
    if not _is_h1_occ(h1, mo_coeff, mo_occ):
        h1 = lib.dot(h1.reshape(-1, nao), mocc)               # AO h1 (symmetric) -> h1 C_occ
    h1 = h1.reshape(natm * 3, nao * nocc)
    s1oo = lib.dot(mocc.T, s1c.reshape(-1, nao, nocc).transpose(1, 0, 2).reshape(nao, -1))
    s1oo = s1oo.reshape(nocc, natm * 3, nocc).transpose(1, 0, 2).reshape(natm * 3, nocc * nocc)
    s1c *= e_o
    me1 = numpy.asarray(mo_e1).reshape(natm * 3, nocc * nocc)
    # *2 for double occupancy, *2 for +c.c.
    r = (4 * lib.dot(h1, mo1.T) - 4 * lib.dot(s1c.reshape(natm * 3, nao * nocc), mo1.T)
         - 2 * lib.dot(s1oo, me1.T))
    de2 += r.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    # pyscf keeps the lower triangle (i >= j) and mirrors it
    for i0 in range(natm):
        for j0 in range(i0):
            de2[j0, i0] = de2[i0, j0].T
    log.timer('RHF hessian (fast assembly)', *time0)
    return de2


def rhf_partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                          atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                   max_memory, verbose, True, hyb=1.)
    return e1 + ej - ek


_libsocdft = False


def _load_socdft():
    '''lib/libsocdft.so (with SOCDFTrks_xc_hess_elem), or None'''
    global _libsocdft
    if _libsocdft is False:
        try:
            from socutils.lib import load_library
            lib_ = load_library('libsocdft', 'SOCUTILS_SOCDFT_LIBRARY')
            _libsocdft = lib_ if hasattr(lib_, 'SOCDFTrks_xc_hess_elem') else None
        except (OSError, ImportError):
            _libsocdft = None
    return _libsocdft


def _ao_stack(ao):
    """pyscf's AO derivative stack (ncomp, g, nao) as (ncomp, nao, g) for the C
    kernels, and the number of doubles between components: no copy when every
    component is contiguous (pyscf's last, shorter grid block keeps the
    full-block stride between components)."""
    aoT = (ao if ao.ndim == 3 else ao[None]).transpose(0, 2, 1)
    if aoT.strides[2] != 8 or aoT.strides[1] != 8 * aoT.shape[2] or aoT.strides[0] % 8:
        aoT = numpy.ascontiguousarray(aoT)
    return aoT, aoT.strides[0] // 8


# indices of the 2nd / 3rd Cartesian derivatives in pyscf's AO derivative stack
_D2 = numpy.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]])
_D3 = numpy.empty((3, 3, 3), dtype=int)
for _x in range(3):
    for _y in range(3):
        for _z in range(3):
            _D3[_x, _y, _z] = {(0, 0, 0): 10, (0, 0, 1): 11, (0, 0, 2): 12, (0, 1, 1): 13, (0, 1, 2): 14,
                               (0, 2, 2): 15, (1, 1, 1): 16, (1, 1, 2): 17, (1, 2, 2): 18,
                               (2, 2, 2): 19}[tuple(sorted((_x, _y, _z)))]


XCHE_LOCAL = True       # xc_hess_explicit: per sub-block on its significant AOs (else the whole pyscf block, all AOs)
XCHE_LOCAL_MIN_NAO = 200   # below this every AO is significant on most sub-blocks: the whole-block path


def _xc_hess_explicit_local(ni, xc_code, xctype, gga, nvar, ao, weight, p0, dm0, atom_of, natm, non0tab,
                            nf_shell, nsub, libdft, term1, diag, M9):
    '''one pyscf grid block of xc_hess_explicit (LDA / GGA) sub-block by sub-block, each on the AOs the
    grid's non0tab flags on it: D phi_k, rho and the functional, R / the on-atom sums / the M operands
    (SOCDFTrks_xc_hess_elem), the R f R GEMM and the atom-pair GEMM over the local AOs only'''
    from pyscf.dft.numint import BLKSIZE
    ngrid = weight.size
    aoT, cs = _ao_stack(ao)
    ncomp = aoT.shape[0]
    nao = aoT.shape[1]
    nd_ = 4 if gga else 1
    K = 2 if gga else 1
    for s0 in range(0, ngrid, nsub):
        s1 = min(ngrid, s0 + nsub)
        gs = s1 - s0
        aom = numpy.repeat(non0tab[(p0 + s0) // BLKSIZE:(p0 + s1 + BLKSIZE - 1) // BLKSIZE].any(axis=0), nf_shell)
        idx = numpy.nonzero(aom)[0].astype(numpy.int32)
        nl = idx.size
        if nl == 0:
            continue
        aol = numpy.empty((ncomp, nl, gs))
        libdft.SOCDFTgather_ao(aol.ctypes, aoT.ctypes, idx.ctypes, ctypes.c_int(ncomp), ctypes.c_int(nao),
                               ctypes.c_int(nl), ctypes.c_int(ngrid), ctypes.c_int(s0), ctypes.c_int(gs),
                               ctypes.c_long(cs))
        dl = numpy.ascontiguousarray(dm0[idx[:, None], idx])
        dT = numpy.empty((nd_, nl, gs))
        for k in range(nd_):
            lib.dot(dl, aol[k], c=dT[k])
        rho = numpy.empty((nvar, gs))
        rho[0] = numpy.einsum('ig,ig->g', aol[0], dT[0])
        if gga:
            for j in range(3):
                rho[1 + j] = 2 * numpy.einsum('ig,ig->g', aol[1 + j], dT[0])
        vxc, fxc = ni.eval_xc_eff(xc_code, rho if gga else rho[0], 2, xctype=xctype)[1:3]
        w = weight[s0:s1]
        wf = numpy.ascontiguousarray(w * fxc.reshape(nvar, nvar, gs))
        wv = numpy.ascontiguousarray(w * vxc.reshape(nvar, gs))
        # d_A rho, the on-atom sums and the R f R block only for the sub-block's atoms (local numbering):
        # R of the other atoms vanishes there, and the full-natm GEMM costs (3 natm)^2 per point
        atoms, atl = numpy.unique(atom_of[idx], return_inverse=True)
        atl = numpy.ascontiguousarray(atl, dtype=numpy.int32)
        na = atoms.size
        R = numpy.empty((3, nvar, na, gs))
        ab = numpy.empty((6 if gga else 3, nl, gs))
        dg = numpy.zeros((3, 3, na))
        libdft.SOCDFTrks_xc_hess_elem(
            R.ctypes, dg.ctypes, ab.ctypes, aol.ctypes, dT.ctypes, wv.ctypes, ctypes.c_int(int(gga)),
            ctypes.c_int(nvar), ctypes.c_int(gs), ctypes.c_int(nl), atl.ctypes, ctypes.c_int(na),
            ctypes.c_long(nl * gs))
        R *= -2
        diag[:, :, atoms] += dg
        wR = numpy.empty((3 * na, nvar, gs))
        libdft.SOCDFTfxc_contract_R(wR.ctypes, wf.ctypes, R.ctypes, ctypes.c_int(na), ctypes.c_int(nvar),
                                    ctypes.c_int(gs), ctypes.c_double(1.), ctypes.c_int(0))
        Rr = numpy.empty((3 * na, nvar, gs))
        libdft.SOCDFTreorder_R(Rr.ctypes, R.ctypes, ctypes.c_int(na), ctypes.c_int(nvar), ctypes.c_int(gs))
        t1 = lib.dot(Rr.reshape(3 * na, -1), wR.reshape(3 * na, -1).T)
        term1[numpy.ix_(range(3), atoms, range(3), atoms)] += t1.reshape(na, 3, na, 3).transpose(1, 0, 3, 2)
        # atom-pair part over the local AOs: [a_x | dphi_x] . [dphi_y | b_y]^T
        L = numpy.empty((3, nl, K * gs))
        Rm = numpy.empty((3, nl, K * gs))
        ca = nl * gs
        if gga:
            libdft.SOCDFTgather_rows2(L.ctypes, ab.ctypes, ctypes.c_long(ca), aol[1:].ctypes, ctypes.c_long(ca),
                                      None, ctypes.c_int(nl), ctypes.c_int(nl), ctypes.c_int(gs),
                                      ctypes.c_int(0), ctypes.c_int(gs))
            libdft.SOCDFTgather_rows2(Rm.ctypes, aol[1:].ctypes, ctypes.c_long(ca), ab[3:].ctypes,
                                      ctypes.c_long(ca), None, ctypes.c_int(nl), ctypes.c_int(nl),
                                      ctypes.c_int(gs), ctypes.c_int(0), ctypes.c_int(gs))
        else:
            L[:] = ab
            Rm[:] = aol[1:4]
        mb = lib.dot(L.reshape(3 * nl, -1), Rm.reshape(3 * nl, -1).T)
        mb = numpy.ascontiguousarray(mb.reshape(3, nl, 3, nl).transpose(0, 2, 1, 3))
        libdft.SOCDFTscatter_add_dm(M9.ctypes, mb.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(9),
                                    ctypes.c_int(nao))


def xc_hess_explicit(hessobj, mo_coeff, mo_occ, max_memory=4000):
    '''Fixed-grid XC part of the explicit RKS Hessian (what pyscf builds from
    _get_vxc_diag and _get_vxc_deriv2), contracted with D on the grid:

        d2E/dA dB = sum_g [ sum_kl f_kl R_Ak R_Bl + sum_k v_k d_A d_B rho_k ],

    rho_k = (rho, nabla rho, tau), R_A = d_A rho_k at fixed D.  The f_kl term
    needs only the per-atom grid vectors R_A, the on-atom part of d_A d_B rho
    is elementwise, and the atom-pair part one set of nao x nao matrices M[x, y]
    (instead of pyscf's per-atom 3 x 3 nao x nao potentials).  (natm, natm, 3, 3).'''
    mol = hessobj.mol
    mf = hessobj.base
    grids = hessobj.grids if hessobj.grids is not None else mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    nao = mo_coeff.shape[0]
    natm = mol.natm
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    aoslices = mol.aoslice_by_atom()
    S = numpy.zeros((nao, natm))            # AO -> atom indicator
    for ia in range(natm):
        S[aoslices[ia][2]:aoslices[ia][3], ia] = 1
    from pyscf.dft import numint
    gga = xctype in ('GGA', 'MGGA')
    mgga = xctype == 'MGGA'
    nvar = 5 if mgga else (4 if gga else 1)
    ao_deriv = 3 if gga else 2
    term1 = numpy.zeros((3, natm, 3, natm))
    diag = numpy.zeros((3, 3, natm))
    M = numpy.zeros((3, 3, nao, nao))
    libdft = _load_socdft()
    atom_of = numpy.zeros(nao, dtype=numpy.int32)
    for ia in range(natm):
        atom_of[aoslices[ia][2]:aoslices[ia][3]] = ia
    from pyscf.dft.numint import BLKSIZE
    nf_shell = ao_loc[1:] - ao_loc[:-1]
    non0tab = getattr(grids, 'non0tab', None)
    nsub = max(1, FXC_SUBBLOCK // BLKSIZE) * BLKSIZE
    M9 = M.reshape(9, nao, nao)
    p1 = 0
    local = (XCHE_LOCAL and libdft is not None and not mgga and non0tab is not None
             and nao >= XCHE_LOCAL_MIN_NAO)
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ngrid = weight.size
        p0, p1 = p1, p1 + ngrid
        if local:
            _xc_hess_explicit_local(ni, mf.xc, xctype, gga, nvar, ao, weight, p0, dm0, atom_of, natm, non0tab,
                                    nf_shell, nsub, libdft, term1, diag, M9)
            continue
        nd_ = 4 if gga else 1
        if libdft is not None:
            aoT, cs = _ao_stack(ao)
            dT = numpy.empty((nd_, nao, ngrid))                  # D phi_k^T (dm0 symmetric)
            for k in range(nd_):
                lib.dot(dm0, aoT[k], c=dT[k])
            d = [dT[k].T for k in range(nd_)]
        else:
            d = [numint._dot_ao_dm(mol, ao[k], dm0, mask, shls_slice, ao_loc) for k in range(nd_)]
        rho = numpy.empty((nvar, ngrid))
        rho[0] = numint._contract_rho(ao[0], d[0])
        if gga:
            for j in range(3):
                rho[1 + j] = 2 * numint._contract_rho(ao[1 + j], d[0])
        if mgga:
            rho[4] = .5 * sum(numint._contract_rho(ao[1 + j], d[1 + j]) for j in range(3))
        vxc, fxc = ni.eval_xc_eff(mf.xc, rho if gga else rho[0], 2, xctype=xctype)[1:3]
        wf = numpy.ascontiguousarray(weight * fxc.reshape(nvar, nvar, ngrid))
        if libdft is not None:
            # R, the on-atom sums and the M operands in one OpenMP pass (lib/dft/xc_hess.c)
            wv = numpy.ascontiguousarray(weight * vxc.reshape(nvar, ngrid))
            R = numpy.empty((3, nvar, natm, ngrid))
            ab = numpy.empty((6 if gga else 3, nao, ngrid))
            dg = numpy.zeros((3, 3, natm))
            libdft.SOCDFTrks_xc_hess_elem(
                R.ctypes, dg.ctypes, ab.ctypes, aoT.ctypes, dT.ctypes, wv.ctypes,
                ctypes.c_int(2 if mgga else int(gga)), ctypes.c_int(nvar), ctypes.c_int(ngrid),
                ctypes.c_int(nao), atom_of.ctypes, ctypes.c_int(natm), ctypes.c_long(cs))
            R *= -2
            diag += dg
            # term1 = sum_g R f R as one GEMM: rows (atom, x), columns (component, grid)
            wR = numpy.empty((3 * natm, nvar, ngrid))
            libdft.SOCDFTfxc_contract_R(wR.ctypes, wf.ctypes, R.ctypes, ctypes.c_int(natm), ctypes.c_int(nvar),
                                        ctypes.c_int(ngrid), ctypes.c_double(1.), ctypes.c_int(0))
            Rr = numpy.empty((3 * natm, nvar, ngrid))
            libdft.SOCDFTreorder_R(Rr.ctypes, R.ctypes, ctypes.c_int(natm), ctypes.c_int(nvar), ctypes.c_int(ngrid))
            t1 = lib.dot(Rr.reshape(3 * natm, -1), wR.reshape(3 * natm, -1).T)
            term1 += t1.reshape(natm, 3, natm, 3).transpose(1, 0, 3, 2)
            # atom-pair part M[x, y] = sum_g a_x dphi_y + dphi_x b_y: per sub-block one
            # GEMM over its significant AOs, [a_x | dphi_x] . [dphi_y | b_y]^T
            ca = nao * ngrid
            K = 2 if gga else 1
            for s0 in range(0, ngrid, nsub):
                s1 = min(ngrid, s0 + nsub)
                gs = s1 - s0
                idx = None
                nl = nao
                if non0tab is not None:
                    aom = numpy.repeat(non0tab[(p0 + s0) // BLKSIZE:(p0 + s1 + BLKSIZE - 1) // BLKSIZE].any(axis=0),
                                       nf_shell)
                    nl = int(aom.sum())
                    if nl == 0:
                        continue
                    if nl < nao:
                        idx = numpy.nonzero(aom)[0].astype(numpy.int32)
                    else:
                        nl = nao
                pidx = None if idx is None else idx.ctypes
                L = numpy.empty((3, nl, K * gs))
                Rm = numpy.empty((3, nl, K * gs))
                if gga:
                    libdft.SOCDFTgather_rows2(L.ctypes, ab.ctypes, ctypes.c_long(ca), aoT[1:].ctypes, ctypes.c_long(cs),
                                              pidx, ctypes.c_int(nao), ctypes.c_int(nl), ctypes.c_int(ngrid),
                                              ctypes.c_int(s0), ctypes.c_int(gs))
                    libdft.SOCDFTgather_rows2(Rm.ctypes, aoT[1:].ctypes, ctypes.c_long(cs), ab[3:].ctypes,
                                              ctypes.c_long(ca), pidx, ctypes.c_int(nao), ctypes.c_int(nl),
                                              ctypes.c_int(ngrid), ctypes.c_int(s0), ctypes.c_int(gs))
                else:
                    libdft.SOCDFTgather_ao(L.ctypes, ab.ctypes, pidx, ctypes.c_int(3), ctypes.c_int(nao),
                                           ctypes.c_int(nl), ctypes.c_int(ngrid), ctypes.c_int(s0), ctypes.c_int(gs),
                                           ctypes.c_long(ca))
                    libdft.SOCDFTgather_ao(Rm.ctypes, aoT[1:].ctypes, pidx, ctypes.c_int(3), ctypes.c_int(nao),
                                           ctypes.c_int(nl), ctypes.c_int(ngrid), ctypes.c_int(s0), ctypes.c_int(gs),
                                           ctypes.c_long(cs))
                mb = lib.dot(L.reshape(3 * nl, -1), Rm.reshape(3 * nl, -1).T)
                mb = numpy.ascontiguousarray(mb.reshape(3, nl, 3, nl).transpose(0, 2, 1, 3))
                if idx is None:
                    M9 += mb.reshape(9, nao, nao)
                else:
                    libdft.SOCDFTscatter_add_dm(M9.ctypes, mb.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(9),
                                                ctypes.c_int(nao))
            if mgga:
                wv = wv[:, :, None]
                for x in range(3):
                    for j in range(3):
                        c = numint._scale_ao(ao[_D2[x, j]], wv[4, :, 0])
                        for y in range(3):
                            M[x, y] += numint._dot_ao_ao(mol, c, ao[_D2[y, j]], mask, shls_slice, ao_loc)
        else:
            wv = (weight * vxc.reshape(nvar, ngrid))[:, :, None]
            # R[x, k, g, A] = d/dR_{A,x} rho_k
            R = numpy.empty((3, nvar, ngrid, natm))
            for x in range(3):
                R[x, 0] = (ao[1 + x] * d[0]).dot(S)
                if gga:
                    for j in range(3):
                        R[x, 1 + j] = (ao[_D2[x, j]] * d[0] + ao[1 + x] * d[1 + j]).dot(S)
                if mgga:
                    R[x, 4] = (.5 * sum(ao[_D2[x, j]] * d[1 + j] for j in range(3))).dot(S)
            R *= -2
            wR = numpy.einsum('klg,ylgB->ykgB', wf, R)
            term1 += numpy.einsum('xkgA,ykgB->xAyB', R, wR)
            # on-atom part of d_A d_A rho_k
            e0 = 2 * wv[0] * d[0]
            if gga:
                e0 += 2 * sum(wv[1 + j] * d[1 + j] for j in range(3))
                f = [2 * wv[1 + j] * d[0] + (wv[4] * d[1 + j] if mgga else 0) for j in range(3)]
            for x in range(3):
                for y in range(x, 3):
                    t = ao[_D2[x, y]] * e0
                    if gga:
                        for j in range(3):
                            t += ao[_D3[x, y, j]] * f[j]
                    diag[x, y] += t.sum(axis=0).dot(S)
            # a_x = 2 w v0 d_x phi + b_x, b_x = 2 sum_j w v_j d_xj phi
            b = [2 * sum(wv[1 + j] * ao[_D2[x, j]] for j in range(3)) for x in range(3)] if gga else None
            a = [2 * wv[0] * ao[1 + x] + (b[x] if gga else 0) for x in range(3)]
            # atom-pair part: M[x, y]_{mu nu} = sum_g a_x,mu dphi_y,nu + dphi_x,mu b_y,nu (+ tau)
            for x in range(3):
                for y in range(3):
                    M[x, y] += numint._dot_ao_ao(mol, a[x], ao[1 + y], mask, shls_slice, ao_loc)
                    if gga:
                        M[x, y] += numint._dot_ao_ao(mol, ao[1 + x], b[y], mask, shls_slice, ao_loc)
            if mgga:
                for x in range(3):
                    for j in range(3):
                        c = numint._scale_ao(ao[_D2[x, j]], wv[4, :, 0])
                        for y in range(3):
                            M[x, y] += numint._dot_ao_ao(mol, c, ao[_D2[y, j]], mask, shls_slice, ao_loc)
    de2 = term1.transpose(1, 3, 0, 2).copy()
    for x in range(3):
        for y in range(3):
            de2[:, :, x, y] += S.T.dot(M[x, y] * dm0).dot(S)
    for ia in range(natm):
        de2[ia, ia] += diag[:, :, ia] + numpy.triu(diag[:, :, ia], 1).T
    return de2


def rks_partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                          atmlst=None, max_memory=4000, verbose=None):
    '''pyscf.hessian.rks.partial_hess_elec with the unique-quartet ej / ek and
    the D-contracted XC terms (range-separated / NLC / atom subsets: pyscf's
    XC code).'''
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if (omega != 0 or mf.do_nlc()
            or (atmlst is not None and list(atmlst) != list(range(mol.natm)))):
        if _df_mode(mf) is not None:
            from pyscf.df.hessian import rks as df_rks_hess
            return df_rks_hess.partial_hess_elec(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                                 max_memory, verbose)
        with lib.temporary_env(rhf_hess, _partial_hess_ejk=_partial_hess_ejk):
            return rks_hess.partial_hess_elec(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                              max_memory, verbose)
    log = logger.new_logger(hessobj, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    de2, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, None, max_memory,
                                    verbose, with_k=hybrid, hyb=hyb)
    de2 += ej - hyb * ek
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    t1 = log.timer_debug1('RKS partial hessian, 1e + 2e', *time0)
    de2 += xc_hess_explicit(hessobj, mo_coeff, mo_occ, max_memory)
    log.timer_debug1('RKS partial hessian, XC', *t1)
    log.timer('RKS partial hessian', *time0)
    return de2


def fock1_jk(mol, dm, with_k=True, tol=ghf_hess.HESS2E_TOL, max_memory=None, with_j=True):
    '''First-order VJ^{A,x}_ab = d/dR_{A,x} (ab|cd) D_cd and
    VK^{A,x}_ad = d/dR_{A,x} (ab|cd) D_bc of all atoms at a symmetric D, in one
    pass over the unique quartets (lib/vhf/fock1.c).  The atoms are batched so
    that nthreads x the per-thread buffers stay within half of max_memory (MB,
    default mol.max_memory); a batch skips the quartets touching none of its
    atoms.  Returns (natm, 3, nao, nao) arrays, or None if the library is
    unavailable.  with_j=False (K only, with_k): vj None, the quartets screened by K's density
    pairs alone (the generated-kernel path; the libcint one computes J too).'''
    import ctypes
    from pyscf.gto import moleintor
    from socutils.scf import jk_multi
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFfock1_jk') or mol.cart:
        return None
    if FOCK1_GEN and hasattr(libv, 'SOCVHFgen_fock1_jk'):
        return _fock1_jk_gen(libv, mol, dm, with_k, tol, max_memory, with_j=with_j)
    dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
    nao = dm.shape[0]
    vj = numpy.empty((mol.natm, 3, nao, nao))
    vk = numpy.empty((mol.natm, 3, nao, nao)) if with_k else numpy.empty(1)
    q = jk_multi._q_cond(mol)
    dcond = ghf_hess._dm_cond(mol, [dm])
    atm, bas, env = mol._atm, mol._bas, mol._env
    opt = moleintor.make_cintopt(atm, bas, env, 'int2e_ip1_sph')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    if max_memory is None:
        max_memory = mol.max_memory
    per_atom = (1 + int(with_k)) * 3 * nao * nao * 8 * lib.num_threads()
    nbatch = max(1, min(mol.natm, int(max_memory * .5e6 / per_atom)))
    for a0 in range(0, mol.natm, nbatch):
        a1 = min(mol.natm, a0 + nbatch)
        libv.SOCVHFfock1_jk(
            vj[a0:a1].ctypes.data_as(ctypes.c_void_p),
            vk[a0:a1].ctypes.data_as(ctypes.c_void_p) if with_k else vk.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(int(with_k)),
            q.ctypes.data_as(ctypes.c_void_p), dcond.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(tol), opt,
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(a0), ctypes.c_int(a1))
    return vj, (vk if with_k else None)


FOCK1_GEN = True            # fock1_jk by the generated derivative-integral kernels (Cartesian), jk_gen.c


def _fock1_jk_gen(libv, mol, dm, with_k, tol, max_memory, prim_tol=None, with_j=True):
    '''fock1_jk in the Cartesian basis by SOCVHFgen_fock1_jk (the three role
    derivatives of a quartet from one HGP pass; libcint for the classes
    without a kernel), transformed to the spherical AOs.'''
    import ctypes
    from pyscf.gto import moleintor
    from socutils.scf import jk_multi
    if prim_tol is None:
        prim_tol = tol * 1e-2
    molc = mol.copy(deep=False)
    molc.cart = True
    c = mol.cart2sph_coeff()
    dmc = numpy.ascontiguousarray(c.dot(numpy.asarray(dm, dtype=numpy.double)).dot(c.T))
    nao, naoc = mol.nao, molc.nao
    q = jk_multi._q_cond(molc)
    dcond = ghf_hess._dm_cond(molc, [dmc])
    atm, bas, env = molc._atm, molc._bas, molc._env
    opt = moleintor.make_cintopt(atm, bas, env, 'int2e_ip1_cart')
    ao_loc = molc.ao_loc_nr().astype(numpy.int32)
    if max_memory is None:
        max_memory = mol.max_memory
    k_only = with_k and not with_j and hasattr(libv, 'SOCVHFgen_fock1_jk2')
    nk = int(not k_only) + int(with_k)
    per_atom = nk * 3 * naoc * naoc * 8 * (lib.num_threads() + 1)
    nbatch = max(1, min(mol.natm, int(max_memory * .5e6 / per_atom)))
    vj = None if k_only else numpy.empty((mol.natm, 3, nao, nao))
    vk = numpy.empty((mol.natm, 3, nao, nao)) if with_k else None
    for a0 in range(0, mol.natm, nbatch):
        a1 = min(mol.natm, a0 + nbatch)
        vjc = numpy.empty(1) if k_only else numpy.empty((a1 - a0, 3, naoc, naoc))
        vkc = numpy.empty((a1 - a0, 3, naoc, naoc)) if with_k else numpy.empty(1)
        drv = libv.SOCVHFgen_fock1_jk2 if k_only else libv.SOCVHFgen_fock1_jk
        flags = (ctypes.c_int(0), ctypes.c_int(1)) if k_only else (ctypes.c_int(int(with_k)),)
        drv(
            vjc.ctypes.data_as(ctypes.c_void_p), vkc.ctypes.data_as(ctypes.c_void_p),
            dmc.ctypes.data_as(ctypes.c_void_p), *flags,
            q.ctypes.data_as(ctypes.c_void_p), dcond.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(tol), ctypes.c_double(prim_tol), opt,
            ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(molc.nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(a0), ctypes.c_int(a1))
        if not k_only:
            vj[a0:a1] = lib.einsum('pi,axpq,qj->axij', c, vjc, c)
        if with_k:
            vk[a0:a1] = lib.einsum('pi,axpq,qj->axij', c, vkc, c)
    return vj, vk


def _df_mode(mf):
    '''None (no density fitting), 'j' (density-fitted J, exact K: mf.density_fit(
    only_dfj=True)) or 'jk' (J and K fitted).'''
    from pyscf.df import df_jk
    if not isinstance(mf, df_jk._DFHF):
        return None
    return 'j' if mf.only_dfj else 'jk'


def _df_auxmol(mf):
    from pyscf import df
    with_df = mf.with_df
    if with_df.auxmol is None:
        return df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    return with_df.auxmol


def _rij_engine(mf):
    '''the RIJ SCF's direct RI-J engine (socutils.scf.md_j, socutils.scf.rij_direct) for its molecule, or None'''
    cache = getattr(mf, '_rij_cache', None)
    eng = cache.get('direct') if cache else None
    return eng if eng is not None and eng.mol is mf.mol else None


def _rij_cderi(mf):
    '''the RIJ SCF's resident L^{-1} (P|ij) (socutils.scf.md_j), or None'''
    cache = getattr(mf, '_rij_cache', None)
    return cache.get('B') if cache else None


def ri_j_deriv(mol, auxmol, dm, with_hess=True, with_vj1=True, tol=ghf_hess.HESS2E_TOL, max_memory=None,
               cderi=None, occ=None, engine=None):
    '''Nuclear derivatives of the density-fitted Coulomb energy
    E_J = 1/2 rho^T M^{-1} rho at a fixed symmetric D (rho_P = (P|ij) D_ij,
    M = (P|Q), d = M^{-1} rho), auxiliary-basis response included (the exact
    derivatives of the DF energy, pyscf's auxbasis_response = 2):

        hess^{xi eta} = d . rho^{xi eta} - 1/2 d M^{xi eta} d + Y^xi M^{-1} Y^eta,
        vj1^xi_ij     = sum_P (ij|P)^xi d_P + (ij|P) (M^{-1} Y^xi)_P,
        Y^xi          = rho^xi - M^xi d.

    The three-centre derivative parts run in one pass over the shell triplets
    with bra derivatives only (lib/vhf/ri_j_deriv.c); the metric terms and
    (ij|P) M^{-1} Y are dense algebra over the auxiliary basis.  Returns
    (hess (natm, natm, 3, 3) or None, vj1 (natm, 3, nao, nao) or None), or None
    if the library is unavailable.  cderi: the SCF's L^{-1} (P|ij) (naux, npair)
    (ri_j_grad): d = L^{-T} (cderi D) and (ij|P) M^{-1} Y = cderi^T L^{-1} Y
    without the three-centre integrals.  occ (with with_vj1, without with_hess): return
    vj1 C_occ (natm, 3, nao, nocc) instead, the auxiliary-centre part of W blocked over the
    auxiliary atoms (SOCVHFri_j_deriv_blk), no (natm, 3, nao, nao) array.'''
    import ctypes
    from pyscf import df
    from pyscf.gto import moleintor
    from pyscf.df.grad.rhf import _gen_metric_solver
    from pyscf.ao2mo.outcore import balance_partition
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFri_j_deriv') or mol.cart:
        return None
    if max_memory is None:
        max_memory = mol.max_memory
    dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
    natm, nao, nbas = mol.natm, mol.nao, mol.nbas
    naux = auxmol.nao
    npair = nao * (nao + 1) // 2
    blk = max(16, min(naux, int(max_memory * .2e6 / 8 / npair)))
    aux_ranges = balance_partition(auxmol.ao_loc, blk)
    # rho_P over the lower triangle of the s2ij integrals
    dm_tril = lib.pack_tril(dm + dm.T - numpy.diag(numpy.diag(dm)))
    solve = _gen_metric_solver(auxmol.intor('int2c2e', aosym='s1'))
    low = None
    if cderi is not None and getattr(cderi, 'shape', None) == (naux, npair):
        import scipy.linalg
        try:
            # the Cholesky pyscf's cholesky_eri took (it falls back to eigh only where this fails)
            low = scipy.linalg.cholesky(auxmol.intor('int2c2e', hermi=1), lower=True)
        except scipy.linalg.LinAlgError:
            low = None
    # the direct RI-J engine (socutils.scf.rij_direct) for the three-centre parts: the Hessian call
    # (no vj1) and the occupied Fock1 call
    direct = None
    if RIJ_DERIV_DIRECT and auxmol.natm == natm and (not with_vj1 or (occ is not None and not with_hess)):
        from socutils.scf import rij_direct
        if engine is not None and engine.mol is mol and engine.auxmol is auxmol:
            direct = engine
        elif rij_direct.available(mol, auxmol):
            direct = rij_direct.RIJDirect(mol, auxmol)
        if direct is not None:
            direct.tol = min(direct.tol, rij_direct.TOL)
    if low is not None:
        c = lib.dot(cderi, dm_tril[:, None])[:, 0]              # L^{-1} rho
        dfit = scipy.linalg.solve_triangular(low, c, lower=True, trans='T')
    elif direct is not None:
        dfit = direct.fit(direct.get_rho(dm))
    else:
        rho = numpy.empty(naux)
        for k0, k1, _ in aux_ranges:
            p0, p1 = auxmol.ao_loc[k0], auxmol.ao_loc[k1]
            int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', shls_slice=(0, nbas, 0, nbas, k0, k1))
            rho[p0:p1] = lib.dot(dm_tril[None], int3c)[0]
        int3c = None
        dfit = solve(rho.copy())

    # Schwarz factors: AO pairs over the bra derivative types, sqrt((P|P))
    q = numpy.max([ghf_hess._q_cond(mol, intor)[k] for intor in ('int2e_ip1', 'int2e_ipip1', 'int2e_ipvip1')
                   for k in (0, 1)], axis=0)
    q = numpy.ascontiguousarray(numpy.maximum(q, q.T))
    aux_loc = auxmol.ao_loc_nr()
    m2 = auxmol.intor('int2c2e', aosym='s1')
    qP = numpy.sqrt(numpy.maximum.reduceat(abs(numpy.diag(m2)), aux_loc[:-1]))
    m2 = None
    dPcond = numpy.ascontiguousarray(numpy.maximum.reduceat(abs(dfit), aux_loc[:-1]))
    dcond = ghf_hess._dm_cond(mol, [dm])
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = moleintor.make_loc(bas, 'int3c2e_sph').astype(numpy.int32)
    opts = [moleintor.make_cintopt(atm, bas, env, s) for s in ('int3c2e_ip1_sph', 'int3c2e_ipip1_sph',
                                                               'int3c2e_ipvip1_sph')]
    hess = numpy.zeros((natm, natm, 3, 3)) if with_hess else None
    hess_md = None
    if with_hess and RIJ_HESS_MD and auxmol.natm == natm:
        hess_md = _md_rij_hess(libv, mol, auxmol, dm, dfit, tol)
    Y = numpy.zeros((natm * 3, naux))
    aux_atom = auxmol._bas[:, gto.ATOM_OF]
    blocked = (occ is not None and with_vj1 and not with_hess and hasattr(libv, 'SOCVHFri_j_deriv_blk')
               and auxmol.natm == natm and bool(numpy.all(numpy.diff(aux_atom) >= 0)))
    W = numpy.zeros((natm, 3, nao, nao)) if with_vj1 and not blocked else None
    null = lib.c_null_ptr()
    if direct is not None and not blocked and not with_vj1 and (not with_hess or hess_md is not None):
        Y = direct.deriv_y(dm)
    elif direct is not None and blocked:
        occ = numpy.ascontiguousarray(occ)
        nocc = occ.shape[1]
        vj1c = numpy.zeros((natm, 3, nao, nocc))
        Y = direct.deriv_y(dm)
        npao = direct.c.shape[0]
        # the auxiliary centres in blocks: Waux (primitive and AO) within a fifth of max_memory
        nc = max(1, min(natm, int(max_memory * .2e6 / (3 * (npao * npao + 2 * npao * nocc) * 8))))
        Wrows = None
        for c0 in range(0, natm, nc):
            c1 = min(natm, c0 + nc)
            Wr, Waux = direct.deriv_w(dfit, c0, c1, occ=occ)       # (V + V^T) C_occ of the block
            if Wrows is None:
                Wrows = Wr
            Wr = None
            vj1c[c0:c1] += Waux
            Waux = None
        aoslices = mol.aoslice_by_atom()
        for ia in range(natm):                                     # the i-derivative rows of atom A
            p0, p1 = aoslices[ia][2:]
            R = numpy.ascontiguousarray(Wrows[:, p0:p1])
            vj1c[ia, :, p0:p1] += lib.dot(R.reshape(-1, nao), occ).reshape(3, p1 - p0, nocc)
            for x in range(3):
                vj1c[ia, x] += lib.dot(R[x].T, occ[p0:p1])
        Wrows = None
    elif blocked:
        occ = numpy.ascontiguousarray(occ)
        nocc = occ.shape[1]
        vj1c = numpy.zeros((natm, 3, nao, nocc))
        Wrows = numpy.zeros((3, nao, nao))
        first = numpy.searchsorted(aux_atom, numpy.arange(natm + 1))
        nc = max(1, min(natm, int(max_memory * .25e6 / (3 * nao * nao * 8))))
        Waux = numpy.empty((nc, 3, nao, nao))
        Yc = numpy.empty_like(Y)
        for c0 in range(0, natm, nc):
            c1 = min(natm, c0 + nc)
            ksh0, ksh1 = nbas + int(first[c0]), nbas + int(first[c1])
            if ksh1 == ksh0:
                continue
            libv.SOCVHFri_j_deriv_blk(
                Yc.ctypes.data_as(ctypes.c_void_p), Wrows.ctypes.data_as(ctypes.c_void_p),
                Waux.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c0), ctypes.c_int(c1 - c0),
                ctypes.c_int(ksh0), ctypes.c_int(ksh1),
                dm.ctypes.data_as(ctypes.c_void_p), dfit.ctypes.data_as(ctypes.c_void_p),
                q.ctypes.data_as(ctypes.c_void_p), qP.ctypes.data_as(ctypes.c_void_p),
                dcond.ctypes.data_as(ctypes.c_void_p), dPcond.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(tol), opts[0], opts[1], opts[2], ao_loc.ctypes.data_as(ctypes.c_void_p),
                atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbas), ctypes.c_int(auxmol.nbas), env.ctypes.data_as(ctypes.c_void_p))
            Y += Yc
            for C in range(c0, c1):
                V = Waux[C - c0]                                   # (V + V^T) C_occ
                vj1c[C] += lib.dot(V.reshape(3 * nao, nao), occ).reshape(3, nao, nocc)
                for x in range(3):
                    vj1c[C, x] += lib.dot(V[x].T, occ)
        Waux = Yc = None
        aoslices = mol.aoslice_by_atom()
        for ia in range(natm):                                     # the i-derivative rows of atom A
            p0, p1 = aoslices[ia][2:]
            R = numpy.ascontiguousarray(Wrows[:, p0:p1])
            vj1c[ia, :, p0:p1] += lib.dot(R.reshape(-1, nao), occ).reshape(3, p1 - p0, nocc)
            for x in range(3):
                vj1c[ia, x] += lib.dot(R[x].T, occ[p0:p1])
        Wrows = None
    else:
        libv.SOCVHFri_j_deriv(
            hess.ctypes.data_as(ctypes.c_void_p) if with_hess and hess_md is None else null,
            Y.ctypes.data_as(ctypes.c_void_p),
            W.ctypes.data_as(ctypes.c_void_p) if with_vj1 else null,
            dm.ctypes.data_as(ctypes.c_void_p), dfit.ctypes.data_as(ctypes.c_void_p),
            q.ctypes.data_as(ctypes.c_void_p), qP.ctypes.data_as(ctypes.c_void_p),
            dcond.ctypes.data_as(ctypes.c_void_p), dPcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol),
            opts[0], opts[1], opts[2], ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            ctypes.c_int(auxmol.nbas), env.ctypes.data_as(ctypes.c_void_p))

    # Y^{A,x} -= M^{A,x} d with M^{A,x}_PQ = d/dR_{A,x} (P|Q), int2c2e_ip1 = (nabla P|Q)
    auxslices = auxmol.aoslice_by_atom()[:, 2:]
    ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
    v = numpy.einsum('xpq,q->xp', ip1, dfit)
    Y = Y.reshape(natm, 3, naux)
    for ia, (p0, p1) in enumerate(auxslices):
        Y[ia, :, p0:p1] += v[:, p0:p1]
        Y[ia] += numpy.einsum('xqp,q->xp', ip1[:, p0:p1], dfit[p0:p1])
    ip1 = v = None
    Y = Y.reshape(natm * 3, naux)
    Z = solve(Y.T.copy())                                     # M^{-1} Y, (naux, 3 natm)
    if hess_md is not None:
        hess += hess_md
    if with_hess:
        hess += lib.dot(Y, Z).reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
        # -1/2 d M^{xi eta} d: for A != B, sum_{P in A, Q in B} d_P d_Q (nabla nabla P|Q)
        # (translational invariance of (P|Q)); H_AA = -sum_{B != A} H_AB
        ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1').reshape(9, naux, naux)
        starts = auxslices[:, 0]
        S = numpy.empty((9, natm, natm))
        for x in range(9):
            u = ipip1[x] * dfit[:, None] * dfit[None, :]
            S[x] = numpy.add.reduceat(numpy.add.reduceat(u, starts, axis=0), starts, axis=1)
        ipip1 = u = None
        S = S.transpose(1, 2, 0).reshape(natm, natm, 3, 3)
        for ia in range(natm):
            S[ia, ia] = 0
            S[ia, ia] = -S[ia].sum(axis=0)
        hess += S
    vj1 = None
    if blocked:
        # + (ij|P) M^{-1} Y, column blocks straight to the occupied form
        kc = max(1, int(max_memory * .1e6 / (nao * nao * 8)))
        vf = vj1c.reshape(natm * 3, nao, nocc)
        if low is not None:
            U = scipy.linalg.solve_triangular(low, Y.T, lower=True)      # L^{-1} Y, (naux, 3 natm)
            for k0 in range(0, natm * 3, kc):
                k1 = min(natm * 3, k0 + kc)
                T = lib.unpack_tril(numpy.ascontiguousarray(lib.dot(cderi.T, U[:, k0:k1]).T))
                vf[k0:k1] += lib.dot(T.reshape(-1, nao), occ).reshape(k1 - k0, nao, nocc)
        elif direct is not None:
            # J of the fitted vectors by the direct engine (no three-centre tensor), several per pass
            npao = direct.c.shape[0]
            kd = max(1, min(256, int(max_memory * .2e6 / ((npao * npao + 2 * nao * nao) * 8))))
            for k0 in range(0, natm * 3, kd):
                k1 = min(natm * 3, k0 + kd)
                vf[k0:k1] += direct.get_vj_fit_multi(Z[:, k0:k1].T, occ)
        else:
            T = numpy.zeros((npair, natm * 3))
            for k0_, k1_, _ in aux_ranges:
                p0, p1 = auxmol.ao_loc[k0_], auxmol.ao_loc[k1_]
                int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij',
                                         shls_slice=(0, nbas, 0, nbas, k0_, k1_))
                T += lib.dot(int3c, Z[p0:p1])
            int3c = None
            for k0 in range(0, natm * 3, kc):
                k1 = min(natm * 3, k0 + kc)
                M = lib.unpack_tril(numpy.ascontiguousarray(T[:, k0:k1].T))
                vf[k0:k1] += lib.dot(M.reshape(-1, nao), occ).reshape(k1 - k0, nao, nocc)
            T = None
        return hess, vj1c
    if with_vj1:
        vj1 = W + W.transpose(0, 1, 3, 2)
        W = None
        if low is not None:
            # (ij|P) M^{-1} Y = cderi^T L^T L^{-T} L^{-1} Y
            T = lib.dot(cderi.T, scipy.linalg.solve_triangular(low, Y.T, lower=True))
        else:
            T = numpy.zeros((npair, natm * 3))
            for k0, k1, _ in aux_ranges:
                p0, p1 = auxmol.ao_loc[k0], auxmol.ao_loc[k1]
                int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij',
                                         shls_slice=(0, nbas, 0, nbas, k0, k1))
                T += lib.dot(int3c, Z[p0:p1])
            int3c = None
        vj1 += lib.unpack_tril(numpy.ascontiguousarray(T.T)).reshape(natm, 3, nao, nao)
    return hess, vj1


RIJ_DERIV_DIRECT = True  # ri_j_deriv's three-centre Y and W (Hessian, occupied Fock1 calls) by socutils.scf.rij_direct
RIJ_GRAD_DIRECT = True  # ri_j_grad by socutils.scf.rij_direct (no three-centre integrals) when available
RIJ_GRAD_MD = True      # ri_j_grad's three-centre part by the McMurchie-Davidson J engine (lib/vhf/md_j.c)


def _md_rij_setup(mol, auxmol, dm, dfit):
    '''the combined decontracted Cartesian basis of the J engine's RIJ passes: the bra, the auxiliary
    primitives, one zero-exponent s shell per auxiliary atom; the density D (bra block dm, auxiliary
    rows d / 2 in their s0 column).  (D, atm, bas, env, nbra, naux, s0sh, ao_loc, natm_total)'''
    from pyscf.gto import moleintor
    pb, cb = mol.decontract_basis(to_cart=True, aggregate=True)
    pa, ca = auxmol.decontract_basis(to_cart=True, aggregate=True)
    dmb = cb.dot(numpy.asarray(dm, dtype=numpy.double)).dot(cb.T)
    dk = ca.dot(dfit)
    natm = mol.natm
    atm, bas, env = gto.conc_env(pb._atm, pb._bas, pb._env, pa._atm, pa._bas, pa._env)
    nbra, naux = pb.nbas, pa.nbas
    env = numpy.hstack([env, [0., numpy.sqrt(4 * numpy.pi)]])
    pexp, pcoef = env.size - 2, env.size - 1
    s0 = []
    for ia in range(pa.natm):
        b = numpy.zeros(gto.BAS_SLOTS, dtype=numpy.int32)
        b[gto.ATOM_OF] = natm + ia
        b[gto.NPRIM_OF] = b[gto.NCTR_OF] = 1
        b[gto.PTR_EXP] = pexp
        b[gto.PTR_COEFF] = pcoef
        s0.append(b)
    bas = numpy.asarray(numpy.vstack([bas, numpy.asarray(s0)]), dtype=numpy.int32)
    s0sh = numpy.asarray(nbra + naux + pa._bas[:, gto.ATOM_OF], dtype=numpy.int32)
    ao_loc = moleintor.make_loc(bas, 'cart').astype(numpy.int32)
    nao = ao_loc[-1]
    nb = ao_loc[nbra]
    D = numpy.zeros((nao, nao))
    D[:nb, :nb] = dmb
    for k in range(naux):
        r0, r1 = ao_loc[nbra + k], ao_loc[nbra + k + 1]
        D[r0:r1, ao_loc[s0sh[k]]] = .5 * dk[r0 - nb:r1 - nb]
    return D, atm, bas, env, nbra, naux, s0sh, ao_loc, natm + pa.natm


def _md_rij_grad(libv, mol, auxmol, dm, dfit, tol):
    '''sum_P d_P sum_ij D_ij d(ij|P)/dR (natm, 3) by SOCVHFmd_rij_grad: the bra and the auxiliary basis
    decontracted to Cartesian primitives, every auxiliary primitive paired with a zero-exponent s
    function on its centre; None if unavailable'''
    import ctypes
    if not hasattr(libv, 'SOCVHFmd_rij_grad') or max(mol._bas[:, gto.ANG_OF].max(),
                                                       auxmol._bas[:, gto.ANG_OF].max()) > MD_J_LMAX:
        return None
    D, atm, bas, env, nbra, naux, s0sh, ao_loc, ntot = _md_rij_setup(mol, auxmol, dm, dfit)
    natm = mol.natm
    grad = numpy.zeros((ntot, 3))
    ex = numpy.zeros(1)
    libv.SOCVHFmd_rij_grad(
        grad.ctypes.data_as(ctypes.c_void_p), ex.ctypes.data_as(ctypes.c_void_p),
        D.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol), ctypes.c_int(nbra), ctypes.c_int(naux),
        s0sh.ctypes.data_as(ctypes.c_void_p), ao_loc.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ntot),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(bas.shape[0]), env.ctypes.data_as(ctypes.c_void_p))
    return grad[:natm] + grad[natm:], ex[0]


RIJ_HESS_MD = True      # ri_j_deriv's three-centre second derivatives by the J engine (SOCVHFmd_rij_hess)


def _md_rij_hess(libv, mol, auxmol, dm, dfit, tol):
    '''sum_P d_P sum_ij D_ij d2(ij|P)/dR dR' (natm, natm, 3, 3) by SOCVHFmd_rij_hess; None if unavailable'''
    import ctypes
    if not hasattr(libv, 'SOCVHFmd_rij_hess') or max(mol._bas[:, gto.ANG_OF].max(),
                                                       auxmol._bas[:, gto.ANG_OF].max()) > MD_J_LMAX:
        return None
    D, atm, bas, env, nbra, naux, s0sh, ao_loc, ntot = _md_rij_setup(mol, auxmol, dm, dfit)
    natm = mol.natm
    hess = numpy.zeros((ntot, ntot, 3, 3))
    libv.SOCVHFmd_rij_hess.restype = ctypes.c_int
    ok = libv.SOCVHFmd_rij_hess(
        hess.ctypes.data_as(ctypes.c_void_p), D.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol),
        ctypes.c_int(nbra), ctypes.c_int(naux), s0sh.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ntot),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(bas.shape[0]), env.ctypes.data_as(ctypes.c_void_p))
    if not ok:
        return None
    # the auxiliary centres are the molecule's atoms (natm + ia): fold
    return hess[:natm, :natm] + hess[:natm, natm:] + hess[natm:, :natm] + hess[natm:, natm:]


def ri_j_grad(mol, auxmol, dm, tol=None, max_memory=None, cderi=None, dfit=None, engine=None):
    '''Gradient of the density-fitted Coulomb energy E_J = 1/2 rho^T M^{-1} rho
    at a symmetric D, auxiliary-basis response included (pyscf's
    auxbasis_response): g^xi = d . rho^xi - 1/2 d M^xi d, d = M^{-1} rho, the
    three-centre part rho^xi by the pass of ri_j_deriv without the Hessian
    and vj1 terms (lib/vhf/ri_j_deriv.c).  cderi: L^{-1} (P|ij) (naux, npair)
    of the SCF (Cholesky L of the metric, as pyscf's cholesky_eri) to take d
    from, d = L^{-T} (cderi D), instead of the three-centre integrals.
    Returns (natm, 3), or None if the library is unavailable.'''
    import ctypes
    from pyscf import df
    from pyscf.gto import moleintor
    from pyscf.df.grad.rhf import _gen_metric_solver
    from pyscf.ao2mo.outcore import balance_partition
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFri_j_deriv') or mol.cart:
        return None
    if tol is None:
        tol = ghf_hess.HESS2E_TOL
    if max_memory is None:
        max_memory = mol.max_memory
    dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
    natm, nao, nbas = mol.natm, mol.nao, mol.nbas
    naux = auxmol.nao
    npair = nao * (nao + 1) // 2
    blk = max(16, min(naux, int(max_memory * .2e6 / 8 / npair)))
    dm_tril = lib.pack_tril(dm + dm.T - numpy.diag(numpy.diag(dm)))
    direct = None
    if RIJ_GRAD_DIRECT and auxmol.natm == mol.natm:
        # the direct RI-J engine (socutils.scf.rij_direct): d without the three-centre integrals
        # (when no factor is given) and the three-centre derivatives by its gradient pass
        from socutils.scf import rij_direct
        if engine is not None and engine.mol is mol and engine.auxmol is auxmol:
            direct = engine
        elif rij_direct.available(mol, auxmol):
            direct = rij_direct.RIJDirect(mol, auxmol)
    if dfit is None and cderi is not None and getattr(cderi, 'shape', None) == (naux, npair):
        import scipy.linalg
        try:
            low = scipy.linalg.cholesky(auxmol.intor('int2c2e', hermi=1), lower=True)
        except scipy.linalg.LinAlgError:
            low = None
        if low is not None:
            c = lib.dot(cderi, dm_tril[:, None])[:, 0]          # L^{-1} rho
            dfit = scipy.linalg.solve_triangular(low, c, lower=True, trans='T')
    if direct is not None:
        return direct.grad(dm, dfit)
    if dfit is None:
        rho = numpy.empty(naux)
        for k0, k1, _ in balance_partition(auxmol.ao_loc, blk):
            p0, p1 = auxmol.ao_loc[k0], auxmol.ao_loc[k1]
            int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', shls_slice=(0, nbas, 0, nbas, k0, k1))
            rho[p0:p1] = lib.dot(dm_tril[None], int3c)[0]
        int3c = None
        dfit = _gen_metric_solver(auxmol.intor('int2c2e', aosym='s1'))(rho.copy())
    g = None
    if RIJ_GRAD_MD and auxmol.natm == mol.natm:
        r = _md_rij_grad(libv, mol, auxmol, dm, dfit, tol)
        if r is not None:
            g = r[0]
    if g is not None:
        v = numpy.einsum('xpq,q->xp', auxmol.intor('int2c2e_ip1', aosym='s1'), dfit)
        for ia, (p0, p1) in enumerate(auxmol.aoslice_by_atom()[:, 2:]):
            g[ia] += v[:, p0:p1].dot(dfit[p0:p1])
        return g
    q = numpy.max([ghf_hess._q_cond(mol, 'int2e_ip1')[k] for k in (0, 1)], axis=0)
    q = numpy.ascontiguousarray(numpy.maximum(q, q.T))
    aux_loc = auxmol.ao_loc_nr()
    qP = numpy.sqrt(numpy.maximum.reduceat(abs(numpy.diag(auxmol.intor('int2c2e', aosym='s1'))), aux_loc[:-1]))
    dPcond = numpy.ascontiguousarray(numpy.maximum.reduceat(abs(dfit), aux_loc[:-1]))
    dcond = ghf_hess._dm_cond(mol, [dm])
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = moleintor.make_loc(bas, 'int3c2e_sph').astype(numpy.int32)
    opts = [moleintor.make_cintopt(atm, bas, env, s) for s in ('int3c2e_ip1_sph', 'int3c2e_ipip1_sph',
                                                               'int3c2e_ipvip1_sph')]
    Y = numpy.zeros((natm * 3, naux))
    null = lib.c_null_ptr()
    libv.SOCVHFri_j_deriv(
        null, Y.ctypes.data_as(ctypes.c_void_p), null,
        dm.ctypes.data_as(ctypes.c_void_p), dfit.ctypes.data_as(ctypes.c_void_p),
        q.ctypes.data_as(ctypes.c_void_p), qP.ctypes.data_as(ctypes.c_void_p),
        dcond.ctypes.data_as(ctypes.c_void_p), dPcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol),
        opts[0], opts[1], opts[2], ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        ctypes.c_int(auxmol.nbas), env.ctypes.data_as(ctypes.c_void_p))
    g = Y.dot(dfit).reshape(natm, 3)                          # d . rho^xi
    # -1/2 d M^{A,x} d = sum_{P in A} d_P (nabla P|Q)_x d_Q
    v = numpy.einsum('xpq,q->xp', auxmol.intor('int2c2e_ip1', aosym='s1'), dfit)
    for ia, (p0, p1) in enumerate(auxmol.aoslice_by_atom()[:, 2:]):
        g[ia] += v[:, p0:p1].dot(dfit[p0:p1])
    return g


def rhf_make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''pyscf.hessian.rhf.make_h1 with the 2e part from fock1_jk (density-fitted
    J: ri_j_deriv).'''
    mol = hessobj.mol
    mf = hessobj.base
    dfm = _df_mode(mf)
    if dfm == 'jk' or (atmlst is not None and list(atmlst) != list(range(mol.natm))):
        if dfm is not None:
            from pyscf.df.hessian import rhf as df_rhf_hess
            return df_rhf_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
        return rhf_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    v = _fock1(hessobj, dm0, True)
    if v is None:
        if dfm is not None:
            from pyscf.df.hessian import rhf as df_rhf_hess
            return df_rhf_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
        return rhf_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    vj, vk = v
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    return [vj[ia] - .5 * vk[ia] + hcore_deriv(ia) for ia in range(mol.natm)]


def _fock1(hessobj, dm0, with_k, occ=None):
    '''(vj1, vk1) of all atoms: fock1_jk, or with density-fitted J ri_j_deriv
    for J and fock1_jk for the exact K; None if a library is missing.  occ:
    the density-fitted vj1 may come as vj1 C_occ (natm, 3, nao, nocc).'''
    mol = hessobj.mol
    mf = hessobj.base
    max_memory = max(1000, mf.max_memory - lib.current_memory()[0])
    if _df_mode(mf) is None:
        return fock1_jk(mol, dm0, with_k=with_k, max_memory=max_memory)
    eng = _rij_engine(mf)
    r = ri_j_deriv(mol, eng.auxmol if eng is not None else _df_auxmol(mf), dm0, with_hess=False,
                   max_memory=max_memory, cderi=_rij_cderi(mf), occ=occ, engine=eng)
    if r is None:
        return None
    vk = None
    if with_k:
        # K only: the density-fitted J comes from ri_j_deriv
        v = fock1_jk(mol, dm0, with_k=True, max_memory=max_memory, with_j=not FOCK1_K_ONLY)
        if v is None:
            return None
        vk = v[1]
    return r[1], vk


FOCK1_K_ONLY = True     # _fock1 with density-fitted J: fock1_jk for K only (else J + K, J dropped)
XCD1_LOCAL_VIP = True   # xc_deriv1_batched: the vxc term per sub-block on its significant AOs (else pyscf's)
XCD1_OCC_LOCAL = True   # xc_deriv1_batched(occ=True): D phi, rho, the functional, d_A rho per sub-block too


def _xc_deriv1_occ_local(ni, xc_code, xctype, gga, nvar, ao, weight, p0, dm0, atom_of, non0tab, nf_shell,
                         nsub, libdft, mocc, vmat, v_ip, dgemm, mkl_local):
    '''one pyscf grid block of xc_deriv1_batched(occ=True), sub-block by sub-block on the AOs the grid's
    non0tab flags on it and their atoms: D phi_k, rho and the functional, the vxc term v_ip, d_A rho
    (SOCDFTrks_dR_rho), f_xc d_A rho and (V + V^T) C_occ (SOCDFTxcd1_occ); vmat (3 natm, nao, nocc)'''
    from pyscf.dft.numint import BLKSIZE
    ngrid = weight.size
    aoT, cs = _ao_stack(ao)
    ncomp = aoT.shape[0]
    nao = aoT.shape[1]
    nocc = mocc.shape[1]
    for s0 in range(0, ngrid, nsub):
        s1 = min(ngrid, s0 + nsub)
        gs = s1 - s0
        aom = numpy.repeat(non0tab[(p0 + s0) // BLKSIZE:(p0 + s1 + BLKSIZE - 1) // BLKSIZE].any(axis=0), nf_shell)
        idx = numpy.nonzero(aom)[0].astype(numpy.int32)
        nl = idx.size
        if nl == 0:
            continue
        ca = nl * gs
        aos = numpy.empty((ncomp, nl, gs))
        libdft.SOCDFTgather_ao(aos.ctypes, aoT.ctypes, idx.ctypes, ctypes.c_int(ncomp), ctypes.c_int(nao),
                               ctypes.c_int(nl), ctypes.c_int(ngrid), ctypes.c_int(s0), ctypes.c_int(gs),
                               ctypes.c_long(cs))
        dl = numpy.ascontiguousarray(dm0[idx[:, None], idx])
        dT = numpy.empty((nvar, nl, gs))
        for k in range(nvar):
            lib.dot(dl, aos[k], c=dT[k])
        rho = numpy.empty((nvar, gs))
        rho[0] = numpy.einsum('ig,ig->g', aos[0], dT[0])
        for j in range(1, nvar):
            rho[j] = 2 * numpy.einsum('ig,ig->g', aos[j], dT[0])
        vxc, fxc = ni.eval_xc_eff(xc_code, rho if gga else rho[0], 2, xctype=xctype)[1:3]
        w = weight[s0:s1]
        wv = w * vxc.reshape(nvar, gs)
        if gga:
            wv[0] *= .5
        # v_ip[x] += sum_g dphi_x (sum_k wv_k phi_k)^T + (wv_0 dphi_x + sum_j wv_j d_xj phi) phi^T (GGA)
        wv = numpy.ascontiguousarray(wv)
        a0 = numpy.empty((nl, gs))
        libdft.SOCDFTaow_multi(a0.ctypes, wv.ctypes, aos.ctypes, ctypes.c_int(1), ctypes.c_int(nvar),
                               ctypes.c_int(nl), ctypes.c_int(gs))
        if gga:
            Lv = numpy.empty((3, nl, 2 * gs))
            Rv = numpy.empty((nl, 2 * gs))
            Lv[:, :, :gs] = aos[1:4]
            for x in range(3):
                t = Lv[x, :, gs:]
                numpy.multiply(aos[1 + x], wv[0], out=t)
                t += aos[_D2[x, 0]] * wv[1]
                t += aos[_D2[x, 1]] * wv[2]
                t += aos[_D2[x, 2]] * wv[3]
            Rv[:, :gs] = a0
            Rv[:, gs:] = aos[0]
        else:
            Lv, Rv = aos[1:4], a0
        vl = lib.dot(Lv.reshape(3 * nl, -1), Rv.T)
        libdft.SOCDFTscatter_add_dm(v_ip.ctypes, vl.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(3),
                                    ctypes.c_int(nao))
        # d_A rho of the sub-block's atoms (local numbering), f_xc d_A rho, (V + V^T) C_occ
        atoms, atl = numpy.unique(atom_of[idx], return_inverse=True)
        atl = numpy.ascontiguousarray(atl, dtype=numpy.int32)
        na_l = atoms.size
        R = numpy.empty((3, nvar, na_l, gs))
        libdft.SOCDFTrks_dR_rho(R.ctypes, aos.ctypes, dT.ctypes, ctypes.c_int(int(gga)), ctypes.c_int(nvar),
                                ctypes.c_int(gs), ctypes.c_int(nl), atl.ctypes, ctypes.c_int(na_l),
                                ctypes.c_long(ca))
        wf = numpy.ascontiguousarray(w * fxc.reshape(nvar, nvar, gs))
        wv1 = numpy.empty((3 * na_l, nvar, gs))
        libdft.SOCDFTfxc_contract_R(wv1.ctypes, wf.ctypes, R.ctypes, ctypes.c_int(na_l), ctypes.c_int(nvar),
                                    ctypes.c_int(gs), ctypes.c_double(2. if gga else 1.), ctypes.c_int(int(gga)))
        rows = numpy.asarray((3 * atoms[:, None] + numpy.arange(3)).ravel(), dtype=numpy.int32)
        cl = numpy.ascontiguousarray(mocc[idx])
        psi = numpy.empty((nvar, gs, nocc))
        for k in range(nvar):
            lib.dot(aos[k].T, cl, c=psi[k])
        libdft.SOCDFTxcd1_occ(vmat.ctypes, wv1.ctypes, aos.ctypes, psi.ctypes, idx.ctypes, rows.ctypes,
                              ctypes.c_int(rows.size), ctypes.c_int(nvar), ctypes.c_int(nl), ctypes.c_int(gs),
                              ctypes.c_int(nocc), ctypes.c_int(nao), dgemm, mkl_local)


def xc_deriv1_batched(hessobj, mo_coeff, mo_occ, max_memory=4000, occ=False):
    """pyscf.hessian.rks._get_vxc_deriv1 (LDA / GGA): the first-order XC
    matrices d V_xc / dR_{A,x} of all atoms at fixed density.  The 3 natm
    first-order potentials f_xc d_A rho of a grid sub-block go through one
    batched aow pass (lib/dft/fxc_multi.c) and one GEMM, restricted to the AOs
    significant on the sub-block, instead of pyscf's three nao x nao GEMMs per
    atom and grid block; d_A rho for all atoms comes from one C pass
    (lib/dft/xc_hess.c).  (natm, 3, nao, nao); occ: projected on the occupied
    orbitals, (natm, 3, nao, nocc), from one (nl x 2 gs) . (2 gs x nocc) GEMM
    per (atom, x) and sub-block (SOCDFTxcd1_occ) instead of the nl x nl one."""
    from pyscf.dft import numint
    from pyscf.dft.numint import BLKSIZE
    from pyscf.grad import rks as rks_grad
    mol = hessobj.mol
    mf = hessobj.base
    grids = hessobj.grids if hessobj.grids is not None else mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    libdft = _load_socdft()
    mocc = numpy.ascontiguousarray(mo_coeff[:, mo_occ > 0])
    if (xctype not in ('LDA', 'GGA') or libdft is None or not hasattr(libdft, 'SOCDFTrks_dR_rho')
            or mol.cart):
        v = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
        return lib.einsum('axpq,qi->axpi', v, mocc) if occ else v
    gga = xctype == 'GGA'
    nvar = 4 if gga else 1
    ao_deriv = 2 if gga else 1
    nao = mo_coeff.shape[0]
    natm = mol.natm
    nd = 3 * natm
    ao_loc = mol.ao_loc_nr()
    nf_shell = ao_loc[1:] - ao_loc[:-1]
    non0tab = getattr(grids, 'non0tab', None)
    aoslices = mol.aoslice_by_atom()
    atom_of = numpy.zeros(nao, dtype=numpy.int32)
    for ia in range(natm):
        atom_of[aoslices[ia][2]:aoslices[ia][3]] = ia
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    v_ip = numpy.zeros((3, nao, nao))
    occ = occ and hasattr(libdft, 'SOCDFTxcd1_occ')
    nocc = mocc.shape[1]
    if occ:
        from socutils.dft.vxc_fast import _blas_ptrs
        dgemm, mkl_local = _blas_ptrs()
        vmat = numpy.zeros((nd, nao, nocc))
        psi_buf = numpy.empty(nvar * (max(1, FXC_SUBBLOCK // BLKSIZE) * BLKSIZE) * nocc)
    else:
        vmat = numpy.zeros((nd, nao, nao))              # index 3 * atom + x
    nsub = max(1, FXC_SUBBLOCK // BLKSIZE) * BLKSIZE
    ncomp_ao = 10 if gga else 4
    # pyscf blocks: AO values, R and the weights within ~30% of max_memory
    per_point = 8 * (ncomp_ao * nao + 2 * 3 * nvar * natm + nvar * nao)
    ng_blk = max(nsub, int(max_memory * .3e6 / per_point) // nsub * nsub)
    blk_mem = ng_blk * (ncomp_ao + 1) * nao * 8 / 1e6
    aow_buf = numpy.empty(nd * nao * nsub)
    local_vip = XCD1_LOCAL_VIP and nao >= XCHE_LOCAL_MIN_NAO
    nga = ncomp_ao if local_vip else nvar               # AO components gathered per sub-block
    aos_buf = numpy.empty(nga * nao * nsub)
    ws_buf = numpy.empty(nd * nvar * nsub)
    p1 = 0
    local_all = occ and XCD1_OCC_LOCAL and non0tab is not None and nao >= XCHE_LOCAL_MIN_NAO
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, blk_mem):
        ng = weight.size
        p0, p1 = p1, p1 + ng
        if local_all:
            _xc_deriv1_occ_local(ni, mf.xc, xctype, gga, nvar, ao, weight, p0, dm0, atom_of, non0tab, nf_shell,
                                 nsub, libdft, mocc, vmat, v_ip, dgemm, mkl_local)
            continue
        aoT, cs = _ao_stack(ao)
        dT = numpy.empty((nvar, nao, ng))                  # D phi_k^T (dm0 symmetric)
        for k in range(nvar):
            lib.dot(dm0, aoT[k], c=dT[k])
        rho = numpy.empty((nvar, ng))
        rho[0] = numint._contract_rho(ao[0], dT[0].T)
        for j in range(1, nvar):
            rho[j] = 2 * numint._contract_rho(ao[j], dT[0].T)
        vxc, fxc = ni.eval_xc_eff(mf.xc, rho if gga else rho[0], 2, xctype=xctype)[1:3]
        wv = weight * vxc.reshape(nvar, ng)
        if gga:
            wv[0] *= .5
        if not local_vip:
            if gga:
                rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)
            else:
                aow = numint._scale_ao(ao[0], wv[0])
                rks_grad._d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)
        R = numpy.empty((3, nvar, natm, ng))
        libdft.SOCDFTrks_dR_rho(R.ctypes, aoT.ctypes, dT.ctypes, ctypes.c_int(int(gga)), ctypes.c_int(nvar),
                                ctypes.c_int(ng), ctypes.c_int(nao), atom_of.ctypes, ctypes.c_int(natm),
                                ctypes.c_long(cs))
        wf = numpy.ascontiguousarray(weight * fxc.reshape(nvar, nvar, ng))
        # first-order potentials of all (atom, x), as pyscf: LDA f R (the *2 of
        # d rho comes with the + c.c. at the end), GGA f (2 R) with row 0 halved
        wv1 = numpy.empty((nd, nvar, ng))
        libdft.SOCDFTfxc_contract_R(wv1.ctypes, wf.ctypes, R.ctypes, ctypes.c_int(natm), ctypes.c_int(nvar),
                                    ctypes.c_int(ng), ctypes.c_double(2. if gga else 1.), ctypes.c_int(int(gga)))
        for s0 in range(0, ng, nsub):
            s1 = min(ng, s0 + nsub)
            gs = s1 - s0
            idx = None
            if non0tab is not None:
                shm = non0tab[(p0 + s0) // BLKSIZE:(p0 + s1 + BLKSIZE - 1) // BLKSIZE].any(axis=0)
                aom = numpy.repeat(shm, nf_shell)
                nl = int(aom.sum())
                if nl == 0:
                    continue
                if nl < nao:
                    idx = numpy.nonzero(aom)[0].astype(numpy.int32)
            if idx is None:
                nl = nao
            aos = aos_buf[:nga * nl * gs].reshape(nga, nl, gs)
            libdft.SOCDFTgather_ao(aos.ctypes, aoT.ctypes, None if idx is None else idx.ctypes, ctypes.c_int(nga),
                                   ctypes.c_int(nao), ctypes.c_int(nl), ctypes.c_int(ng), ctypes.c_int(s0),
                                   ctypes.c_int(gs), ctypes.c_long(cs))
            if local_vip:
                # v_ip[x] += sum_g dphi_x (sum_k wv_k phi_k)^T + (wv_0 dphi_x + sum_j wv_j d_xj phi) phi^T (GGA, wv_0 halved)
                wvs = numpy.ascontiguousarray(wv[:, s0:s1])
                a0 = numpy.empty((nl, gs))
                libdft.SOCDFTaow_multi(a0.ctypes, wvs.ctypes, aos.ctypes, ctypes.c_int(1), ctypes.c_int(nvar),
                                       ctypes.c_int(nl), ctypes.c_int(gs))
                if gga:
                    Lv = numpy.empty((3, nl, 2 * gs))
                    Rv = numpy.empty((nl, 2 * gs))
                    Lv[:, :, :gs] = aos[1:4]
                    for x in range(3):
                        t = Lv[x, :, gs:]
                        numpy.multiply(aos[1 + x], wvs[0], out=t)
                        t += aos[_D2[x, 0]] * wvs[1]
                        t += aos[_D2[x, 1]] * wvs[2]
                        t += aos[_D2[x, 2]] * wvs[3]
                    Rv[:, :gs] = a0
                    Rv[:, gs:] = aos[0]
                else:
                    Lv, Rv = aos[1:4], a0
                vl = lib.dot(Lv.reshape(3 * nl, -1), Rv.T)
                if idx is None:
                    v_ip += vl.reshape(3, nao, nao)
                else:
                    libdft.SOCDFTscatter_add_dm(v_ip.ctypes, vl.ctypes, idx.ctypes, ctypes.c_int(nl),
                                                ctypes.c_int(3), ctypes.c_int(nao))
            if occ:
                if idx is None:
                    idx = numpy.arange(nao, dtype=numpy.int32)
                atoms = numpy.unique(atom_of[idx])
                rows = numpy.asarray((3 * atoms[:, None] + numpy.arange(3)).ravel(), dtype=numpy.int32)
                ws = numpy.ascontiguousarray(wv1[rows, :, s0:s1])
                cl = numpy.ascontiguousarray(mocc[idx])
                psi = psi_buf[:nvar * gs * nocc].reshape(nvar, gs, nocc)
                for k in range(nvar):
                    lib.dot(aos[k].T, cl, c=psi[k])
                libdft.SOCDFTxcd1_occ(vmat.ctypes, ws.ctypes, aos.ctypes, psi.ctypes, idx.ctypes, rows.ctypes,
                                      ctypes.c_int(rows.size), ctypes.c_int(nvar), ctypes.c_int(nl),
                                      ctypes.c_int(gs), ctypes.c_int(nocc), ctypes.c_int(nao), dgemm, mkl_local)
                continue
            if idx is None or not XCD1_LOCAL_ATOMS:
                ws = ws_buf[:nd * nvar * gs].reshape(nd, nvar, gs)
                libdft.SOCDFTslice_grid(ws.ctypes, wv1.ctypes, ctypes.c_long(nd * nvar), ctypes.c_int(ng),
                                        ctypes.c_int(s0), ctypes.c_int(gs))
                aow = aow_buf[:nd * nl * gs].reshape(nd * nl, gs)
                libdft.SOCDFTaow_multi(aow.ctypes, ws.ctypes, aos.ctypes, ctypes.c_int(nd), ctypes.c_int(nvar),
                                       ctypes.c_int(nl), ctypes.c_int(gs))
                if idx is None:
                    lib.dot(aow, aos[0].T, beta=1, c=vmat.reshape(nd * nao, nao))
                else:
                    vl = lib.dot(aow, aos[0].T)
                    libdft.SOCDFTscatter_add_dm(vmat.ctypes, vl.ctypes, idx.ctypes, ctypes.c_int(nl),
                                                ctypes.c_int(nd), ctypes.c_int(nao))
                continue
            # d_A rho, hence the first-order potential of atom A, vanishes where none of A's AOs is
            # significant: only the atoms of the sub-block's AOs (their 3 rows each)
            atoms = numpy.unique(atom_of[idx])
            na_l = atoms.size
            nd_l = 3 * na_l
            rows = (3 * atoms[:, None] + numpy.arange(3)).ravel()
            ws = numpy.ascontiguousarray(wv1[rows, :, s0:s1])
            aow = aow_buf[:nd_l * nl * gs].reshape(nd_l * nl, gs)
            libdft.SOCDFTaow_multi(aow.ctypes, ws.ctypes, aos.ctypes, ctypes.c_int(nd_l), ctypes.c_int(nvar),
                                   ctypes.c_int(nl), ctypes.c_int(gs))
            vl = lib.dot(aow, aos[0].T)                              # (nd_l nl, nl)
            vmat_ = vmat.reshape(natm, 3 * nao * nao)
            vl_ = vl.reshape(na_l, 3 * nl * nl)
            for k, ia in enumerate(atoms):
                libdft.SOCDFTscatter_add_dm(vmat_[ia].ctypes, vl_[k].ctypes, idx.ctypes, ctypes.c_int(nl),
                                            ctypes.c_int(3), ctypes.c_int(nao))
    if occ:
        # vmat already holds (V + V^T) C; the v_ip rows of atom A: (P v) C + (P v)^T C
        vmat = vmat.reshape(natm, 3, nao, nocc)
        vc = lib.dot(v_ip.reshape(3 * nao, nao), mocc).reshape(3, nao, nocc)
        for ia in range(natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia, :, p0:p1] += vc[:, p0:p1]
            for x in range(3):
                vmat[ia, x] += lib.dot(v_ip[x, p0:p1].T, mocc[p0:p1])
        vmat *= -1
        return vmat
    vmat = vmat.reshape(natm, 3, nao, nao)
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        vmat[ia, :, p0:p1] += v_ip[:, p0:p1]
        vmat[ia] = -vmat[ia] - vmat[ia].transpose(0, 2, 1)
    return vmat


def rks_make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''pyscf.hessian.rks.make_h1 with the 2e part from fock1_jk (range-separated
    functionals and atom subsets fall back to pyscf).'''
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    dfm = _df_mode(mf)
    if (omega != 0 or (atmlst is not None and list(atmlst) != list(range(mol.natm)))
            or (dfm == 'jk' and hybrid)):
        if dfm is not None:
            from pyscf.df.hessian import rks as df_rks_hess
            return df_rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
        return rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    occ = getattr(hessobj, '_h1_occ', False)
    # the first-order K by COSX (occupied form only): d K / dR C on the SCF's COSX grid
    cx = _hess_cosx(mf, HESS_COSX_FOCK1) if hybrid and occ else None
    vk_cx = None
    if cx is not None:
        cached = getattr(hessobj, '_cosx_fock1', None)
        if cached is not None and cached[0] == mocc.tobytes():
            vk_cx = cached[1]
            hessobj._cosx_fock1 = None
        else:
            t0 = (logger.process_clock(), logger.perf_counter())
            vk_cx = cx.fock1(mocc)
            logger.timer(hessobj, 'Fock1 K (COSX)', *t0)
    v = _fock1(hessobj, dm0, hybrid and vk_cx is None, occ=mocc if occ else None)
    if v is None:
        if dfm is not None:
            from pyscf.df.hessian import rks as df_rks_hess
            return df_rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
        return rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    vj, vk = v
    vj_occ = occ and vj.shape[-1] == mocc.shape[1] != vj.shape[-2]
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    h1ao = xc_deriv1_batched(hessobj, mo_coeff, mo_occ, max_memory, occ=occ)
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    for ia in range(mol.natm):
        h = hcore_deriv(ia) if vj_occ else vj[ia] + hcore_deriv(ia)
        if hybrid and vk_cx is None:
            h -= .5 * hyb * vk[ia]
        if occ:
            h1ao[ia] += lib.dot(h.reshape(-1, h.shape[-1]), mocc).reshape(3, -1, mocc.shape[1])
            if vj_occ:
                h1ao[ia] += vj[ia]
            if vk_cx is not None:
                h1ao[ia] -= .5 * hyb * vk_cx[ia]
        else:
            h1ao[ia] += h
    return h1ao


CPHF_COSX = None        # K of the CPHF densities by COSX (socutils.scf.cosx): None -> auto (SCF ran COSX, large basis)
CPHF_COSX_GRID = 2      # its grid (cosx.GRIDX level)
CPHF_COSX_OCC = False   # the CPHF K as K C_occ (cosx.COSX.get_k_occ, one-sided fitted K, no K matrix)
CPHF_COSX_NAO_PER_ATOM = 12     # auto mode: AOs per atom from which the CPHF K goes to COSX
CPHF_COSX_FLOAT = True  # the many-density COSX contraction in single precision (False: double); K within ~1e-6
                        # relative (the COSX grid error ~1e-3), Hessians within 2e-9, harmonic frequencies within
                        # 1e-4 cm-1 and the same CPHF iterations (caffeine / C16H34 cc-pVDZ, C16H34 def2-TZVP);
                        # CPHF 12 - 20 % faster


HESS_COSX_FOCK1 = None  # the first-order K (make_h1) by COSX: None -> with the CPHF's auto mode (CPHF_COSX)
HESS_COSX_EXPLICIT = None   # the K second derivatives of the explicit part by COSX: None -> likewise


def _hess_cosx(mf, switch):
    '''the COSX object for a K term of the Hessian (switch None: as the CPHF), or None'''
    if switch is False:
        return None
    if switch is None:
        return _cphf_cosx(mf)
    with lib.temporary_env(sys.modules[__name__], CPHF_COSX=True):
        return _cphf_cosx(mf)


def _cphf_cosx(mf):
    '''the COSX object for the CPHF K, or None'''
    use = CPHF_COSX
    st = getattr(mf, '_cosx', None)
    if use is None:
        # auto: when the SCF ran COSX and the basis is large enough for COSX to beat jk_multi on many
        # densities (cc-pVDZ c16 / caffeine: exact faster; def2-TZVP c16: COSX 1.2 against 2.1 s per density)
        use = bool(st) and mf.mol.nao_nr() >= CPHF_COSX_NAO_PER_ATOM * mf.mol.natm
    if not use:
        return None
    from socutils.scf import cosx
    if not cosx.available(mf.mol):
        return None
    if st and CPHF_COSX_GRID in st.get('k', {}):
        return st['k'][CPHF_COSX_GRID]
    k = cosx.COSX(mf.mol, cosx.make_grids(mf.mol, **cosx.GRIDX[CPHF_COSX_GRID]))
    k.verbose = mf.verbose
    k.stdout = mf.stdout
    if st is not None:
        st.setdefault('k', {})[CPHF_COSX_GRID] = k
    return k.build()


def _multi_jk_env(mf):
    '''mf.get_jk / get_j through socutils.scf.jk_multi (all CPHF densities in
    one direct pass); with density fitting J from the stored three-index
    tensor (_df_get_j) and exact K from jk_multi; range-separated calls and
    fitted K unchanged.  K by COSX (all densities sharing the potential
    integrals) when the SCF ran COSX (CPHF_COSX).'''
    from socutils.scf import jk_multi
    mol = mf.mol
    dfm = _df_mode(mf)
    multi_k = jk_multi.available(mol)
    cx = _cphf_cosx(mf)
    if dfm is None and not multi_k and cx is None:
        return {}
    orig_jk = mf.get_jk

    def get_jk(mol_=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if (mol_ is not None and mol_ is not mol) or omega is not None or dm is None:
            return orig_jk(mol_, dm, hermi, with_j, with_k, omega)
        dm = numpy.asarray(dm)
        if numpy.iscomplexobj(dm) or (dfm == 'jk' and with_k):
            return orig_jk(mol_, dm, hermi, with_j, with_k, omega)
        max_memory = max(1000, mf.max_memory - lib.current_memory()[0])
        if cx is not None and with_k:
            if dfm is None:
                vj = jk_multi.get_jk(mol, dm, True, False, hermi=hermi, max_memory=max_memory)[0] if with_j else None
            else:
                vj = _df_get_j(mf.with_df, dm) if with_j else None
            from socutils.scf import cosx
            fl = cosx.MULTI_FLOAT
            cosx.set_multi_float(CPHF_COSX_FLOAT)
            try:
                vk = cx.get_k(dm, hermi=hermi)
            finally:
                cosx.set_multi_float(fl)
            return vj, vk
        if dfm is None:
            return jk_multi.get_jk(mol, dm, with_j, with_k, hermi=hermi, max_memory=max_memory)
        vj = _df_get_j(mf.with_df, dm) if with_j else None
        vk = None
        if with_k:
            if multi_k:
                vk = jk_multi.get_jk(mol, dm, False, True, hermi=hermi, max_memory=max_memory)[1]
            else:
                vk = orig_jk(mol_, dm, hermi, False, True, omega)[1]
        return vj, vk

    def get_j(mol_=None, dm=None, hermi=1, omega=None):
        return get_jk(mol_, dm, hermi, True, False, omega)[0]

    def get_k(mol_=None, dm=None, hermi=1, omega=None):
        return get_jk(mol_, dm, hermi, False, True, omega)[1]
    return dict(get_jk=get_jk, get_j=get_j, get_k=get_k)


def _df_get_j(with_df, dm):
    '''Density-fitted J of many densities from the stored three-index tensor
    (with_df._cderi, built by the SCF): rho = D L^T, J = rho L, two GEMMs per
    auxiliary block over all densities (pyscf's DF get_j for a pure functional
    evaluates the 3c integrals again on every call).  The symmetric part of
    each D (all J sees).'''
    dm = numpy.asarray(dm)
    shape = dm.shape
    nao = shape[-1]
    dms = dm.reshape(-1, nao, nao)
    dmtril = lib.pack_tril(dms + dms.transpose(0, 2, 1))
    diag = numpy.arange(nao) * (numpy.arange(nao) + 3) // 2      # diagonal of the packed triangle
    dmtril[:, diag] *= .5
    vj = numpy.zeros_like(dmtril)
    if with_df._cderi is None:
        with_df.build()
    for eri1 in with_df.loop():
        rho = lib.dot(dmtril, eri1.T)
        vj += lib.dot(rho, eri1)
    return lib.unpack_tril(vj, 1).reshape(shape)


KRYLOV_FAST_QR = True   # _krylov: Cholesky QR twice (GEMMs) instead of pyscf's Python-loop Gram-Schmidt


KRYLOV_QR_BLOCKED = True   # _qr_fast: blocked Gram-Schmidt (also for nearly dependent vectors) rather than CholQR2
QR_BLOCK = 32


def _qr_blocked(X, lindep=1e-14):
    '''pyscf.lib.linalg_helper._qr's modified Gram-Schmidt result (the vectors in order, one dropped when its
    squared residual is <= lindep) for _krylov, which only uses the orthonormal rows and the residual norms
    (the diagonal of r): blocks of QR_BLOCK vectors orthogonalised against the accepted ones by two GEMM
    passes (classical Gram-Schmidt twice), within the block one vector after the other against the
    block's accepted vectors.  Returns (qs, diag(residual norms)).'''
    n, dim = X.shape
    Q = numpy.empty((n, dim), dtype=X.dtype)
    norms = numpy.empty(n)
    nv = 0
    for b0 in range(0, n, QR_BLOCK):
        Xb = numpy.array(X[b0:b0 + QR_BLOCK], copy=True)
        if nv:
            Qa = Q[:nv]
            for _ in range(2):
                Xb -= lib.dot(lib.dot(Xb, Qa.conj().T), Qa)
        nb0 = nv
        for x in Xb:
            if nv > nb0:
                Qb = Q[nb0:nv]
                for _ in range(2):
                    x -= numpy.dot(numpy.dot(Qb.conj(), x), Qb)
            ip = numpy.dot(x.conj(), x).real
            if ip > lindep:
                nrm = numpy.sqrt(ip)
                Q[nv] = x / nrm
                norms[nv] = nrm
                nv += 1
    return Q[:nv], numpy.diag(norms[:nv])


def _qr_fast(xs, dot, lindep=1e-14):
    '''pyscf.lib.linalg_helper._qr (xs = r^T qs, qs orthonormal rows, r upper triangular with the
    Gram-Schmidt residual norms on its diagonal) by Cholesky QR applied twice (CholQR2): all GEMMs
    over the stacked vectors, instead of pyscf's modified Gram-Schmidt with a Python loop over the
    vector pairs (half of a large CP-SCF).  Exact-arithmetic identical; pyscf's _qr is used when a
    vector is (nearly) linearly dependent, the only case it drops vectors.'''
    from pyscf.lib import linalg_helper
    X = numpy.asarray(xs)
    if not KRYLOV_FAST_QR or X.ndim != 2 or X.shape[0] < 2:
        return linalg_helper._qr(xs, dot, lindep)
    if KRYLOV_QR_BLOCKED:
        return _qr_blocked(X, lindep)
    # rows scaled to unit norm first: CholQR's accuracy then depends on the vectors' (near) linear
    # dependence only, not on their very different lengths (converging and fresh directions)
    nrm = numpy.sqrt(numpy.einsum('ij,ij->i', X.conj(), X).real)
    if nrm.min() ** 2 <= lindep:
        return linalg_helper._qr(xs, dot, lindep)
    X = X / nrm[:, None]
    try:
        G = lib.dot(X, X.conj().T)
        L1 = scipy.linalg.cholesky(G, lower=True)
        Q1 = scipy.linalg.solve_triangular(L1, X, lower=True, check_finite=False)
        G = lib.dot(Q1, Q1.conj().T)
        L2 = scipy.linalg.cholesky(G, lower=True)
        Q = scipy.linalg.solve_triangular(L2, Q1, lower=True, check_finite=False)
    except (scipy.linalg.LinAlgError, ValueError):
        return linalg_helper._qr(xs, dot, lindep)
    rn = lib.dot(L1, L2)                                  # X / nrm = rn Q
    dn = rn.diagonal().real                               # relative Gram-Schmidt residuals
    if not numpy.all(numpy.isfinite(dn)) or dn.min() ** 2 <= 1e-10 or ((nrm * dn) ** 2).min() <= lindep:
        return linalg_helper._qr(xs, dot, lindep)
    rT = rn * nrm[:, None]                                # xs = r^T qs, r^T = diag(nrm) rn
    return Q, rT.conj().T


def _krylov(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=numpy.dot, lindep=None, callback=None,
            hermi=False, max_memory=None, verbose=logger.WARN):
    '''pyscf.lib.krylov (Pople's method for (1 + a) x = b, all right-hand sides
    together, converged directions dropped) with the projections onto the trial
    space, the subspace matrix and the solution as GEMMs over the stacked trial
    vectors; pyscf loops over the vectors in Python with one dot product each,
    which dominates for the 3 natm right-hand sides of a Hessian.  Same
    iterations and result.'''
    from pyscf.lib import linalg_helper
    if lindep is None:
        lindep = linalg_helper.DSOLVE_LINDEP
    if isinstance(aop, numpy.ndarray) and aop.ndim == 2:
        return numpy.linalg.solve(aop + numpy.eye(aop.shape[0]), b)
    log = logger.new_logger(verbose=verbose)
    b = numpy.asarray(b)
    if x0 is not None:
        b = b - (x0 + aop(x0))
    b1d = b.ndim == 1
    x1 = b.reshape(1, -1) if b1d else b
    nroots, ndim = x1.shape
    x1, rmat = _qr_fast(x1, dot)
    x1 *= rmat.diagonal()[:, None]
    innerprod = list(rmat.diagonal().real ** 2)
    if not innerprod or max(innerprod) < lindep or max(innerprod) < tol ** 2:
        return numpy.zeros_like(b) if x0 is None else x0
    xs = []
    ax = []
    XS = numpy.empty((0, ndim), dtype=x1.dtype)
    max_cycle = min(max_cycle, ndim)
    for cycle in range(max_cycle):
        axt = aop(x1).reshape(-1, ndim)
        xs.append(x1)
        ax.append(axt)
        XS = numpy.vstack([XS, x1])
        if callable(callback):
            callback(cycle, list(XS), list(numpy.vstack(ax)))
        # x1 = axt - sum_i <axt, xs_i> / <xs_i, xs_i> xs_i (classical Gram-Schmidt)
        w = lib.dot(axt, XS.conj().T) / numpy.asarray(innerprod)[None, :]
        x1 = axt - lib.dot(w, XS)
        axt = None
        x1, rmat = _qr_fast(x1, dot)
        x1 *= rmat.diagonal()[:, None]
        innerprod1 = rmat.diagonal().real ** 2
        max_innerprod = max(innerprod1, default=0.)
        log.debug('krylov cycle %d  r = %g', cycle, max_innerprod ** .5)
        if max_innerprod < lindep or max_innerprod < tol ** 2:
            break
        mask = (innerprod1 > lindep) & (innerprod1 > tol ** 2)
        x1 = x1[mask]
        innerprod.extend(innerprod1[mask])
    else:
        raise RuntimeError("Krylov solver failed to converge.")
    AX = numpy.vstack(ax)
    h = lib.dot(XS.conj(), AX.T)
    if hermi:
        h = numpy.tril(h) + numpy.tril(h, -1).conj().T
    h[numpy.diag_indices_from(h)] += innerprod
    g = lib.dot(XS.conj(), numpy.ascontiguousarray(b.reshape(-1, ndim).T))
    c = numpy.linalg.solve(h, g)
    x = lib.dot(c.T, XS)
    if b1d:
        x = x[0]
    if x0 is not None:
        x += x0
    return x


XC_AO_CUTOFF = 1e-13


def _xc_response_mo(ni, mol, grids, xctype, fxc, P, Q, max_memory):
    '''V[dm1] Q of the RKS XC kernel for dm1 = P Q^T + Q P^T, P (nset, nao, nocc),
    Q (nao, nocc): with phi dm1 = (phi P) Q^T + (phi Q) P^T and only V Q
    needed, the GEMMs run over nocc instead of nao (LDA 2, GGA 8 g nao nocc
    per density instead of pyscf's 2 g nao^2).  Arrays are kept transposed,
    (n, g), for pyscf's C contractions.'''
    from pyscf.dft import numint
    nset, nao, nocc = P.shape
    vq = numpy.zeros((nset, nao, nocc))
    PT = P.transpose(0, 2, 1)
    ao_deriv = 0 if xctype == 'LDA' else 1
    # grid block size for the (4, nset, nocc, g) intermediates X and R
    ng_blk = max(numint.BLKSIZE, int(max_memory * .4e6 / (8 * 8 * nset * nocc)))
    blk_mem = ng_blk * 5 * nao * 8 / 1e6
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, blk_mem):
        ng = weight.size
        p0, p1 = p1, p1 + ng
        ao = ao.reshape(-1, ng, nao)
        idx = numpy.where(abs(ao).max(axis=(0, 1)) > XC_AO_CUTOFF)[0]
        if idx.size == 0:
            continue
        nl = idx.size
        aoT = numpy.ascontiguousarray(ao[:, :, idx].transpose(0, 2, 1))     # (ncomp, nl, g)
        Ql = numpy.ascontiguousarray(Q[idx])
        PlT = numpy.ascontiguousarray(PT[:, :, idx])                         # (nset, nocc, nl)
        qT0 = Ql.T.dot(aoT[0])                                               # (nocc, g)
        X0T = PlT.reshape(nset * nocc, nl).dot(aoT[0]).reshape(nset, nocc, ng)
        if xctype == 'LDA':
            wf = weight * fxc[0, 0, p0:p1]
            for m in range(nset):
                u = 2 * numint._contract_rho(X0T[m].T, qT0.T) * wf
                vq[m, idx] += aoT[0].dot((qT0 * u).T)
            continue
        # GGA: X_k = P^T d_k phi for all densities (one GEMM over the AOs),
        # rho1 and R_k elementwise over nocc, V Q = sum_k d_k phi^T R_k (one
        # GEMM over the grid)
        wf = fxc[:, :, p0:p1] * weight
        qT = numpy.empty((4, nocc, ng))
        XT = numpy.empty((4, nset, nocc, ng))
        qT[0] = qT0
        XT[0] = X0T
        for k in range(1, 4):
            numpy.dot(Ql.T, aoT[k], out=qT[k])
            numpy.dot(PlT.reshape(nset * nocc, nl), aoT[k], out=XT[k].reshape(nset * nocc, ng))
        R = numpy.empty((4, nset, nocc, ng))
        for m in range(nset):
            rho1 = numpy.empty((4, ng))
            rho1[0] = numint._contract_rho(XT[0, m].T, qT[0].T)
            for j in range(1, 4):
                rho1[j] = (numint._contract_rho(XT[j, m].T, qT[0].T)
                           + numint._contract_rho(XT[0, m].T, qT[j].T))
            rho1 *= 2
            u = numpy.einsum('yg,xyg->xg', rho1, wf)
            R[0, m] = numint._scale_ao(qT.transpose(0, 2, 1), u).T
            for j in range(1, 4):
                numpy.multiply(qT[0], u[j], out=R[j, m])
        v = sum(aoT[k].dot(R[k].reshape(nset * nocc, ng).T) for k in range(4))   # (nl, nset*nocc)
        vq[:, idx] += v.reshape(nl, nset, nocc).transpose(1, 0, 2)
    return vq


def gen_vind(mf, mo_coeff, mo_occ):
    '''pyscf.hessian.rhf.gen_vind, for RKS (LDA / GGA, no range separation /
    NLC) with the XC response of _xc_response_mo; otherwise pyscf's.'''
    from pyscf.scf import hf
    mol = mf.mol
    if not isinstance(mf, hf.KohnShamDFT) or mol.cart:
        return rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if xctype not in ('LDA', 'GGA') or omega != 0 or mf.do_nlc():
        return rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = mocc.shape[1]
    if xctype == 'GGA' and 8 * nocc > nao:
        # GGA in the MO form costs 8 g nao nocc per density against pyscf's
        # 2 g nao^2 (which runs at the GEMM bound): only for nao > 8 nocc
        return rhf_hess.gen_vind(mf, mo_coeff, mo_occ)
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 0)
    max_memory = max(2000, mf.max_memory * .8 - lib.current_memory()[0])
    # K C_occ directly from COSX (CPHF_COSX_OCC): A C_occ per grid point shared by all densities
    cx = _cphf_cosx(mf) if hybrid and CPHF_COSX_OCC else None

    def fx(mo1):
        # the MO <-> AO transforms as single GEMMs over all densities (MKL through lib.dot / lib.einsum;
        # numpy.einsum without optimize is one thread, numpy.matmul runs numpy's own BLAS)
        mo1 = mo1.reshape(-1, nmo, nocc)
        nset = mo1.shape[0]
        P = numpy.ascontiguousarray(lib.einsum('pq,mqi->mpi', mo_coeff, mo1 * 2))   # *2: double occupancy
        dm1 = lib.dot(P.reshape(-1, nocc), mocc.T).reshape(nset, nao, nao)
        dm1 = lib.hermi_sum(dm1, axes=(0, 2, 1), inplace=True)
        vq = _xc_response_mo(ni, mol, mf.grids, xctype, fxc, P, mocc, max_memory)
        if cx is not None:
            from socutils.scf import cosx
            vj = mf.get_j(mol, dm1, hermi=1)
            fl = cosx.MULTI_FLOAT
            cosx.set_multi_float(CPHF_COSX_FLOAT)
            try:
                vkc = cx.get_k_occ(dm1, mocc)
            finally:
                cosx.set_multi_float(fl)
            vq += numpy.matmul(vj, mocc) - .5 * hyb * vkc
        else:
            if hybrid:
                vj, vk = mf.get_jk(mol, dm1, hermi=1)
                v = vj - .5 * hyb * vk
            else:
                v = mf.get_j(mol, dm1, hermi=1)
            vq += lib.dot(v.reshape(-1, nao), mocc).reshape(nset, nao, nocc)
        return lib.einsum('pq,mpi->mqi', mo_coeff, vq)
    return fx


class _FxcBatched:
    '''numint.nr_rks_fxc for many symmetric first-order densities (CPHF): the
    same flops as pyscf's per-density sparse loops, but per grid block

        c   = D phi^T                       one GEMM, all densities stacked
        aow = f(rho1[c], fxc) phi           one fused OpenMP C pass over c
        V  += aow phi                       one GEMM

    restricted to the AOs that are significant on the block (grids.non0tab), so that the cost follows the local AO count of every block, not
    nao.  The fused pass (lib/dft/fxc_multi.c) evaluates rho1, contracts it
    with fxc and overwrites c with aow chunk by chunk while the chunk is in
    cache.  The GEMMs go through pyscf's lib.dot (MKL): numpy's own OpenBLAS
    runs the small-K product D phi^T at a third of the speed.  LDA / GGA with
    hermi=1 and a cached fxc; other calls go to pyscf.'''

    def __init__(self, ni, max_memory=4000):
        self.ni = ni
        self.orig = ni.nr_rks_fxc
        self.max_memory = max_memory

    def __call__(self, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                 rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        ni = self.ni
        xctype = ni._xc_type(xc_code)
        libdft = _load_socdft()
        dms_a = numpy.asarray(dms)
        if (hermi != 1 or fxc is None or xctype not in ('LDA', 'GGA') or libdft is None
                or not hasattr(libdft, 'SOCDFTfxc_fused') or numpy.iscomplexobj(dms_a) or mol.cart):
            return self.orig(mol, grids, xc_code, dm0, dms, relativity, hermi, rho0, vxc, fxc,
                             max_memory, verbose)
        nao = dms_a.shape[-1]
        d = numpy.ascontiguousarray(dms_a.reshape(-1, nao, nao))
        n = d.shape[0]
        ncomp = 1 if xctype == 'LDA' else 4
        fxc = numpy.asarray(fxc).reshape(ncomp, ncomp, -1)
        from pyscf.dft.numint import BLKSIZE
        ao_loc = mol.ao_loc_nr()
        nf_shell = ao_loc[1:] - ao_loc[:-1]
        non0tab = getattr(grids, 'non0tab', None)
        vmat = numpy.zeros((n, nao, nao))
        nsub = max(1, FXC_SUBBLOCK // BLKSIZE) * BLKSIZE              # grid points per sub-block
        # pyscf blocks (AO evaluation) of ~20% of max_memory; c / aow per sub-block
        ng_blk = max(nsub, int(self.max_memory * .2e6 / (8 * (ncomp + 1) * nao)) // nsub * nsub)
        blk_mem = ng_blk * (ncomp + 1) * nao * 8 / 1e6
        cbuf = numpy.empty(n * nao * nsub)
        aos_buf = numpy.empty(ncomp * nao * nsub)
        fws_buf = numpy.empty(ncomp * ncomp * nsub)
        p1 = 0
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ncomp // 4, blk_mem):
            ng = weight.size
            p0, p1 = p1, p1 + ng
            aoT, cs = _ao_stack(ao)                                    # (ncomp, nao, g)
            fwb = numpy.ascontiguousarray(fxc[:, :, p0:p1] * weight)
            if xctype != 'LDA':
                fwb[0] *= .5                                           # *.5 for V + V^T
            for s0 in range(0, ng, nsub):
                s1 = min(ng, s0 + nsub)
                gs = s1 - s0
                # AOs of the shells significant on this sub-block
                idx = None
                if non0tab is not None:
                    shm = non0tab[(p0 + s0) // BLKSIZE:(p0 + s1 + BLKSIZE - 1) // BLKSIZE].any(axis=0)
                    aom = numpy.repeat(shm, nf_shell)
                    nl = int(aom.sum())
                    if nl == 0:
                        continue
                    if nl < nao:
                        idx = numpy.nonzero(aom)[0].astype(numpy.int32)
                if idx is None:
                    nl = nao
                aos = aos_buf[:ncomp * nl * gs].reshape(ncomp, nl, gs)
                libdft.SOCDFTgather_ao(aos.ctypes, aoT.ctypes, None if idx is None else idx.ctypes,
                                       ctypes.c_int(ncomp), ctypes.c_int(nao), ctypes.c_int(nl), ctypes.c_int(ng),
                                       ctypes.c_int(s0), ctypes.c_int(gs), ctypes.c_long(cs))
                if idx is None:
                    dl = d.reshape(n * nao, nao)
                else:
                    dl = numpy.empty((n * nl, nl))
                    libdft.SOCDFTgather_dm(dl.ctypes, d.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(n),
                                           ctypes.c_int(nao))
                c = cbuf[:n * nl * gs].reshape(n * nl, gs)
                lib.dot(dl, aos[0], c=c)                               # D phi^T
                fws = fws_buf[:ncomp * ncomp * gs].reshape(ncomp, ncomp, gs)
                libdft.SOCDFTslice_grid(fws.ctypes, fwb.ctypes, ctypes.c_long(ncomp * ncomp), ctypes.c_int(ng),
                                        ctypes.c_int(s0), ctypes.c_int(gs))
                libdft.SOCDFTfxc_fused(c.ctypes, aos.ctypes, fws.ctypes, ctypes.c_int(n), ctypes.c_int(ncomp),
                                       ctypes.c_int(nl), ctypes.c_int(gs))  # c <- aow
                if idx is None:
                    lib.dot(c, aos[0].T, beta=1, c=vmat.reshape(n * nao, nao))
                else:
                    vl = lib.dot(c, aos[0].T)
                    libdft.SOCDFTscatter_add_dm(vmat.ctypes, vl.ctypes, idx.ctypes, ctypes.c_int(nl),
                                                ctypes.c_int(n), ctypes.c_int(nao))
        if xctype != 'LDA':
            # V + V^T in place (OpenMP C; numpy's vmat + vmat.transpose allocates and reads strided)
            vmat = lib.hermi_sum(vmat, axes=(0, 2, 1), inplace=True)
        return vmat.reshape(dms_a.shape)


# grid points per sub-block of the batched XC response: small enough that the
# sub-block sees only the AOs near it (the GEMMs scale with their square), large
# enough for efficient GEMMs (the sub-matrix gather costs ~1 / (2 FXC_SUBBLOCK)
# of them)
FXC_SUBBLOCK = 448
XCD1_LOCAL_ATOMS = True     # xc_deriv1_batched: per sub-block only the atoms with significant AOs on it


# the default ('auto') grid of the CP-SCF XC response: socutils.dft.grids.hessian_grids arguments
CPSCF_GRID_ARGS = dict(angular_grid=2, intacc=3.467)


class _MultiJKCPHF:
    '''solve_mo1 with the response J / K of jk_multi (density fitting:
    _df_get_j), the batched (or, for nao > 8 nocc, MO-basis) XC response, and
    the Krylov solver as GEMMs (_krylov).

    Options:
        sum_rule : impose translational invariance on the final Hessian,
            H_AA = -sum_{B!=A} H_AB (socutils.dft.grids.translational_sum_rule);
            pyscf's RKS Hessian has no grid response, and on finite grids its
            error sits almost entirely in H_AA.  Off by default.
        cpscf_grids : the grid of the XC response inside the CP-SCF equations
            only; the explicit terms and the right-hand sides keep the SCF grid.
            'auto' (default): socutils.dft.grids.hessian_grids(mol,
            **CPSCF_GRID_ARGS), a coarse grid as Gaussian's CPHF (SG-1 there,
            the SCF on UltraFine); (H2O)_8 cc-pVDZ PBE0 on a level-4 SCF grid:
            the CP-SCF 189 -> 37 s, the frequencies within 0.25 cm-1.  A Grids
            object: that grid.  None: the SCF grid (the result equals pyscf's).'''
    sum_rule = False
    cpscf_grids = 'auto'
    auxbasis_response = 2   # the fitted-K (RI-JK) fallbacks to pyscf.df.hessian read it, as its DF Hessian
    _keys = {'sum_rule', 'cpscf_grids', '_cpscf_auto', 'auxbasis_response'}

    def kernel(self, *args, **kwargs):
        # the SCF's semi-direct integral store (scf.md_j) serves no Hessian build: its memory to the Hessian
        if getattr(self.base, '_jk_store', None) is not None:
            self.base._jk_store = None
        de = super().kernel(*args, **kwargs)
        if self.sum_rule:
            from socutils.dft.grids import translational_sum_rule
            self.de = de = translational_sum_rule(de)
        return de

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx=None, atmlst=None,
                  max_memory=4000, verbose=None):
        from pyscf.dft import numint
        env_xc = {}
        ni = getattr(self.base, '_numint', None)
        if isinstance(ni, numint.NumInt):
            env_xc = dict(nr_rks_fxc=_FxcBatched(ni, max_memory))
        env_grid = {}
        cg = self.cpscf_grids
        if isinstance(cg, str) and cg == 'auto':
            cg = None
            if hasattr(self.base, 'grids'):
                cg = getattr(self, '_cpscf_auto', None)
                if cg is None or cg.mol is not self.mol:
                    from socutils.dft.grids import hessian_grids
                    cg = self._cpscf_auto = hessian_grids(self.mol, **CPSCF_GRID_ARGS)
        if cg is not None and hasattr(self.base, 'grids'):
            if cg.coords is None:
                cg.build(with_non0tab=True)
            env_grid = dict(grids=cg)
        with lib.temporary_env(self.base, **_multi_jk_env(self.base), **env_grid), \
                lib.temporary_env(lib, krylov=_krylov):
            with lib.temporary_env(ni, **env_xc) if env_xc else _nullcontext():
                if fx is None:
                    fx = gen_vind(self.base, mo_coeff, mo_occ)
                if (isinstance(h1ao_or_chkfile, numpy.ndarray)
                        and _is_h1_occ(h1ao_or_chkfile, mo_coeff, mo_occ)):
                    return _solve_mo1_occ(self.base, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx, atmlst,
                                          max_memory, max_cycle=self.max_cycle, level_shift=self.level_shift)
                return super().solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx,
                                         atmlst, max_memory, verbose)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class RHFHessian(_MultiJKCPHF, rhf_hess.Hessian):
    hess_elec = hess_elec
    partial_hess_elec = rhf_partial_hess_elec
    make_h1 = rhf_make_h1


class RKSHessian(_MultiJKCPHF, rks_hess.Hessian):
    hess_elec = hess_elec
    partial_hess_elec = rks_partial_hess_elec
    make_h1 = rks_make_h1
