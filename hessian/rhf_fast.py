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
import numpy
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess
from socutils.hessian import ghf as ghf_hess


def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True):
    '''pyscf.hessian.rhf._partial_hess_ejk with ej, ek from the unique-quartet
    driver: E_J = 1/2 (ab|cd) D_ba D_dc, E_K = 1/4 (ab|cd) D_bc D_da.'''
    log = logger.new_logger(hessobj, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)
    atmlst = list(atmlst)
    if atmlst != list(range(mol.natm)) or ghf_hess.deriv2_energies_direct(mol, []) is None:
        return rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                          max_memory, verbose, with_k)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ > 0]) * 2
    natm = len(atmlst)
    # J and K from the same integral pass (groups 0 and 1): the combined
    # Cartesian kernel up to g functions, else the libcint-based driver
    out = hess2e_jk_cart(mol, dm0, [(1, 0), (0, 1)] if with_k else [(1, 0)])
    if out is None:
        terms = [(dm0, 'ba', dm0, 'dc', .5, 0)]
        if with_k:
            terms.append((dm0, 'bc', dm0, 'da', .25, 1))
        out = ghf_hess.deriv2_energies_direct(mol, terms)
    ej = out[0]
    ek = out[1] if with_k else numpy.zeros_like(ej)
    log.timer_debug1('two-electron second derivatives (unique quartets)', *time0)

    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    e1 = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia][2:]
        e1[i0, i0] -= numpy.einsum('xypq,pq->xy', s1aa[:, :, p0:p1], dme0[p0:p1]) * 2
        for j0, ja in enumerate(atmlst[:i0 + 1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0, j0] -= numpy.einsum('xypq,pq->xy', s1ab[:, :, p0:p1, q0:q1], dme0[p0:p1, q0:q1]) * 2
            e1[i0, j0] += numpy.einsum('xypq,pq->xy', hcore_deriv(ia, ja), dm0)
        for j0 in range(i0):
            e1[j0, i0] = e1[i0, j0].T
    log.timer('RHF partial hessian (unique quartets)', *time0)
    return e1, ej, ek


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
    import ctypes
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFhess2e_jk_cart'):
        return None
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
    dmc = numpy.ascontiguousarray(dmc)
    # integral estimate per shell pair, bounding every derivative type
    q = numpy.max([ghf_hess._q_cond(molc, intor)[k] for intor in ('int2e_ipip1', 'int2e_ipvip1', 'int2e_ip1ip2')
                   for k in (0, 1)], axis=0)
    q = numpy.ascontiguousarray(numpy.maximum(q, q.T))
    dcond = ghf_hess._dm_cond(molc, [dmc])
    jfac = numpy.asarray([g[0] for g in groups], dtype=numpy.double)
    kfac = numpy.asarray([g[1] for g in groups], dtype=numpy.double)
    natm = mol.natm
    out = numpy.zeros((ngroup, natm, natm, 3, 3)) if mode == 0 else numpy.zeros(ngroup)
    ao_loc = molc.ao_loc_nr().astype(numpy.int32)
    atm, bas, env = molc._atm, molc._bas, molc._env
    libv.SOCVHFhess2e_jk_cart(
        out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mode), ctypes.c_int(ngroup),
        jfac.ctypes.data_as(ctypes.c_void_p), kfac.ctypes.data_as(ctypes.c_void_p),
        dmc.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
        dcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol), ctypes.c_double(prim_tol),
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(molc.nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    return out


def rhf_partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                          atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                   max_memory, verbose, True)
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


# indices of the 2nd / 3rd Cartesian derivatives in pyscf's AO derivative stack
_D2 = numpy.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]])
_D3 = numpy.empty((3, 3, 3), dtype=int)
for _x in range(3):
    for _y in range(3):
        for _z in range(3):
            _D3[_x, _y, _z] = {(0, 0, 0): 10, (0, 0, 1): 11, (0, 0, 2): 12, (0, 1, 1): 13, (0, 1, 2): 14,
                               (0, 2, 2): 15, (1, 1, 1): 16, (1, 1, 2): 17, (1, 2, 2): 18,
                               (2, 2, 2): 19}[tuple(sorted((_x, _y, _z)))]


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
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ngrid = weight.size
        d = [numint._dot_ao_dm(mol, ao[k], dm0, mask, shls_slice, ao_loc) for k in range(4 if gga else 1)]
        rho = numpy.empty((nvar, ngrid))
        rho[0] = numint._contract_rho(ao[0], d[0])
        if gga:
            for j in range(3):
                rho[1 + j] = 2 * numint._contract_rho(ao[1 + j], d[0])
        if mgga:
            rho[4] = .5 * sum(numint._contract_rho(ao[1 + j], d[1 + j]) for j in range(3))
        vxc, fxc = ni.eval_xc_eff(mf.xc, rho if gga else rho[0], 2, xctype=xctype)[1:3]
        wf = weight * fxc.reshape(nvar, nvar, ngrid)
        if libdft is not None:
            # R, the on-atom sums and the M operands in one OpenMP pass (lib/dft/xc_hess.c)
            wv = numpy.ascontiguousarray(weight * vxc.reshape(nvar, ngrid))
            aoT = ao.transpose(0, 2, 1)
            if not aoT.flags.c_contiguous:
                aoT = numpy.ascontiguousarray(aoT)
            dT = numpy.ascontiguousarray(numpy.asarray(d).transpose(0, 2, 1))
            R = numpy.empty((3, nvar, natm, ngrid))
            ab = numpy.empty((6 if gga else 3, nao, ngrid))
            dg = numpy.zeros((3, 3, natm))
            libdft.SOCDFTrks_xc_hess_elem(
                R.ctypes, dg.ctypes, ab.ctypes, aoT.ctypes, dT.ctypes, wv.ctypes,
                ctypes.c_int(2 if mgga else int(gga)), ctypes.c_int(nvar), ctypes.c_int(ngrid),
                ctypes.c_int(nao), atom_of.ctypes, ctypes.c_int(natm))
            R *= -2
            diag += dg
            wR = numpy.einsum('klg,ylBg->ykBg', wf, R)
            term1 += numpy.tensordot(R, wR, axes=([1, 3], [1, 3]))
            a = [ab[x].T for x in range(3)]
            b = [ab[3 + x].T for x in range(3)] if gga else None
            wv = wv[:, :, None]
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
        with lib.temporary_env(rhf_hess, _partial_hess_ejk=_partial_hess_ejk):
            return rks_hess.partial_hess_elec(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                              max_memory, verbose)
    log = logger.new_logger(hessobj, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    de2, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, None, max_memory,
                                    verbose, with_k=hybrid)
    de2 += ej - hyb * ek
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    t1 = log.timer_debug1('RKS partial hessian, 1e + 2e', *time0)
    de2 += xc_hess_explicit(hessobj, mo_coeff, mo_occ, max_memory)
    log.timer_debug1('RKS partial hessian, XC', *t1)
    log.timer('RKS partial hessian', *time0)
    return de2


def fock1_jk(mol, dm, with_k=True, tol=ghf_hess.HESS2E_TOL):
    '''First-order VJ^{A,x}_ab = d/dR_{A,x} (ab|cd) D_cd and
    VK^{A,x}_ad = d/dR_{A,x} (ab|cd) D_bc of all atoms at a symmetric D, in one
    pass over the unique quartets (lib/vhf/fock1.c).  Returns (natm, 3, nao, nao)
    arrays, or None if the library is unavailable.'''
    import ctypes
    from pyscf.gto import moleintor
    from socutils.scf import jk_multi
    libv = ghf_hess._load_socvhf()
    if not libv or not hasattr(libv, 'SOCVHFfock1_jk') or mol.cart:
        return None
    dm = numpy.ascontiguousarray(dm, dtype=numpy.double)
    nao = dm.shape[0]
    vj = numpy.empty((mol.natm, 3, nao, nao))
    vk = numpy.empty((mol.natm, 3, nao, nao)) if with_k else numpy.empty(1)
    q = jk_multi._q_cond(mol)
    dcond = ghf_hess._dm_cond(mol, [dm])
    atm, bas, env = mol._atm, mol._bas, mol._env
    opt = moleintor.make_cintopt(atm, bas, env, 'int2e_ip1_sph')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    libv.SOCVHFfock1_jk(
        vj.ctypes.data_as(ctypes.c_void_p), vk.ctypes.data_as(ctypes.c_void_p),
        dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(int(with_k)),
        q.ctypes.data_as(ctypes.c_void_p), dcond.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(tol), opt,
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    return vj, (vk if with_k else None)


def rhf_make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''pyscf.hessian.rhf.make_h1 with the 2e part from fock1_jk.'''
    mol = hessobj.mol
    if atmlst is not None and list(atmlst) != list(range(mol.natm)):
        return rhf_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    v = fock1_jk(mol, dm0)
    if v is None:
        return rhf_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    vj, vk = v
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    return [vj[ia] - .5 * vk[ia] + hcore_deriv(ia) for ia in range(mol.natm)]


def rks_make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''pyscf.hessian.rks.make_h1 with the 2e part from fock1_jk (range-separated
    functionals and atom subsets fall back to pyscf).'''
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if omega != 0 or (atmlst is not None and list(atmlst) != list(range(mol.natm))):
        return rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    v = fock1_jk(mol, dm0, with_k=hybrid)
    if v is None:
        return rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    vj, vk = v
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    h1ao = rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    for ia in range(mol.natm):
        h1ao[ia] += vj[ia] + hcore_deriv(ia)
        if hybrid:
            h1ao[ia] -= .5 * hyb * vk[ia]
    return h1ao


def _multi_jk_env(mf):
    '''mf.get_jk / get_j through socutils.scf.jk_multi (all CPHF densities in
    one direct pass); range-separated calls and density fitting unchanged.'''
    from pyscf.df import df_jk
    from socutils.scf import jk_multi
    mol = mf.mol
    if isinstance(mf, df_jk._DFHF) or not jk_multi.available(mol):
        return {}
    orig_jk = mf.get_jk

    def get_jk(mol_=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if (mol_ is not None and mol_ is not mol) or omega is not None or dm is None:
            return orig_jk(mol_, dm, hermi, with_j, with_k, omega)
        dm = numpy.asarray(dm)
        if numpy.iscomplexobj(dm):
            return orig_jk(mol_, dm, hermi, with_j, with_k, omega)
        return jk_multi.get_jk(mol, dm, with_j, with_k, hermi=hermi)

    def get_j(mol_=None, dm=None, hermi=1, omega=None):
        return get_jk(mol_, dm, hermi, True, False, omega)[0]

    def get_k(mol_=None, dm=None, hermi=1, omega=None):
        return get_jk(mol_, dm, hermi, False, True, omega)[1]
    return dict(get_jk=get_jk, get_j=get_j, get_k=get_k)


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

    def fx(mo1):
        mo1 = mo1.reshape(-1, nmo, nocc)
        P = numpy.einsum('pq,mqi->mpi', mo_coeff, mo1 * 2)        # *2 for double occupancy
        dm1 = numpy.matmul(P, mocc.T)
        dm1 = dm1 + dm1.transpose(0, 2, 1)
        vq = _xc_response_mo(ni, mol, mf.grids, xctype, fxc, P, mocc, max_memory)
        if hybrid:
            vj, vk = mf.get_jk(mol, dm1, hermi=1)
            v = vj - .5 * hyb * vk
        else:
            v = mf.get_j(mol, dm1, hermi=1)
        vq += numpy.matmul(v, mocc)
        return numpy.einsum('pq,mpi->mqi', mo_coeff, vq)
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
        non0tab = getattr(grids, 'non0tab', None)
        vmat = numpy.zeros((n, nao, nao))
        # grid block: c / aow (n, nao, g) within ~20% of max_memory
        ng_blk = max(BLKSIZE_FXC, int(self.max_memory * .2e6 / (8 * n * nao)) // BLKSIZE_FXC * BLKSIZE_FXC)
        blk_mem = ng_blk * (ncomp + 1) * nao * 8 / 1e6
        # one buffer for c / aow, reused by every block (a fresh (n*nao, g)
        # array per block costs more in page faults than the GEMM itself)
        cbuf = numpy.empty(n * nao * ng_blk)
        p1 = 0
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ncomp // 4, blk_mem):
            ng = weight.size
            p0, p1 = p1, p1 + ng
            aoT = (ao if ao.ndim == 3 else ao[None]).transpose(0, 2, 1)  # (ncomp, nao, g)
            if not aoT.flags.c_contiguous:
                aoT = numpy.ascontiguousarray(aoT)
            # AOs of the shells significant anywhere on this block
            idx = None
            if non0tab is not None:
                sh = numpy.nonzero(non0tab[p0 // BLKSIZE:(p1 + BLKSIZE - 1) // BLKSIZE].any(axis=0))[0]
                if sh.size == 0:
                    continue
                nsig = (ao_loc[sh + 1] - ao_loc[sh]).sum()
                if nsig < FXC_SUBSET_FRACTION * nao:
                    idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i + 1]) for i in sh]).astype(numpy.int32)
            if idx is None:
                nl = nao
                dl = d.reshape(n * nao, nao)
            else:
                nl = idx.size
                aoT = numpy.ascontiguousarray(aoT[:, idx])           # whole rows
                dl = numpy.empty((n * nl, nl))
                libdft.SOCDFTgather_dm(dl.ctypes, d.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(n),
                                       ctypes.c_int(nao))
            if n * nl * ng > cbuf.size:
                cbuf = numpy.empty(n * nl * ng)
            c = cbuf[:n * nl * ng].reshape(n * nl, ng)
            lib.dot(dl, aoT[0], c=c)                                   # D phi^T
            fw = fxc[:, :, p0:p1] * weight
            if xctype != 'LDA':
                fw[0] *= .5                                            # *.5 for V + V^T
            fw = numpy.ascontiguousarray(fw)
            libdft.SOCDFTfxc_fused(c.ctypes, aoT.ctypes, fw.ctypes, ctypes.c_int(n), ctypes.c_int(ncomp),
                                   ctypes.c_int(nl), ctypes.c_int(ng))  # c <- aow
            if idx is None:
                lib.dot(c, aoT[0].T, beta=1, c=vmat.reshape(n * nao, nao))
            else:
                vl = lib.dot(c, aoT[0].T)
                libdft.SOCDFTscatter_add_dm(vmat.ctypes, vl.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(n),
                                            ctypes.c_int(nao))
        if xctype != 'LDA':
            vmat = vmat + vmat.transpose(0, 2, 1)
        return vmat.reshape(dms_a.shape)


BLKSIZE_FXC = 128
# block-local AO subsets only when they are this much smaller than nao (the
# gather / scatter is not free and the GEMMs scale with the square)
FXC_SUBSET_FRACTION = .95


class _MultiJKCPHF:
    '''solve_mo1 with the response J / K of jk_multi and the batched (or, for
    nao > 8 nocc, MO-basis) XC response'''

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx=None, atmlst=None,
                  max_memory=4000, verbose=None):
        from pyscf.dft import numint
        env_xc = {}
        ni = getattr(self.base, '_numint', None)
        if isinstance(ni, numint.NumInt):
            env_xc = dict(nr_rks_fxc=_FxcBatched(ni, max_memory))
        with lib.temporary_env(self.base, **_multi_jk_env(self.base)):
            with lib.temporary_env(ni, **env_xc) if env_xc else _nullcontext():
                if fx is None:
                    fx = gen_vind(self.base, mo_coeff, mo_occ)
                return super().solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile, fx,
                                         atmlst, max_memory, verbose)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class RHFHessian(_MultiJKCPHF, rhf_hess.Hessian):
    partial_hess_elec = rhf_partial_hess_elec
    make_h1 = rhf_make_h1


class RKSHessian(_MultiJKCPHF, rks_hess.Hessian):
    partial_hess_elec = rks_partial_hess_elec
    make_h1 = rks_make_h1
