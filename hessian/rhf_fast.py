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
    if dfm is not None:
        # density-fitted J (exact derivatives of the DF energy), exact K below
        r = ri_j_deriv(mol, _df_auxmol(mf), dm0, with_vj1=False,
                       max_memory=max(1000, mf.max_memory - lib.current_memory()[0]))
        if r is None:
            from pyscf.df.hessian import rhf as df_rhf_hess
            return df_rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ, atmlst,
                                                 max_memory, verbose, with_k)
        ej = r[0]
        groups = [(0, 1)] if with_k else []
    else:
        # J and K from the same integral pass (groups 0 and 1)
        groups = [(1, 0), (0, 1)] if with_k else [(1, 0)]
    out = []
    if groups:
        # the combined Cartesian kernel up to g functions, else the
        # libcint-based driver
        out = hess2e_jk_cart(mol, dm0, groups)
        if out is None:
            terms = [] if ej is not None else [(dm0, 'ba', dm0, 'dc', .5, 0)]
            if with_k:
                terms.append((dm0, 'bc', dm0, 'da', .25, len(terms)))
            out = ghf_hess.deriv2_energies_direct(mol, terms)
    if ej is None:
        ej = out[0]
        ek = out[1] if with_k else numpy.zeros_like(ej)
    else:
        ek = out[0] if with_k else numpy.zeros_like(ej)
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


def _deriv2e_cart(mol, dm, groups, order, mode, tol, prim_tol):
    import ctypes
    libv = ghf_hess._load_socvhf()
    fname = 'SOCVHFhess2e_jk_cart' if order == 2 else 'SOCVHFgrad2e_jk_cart'
    if not libv or not hasattr(libv, fname):
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
    getattr(libv, fname)(
        out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mode), ctypes.c_int(ngroup),
        jfac.ctypes.data_as(ctypes.c_void_p), kfac.ctypes.data_as(ctypes.c_void_p),
        dmc.ctypes.data_as(ctypes.c_void_p), q.ctypes.data_as(ctypes.c_void_p),
        dcond.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(tol), ctypes.c_double(prim_tol),
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(natm), bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(molc.nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    return out


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
    e1 += numpy.einsum('xypq,pA,qB->ABxy', hab, S, S)
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
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, None, log)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao, None, None, max_memory, log)
    nao = mo_coeff.shape[0]
    mocc = mo_coeff[:, mo_occ > 0]
    nocc = mocc.shape[1]
    e_o = mo_energy[mo_occ > 0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    aoslices = mol.aoslice_by_atom()
    s1ao = numpy.zeros((natm, 3, nao, nao))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        s1ao[ia, :, p0:p1] += s1a[:, p0:p1]
        s1ao[ia, :, :, p0:p1] += s1a[:, p0:p1].transpose(0, 2, 1)
    mo1 = numpy.asarray(mo1).reshape(natm * 3 * nao, nocc)
    dm1 = lib.dot(mo1, mocc.T).reshape(natm * 3, nao * nao)
    dme1 = lib.dot(mo1, (mocc * e_o).T).reshape(natm * 3, nao * nao)
    h1 = numpy.asarray(h1ao).reshape(natm * 3, nao * nao)
    s1 = s1ao.reshape(natm * 3, nao * nao)
    s1oo = numpy.einsum('xpi,pj->xij', lib.dot(s1ao.reshape(-1, nao), mocc).reshape(natm * 3, nao, nocc),
                        mocc).reshape(natm * 3, nocc * nocc)
    me1 = numpy.asarray(mo_e1).reshape(natm * 3, nocc * nocc)
    # *2 for double occupancy, *2 for +c.c.
    r = 4 * lib.dot(h1, dm1.T) - 4 * lib.dot(s1, dme1.T) - 2 * lib.dot(s1oo, me1.T)
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
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ngrid = weight.size
        p0, p1 = p1, p1 + ngrid
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
                                    verbose, with_k=hybrid)
    de2 += ej - hyb * ek
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    t1 = log.timer_debug1('RKS partial hessian, 1e + 2e', *time0)
    de2 += xc_hess_explicit(hessobj, mo_coeff, mo_occ, max_memory)
    log.timer_debug1('RKS partial hessian, XC', *t1)
    log.timer('RKS partial hessian', *time0)
    return de2


def fock1_jk(mol, dm, with_k=True, tol=ghf_hess.HESS2E_TOL, max_memory=None):
    '''First-order VJ^{A,x}_ab = d/dR_{A,x} (ab|cd) D_cd and
    VK^{A,x}_ad = d/dR_{A,x} (ab|cd) D_bc of all atoms at a symmetric D, in one
    pass over the unique quartets (lib/vhf/fock1.c).  The atoms are batched so
    that nthreads x the per-thread buffers stay within half of max_memory (MB,
    default mol.max_memory); a batch skips the quartets touching none of its
    atoms.  Returns (natm, 3, nao, nao) arrays, or None if the library is
    unavailable.'''
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


def ri_j_deriv(mol, auxmol, dm, with_hess=True, with_vj1=True, tol=ghf_hess.HESS2E_TOL, max_memory=None):
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
    if the library is unavailable.'''
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
    rho = numpy.empty(naux)
    for k0, k1, _ in aux_ranges:
        p0, p1 = auxmol.ao_loc[k0], auxmol.ao_loc[k1]
        int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', shls_slice=(0, nbas, 0, nbas, k0, k1))
        rho[p0:p1] = lib.dot(dm_tril[None], int3c)[0]
    int3c = None
    solve = _gen_metric_solver(auxmol.intor('int2c2e', aosym='s1'))
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
    Y = numpy.zeros((natm * 3, naux))
    W = numpy.zeros((natm, 3, nao, nao)) if with_vj1 else None
    null = lib.c_null_ptr()
    libv.SOCVHFri_j_deriv(
        hess.ctypes.data_as(ctypes.c_void_p) if with_hess else null, Y.ctypes.data_as(ctypes.c_void_p),
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
    if with_vj1:
        vj1 = W + W.transpose(0, 1, 3, 2)
        W = None
        T = numpy.zeros((npair, natm * 3))
        for k0, k1, _ in aux_ranges:
            p0, p1 = auxmol.ao_loc[k0], auxmol.ao_loc[k1]
            int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', shls_slice=(0, nbas, 0, nbas, k0, k1))
            T += lib.dot(int3c, Z[p0:p1])
        int3c = None
        vj1 += lib.unpack_tril(numpy.ascontiguousarray(T.T)).reshape(natm, 3, nao, nao)
    return hess, vj1


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


def _fock1(hessobj, dm0, with_k):
    '''(vj1, vk1) of all atoms: fock1_jk, or with density-fitted J ri_j_deriv
    for J and fock1_jk for the exact K; None if a library is missing.'''
    mol = hessobj.mol
    mf = hessobj.base
    max_memory = max(1000, mf.max_memory - lib.current_memory()[0])
    if _df_mode(mf) is None:
        return fock1_jk(mol, dm0, with_k=with_k, max_memory=max_memory)
    r = ri_j_deriv(mol, _df_auxmol(mf), dm0, with_hess=False, max_memory=max_memory)
    if r is None:
        return None
    vk = None
    if with_k:
        v = fock1_jk(mol, dm0, with_k=True, max_memory=max_memory)
        if v is None:
            return None
        vk = v[1]
    return r[1], vk


def xc_deriv1_batched(hessobj, mo_coeff, mo_occ, max_memory=4000):
    """pyscf.hessian.rks._get_vxc_deriv1 (LDA / GGA): the first-order XC
    matrices d V_xc / dR_{A,x} of all atoms at fixed density.  The 3 natm
    first-order potentials f_xc d_A rho of a grid sub-block go through one
    batched aow pass (lib/dft/fxc_multi.c) and one GEMM, restricted to the AOs
    significant on the sub-block, instead of pyscf's three nao x nao GEMMs per
    atom and grid block; d_A rho for all atoms comes from one C pass
    (lib/dft/xc_hess.c).  The vxc term (v_ip) is pyscf's.  (natm, 3, nao, nao)."""
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
    if (xctype not in ('LDA', 'GGA') or libdft is None or not hasattr(libdft, 'SOCDFTrks_dR_rho')
            or mol.cart):
        return rks_hess._get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
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
    vmat = numpy.zeros((nd, nao, nao))                  # index 3 * atom + x
    nsub = max(1, FXC_SUBBLOCK // BLKSIZE) * BLKSIZE
    ncomp_ao = 10 if gga else 4
    # pyscf blocks: AO values, R and the weights within ~30% of max_memory
    per_point = 8 * (ncomp_ao * nao + 2 * 3 * nvar * natm + nvar * nao)
    ng_blk = max(nsub, int(max_memory * .3e6 / per_point) // nsub * nsub)
    blk_mem = ng_blk * (ncomp_ao + 1) * nao * 8 / 1e6
    aow_buf = numpy.empty(nd * nao * nsub)
    aos_buf = numpy.empty(nvar * nao * nsub)
    ws_buf = numpy.empty(nd * nvar * nsub)
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, blk_mem):
        ng = weight.size
        p0, p1 = p1, p1 + ng
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
            aos = aos_buf[:nvar * nl * gs].reshape(nvar, nl, gs)
            libdft.SOCDFTgather_ao(aos.ctypes, aoT.ctypes, None if idx is None else idx.ctypes, ctypes.c_int(nvar),
                                   ctypes.c_int(nao), ctypes.c_int(nl), ctypes.c_int(ng), ctypes.c_int(s0),
                                   ctypes.c_int(gs), ctypes.c_long(cs))
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
    v = _fock1(hessobj, dm0, hybrid)
    if v is None:
        if dfm is not None:
            from pyscf.df.hessian import rks as df_rks_hess
            return df_rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
        return rks_hess.make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    vj, vk = v
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    max_memory = max(2000, mf.max_memory * .9 - lib.current_memory()[0])
    h1ao = xc_deriv1_batched(hessobj, mo_coeff, mo_occ, max_memory)
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    for ia in range(mol.natm):
        h1ao[ia] += vj[ia] + hcore_deriv(ia)
        if hybrid:
            h1ao[ia] -= .5 * hyb * vk[ia]
    return h1ao


def _multi_jk_env(mf):
    '''mf.get_jk / get_j through socutils.scf.jk_multi (all CPHF densities in
    one direct pass); with density fitting J from the stored three-index
    tensor (_df_get_j) and exact K from jk_multi; range-separated calls and
    fitted K unchanged.'''
    from socutils.scf import jk_multi
    mol = mf.mol
    dfm = _df_mode(mf)
    multi_k = jk_multi.available(mol)
    if dfm is None and not multi_k:
        return {}
    orig_jk = mf.get_jk

    def get_jk(mol_=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if (mol_ is not None and mol_ is not mol) or omega is not None or dm is None:
            return orig_jk(mol_, dm, hermi, with_j, with_k, omega)
        dm = numpy.asarray(dm)
        if numpy.iscomplexobj(dm) or (dfm == 'jk' and with_k):
            return orig_jk(mol_, dm, hermi, with_j, with_k, omega)
        max_memory = max(1000, mf.max_memory - lib.current_memory()[0])
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
    x1, rmat = linalg_helper._qr(x1, dot)
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
        x1, rmat = linalg_helper._qr(x1, dot)
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
            vmat = vmat + vmat.transpose(0, 2, 1)
        return vmat.reshape(dms_a.shape)


# grid points per sub-block of the batched XC response: small enough that the
# sub-block sees only the AOs near it (the GEMMs scale with their square), large
# enough for efficient GEMMs (the sub-matrix gather costs ~1 / (2 FXC_SUBBLOCK)
# of them)
FXC_SUBBLOCK = 448


class _MultiJKCPHF:
    '''solve_mo1 with the response J / K of jk_multi (density fitting:
    _df_get_j), the batched (or, for nao > 8 nocc, MO-basis) XC response, and
    the Krylov solver as GEMMs (_krylov).

    Options (off by default, so that the result equals pyscf's):
        sum_rule : impose translational invariance on the final Hessian,
            H_AA = -sum_{B!=A} H_AB (socutils.dft.grids.translational_sum_rule);
            pyscf's RKS Hessian has no grid response, and on finite grids its
            error sits almost entirely in H_AA.
        cpscf_grids : a (coarser) Grids object for the XC response inside the
            CP-SCF equations only (socutils.dft.grids.hessian_grids(mol,
            angular_grid=2, intacc=3.467) is ~5x smaller than the SCF grid at
            <= 0.15 cm-1 in the frequencies); the explicit terms and the
            right-hand sides keep the SCF grid.'''
    sum_rule = False
    cpscf_grids = None
    _keys = {'sum_rule', 'cpscf_grids'}

    def kernel(self, *args, **kwargs):
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
        if self.cpscf_grids is not None and hasattr(self.base, 'grids'):
            if self.cpscf_grids.coords is None:
                self.cpscf_grids.build(with_non0tab=True)
            env_grid = dict(grids=self.cpscf_grids)
        with lib.temporary_env(self.base, **_multi_jk_env(self.base), **env_grid), \
                lib.temporary_env(lib, krylov=_krylov):
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
    hess_elec = hess_elec
    partial_hess_elec = rhf_partial_hess_elec
    make_h1 = rhf_make_h1


class RKSHessian(_MultiJKCPHF, rks_hess.Hessian):
    hess_elec = hess_elec
    partial_hess_elec = rks_partial_hess_elec
    make_h1 = rks_make_h1
