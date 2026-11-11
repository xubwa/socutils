#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
pyscf RHF / RKS nuclear gradients with the exact two-electron part and the XC
part as contracted energy derivatives instead of (3, nao, nao) derivative
matrices contracted with the density atom by atom:

* exact J: the McMurchie-Davidson J engine (lib/vhf/md_j.c,
  socutils.hessian.rhf_fast.md_j_grad): decontracted Cartesian primitives,
  the density in Hermite form once per primitive pair, no Cartesian integrals;
* exact K: the unique-quartet Rys kernel of the Hessian at derivative order 1
  (lib/vhf/hess2e_cart.c, socutils.hessian.rhf_fast.grad2e_jk_cart): one
  pass, density contracted on the fly, the fourth centre by translational
  invariance, K-only density screening;
* XC (fixed grid, LDA / GGA / MGGA): per grid block U = D aow, V = D phi
  (two GEMMs, pyscf: six) and one C pass (lib/dft/xc_hess.c
  SOCDFTrks_xc_grad) for sum_{mu in A} d_A phi_mu ... of every atom.

Drop-in classes (same results as pyscf's):

    from socutils.grad import rhf_fast
    g = rhf_fast.Gradients(mf).kernel()     # mf: scf.RHF / dft.RKS, also
                                            # density_fit() (RI-J; RI-J + exact K
                                            # with only_dfj=True)

fitted K (RI-JK, density_fit() without only_dfj): socutils.grad.rik_grad.
Range-separated functionals, NLC and grid response keep pyscf's code.
'''

import ctypes
import numpy
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from socutils.hessian import rhf_fast as hess_fast

# quartet cutoff of the exact 2e gradient (Schwarz x density bounds) and the primitive-quartet
# cutoff; the Hessians keep socutils.hessian.ghf.HESS2E_TOL (1e-13).  Against tol 1e-14 / no
# primitive screening (HF / cc-pVDZ): 1e-10 / 1e-11 max |dg| 3.0e-8 (pterin), 3.3e-8 (C16H34),
# 6.3e-8 (H2O)_16 Eh/Bohr (Gaussian's Opt threshold 4.5e-4), the 2e part 10 / 12 / 20 % faster
# than 1e-11 / 1e-13
GRAD2E_TOL = 1e-10
RIJ_GRAD_FAST = True    # RIJ (only_dfj): the DF-J gradient by hessian.rhf_fast.ri_j_grad, else pyscf's
RIK_GRAD_FAST = True    # RI-JK: the fitted K gradient by socutils.grad.rik_grad (and J as for RIJ), else pyscf's
GRAD2E_PRIM_TOL = 1e-11
GRAD_SUBBLOCK = 112     # grid points per sub-block of the fused XC gradient (10 AO components per AO)


USE_FUSED_XC = True     # LDA / GGA XC gradient by the fused one-pass kernel (_xc_grad_fused)


def xc_grad(ks_grad, mol, dm, max_memory=2000):
    '''XC contribution to the RKS gradient on the fixed grid (pyscf's
    get_vxc contracted as in grad_elec), (natm, 3); None if not applicable.'''
    mf = ks_grad.base
    ni = mf._numint
    libdft = hess_fast._load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTrks_xc_grad') or mol.cart:
        return None
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return None
    grids, _ = rks_grad._initialize_grids(ks_grad)
    if xctype != 'MGGA' and USE_FUSED_XC and hasattr(libdft, 'SOCDFTxc_grad_fused'):
        de = _xc_grad_fused(libdft, ni, mol, grids, mf.xc, dm, xctype)
        if de is not None:
            return de
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    D = numpy.ascontiguousarray(dm)
    ao_deriv = 1 if xctype == 'LDA' else 2
    nw = 1 if xctype == 'LDA' else 4
    itype = {'LDA': 0, 'GGA': 1, 'MGGA': 2}[xctype]
    out = numpy.zeros((3, nao))
    null = lib.c_null_ptr()
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        ng = weight.size
        if xctype == 'LDA':
            rho = make_rho(0, ao[0], mask, xctype)
        elif xctype == 'GGA':
            rho = make_rho(0, ao[:4], mask, xctype)
        else:
            rho = make_rho(0, ao[:10], mask, xctype)
        vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
        wv = numpy.ascontiguousarray(vxc.reshape(-1, ng) * weight)
        if xctype != 'LDA':
            wv[0] *= .5
        if xctype == 'MGGA':
            wv[4] *= .5
        aoT, cs = hess_fast._ao_stack(ao)                      # (ncomp, nao, g)
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
    de = numpy.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = -2 * out[:, p0:p1].sum(axis=1)
    return de


def _xc_grad_fused(libdft, ni, mol, grids, xc_code, dm, xctype):
    '''LDA / GGA XC gradient in one pass over the grid (lib/dft/vxc_rks.c
    SOCDFTxc_grad_fused: the AOs to the second derivatives, rho, the functional
    and the contraction per sub-block in cache); None if not applicable.'''
    from pyscf.gto import moleintor
    from socutils.dft import vxc_fast
    if grids.non0tab is None:
        grids.build(with_non0tab=True)
    if not vxc_fast._fusable(ni, xc_code, grids):
        return None
    xc = ni.libxc._get_xc(xc_code, 0)
    if xc.nfunc == 0:
        return numpy.zeros((mol.natm, 3))
    nao = mol.nao
    nthreads = lib.num_threads()
    ngrid = grids.weights.size
    coords = numpy.ascontiguousarray(grids.coords.T)
    weights = numpy.ascontiguousarray(grids.weights)
    non0tab = numpy.ascontiguousarray(grids.non0tab, dtype=numpy.uint8)
    lcg = moleintor.libcgto
    feval = ctypes.cast(lcg.GTOval_sph_deriv1 if xctype == 'LDA' else lcg.GTOval_sph_deriv2, ctypes.c_void_p)
    if (vxc_fast.OWN_GTOVAL and hasattr(libdft, 'SOCDFTgtoval_sph_deriv2')
            and mol._bas[:, 1].max() <= libdft.SOCDFTgtoval_lmax()):
        # the SIMD evaluator of lib/dft/gtoval.c
        feval = ctypes.cast(libdft.SOCDFTgtoval_sph_deriv1 if xctype == 'LDA' else libdft.SOCDFTgtoval_sph_deriv2,
                            ctypes.c_void_p)
    fxc = ctypes.cast(ni.libxc._itrf.LIBXC_eval_xc, ctypes.c_void_p)
    facs = (ctypes.c_double * xc.nfunc)(*xc.facs)
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    D = numpy.ascontiguousarray(dm)
    out = numpy.zeros((nthreads, 3, nao))
    libdft.SOCDFTxc_grad_fused(
        out.ctypes.data_as(ctypes.c_void_p), D.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p), weights.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrid), ctypes.c_int(GRAD_SUBBLOCK),
        ctypes.c_double(vxc_fast.AO_CUTOFF), ctypes.c_int(0 if xctype == 'LDA' else 1), feval,
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
        fxc, ctypes.c_int(xc.nfunc), ctypes.cast(xc.xc_arr, ctypes.c_void_p), facs, *vxc_fast._blas_ptrs())
    out = out.sum(axis=0)
    de = numpy.zeros((mol.natm, 3))
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] = -2 * out[:, p0:p1].sum(axis=1)
    return de


def _gen_grad_usable(mol):
    '''the generated first-derivative kernels run for this basis: every shell l <= 2 and <= 4
    contractions, or any basis when the Rys kernel can skip the generated quartets (the f shells'
    quartets to Rys, the rest generated, J and K in one pass)'''
    from socutils.hessian import ghf as ghf_hess
    libv = ghf_hess._load_socvhf()
    return (hess_fast.GRAD2E_GEN and bool(libv) and hasattr(libv, 'SOCVHFgen_grad2e')
            and (hasattr(libv, 'SOCVHFcart_set_skip_gen')
                 or (mol._bas[:, 1].max() <= 2 and mol._bas[:, 3].max() <= 4)))


def _fast_terms(grad, mol, dm):
    '''(natm, 3): the exact 2e (J unless density-fitted, K for hybrids) and XC
    parts of the gradient; None if this case keeps pyscf's code.'''
    mf = grad.base
    dfm = hess_fast._df_mode(mf)
    is_ks = isinstance(mf, hf.KohnShamDFT)
    if is_ks:
        ni = mf._numint
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        if omega != 0 or mf.do_nlc() or grad.grid_response:
            if grad.grid_response and getattr(mf, '_cosx', None):
                logger.warn(grad, 'grid_response: the whole gradient by pyscf, the exchange exact '
                            '(the COSX K gradient of the SCF is not used)')
            return None
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    else:
        hyb, hybrid = 1., True
    t0 = (logger.process_clock(), logger.perf_counter())
    jf = 1. if dfm is None else 0.
    kf = -hyb if hybrid else 0.
    de = numpy.zeros((mol.natm, 3))
    grad._rij_fast = False
    dfit = None
    if dfm == 'jk' and hybrid:
        if not (RIK_GRAD_FAST and grad.auxbasis_response):
            return None
        # RI-JK: the fitted K gradient (socutils.grad.rik_grad) and, from the same pass over the
        # factor, the fitted Coulomb coefficients of the RI-J gradient below
        from socutils.grad import rik_grad
        r = rik_grad.jk_grad_terms(mf, mol, dm, hyb, max_memory=max(1000, mf.max_memory - lib.current_memory()[0]))
        if r is None:
            return None
        dfit, dk = r
        de += dk
        kf = 0.
        t0 = logger.timer(grad, 'RI-K gradient', *t0)
    if dfm is not None and RIJ_GRAD_FAST and grad.auxbasis_response:
        # RIJ: the density-fitted J gradient (with its auxiliary-basis response) from the C pass
        cache = getattr(mf, '_rij_cache', None) or {}
        key = (mol._atm.tobytes(), mol._bas.tobytes(), mol._env.tobytes())
        cderi = cache.get('B') if cache.get('key') == key else None
        engine = cache.get('direct') if cache.get('key') == key else None
        r = hess_fast.ri_j_grad(mol, engine.auxmol if engine is not None else hess_fast._df_auxmol(mf), dm,
                                tol=GRAD2E_TOL, cderi=cderi, dfit=dfit, engine=engine,
                                max_memory=max(1000, mf.max_memory - lib.current_memory()[0]))
        if r is not None:
            de += r
            grad._rij_fast = True
            t0 = logger.timer(grad, 'RIJ gradient (three-centre C pass)', *t0)
    if dfm == 'jk' and not grad._rij_fast:
        return None
    if kf != 0 and getattr(mf, '_cosx', None):
        # K from COSX (socutils.scf.cosx.apply): the gradient of its energy on the SCF's grid
        from socutils.scf import cosx
        r = cosx.grad_k(mf, dm, accuracy=getattr(grad, 'cosx_grad_accuracy', None))
        if r is not None:
            de += r
            kf = 0.
            t0 = logger.timer(grad, 'COSX exchange gradient', *t0)
    if jf != 0 and kf != 0 and _gen_grad_usable(mol):
        # hybrids: J and K in one pass of the generated kernels (J costs little more
        # there than the separate MD J pass)
        g = hess_fast.grad2e_jk_cart(mol, dm, [(jf, kf)], tol=GRAD2E_TOL, prim_tol=GRAD2E_PRIM_TOL)
        if g is not None:
            de += g[0]
            jf = kf = 0.
            t0 = logger.timer(grad, 'exact 2e gradient (generated kernels, J + K)', *t0)
    if jf != 0:
        # exact J from the McMurchie-Davidson J engine
        r = hess_fast.md_j_grad(mol, dm)
        if r is not None:
            de += r[0]
            jf = 0.
            t0 = logger.timer(grad, 'exact J gradient (MD J engine)', *t0)
    if jf != 0 or kf != 0:
        g = hess_fast.grad2e_jk_cart(mol, dm, [(jf, kf)], tol=GRAD2E_TOL, prim_tol=GRAD2E_PRIM_TOL)
        if g is None:
            return None
        de += g[0]
        t0 = logger.timer(grad, 'exact 2e gradient (unique quartets)', *t0)
    if is_ks:
        max_memory = max(2000, grad.max_memory * .9 - lib.current_memory()[0])
        dx = xc_grad(grad, mol, dm, max_memory)
        if dx is None:
            return None
        de += dx
        logger.timer(grad, 'XC gradient (contracted)', *t0)
    return de


HCORE_FAST = True       # grad_elec: the hcore gradient of all atoms at once (kinetic + point-charge RI-J engine)


def _hcore_grad(mf_grad, mol, dm):
    '''sum_ij D_ij d h_ij / dR_A (natm, 3) for a symmetric dm: the kinetic part from one int1e_ipkin, the
    nuclear attraction -sum_C Z_C (ij|C) of unit point charges (gto.fakemol_for_charges) by the direct
    RI-J engine's three-centre gradient (bra and charge centres, socutils.scf.rij_direct) -- instead of
    pyscf's hcore_generator, which evaluates the (3, nao, nao) rinv derivatives once per atom (30 s of
    a 43 s gradient on a 176-atom peptide).  None (pyscf's path) with ECPs, finite nuclei, X2C.'''
    mf = mf_grad.base
    if (not HCORE_FAST or mol.has_ecp() or getattr(mol, '_pseudo', None) or mol.cart
            or getattr(mf, 'with_x2c', None) or any(mol.nucmod.values() if isinstance(mol.nucmod, dict)
                                                    else [mol.nucmod])):
        return None
    try:
        from socutils.scf import rij_direct
        fake = gto.fakemol_for_charges(mol.atom_coords())
        if not rij_direct.available(mol, fake):
            return None
        eng = rij_direct.RIJDirect(mol, fake)
        dm = numpy.asarray(dm, dtype=numpy.double)
        dm = (dm + dm.T) * .5
        gb, ga = eng.grad_3c(dm, -mol.atom_charges().astype(numpy.double))
    except Exception:
        return None
    ipk = mol.intor('int1e_ipkin', comp=3)
    de = gb + ga
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:, 2:]):
        de[ia] -= 2 * numpy.einsum('xij,ij->x', ipk[:, p0:p1], dm[p0:p1])
    return de


NET_FORCE_WARN = 2e-4   # Eh/bohr: the net force above which grad_elec warns


class _FastGradMixin:
    '''get_veff returns only what pyscf's grad_elec still needs as a matrix (the
    density-fitted J with its auxiliary-basis response, else zero); the exact
    2e and XC parts enter through extra_force as energy derivatives; the hcore
    part of all atoms at once (_hcore_grad).'''

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''pyscf.grad.rhf.grad_elec with the hcore term from _hcore_grad (pyscf's per-atom loop over
        hcore_generator otherwise)'''
        mf = self.base
        mol = self.mol
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if atmlst is not None and list(atmlst) != list(range(mol.natm)):
            return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        log = logger.Logger(self.stdout, self.verbose)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0 = self._tag_rdm1(dm0, mo_coeff, mo_occ)
        t0 = (logger.process_clock(), logger.perf_counter())
        dh = _hcore_grad(self, mol, dm0)
        if dh is None:
            return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        t0 = log.timer('hcore gradient (kinetic + point-charge engine)', *t0)
        s1 = self.get_ovlp(mol)
        vhf = self.get_veff(mol, dm0)
        log.timer('gradients of 2e part', *t0)
        dme0 = self.make_rdm1e(mo_energy, mo_coeff, mo_occ)
        aoslices = mol.aoslice_by_atom()
        atmlst = range(mol.natm)
        de = numpy.zeros((mol.natm, 3))
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            de[k] += dh[ia]
            de[k] += numpy.einsum('xij,ij->x', vhf[:, p0:p1], dm0[p0:p1]) * 2
            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2
            de[k] += self.extra_force(ia, locals())
        # the net force: zero for the exact derivative; fixed-grid XC / COSX terms break it, and it tracks
        # the largest error of the gradient (median over 67 molecules: net force 0.9 x the max error)
        net = float(abs(de.sum(axis=0)).max())
        self.net_force = net
        if getattr(mf, '_cosx', None):
            # the max force of this gradient, for the K-gradient grid of the next one (cosx.set_grad_accuracy
            # with auto=True: the grid follows the forces of an optimisation)
            try:
                mf._cosx['last_gmax'] = float(abs(de + self.grad_nuc(atmlst=atmlst)).max())
            except Exception:       # noqa: a gradient method without a plain nuclear term
                mf._cosx['last_gmax'] = None
        log.info('net force |sum_A g_A| %.2e Eh/bohr', net)
        if net > NET_FORCE_WARN:
            log.warn('net force %.1e Eh/bohr: the fixed-grid terms of the gradient (XC, COSX) are '
                     'not converged with their grids', net)
        if log.verbose >= logger.DEBUG:
            log.debug('gradients of electronic part')
            rhf_grad._write(log, mol, de, atmlst)
        return de

    def get_veff(self, mol=None, dm=None):
        mf = self.base
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = mf.make_rdm1()
        de = _fast_terms(self, mol, dm)
        self._de_fast = de
        if de is None:
            return super().get_veff(mol, dm)
        if hess_fast._df_mode(mf) is None:
            nao = dm.shape[-1]
            return lib.tag_array(numpy.zeros((3, nao, nao)), exc1_grid=None)
        if getattr(self, '_rij_fast', False):
            nao = dm.shape[-1]
            return lib.tag_array(numpy.zeros((3, nao, nao)), exc1_grid=None, aux=numpy.zeros((mol.natm, 3)))
        vj = self.get_j(mol, dm)
        if self.auxbasis_response:
            return lib.tag_array(numpy.asarray(vj), exc1_grid=None, aux=vj.aux.sum((0, 1)))
        return lib.tag_array(numpy.asarray(vj), exc1_grid=None)

    def extra_force(self, atom_id, envs):
        f = super().extra_force(atom_id, envs)
        de = getattr(self, '_de_fast', None)
        if de is not None:
            f = f + de[atom_id]
        return f


class RHFGradients(_FastGradMixin, rhf_grad.Gradients):
    pass


class RKSGradients(_FastGradMixin, rks_grad.Gradients):
    pass


def Gradients(mf):
    '''The fast gradient object for an RHF / RKS mean field (density-fitted
    ones included).'''
    dft = isinstance(mf, hf.KohnShamDFT)
    if hess_fast._df_mode(mf) is not None:
        from pyscf.df.grad import rhf as df_rhf_grad, rks as df_rks_grad
        base = df_rks_grad.Gradients if dft else df_rhf_grad.Gradients
        return type('DF' + ('RKS' if dft else 'RHF') + 'Gradients', (_FastGradMixin, base), {})(mf)
    return (RKSGradients if dft else RHFGradients)(mf)
