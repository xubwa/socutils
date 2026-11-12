#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
The SCF Vxc of two-component (GKS / spinor) DFT in one pass over the grid
(lib/dft/vxc_2c.h SOCDFTvxc_fused_2c; the scalar analogue is
socutils.dft.vxc_fast._nr_rks_fused): per sub-block of vxc_fast.SUBBLOCK points
the AOs of the flagged shells once, the rows of every real density matrix, the
functional (libxc inside the OpenMP region), and the Vxc channels; LDA / GGA.

    collinear 'col'   D_a = Re D_aa, D_b = Re D_bb, spin-polarised libxc (pyscf's
                      NumInt2C.nr_vxc -> nr_uks).  A time-reversal-even (Kramers
                      closed-shell) density has D_a = D_b: the restricted pass of
                      vxc_fast on D_a + D_b (V_aa = V_bb), no spin channels at all.
    'ncol' (and 'mcol' LDA, which is the same functional)
                      D_n, D_mx, D_my, D_mz (noncollinear._rho_mag_dms), the locally
                      collinear chain rule in C; V_aa = V_n + V_z, V_bb = V_n - V_z,
                      V_ab = V_x - i V_y.

Everything else (mcol GGA / MGGA, meta-GGA, range separation, hermi != 1, many
densities, no non0tab, too little memory for the per-thread V) returns None and
the caller keeps its own path.  The AOs come from the SIMD evaluator of
lib/dft/gtoval.c for l <= SOCDFTgtoval_lmax(), else from pyscf's GTOval
(same kernel, slower AO values).

    from socutils.dft import vxc_2c
    vxc_2c.ENABLE = False        # back to the pyscf / noncollinear paths
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger

ENABLE = True
MCOL_GGA = True         # multi-collinear GGA fused too (the sphere average per sub-block)
KRAMERS_TOL = 1e-12     # max |Re D_aa - Re D_bb| / max |Re D_aa| below which 'col' runs the restricted pass
MEM_FRAC = .5           # of the free max_memory the per-thread V buffers may take


def _common(ni, mol, grids, xc_code):
    '''the shared ctypes arguments, or None when the fused pass does not apply'''
    from pyscf.dft import libxc
    from pyscf.gto import moleintor
    from socutils.hessian.rhf_fast import _load_socdft
    from socutils.dft import vxc_fast
    libdft = _load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTvxc_fused_2c') or mol.cart:
        return None
    if ni.libxc is not libxc or ni.rsh_and_hybrid_coeff(xc_code)[0] != 0:
        return None
    xctype = ni._xc_type(xc_code)
    if xctype not in ('LDA', 'GGA'):
        return None
    if grids.coords is None or grids.non0tab is None:
        grids.build(with_non0tab=True)
    xc = libxc._get_xc(xc_code, 1)
    if xc.nfunc == 0:
        return None
    lcg = moleintor.libcgto
    feval = ctypes.cast(lcg.GTOval_sph if xctype == 'LDA' else lcg.GTOval_sph_deriv1, ctypes.c_void_p)
    own = vxc_fast.OWN_GTOVAL and mol._bas[:, 1].max() <= libdft.SOCDFTgtoval_lmax()
    if own:
        feval = ctypes.cast(libdft.SOCDFTgtoval_sph if xctype == 'LDA' else libdft.SOCDFTgtoval_sph_deriv1,
                            ctypes.c_void_p)
    return dict(libdft=libdft, xctype=xctype, xc=xc, feval=feval, own=own)


def _run(c, mol, grids, dms, scheme, s_thrd, mcol=None):
    '''SOCDFTvxc_fused_2c on the real symmetric matrices dms (nd, nao, nao): (n, exc, V (nd, nao, nao));
    mcol (scheme 2) = (collinear_thrd, u, wu, xcmap, xclen)'''
    from pyscf.dft import libxc
    from socutils.dft import vxc_fast
    libdft, xc = c['libdft'], c['xc']
    nd, nao = dms.shape[:2]
    nthreads = lib.num_threads()
    dms = numpy.ascontiguousarray(dms, dtype=numpy.double)
    coords = numpy.ascontiguousarray(grids.coords.T)
    weights = numpy.ascontiguousarray(grids.weights)
    non0tab = numpy.ascontiguousarray(grids.non0tab, dtype=numpy.uint8)
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    ao_loc = mol.ao_loc_nr().astype(numpy.int32)
    vbuf = numpy.zeros((nthreads, nd, nao, nao))
    ex = numpy.zeros(nthreads)
    ne = numpy.zeros(nthreads)
    facs = (ctypes.c_double * max(1, xc.nfunc))(*xc.facs)
    if mcol is None:
        col_thrd, u, wu, xcmap, xclen = 1., numpy.zeros(3), numpy.zeros(1), numpy.zeros(1, numpy.int32), 0
    else:
        col_thrd, u, wu, xcmap, xclen = mcol
    libdft.SOCDFTvxc_fused_2c(
        vbuf.ctypes.data_as(ctypes.c_void_p), ex.ctypes.data_as(ctypes.c_void_p),
        ne.ctypes.data_as(ctypes.c_void_p), dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nd),
        ctypes.c_int(scheme), ctypes.c_double(s_thrd), ctypes.c_double(col_thrd),
        u.ctypes.data_as(ctypes.c_void_p), wu.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(wu.size),
        xcmap.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(xclen),
        coords.ctypes.data_as(ctypes.c_void_p), weights.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(weights.size), ctypes.c_int(vxc_fast.SUBBLOCK),
        ctypes.c_double(vxc_fast.AO_CUTOFF), ctypes.c_int(0 if c['xctype'] == 'LDA' else 1), c['feval'],
        ao_loc.ctypes.data_as(ctypes.c_void_p), atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas), env.ctypes.data_as(ctypes.c_void_p),
        ctypes.cast(libxc._itrf.LIBXC_eval_xc, ctypes.c_void_p), ctypes.c_int(xc.nfunc),
        ctypes.cast(xc.xc_arr, ctypes.c_void_p), facs, *vxc_fast._blas_ptrs())
    v = vbuf.sum(axis=0)
    vbuf = None
    if c['xctype'] != 'LDA':
        v = v + v.transpose(0, 2, 1)
    return ne.sum(), ex.sum(), v


def _fits(mol, nd, max_memory):
    need = lib.num_threads() * nd * mol.nao ** 2 * 8e-6
    return need < MEM_FRAC * max(0, max_memory - lib.current_memory()[0])


def nr_vxc(ni, mol, grids, xc_code, dm, hermi=1, max_memory=2000):
    '''(n, exc, vmat) of one Hermitian GHF density dm (2 nao, 2 nao) by the fused pass, or None'''
    if not ENABLE or hermi != 1:
        return None
    dm = numpy.asarray(dm)
    if dm.ndim != 2:
        return None
    scheme = ni.collinear[0]
    c = _common(ni, mol, grids, xc_code)
    if c is None or scheme not in ('c', 'n', 'm'):
        return None
    if scheme == 'm' and c['xctype'] != 'LDA' and (getattr(ni, 'mcol_backend', 'native') != 'native'
                                                  or not MCOL_GGA):
        return None
    t0 = (logger.process_clock(), logger.perf_counter())
    nao = mol.nao
    vmat = numpy.zeros((2 * nao, 2 * nao), dtype=numpy.result_type(dm.dtype, numpy.double) if scheme == 'c'
                       else numpy.complex128)
    if scheme == 'c':
        da = dm[:nao, :nao].real
        db = dm[nao:, nao:].real
        if abs(da - db).max() <= KRAMERS_TOL * max(abs(da).max(), 1e-300):
            # rho_a = rho_b: the spin-polarised functional of equal spins is the restricted one
            from socutils.dft import vxc_fast
            ni1 = ni._to_numint1c()
            if not _fits(mol, 1, max_memory) or not vxc_fast._fusable(ni1, xc_code, grids):
                return None
            d = numpy.ascontiguousarray(da + db)
            n, exc, v = vxc_fast._nr_rks_fused(c['libdft'], ni1, mol, grids, xc_code, d, d[None], c['xctype'])
            vmat[:nao, :nao] = vmat[nao:, nao:] = v
            logger.timer_debug1(ni, 'two-component Vxc (fused, restricted)', *t0)
            return n, exc, vmat
        if not _fits(mol, 2, max_memory):
            return None
        n, exc, v = _run(c, mol, grids, numpy.stack([da, db]), 0, 0.)
        vmat[:nao, :nao] = v[0]
        vmat[nao:, nao:] = v[1]
    else:
        if not _fits(mol, 4, max_memory):
            return None
        from socutils.dft.noncollinear import _rho_mag_dms
        ds = _rho_mag_dms(dm)[0::2]             # the real symmetric parts of D_n, D_mx, D_my, D_mz
        if scheme == 'm' and c['xctype'] == 'GGA':
            # multi-collinear GGA: the half-sphere average of lib/dft/mcol.c per sub-block
            from pyscf.dft import libxc
            from socutils.dft.noncollinear import _half_sphere
            u, wu = _half_sphere(ni.spin_samples)
            idx = libxc._libxc_to_xcfun_indices('GGA', 1, 2)
            xclen = len(idx)
            xcmap = numpy.asarray(numpy.arange(xclen)[idx], dtype=numpy.int32)
            thrd = ni.collinear_thrd
            mc = (1e300 if thrd is None else thrd, numpy.ascontiguousarray(u), numpy.ascontiguousarray(wu),
                  xcmap, xclen)
            n, exc, v = _run(c, mol, grids, numpy.stack(ds), 2, 0., mc)
        else:
            n, exc, v = _run(c, mol, grids, numpy.stack(ds), 1, ni.ncol_s_thrd)
        vr, vx, vy, vz = v
        vmat[:nao, :nao] = vr + vz
        vmat[nao:, nao:] = vr - vz
        vmat[:nao, nao:] = vx - 1j * vy
        vmat[nao:, :nao] = vx + 1j * vy
    logger.timer_debug1(ni, 'two-component Vxc (fused, %s)' % ni.collinear, *t0)
    return n, exc, vmat
