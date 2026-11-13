#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
The two-component (GKS / spinor) f_xc response of a closed-shell reference for
many trial densities at once (TDDFT / TDA, CPHF): LDA / GGA, the ncol kernel
(and mcol, the same kernel at m = 0).

At m = 0 the non-collinear kernel is isotropic,

    v_n = f_tt n1,     v_m_k = f_ss m1_k  (k = x, y, z),

f_tt the restricted kernel of the density and f_ss the spin kernel at zero
polarisation (noncollinear.eval_xc_ncol / eval_xc_mcol give exactly this block
structure there; checked on the cached kernel, else None and the caller keeps
its path).  A complex GHF trial density gives the channels D_n = D_aa + D_bb,
D_mx = D_ab + D_ba, D_my = i (D_ab - D_ba), D_mz = D_aa - D_bb, of which only
the symmetric parts contribute to the grid densities: 8 real symmetric
matrices per trial density, 2 with f_tt and 6 with f_ss.  All of them go
through one pass over the grid per AO block (the scalar CPHF machinery of
hessian.rhf_fast._FxcBatched: c = D phi^T for every density as one GEMM per
sub-block on its significant AOs, the fused C pass lib/dft/fxc_multi.c
SOCDFTfxc_fused for rho1 -> f rho1 -> aow, V += aow phi as one GEMM), the two
kernels sharing the AO values.

MODE: 'auto' (AO form; the MO form of mo_response where it wins), 'ao', 'mo',
'old' (the noncollinear path).
'''

import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger

MODE = 'auto'
ISO_TOL = 1e-12         # relative size of the non-isotropic blocks of the cached kernel accepted as zero
ZERO_TOL = 1e-10        # relative size below which a channel part of all trial densities counts as zero (skipped;
                        # as spinor_jk.KRAMERS_TOL: TR-even BiPh3 trial densities carry ~1e-12 of the vanishing parts)


def isotropic_kernel(fxc, xctype):
    '''(f_tt, f_ss) (nvar, nvar, ng) of a cached ncol / mcol kernel (4, nvar, 4, nvar, ng) at m = 0, or None'''
    fxc = numpy.asarray(fxc)
    nvar = 1 if xctype == 'LDA' else 4
    fxc = fxc.reshape(4, nvar, 4, nvar, -1)
    scale = max(abs(fxc[0, :, 0]).max(), abs(fxc[1, :, 1]).max(), 1e-300)
    tol = ISO_TOL * scale
    if abs(fxc[0, :, 1:]).max() > tol or abs(fxc[1:, :, 0]).max() > tol:
        return None
    for k in range(1, 4):
        for l in range(1, 4):
            if k != l and abs(fxc[k, :, l]).max() > tol:
                return None
        if k > 1 and abs(fxc[k, :, k] - fxc[1, :, 1]).max() > tol:
            return None
    return numpy.ascontiguousarray(fxc[0, :, 0]), numpy.ascontiguousarray(fxc[1, :, 1])


def _batched_v(ni, mol, grids, groups, xctype, max_memory):
    '''V (n_k, nao, nao) of the real symmetric densities of every group (d_k (n_k, nao, nao), f_k (nvar, nvar,
    ng)), one AO pass for all groups (hessian.rhf_fast._FxcBatched per sub-block)'''
    from socutils.hessian.rhf_fast import _load_socdft, _ao_stack, FXC_SUBBLOCK
    from pyscf.dft.numint import BLKSIZE
    libdft = _load_socdft()
    nao = mol.nao
    ncomp = 1 if xctype == 'LDA' else 4
    ao_loc = mol.ao_loc_nr()
    nf_shell = ao_loc[1:] - ao_loc[:-1]
    non0tab = getattr(grids, 'non0tab', None)
    ntot = sum(d.shape[0] for d, _ in groups)
    vmats = [numpy.zeros((d.shape[0], nao, nao)) for d, _ in groups]
    nsub = max(1, FXC_SUBBLOCK // BLKSIZE) * BLKSIZE
    ng_blk = max(nsub, int(max_memory * .2e6 / (8 * (ncomp + 1) * nao)) // nsub * nsub)
    blk_mem = ng_blk * (ncomp + 1) * nao * 8 / 1e6
    nmax = max(d.shape[0] for d, _ in groups)
    cbuf = numpy.empty(nmax * nao * nsub)
    aos_buf = numpy.empty(ncomp * nao * nsub)
    fws_buf = numpy.empty(ncomp * ncomp * nsub)
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ncomp // 4, blk_mem):
        ng = weight.size
        p0, p1 = p1, p1 + ng
        aoT, cs = _ao_stack(ao)
        fwbs = []
        for _, f in groups:
            fwb = numpy.ascontiguousarray(f[:, :, p0:p1] * weight)
            if xctype != 'LDA':
                fwb[0] *= .5
            fwbs.append(fwb)
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
            aos = aos_buf[:ncomp * nl * gs].reshape(ncomp, nl, gs)
            libdft.SOCDFTgather_ao(aos.ctypes, aoT.ctypes, None if idx is None else idx.ctypes,
                                   ctypes.c_int(ncomp), ctypes.c_int(nao), ctypes.c_int(nl), ctypes.c_int(ng),
                                   ctypes.c_int(s0), ctypes.c_int(gs), ctypes.c_long(cs))
            for (d, _), fwb, vmat in zip(groups, fwbs, vmats):
                n = d.shape[0]
                if idx is None:
                    dl = d.reshape(n * nao, nao)
                else:
                    dl = numpy.empty((n * nl, nl))
                    libdft.SOCDFTgather_dm(dl.ctypes, d.ctypes, idx.ctypes, ctypes.c_int(nl), ctypes.c_int(n),
                                           ctypes.c_int(nao))
                c = cbuf[:n * nl * gs].reshape(n * nl, gs)
                lib.dot(dl, aos[0], c=c)
                fws = fws_buf[:ncomp * ncomp * gs].reshape(ncomp, ncomp, gs)
                libdft.SOCDFTslice_grid(fws.ctypes, fwb.ctypes, ctypes.c_long(ncomp * ncomp), ctypes.c_int(ng),
                                        ctypes.c_int(s0), ctypes.c_int(gs))
                libdft.SOCDFTfxc_fused(c.ctypes, aos.ctypes, fws.ctypes, ctypes.c_int(n), ctypes.c_int(ncomp),
                                       ctypes.c_int(nl), ctypes.c_int(gs))
                if idx is None:
                    lib.dot(c, aos[0].T, beta=1, c=vmat.reshape(n * nao, nao))
                else:
                    vl = lib.dot(c, aos[0].T)
                    libdft.SOCDFTscatter_add_dm(vmat.ctypes, vl.ctypes, idx.ctypes, ctypes.c_int(nl),
                                                ctypes.c_int(n), ctypes.c_int(nao))
    if xctype != 'LDA':
        vmats = [lib.hermi_sum(v, axes=(0, 2, 1), inplace=True) for v in vmats]
    return vmats


def channels(dms):
    '''the real symmetric matrices of the grid densities of GHF densities (nset, 2 nao, 2 nao): density (nset, 2,
    nao, nao) [Re, Im of D_n] and magnetisation (nset, 3, 2, nao, nao) [k = x, y, z; Re, Im]'''
    nset, n2 = dms.shape[:2]
    nao = n2 // 2
    aa, ab = dms[:, :nao, :nao], dms[:, :nao, nao:]
    ba, bb = dms[:, nao:, :nao], dms[:, nao:, nao:]

    def sym(x):
        x = numpy.asarray(x, dtype=numpy.complex128)
        x = (x + x.transpose(0, 2, 1)) * .5
        return numpy.stack([x.real, x.imag], axis=1)
    dn = sym(aa + bb)
    dm = numpy.stack([sym(ab + ba), sym(1j * (ab - ba)), sym(aa - bb)], axis=1)
    return dn, dm


def assemble(vn, vm, nao):
    '''GHF matrices from the channel potentials vn (nset, 2, nao, nao) and vm (nset, 3, 2, nao, nao) (Re / Im)'''
    vr = vn[:, 0] + 1j * vn[:, 1]
    vx = vm[:, 0, 0] + 1j * vm[:, 0, 1]
    vy = vm[:, 1, 0] + 1j * vm[:, 1, 1]
    vz = vm[:, 2, 0] + 1j * vm[:, 2, 1]
    nset = vn.shape[0]
    out = numpy.empty((nset, 2 * nao, 2 * nao), dtype=numpy.complex128)
    out[:, :nao, :nao] = vr + vz
    out[:, nao:, nao:] = vr - vz
    out[:, :nao, nao:] = vx - 1j * vy
    out[:, nao:, :nao] = vx + 1j * vy
    return out


def nr_fxc(ni, mol, grids, xc_code, dm0, dms, hermi=0, rho0=None, vxc=None, fxc=None, max_memory=2000):
    '''the response of the GHF densities dms ((nset,) 2 nao, 2 nao) by the batched AO pass, or None'''
    from socutils.hessian.rhf_fast import _load_socdft
    if MODE in ('old', 'mo') or fxc is None or ni.collinear[0] not in ('n', 'm'):
        return None
    xctype = ni._xc_type(xc_code)
    libdft = _load_socdft()
    if (xctype not in ('LDA', 'GGA') or libdft is None or not hasattr(libdft, 'SOCDFTfxc_fused')
            or mol.cart):
        return None
    iso = isotropic_kernel(fxc, xctype)
    if iso is None:
        return None
    t0 = (logger.process_clock(), logger.perf_counter())
    dms_a = numpy.asarray(dms)
    shape = dms_a.shape
    d = dms_a.reshape(-1, *shape[-2:])
    nset = d.shape[0]
    nao = mol.nao
    if grids.non0tab is None:
        grids.build(with_non0tab=True)
    dn, dmg = channels(d)
    # the Re / Im parts of the density and of the magnetisation channels that vanish for every density are
    # skipped: a time-reversal even transition density (Kramers-adapted response) has Im n = 0 and Re m = 0, half
    # the channels
    scale = max(abs(dn).max(), abs(dmg).max(), 1e-300)
    tol = ZERO_TOL * scale
    groups, slots = [], []
    for part in range(2):
        x = dn[:, part]
        if abs(x).max() > tol:
            groups.append((numpy.ascontiguousarray(x), iso[0]))
            slots.append(('n', part))
        x = dmg[:, :, part]
        if abs(x).max() > tol:
            groups.append((numpy.ascontiguousarray(x.reshape(nset * 3, nao, nao)), iso[1]))
            slots.append(('m', part))
    vn = numpy.zeros((nset, 2, nao, nao))
    vm = numpy.zeros((nset, 3, 2, nao, nao))
    if groups:
        vs = _batched_v(ni, mol, grids, groups, xctype, max(1000, max_memory))
        for (kind, part), v in zip(slots, vs):
            if kind == 'n':
                vn[:, part] = v
            else:
                vm[:, :, part] = v.reshape(nset, 3, nao, nao)
    v = assemble(vn, vm, nao)
    logger.timer_debug1(ni, 'two-component f_xc (batched, %d densities)' % nset, *t0)
    return v.reshape(shape[:-2] + v.shape[-2:])


def mo_response(ni, mol, grids, iso, orbo, orbv, zs, xctype, max_memory=2000):
    '''the f_xc part of the TDA matvec in the MO form (nvec, nocc, nvir), without transition density matrices:
    per AO block the occupied spinors psi_i and, for all trial vectors at once, psi~_i = sum_a z_ia psi_a =
    phi^T (C_v z^T) by GEMMs; the channels n1, m1 from P_st = sum_i psi~_i,s psi_i,t^*; the isotropic kernel
    iso = (f_tt, f_ss); the potential W_st = v_n 1 + v_m . sigma applied to psi_i and projected on the AOs
    (R = sum_g phi (W psi)), then v_ia = sum C_v^* R.  orbo, orbv: GHF MO coefficients (2 nao, nocc / nvir).'''
    nao = mol.nao
    ncomp = 1 if xctype == 'LDA' else 4
    zs = numpy.asarray(zs).reshape(-1, orbo.shape[1], orbv.shape[1])
    nvec, nocc, nvir = zs.shape
    f_tt, f_ss = iso
    co = [numpy.ascontiguousarray(orbo[:nao]), numpy.ascontiguousarray(orbo[nao:])]
    cv = [orbv[:nao], orbv[nao:]]
    # Z_s = C_v,s z^T: (nao, nvec * nocc)
    zt = numpy.ascontiguousarray(zs.transpose(2, 0, 1).reshape(nvir, nvec * nocc))
    Z = [numpy.ascontiguousarray(numpy.dot(c, zt)) for c in cv]
    R = [numpy.zeros((nao, nvec * nocc), dtype=numpy.complex128) for _ in range(2)]
    p1 = 0
    blk_mem = max(200, max_memory * .2)
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ncomp // 4, blk_mem):
        ng = weight.size
        p0, p1 = p1, p1 + ng
        ao = numpy.ascontiguousarray(ao.reshape(ncomp, ng, nao))
        # psi[c][s] (ng, nocc), psit[c][s] (ng, nvec, nocc)
        psi = [[numpy.dot(ao[c], co[s]) for s in range(2)] for c in range(ncomp)]
        psit = [[numpy.dot(ao[c], Z[s]).reshape(ng, nvec, nocc) for s in range(2)] for c in range(ncomp)]
        # P_st rows (ncomp, ng, nvec) complex
        P = numpy.empty((2, 2, ncomp, ng, nvec), dtype=numpy.complex128)
        for s in range(2):
            for t in range(2):
                pc = psi[0][t].conj()
                P[s, t, 0] = numpy.einsum('gxi,gi->gx', psit[0][s], pc)
                for c in range(1, ncomp):
                    P[s, t, c] = (numpy.einsum('gxi,gi->gx', psit[c][s], pc)
                                  + numpy.einsum('gxi,gi->gx', psit[0][s], psi[c][t].conj()))
        rn = P[0, 0] + P[1, 1]
        rm = [P[0, 1] + P[1, 0], 1j * (P[0, 1] - P[1, 0]), P[0, 0] - P[1, 1]]
        w = weight[None, :, None]
        ftt = f_tt[:, :, p0:p1]
        fss = f_ss[:, :, p0:p1]
        vn = numpy.einsum('xyg,ygv->xgv', ftt, rn) * w
        vm = [numpy.einsum('xyg,ygv->xgv', fss, r) * w for r in rm]
        # W_st rows: [[vn + vz, vx - i vy], [vx + i vy, vn - vz]]
        W = [[vn + vm[2], vm[0] - 1j * vm[1]], [vm[0] + 1j * vm[1], vn - vm[2]]]
        for s in range(2):
            # A_i,s = sum_t W0_st psi_i,t + sum_c Wc_st d_c psi_i,t ;  B^c_i,s = sum_t Wc_st psi_i,t
            A = numpy.zeros((ng, nvec, nocc), dtype=numpy.complex128)
            for t in range(2):
                A += W[s][t][0][:, :, None] * psi[0][t][:, None, :]
                for c in range(1, ncomp):
                    A += W[s][t][c][:, :, None] * psi[c][t][:, None, :]
            R[s] += numpy.dot(ao[0].T, A.reshape(ng, nvec * nocc))
            for c in range(1, ncomp):
                B = sum(W[s][t][c][:, :, None] * psi[0][t][:, None, :] for t in range(2))
                R[s] += numpy.dot(ao[c].T, B.reshape(ng, nvec * nocc))
    # v_ia = sum_s C_v,s^H R_s
    v = sum(numpy.dot(cv[s].conj().T, R[s]) for s in range(2))            # (nvir, nvec * nocc)
    return v.reshape(nvir, nvec, nocc).transpose(1, 2, 0)
