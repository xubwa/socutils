#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# Non-collinear (ncol) and multi-collinear (mcol) exchange-correlation
# functionals for two-component (GKS / spinor) DFT.
#
# Both schemes turn a collinear functional e(rho, s) into a functional of the
# density and the magnetization vector m.  The evaluators below return the XC
# energy per particle and its derivatives with respect to the (rho, mx, my, mz)
# components -- and their gradients / tau -- in the layout of the multi-collinear
# matrix builders of pyscf.dft.numint2c (vxc: (4, nvar, ngrids); fxc:
# (4, nvar, 4, nvar, ngrids)), so one code path builds Vxc and the fxc response
# for both.  Rows of every density component are
#     LDA  : [value]                               nvar = 1
#     GGA  : [value, d/dx, d/dy, d/dz]             nvar = 4
#     MGGA : [value, d/dx, d/dy, d/dz, tau]        nvar = 5
#
# ncol -- locally collinear (Kubler; GGA after Scalmani & Frisch, JCTC 8, 2193
#   (2012)).  The spin density is |m| and every derivative row is projected on
#   the local axis n = m/|m|:
#       s = |m|,   grad s = sum_k n_k grad m_k,   tau_s = sum_k n_k tau_mk.
#   vxc and fxc are the analytic chain rule of these definitions.  Where
#   |m| -> 0 the axis n is undefined; there the functional is evaluated in its
#   isotropic, spin-unpolarized limit (zero spin potential, transverse kernel =
#   longitudinal kernel), which coincides with the multi-collinear kernel at
#   m = 0.  For GGA/MGGA the kernel is singular (~1/|m|, 1/|m|^2) as |m| -> 0
#   -- the known shortcoming of the locally collinear GGA -- so the kernel uses
#   the isotropic limit below the larger threshold |m| < ncol_fxc_thrd * rho
#   (SCF noise in m of a closed shell is ~1e-7 rho).  mcol is the method of
#   choice for the TDDFT kernel.
#
# mcol -- multi-collinear (Pu, Li, Zhang, Sun, Xiao, Liu, JCTC 19, 2006 (2023);
#   the mcfun library).  The collinear functional is averaged over all spin
#   projection directions u on the unit sphere,
#       e_mc(rho, m) = < e(rho, s_u) + s_u . de/ds(rho, s_u) >_u ,
#       s_u = u . m  (applied to every derivative row),
#   which gives potentials and kernels that are well defined at m = 0.
#   * LDA: s_u depends on u only through u.n, the sphere average collapses and
#     e_mc(rho, m) = e(rho, |m|) exactly (spin-flip symmetry of e).  Multi-
#     collinear LDA *is* locally collinear LDA, so it is evaluated in closed form
#     with no direction sampling and no third functional derivative.
#   * GGA/MGGA: the average is a genuine 2D sphere integral, done on a Lebedev
#     grid of `spin_samples` points.  The integrand is even under u -> -u, so
#     only half of the (inversion-symmetric) Lebedev points are evaluated, all
#     directions in one batched libxc call.  Points with |m| >= collinear_thrd
#     * rho (nearly fully polarized, where libxc derivatives on the sampled
#     directions degrade) are evaluated on the local principal axis, exactly as
#     mcfun does.
#   This reproduces mcfun.eval_xc_eff (same quadrature) without the mcfun
#   dependency.
#

import ctypes
import numpy as np
from pyscf import lib
from pyscf.dft import numint, numint2c, xc_deriv
from pyscf.dft.numint import _dot_ao_ao, _scale_ao, _tau_dot
from pyscf.dft.LebedevGrid import MakeAngularGrid


def _xc_type_of(ni, xc_code, xctype):
    return ni._xc_type(xc_code) if xctype is None else xctype


def eval_xc_ts(ni, xc_code, rho_ts, deriv, xctype, omega=None):
    '''Collinear functional on total/spin density rows.

    rho_ts: (2, nvar, ngrids).  Returns [exc, v, f, k] with derivatives in the
    total/spin (t, s) representation, v: (2, nvar, ngrids), f: (2, nvar, 2,
    nvar, ngrids), ...  Missing orders are None.
    '''
    if omega is None:
        omega = ni.omega
    t, s = rho_ts
    rho = np.stack([(t + s) * .5, (t - s) * .5])
    if xctype == 'LDA':
        rho = rho[:, 0]
    out = ni.libxc.eval_xc1(xc_code, rho, 1, deriv, omega)
    evfk = [out[0]]
    for order in range(1, deriv + 1):
        evfk.append(xc_deriv.ud2ts(xc_deriv.transform_xc(rho, out, xctype, 1, order)))
    evfk.extend([None] * (3 - deriv))
    return evfk


def _as_tm(rho_tm, xctype):
    rho_tm = np.asarray(rho_tm)
    if rho_tm.dtype != np.double:
        raise RuntimeError('rho and m must be real')
    if xctype == 'LDA':
        rho_tm = rho_tm.reshape(4, 1, -1)
    elif xctype == 'MGGA' and rho_tm.shape[1] == 6:
        # drop the laplacian row: [rho, dx, dy, dz, lapl, tau] -> [..., tau]
        rho_tm = rho_tm[:, [0, 1, 2, 3, 5]]
    return rho_tm


def _ncol_core(ni, xc_code, rho_tm, deriv, xctype, s_thrd):
    '''Locally collinear functional and its chain rule.  rho_tm: (4, nvar, ng).'''
    nvar, ng = rho_tm.shape[1:]
    rho = rho_tm[0]
    M = rho_tm[1:]                                  # (3, nvar, ng)
    s = np.sqrt(np.einsum('kg,kg->g', M[:, 0], M[:, 0]))
    pol = s > s_thrd * np.maximum(rho[0], 1e-300)   # local axis well defined
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_s = np.where(pol, 1. / s, 0.)
    nhat = M[:, 0] * inv_s                          # (3, ng), 0 where unpolarized

    rho_ts = np.empty((2, nvar, ng))
    rho_ts[0] = rho
    rho_ts[1] = np.einsum('kg,kvg->vg', nhat, M)    # s and projected rows
    rho_ts[1, 0] = np.where(pol, s, 0.)

    exc, v, f = eval_xc_ts(ni, xc_code, rho_ts, max(deriv, 1), xctype)[:3]
    ret = [exc]
    if deriv == 0:
        return ret

    vt, vs = v                                      # (nvar, ng)
    if deriv > 1:
        ftt = f[0, :, 0]
        fts = f[0, :, 1]
        fss = f[1, :, 1]

    sv = rho_ts[1]
    # The Jacobian d s_v / d M_kw is sparse:
    #     J[v, k, w] = delta_vw n_k + delta_w0 [v > 0] P[k, v],
    #     P[k, v] = d s_v / d m_k0 = (M_kv - n_k s_v) / s     (P[:, 0] == 0)
    # and every contraction with it below is written out explicitly.
    P = (M - nhat[:, None] * sv) * inv_s            # (3, nvar, ng)
    P1 = P[:, 1:]

    vxc = np.empty((4, nvar, ng))
    vxc[0] = vt
    vxc[1:] = nhat[:, None] * vs
    vxc[1:, 0] += np.einsum('vg,kvg->kg', vs[1:], P1)
    ret.append(vxc)
    if deriv == 1:
        return ret

    fxc = np.empty((4, nvar, 4, nvar, ng))
    fxc[0, :, 0] = ftt
    # sum_v fts[a, v] J[v, l, w] = fts[a, w] n_l + delta_w0 R[a, l]
    fxc[0, :, 1:] = fts[:, None] * nhat[None, :, None]
    fxc[0, :, 1:, 0] += np.einsum('avg,lvg->alg', fts[:, 1:], P1)
    fxc[1:, :, 0] = fxc[0, :, 1:].transpose(1, 2, 0, 3)
    # sum_vu J[v, k, w] fss[v, u] J[u, l, x]
    #   = n_k n_l fss[w, x] + delta_x0 n_k S[w, l] + delta_w0 n_l S[x, k]
    #     + delta_w0 delta_x0 T[k, l],   S = fss P^T,  T = P fss P^T  (v, u > 0)
    S = np.einsum('wug,lug->wlg', fss[:, 1:], P1)              # (nvar, 3, ng)
    T = np.einsum('kvg,vlg->klg', P1, S[1:])                   # (3, 3, ng)
    fkk = (nhat[:, None, None, None] * nhat[None, None, :, None]) * fss[None, :, None]
    fkk[:, :, :, 0] += nhat[:, None, None] * S[None]
    fkk[:, 0] += S.transpose(1, 0, 2)[:, None] * nhat[None, :, None]
    fkk[:, 0, :, 0] += T

    # second derivatives of s_v contracted with vs
    Q = np.eye(3)[:, :, None] - np.einsum('kg,lg->klg', nhat, nhat)   # (3, 3, ng)
    # s_0 = |m_0| :  d2 s / dm_k dm_l = Q_kl / s
    H00 = vs[0] * inv_s
    # s_v (v > 0) : d2/dm_k0 dm_l0 = -(P_kv n_l + n_k P_lv)/s - s_v Q_kl / s^2
    #               d2/dm_k0 dm_lv =  Q_kl / s
    Pvs = np.einsum('vg,kvg->kg', vs[1:], P[:, 1:])
    svs = np.einsum('vg,vg->g', vs[1:], sv[1:])
    fkk[:, 0, :, 0] += (Q * (H00 - svs * inv_s**2)
                        - (np.einsum('kg,lg->klg', Pvs, nhat)
                           + np.einsum('kg,lg->klg', nhat, Pvs)) * inv_s)
    for w in range(1, nvar):
        hw = Q * (vs[w] * inv_s)
        fkk[:, 0, :, w] += hw
        fkk[:, w, :, 0] += hw

    # Unpolarized points (n = 0, 1/s := 0, so P = 0): every term above already
    # vanishes there; add the isotropic limit f_kl = fss delta_kl (= the
    # multi-collinear kernel at m = 0).
    fss_unpol = fss * (~pol)
    for k in range(3):
        fkk[k, :, k] += fss_unpol
    fxc[1:, :, 1:] = fkk
    ret.append(fxc)
    return ret


def eval_xc_ncol(ni, xc_code, rho_tm, deriv=1, omega=None, xctype=None,
                 verbose=None, spin=None):
    '''Locally collinear (non-collinear) XC: [exc, vxc, fxc, None].'''
    if omega is not None and omega != ni.omega:
        raise NotImplementedError('omega')
    if deriv > 2:
        raise NotImplementedError('kxc for non-collinear functionals')
    xctype = _xc_type_of(ni, xc_code, xctype)
    rho_tm = _as_tm(rho_tm, xctype)
    s_thrd = ni.ncol_s_thrd
    if deriv > 1 and xctype != 'LDA':
        # the GGA/MGGA kernel has 1/|m| and 1/|m|^2 terms that turn SCF noise in
        # m (|m|/rho ~ 1e-7 for a converged closed shell) into garbage
        s_thrd = max(s_thrd, ni.ncol_fxc_thrd)
    ret = _ncol_core(ni, xc_code, rho_tm, deriv, xctype, s_thrd)
    ret.extend([None] * (4 - len(ret)))
    return ret


#
# multi-collinear
#
def _half_sphere(spin_samples):
    '''Lebedev directions with u and -u merged (weights normalized to 1).'''
    grid = MakeAngularGrid(spin_samples)
    u, w = grid[:, :3], grid[:, 3]
    w = w / w.sum()
    eps = 1e-12
    keep = ((u[:, 2] > eps)
            | ((abs(u[:, 2]) <= eps) & (u[:, 1] > eps))
            | ((abs(u[:, 2]) <= eps) & (abs(u[:, 1]) <= eps) & (u[:, 0] > 0)))
    u, w = u[keep], w[keep] * 2
    assert abs(w.sum() - 1) < 1e-10, 'Lebedev grid is not inversion symmetric'
    return np.ascontiguousarray(u), w


_libsocdft = False


def _load_libsocdft():
    global _libsocdft
    if _libsocdft is False:
        try:
            from socutils.lib import load_library
            _libsocdft = load_library('libsocdft', 'SOCUTILS_SOCDFT_LIBRARY')
        except OSError:
            _libsocdft = None
    return _libsocdft


def _mcol_sphere_c(ni, xc_code, rho_tm, deriv, xctype, u, wu):
    '''C kernel of _mcol_sphere (lib/dft/mcol.c): libxc derivatives are
    contracted per point, without the unfolded (2 nvar)^n tensors.'''
    from pyscf.dft import libxc
    lib_c = _load_libsocdft()
    rho_tm = np.asarray(rho_tm, order='C')
    nvar, ng = rho_tm.shape[1:]
    nu = wu.size
    rho_ud = np.empty((2, nvar, ng * nu))
    lib_c.SOCmcol_project(
        rho_ud.ctypes.data_as(ctypes.c_void_p), rho_tm.ctypes.data_as(ctypes.c_void_p),
        u.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nvar), ctypes.c_int(ng),
        ctypes.c_int(nu))
    # raw libxc rows; the xcfun-order permutation of eval_xc1 is applied in C
    xc = libxc._eval_xc(xc_code, rho_ud, 1, deriv + 1, ni.omega)
    rho_ud = None
    xcmap = np.arange(xc.shape[0])[libxc._libxc_to_xcfun_indices(xctype, 1, deriv + 1)]
    xcmap = np.asarray(xcmap, dtype=np.int32)
    exc = np.empty(ng)
    vxc = np.empty((4, nvar, ng))
    fxc = np.empty((4, nvar, 4, nvar, ng)) if deriv > 1 else np.empty(1)
    lib_c.SOCmcol_sphere(
        exc.ctypes.data_as(ctypes.c_void_p), vxc.ctypes.data_as(ctypes.c_void_p),
        fxc.ctypes.data_as(ctypes.c_void_p), rho_tm.ctypes.data_as(ctypes.c_void_p),
        u.ctypes.data_as(ctypes.c_void_p), wu.ctypes.data_as(ctypes.c_void_p),
        xc.ctypes.data_as(ctypes.c_void_p), xcmap.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nvar), ctypes.c_int(ng), ctypes.c_int(nu), ctypes.c_int(deriv))
    ret = [exc, vxc, fxc]
    return ret[:deriv + 1]


def _mcol_sphere(ni, xc_code, rho_tm, deriv, xctype, u, wu):
    '''Multi-collinear e, v, f by direction sampling.  rho_tm: (4, nvar, ng).'''
    nvar, ng = rho_tm.shape[1:]
    nu = wu.size
    rho = rho_tm[0]
    # projected spin rows s_u = u . m, layout (nvar, ng, nu) -> (nvar, ng*nu)
    rho_ts = np.empty((2, nvar, ng, nu))
    rho_ts[0] = rho[:, :, None]
    rho_ts[1] = np.einsum('kvg,uk->vgu', rho_tm[1:], u)
    sp = rho_ts[1]
    evfk = eval_xc_ts(ni, xc_code, rho_ts.reshape(2, nvar, ng * nu), deriv + 1, xctype)

    e = evfk[0].reshape(ng, nu)
    v = evfk[1].reshape(2, nvar, ng, nu)
    # rho * e_mc = < rho e + s_u . v_s >
    erho = np.einsum('gu,u->g', e * rho[0][:, None] + np.einsum('vgu,vgu->gu', v[1], sp), wu)
    with np.errstate(divide='ignore', invalid='ignore'):
        exc = np.where(rho[0] != 0, erho / rho[0], 0.)
    ret = [exc]
    if deriv == 0:
        return ret

    f = evfk[2].reshape(2, nvar, 2, nvar, ng, nu)
    # g = d/d(t, s) of (e rho + s . v_s)
    g = v.copy()
    g[1] *= 2
    g += np.einsum('vbygu,vgu->bygu', f[1], sp)
    vxc = np.empty((4, nvar, ng))
    vxc[0] = np.einsum('ygu,u->yg', g[0], wu)
    vxc[1:] = np.einsum('ygu,uk,u->kyg', g[1], u, wu)
    ret.append(vxc)
    if deriv == 1:
        return ret

    k = evfk[3].reshape(2, nvar, 2, nvar, 2, nvar, ng, nu)
    h = f.copy()
    h[1, :, 1] *= 3
    h[0, :, 1] *= 2
    h[1, :, 0] *= 2
    h += np.einsum('vaxbygu,vgu->axbygu', k[1], sp)
    fxc = np.empty((4, nvar, 4, nvar, ng))
    fxc[0, :, 0] = np.einsum('xygu,u->xyg', h[0, :, 0], wu)
    tmp = np.einsum('xygu,uk,u->xkyg', h[0, :, 1], u, wu)
    fxc[0, :, 1:] = tmp
    fxc[1:, :, 0] = tmp.transpose(1, 2, 0, 3)
    fxc[1:, :, 1:] = np.einsum('xygu,uk,ul,u->kxlyg', h[1, :, 1], u, u, wu)
    ret.append(fxc)
    return ret


def _mcol_paxis(ni, xc_code, rho_tm, deriv, xctype, nsamples):
    '''Multi-collinear functional for (nearly) collinear spins: all rows are
    projected on the local axis n; the transverse kernel is the average of the
    longitudinal one along the principal axis, int_0^1 f_ss(rho, t s) dt.'''
    nvar, ng = rho_tm.shape[1:]
    M = rho_tm[1:]
    s = np.sqrt(np.einsum('kg,kg->g', M[:, 0], M[:, 0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        nhat = np.where(s > 0, M[:, 0] / s, 0.)
    rho_ts = np.empty((2, nvar, ng))
    rho_ts[0] = rho_tm[0]
    rho_ts[1] = np.einsum('kg,kvg->vg', nhat, M)
    rho_ts[1, 0] = s
    evf = eval_xc_ts(ni, xc_code, rho_ts, max(deriv, 1), xctype)
    ret = [evf[0]]
    if deriv == 0:
        return ret
    v = evf[1]
    vxc = np.empty((4, nvar, ng))
    vxc[0] = v[0]
    vxc[1:] = np.einsum('kg,vg->kvg', nhat, v[1])
    ret.append(vxc)
    if deriv == 1:
        return ret

    f = eval_xc_ts(ni, xc_code, rho_ts, 2, xctype)[2]
    fxc = np.empty((4, nvar, 4, nvar, ng))
    fxc[0, :, 0] = f[0, :, 0]
    tmp = np.einsum('xyg,kg->xkyg', f[0, :, 1], nhat)
    fxc[0, :, 1:] = tmp
    fxc[1:, :, 0] = tmp.transpose(1, 2, 0, 3)
    nn = np.einsum('kg,lg->klg', nhat, nhat)
    fxc[1:, :, 1:] = np.einsum('xyg,klg->kxlyg', f[1, :, 1], nn)

    tq, wq = np.polynomial.legendre.leggauss(nsamples)
    tq = tq * .5 + .5
    wq = wq * .5
    rho_q = np.empty((2, nvar, ng, nsamples))
    rho_q[0] = rho_ts[0][:, :, None]
    rho_q[1] = rho_ts[1][:, :, None] * tq
    fq = eval_xc_ts(ni, xc_code, rho_q.reshape(2, nvar, ng * nsamples), 2, xctype)[2]
    fsf = np.einsum('xygq,q->xyg', fq[1, :, 1].reshape(nvar, nvar, ng, nsamples), wq)
    trans = np.eye(3)[:, :, None] - nn
    fxc[1:, :, 1:] += np.einsum('xyg,klg->kxlyg', fsf, trans)
    ret.append(fxc)
    return ret


def eval_xc_mcol(ni, xc_code, rho_tm, deriv=1, omega=None, xctype=None,
                 verbose=None, spin=None):
    '''Multi-collinear XC: [exc, vxc, fxc, None].'''
    if omega is not None and omega != ni.omega:
        raise NotImplementedError('omega')
    if deriv > 2:
        raise NotImplementedError('kxc for multi-collinear functionals')
    xctype = _xc_type_of(ni, xc_code, xctype)
    rho_tm = _as_tm(rho_tm, xctype)
    if xctype == 'LDA':
        # e_mc(rho, m) = e(rho, |m|): closed form, identical to ncol
        ret = _ncol_core(ni, xc_code, rho_tm, deriv, xctype, ni.ncol_s_thrd)
        ret.extend([None] * (4 - len(ret)))
        return ret

    nvar, ng = rho_tm.shape[1:]
    u, wu = _half_sphere(ni.spin_samples)
    ret = [np.zeros(ng)]
    if deriv > 0:
        ret.append(np.zeros((4, nvar, ng)))
    if deriv > 1:
        ret.append(np.zeros((4, nvar, 4, nvar, ng)))

    rho = rho_tm[0, 0]
    s = np.sqrt(np.einsum('kg,kg->g', rho_tm[1:, 0], rho_tm[1:, 0]))
    thrd = ni.collinear_thrd
    if thrd is None:
        cs = np.zeros(ng, dtype=bool)
    else:
        cs = s >= rho * thrd
    sph = np.where(~cs)[0]
    cs = np.where(cs)[0]

    # bound the size of the libxc output: (2 nvar)^(deriv+2) doubles per point
    if ni.mcol_kernel == 'c' and _load_libsocdft() is not None:
        sphere = _mcol_sphere_c
        # libxc output: comb(nq + deriv + 1, deriv + 1) doubles per point
        nq = 7 if xctype == 'MGGA' else 5
        per_point = lib.comb(nq + deriv + 1, deriv + 1)
    else:
        sphere = _mcol_sphere
        # unfolded tensors: (2 nvar)^(deriv+2) doubles per point
        per_point = (2 * nvar)**(deriv + 2)
    blk = max(1, int(ni.mcol_max_points / (per_point * wu.size)))
    for p0, p1 in lib.prange(0, sph.size, blk):
        idx = sph[p0:p1]
        r = sphere(ni, xc_code, rho_tm[..., idx], deriv, xctype, u, wu)
        for x, y in zip(ret, r):
            x[..., idx] = y
    if cs.size > 0:
        r = _mcol_paxis(ni, xc_code, rho_tm[..., cs], deriv, xctype,
                        ni.collinear_samples)
        for x, y in zip(ret, r):
            x[..., cs] = y
    ret.extend([None] * (4 - len(ret)))
    return ret


#
# Vxc / fxc matrices
#
_VXC_MAT = {'LDA': (numint2c._mcol_lda_vxc_mat, 0),
            'GGA': (numint2c._mcol_gga_vxc_mat, 1),
            'MGGA': (numint2c._mcol_mgga_vxc_mat, 1)}
_FXC_MAT = {'LDA': (numint2c._mcol_lda_fxc_mat, 0),
            'GGA': (numint2c._mcol_gga_fxc_mat, 1),
            'MGGA': (numint2c._mcol_mgga_fxc_mat, 1)}


# Transition densities and response potentials with real AOs.  For a GHF
# density matrix D = [[aa, ab], [ba, bb]] (complex, any symmetry) the density
# and magnetization are  x_k(r) = sum_uv ao_u(r) D_k[u,v] ao_v(r)  with
#     D_rho = aa + bb,  D_mx = ab + ba,  D_my = i (ab - ba),  D_mz = aa - bb,
# and only the symmetric part of D_k contributes.  Each D_k is therefore split
# into two real symmetric matrices and evaluated with the real, screened
# routines of pyscf.dft.numint -- no complex AO arithmetic.

def _rho_mag_dms(dm):
    nao = dm.shape[-1] // 2
    aa, ab = dm[:nao, :nao], dm[:nao, nao:]
    ba, bb = dm[nao:, :nao], dm[nao:, nao:]
    ds = []
    for d in (aa + bb, ab + ba, 1j * (ab - ba), aa - bb):
        d = np.asarray(d, dtype=np.complex128)
        d = (d + d.T) * .5
        ds.append(np.ascontiguousarray(d.real))
        ds.append(np.ascontiguousarray(d.imag))
    return ds


def _eval_rho1_real_ao(mol, ao, ds, mask, xctype):
    '''(rho, m) of a complex (transition) density matrix, as numint2c.eval_rho
    with hermi=0, from the output of _rho_mag_dms.'''
    rho = [numint.eval_rho(mol, ao, d, mask, xctype, hermi=1, with_lapl=False)
           for d in ds]
    rho = np.asarray(rho)
    return rho[0::2] + rho[1::2] * 1j


def eval_rho_2c(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
                with_lapl=False, verbose=None):
    '''Drop-in for pyscf.dft.numint2c.eval_rho (real AOs, no laplacian):
    (rho, mx, my, mz) rows of any GHF density matrix; real for hermi=1.'''
    assert not with_lapl
    rho = _eval_rho1_real_ao(mol, ao, _rho_mag_dms(dm), non0tab, xctype.upper())
    return rho.real.copy() if hermi else rho


def _vmat_real_ao(mol, ao, weight, v, mask, shls_slice, ao_loc, xctype):
    '''GHF matrix of the (complex) potential v = dE/d(rho, mx, my, mz)
    (layout (4, nvar, ngrids)), as numint2c._mcol_*_vxc_mat with hermi=0.'''
    ao0 = ao if xctype == 'LDA' else ao[0]
    mats = []
    for vk in v:
        wv = vk * weight
        if xctype == 'LDA':
            wv = wv.reshape(-1)
        else:
            wv = np.array(wv)
            wv[0] *= .5
        for w in (wv.real.copy(), wv.imag.copy()):
            if xctype == 'LDA':
                mat = _dot_ao_ao(mol, ao0, _scale_ao(ao0, w * .5), mask,
                                 shls_slice, ao_loc)
            else:
                mat = _dot_ao_ao(mol, ao0, _scale_ao(ao[:4], w[:4]), mask,
                                 shls_slice, ao_loc)
                if xctype == 'MGGA':
                    mat += _tau_dot(mol, ao, ao, w[4] * .25, mask, shls_slice,
                                    ao_loc)
            mats.append(mat + mat.T)
    vr, vx, vy, vz = [mats[2 * k] + mats[2 * k + 1] * 1j for k in range(4)]
    return np.block([[vr + vz, vx - vy * 1j], [vx + vy * 1j, vr - vz]])


def _contract_fxc(rho1, fxc):
    '''einsum('axg,axbyg->byg', rho1, fxc) for complex rho1, real fxc.'''
    nvar = rho1.shape[1]
    n = 4 * nvar
    rho1 = np.asarray(rho1, dtype=np.complex128, order='C').reshape(n, -1)
    ng = rho1.shape[1]
    fxc = fxc.reshape(n, n, -1)
    lib_c = _load_libsocdft()
    if lib_c is not None and hasattr(lib_c, 'SOCfxc_contract'):
        fxc = np.asarray(fxc, order='C')
        v = np.empty((n, ng), dtype=np.complex128)
        lib_c.SOCfxc_contract(v.ctypes.data_as(ctypes.c_void_p),
                              fxc.ctypes.data_as(ctypes.c_void_p),
                              rho1.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(n), ctypes.c_int(ng))
        return v.reshape(4, nvar, ng)
    re = np.zeros(fxc.shape[1:])
    im = np.zeros(fxc.shape[1:])
    for k in range(4 * nvar):
        re += fxc[k] * rho1[k].real
        im += fxc[k] * rho1[k].imag
    return (re + im * 1j).reshape(4, nvar, -1)


def _gks_noncol_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=0,
                    max_memory=2000, verbose=None):
    '''Vxc for ncol and mcol (both evaluators share the (rho, m) layout).'''
    xctype = ni._xc_type(xc_code)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    make_rho, nset, n2c = ni._gen_rho_evaluator(mol, dms, hermi, False)
    nao = n2c // 2

    nelec = np.zeros(nset)
    excsum = np.zeros(nset)
    vmat = np.zeros((nset, n2c, n2c), dtype=np.complex128)

    if xctype in _VXC_MAT:
        fmat, ao_deriv = _VXC_MAT[xctype]
        eval_xc = ni.noncol_eval_xc
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                exc, vxc = eval_xc(xc_code, rho, deriv=1, xctype=xctype)[:2]
                den = (rho[0] if xctype == 'LDA' else rho[0, 0]) * weight
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc)
                vmat[i] += fmat(mol, ao, weight, rho, vxc, mask, shls_slice,
                                ao_loc, hermi)
    elif xctype != 'HF':
        raise NotImplementedError(f'non-collinear get_vxc for functional {xc_code}')

    if hermi:
        vmat = vmat + vmat.conj().transpose(0, 2, 1)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
        nelec = nelec[0]
        excsum = excsum[0]
    return nelec, excsum, vmat


def _gks_noncol_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                    rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''fxc response for ncol and mcol.'''
    xctype = ni._xc_type(xc_code)
    if xctype == 'HF':
        return np.zeros_like(np.asarray(dms), dtype=np.complex128)
    if xctype not in _FXC_MAT:
        raise NotImplementedError(f'non-collinear get_fxc for functional {xc_code}')
    if fxc is None:
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, max_memory=max_memory)[2]
    if xctype == 'LDA':
        fxc = fxc.reshape(4, 1, 4, 1, -1)
    ao_deriv = _FXC_MAT[xctype][1]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dms_ = np.asarray(dms).reshape(-1, *np.shape(dms)[-2:])
    nset, n2c = dms_.shape[:2]
    nao = n2c // 2
    ds = [_rho_mag_dms(dm) for dm in dms_]
    vmat = np.zeros((nset, n2c, n2c), dtype=np.complex128)

    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        _fxc = fxc[..., p0:p1]
        for i in range(nset):
            rho1 = _eval_rho1_real_ao(mol, ao, ds[i], mask, xctype)
            if xctype == 'LDA':
                rho1 = rho1.reshape(4, 1, -1)
            v1 = _contract_fxc(rho1, _fxc)
            vmat[i] += _vmat_real_ao(mol, ao, weight, v1, mask, shls_slice,
                                     ao_loc, xctype)
    if isinstance(dms, np.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    return vmat


def _col_fxc_spin_symmetric(ni, mol, grids, xc_code, dm0, dms, relativity, hermi, rho0,
                            max_memory, verbose, thresh=1e-10):
    '''Spin-symmetric response (Re D_aa = Re D_bb in dm0 and in every dm1,
    e.g. the time-even perturbations of a Kramers closed shell): the collinear
    kernel then reduces to the restricted one, v_aa = v_bb = f_RKS rho1 with
    rho1 = Re(D_aa + D_bb), a single density channel.  None if not applicable.'''
    if not hermi or (dm0 is None and rho0 is None):
        return None
    nao = dms.shape[-1] // 2
    r1 = (dms[:, :nao, :nao] + dms[:, nao:, nao:]).real
    if abs((dms[:, :nao, :nao] - dms[:, nao:, nao:]).real).max() > thresh * max(1e-300, abs(r1).max()):
        return None
    ni1c = ni._to_numint1c()
    if dm0 is not None:
        r0 = (dm0[:nao, :nao] + dm0[nao:, nao:]).real
        if abs((dm0[:nao, :nao] - dm0[nao:, nao:]).real).max() > thresh * max(1, abs(r0).max()):
            return None
        r0 = np.ascontiguousarray(r0)
        key = (id(grids), xc_code, 'dm', hash(r0.tobytes()))
        make = lambda: ni1c.cache_xc_kernel1(mol, grids, xc_code, r0, spin=0, max_memory=max_memory)
    else:
        # the cached (UKS) reference rho0 of pyscf's GHF response function
        rho0 = np.asarray(rho0)
        if rho0.shape[0] != 2:
            return None
        if abs(rho0[0] - rho0[1]).max() > thresh * max(1, abs(rho0).max()):
            return None
        key = (id(grids), xc_code, 'rho', hash(rho0.tobytes()))
        xctype = ni1c._xc_type(xc_code)

        def make():
            rt = rho0[0] + rho0[1]
            v, f = ni1c.eval_xc_eff(xc_code, rt, deriv=2, xctype=xctype)[1:3]
            return rt, v, f
    cache = getattr(ni, '_rks_fxc_cache', None)
    if cache is None or cache[0] != key:
        ni._rks_fxc_cache = cache = (key,) + tuple(make())
    v = ni1c.nr_rks_fxc(mol, grids, xc_code, None, np.ascontiguousarray(r1), relativity, 1,
                        cache[1], cache[2], cache[3], max_memory, verbose)
    out = np.zeros(dms.shape, dtype=np.result_type(dms.dtype, np.double))
    out[:, :nao, :nao] = v
    out[:, nao:, nao:] = v
    return out


def _col_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
             rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Collinear fxc response (numint2c.NumInt2C.nr_fxc) with real density
    matrices only: the kernel is real and rho1 is linear in D, so
    v[D] = v[Re D] + i v[Im D], and Im D of a Hermitian D (anti-symmetric)
    gives no density.  pyscf's UKS kernel then runs its real, BLAS path
    instead of contracting complex views element-wise.'''
    dms = np.asarray(dms)
    nao = dms.shape[-1] // 2
    lead = dms.shape[:-2]
    dms = dms.reshape(-1, 2 * nao, 2 * nao)
    out = _col_fxc_spin_symmetric(ni, mol, grids, xc_code, dm0, dms, relativity, hermi, rho0,
                                  max_memory, verbose)
    if out is not None:
        return out.reshape(lead + (2 * nao, 2 * nao))
    if dm0 is not None:
        dm0 = (dm0[:nao, :nao].real.copy('C'), dm0[nao:, nao:].real.copy('C'))
    parts = [dms.real]
    with_imag = np.iscomplexobj(dms) and not hermi
    if with_imag:
        parts.append(dms.imag)
    d = np.concatenate(parts)
    dm1 = (np.ascontiguousarray(d[:, :nao, :nao]), np.ascontiguousarray(d[:, nao:, nao:]))
    va, vb = ni._to_numint1c().nr_uks_fxc(mol, grids, xc_code, dm0, dm1, relativity, hermi,
                                          rho0, vxc, fxc, max_memory, verbose)
    nset = dms.shape[0]
    out = np.zeros(dms.shape, dtype=np.result_type(dms.dtype, np.double))
    out[:, :nao, :nao] = va[:nset]
    out[:, nao:, nao:] = vb[:nset]
    if with_imag:
        out[:, :nao, :nao] += 1j * va[nset:]
        out[:, nao:, nao:] += 1j * vb[nset:]
    return out.reshape(lead + (2 * nao, 2 * nao))


class NumInt2C(numint2c.NumInt2C):
    '''GKS numerical integration with native non-collinear schemes.

    collinear:
        'col'  -- collinear (spin-diagonal) functional, pyscf default.
        'ncol' -- locally collinear; LDA, GGA and MGGA Vxc and fxc.
        'mcol' -- multi-collinear; LDA in closed form, GGA/MGGA on a
                  half Lebedev sphere of `spin_samples` directions.
                  Set ``mcol_backend = 'mcfun'`` to use the mcfun library
                  instead (reference implementation).
    '''

    mcol_backend = 'native'
    # native mcol GGA/MGGA sphere kernel: 'c' (lib/libsocdft.so, falls back to
    # numpy when it is not built) or 'numpy'
    mcol_kernel = 'c'
    # |m| / rho below which the local spin axis is treated as undefined (ncol)
    ncol_s_thrd = 1e-10
    # same for the ncol GGA/MGGA kernel (fxc), which is singular as |m| -> 0;
    # below it the kernel is the isotropic limit (= the mcol kernel at m = 0)
    ncol_fxc_thrd = 1e-4
    # bound on libxc output elements per batched multi-collinear call
    mcol_max_points = 2e7

    def noncol_eval_xc(self, xc_code, rho, deriv=1, omega=None, xctype=None,
                       verbose=None):
        c = self.collinear[0]
        if c == 'n':
            return eval_xc_ncol(self, xc_code, rho, deriv, omega, xctype, verbose)
        if c == 'm':
            if self.mcol_backend == 'mcfun':
                return self.mcfun_eval_xc_adapter(xc_code)(
                    xc_code, rho, deriv=deriv, xctype=xctype)
            return eval_xc_mcol(self, xc_code, rho, deriv, omega, xctype, verbose)
        raise RuntimeError(f'collinear={self.collinear} is not a non-collinear scheme')

    def mcfun_eval_xc_adapter(self, xc_code):
        '''Used by pyscf's own mcol code paths; routes to the active scheme.'''
        if self.collinear[0] == 'm' and self.mcol_backend == 'mcfun':
            return numint2c.mcfun_eval_xc_adapter(self, xc_code)
        return self.noncol_eval_xc

    def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None, spin=None):
        # ncol: (rho, m) derivatives, in the same layout as mcol.
        # col and mcfun's collinear callback (which passes (t, s) rows): pyscf.
        if self.collinear[0] == 'n':
            return eval_xc_ncol(self, xc_code, rho, deriv, omega, xctype, verbose)
        return numint2c._eval_xc_eff(self, xc_code, rho, deriv, omega, xctype,
                                     verbose)

    def _gks_mcol_vxc(self, mol, grids, xc_code, dms, relativity=0, hermi=0,
                      max_memory=2000, verbose=None):
        return _gks_noncol_vxc(self, mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)

    def _gks_mcol_fxc(self, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
                      rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        return _gks_noncol_fxc(self, mol, grids, xc_code, dm0, dms, relativity,
                               hermi, rho0, vxc, fxc, max_memory, verbose)

    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if self.collinear[0] in ('m', 'n'):
            return self._gks_mcol_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                      hermi, rho0, vxc, fxc, max_memory, verbose)
        return _col_fxc(self, mol, grids, xc_code, dm0, dms, relativity, hermi,
                        rho0, vxc, fxc, max_memory, verbose)
    get_fxc = nr_gks_fxc = nr_fxc

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        if self.collinear[0] not in ('m', 'n'):
            return numint2c.NumInt2C.cache_xc_kernel(
                self, mol, grids, xc_code, mo_coeff, mo_occ, spin, max_memory)
        dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
        return self.cache_xc_kernel1(mol, grids, xc_code, dm, spin, max_memory)

    def cache_xc_kernel1(self, mol, grids, xc_code, dm, spin=0, max_memory=2000):
        if self.collinear[0] not in ('m', 'n'):
            return numint2c.NumInt2C.cache_xc_kernel1(
                self, mol, grids, xc_code, dm, spin, max_memory)
        xctype = self._xc_type(xc_code)
        ao_deriv = 0 if xctype == 'LDA' else 1
        nao = dm.shape[-1] // 2
        rho = []
        for ao, mask, weight, coords \
                in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho.append(self.eval_rho1(mol, ao, dm, mask, xctype, 1, False))
        rho = np.concatenate(rho, axis=-1)
        assert rho.dtype == np.double
        if xctype == 'HF':
            return rho, None, None
        vxc, fxc = self.noncol_eval_xc(xc_code, rho, deriv=2, xctype=xctype)[1:3]
        return rho, vxc, fxc
