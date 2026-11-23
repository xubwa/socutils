#!/usr/bin/env python
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Grid response of the XC parts of the RKS Hessian, as corrections to the fixed-grid terms of
socutils.hessian.rhf_fast (xc_hess_explicit, the first-order V_xc of rks_make_h1).

A point g of atom o sits at r_g = R_o + u_g with partition weight w_g(R); an AO mu on atom b(mu) is
phi_mu(r_g - R_b), so d phi_mu / dR_A = c_A(mu) grad phi, c_A = delta_Ao - delta_Ab.  With the fixed-grid
derivative d^f_A (only the AOs of A move) and S = sum_C d^f_C = -grad_r,

    d_A = d^f_A - delta_Ao S,
    rho'_A = R_A - delta_Ao s,                          R_A = d^f_A rho,  s = sum_C R_C,
    rho''_AB = rho''^f_AB - delta_Bo Q_A - delta_Ao Q_B^T + delta_Ao delta_Bo sum_C Q_C,   Q_A = d^f_A S rho.

With the energy density e = exc rho, potential v and kernel f at the point,

    E''_AB = sum_g [ w''_AB e + w'_A v.rho'_B + w'_B v.rho'_A + w rho'_A f rho'_B + w v.rho''_AB ]
           = (fixed grid: sum_g w (R_A f R_B + v.rho''^f_AB))
           + sum_g w''_AB e                                                                   (C1)
           + sum_g (w'_A v.rho'_B + w'_B v.rho'_A)                                             (C2)
           - delta_Bo sum_g w R_A f s - delta_Ao sum_g w s f R_B + delta_Ao delta_Bo sum_g w s f s   (C3)
           - delta_Bo sum_g w v.Q_A - delta_Ao sum_g w v.Q_B^T + delta_Ao delta_Bo sum_g w v.sum_C Q_C (C4)

v.Q_A is element-wise in the AOs of A: with the density rows as bilinear forms rho_k = sum c_k^ab phi_a D phi_b
(phi_a the AO derivative rows) and V^ab = sum_k v_k c_k^ab,

    v.Q_A[x, y] = sum_{mu in A} sum_ab V^ab [phi_{a+x+y} (D phi_b) + phi_{b+x} (D phi_{a+y})
                                             + phi_{a+x} (D phi_{b+y}) + phi_{b+x+y} (D phi_a)]_mu.

The first-order V_xc (at fixed D) gains

    sum_g (w'_A v - delta_Ao w f s) . d rho / d D   and   delta_Ao (G + G^T),  G_x = sum_g w v . B(d_x phi, phi)

(B the bilinear density form, socutils.hessian.xc_grid_response._vmat_bilinear).  The points are visited by
owner atom (atm_idx); the weight derivatives come from lib/dft/becke_hess.c for the partition the grid ran
(Becke, SSF; socutils.grad.grid_response.partition_deriv1 / partition_hess, C1 the latter).
socutils.hessian.xc_grid_response is the reference (the full Hessian from displaced AOs, numpy, Becke only).
'''

import ctypes
import itertools
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint, gen_grid

AO_CUTOFF = 1e-13
W1_CUTOFF = 1e-11       # (atom, x) whose weight derivative x |v| on a sub-block is below this: skipped (Fock1)
BLKSIZE = 256           # points per sub-block

# AO derivative multi-index -> row of pyscf's AO array
_AO_IDX = {(): 0}
_n = 1
for _k in (1, 2, 3, 4):
    for _c in itertools.combinations_with_replacement(range(3), _k):
        _AO_IDX[_c] = _n
        _n += 1


def _r(*axes):
    return _AO_IDX[tuple(sorted(axes))]


def _bilinear_spec(xctype):
    '''rho_k = sum (a, b, c) c phi_a D phi_b over the density rows k'''
    spec = [[((), (), 1.)]]
    if xctype != 'LDA':
        for i in range(3):
            spec.append([((i,), (), 1.), ((), (i,), 1.)])
    if xctype == 'MGGA':
        spec.append([((i,), (i,), .5) for i in range(3)])
    return spec


def _vmat_occ(ao, wv, phic, xctype):
    """(sum_g wv . d rho / d D) C on the local AOs for n potentials at once: ao (nrow, ng, nl), wv (n, nvar, ng),
    phic[r] = ao[r] C (ng, ncol).  The terms are grouped by their left AO row, one GEMM each over all the potentials
    (columns (potential, ncol)).  Returns (n, nl, ncol)."""
    n = wv.shape[0]
    ng, ncol = phic[0].shape
    nl = ao.shape[2]
    # Y_left (ng, n, ncol): left ao0 <- wv0 phic0 + sum_k wv_k phic_k;  left ao_k <- wv_k phic0 + 1/2 wv_tau phic_k
    y0 = wv[:, 0].T[:, :, None] * phic[0][:, None, :]
    if xctype != 'LDA':
        for k in range(1, 4):
            y0 += wv[:, k].T[:, :, None] * phic[k][:, None, :]
    out = lib.dot(ao[0].T, y0.reshape(ng, n * ncol))
    if xctype != 'LDA':
        for k in range(1, 4):
            yk = wv[:, k].T[:, :, None] * phic[0][:, None, :]
            if xctype == 'MGGA':
                yk += .5 * wv[:, 4].T[:, :, None] * phic[k][:, None, :]
            lib.dot(ao[k].T, yk.reshape(ng, n * ncol), 1, out, 1)
    return out.reshape(nl, n, ncol).transpose(1, 0, 2)


def _tables(xctype, spec, dpos):
    """index tables of SOCDFTgr_point_terms: R terms (x, k, ao row, phid row) with coefficients, the V^ab keys,
    and the Q terms (x, y, ao row, phid row, V index)"""
    rtab, rcoef = [], []
    for x in range(3):
        for k, terms in enumerate(spec):
            for a, b, c in terms:
                rtab.append((x, k, _r(*(a + (x,))), dpos[b]))
                rcoef.append(-c)
                rtab.append((x, k, _r(*(b + (x,))), dpos[a]))
                rcoef.append(-c)
    vkeys = []
    for terms in spec:
        for a, b, c in terms:
            if (a, b) not in vkeys:
                vkeys.append((a, b))
    qtab = []
    for v, (a, b) in enumerate(vkeys):
        for x in range(3):
            for y in range(3):
                qtab.append((x, y, _r(*(a + (x, y))), dpos[b], v))
                qtab.append((x, y, _r(*(b + (x,))), dpos[tuple(sorted(a + (y,)))], v))
                qtab.append((x, y, _r(*(a + (x,))), dpos[tuple(sorted(b + (y,)))], v))
                qtab.append((x, y, _r(*(b + (x, y))), dpos[a], v))
    return (numpy.asarray(rtab, dtype=numpy.int32), numpy.asarray(rcoef), vkeys,
            numpy.asarray(qtab, dtype=numpy.int32))


def corrections(hessobj, mo_coeff, mo_occ, C=None, with_hess=True, with_fock=True, grids=None):
    '''(dH (natm, natm, 3, 3), dV1 (natm, 3, nao, ncol)): the grid-response corrections of the XC Hessian and of
    the first-order V_xc (times C, (nao, ncol); None: the AO matrices) at the RKS density of mo_coeff, mo_occ'''
    mf = hessobj.base
    mol = hessobj.mol
    ni = mf._numint
    xc = mf.xc
    xctype = ni._xc_type(xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError('grid response for %s' % xctype)
    if grids is None:
        grids = hessobj.grids if getattr(hessobj, 'grids', None) is not None else mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    from socutils.grad import grid_response as gresp
    if not gresp.available(grids):
        raise NotImplementedError('grid response: partition %s / lib/dft' % getattr(grids, 'becke_scheme', None))
    natm = mol.natm
    nao = mol.nao_nr()
    mocc = mo_coeff[:, mo_occ > 0]
    dm0 = numpy.dot(mocc * mo_occ[mo_occ > 0], mocc.T)
    if C is None:
        C = numpy.eye(nao)
    ncol = C.shape[1]
    gga = xctype != 'LDA'
    mgga = xctype == 'MGGA'
    nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    ao_deriv = 3 if gga else 2
    spec = _bilinear_spec(xctype)
    # D phi rows needed: up to order 1 (LDA), 2 (GGA / MGGA)
    drows = [()] + [(i,) for i in range(3)]
    if gga:
        drows += [(i, j) for i in range(3) for j in range(i, 3)]
    dpos = {r: n for n, r in enumerate(drows)}
    rtab, rcoef, vkeys, qtab = _tables(xctype, spec, dpos)
    from socutils.hessian.rhf_fast import _load_socdft
    libdft = _load_socdft()
    if libdft is None or not hasattr(libdft, 'SOCDFTgr_point_terms'):
        raise NotImplementedError('grid response: lib/dft not built')
    atom_of = numpy.zeros(nao, dtype=int)
    aoslices = mol.aoslice_by_atom()
    for ia in range(natm):
        atom_of[aoslices[ia][2]:aoslices[ia][3]] = ia

    dH = numpy.zeros((natm, natm, 3, 3))
    dV = numpy.zeros((natm, 3, nao, ncol)) if with_fock else None
    egrid = numpy.zeros(grids.weights.size) if with_hess else None
    t0 = (logger.process_clock(), logger.perf_counter())
    owners = numpy.asarray(grids.atm_idx)
    order = numpy.argsort(owners, kind='stable')
    bounds = numpy.searchsorted(owners[order], numpy.arange(natm + 1))
    for o in range(natm):
        pts_o = order[bounds[o]:bounds[o + 1]]
        for p0, p1 in lib.prange(0, pts_o.size, BLKSIZE):
            pts = pts_o[p0:p1]
            coords = grids.coords[pts]
            w = grids.weights[pts]
            w1 = gresp.partition_deriv1(grids, pts, mol).transpose(1, 2, 0)    # (natm, 3, ng)
            ng = w.size
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask)    # (ncomp, ng, nao)
            idx = numpy.where(abs(ao).max(axis=(0, 1)) > AO_CUTOFF)[0]
            if idx.size == 0:
                continue
            ao = numpy.ascontiguousarray(ao[:, :, idx])
            nl = idx.size
            dl = numpy.ascontiguousarray(dm0[idx[:, None], idx])
            phid = numpy.empty((len(drows), ng, nl))                       # D phi_r
            for n_, r in enumerate(drows):
                lib.dot(ao[_r(*r)], dl, c=phid[n_])
            # densities and the functional
            rho = numpy.empty((nvar, ng))
            for k, terms in enumerate(spec):
                rho[k] = sum(c * numpy.einsum('gm,gm->g', ao[_r(*a)], phid[dpos[b]]) for a, b, c in terms)
            exc, vxc, fxc = ni.eval_xc_eff(xc, rho if gga else rho[0], 2, xctype=xctype)[:3]
            vxc = vxc.reshape(nvar, ng)
            fxc = fxc.reshape(nvar, nvar, ng)
            e = exc.ravel() * rho[0]
            if with_hess:
                egrid[pts] = e
            # the local atoms (the AOs are sorted by atom) and the owner's slot (appended if it has no AO here)
            aidx = atom_of[idx]
            atoms = numpy.unique(aidx)
            seg = numpy.searchsorted(aidx, numpy.append(atoms, natm)).astype(numpy.int32)
            na = atoms.size
            wv = vxc * w
            V = numpy.zeros((len(vkeys), ng))
            for k, terms in enumerate(spec):
                for a, b, c in terms:
                    V[vkeys.index((a, b))] += c * wv[k]
            R = numpy.empty((3, nvar, ng, na))
            Q = numpy.zeros((na, 3, 3))
            libdft.SOCDFTgr_point_terms(
                R.ctypes.data_as(ctypes.c_void_p), Q.ctypes.data_as(ctypes.c_void_p),
                ao.ctypes.data_as(ctypes.c_void_p), phid.ctypes.data_as(ctypes.c_void_p),
                V.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ng), ctypes.c_int(nl),
                seg.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(na),
                rtab.ctypes.data_as(ctypes.c_void_p), rcoef.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(rcoef)), ctypes.c_int(nvar),
                qtab.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(qtab) if with_hess else 0))
            s_ = R.sum(axis=-1)                                           # (3, nvar, ng)
            ol = numpy.where(atoms == o)[0]
            if ol.size == 0:
                atoms = numpy.append(atoms, o)
                R = numpy.concatenate([R, numpy.zeros((3, nvar, ng, 1))], axis=-1)
                Q = numpy.concatenate([Q, numpy.zeros((1, 3, 3))])
                ol = na
                na += 1
            else:
                ol = int(ol[0])
            fs = numpy.einsum('klg,xlg->xkg', fxc, s_)                     # f s (3, nvar, ng)
            if with_hess:
                # C2: w'_A (v . rho'_B) + w'_B (v . rho'_A), rho'_B = R_B - delta_Bo s
                eps1 = numpy.einsum('kg,xkga->axg', vxc, R)                 # (na, 3, ng)
                eps1[ol] -= numpy.einsum('kg,xkg->xg', vxc, s_)
                t = lib.dot(w1.reshape(3 * natm, ng), eps1.reshape(3 * na, ng).T).reshape(natm, 3, na, 3)
                t = t.transpose(0, 2, 1, 3)                                # (natm, na, 3, 3)
                dH[:, atoms] += t
                dH[atoms, :] += t.transpose(1, 0, 3, 2)
                # C3: - delta_Bo w R_A f s - delta_Ao w s f R_B + delta_Ao delta_Bo w s f s
                wfs = fs * w
                rfs = numpy.einsum('xkga,ykg->axy', R, wfs)                 # sum_g w R_A,x f s_y
                dH[atoms, o] -= rfs
                dH[o, atoms] -= rfs.transpose(0, 2, 1)
                dH[o, o] += numpy.einsum('xkg,ykg->xy', s_, wfs)
                # C4: -delta_Bo w v.Q_A - delta_Ao w v.Q_B^T + delta_Ao delta_Bo w v.sum_C Q_C
                dH[atoms, o] -= Q
                dH[o, atoms] -= Q.transpose(0, 2, 1)
                dH[o, o] += Q.sum(axis=0)
            if with_fock:
                Cl = numpy.ascontiguousarray(C[idx])
                phic = [ao[_r(*r)].dot(Cl) for r in ([()] + [(i,) for i in range(3)])]
                # (w'_A v - delta_Ao w f s) . d rho / d D for the (A, x) with a weight derivative here
                wmax = abs(w1).max(axis=2)                                 # (natm, 3)
                sel = numpy.argwhere(wmax * abs(vxc).max() > W1_CUTOFF)
                pots = numpy.empty((len(sel) + 3, nvar, ng))
                for n_, (A, x) in enumerate(sel):
                    numpy.multiply(w1[A, x], vxc, out=pots[n_])
                for x in range(3):
                    numpy.multiply(-w, fs[x], out=pots[len(sel) + x])
                keys = [(A, x) for A, x in sel] + [(o, x) for x in range(3)]
                m = _vmat_occ(ao, pots, phic, xctype)
                for (A, x), mm in zip(keys, m):
                    dV[A, x][idx] += mm
                # owner: (G_x + G_x^T) C, G_x = sum_g w v . B(d_x phi, phi)
                for x in range(3):
                    dx = [ao[_r(x)]] + ([ao[_r(i, x)] for i in range(3)] if gga else [])
                    a0r = [ao[0]] + ([ao[_r(i)] for i in range(3)] if gga else [])
                    dxc = [d.dot(Cl) for d in dx]
                    # G C = sum U^T (wv V C);  G^T C = sum V^T (wv U C)
                    g = dx[0].T.dot(wv[0][:, None] * phic[0]) + a0r[0].T.dot(wv[0][:, None] * dxc[0])
                    if gga:
                        for k in range(1, 4):
                            g += dx[k].T.dot(wv[k][:, None] * phic[0]) + dx[0].T.dot(wv[k][:, None] * phic[k])
                            g += a0r[0].T.dot(wv[k][:, None] * dxc[k]) + a0r[k].T.dot(wv[k][:, None] * dxc[0])
                    if mgga:
                        for k in range(1, 4):
                            g += .5 * (dx[k].T.dot(wv[4][:, None] * phic[k])
                                       + a0r[k].T.dot(wv[4][:, None] * dxc[k]))
                    dV[o, x][idx] += g
    t0 = logger.timer(hessobj, 'XC grid response: point terms', *t0)
    if with_hess:
        dH += gresp.partition_hess(grids, egrid, mol)                   # (A, B, x, y)
        logger.timer(hessobj, 'XC grid response: second weight derivatives', *t0)
    return dH, dV


# ---------------------------------------------------------------- the grid response inside the local passes
class _GRBase:
    def __init__(self, hessobj, grids, xctype):
        from socutils.grad import grid_response as gresp
        from socutils.hessian.rhf_fast import _load_socdft
        if not gresp.available(grids):
            raise NotImplementedError('grid response: partition %s / lib/dft' % getattr(grids, 'becke_scheme', None))
        self.gresp = gresp
        self.grids = grids
        self.mol = hessobj.mol
        self.natm = self.mol.natm
        self.xctype = xctype
        self.nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
        self.owner = numpy.asarray(grids.atm_idx)
        self.libdft = _load_socdft()
        self.w1 = None
        self.p0 = 0

    def block(self, p0, ngrid):
        '''the weight derivatives (natm, 3, ngrid) of the pyscf grid block of points p0 .. p0 + ngrid'''
        self.p0 = p0
        self.w1 = numpy.ascontiguousarray(
            self.gresp.partition_deriv1(self.grids, numpy.arange(p0, p0 + ngrid), self.mol).transpose(1, 2, 0))


class HessGR(_GRBase):
    '''the grid-response corrections C1 .. C4 (module doc) accumulated inside rhf_fast._xc_hess_explicit_local'''
    def __init__(self, hessobj, grids, xctype):
        _GRBase.__init__(self, hessobj, grids, xctype)
        spec = _bilinear_spec(xctype)
        drows = [()] + [(i,) for i in range(3)]
        if xctype != 'LDA':
            drows += [(i, j) for i in range(3) for j in range(i, 3)]
        self.spec, self.drows = spec, drows
        self.dpos = {r: n for n, r in enumerate(drows)}
        self.rtab, self.rcoef, self.vkeys, self.qtab = _tables(xctype, spec, self.dpos)
        self.dH = numpy.zeros((self.natm, self.natm, 3, 3))
        self.egrid = numpy.zeros(grids.weights.size)

    def sub(self, s0, s1, aol, dl, dT, rho, exc, vxc, fxc, w, R, atoms, idx, atom_of):
        '''one sub-block (points p0 + s0 .. p0 + s1, local AOs idx): aol (ncomp, nl, gs), dT = D aol[:nd],
        R (3, nvar, na, gs) the true d^f_A rho of the local atoms, vxc (nvar, gs), fxc (nvar, nvar, gs) unweighted'''
        natm = self.natm
        gs = s1 - s0
        nl = idx.size
        g0 = self.p0 + s0
        own = self.owner[g0:g0 + gs]
        self.egrid[g0:g0 + gs] = exc.ravel() * rho[0]
        phid = numpy.empty((len(self.drows), nl, gs))
        nd_ = dT.shape[0]
        for n_, r in enumerate(self.drows):
            if n_ < nd_:
                phid[n_] = dT[n_]
            else:
                lib.dot(dl, aol[_r(*r)], c=phid[n_])
        uo, ownl = numpy.unique(own, return_inverse=True)
        ownl = numpy.ascontiguousarray(ownl, dtype=numpy.int32)
        nown = uo.size
        na = atoms.size
        seg = numpy.searchsorted(atom_of[idx], numpy.append(atoms, natm)).astype(numpy.int32)
        wv = vxc * w
        V = numpy.zeros((len(self.vkeys), gs))
        for k, terms in enumerate(self.spec):
            for a, b, c in terms:
                V[self.vkeys.index((a, b))] += c * wv[k]
        Q = numpy.zeros((na, nown, 3, 3))
        aolc = numpy.ascontiguousarray(aol)
        self.libdft.SOCDFTgr_q_owner(
            Q.ctypes.data_as(ctypes.c_void_p), aolc.ctypes.data_as(ctypes.c_void_p),
            phid.ctypes.data_as(ctypes.c_void_p), V.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(gs),
            ctypes.c_int(nl), seg.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(na),
            ownl.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nown),
            self.qtab.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(self.qtab)))
        s = R.sum(axis=2)                                                 # (3, nvar, gs)
        dH = self.dH
        # C2 over the local atoms and the owners: eps1 = v . rho'_B
        L = numpy.union1d(atoms, uo)
        pos = numpy.searchsorted(L, atoms)
        eps1 = numpy.zeros((L.size, 3, gs))
        eps1[pos] = numpy.einsum('kg,xkag->axg', vxc, R)
        vs = numpy.einsum('kg,xkg->xg', vxc, s)
        po = numpy.searchsorted(L, own)
        numpy.subtract.at(eps1, (po, slice(None), numpy.arange(gs)), vs.T)
        w1 = self.w1[:, :, s0:s1]
        t = lib.dot(w1.reshape(3 * natm, gs), eps1.reshape(3 * L.size, gs).T).reshape(natm, 3, L.size, 3)
        t = t.transpose(0, 2, 1, 3)
        dH[:, L] += t
        dH[L, :] += t.transpose(1, 0, 3, 2)
        # C3, C4 per owner
        wfs = numpy.einsum('klg,xlg->xkg', fxc, s) * w
        for iu, u in enumerate(uo):
            m = ownl == iu
            rfs = numpy.einsum('xkag,ykg->axy', R[:, :, :, m], wfs[:, :, m])
            dH[atoms, u] -= rfs + Q[:, iu]
            dH[u, atoms] -= (rfs + Q[:, iu]).transpose(0, 2, 1)
            dH[u, u] += numpy.einsum('xkg,ykg->xy', s[:, :, m], wfs[:, :, m]) + Q[:, iu].sum(axis=0)

    def finish(self):
        return self.dH + self.gresp.partition_hess(self.grids, self.egrid, self.mol)


class Fock1GR(_GRBase):
    '''the grid-response corrections of the first-order V_xc inside rhf_fast._xc_deriv1_occ_local, in its internal
    sign (the pass negates vmat at the end) and potential convention (row 0 / 2, tau / 4 against v . d rho / d D)'''
    def __init__(self, hessobj, grids, xctype):
        _GRBase.__init__(self, hessobj, grids, xctype)
        conv = numpy.ones(self.nvar)
        conv[0] = .5
        if xctype == 'MGGA':
            conv[4] = .25
        self.conv = conv[:, None]

    def rows(self, s0, s1, atoms, wv1, R_raw, vxc, wf):
        '''(rows, potentials) for SOCDFTxcd1_occ: the local atoms' wv1 (3 na, nvar, gs) plus the weight and owner
        terms -(w'_A v - delta_Ao w f s) (converted), R_raw = -d^f rho / 2 of the local atoms (3, nvar, na, gs)'''
        nvar = self.nvar
        gs = s1 - s0
        g0 = self.p0 + s0
        own = self.owner[g0:g0 + gs]
        w1 = self.w1[:, :, s0:s1]
        sel = numpy.where(abs(w1).max(axis=(1, 2)) * abs(vxc).max() > W1_CUTOFF)[0]
        L = numpy.union1d(numpy.union1d(atoms, sel), numpy.unique(own))
        pots = numpy.zeros((L.size, 3, nvar, gs))
        pots[numpy.searchsorted(L, atoms)] = wv1.reshape(atoms.size, 3, nvar, gs)
        ps = numpy.searchsorted(L, sel)
        # -(w'_A v): converted
        pots[ps] -= w1[sel][:, :, None, :] * (vxc * self.conv)[None, None]
        # +(w f s) at the owner, s = -2 sum R_raw
        s = -2 * R_raw.sum(axis=2)                                        # (3, nvar, gs)
        fs = numpy.einsum('klg,xlg->xkg', wf, s) * self.conv[None]
        po = numpy.searchsorted(L, own)
        numpy.add.at(pots, (po, slice(None), slice(None), numpy.arange(gs)), fs.transpose(2, 0, 1))
        rows = numpy.asarray((3 * L[:, None] + numpy.arange(3)).ravel(), dtype=numpy.int32)
        return rows, numpy.ascontiguousarray(pots.reshape(3 * L.size, nvar, gs))

    def owner_terms(self, s0, s1, vmat, idx, Lv, Rv, cl):
        '''-(G_x + G_x^T) C of each owner on its points (internal sign), G_x the v_ip integrand: Lv (3, nl, nb gs),
        Rv (nl, nb gs) column blocks of gs points'''
        gs = s1 - s0
        g0 = self.p0 + s0
        own = self.owner[g0:g0 + gs]
        nb = Rv.shape[1] // gs
        nocc = cl.shape[1]
        vm = vmat.reshape(self.natm, 3, -1, nocc)
        for u in numpy.unique(own):
            m = numpy.where(own == u)[0]
            cols = (numpy.arange(nb)[:, None] * gs + m).ravel()
            Ru = Rv[:, cols]
            rc = lib.dot(Ru.T, cl)                                        # (ncols, nocc)
            for x in range(3):
                Lu = Lv[x][:, cols]
                g = lib.dot(Lu, rc) + lib.dot(Ru, lib.dot(Lu.T, cl))
                vm[u, x][idx] -= g
