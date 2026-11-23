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

import itertools
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint, gen_grid

AO_CUTOFF = 1e-13
W1_CUTOFF = 1e-14       # (atom, x) whose weight derivative on a sub-block is below this are skipped
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


def _vmat_occ(ao, wv, phic, xctype, rows0):
    '''(sum_g wv . d rho / d D) C on the local AOs: ao (nrow, ng, nl), wv (n, nvar, ng) for n potentials,
    phic[r] = ao[r] C (ng, ncol).  Returns (n, nl, ncol).'''
    a0 = ao[0]
    # columns per potential: sum over the rows of ao_r^T (wv_r' o phic_r'')
    out = lib.einsum('gm,ngc->nmc', a0, wv[:, 0, :, None] * phic[0][None])
    if xctype != 'LDA':
        for k in range(1, 4):
            out += lib.einsum('gm,ngc->nmc', ao[k], wv[:, k, :, None] * phic[0][None])
            out += lib.einsum('gm,ngc->nmc', a0, wv[:, k, :, None] * phic[k][None])
    if xctype == 'MGGA':
        for k in range(1, 4):
            out += .5 * lib.einsum('gm,ngc->nmc', ao[k], wv[:, 4, :, None] * phic[k][None])
    return out


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

    def dkey(*axes):
        return dpos[tuple(sorted(axes))]
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
            dl = numpy.ascontiguousarray(dm0[idx[:, None], idx])
            phid = {r: ao[_r(*r)].dot(dl) for r in drows}                  # (ng, nl), D phi_r
            # densities and the functional
            rho = numpy.empty((nvar, ng))
            for k, terms in enumerate(spec):
                rho[k] = sum(c * numpy.einsum('gm,gm->g', ao[_r(*a)], phid[b]) for a, b, c in terms)
            exc, vxc, fxc = ni.eval_xc_eff(xc, rho if gga else rho[0], 2, xctype=xctype)[:3]
            vxc = vxc.reshape(nvar, ng)
            fxc = fxc.reshape(nvar, nvar, ng)
            e = exc.ravel() * rho[0]
            if with_hess:
                egrid[pts] = e
            # R_A (3, nvar, nl -> atoms, ng) element-wise per AO, reduced over the AOs of each atom
            atoms, atl = numpy.unique(atom_of[idx], return_inverse=True)
            na = atoms.size
            Rmu = numpy.zeros((3, nvar, ng, idx.size))
            for x in range(3):
                for k, terms in enumerate(spec):
                    for a, b, c in terms:
                        Rmu[x, k] -= c * (ao[_r(*(a + (x,)))] * phid[b] + ao[_r(*(b + (x,)))] * phid[a])
            R = numpy.zeros((3, nvar, ng, na))
            for j in range(na):
                R[..., j] = Rmu[..., atl == j].sum(axis=-1)
            s = R.sum(axis=-1)                                            # (3, nvar, ng)
            # the owner's slot among the local atoms (appended if it has no AO here)
            ol = numpy.where(atoms == o)[0]
            if ol.size == 0:
                atoms = numpy.append(atoms, o)
                R = numpy.concatenate([R, numpy.zeros((3, nvar, ng, 1))], axis=-1)
                ol = na
                na += 1
            else:
                ol = int(ol[0])
            fs = numpy.einsum('klg,xlg->xkg', fxc, s)                      # f s (3, nvar, ng)
            if with_hess:
                # C2: w'_A (v . rho'_B) + w'_B (v . rho'_A), rho'_B = R_B - delta_Bo s
                eps1 = numpy.einsum('kg,xkga->axg', vxc, R)                 # (na, 3, ng)
                eps1[ol] -= numpy.einsum('kg,xkg->xg', vxc, s)
                t = lib.dot(w1.reshape(3 * natm, ng), eps1.reshape(3 * na, ng).T).reshape(natm, 3, na, 3)
                t = t.transpose(0, 2, 1, 3)                                # (natm, na, 3, 3)
                dH[:, atoms] += t
                dH[atoms, :] += t.transpose(1, 0, 3, 2)
                # C3: - delta_Bo w R_A f s - delta_Ao w s f R_B + delta_Ao delta_Bo w s f s
                wfs = fs * w
                rfs = numpy.einsum('xkga,ykg->axy', R, wfs)                 # sum_g w R_A,x f s_y
                dH[atoms, o] -= rfs
                dH[o, atoms] -= rfs.transpose(0, 2, 1)
                dH[o, o] += numpy.einsum('xkg,ykg->xy', s, wfs)
                # C4: v.Q_A[x, y] element-wise in the AOs, summed over the points with w
                wv = vxc * w
                V = {}
                for k, terms in enumerate(spec):
                    for a, b, c in terms:
                        V[a, b] = V.get((a, b), 0) + c * wv[k]
                qmu = numpy.zeros((3, 3, idx.size))
                for (a, b), vab in V.items():
                    for x in range(3):
                        for y in range(3):
                            t4 = (ao[_r(*(a + (x, y)))] * phid[b]
                                  + ao[_r(*(b + (x,)))] * phid[tuple(sorted(a + (y,)))]
                                  + ao[_r(*(a + (x,)))] * phid[tuple(sorted(b + (y,)))]
                                  + ao[_r(*(b + (x, y)))] * phid[a])
                            qmu[x, y] += vab.dot(t4)
                Q = numpy.zeros((na, 3, 3))
                for j in range(na):
                    sel = atl == j
                    if sel.any():
                        Q[j] = qmu[:, :, sel].sum(axis=-1)
                dH[atoms, o] -= Q
                dH[o, atoms] -= Q.transpose(0, 2, 1)
                dH[o, o] += Q.sum(axis=0)
            if with_fock:
                Cl = C[idx]
                phic = [ao[_r(*r)].dot(Cl) for r in ([()] + [(i,) for i in range(3)])]
                # (w'_A v - delta_Ao w f s) . d rho / d D for the (A, x) with a weight derivative here
                wmax = abs(w1).max(axis=2)                                 # (natm, 3)
                sel = numpy.argwhere(wmax * abs(vxc).max() > W1_CUTOFF)
                pots = [w1[A, x][None] * vxc for A, x in sel]
                keys = [(A, x) for A, x in sel]
                for x in range(3):
                    pots.append(-w * fs[x])
                    keys.append((o, x, 'f'))
                if pots:
                    m = _vmat_occ(ao, numpy.asarray(pots), phic, xctype, None)
                    for (key, mm) in zip(keys, m):
                        dV[key[0], key[1]][idx] += mm
                # owner: (G_x + G_x^T) C, G_x = sum_g w v . B(d_x phi, phi)
                wv = vxc * w
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
