'''
Derivative of the PC-NEVPT2 second-order energy with respect to the CI vector of a singlet
model state at fixed orbitals (Python reference, real orbitals, the spin-orbital route of
zmr.nr.density), in the parametrization of PySCF's CASSCF, c' = (c + x) / |c + x|.

For a class with functions phi_X = O_X |Psi>, metric S = phi phi^T and
K = phi (H_act - s) phi^T (s = <Psi|H_act|Psi>), the energy of the external labels e is

    E = -sum_e v_e^T S (K + Delta_e S)^+ S v_e,

and with y_e = T^+ c_e (T[k, X] = <b_k|phi_X>, c the amplitudes)

    dE = -tr(dS Q_S) + tr(dK Q_K),  Q_S = sum_e (y v^T + v y^T - Delta y y^T),  Q_K = sum_e y y^T,

i.e. dE = 2 <dPsi|g> with

    g = -sum_X O_X^+ (Q_S^T phi)_X + sum_X O_X^+ (H - s)(Q_K^T phi)_X - tr(S Q_K)(H - s) Psi.

The denominators add sum_rs f_rs r^+ s Psi with f = dE/dgamma_act, the derivative through
the generalized Fock matrix: f_rs = sum_P [tr(D_gen B_P) B_P - B_P D_gen^T B_P]_rs.
Finally dE/dx = 2 (g - (c.g) c) on the determinants of the PySCF sector.
'''

import numpy

from socutils.zmr.nr.active import sector_index
from socutils.zmr.reference import orbital_gradient as og
from socutils.zmr.uc import classes as uc_classes
from socutils.zmr.uc import density, engine


def _adjoint(ops):
    return [(tuple(not d for d in reversed(dag)), numpy.ascontiguousarray(idx[:, ::-1]))
            for dag, idx in ops]


def _sum_adjoint(act, pclass, w, ntarget):
    '''sum_X O_X^+ w_X for vectors w[X, :] of the target sector.'''
    adj = _adjoint(pclass.operators(act.norb))
    y = act.apply_strings(numpy.ascontiguousarray(w), ntarget, adj)     # (X, X', D_N)
    return numpy.einsum('xxd->d', y)


def class_g(pt, ctx, name, state=0):
    '''g of one class with active operators (without the Fock term).'''
    pclass = uc_classes.CLASS_BY_NAME[name]
    act = pt.active_space
    psi = pt.civecs[:, state]
    ntarget = act.target(pt.nelec, pclass.dn)
    b, mu, t, s = density.first_order_space(pt, pclass, state)
    phi = engine.class_functions(pclass, act, psi[:, None], pt.nelec)[0]   # (X, D_t)
    nx = phi.shape[0]
    tp = numpy.linalg.pinv(t)                                               # (X, k)
    q_s = numpy.zeros((nx, nx), dtype=numpy.complex128)
    q_k = numpy.zeros((nx, nx), dtype=numpy.complex128)
    for delta, v in pclass.blocks(ctx):
        u = v @ t.T
        c = u / (delta[:, None] + mu[None, :] - s)
        y = c @ tp.T                                                        # (e, X)
        q_s += y.T @ v + v.T @ y - (y.T * delta[None, :]) @ y
        q_k += y.T @ y
    w_s = q_s.T @ phi
    w_k = q_k.T @ phi
    hw_k = act.apply_h(ntarget, w_k.T).T - s * w_k
    hpsi = act.apply_h(pt.nelec, psi[:, None])[:, 0] - s * psi
    smat = phi.conj() @ phi.T
    return (-_sum_adjoint(act, pclass, w_s, ntarget) + _sum_adjoint(act, pclass, hw_k, ntarget)
            - numpy.trace(smat @ q_k) * hpsi)


def fock_g(pt, ctx, d_gen, state=0):
    '''sum_rs f_rs r^+ s Psi, f = dE/dgamma_act through the generalized Fock matrix.'''
    sp = ctx.sp
    bmo = og.mo_vectors(pt.quasi_canonical.eris)
    bmo = bmo.real if numpy.iscomplexobj(bmo) else bmo
    a = sp.slice('a')
    tr = numpy.einsum('pq,Ppq->P', d_gen, bmo)
    f = (numpy.einsum('P,Prs->rs', tr, bmo[:, a, a])
         - numpy.einsum('Prq,pq,Pps->rs', bmo[:, a, :], d_gen, bmo[:, :, a]))
    act = pt.active_space
    na = act.norb
    psi = pt.civecs[:, state]
    pairs = numpy.indices((na, na)).reshape(2, -1).T
    ops = act.apply_strings(psi[None], pt.nelec, [((True, False), pairs)])[0]   # (rs, D)
    return numpy.einsum('r,rd->d', f.reshape(-1), ops)


def ci_gradient(pt, route, state=0, names=None):
    '''dE2/dx on the CI vector of the PySCF sector (route: zmr.nr.density.Route), of the
    classes names (default all).'''
    ctx = density.context(pt)
    sp = ctx.sp
    names = names or ('ccvv',) + tuple(density._LABELS)
    d_gen = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    if 'ccvv' in names:
        _, dcc, dvv = density.ccvv(ctx, kramers=False)
        d_gen = d_gen + density.assemble(sp, {'cc': dcc, 'vv': dvv})
    g = numpy.zeros(pt.civecs.shape[0], dtype=numpy.complex128)
    for name in names:
        if name == 'ccvv':
            continue
        _, dblk = density.class_reference(pt, ctx, name, state)
        for lab in ('cc', 'vv'):
            if lab in dblk:
                d_gen = d_gen + density.assemble(sp, {lab: dblk[lab]})
        g += class_g(pt, ctx, name, state)
    g += fock_g(pt, ctx, d_gen.real, state)
    idx = sector_index(pt.active_space.norb // 2, route.nelec_ab)
    gx = g[idx].real
    c = route.ci
    return 2 * (gx - numpy.dot(c, gx) * c)
