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
    '''sum_X O_X^+ w_X for vectors w[X, :] of the target sector: every string applied to
    its own vector only.'''
    out = None
    x0 = 0
    for dag, idx in _adjoint(pclass.operators(act.norb)):
        for k in range(len(idx)):
            y = act.apply_strings(numpy.ascontiguousarray(w[x0 + k][None]), ntarget,
                                  [(dag, idx[k:k + 1])])[0, 0]
            out = y if out is None else out + y
        x0 += len(idx)
    return out


def class_g(pt, ctx, name, state=0, rho=None):
    '''g of one class with active operators (without the Fock term) for one model state,
    or with rho (zmr.uc.density.model_spaces) the vectors g[J] of all model states for the
    target sum_IJ rho_IJ h2_IJ of the union contraction: dT = 2 sum_J <dPsi_J|g_J>.

    The functions A = (J, X) of all states span the first-order space; with the vectors
    R^J_e[A] = delta_{J_A J} v[e, X_A], L^J_e[A] = conj(rho_{J_A J}) v[e, X_A],
    Y_R = T^+ c^J, Y_L = T^+ l^J (zmr.uc.density.pair_amplitudes) and the metric S and
    K0 = phi H phi^T over A, every ket J contributes

        dT = -tr(dS Q_S) + tr(dK0 Q_K) - ds_J tau_J,
        Q_S = sum_e (L Y_R^T + Y_L R^T - (delta_e - s_J) Y_L Y_R^T),  Q_K = sum_e Y_L Y_R^T,
        tau_J = sum_e Y_L^T S Y_R.'''
    pclass = uc_classes.CLASS_BY_NAME[name]
    b, mu, t, s, rho_, states = density.model_spaces(pt, pclass, state, rho)
    acc = QAccumulator(pt, pclass, t, s, rho_, states)
    for delta, v in pclass.blocks(ctx):
        _, c, l, _ = density.pair_amplitudes(delta, v, mu, t, s, rho_)
        acc.add(c, l, v, delta)
    out = acc.g()
    return out[0] if rho is None else out


class QAccumulator:
    '''The matrices Q_S, Q_K over the functions A = (J, X) of the states and the numbers
    tau_J of class_g, accumulated block by block of external labels (add), and the
    vectors g[J] from them (g).'''

    def __init__(self, pt, pclass, t, s, rho, states):
        self.pt, self.pclass, self.s, self.rho, self.states = pt, pclass, s, rho, states
        nj, nk, nx = t.shape
        self.nj, self.nx = nj, nx
        na_ = nj * nx
        self.tp = numpy.ascontiguousarray(
            numpy.linalg.pinv(t.transpose(1, 0, 2).reshape(nk, na_)).T)     # (k, A)
        psi = pt.civecs[:, states]
        self.phi = engine.class_functions(pclass, pt.active_space, psi, pt.nelec)
        pf = self.phi.reshape(na_, -1)
        self.smat = pf.conj() @ pf.T
        self.q_s = numpy.zeros((na_, na_), dtype=numpy.complex128)
        self.q_k = numpy.zeros((na_, na_), dtype=numpy.complex128)
        self.tau = numpy.zeros(nj, dtype=numpy.complex128)

    def add(self, c, l, v, delta, f=1.0):
        '''Labels e of one block: amplitudes c, l (J, e, k) of pair_amplitudes, v[e, X],
        delta[e]; weight f (labels counted twice: 1/2).'''
        nj, nx = self.nj, self.nx
        ne = len(delta)
        for j in range(nj):
            yr = c[j] @ self.tp                                         # (e, A)
            yl = l[j] @ self.tp
            ylh = yl.T.conj()
            # L^J_e[A] = conj(rho_{J_A J}) v[e, X_A], R^J_e[A] = delta_{J_A J} v[e, X_A]
            lv = numpy.einsum('i,ex->ixe', self.rho[:, j].conj(), v).reshape(nj * nx, ne)
            lr = lv.conj() @ yr
            rr = numpy.zeros((nj * nx, ne), dtype=numpy.result_type(yl, v))
            rr[j * nx:(j + 1) * nx] = v.T
            self.q_s += f * (lr + ylh @ rr.T - (ylh * (delta - self.s[j])[None, :]) @ yr)
            self.q_k += f * (ylh @ yr)
            self.tau[j] += f * numpy.einsum('ea,ab,eb->', yl.conj(), self.smat, yr)

    def g(self):
        pt, pclass, act = self.pt, self.pclass, self.pt.active_space
        ntarget = act.target(pt.nelec, pclass.dn)
        pf = self.phi.reshape(self.nj * self.nx, -1)
        q_s = 0.5 * (self.q_s + self.q_s.T)
        q_k = 0.5 * (self.q_k + self.q_k.T)
        w_s = (q_s @ pf).reshape(self.nj, self.nx, -1)
        w_k = (q_k @ pf).reshape(self.nj, self.nx, -1)
        out = []
        for j, jj in enumerate(self.states):
            psi = pt.civecs[:, jj]
            hw = act.apply_h(ntarget, w_k[j].T).T
            hpsi = act.apply_h(pt.nelec, psi[:, None])[:, 0] - self.s[j] * psi
            out.append(-_sum_adjoint(act, pclass, w_s[j], ntarget)
                       + _sum_adjoint(act, pclass, hw, ntarget) - self.tau[j].real * hpsi)
        return numpy.array(out)


def fock_g(pt, ctx, d_gen, state=0, w=None, states=None):
    '''sum_rs f_rs r^+ s Psi, f = dE/dgamma_act through the generalized Fock matrix:
    f = (J - K)[D_gen] in the active block (an AO Coulomb/exchange build, or w, that
    matrix in the MO basis if known).  states: instead the vectors w_J sum_rs f_rs r^+ s
    Psi_J of these model states (w_J the weights of the averaged density).'''
    sp = ctx.sp
    eris = pt.quasi_canonical.eris
    c = eris.mo_coeff
    a = sp.slice('a')
    if w is None:
        ca = c[:, a]
        f = ca.conj().T @ eris.veff_fn(c @ numpy.asarray(d_gen).T @ c.conj().T) @ ca
    else:
        f = w[a, a]
    act = pt.active_space
    na = act.norb
    pairs = numpy.indices((na, na)).reshape(2, -1).T

    def one(j):
        ops = act.apply_strings(pt.civecs[:, j][None], pt.nelec, [((True, False), pairs)])[0]
        return numpy.einsum('r,rd->d', f.reshape(-1), ops)
    if states is None:
        return one(state)
    return numpy.array([pt.weights[j] * one(j) for j in states])


def ci_gradient_states(pt, rho, names=None):
    '''The vectors g[J] (determinants of the spinor sector) of all model states for the
    target sum_IJ rho_IJ h2_IJ of the union contraction, dT = 2 sum_J <dPsi_J|g_J>, of
    the classes names (default all nine; ccvv included: with several model states its
    first-order space is the model space).'''
    ctx = density.context(pt)
    sp = ctx.sp
    names = names or ('ccvv',) + tuple(density._LABELS)
    nstate = pt.civecs.shape[1]
    d_gen = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    g = numpy.zeros((nstate, pt.civecs.shape[0]), dtype=numpy.complex128)
    for name in names:
        if name == 'ccvv':
            _, dcc, dvv = density.ccvv(ctx, kramers=False)
            d_gen = d_gen + density.assemble(sp, {'cc': dcc, 'vv': dvv})
        else:
            _, dblk = density.class_reference(pt, ctx, name, rho=rho)
            for lab in ('cc', 'vv'):
                if lab in dblk:
                    d_gen = d_gen + density.assemble(sp, {lab: dblk[lab]})
        g += class_g(pt, ctx, name, rho=rho)
    g += fock_g(pt, ctx, d_gen.real, states=range(nstate))
    return g


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


def project(g, pt, route):
    '''dE/dx on the CI vector of the PySCF sector from g (dE = 2 <dPsi|g>).'''
    idx = sector_index(pt.active_space.norb // 2, route.nelec_ab)
    gx = numpy.asarray(g)[idx].real
    c = route.ci
    return 2 * (gx - numpy.dot(c, gx) * c)
