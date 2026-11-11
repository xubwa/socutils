'''
Unrelaxed (response) one-particle densities of NEVPT2: derivatives of the second-order
energy with respect to the one-electron Hamiltonian at fixed orbitals and fixed model
states, so that for h -> h + lambda O

    dE2 / dlambda = sum_pq D[p, q] O[p, q]

(finite field with frozen orbitals and CI vectors).  A perturbation of h changes the
generalized Fock matrix by O itself; the quasi-canonical orbitals and their energies
follow it, and the energy of a class, written without canonical orbitals, depends on
the core-core and virtual-virtual blocks of the Fock matrix through its denominators.

ccvv (state independent), with t[ij, ab] = v[ij, ab] / Delta, v = <ab||ij> and all
orderings of the indices (E = -1/4 sum |v|^2 / Delta):

    D_vv[a, a'] =  1/2 sum_{ij b} conj(t[ij, ab]) t[ij, a'b]
    D_cc[i, i'] = -1/2 sum_{j ab} conj(t[ij, ab]) t[i'j, ab]

in the quasi-canonical orbitals.  The sum runs over the core orbitals j in the outer
loop, so that the amplitudes t[ij] of all i are at hand for D_cc: every pair is formed
twice (twice the cost of the energy alone).
'''

import numpy

from socutils.zmr.lib import linalg as la
from socutils.zmr.uc import classes
from socutils.zmr.uc.classes import _slice


def ccvv(ctx, kramers=True):
    '''(E_ccvv, D_cc, D_vv) in the orbitals of ctx (zmr.uc.classes.Context of
    quasi-canonical orbitals, factorized integrals): the energy and its derivatives
    with respect to the core-core and virtual-virtual blocks of the Fock matrix,
    dE = sum dF_cc * D_cc + sum dF_vv * D_vv (elementwise).  Native kernel; with Kramers
    pairs (eris.time_reversal, kramers) the core orbitals j of one member of every pair
    only, the others from time reversal: E twice, D + Theta conj(D) Theta^+.'''
    from socutils.zmr.lib import kernels
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    if nc < 2 or nv < 2:
        return (0.0, numpy.zeros((nc, nc), dtype=numpy.complex128),
                numpy.zeros((nv, nv), dtype=numpy.complex128))
    jlist = th_c = th_v = None
    if kramers and classes.kramers_pairs(ctx) is not None:
        sp = ctx.sp
        tr = ctx.eris.time_reversal
        core = tr.subset(numpy.arange(sp.ncore))
        th_c = core.matrix()
        th_v = tr.subset(numpy.arange(sp.ncore + sp.nact, sp.nmo)).matrix()
        jlist = core.representatives()
    res = kernels.d_ccvv(ctx.eris.cd_block('vc'), ctx.eps_v, ctx.eps_c, jlist=jlist)
    if res is None:
        return ccvv_reference(ctx)
    e, dcc, dvv = res
    if jlist is not None:
        e = 2 * e
        dcc = dcc + th_c @ dcc.conj() @ th_c.conj().T
        dvv = dvv + th_v @ dvv.conj() @ th_v.conj().T
    return e, dcc, dvv


def ccvv_reference(ctx):
    '''ccvv as in ccvv, in Python (all pairs; for checks and without libzmr).'''
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    dcc = numpy.zeros((nc, nc), dtype=numpy.complex128)
    dvv = numpy.zeros((nv, nv), dtype=numpy.complex128)
    if nc < 2 or nv < 2:
        return 0.0, dcc, dvv
    bvc = ctx.eris.cd_block('vc')                  # (L, a, i)
    ev, ec = numpy.asarray(ctx.eps_v), numpy.asarray(ctx.eps_c)
    dab = ev[:, None] + ev[None, :]
    slices = [_slice(bvc, i) for i in range(nc)]   # B[:, :, i] as (L, nv)
    e = 0.0
    t = numpy.zeros((nc, nv, nv), dtype=numpy.complex128)
    for j in range(nc):
        for i in range(nc):
            if i == j:
                t[i] = 0.0
                continue
            m = la.gemm(slices[i], slices[j], 'T', 'N')     # (ai|bj)
            v = m - m.T                                       # <ab||ij>
            t[i] = v / (dab - ec[i] - ec[j])
            e -= 0.25 * float(numpy.vdot(v, t[i]).real)
        flat = t.reshape(nc, nv * nv)
        dcc -= 0.5 * la.gemm(flat.conj(), flat, 'N', 'T')
        for i in range(nc):
            if i != j:
                dvv += 0.5 * la.gemm(t[i].conj(), t[i], 'N', 'T')
    return e, dcc, dvv


# ----------------------------------------------------------------------
# classes with active operators (Python reference: all amplitudes of a class in memory)

def first_order_space(pt, pclass, state=0):
    '''(b, mu, t, s) of the class for one model state of a finished PC-NEVPT2 pt: the
    orthonormal basis b (D, nk) of the first-order space in the determinants of the
    target sector, its H_act energies mu, the projected functions
    t[k, X] = <b_k|O_X|Psi>, and the shift s = E0 - E_core of the state.'''
    from socutils.zmr.pc import spectral
    from socutils.zmr.uc import engine
    act, civecs = pt.active_space, pt.civecs
    ntarget = act.target(pt.nelec, pclass.dn)
    phi = engine.class_functions(pclass, act, civecs[:, [state]], pt.nelec)
    b, mu = spectral.contracted_space(phi.reshape(-1, phi.shape[2]),
                                      lambda q: act.apply_h(ntarget, q), lambda: None, pt.lindep)
    t = engine.project_functions(b, phi)[0]
    return b, mu, t, float(pt.shifts[state])


def _active_block(pt, pclass, b, g):
    '''D_aa[t, u] = sum_kk' g[k, k'] <b_k'|t^+ u|b_k> - tr(g) <Psi|t^+ u|Psi>: the
    derivative of the class energy with respect to the active block of h through
    H_act in the first-order space and the zeroth-order energy of the model state
    (g[k, k'] = sum_e c[e, k] conj(c[e, k'])).'''
    from socutils.zmr.active import rdm as active_rdm
    act = pt.active_space
    ntarget = act.target(pt.nelec, pclass.dn)
    dm = active_rdm.make_rdm(act, b, ntarget, 1)              # [k', k, t, u]
    d = numpy.einsum('kl,lktu->tu', g, dm)
    gamma = active_rdm.make_rdm(act, pt.civecs[:, [0]], pt.nelec, 1)[0, 0]
    return d - numpy.trace(g) * gamma


def cavv_reference(pt, ctx, state=0):
    '''(E, D_cc, D_aa, D_vv) of cavv for one model state, gamma convention
    (dE = sum O[p, q] D[p, q] for Hermitian O): amplitudes
    c[i, ab, k] = sum_t v[i ab, t] t[k, t] / (mu_k + delta - s).'''
    pclass = [p for p in pt.perturber_classes() if p.name == 'cavv'][0]
    b, mu, t, s = first_order_space(pt, pclass, state)
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    aa, bb = numpy.triu_indices(nv, 1)
    nk = len(mu)
    c = numpy.zeros((nc, nv, nv, nk), dtype=numpy.complex128)
    e = 0.0
    for i, (delta, v) in enumerate(pclass.blocks(ctx)):
        u = v @ t.T                                           # (ab, k)
        ci = u / (delta[:, None] + mu[None, :] - s)
        e -= float(numpy.vdot(u, ci).real)
        c[i, aa, bb] = ci
        c[i, bb, aa] = -ci
    g = 0.5 * numpy.einsum('iabk,iabl->kl', c, c.conj())
    dcc = -0.5 * numpy.einsum('iabk,jabk->ij', c.conj(), c)
    dvv = numpy.einsum('iabk,idbk->ad', c.conj(), c)
    return e, dcc, _active_block(pt, pclass, b, g), dvv


def ccav_reference(pt, ctx, state=0):
    '''(E, D_cc, D_aa, D_vv) of ccav for one model state:
    c[ij, a, k] = sum_t v[ij a, t] t[k, t] / (mu_k + delta - s).'''
    pclass = [p for p in pt.perturber_classes() if p.name == 'ccav'][0]
    b, mu, t, s = first_order_space(pt, pclass, state)
    nc, nv = ctx.sp.ncore, ctx.sp.nvirt
    ii, jj = numpy.triu_indices(nc, 1)
    nk = len(mu)
    c = numpy.zeros((nc, nc, nv, nk), dtype=numpy.complex128)
    e = 0.0
    for (i, j), (delta, v) in zip(zip(ii, jj), pclass.blocks(ctx)):
        u = v @ t.T                                           # (a, k)
        cij = u / (delta[:, None] + mu[None, :] - s)
        e -= float(numpy.vdot(u, cij).real)
        c[i, j] = cij
        c[j, i] = -cij
    g = 0.5 * numpy.einsum('ijak,ijal->kl', c, c.conj())
    dcc = -numpy.einsum('ijak,ljak->il', c.conj(), c)
    dvv = 0.5 * numpy.einsum('ijak,ijdk->ad', c.conj(), c)
    return e, dcc, _active_block(pt, pclass, b, g), dvv
