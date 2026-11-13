'''
Relaxed one-particle density of the state-specific spin-free PC-NEVPT2 of a singlet state of
a PySCF CASSCF (single-state or state-averaged), through the Lagrangian:

    D = D_state + D_unrelaxed + D_z,   J z = -dE/dx,

E = E_state(CASSCF) + E2(NEVPT2) of the state, x the parameters of PySCF's second-order
CASSCF (mo' = mo expm(unpack(x_orb)), c_k' = (c_k + x_k) / |c_k + x_k|), J its exact
Hessian (applied to vectors, newton_casscf.gen_g_hop), D_z the derivative of the
averaged CASSCF density along z (zmr.reference.relaxed_fd.lagrangian_density).
dE2/dx: the orbital part from zmr.uc.gradient, the CI part from
zmr.reference.ci_gradient; the unrelaxed density from zmr.uc.density (through the
spin-orbital route of zmr.nr.density).

    dm = relaxed_rdm1(mc, cderi, state=0)       AO, spin-summed: <O> = tr(O dm)
'''

import numpy
import scipy.sparse.linalg
from pyscf.mcscf import newton_casscf

from socutils.zmr.nr import density as nr_density
from socutils.zmr.reference import ci_gradient as cg
from socutils.zmr.reference import relaxed_fd as rf
from socutils.zmr.uc import density as uc_density
from socutils.zmr.uc import gradient as uc_gradient


def _null_space(mc):
    '''Orthonormal columns spanning the zero modes of the Jacobian of the gradient: the
    norm of every CI vector.  (The rotations among the averaged states are not zero modes
    of the Jacobian: the CI gradients w_I (H - E_I) c_I couple them with the symmetric
    mixings c_I + t c_J, c_J + t c_I; together they fix the model states as eigenvectors.)'''
    ci = rf._ci_list(mc)
    n = rf.ngorb(mc)
    sizes = [c.size for c in ci]
    offs = n + numpy.concatenate([[0], numpy.cumsum(sizes)])
    vecs = numpy.zeros((offs[-1], len(ci)))
    for k, c in enumerate(ci):
        vecs[offs[k]:offs[k + 1], k] = c.ravel() / numpy.linalg.norm(c)
    return vecs


def zvector(mc, rhs, tol=1e-10, maxiter=500):
    '''z with J z = -rhs (J symmetric at the CASSCF solution) by preconditioned MINRES on
    the complement of the zero modes.'''
    ci = rf._ci_list(mc)
    arg = ci if len(ci) > 1 else ci[0]
    g, _, hop, hdiag = newton_casscf.gen_g_hop(mc, mc.mo_coeff, arg, mc.ao2mo(mc.mo_coeff))[:4]
    hdiag = numpy.asarray(hdiag, dtype=float)
    q = _null_space(mc)

    def proj(x):
        return x - q @ (q.T @ x)
    n = len(rhs)
    a = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda x: proj(numpy.asarray(hop(proj(x))).real))
    prec = numpy.where(abs(hdiag) > 1e-8, 1.0 / numpy.maximum(abs(hdiag), 1e-8), 1.0)
    m = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda x: proj(prec * proj(x)))
    b = proj(-numpy.asarray(rhs, dtype=float))
    z, info = scipy.sparse.linalg.minres(a, b, M=m, rtol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError('Z-vector equations not converged (minres info %d)' % info)
    return proj(z)


def relaxed_rdm1(mc, cderi, state=0, max_memory=4000, return_parts=False, timings=None):
    '''AO density of the state `state` of mc whose expectation values are the derivatives
    of E_state(CASSCF) + E2(state-specific PC-NEVPT2) (all orbitals correlated).
    return_parts: also (D_state, D_unrelaxed, D_z).  timings: dict that receives the
    seconds of every step.'''
    import time
    tim = {} if timings is None else timings
    t0 = time.time()

    def lap(name):
        nonlocal t0
        tim[name] = tim.get(name, 0.0) + time.time() - t0
        t0 = time.time()
    ci = rf._ci_list(mc)
    route = nr_density.spin_orbital_route(mc, cderi, ci=ci[state])
    pt = route.pt(max_memory)
    lap('spin-orbital NEVPT2 energy')
    # unrelaxed density, orbital gradient and CI gradient of E2 in one pass
    out = uc_gradient.derivatives(pt, state=0, max_memory=max_memory)
    u = out['u']
    d2 = route.spin_sum(u.conj() @ out['d'] @ u.T).real
    g_orb = route.spin_sum(numpy.conj(u) @ out['grad'] @ u.T).real
    lap('E2 derivatives')
    rhs = rf.state_gradient(mc, state) if len(ci) > 1 else numpy.zeros(
        rf.ngorb(mc) + sum(c.size for c in ci))
    rhs[:rf.ngorb(mc)] += mc.pack_uniq_var(g_orb - g_orb.T)
    rhs += rf.ci_block(mc, state, cg.project(out['g_ci'], pt, route))
    z = zvector(mc, rhs)
    lap('Z-vector')
    c = mc.mo_coeff
    d_state = rf.state_density(mc, state)
    d_unrel = c @ d2.T @ c.T
    d_z = rf.lagrangian_density(mc, z)
    dm = d_state + d_unrel + d_z
    lap('Lagrangian density')
    if return_parts:
        return dm, (d_state, d_unrel, d_z)
    return dm


# ----------------------------------------------------------------------
# levels of the multi-state (QD) PC-NEVPT2 on a state-averaged CASSCF

def _transition_rdms(mc, rho):
    '''sum_IJ rho_IJ <c_I|E|c_J> (1- and 2-RDM of PySCF, active orbitals).'''
    from pyscf.fci import direct_spin1
    ci = rf._ci_list(mc)
    ncas, nelecas = mc.ncas, mc.nelecas
    dm1 = numpy.zeros((ncas, ncas))
    dm2 = numpy.zeros((ncas,) * 4)
    for i, ci_i in enumerate(ci):
        for j, ci_j in enumerate(ci):
            if abs(rho[i, j]) > 0:
                d1, d2 = direct_spin1.trans_rdm12(ci_i, ci_j, ncas, nelecas)
                dm1 += rho[i, j] * d1
                dm2 += rho[i, j] * d2
    return dm1, dm2


def model_gradient(mc, rho):
    '''dT/dx of T = sum_IJ rho_IJ hmodel_IJ, hmodel_IJ = <c_I|H_act|c_J> + E_core delta_IJ
    (rho real symmetric with trace 1), in the parameters of the state-averaged CASSCF.'''
    from pyscf.fci import direct_spin1
    from pyscf.mcscf import mc1step
    mo, ci = mc.mo_coeff, rf._ci_list(mc)
    ncas, nelecas = mc.ncas, mc.nelecas
    dm1, dm2 = _transition_rdms(mc, rho)
    # mc1step returns half the orbital gradient of newton_casscf's parameters
    g_orb = 2 * numpy.asarray(mc1step.gen_g_hop(mc, mo, 1, dm1, dm2, mc.ao2mo(mo))[0])
    h1, _ = mc.get_h1eff(mo)
    h2e = direct_spin1.absorb_h1e(h1, mc.get_h2eff(mo), ncas, nelecas, .5)
    hc = [direct_spin1.contract_2e(h2e, c, ncas, nelecas).ravel() for c in ci]
    g_ci = []
    for i, c in enumerate(ci):
        g = sum(rho[i, j] * hc[j] for j in range(len(ci)))
        g_ci.append(2 * (g - numpy.dot(c.ravel(), g) * c.ravel()))
    return numpy.concatenate([g_orb] + g_ci)


def _ci_slices(mc):
    ci = rf._ci_list(mc)
    p = rf.ngorb(mc)
    out = []
    for c in ci:
        out.append(slice(p, p + c.size))
        p += c.size
    return out


def qd_derivatives(pt, rho):
    '''{'d', 'grad', 'g', 'u'} of sum_IJ rho_IJ h2_IJ of a multi-state PC-NEVPT2 (union
    contraction): the unrelaxed density and the orbital gradient in the quasi-canonical
    orbitals, the vectors g[J] of the CI gradient (Python reference: all amplitudes of a
    class in memory; zmr.uc.gradient.derivatives with rho is the scalable form).'''
    from socutils.zmr.reference import orbital_gradient as og
    ctx = uc_density.context(pt)
    sp = ctx.sp
    d = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    _, dcc, dvv = uc_density.ccvv(ctx, kramers=False)
    d += uc_density.assemble(sp, {'cc': dcc, 'vv': dvv, 'aa': uc_density.ccvv_active(pt, ctx, rho)[0]})
    for name in uc_density._LABELS:
        d += uc_density.assemble(sp, uc_density.class_reference(pt, ctx, name, rho=rho)[1])
    grad, u = og.gradient(pt, rho=rho)
    g = cg.ci_gradient_states(pt, rho)
    return {'d': d, 'grad': grad, 'g': g, 'u': u}


def relaxed_rdm1_levels(mc, cderi, levels=None, max_memory=4000, return_parts=False):
    '''AO densities of levels of the multi-state PC-NEVPT2 (union contraction, the
    zeroth-order energy of every state) on all states of the state-averaged CASSCF mc,
    whose expectation values are the derivatives of the eigenvalues `levels` (ascending;
    default all) of heff = hmodel + h2, through the Lagrangian of the averaged CASSCF.  The
    derivatives of all levels come from one pass over the classes (the amplitudes of the
    model states are those of every level; only the weights rho = c_k c_k^+ differ), one
    Z-vector per level.  Returns a list; return_parts: also (D_model, D_unrelaxed, D_z) and
    the eigenvalue of every level.'''
    from socutils.zmr.nr.active import sector_index
    ci = rf._ci_list(mc)
    route = nr_density.spin_orbital_route(mc, cderi, ci=ci)
    pt = route.pt(max_memory)
    e, vec = numpy.linalg.eigh(pt.heff)
    levels = list(range(len(e))) if levels is None else list(levels)
    rhos = []
    for k in levels:
        ck = vec[:, k]
        ck = ck * abs(ck[abs(ck).argmax()]) / ck[abs(ck).argmax()]
        if abs(ck.imag).max() > 1e-10 * abs(ck).max():
            raise NotImplementedError('real model states only')
        rhos.append(numpy.outer(ck.real, ck.real))
    outs = uc_gradient.derivatives(pt, max_memory=max_memory, rho=numpy.array(rhos))
    idx = sector_index(pt.active_space.norb // 2, route.nelec_ab)
    mo = mc.mo_coeff
    nmo = mo.shape[1]
    a = slice(mc.ncore, mc.ncore + mc.ncas)
    res = []
    for k, rho, out in zip(levels, rhos, outs):
        u = out['u']
        d2 = route.spin_sum(u.conj() @ out['d'] @ u.T).real
        g_orb = route.spin_sum(numpy.conj(u) @ out['grad'] @ u.T).real
        rhs = numpy.zeros(rf.ngorb(mc) + sum(c.size for c in ci))
        rhs[:rf.ngorb(mc)] = mc.pack_uniq_var(g_orb - g_orb.T)
        for j, sl in enumerate(_ci_slices(mc)):
            gx = out['g_ci'][j][idx].real
            c = ci[j].ravel()
            rhs[sl] = 2 * (gx - numpy.dot(c, gx) * c)
        rhs += model_gradient(mc, rho)
        z = zvector(mc, rhs)
        gam = numpy.zeros((nmo, nmo))
        gam[:mc.ncore, :mc.ncore] = 2 * numpy.eye(mc.ncore)
        gam[a, a] = _transition_rdms(mc, rho)[0]
        d_model = mo @ gam @ mo.T
        d_unrel = mo @ d2.T @ mo.T
        d_z = rf.lagrangian_density(mc, z)
        dm = d_model + d_unrel + d_z
        res.append((dm, (d_model, d_unrel, d_z), float(e[k])) if return_parts else dm)
    return res


def relaxed_rdm1_level(mc, cderi, level=0, max_memory=4000, return_parts=False):
    '''relaxed_rdm1_levels of one level.'''
    return relaxed_rdm1_levels(mc, cderi, [level], max_memory, return_parts)[0]
