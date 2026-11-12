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
    '''Orthonormal columns spanning the zero modes of the averaged energy: the norm of
    every CI vector and the rotations among the averaged states.'''
    ci = rf._ci_list(mc)
    n = rf.ngorb(mc)
    sizes = [c.size for c in ci]
    offs = n + numpy.concatenate([[0], numpy.cumsum(sizes)])
    ntot = offs[-1]
    vecs = []
    for k, c in enumerate(ci):
        v = numpy.zeros(ntot)
        v[offs[k]:offs[k + 1]] = c.ravel()
        vecs.append(v)
        for l in range(k):
            v = numpy.zeros(ntot)
            v[offs[k]:offs[k + 1]] = ci[l].ravel()
            v[offs[l]:offs[l + 1]] = -c.ravel()
            vecs.append(v)
    q, _ = numpy.linalg.qr(numpy.array(vecs).T)
    return q


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


def relaxed_rdm1(mc, cderi, state=0, max_memory=4000, return_parts=False):
    '''AO density of the state `state` of mc whose expectation values are the derivatives
    of E_state(CASSCF) + E2(state-specific PC-NEVPT2) (all orbitals correlated).
    return_parts: also (D_state, D_unrelaxed, D_z).'''
    ci = rf._ci_list(mc)
    route = nr_density.spin_orbital_route(mc, cderi, ci=ci[state])
    pt = route.pt(max_memory)
    # unrelaxed density
    _, d2so = uc_density.make_rdm1(pt, max_memory=max_memory)
    d2 = route.spin_sum(d2so).real
    # dE/dx
    g_qc, u = uc_gradient.gradient(pt, max_memory=max_memory)
    g_orb = route.spin_sum(numpy.conj(u) @ g_qc @ u.T).real
    rhs = rf.state_gradient(mc, state) if len(ci) > 1 else numpy.zeros(
        rf.ngorb(mc) + sum(c.size for c in ci))
    rhs[:rf.ngorb(mc)] += mc.pack_uniq_var(g_orb - g_orb.T)
    rhs += rf.ci_block(mc, state, cg.ci_gradient(pt, route))
    z = zvector(mc, rhs)
    c = mc.mo_coeff
    d_state = rf.state_density(mc, state)
    d_unrel = c @ d2.T @ c.T
    d_z = rf.lagrangian_density(mc, z)
    dm = d_state + d_unrel + d_z
    if return_parts:
        return dm, (d_state, d_unrel, d_z)
    return dm
