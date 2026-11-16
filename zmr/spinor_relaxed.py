'''
Relaxed one-particle density of the state-specific PC-NEVPT2 of a state of a state-averaged
spinor (two-component) CASSCF of socutils, through the Lagrangian

    D = D_k + D_unrelaxed + D_z,   J z = -G,

G the derivative of E_k(CASSCF) + E2(PC-NEVPT2 of state k) with respect to the orbital and
CI parameters of socutils.mcscf.zmc_response (orbitals C exp(K), CI vectors (c + x)/|c + x|),
J the exact orbital + CI Hessian of the averaged CASSCF energy there.  The derivatives of
E2 come from one pass of zmr.uc.gradient.derivatives (complex orbitals; orbital_gradient
and ci_gradient here are the Python references); the zmr Hamiltonian is built from the
same three-index integrals as the CASSCF (cderi), in which the CI vectors of socutils are
those of the zmr spinor sector.

    dm = relaxed_rdm1(mc, mf, mo, ci, weights, state=0, cderi=mf.with_df._cderi)
    <O> = tr(O dm)  for a spinor AO operator O (dm in the j-spinor AO basis)
'''

import numpy

from socutils.mcscf import zmc_response
from socutils.zmr.interface import socutils_x2c
from socutils.zmr.pc.nevpt2 import PCNEVPT2


def build_pt(mf, mc, mo, civecs, cderi, weights=None, max_memory=4000):
    '''The spinor PC-NEVPT2 (all orbitals correlated) of the CI vectors civecs (columns).'''
    eris = socutils_x2c.mo_eris(mf, mf.mol, mo, 0, mc.ncore, mc.ncas, cderi=cderi)
    pt = PCNEVPT2(eris, mc.nelecas, civecs=civecs, weights=weights, e0='state')
    pt.verbose = 0
    pt.max_memory = max(pt.max_memory, max_memory)
    pt.kernel()
    return pt


def orbital_gradient(mc, pt, rho=None):
    '''G_orb of E2 (zmc_response layout: dE2 = Re<G, x>) in the orbitals pt was built on.'''
    from socutils.zmr.reference import orbital_gradient as og
    m_qc, u = og.gradient_general(pt, rho=rho)
    m = numpy.conj(u) @ m_qc @ u.T
    nmo = m.shape[0]
    p, q = numpy.nonzero(mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen))
    return numpy.conj(m[p, q]) - m[q, p]


def ci_gradient(pt):
    '''g of E2 for the single model state of pt (dE2 = 2 Re<dPsi|g>).'''
    from socutils.zmr.reference import ci_gradient as cg
    from socutils.zmr.uc import density
    ctx = density.context(pt)
    sp = ctx.sp
    _, dcc, dvv = density.ccvv(ctx, kramers=False)
    d_gen = density.assemble(sp, {'cc': dcc, 'vv': dvv})
    g = numpy.zeros(pt.civecs.shape[0], dtype=complex)
    for name in density._LABELS:
        blk = density.class_reference(pt, ctx, name)[1]
        d_gen = d_gen + density.assemble(sp, {lab: blk[lab] for lab in ('cc', 'vv') if lab in blk})
        g += cg.class_g(pt, ctx, name)
    return g + cg.fock_g(pt, ctx, d_gen)


def relaxed_rdm1(mc, mf, mo, ci, weights, state=0, cderi=None, max_memory=4000,
                 return_parts=False):
    '''AO (j-spinor) density of state `state` of the averaged CASSCF (orbitals mo, CI
    vectors ci) whose expectation values are the derivatives of E_k(CASSCF) +
    E2(state-specific PC-NEVPT2), all orbitals correlated.  return_parts: also
    (D_k, D_unrelaxed, D_z).'''
    cderi = mf.with_df._cderi if cderi is None else cderi
    ci = [numpy.asarray(c, dtype=complex).ravel() for c in ci]
    resp = zmc_response.SAResponse(mc, mo, ci, weights, cderi=cderi)
    pt = build_pt(mf, mc, mo, ci[state][:, None], cderi, max_memory=max_memory)
    rhs = zmc_response.state_gradient(resp, state)
    # E2: unrelaxed density, orbital and CI gradients in one pass (zmr.uc.gradient)
    from socutils.zmr.uc import gradient
    out = gradient.derivatives(pt, max_memory=max_memory)
    u = numpy.asarray(out['u'])
    m = numpy.conj(u) @ out['grad'] @ u.T            # input orbitals: phi_qc = phi u
    p, q = numpy.nonzero(mc.uniq_var_indices(m.shape[0], mc.ncore, mc.ncas, mc.frozen))
    rhs[:resp.n_orb] += numpy.conj(m[p, q]) - m[q, p]
    g = out['g_ci']
    c = ci[state]
    parts = resp.split(rhs)
    parts[1 + state] = parts[1 + state] + 2 * (g - numpy.vdot(c, g).real * c)
    rhs = resp.join(parts)
    z = zmc_response.zvector(resp, rhs)
    d_k = zmc_response.state_density(resp, state)
    d2 = numpy.conj(u) @ out['d'] @ u.T               # D[p, q] = dE2/dO[p, q], input orbitals
    d_unrel = mo @ d2.T @ mo.conj().T
    d_z = zmc_response.lagrangian_density(resp, z)
    dm = d_k + d_unrel + d_z
    if return_parts:
        return dm, (d_k, d_unrel, d_z)
    return dm


# ----------------------------------------------------------------------
# levels of the multi-state PC-NEVPT2 (time-even properties)

def level_weights(heff, levels, tol=1e-8):
    '''rho (L, J, J): for every level the average of conj(c) c^T over the eigenvectors of
    heff degenerate with it (|e - e_k| < tol: a Kramers pair, or any exactly degenerate
    multiplet), invariant to rotations within the multiplet.  The derivative of the
    level for a perturbation that keeps the degeneracy (time-even operators); a
    time-odd one splits it and needs the matrix within the multiplet.'''
    e, vec = numpy.linalg.eigh(heff)
    out = []
    for k in levels:
        group = numpy.nonzero(abs(e - e[k]) < tol)[0]
        rho = sum(numpy.outer(vec[:, n].conj(), vec[:, n]) for n in group) / len(group)
        out.append(rho)
    return numpy.array(out), e


def relaxed_rdm1_levels(mc, mf, mo, ci, weights, levels=None, cderi=None, max_memory=4000,
                        degeneracy_tol=1e-8, return_parts=False):
    '''AO densities of levels of the multi-state PC-NEVPT2 (union contraction, the
    zeroth-order energy of every state) on all states of the averaged spinor CASSCF, whose
    expectation values are the derivatives of the eigenvalues `levels` of heff (default
    all; a degenerate multiplet averaged, level_weights) for time-even operators.  One
    pass over the classes for all levels, one Z-vector each; the rotations among
    degenerate averaged states are projected out of the Z-vector equations.  Returns a
    list; return_parts: also (D_model, D_unrelaxed, D_z) and the eigenvalue.'''
    from socutils.zmr.uc import gradient
    cderi = mf.with_df._cderi if cderi is None else cderi
    ci = [numpy.asarray(c, dtype=complex).ravel() for c in ci]
    resp = zmc_response.SAResponse(mc, mo, ci, weights, cderi=cderi)
    pt = build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights, max_memory=max_memory)
    levels = list(range(len(ci))) if levels is None else list(levels)
    rhos, e = level_weights(pt.heff, levels, degeneracy_tol)
    outs = gradient.derivatives(pt, max_memory=max_memory, rho=rhos)
    p, q = numpy.nonzero(mc.uniq_var_indices(mo.shape[1], mc.ncore, mc.ncas, mc.frozen))
    null = zmc_response.degenerate_rotations(resp)
    res = []
    for k, rho, out in zip(levels, rhos, outs):
        u = numpy.asarray(out['u'])
        m = numpy.conj(u) @ out['grad'] @ u.T
        parts = resp.split(zmc_response.model_gradient(resp, rho))
        parts[0] = parts[0] + numpy.conj(m[p, q]) - m[q, p]
        for j, c in enumerate(ci):
            g = out['g_ci'][j]
            parts[1 + j] = parts[1 + j] + 2 * (g - numpy.vdot(c, g).real * c)
        z = zmc_response.zvector(resp, resp.join(parts), extra_null=null)
        d_model = zmc_response.model_density(resp, rho)
        d2 = numpy.conj(u) @ out['d'] @ u.T
        d_unrel = mo @ d2.T @ mo.conj().T
        d_z = zmc_response.lagrangian_density(resp, z)
        dm = d_model + d_unrel + d_z
        res.append((dm, (d_model, d_unrel, d_z), float(e[k])) if return_parts else dm)
    return res
