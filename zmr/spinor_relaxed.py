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


def build_pt(mf, mc, mo, civecs, cderi, weights=None, max_memory=4000, degeneracy_tol=None):
    '''The spinor PC-NEVPT2 (all orbitals correlated) of the CI vectors civecs (columns);
    degeneracy_tol: NEVPT2Base.degeneracy_tol (states sharing their E0).'''
    eris = socutils_x2c.mo_eris(mf, mf.mol, mo, 0, mc.ncore, mc.ncas, cderi=cderi)
    pt = PCNEVPT2(eris, mc.nelecas, civecs=civecs, weights=weights, e0='state')
    if degeneracy_tol is not None:
        pt.degeneracy_tol = degeneracy_tol
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


def _relaxed_for_rhos(mc, mf, mo, ci, weights, rhos, cderi, max_memory):
    '''(AO densities D[n], parts, pt): the relaxed densities of the targets
    T_n = sum_IJ rhos[n]_IJ heff_IJ (rhos Hermitian) of the multi-state PC-NEVPT2 on all
    states of the averaged spinor CASSCF: dT_n/dlambda = tr(O D[n]).  One pass over the
    classes, one Z-vector per target (rotations among degenerate averaged states
    projected out).'''
    from socutils.zmr.uc import gradient
    ci = [numpy.asarray(c, dtype=complex).ravel() for c in ci]
    resp = zmc_response.SAResponse(mc, mo, ci, weights, cderi=cderi)
    pt = build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights, max_memory=max_memory)
    rhos = numpy.asarray(rhos)
    outs = gradient.derivatives(pt, max_memory=max_memory, rho=rhos)
    p, q = numpy.nonzero(mc.uniq_var_indices(mo.shape[1], mc.ncore, mc.ncas, mc.frozen))
    null = zmc_response.degenerate_rotations(resp)
    dms, parts_all = [], []
    for rho, out in zip(rhos, outs):
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
        dms.append(d_model + d_unrel + d_z)
        parts_all.append((d_model, d_unrel, d_z))
    return dms, parts_all, pt


def relaxed_rdm1_levels(mc, mf, mo, ci, weights, levels=None, cderi=None, max_memory=4000,
                        degeneracy_tol=1e-8, return_parts=False):
    '''AO densities of levels of the multi-state PC-NEVPT2 (union contraction, the
    zeroth-order energy of every state) on all states of the averaged spinor CASSCF, whose
    expectation values are the derivatives of the eigenvalues `levels` of heff (default
    all; a degenerate multiplet averaged, level_weights) for time-even operators.  Returns
    a list; return_parts: also (D_model, D_unrelaxed, D_z) and the eigenvalue.'''
    cderi = mf.with_df._cderi if cderi is None else cderi
    probe = build_pt(mf, mc, mo, numpy.array([numpy.asarray(c).ravel() for c in ci]).T, cderi,
                     weights=weights, max_memory=max_memory)
    levels = list(range(len(ci))) if levels is None else list(levels)
    rhos, e = level_weights(probe.heff, levels, degeneracy_tol)
    dms, parts, _ = _relaxed_for_rhos(mc, mf, mo, ci, weights, rhos, cderi, max_memory)
    if return_parts:
        return [(d, x, float(e[k])) for d, x, k in zip(dms, parts, levels)]
    return dms


def relaxed_level_matrix(mc, mf, mo, ci, weights, level=0, cderi=None, max_memory=4000,
                         degeneracy_tol=1e-8, return_unrelaxed=False):
    '''(e, D): the degenerate multiplet of level `level` of heff (eigenvectors c_m, energy
    e) and the relaxed transition densities D[m][n] (AO) with

        d<c_m|heff|c_n>/dlambda = tr(O D[m][n])   for h -> h + lambda O,

    the matrix whose eigenvalues are the first-order splittings of the multiplet by any
    one-electron perturbation (time-odd ones split a Kramers pair: magnetic properties).
    T_mn is complex: its real and imaginary parts, sum rho_IJ heff_IJ with the Hermitian
    rho = (r + r^+)/2 and (r - r^+)/2i, r_IJ = conj(c_mI) c_nJ, are two targets.
    return_unrelaxed: also the matrix at fixed orbitals and CI vectors.'''
    cderi = mf.with_df._cderi if cderi is None else cderi
    probe = build_pt(mf, mc, mo, numpy.array([numpy.asarray(c).ravel() for c in ci]).T, cderi,
                     weights=weights, max_memory=max_memory)
    e, vec = numpy.linalg.eigh(probe.heff)
    group = numpy.nonzero(abs(e - e[level]) < degeneracy_tol)[0]
    keys, rhos = [], []
    for a, m in enumerate(group):
        for b, n in enumerate(group):
            if b < a:
                continue
            r = numpy.outer(vec[:, m].conj(), vec[:, n])
            keys.append((a, b, 're'))
            rhos.append(0.5 * (r + r.conj().T))
            if b != a:
                keys.append((a, b, 'im'))
                rhos.append((r - r.conj().T) / 2j)
    dms, parts, _ = _relaxed_for_rhos(mc, mf, mo, ci, weights, rhos, cderi, max_memory)

    def matrix(ds):
        k = len(group)
        nao = ds[0].shape[0]
        out = [[numpy.zeros((nao, nao), dtype=complex) for _ in range(k)] for _ in range(k)]
        for (a, b, part), d in zip(keys, ds):
            if part == 're':
                out[a][b] = out[a][b] + d
                if b != a:
                    out[b][a] = out[b][a] + d
            else:
                out[a][b] = out[a][b] + 1j * d
                out[b][a] = out[b][a] - 1j * d
        return out
    if return_unrelaxed:
        return float(e[level]), matrix(dms), matrix([x[0] + x[1] for x in parts])
    return float(e[level]), matrix(dms)


# ----------------------------------------------------------------------
# magnetic properties of a Kramers doublet

def magnetic_moment_ops(mf, origin=None, utm=True):
    '''(3, n2c, n2c) the X2C picture-changed operator 1/2 (L + g_e S) of socutils.prop.gfac
    (Zeeman term B . O in atomic units, mu_B = 1/2), gauge origin `origin` (default the
    centre of nuclear charge).'''
    from socutils.prop.gfac import spinor_hf as gfac
    mol = mf.mol
    if origin is None:
        q = mol.atom_charges()
        origin = (q[:, None] * mol.atom_coords()).sum(0) / q.sum()
    with mol.with_common_orig(origin):
        return gfac.int_gfac_2c(mf, utm=utm)


def g_tensor_from_block(m):
    '''g = 2 P of the polar decomposition of the pseudo-spin tensor of a (3, 2, 2) block
    M_a[m, n] = <m|O_a|n> of a Kramers doublet (socutils.prop.observables): principal
    values non-negative, the overall sign not claimed.'''
    from socutils.prop import observables
    p, _ = observables.pseudo_spin_frame(observables.pseudo_spin_tensor(m))
    return 2 * p


def relaxed_g_tensor(mc, mf, mo, ci, weights, level=0, cderi=None, max_memory=4000,
                     origin=None, ops=None):
    '''g tensors of the Kramers doublet of level `level` of the multi-state PC-NEVPT2:
    {'model', 'unrelaxed', 'relaxed'}, from the matrices <m|O_a|n> of the CASSCF-model
    part, the derivative at fixed orbitals and CI vectors, and the relaxed one
    (relaxed_level_matrix).  ops: the magnetic-moment operators (default
    magnetic_moment_ops).'''
    if ops is None:
        ops = magnetic_moment_ops(mf, origin)
    cderi = mf.with_df._cderi if cderi is None else cderi
    probe = build_pt(mf, mc, mo, numpy.array([numpy.asarray(c).ravel() for c in ci]).T, cderi,
                     weights=weights, max_memory=max_memory)
    e, vec = numpy.linalg.eigh(probe.heff)
    group = numpy.nonzero(abs(e - e[level]) < 1e-8)[0]
    if len(group) != 2:
        raise ValueError('level %d is %d-fold degenerate, not a Kramers doublet'
                         % (level, len(group)))
    e_k, d_rel, d_un = relaxed_level_matrix(mc, mf, mo, ci, weights, level=level, cderi=cderi,
                                            max_memory=max_memory, return_unrelaxed=True)
    resp = zmc_response.SAResponse(mc, mo, ci, weights, cderi=cderi)
    d_model = [[None, None], [None, None]]
    for a, m in enumerate(group):
        for b, n in enumerate(group):
            r = numpy.outer(vec[:, m].conj(), vec[:, n])
            d_model[a][b] = (zmc_response.model_density(resp, 0.5 * (r + r.conj().T))
                             + 1j * zmc_response.model_density(resp, (r - r.conj().T) / 2j))

    def block(d):
        return numpy.array([[[numpy.einsum('ij,ji', o, d[a][b]) for b in range(2)]
                             for a in range(2)] for o in ops])
    return {'energy': e_k,
            'model': g_tensor_from_block(block(d_model)),
            'unrelaxed': g_tensor_from_block(block(d_un)),
            'relaxed': g_tensor_from_block(block(d_rel))}
