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


def build_pt(mf, mc, mo, civecs, cderi, weights=None, max_memory=4000, degeneracy_tol=None,
             nfrozen=0):
    '''The spinor PC-NEVPT2 of the CI vectors civecs (columns), the first nfrozen orbitals of
    mo frozen (folded into the integrals; frozen_partition picks them); degeneracy_tol:
    NEVPT2Base.degeneracy_tol (states sharing their E0).'''
    eris = socutils_x2c.mo_eris(mf, mf.mol, mo, nfrozen, mc.ncore - nfrozen, mc.ncas,
                                cderi=cderi)
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


def frozen_partition(mf, mc, mo, ci, weights, nfrozen, cderi=None):
    '''(mo', eps): the core orbitals of mo replaced by the eigenvectors of the core block of
    the averaged generalized Fock matrix F = h + (J - K)[core + active gamma_SA], ascending
    (the averaged CASSCF energy does not change); the first nfrozen of them are the frozen
    core of the NEVPT2.  A degeneracy across the boundary (a Kramers pair split) is an
    error.'''
    ncore = mc.ncore
    if nfrozen == 0:
        return mo, None
    cderi = mf.with_df._cderi if cderi is None else cderi
    from socutils.zmr.active import rdm as active_rdm
    from socutils.zmr.active.sectors import ActiveSpace
    na = mc.ncas
    cas = ActiveSpace(numpy.zeros((na, na)), numpy.zeros((na,) * 4), 0.0)
    cmat = numpy.array([numpy.asarray(c, dtype=complex).ravel() for c in ci]).T
    d1 = active_rdm.make_rdm(cas, cmat, mc.nelecas, 1)
    gamma = sum(w * d1[i, i] for i, w in enumerate(weights))
    eris0 = socutils_x2c.mo_eris(mf, mf.mol, mo, 0, ncore, mc.ncas, cderi=cderi)
    f = eris0.generalized_fock(gamma)
    e, v = numpy.linalg.eigh(0.5 * (f[:ncore, :ncore] + f[:ncore, :ncore].conj().T))
    if nfrozen < ncore and e[nfrozen] - e[nfrozen - 1] < 1e-6:
        raise ValueError('frozen core: no gap after %d core orbitals (%.2e)'
                         % (nfrozen, e[nfrozen] - e[nfrozen - 1]))
    mo = numpy.array(mo, copy=True)
    mo[:, :ncore] = mo[:, :ncore] @ v
    return mo, e


def _e2_terms(mc, mf, mo, resp, out, nfrozen, states):
    '''(G_orb, {J: G_J}, D_unrelaxed (AO)) of E2 from one zmr.uc.gradient.derivatives result
    out of a PC-NEVPT2 of the model states `states` (indices into resp.ci): the gradient in
    the parameters of zmc_response and the unrelaxed density in the orbitals mo.  With a
    frozen core (mo from frozen_partition) also the multipliers zeta_fc of the condition
    F_fc = 0 (f frozen, c correlated core, F the averaged generalized Fock matrix) that fixes
    the frozen space: zeta_fc = -G_fc / (e_f - e_c), and the target Re sum conj(zeta) F_fc
    adds its orbital and CI gradients (through gamma_SA) and its density.'''
    from socutils.zmr.reference import orbital_gradient as og
    nf = nfrozen
    n_all = mo.shape[1]
    u = numpy.eye(n_all, dtype=complex)
    u[nf:, nf:] = out['u']
    m = numpy.conj(u) @ out['grad'] @ u.T             # input orbitals: phi_qc = phi u
    d = numpy.zeros((n_all, n_all), dtype=complex)
    d[nf:, nf:] = out['d']
    d = numpy.conj(u) @ d @ u.T                       # D[p, q] = dE2/dO[p, q]
    g_ci = numpy.atleast_2d(out['g_ci'])
    g_states = {j: g_ci[n] for n, j in enumerate(states)}
    if nf:
        ncore, ncas = mc.ncore, mc.ncas
        act = slice(ncore, ncore + ncas)
        eris0 = socutils_x2c.mo_eris(mf, mf.mol, mo, 0, ncore, ncas, cderi=resp.cderi)
        gam = numpy.zeros((n_all, n_all), dtype=complex)
        gam[:ncore, :ncore] = numpy.eye(ncore)
        gam[act, act] = resp.dm1
        f = eris0.generalized_fock(resp.dm1)
        eps = f.diagonal().real
        fi, cj = numpy.arange(nf), numpy.arange(nf, ncore)
        g_fc = numpy.conj(m[numpy.ix_(fi, cj)]) - m[numpy.ix_(cj, fi)].T
        zeta = -g_fc / (eps[fi][:, None] - eps[cj][None, :])
        dz = numpy.zeros((n_all, n_all), dtype=complex)
        dz[numpy.ix_(fi, cj)] = 0.5 * zeta.conj()
        dz[numpy.ix_(cj, fi)] = 0.5 * zeta.T
        mz, y = og._fock_m(eris0, dz, gam, f, return_y=True, c=mo)
        m = m + mz
        d = d + dz
        fc = resp.apply_h1(y[act, act], numpy.array(resp.ci).T).T
        for j, (w, gz) in enumerate(zip(resp.weights, fc)):
            g_states[j] = g_states.get(j, 0) + w * gz
    p, q = numpy.nonzero(mc.uniq_var_indices(n_all, mc.ncore, mc.ncas, mc.frozen))
    g_orb = numpy.conj(m[p, q]) - m[q, p]
    g_parts = {j: 2 * (g - numpy.vdot(resp.ci[j], g).real * resp.ci[j]) for j, g in g_states.items()}
    return g_orb, g_parts, mo @ d.T @ mo.conj().T


def relaxed_rdm1(mc, mf, mo, ci, weights, state=0, cderi=None, max_memory=4000,
                 return_parts=False, nfrozen=0):
    '''AO (j-spinor) density of state `state` of the averaged CASSCF (orbitals mo, CI
    vectors ci) whose expectation values are the derivatives of E_k(CASSCF) +
    E2(state-specific PC-NEVPT2); nfrozen: the frozen core of the NEVPT2 (the lowest
    orbitals of frozen_partition).  return_parts: also (D_k, D_unrelaxed, D_z).'''
    from socutils.zmr.uc import gradient
    cderi = mf.with_df._cderi if cderi is None else cderi
    ci = [numpy.asarray(c, dtype=complex).ravel() for c in ci]
    mo, _ = frozen_partition(mf, mc, mo, ci, weights, nfrozen, cderi)
    resp = zmc_response.SAResponse(mc, mo, ci, weights, cderi=cderi)
    resp.cderi = cderi
    pt = build_pt(mf, mc, mo, ci[state][:, None], cderi, max_memory=max_memory, nfrozen=nfrozen)
    out = gradient.derivatives(pt, max_memory=max_memory,
                               frozen_coeff=mo[:, :nfrozen] if nfrozen else None)
    g_orb, g_parts, d_unrel = _e2_terms(mc, mf, mo, resp, out, nfrozen, [state])
    parts = resp.split(zmc_response.state_gradient(resp, state))
    parts[0] = parts[0] + g_orb
    for j, g in g_parts.items():
        parts[1 + j] = parts[1 + j] + g
    z = zmc_response.zvector(resp, resp.join(parts))
    d_k = zmc_response.state_density(resp, state)
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


def _relaxed_for_rhos(mc, mf, mo, ci, weights, rho_fn, cderi, max_memory, nfrozen=0):
    '''(AO densities D[n], parts, pt): the relaxed densities of the targets
    T_n = sum_IJ rhos[n]_IJ heff_IJ (rhos Hermitian) of the multi-state PC-NEVPT2 on all
    states of the averaged spinor CASSCF: dT_n/dlambda = tr(O D[n]), rhos = rho_fn(heff)
    from the heff of that NEVPT2.  One pass over the classes, one Z-vector per target
    (rotations among degenerate averaged states projected out); nfrozen: frozen core (mo
    is taken from frozen_partition).'''
    from socutils.zmr.uc import gradient
    ci = [numpy.asarray(c, dtype=complex).ravel() for c in ci]
    mo, _ = frozen_partition(mf, mc, mo, ci, weights, nfrozen, cderi)
    resp = zmc_response.SAResponse(mc, mo, ci, weights, cderi=cderi)
    resp.cderi = cderi
    pt = build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights, max_memory=max_memory,
                  nfrozen=nfrozen)
    rhos = numpy.asarray(rho_fn(pt.heff))
    outs = gradient.derivatives(pt, max_memory=max_memory, rho=rhos,
                                frozen_coeff=mo[:, :nfrozen] if nfrozen else None)
    null = zmc_response.degenerate_rotations(resp)
    dms, parts_all = [], []
    for rho, out in zip(rhos, outs):
        g_orb, g_parts, d_unrel = _e2_terms(mc, mf, mo, resp, out, nfrozen, range(len(ci)))
        parts = resp.split(zmc_response.model_gradient(resp, rho))
        parts[0] = parts[0] + g_orb
        for j, g in g_parts.items():
            parts[1 + j] = parts[1 + j] + g
        z = zmc_response.zvector(resp, resp.join(parts), extra_null=null)
        d_model = zmc_response.model_density(resp, rho)
        d_z = zmc_response.lagrangian_density(resp, z)
        dms.append(d_model + d_unrel + d_z)
        parts_all.append((d_model, d_unrel, d_z))
    return dms, parts_all, pt


def relaxed_rdm1_levels(mc, mf, mo, ci, weights, levels=None, cderi=None, max_memory=4000,
                        degeneracy_tol=1e-8, return_parts=False, nfrozen=0):
    '''AO densities of levels of the multi-state PC-NEVPT2 (union contraction, the
    zeroth-order energy of every state) on all states of the averaged spinor CASSCF, whose
    expectation values are the derivatives of the eigenvalues `levels` of heff (default
    all; a degenerate multiplet averaged, level_weights) for time-even operators; nfrozen:
    frozen core.  Returns a list; return_parts: also (D_model, D_unrelaxed, D_z) and the
    eigenvalue.'''
    cderi = mf.with_df._cderi if cderi is None else cderi
    levels = list(range(len(ci))) if levels is None else list(levels)
    energies = {}

    def rho_fn(heff):
        rhos, e = level_weights(heff, levels, degeneracy_tol)
        energies['e'] = e
        return rhos
    dms, parts, _ = _relaxed_for_rhos(mc, mf, mo, ci, weights, rho_fn, cderi, max_memory,
                                      nfrozen)
    if return_parts:
        return [(d, x, float(energies['e'][k])) for d, x, k in zip(dms, parts, levels)]
    return dms


def relaxed_level_matrix(mc, mf, mo, ci, weights, level=0, cderi=None, max_memory=4000,
                         degeneracy_tol=1e-8, return_unrelaxed=False, nfrozen=0,
                         return_all=False):
    '''(e, D): the degenerate multiplet of level `level` of heff (eigenvectors c_m, energy
    e) and the relaxed transition densities D[m][n] (AO) with

        d<c_m|heff|c_n>/dlambda = tr(O D[m][n])   for h -> h + lambda O,

    the matrix whose eigenvalues are the first-order splittings of the multiplet by any
    one-electron perturbation (time-odd ones split a Kramers pair: magnetic properties).
    T_mn is complex: its real and imaginary parts, sum rho_IJ heff_IJ with the Hermitian
    rho = (r + r^+)/2 and (r - r^+)/2i, r_IJ = conj(c_mI) c_nJ, are two targets.
    return_unrelaxed: also the matrix at fixed orbitals and CI vectors; return_all: (e,
    {'model', 'unrelaxed', 'relaxed'}) instead; nfrozen: frozen core.'''
    cderi = mf.with_df._cderi if cderi is None else cderi
    info = {}

    def rho_fn(heff):
        e, vec = numpy.linalg.eigh(heff)
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
        info.update(e=e[level], k=len(group), keys=keys)
        return rhos
    dms, parts, _ = _relaxed_for_rhos(mc, mf, mo, ci, weights, rho_fn, cderi, max_memory,
                                      nfrozen)

    def matrix(ds):
        k = info['k']
        nao = ds[0].shape[0]
        out = [[numpy.zeros((nao, nao), dtype=complex) for _ in range(k)] for _ in range(k)]
        for (a, b, part), d in zip(info['keys'], ds):
            if part == 're':
                out[a][b] = out[a][b] + d
                if b != a:
                    out[b][a] = out[b][a] + d
            else:
                out[a][b] = out[a][b] + 1j * d
                out[b][a] = out[b][a] - 1j * d
        return out
    e = float(info['e'])
    if return_all:
        return e, {'model': matrix([x[0] for x in parts]),
                   'unrelaxed': matrix([x[0] + x[1] for x in parts]), 'relaxed': matrix(dms)}
    if return_unrelaxed:
        return e, matrix(dms), matrix([x[0] + x[1] for x in parts])
    return e, matrix(dms)


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


def hyperfine_ops(mf, atom, utm=True):
    '''(3, n2c, n2c) the X2C picture-changed hyperfine operator at nucleus `atom` of
    socutils.prop.hfc (the coupling to a unit nuclear spin, the isotope's g factor
    included).'''
    from socutils.prop.hfc import spinor_hf as hfc
    return hfc.int_hfc_2c(mf, atom, utm=utm)


def relaxed_doublet_blocks(mc, mf, mo, ci, weights, ops, level=0, cderi=None,
                           max_memory=4000, nfrozen=0):
    '''(energy, {'model', 'unrelaxed', 'relaxed'}: (ncomp, 2, 2)) the matrices
    <m|O_c|n> over the Kramers doublet of level `level` of the multi-state PC-NEVPT2 for
    the operators ops (ncomp, n2c, n2c): the CASSCF-model part, the derivative at fixed
    orbitals and CI vectors, and the relaxed one (relaxed_level_matrix; one solution for
    all operators); nfrozen: frozen core.'''
    e_k, mats = relaxed_level_matrix(mc, mf, mo, ci, weights, level=level, cderi=cderi,
                                     max_memory=max_memory, nfrozen=nfrozen, return_all=True)
    if len(mats['relaxed']) != 2:
        raise ValueError('level %d is %d-fold degenerate, not a Kramers doublet'
                         % (level, len(mats['relaxed'])))

    def block(d):
        return numpy.array([[[numpy.einsum('ij,ji', o, d[a][b]) for b in range(2)]
                             for a in range(2)] for o in ops])
    return e_k, {k: block(d) for k, d in mats.items()}


def relaxed_g_tensor(mc, mf, mo, ci, weights, level=0, cderi=None, max_memory=4000,
                     origin=None, ops=None, nfrozen=0):
    '''g tensors {'energy', 'model', 'unrelaxed', 'relaxed'} of the Kramers doublet of
    level `level` (relaxed_doublet_blocks of the magnetic-moment operators,
    g_tensor_from_block).'''
    if ops is None:
        ops = magnetic_moment_ops(mf, origin)
    e_k, blocks = relaxed_doublet_blocks(mc, mf, mo, ci, weights, ops, level, cderi, max_memory,
                                         nfrozen)
    out = {k: g_tensor_from_block(b) for k, b in blocks.items()}
    out['energy'] = e_k
    return out


def relaxed_hyperfine_tensor(mc, mf, mo, ci, weights, atoms, level=0, cderi=None,
                             max_memory=4000, origin=None, nfrozen=0):
    '''g and hyperfine tensors of the Kramers doublet of level `level`:
    {'energy', 'g': {kind: (3, 3)}, 'A': {kind: (natoms, 3, 3)}} for kind in model /
    unrelaxed / relaxed.  A_ab = Tr(M_a sigma_b) in the pseudo-spin frame fixed by g of
    the same kind (socutils.prop.observables.hyperfine_tensor): H = sum I_a A_ab S~_b,
    a.u., the sign relative to the non-negative g.'''
    from socutils.prop import observables
    ops = [magnetic_moment_ops(mf, origin)] + [hyperfine_ops(mf, a) for a in atoms]
    e_k, blocks = relaxed_doublet_blocks(mc, mf, mo, ci, weights, numpy.concatenate(ops),
                                         level, cderi, max_memory, nfrozen)
    out = {'energy': e_k, 'g': {}, 'A': {}}
    for kind, b in blocks.items():
        p, r = observables.pseudo_spin_frame(observables.pseudo_spin_tensor(b[:3]))
        out['g'][kind] = 2 * p
        out['A'][kind] = numpy.array([observables.pseudo_spin_tensor(b[3 + 3 * k:6 + 3 * k]) @ r.T
                                      for k in range(len(atoms))])
    return out


# ----------------------------------------------------------------------
# electric properties at the nuclei (time-even; any level, a degenerate one averaged)

def efg_ops(mf, atom, x_response=True):
    '''(3, 3, n2c, n2c) the X2C picture-changed electronic EFG operator at nucleus `atom`
    (socutils.prop.efg.spinor_hf.int_efg_2c): V = V_nuc - tr(int dm).'''
    from socutils.prop.efg import spinor_hf as efg
    return efg.int_efg_2c(mf, atom, x_response=x_response)


def contact_density_ops(mf, atom, x_response=True):
    '''(n2c, n2c) the X2C picture-changed contact-density operator at nucleus `atom`
    (socutils.prop.contact_density.spinor_hf.int_cd_2c): rho(R) = tr(int dm).'''
    from socutils.prop.contact_density import spinor_hf as cd
    return cd.int_cd_2c(mf, atom, x_response=x_response)


def efg_tensor(mf, dm, atom, ops=None):
    '''Total EFG (3, 3) at nucleus `atom` for an AO density dm (relaxed or not): the
    nuclear part minus the electronic one, as socutils.prop.efg.spinor_hf.kernel.'''
    from socutils.prop.efg.rhf import _get_quad_nuc
    if ops is None:
        ops = efg_ops(mf, atom)
    v_e = numpy.einsum('xyij,ji->xy', ops, dm).real
    return _get_quad_nuc(mf.mol, atom) - v_e
