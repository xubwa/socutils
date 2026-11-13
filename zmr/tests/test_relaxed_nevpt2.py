'''The relaxed density of the spin-free PC-NEVPT2 of a singlet (analytic orbital and CI
gradients of E2, Z-vector of the CASSCF, Lagrangian density) against finite fields with
the CASSCF and the NEVPT2 redone in the field (water, 6-31G, CAS(4,4), density fitting).'''
import numpy
import pytest
from pyscf import gto, scf, mcscf
from pyscf.df import incore

AUX = 'weigend'


@pytest.mark.slow
def test_relaxed_dipole():
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2 as NRPC
    from socutils.zmr.nr import density as nr_density
    from socutils.zmr.reference import ci_gradient as cg, orbital_gradient as og
    from socutils.zmr.reference import relaxed_fd as rf
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis=AUX)
    q = mol.atom_charges()
    with mol.with_common_orig((q[:, None] * mol.atom_coords()).sum(0) / q.sum()):
        op = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()

    def casscf(h, mo0=None):
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf.get_hcore = lambda *a: h
        mf.conv_tol = 1e-13
        mf.kernel()
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.conv_tol = 1e-12
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 300
        mc.kernel(mo0)
        return rf.polish(mc)

    def e_tot(mc, h):
        eris = pyscf_nr.nr_eris(mol, h, mc.mo_coeff, 0, mc.ncore, mc.ncas, cderi=cderi)
        pt = NRPC(eris, (2, 2), civecs=numpy.asarray(mc.ci).reshape(-1, 1))
        pt.verbose = 0
        pt.kernel()
        return mc.e_tot + float(sum(v[0, 0].real for v in pt.h2_by_class.values()))

    mc = casscf(h0)
    route = nr_density.spin_orbital_route(mc, cderi)
    pt = route.pt()
    g_qc, u = og.gradient(pt)
    g = route.spin_sum(numpy.conj(u) @ g_qc @ u.T).real
    z = rf.zvector(mc, numpy.concatenate([mc.pack_uniq_var(g - g.T),
                                          cg.ci_gradient(pt, route)]))
    _, d2, _ = nr_density.make_rdm1(mc, cderi)
    c = mc.mo_coeff
    dm = mc.make_rdm1() + c @ d2.real.T @ c.T + rf.lagrangian_density(mc, z)
    relaxed = numpy.einsum('ij,ji', op, dm)
    ff = []
    for sgn in (1, -1):
        m2 = casscf(h0 + sgn * 1e-3 * op, mc.mo_coeff)
        ff.append(e_tot(m2, h0 + sgn * 1e-3 * op))
    ff = (ff[0] - ff[1]) / 2e-3
    # a single step of 1e-3: second order in the field, ~1e-6
    assert abs(relaxed - ff) < 5e-6, (relaxed, ff)
    # the unrelaxed density is off by much more
    assert abs(numpy.einsum('ij,ji', op, mc.make_rdm1() + c @ d2.real.T @ c.T) - ff) > 1e-3


@pytest.mark.slow
def test_relaxed_dipole_sa_casscf():
    '''State-specific PC-NEVPT2 of the states of an SA-CASSCF (two singlets, equal
    weights): the right-hand side has the gradient of the CASSCF energy of the state.'''
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2 as NRPC
    from socutils.zmr.nr import density as nr_density
    from socutils.zmr.reference import ci_gradient as cg, orbital_gradient as og
    from socutils.zmr.reference import relaxed_fd as rf
    from socutils.zmr.uc import density as uc_density
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis=AUX)
    q = mol.atom_charges()
    with mol.with_common_orig((q[:, None] * mol.atom_coords()).sum(0) / q.sum()):
        op = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()

    def casscf(h, mo0=None):
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf.get_hcore = lambda *a: h
        mf.conv_tol = 1e-13
        mf.kernel()
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.fix_spin_(ss=0)
        mc = mc.state_average_([0.5, 0.5])
        mc.conv_tol = 1e-12
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 300
        mc.kernel(mo0)
        return rf.polish(mc)

    def e_state(mc, h, k):
        eris = pyscf_nr.nr_eris(mol, h, mc.mo_coeff, 0, mc.ncore, mc.ncas, cderi=cderi)
        pt = NRPC(eris, (2, 2), civecs=numpy.asarray(mc.ci[k]).reshape(-1, 1))
        pt.verbose = 0
        pt.kernel()
        return (rf.state_energies(mc, mc.mo_coeff, list(mc.ci))[k]
                + float(sum(v[0, 0].real for v in pt.h2_by_class.values())))

    mc = casscf(h0)
    c = mc.mo_coeff
    m_p = casscf(h0 + 1e-3 * op, c)
    m_m = casscf(h0 - 1e-3 * op, c)
    for k in range(2):
        route = nr_density.spin_orbital_route(mc, cderi, ci=mc.ci[k])
        pt = route.pt()
        g_qc, u = og.gradient(pt)
        g = route.spin_sum(numpy.conj(u) @ g_qc @ u.T).real
        rhs = rf.state_gradient(mc, k)
        rhs[:rf.ngorb(mc)] += mc.pack_uniq_var(g - g.T)
        rhs += rf.ci_block(mc, k, cg.ci_gradient(pt, route))
        z = rf.zvector(mc, rhs)
        d2 = route.spin_sum(uc_density.make_rdm1(pt)[1]).real
        dm = rf.state_density(mc, k) + c @ d2.T @ c.T + rf.lagrangian_density(mc, z)
        relaxed = numpy.einsum('ij,ji', op, dm)
        ff = (e_state(m_p, h0 + 1e-3 * op, k) - e_state(m_m, h0 - 1e-3 * op, k)) / 2e-3
        assert abs(relaxed - ff) < 5e-6, (k, relaxed, ff)


@pytest.mark.slow
def test_relaxed_rdm1_interface():
    '''zmr.nr.relaxed.relaxed_rdm1 (iterative Z-vector, the gradients for large external
    spaces) against the dense Z-vector and the reference gradients, SA-CASSCF states.'''
    from socutils.zmr.nr import density as nr_density
    from socutils.zmr.nr import relaxed
    from socutils.zmr.reference import ci_gradient as cg, orbital_gradient as og
    from socutils.zmr.reference import relaxed_fd as rf
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis=AUX)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.fix_spin_(ss=0)
    mc = mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1e-12
    mc.kernel()
    mc = rf.polish(mc)
    for k in range(2):
        dm = relaxed.relaxed_rdm1(mc, cderi, state=k, max_memory=0.5)
        route = nr_density.spin_orbital_route(mc, cderi, ci=mc.ci[k])
        pt = route.pt()
        g_qc, u = og.gradient(pt)
        g = route.spin_sum(numpy.conj(u) @ g_qc @ u.T).real
        rhs = rf.state_gradient(mc, k)
        rhs[:rf.ngorb(mc)] += mc.pack_uniq_var(g - g.T)
        rhs += rf.ci_block(mc, k, cg.ci_gradient(pt, route))
        z = rf.zvector(mc, rhs)
        from socutils.zmr.uc import density as uc_density
        d2 = route.spin_sum(uc_density.make_rdm1(pt)[1]).real
        c = mc.mo_coeff
        ref = rf.state_density(mc, k) + c @ d2.T @ c.T + rf.lagrangian_density(mc, z)
        assert abs(dm - ref).max() < 1e-8, abs(dm - ref).max()


@pytest.mark.slow
@pytest.mark.parametrize('case', ['c2v', 'cs'])
def test_relaxed_dipole_qd_levels(case):
    '''Levels of the multi-state PC-NEVPT2 (union contraction, the zeroth-order energy of
    every state) on all states of an SA-CASSCF: zmr.nr.relaxed.relaxed_rdm1_level against
    finite fields.  c2v: two singlets of different symmetry; cs: three singlets of
    asymmetric water, two of one symmetry (heff and the Z-vector couple them).'''
    from socutils.zmr.interface import pyscf_nr
    from socutils.zmr.nr import PCNEVPT2 as NRPC
    from socutils.zmr.nr import relaxed
    from socutils.zmr.reference import relaxed_fd as rf
    if case == 'c2v':
        geom, weights = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', [0.5, 0.5]
    else:
        geom, weights = 'O 0 0 0.1173; H 0 0.7172 -0.4892; H 0 -0.8072 -0.4492', [1 / 3.] * 3
    mol = gto.M(atom=geom, basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis=AUX)
    q = mol.atom_charges()
    with mol.with_common_orig((q[:, None] * mol.atom_coords()).sum(0) / q.sum()):
        op = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()

    def casscf(h, mo0=None, ci0=None):
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        mf.get_hcore = lambda *a: h
        mf.conv_tol = 1e-13
        mf.kernel()
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.fix_spin_(ss=0)
        mc = mc.state_average_(weights)
        mc.conv_tol = 1e-12
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 300
        mc.kernel(mo0, ci0)
        return rf.polish(mc)

    def levels(mc, h):
        eris = pyscf_nr.nr_eris(mol, h, mc.mo_coeff, 0, mc.ncore, mc.ncas, cderi=cderi)
        pt = NRPC(eris, (2, 2), civecs=numpy.array([c.ravel() for c in mc.ci]).T,
                  weights=weights, e0='state', contraction='union')
        pt.verbose = 0
        return numpy.sort(numpy.asarray(pt.kernel()).real)

    mc = casscf(h0)
    # the levels need well converged CASSCF solutions (E2 is not stationary in them)
    assert mc.gnorm < 1e-10
    lv = []
    for sgn in (1, -1):
        m2 = casscf(h0 + sgn * 1e-3 * op, mc.mo_coeff, list(mc.ci))
        lv.append(levels(m2, h0 + sgn * 1e-3 * op))
    ff = (lv[0] - lv[1]) / 2e-3
    for k in range(len(weights)):
        dm, parts, e = relaxed.relaxed_rdm1_level(mc, cderi, k, return_parts=True)
        assert abs(e - levels(mc, h0)[k]) < 1e-10
        relaxed_k = numpy.einsum('ij,ji', op, dm)
        # a single step of 1e-3: second order in the field, a few 1e-6
        assert abs(relaxed_k - ff[k]) < 5e-6, (k, relaxed_k, ff[k])


def test_qd_derivatives_scalable_vs_reference():
    '''zmr.uc.gradient.derivatives with rho (segmented passes, small segments) against the
    Python reference zmr.nr.relaxed.qd_derivatives for a level of an SA-CASSCF with three
    states (asymmetric water).'''
    from socutils.zmr.nr import density as nr_density
    from socutils.zmr.nr import relaxed
    from socutils.zmr.uc import gradient as uc_gradient
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7172 -0.4892; H 0 -0.8072 -0.4492',
                basis='6-31g', verbose=0)
    cderi = incore.cholesky_eri(mol, auxbasis=AUX)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.fix_spin_(ss=0)
    mc = mc.state_average_([1 / 3.] * 3)
    mc.kernel()
    route = nr_density.spin_orbital_route(mc, cderi, ci=list(mc.ci))
    pt = route.pt()
    _, vec = numpy.linalg.eigh(pt.heff)
    for k in (1, 2):
        rho = numpy.outer(vec[:, k].real, vec[:, k].real)
        ref = relaxed.qd_derivatives(pt, rho)
        out = uc_gradient.derivatives(pt, max_memory=0.05, rho=rho)
        assert abs(out['d'] - ref['d']).max() < 1e-10
        assert abs(out['grad'] - ref['grad']).max() < 1e-10
        assert abs(out['g_ci'] - ref['g']).max() < 1e-10
