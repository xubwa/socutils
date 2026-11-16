'''Relaxed density of the PC-NEVPT2 of a state of a state-averaged spinor CASSCF
(zmr.spinor_relaxed) against finite fields with the CASSCF and the NEVPT2 redone in the
field: water 6-31G, 4 electrons in 8 spinors, SA over the singlet ground state and the
triplet, the ground state; the operator a complex Hermitian j-spinor dipole component.'''
import numpy
import pytest


@pytest.mark.slow
def test_relaxed_spinor_pc_nevpt2():
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf, zmc_response
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    nroots = 4
    weights = [1.0 / nroots] * nroots
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7172 -0.4892; H 0 -0.8072 -0.4492',
                basis='6-31g', verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    cderi = mf.with_df._cderi
    h0 = mf.get_hcore()
    r = mol.intor('int1e_r_spinor')
    op = r[0] + 0.7 * r[1] + 0.3 * r[2]

    def casscf(h, mo0):
        mc = zmcscf.CASSCF(mf, 8, 4)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.get_hcore = lambda *a: h
        mc.conv_tol = 1e-11
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        mc.kernel(mo0)
        mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), weights, verbose=False)
        assert gn < 1e-9
        return mc, mo, ci

    def e_state(h, mo0, k=0):
        mc, mo, ci = casscf(h, mo0)
        e_cas = zmc_response.sa_energy(mc, mo, [ci[k]], [1.0])
        saved = mf.get_hcore
        mf.get_hcore = lambda *a: h
        try:
            pt = spinor_relaxed.build_pt(mf, mc, mo, ci[k][:, None], cderi)
        finally:
            mf.get_hcore = saved
        return e_cas + sum(v[0, 0] for v in pt.h2_by_class.values()).real

    mc, mo, ci = casscf(h0, mf.mo_coeff)
    dm, (d_k, d_un, d_z) = spinor_relaxed.relaxed_rdm1(mc, mf, mo, ci, weights, state=0,
                                                       cderi=cderi, return_parts=True)
    relaxed = numpy.einsum('ij,ji', op, dm)
    assert abs(relaxed.imag) < 1e-10
    # small fields: E2 is strongly nonlinear in the CI vector here (a near-dependent
    # first-order space of cava, metric eigenvalue 2e-9), so the curvature in the field
    # is large on the scale of 1e-3
    fd = {}
    for h in (1e-4, 2e-5):
        fd[h] = (e_state(h0 + h * op, mo) - e_state(h0 - h * op, mo)) / (2 * h)
        print('h %.0e: finite field %.8f' % (h, fd[h]))
    ff = fd[2e-5]
    print('relaxed %.8f  finite field %.8f  (CASSCF state %.8f, unrelaxed %.8f)'
          % (relaxed.real, ff, numpy.einsum('ij,ji', op, d_k).real,
             numpy.einsum('ij,ji', op, d_k + d_un).real))
    assert abs(relaxed.real - ff) < 1e-6, (relaxed, ff)


def test_spinor_derivatives_scalable_vs_reference():
    '''zmr.uc.gradient.derivatives (segments forced) for a spinor PC-NEVPT2 against the
    complex-orbital references: the orbital gradient M of gradient_general, the CI
    gradient and the unrelaxed density.'''
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    from socutils.zmr.reference import orbital_gradient as og
    from socutils.zmr.uc import density, gradient
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7172 -0.4892; H 0 -0.8072 -0.4492',
                basis='6-31g', verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    mc = zmcscf.CASSCF(mf, 8, 4)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.verbose = 0
    mc.kernel()
    ci = numpy.asarray(mc.ci).ravel()
    pt = spinor_relaxed.build_pt(mf, mc, mc.mo_coeff, ci[:, None], mf.with_df._cderi)
    out = gradient.derivatives(pt, max_memory=0.05)
    m_ref, _ = og.gradient_general(pt)
    assert abs(out['grad'] - m_ref).max() < 1e-10, abs(out['grad'] - m_ref).max()
    g_ref = spinor_relaxed.ci_gradient(pt)
    assert abs(out['g_ci'] - g_ref).max() < 1e-10, abs(out['g_ci'] - g_ref).max()
    u = out['u']
    assert abs(u.conj() @ out['d'] @ u.T - density.make_rdm1(pt)[1]).max() < 1e-10
