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


@pytest.mark.slow
def test_relaxed_spinor_qd_levels_kramers():
    '''Levels of the multi-state spinor PC-NEVPT2 with Kramers pairs (OH radical, 5
    electrons in 8 spinors on the Kramers-restricted orbitals of OH-, SA over the four
    components of X 2Pi): relaxed_rdm1_levels (the pair averaged) against finite fields
    of the levels for a time-even operator, which keeps every pair degenerate.'''
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf, zmc_response
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    nroots = 4
    weights = [1.0 / nroots] * nroots
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', charge=-1, verbose=0)
    # X2CAMF: spin-orbit coupling splits X 2Pi into the Kramers pairs 2Pi_3/2, 2Pi_1/2
    mf = spinor_hf.KRHF(mol).x2camf().density_fit()
    mf.kernel()
    cderi = mf.with_df._cderi
    h0 = mf.get_hcore()
    r = mol.intor('int1e_r_spinor')
    op = r[2] + 0.5 * r[0]

    def casscf(h, mo0):
        mc = zmcscf.CASSCF(mf, 8, 5, ncore=4)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.get_hcore = lambda *a: h
        mc.conv_tol = 1e-11
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        mc.kernel(mo0)
        mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), weights, verbose=False)
        assert gn < 1e-9, gn
        return mc, mo, ci

    def levels(h, mo0):
        mc, mo, ci = casscf(h, mo0)
        saved = mf.get_hcore
        mf.get_hcore = lambda *a: h
        try:
            pt = spinor_relaxed.build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights)
        finally:
            mf.get_hcore = saved
        return numpy.linalg.eigvalsh(pt.heff)

    mc, mo, ci = casscf(h0, mf.mo_coeff)
    res = spinor_relaxed.relaxed_rdm1_levels(mc, mf, mo, ci, weights, cderi=cderi,
                                             return_parts=True)
    e = [x[2] for x in res]
    print('QD levels', numpy.round(e, 8))
    assert abs(e[0] - e[1]) < 1e-9 and abs(e[2] - e[3]) < 1e-9      # Kramers pairs
    assert e[2] - e[1] > 1e-4                                       # split by spin-orbit
    h = 2e-5
    ff = (levels(h0 + h * op, mo) - levels(h0 - h * op, mo)) / (2 * h)
    for k, (dm, parts, _) in enumerate(res):
        relaxed = numpy.einsum('ij,ji', op, dm)
        unrel = numpy.einsum('ij,ji', op, parts[0] + parts[1]).real
        print('level %d: unrelaxed %.8f relaxed %.8f%+.1ei finite field %.8f'
              % (k, unrel, relaxed.real, relaxed.imag, ff[k]))
        assert abs(relaxed.imag) < 1e-9
        assert abs(relaxed.real - ff[k]) < 1e-6, (k, relaxed, ff[k])


@pytest.mark.slow
def test_relaxed_level_matrix_time_odd():
    '''The relaxed matrix of a Kramers pair (relaxed_level_matrix) for a time-odd operator
    (spin): its eigenvalues against the finite-field splitting of the pair.  OH radical,
    X2CAMF, SA over X 2Pi; the pair shares its E0 also in the field (degeneracy_tol), so
    heff stays covariant within it.  The splitting is odd in the field: half of it over h,
    Richardson in h.'''
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf, zmc_response
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    nroots = 4
    weights = [1.0 / nroots] * nroots
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', charge=-1, verbose=0)
    mf = spinor_hf.KRHF(mol).x2camf().density_fit()
    mf.kernel()
    cderi = mf.with_df._cderi
    h0 = mf.get_hcore()
    sig = mol.intor('int1e_sigma_spinor')
    op = sig[2] + 0.3 * sig[0]

    def casscf(h, mo0):
        mc = zmcscf.CASSCF(mf, 8, 5, ncore=4)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.get_hcore = lambda *a: h
        mc.conv_tol = 1e-11
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        mc.kernel(mo0)
        mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), weights, tol=1e-12,
                                         max_cycle=25, verbose=False)
        assert gn < 1e-11, gn
        return mc, mo, ci

    def pair(h, mo0):
        mc, mo, ci = casscf(h, mo0)
        saved = mf.get_hcore
        mf.get_hcore = lambda *a: h
        try:
            pt = spinor_relaxed.build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights,
                                         degeneracy_tol=1e-4)
        finally:
            mf.get_hcore = saved
        return numpy.linalg.eigvalsh(pt.heff)[:2]

    mc, mo, ci = casscf(h0, mf.mo_coeff)
    _, d = spinor_relaxed.relaxed_level_matrix(mc, mf, mo, ci, weights, level=0, cderi=cderi)
    w = numpy.array([[numpy.einsum('ij,ji', op, d[a][b]) for b in range(2)] for a in range(2)])
    assert abs(w - w.conj().T).max() < 1e-9
    w_eig = numpy.linalg.eigvalsh(w)
    half = {}
    for h in (1e-5, 2e-5):
        e = pair(h0 + h * op, mo)
        half[h] = (e[1] - e[0]) / (2 * h)
    ff = (4 * half[1e-5] - half[2e-5]) / 3
    print('eig(W) %s, finite-field half splitting %.8f' % (w_eig, ff))
    assert abs(w_eig[0] + w_eig[1]) < 1e-7                     # +-w: a Kramers pair
    assert abs(w_eig[1] - ff) < 2e-6, (w_eig, ff)


@pytest.mark.slow
def test_relaxed_g_tensor_finite_field():
    '''relaxed_g_tensor of the X 2Pi_3/2 doublet of OH (X2CAMF, X2C picture-changed
    1/2 (L + g_e S)) against the finite-field Zeeman splitting along the bond: a field B
    along z splits the doublet by B g_parallel / 2 (mu_B = 1/2), so half the splitting
    over B is g_parallel / 4.  The pair shares its E0 in the field.'''
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf, zmc_response
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    nroots = 4
    weights = [1.0 / nroots] * nroots
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', charge=-1, verbose=0)
    mf = spinor_hf.KRHF(mol).x2camf().density_fit()
    mf.kernel()
    cderi = mf.with_df._cderi
    h0 = mf.get_hcore()
    ops = spinor_relaxed.magnetic_moment_ops(mf)

    def casscf(h, mo0):
        mc = zmcscf.CASSCF(mf, 8, 5, ncore=4)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.get_hcore = lambda *a: h
        mc.conv_tol = 1e-11
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        mc.kernel(mo0)
        mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), weights, tol=1e-12,
                                         max_cycle=25, verbose=False)
        assert gn < 1e-11, gn
        return mc, mo, ci

    def pair(h, mo0):
        mc, mo, ci = casscf(h, mo0)
        saved = mf.get_hcore
        mf.get_hcore = lambda *a: h
        try:
            pt = spinor_relaxed.build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights,
                                         degeneracy_tol=1e-4)
        finally:
            mf.get_hcore = saved
        return numpy.linalg.eigvalsh(pt.heff)[:2]

    mc, mo, ci = casscf(h0, mf.mo_coeff)
    g = spinor_relaxed.relaxed_g_tensor(mc, mf, mo, ci, weights, level=0, cderi=cderi, ops=ops)
    g_par = numpy.linalg.eigvalsh(g['relaxed'])[-1]
    assert numpy.linalg.eigvalsh(g['relaxed'])[1] < 1e-4             # g_perp of 2Pi_3/2
    half = {}
    for b in (1e-5, 2e-5):
        e = pair(h0 + b * ops[2], mo)
        half[b] = (e[1] - e[0]) / (2 * b)
    ff = (4 * half[1e-5] - half[2e-5]) / 3
    print('g_parallel relaxed %.7f, model %.7f; finite field %.7f'
          % (g_par, numpy.linalg.eigvalsh(g['model'])[-1], 4 * ff))
    assert abs(g_par - 4 * ff) < 1e-5, (g_par, 4 * ff)


@pytest.mark.slow
def test_relaxed_hyperfine_finite_field():
    '''relaxed_hyperfine_tensor at the proton of OH X 2Pi_3/2 against the finite-field
    splitting of the doublet by a classical nuclear moment m along z (H += m h_z): half the
    splitting over m is |T_A[z, :]| / 2 (M_z = 1/2 sum_b T_zb sigma_b), T_A the pseudo-spin
    tensor before the frame rotation, whose row norms the rotation keeps.'''
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf, zmc_response
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    nroots = 4
    weights = [1.0 / nroots] * nroots
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', charge=-1, verbose=0)
    mf = spinor_hf.KRHF(mol).x2camf().density_fit()
    mf.kernel()
    cderi = mf.with_df._cderi
    h0 = mf.get_hcore()
    hz = spinor_relaxed.hyperfine_ops(mf, 1)[2]

    def casscf(h, mo0):
        mc = zmcscf.CASSCF(mf, 8, 5, ncore=4)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.get_hcore = lambda *a: h
        mc.conv_tol = 1e-11
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        mc.kernel(mo0)
        mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), weights, tol=1e-12,
                                         max_cycle=25, verbose=False)
        assert gn < 1e-11, gn
        return mc, mo, ci

    def pair(h, mo0):
        mc, mo, ci = casscf(h, mo0)
        saved = mf.get_hcore
        mf.get_hcore = lambda *a: h
        try:
            pt = spinor_relaxed.build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights,
                                         degeneracy_tol=1e-4)
        finally:
            mf.get_hcore = saved
        return numpy.linalg.eigvalsh(pt.heff)[:2]

    mc, mo, ci = casscf(h0, mf.mo_coeff)
    res = spinor_relaxed.relaxed_hyperfine_tensor(mc, mf, mo, ci, weights, atoms=[1], cderi=cderi)
    a_rel = res['A']['relaxed'][0]
    target = 0.5 * numpy.linalg.norm(a_rel[2])                       # rows kept by the frame
    m = 1e-5 / max(target, 1e-300)                                   # splitting ~ 1e-5 Eh
    half = {}
    for f in (1.0, 2.0):
        e = pair(h0 + f * m * hz, mo)
        half[f] = (e[1] - e[0]) / (2 * f * m)
    ff = (4 * half[1.0] - half[2.0]) / 3
    print('proton A (relaxed) principal', numpy.linalg.svd(a_rel)[1],
          'half splitting / m: analytic %.6e finite field %.6e' % (target, ff))
    assert abs(target - ff) < 1e-5 * abs(target), (target, ff)


@pytest.mark.slow
def test_relaxed_efg_contact_density_levels():
    '''Relaxed X2C picture-changed EFG (zz at the proton) and contact density (oxygen) of the
    X 2Pi_3/2 level of OH (multi-state PC-NEVPT2, the Kramers pair averaged) against finite
    fields of the level with the operator added to h.'''
    from pyscf import gto
    from socutils.fci import zfci
    from socutils.mcscf import zmcscf, zmc_response
    from socutils.scf import spinor_hf
    from socutils.zmr import spinor_relaxed
    nroots = 4
    weights = [1.0 / nroots] * nroots
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='6-31g', charge=-1, verbose=0)
    mf = spinor_hf.KRHF(mol).x2camf().density_fit()
    mf.kernel()
    cderi = mf.with_df._cderi
    h0 = mf.get_hcore()
    ops = {'EFG_zz(H)': spinor_relaxed.efg_ops(mf, 1)[2, 2],
           'rho(O)': spinor_relaxed.contact_density_ops(mf, 0)}
    for o in ops.values():
        assert abs(o - o.conj().T).max() < 1e-8 * abs(o).max()

    def casscf(h, mo0):
        mc = zmcscf.CASSCF(mf, 8, 5, ncore=4)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.get_hcore = lambda *a: h
        mc.conv_tol = 1e-11
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        mc.kernel(mo0)
        mo, ci, gn = zmc_response.polish(mc, mc.mo_coeff, list(mc.ci), weights, tol=1e-12,
                                         max_cycle=25, verbose=False)
        assert gn < 1e-11, gn
        return mc, mo, ci

    def level0(h, mo0):
        mc, mo, ci = casscf(h, mo0)
        saved = mf.get_hcore
        mf.get_hcore = lambda *a: h
        try:
            pt = spinor_relaxed.build_pt(mf, mc, mo, numpy.array(ci).T, cderi, weights=weights)
        finally:
            mf.get_hcore = saved
        return numpy.linalg.eigvalsh(pt.heff)[0]

    mc, mo, ci = casscf(h0, mf.mo_coeff)
    dm, (d_model, d_un, _), _ = spinor_relaxed.relaxed_rdm1_levels(
        mc, mf, mo, ci, weights, levels=[0], cderi=cderi, return_parts=True)[0]
    for name, o in ops.items():
        relaxed = numpy.einsum('ij,ji', o, dm)
        assert abs(relaxed.imag) < 1e-8 * abs(relaxed)
        h = 1e-5 / abs(relaxed.real)
        ff = (level0(h0 + h * o, mo) - level0(h0 - h * o, mo)) / (2 * h)
        print('%s: model %.8f unrelaxed %.8f relaxed %.8f finite field %.8f'
              % (name, numpy.einsum('ij,ji', o, d_model).real,
                 numpy.einsum('ij,ji', o, d_model + d_un).real, relaxed.real, ff))
        assert abs(relaxed.real - ff) < 1e-6 * abs(relaxed.real), (name, relaxed, ff)
