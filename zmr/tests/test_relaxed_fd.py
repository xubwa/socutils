'''zmr.reference.relaxed_fd: relaxed state properties of a state-averaged CASSCF through
the Z-vector and the Lagrangian density, against finite fields (water, 6-31G).'''
import numpy
import pytest
from pyscf import gto, scf, mcscf
from pyscf.fci import direct_spin1

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def _casscf(mol, h, weights, mo0=None):
    from socutils.zmr.reference import relaxed_fd as rf
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a: h
    mf.conv_tol = 1e-13
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc.fix_spin_(ss=0)
    if len(weights) > 1:
        mc = mc.state_average_(weights)
    mc.conv_tol = 1e-12
    mc.conv_tol_grad = 1e-6
    mc.max_cycle_macro = 300
    mc.kernel(mo0)
    return rf.polish(mc)


@pytest.mark.slow
def test_sa_casscf_state_dipole():
    from socutils.zmr.reference import relaxed_fd as rf
    mol = gto.M(atom=ATOM, basis='6-31g', verbose=0)
    q = mol.atom_charges()
    org = (q[:, None] * mol.atom_coords()).sum(0) / q.sum()
    with mol.with_common_orig(org):
        op = mol.intor('int1e_r')[2]
    h0 = scf.RHF(mol).get_hcore()
    w = [0.5, 0.5]
    mc = _casscf(mol, h0, w)
    assert mc.gnorm < 1e-9
    mo, ci = mc.mo_coeff, list(mc.ci)
    ff = []
    for sgn in (1, -1):
        m2 = _casscf(mol, h0 + sgn * 1e-3 * op, w, mo)
        assert m2.gnorm < 1e-9
        ff.append(rf.state_energies(m2, m2.mo_coeff, list(m2.ci)))
    ff = (ff[0] - ff[1]) / 2e-3
    for k in range(2):
        rhs = rf.fd_gradient(lambda m, c: rf.state_energies(mc, m, c)[k], mc)
        z = rf.zvector(mc, rhs)
        dz = rf.lagrangian_density(mc, z)
        # the state density in the AO basis
        nmo = mo.shape[1]
        g = numpy.zeros((nmo, nmo))
        g[:mc.ncore, :mc.ncore] = 2 * numpy.eye(mc.ncore)
        a = slice(mc.ncore, mc.ncore + mc.ncas)
        g[a, a] = direct_spin1.make_rdm1(ci[k], mc.ncas, mc.nelecas)
        d_state = mo @ g @ mo.T
        relaxed = numpy.einsum('ij,ji', op, d_state + dz)
        # the step of 1e-3 alone (no extrapolation): second order in the field, 1e-6
        assert abs(relaxed - ff[k]) < 1e-5, (k, relaxed, ff[k])
        # and the expectation value is not the derivative
        assert abs(numpy.einsum('ij,ji', op, d_state) - ff[k]) > 1e-3
