'''Finite-difference check of the exact orbital Hessian of zmc_twostep.

For fixed density matrices the energy in the orbitals C exp(eps kappa) is

    E(eps) = E0 + 2 eps Re<g, x> + eps^2 <x, H x> + O(eps^3),   x = pack(kappa)

so central differences of the fixed-density energy give 2 Re<g, x> and
2 <x, H x>.  The check is made away from a stationary point (randomly rotated
orbitals) and with a complex active density matrix, where the gradient, the
commutator term of H x and the transposes of the density matrices all matter.
H must also be symmetric, Re<u, H v> = Re<v, H u>.

    python mcscf/test/test_exact_hessian_fd.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
import scipy.linalg
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_ao2mo, zmc_superci, zmc_twostep
from socutils.fci import zfci


def run_check(atom, basis, ncas, nelecas, seed=4):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    nmo = mf.mo_coeff.shape[1]
    rng = np.random.default_rng(seed)
    k0 = 0.1 * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k0 - k0.conj().T)
    mc.mo_coeff = mo
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    e_ci, _, ci = mci.kernel(mo, verbose=0)
    dm1, dm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

    e0, g, hop = zmc_twostep.gen_g_hop_exact(mc, mo, dm1, dm2, eris)
    g_sci = zmc_superci.gen_g_hop(mc, mo.copy(), dm1, dm2, eris)[0]
    print('max |Im casdm1| = %.2e' % abs(dm1.imag).max())
    print('fixed-density energy - CI energy: %.2e' % (e0 - np.asarray(e_ci).real))
    print('gradient vs zmc_superci.gen_g_hop: %.2e  (|g| = %.2e)' % (np.linalg.norm(g - g_sci), np.linalg.norm(g)))
    assert abs(e0 - np.asarray(e_ci).real) < 1e-9
    assert np.linalg.norm(g - g_sci) < 1e-10

    def energy(x):
        m = mo @ scipy.linalg.expm(mc.unpack_uniq_var(x))
        return zmc_twostep.gen_g_hop_exact(mc, m, dm1, dm2, zmc_ao2mo._CDERIS(mc, m, level=2))[0]

    eps = 2e-3
    worst = 0.0
    for trial in range(3):
        x = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
        x /= np.linalg.norm(x)
        ep, em = energy(eps * x), energy(-eps * x)
        ep2, em2 = energy(2 * eps * x), energy(-2 * eps * x)
        d1 = (8 * (ep - em) - (ep2 - em2)) / (12 * eps)                      # fourth-order formulas
        d2 = (16 * (ep + em) - (ep2 + em2) - 30 * e0) / (12 * eps**2)
        a1, a2 = 2 * np.vdot(g, x).real, 2 * np.vdot(x, hop(x)).real
        print('trial %d: dE/deps  FD %.9f  analytic %.9f ;  d2E/deps2  FD %.7f  analytic %.7f' % (trial, d1, a1, d2, a2))
        worst = max(worst, abs(d1 - a1), abs(d2 - a2))
    u = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
    v = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
    sym = abs(np.vdot(u, hop(v)).real - np.vdot(v, hop(u)).real)
    print('symmetry |Re<u,Hv> - Re<v,Hu>| = %.2e' % sym)
    return worst, sym


if __name__ == '__main__':
    for args in (('N 0 0 0; N 0 0 1.6', 'sto-3g', 8, 6), ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', '6-31g', 8, 4)):
        worst, sym = run_check(*args)
        assert worst < 2e-6, f'exact orbital Hessian FAILS the finite-difference check: {worst:.3e}'
        assert sym < 1e-10, f'exact orbital Hessian is not symmetric: {sym:.3e}'
    print('exact orbital Hessian PASSES the finite-difference and symmetry checks')
