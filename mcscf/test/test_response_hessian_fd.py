'''Finite-difference check of the orbital + CI gradient and Hessian of the state-averaged
spinor CASSCF energy (zmc_response.SAResponse).

Along a direction z of orbital rotations and CI changes together,

    E(eps) = E(z = eps z),   dE/deps = Re<G, z>,   d^2E/deps^2 = Re<z, J z>

at any point (the parametrization is exact: C exp(K), (c + x)/|c + x|), so the check
is made away from a stationary point: randomly rotated orbitals, CASCI vectors of
these orbitals plus a random admixture.  J must be symmetric, Re<u, J v> = Re<v, J u>
(|J (norm, phase)| is printed only: those are zero modes at a stationary point alone).

    python mcscf/test/test_response_hessian_fd.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
import scipy.linalg
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_ao2mo, zmc_response
from socutils.fci import zfci


def run_check(atom, basis, ncas, nelecas, nroots, seed=7):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.fcisolver.nroots = nroots
    nmo = mf.mo_coeff.shape[1]
    rng = np.random.default_rng(seed)
    k0 = 0.05 * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k0 - k0.conj().T)
    mc.mo_coeff = mo
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    _, _, ci = mci.kernel(mo, verbose=0)
    ci = [np.asarray(c).ravel() for c in ci][:nroots]
    # away from the CASCI solution as well
    ci = [c + 0.05 * (rng.standard_normal(c.size) + 1j * rng.standard_normal(c.size)) for c in ci]
    ci = [c / np.linalg.norm(c) for c in ci]
    weights = np.full(nroots, 1.0 / nroots)
    resp = zmc_response.SAResponse(mc, mo, ci, weights, eris=eris)
    e0 = zmc_response.sa_energy(mc, mo, ci, weights)
    print('SA energy %.10f, fixed-density orbital energy %.10f' % (e0, resp.e_orb))
    assert abs(e0 - resp.e_orb) < 1e-9
    g = resp.gradient()
    n = len(g)

    def energy(z):
        m, c = zmc_response.move(mc, mo, ci, z)
        return zmc_response.sa_energy(mc, m, c, weights)

    def direction(kind):
        z = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        if kind == 'orbital':
            z[resp.n_orb:] = 0
        elif kind == 'ci':
            z[:resp.n_orb] = 0
        return z / np.linalg.norm(z)

    eps = 2e-3
    worst = 0.0
    for kind in ('orbital', 'ci', 'both', 'both'):
        z = direction(kind)
        ep, em = energy(eps * z), energy(-eps * z)
        ep2, em2 = energy(2 * eps * z), energy(-2 * eps * z)
        d1 = (8 * (ep - em) - (ep2 - em2)) / (12 * eps)
        d2 = (16 * (ep + em) - (ep2 + em2) - 30 * e0) / (12 * eps**2)
        a1, a2 = np.vdot(g, z).real, np.vdot(z, resp.hop(z)).real
        print('%-8s dE  FD %.9f  analytic %.9f ;  d2E  FD %.7f  analytic %.7f'
              % (kind, d1, a1, d2, a2))
        worst = max(worst, abs(d1 - a1), abs(d2 - a2))
    u, v = direction('both'), direction('both')
    sym = abs(np.vdot(u, resp.hop(v)).real - np.vdot(v, resp.hop(u)).real)
    print('symmetry |Re<u,Jv> - Re<v,Ju>| = %.2e' % sym)
    null = resp.null_space()
    nul = max(np.linalg.norm(resp.hop(null[:, k])) for k in range(null.shape[1]))
    print('|J (norm, phase)| = %.2e' % nul)
    return worst, sym, nul


if __name__ == '__main__':
    cases = (('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', '6-31g', 8, 4, 2),
             ('N 0 0 0; N 0 0 1.6', 'sto-3g', 8, 6, 3))
    for args in cases:
        worst, sym, nul = run_check(*args)
        assert worst < 2e-6, f'SA-CASSCF response FAILS the finite-difference check: {worst:.3e}'
        assert sym < 1e-9, f'SA-CASSCF Jacobian is not symmetric: {sym:.3e}'
    print('SA-CASSCF orbital + CI Hessian PASSES')
