'''Mixed CD / COSX core exchange of the spinor CASSCF (mc.core_exchange = 'cosx', mcscf.core_cosx).

1. energy of the same orbitals and density matrices, all-CD vs mixed: the difference is the COSX error of
   the core-core exchange energy alone
2. finite-difference check of the gradient and of the exact Hessian (zmc_twostep) of the mixed functional,
   away from a stationary point (randomly rotated orbitals): the analytic quantities are the derivatives of
   the mixed energy, not of the all-CD one
3. optionally (argv 'scf' / 'twostep'): converged SO-SCI / two-step CASSCF energies, all-CD vs mixed, and the
   core-core exchange energy of CD and COSX against the exact four-index value at the mixed orbitals

    python mcscf/test/test_core_cosx.py [HBr] [AuH] [HgCl2] [scf] [twostep] [nofd]
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys, time
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
import scipy.linalg
from pyscf import gto, lib
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_ao2mo, zmc_superci, zmc_twostep
from socutils.fci import zfci


def make(atom, basis, ncas, nelecas, x2c=True):
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=8000)
    mf = spinor_hf.KRHF(mol)
    if x2c:
        mf = mf.x2camf()
    # acCD auxiliary basis (socutils.cd.accd): a JK-fit basis leaves errors of order 1 Eh in the deep-core
    # exchange of a heavy atom (HBr cc-pVDZ, default auxbasis: 1.3 Eh in tr K[D_c]), which would hide the COSX
    # error the comparison is about
    from socutils.cd import accd
    mf = mf.density_fit(auxbasis=accd.accd_basis(mol))
    mf.kernel()
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    return mol, mf, mc


def core_exchange_energies(mol, mf, mo, ncore, eris_cd, eris):
    '''the core-core exchange energy -1/2 tr(D_c K[D_c]): exact four-index, CD factor, COSX'''
    cc = mo[:, :ncore]
    ex = spinor_hf.KRHF(mol).get_jk(mol, cc @ cc.conj().T)[1]
    e_ex = -.5 * np.trace(cc.conj().T @ ex @ cc).real
    e_cd = -.5 * np.trace(cc.conj().T @ eris_cd.vk_core @ cc).real
    e_x = -.5 * np.trace(eris.vkx_mo[:ncore, :ncore]).real
    print('core-core exchange energy: exact %.8f  CD - exact %.3e  COSX - exact %.3e' % (e_ex, e_cd - e_ex, e_x - e_ex))
    return e_ex, e_cd, e_x


def run_check(mol, mf, mc, seed=4, kscale=0.05):
    ncas, nelecas = mc.ncas, mc.nelecas
    nmo = mf.mo_coeff.shape[1]
    rng = np.random.default_rng(seed)
    k0 = kscale * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k0 - k0.conj().T)
    mc.mo_coeff = mo
    mc.core_exchange = 'cd'
    eris_cd = zmc_ao2mo._CDERIS(mc, mo, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris_cd)
    e_cd, _, ci = mci.kernel(mo, verbose=0)
    dm1, dm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    mc.core_exchange = 'cosx'
    t0 = time.perf_counter()
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    print('mixed _CDERIS %.2f s' % (time.perf_counter() - t0))
    e_cd0 = zmc_twostep.gen_g_hop_exact(mc, mo, dm1, dm2, eris_cd)[0]
    e0, g, hop = zmc_twostep.gen_g_hop_exact(mc, mo, dm1, dm2, eris)
    g_sci = zmc_superci.gen_g_hop(mc, mo.copy(), dm1, dm2, eris)[0]
    g_cd = zmc_twostep.gen_g_hop_exact(mc, mo, dm1, dm2, eris_cd)[1]
    print('fixed-density energy: all-CD %.10f  mixed %.10f  diff %.3e' % (e_cd0, e0, e0 - e_cd0))
    print('gradient: |g| %.3e, |g_mixed - g_cd| %.3e, twostep vs super-CI %.2e' % (
        np.linalg.norm(g), np.linalg.norm(g - g_cd), np.linalg.norm(g - g_sci)))
    core_exchange_energies(mol, mf, mo, mc.ncore, eris_cd, eris)

    def energy(x):
        m = mo @ scipy.linalg.expm(mc.unpack_uniq_var(x))
        return zmc_twostep.gen_g_hop_exact(mc, m, dm1, dm2, zmc_ao2mo._CDERIS(mc, m, level=2))[0]

    eps = 2e-3
    wg = wh = 0.0
    for trial in range(2):
        x = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
        x /= np.linalg.norm(x)
        ep, em = energy(eps * x), energy(-eps * x)
        ep2, em2 = energy(2 * eps * x), energy(-2 * eps * x)
        d1 = (8 * (ep - em) - (ep2 - em2)) / (12 * eps)
        d2 = (16 * (ep + em) - (ep2 + em2) - 30 * e0) / (12 * eps**2)
        a1, a2 = 2 * np.vdot(g, x).real, 2 * np.vdot(x, hop(x)).real
        a1cd = 2 * np.vdot(g_cd, x).real
        print('trial %d: dE/deps FD %.10f analytic %.10f (all-CD gradient %.10f);  d2E/deps2 FD %.7f analytic %.7f'
              % (trial, d1, a1, a1cd, d2, a2))
        wg, wh = max(wg, abs(d1 - a1)), max(wh, abs(d2 - a2))
    u = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
    v = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
    sym = abs(np.vdot(u, hop(v)).real - np.vdot(v, hop(u)).real)
    print('worst |FD - analytic|: gradient %.2e, Hessian %.2e; symmetry %.2e' % (wg, wh, sym))
    return wg, wh, sym


def run_scf(mol, mf, mc0, solver='sosci'):
    '''all-CD CASSCF from the HF orbitals, then the mixed one from the all-CD orbitals (from the same start the
    two can end on different stationary points of these small active spaces)'''
    from socutils.mcscf import zmc_sosci
    kern = {'twostep': zmc_twostep.mcscf_twostep, 'sosci': zmc_sosci.mcscf_sosci}.get(solver)
    out = {}
    mo = mf.mo_coeff
    for mode in ('cd', 'cosx'):
        mc = zmcscf.CASSCF(mf, mc0.ncas, mc0.nelecas)
        mc.fcisolver = zfci.FCISolver(mol)
        mc.core_exchange = mode
        mc.verbose = 0
        mc.conv_tol = 1e-10
        mc.max_cycle_macro = 200
        t0 = time.perf_counter()
        if kern is None:
            mc.kernel(mo)
        else:
            mc.superci(mo, _kern=kern)
        mo = mc.mo_coeff
        out[mode] = mc.e_tot
        print('%s CASSCF %s: E = %.10f  converged %s  %.1f s' % (solver, mode, mc.e_tot, mc.converged,
                                                               time.perf_counter() - t0))
    print('%s CASSCF mixed - all-CD = %.3e' % (solver, out['cosx'] - out['cd']))
    # what the difference is made of, at the converged mixed orbitals
    mc.core_exchange = 'cd'
    e_cd = zmc_ao2mo._CDERIS(mc, mo, level=2)
    mc.core_exchange = 'cosx'
    e_x = zmc_ao2mo._CDERIS(mc, mo, level=2)
    core_exchange_energies(mol, mf, mo, mc.ncore, e_cd, e_x)
    return out


SYSTEMS = {
    'HBr': ('H 0 0 0; Br 0 0 1.414', 'cc-pvdz', 4, 2),
    'AuH': ('Au 0 0 0; H 0 0 1.524', 'x2c-svpall', 4, 2),
    'HgCl2': ('Hg 0 0 0; Cl 0 0 2.252; Cl 0 0 -2.252', 'x2c-svpall', 6, 4),
}

if __name__ == '__main__':
    args = sys.argv[1:]
    names = [a for a in args if a in SYSTEMS] or ['HBr']
    for name in names:
        atom, basis, ncas, nel = SYSTEMS[name]
        print('==', name, basis, 'CAS(%d,%d)' % (nel, ncas))
        mol, mf, mc = make(atom, basis, ncas, nel)
        print('nao_nr %d, n2c %d, E(HF) %.10f' % (mol.nao_nr(), mol.nao_2c(), mf.e_tot))
        if 'nofd' not in args:
            wg, wh, sym = run_check(mol, mf, mc)
            assert wg < 2e-6 and wh < 2e-5 and sym < 1e-9, 'mixed functional derivatives FAIL'
        if 'scf' in args:
            run_scf(mol, mf, mc, 'sosci')
        if 'twostep' in args:
            run_scf(mol, mf, mc, 'twostep')
