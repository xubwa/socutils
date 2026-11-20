'''Spinor CASSCF / CAHF with double-group symmetry.

1. O2 (D∞h), SA-3 CASSCF(2e, 4 pi_g* spinors) on a SymAdaptedRHF: the
   irreps are picked up into mc.orbsym, the rotations stay within one irrep.
   Checked: energy equal to the run without symmetry from the same orbitals
   (super-CI; SO-SCI and the exact-Hessian two-step with symmetry), all orbitals
   pure in 2m_j and parity, symmetry-forbidden active-space integrals zero,
   Kramers partners consistent with time reversal and with the conjugate irreps.
2. O2 CAHF through cahf_diis: the same checks against a run on a Kramers-
   restricted SCF without symmetry.
3. H2O (C2v -> C2*), CASSCF(4e, 6 spinors): energy equal to the C1 run.

    python mcscf/test/test_double_group_casscf.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
from pyscf import gto
from pyscf.x2c import x2c
from socutils.symm import to_symm_frame
from socutils.symm import double_group as dg
from socutils.scf import double_group_symm, spinor_hf
from socutils.mcscf import zmcscf, zmc_ao2mo, zmc_twostep, zmc_sosci, cahf_diis, zcahf
from socutils.fci import zfci


def scf(mol, symm=True):
    cls = double_group_symm.SymAdaptedRHF if symm else spinor_hf.RHF
    mf = cls(mol).density_fit()
    mf.with_x2c = x2c.SpinorX2CHelper(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    h = mf.get_hcore()
    mf.get_hcore = lambda *args, **kwargs: h          # the X2C helper rebuilds it on every call
    return mf


def casscf(mf, ncas, nelecas, nroots=1, symmetry=True, mo=None, kern=None):
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mf.mol)
    mc.symmetry = symmetry
    mc.conv_tol = 1e-11
    mc.max_cycle_macro = 50
    mc.verbose = 0
    if nroots > 1:
        mc = mc.state_average([1. / nroots] * nroots)
    if kern is None:
        mc.kernel(mo)
    else:
        mc.superci(mo, _kern=kern)
    return mc


def check_symmetry(mc, mf):
    '''Orbital purity, forbidden active integrals, Kramers partners.'''
    gname = mf.doublegroupname
    so, irrep_id = mf._dg_symm_orb()
    mo = mc.mo_coeff
    orbsym, impurity = double_group_symm.orbital_purity(mo, so, irrep_id)
    assert np.array_equal(orbsym, mc.orbsym)
    assert np.array_equal(orbsym, mo.orbsym)

    ncore, ncas = mc.ncore, mc.ncas
    act = orbsym[ncore:ncore + ncas]
    eri = zmc_ao2mo._CDERIS(mc, mo, level=2).aaaa           # (tu|vw) = int t* u v* w
    conj = np.array([dg.conj(ir, gname) for ir in act])
    tu = dg.direct_prod(conj, act, gname)
    tot = dg.direct_prod(tu.ravel(), tu.ravel(), gname).reshape(eri.shape)
    forbidden = abs(eri[tot != 0]).max()
    allowed = abs(eri[tot == 0]).max()

    partner, bar = mc.kramers_pairs
    nmo = mo.shape[1]
    assert np.array_equal(partner, np.arange(nmo) ^ 1)
    assert np.array_equal(bar, np.arange(nmo) % 2)
    assert all(orbsym[partner[i]] == dg.conj(orbsym[i], gname) for i in range(nmo))
    print('  active irreps %s' % ' '.join(dg.irrep_id2name(gname, ir) for ir in act))
    print('  max irrep mixing %.1e, forbidden (tu|vw) %.1e (allowed up to %.2f)'
          % (impurity.max(), forbidden, allowed))
    assert impurity.max() < 1e-10
    assert forbidden < 1e-10
    return act


def test_o2():
    mol = gto.M(atom='O 0 0 0; O 0 0 1.21', basis='ccpvdz', symmetry=True, spin=2, verbose=0)
    mol = to_symm_frame(mol)
    mf = scf(mol)
    mo0 = mf.mo_coeff

    print('O2 SA-3 CASSCF(2e, 4o), D∞h')
    mc = casscf(mf, 4, 2, nroots=3, mo=mo0)
    act = check_symmetry(mc, mf)
    names = sorted(dg.irrep_id2name(mf.doublegroupname, ir) for ir in act)
    assert names == sorted(['1/2g', '-1/2g', '3/2g', '-3/2g']), names
    # natural orbitals within the irreps, Kramers pairs kept
    ncore, nocc = mc.ncore, mc.ncore + mc.ncas
    dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
    natocc, u = zmcscf.eig_sub(mc, -dm1.T, np.arange(ncore, nocc))
    mo_no = mc.mo_coeff.copy()
    mo_no[:, ncore:nocc] = mo_no[:, ncore:nocc] @ u
    mc_no = zmcscf.CASSCF(mf, 4, 2)
    mc_no.mo_coeff = mo_no
    mc_no._setup_orbsym(mo_no)
    assert np.array_equal(mc_no.orbsym, mc.orbsym)
    assert np.array_equal(mc_no.kramers_pairs[0], np.arange(mo_no.shape[1]) ^ 1)
    print('  natural occupations %s' % np.round(-natocc, 6))

    mc_c1 = casscf(mf, 4, 2, nroots=3, symmetry=False, mo=mo0)
    assert mc_c1.orbsym is None
    print('  E(D∞h) = %.12f  E(no symmetry) = %.12f  diff %.1e' % (mc.e_tot, mc_c1.e_tot, mc.e_tot - mc_c1.e_tot))
    assert mc.converged and mc_c1.converged
    assert abs(mc.e_tot - mc_c1.e_tot) < 1e-8

    mc2 = casscf(mf, 4, 2, nroots=3, mo=mo0, kern=zmc_twostep.mcscf_twostep)
    print('  exact-Hessian two-step: E = %.12f  diff %.1e' % (mc2.e_tot, mc2.e_tot - mc.e_tot))
    check_symmetry(mc2, mf)
    assert abs(mc2.e_tot - mc.e_tot) < 1e-8

    mc3 = casscf(mf, 4, 2, nroots=3, mo=mo0, kern=zmc_sosci.mcscf_sosci)
    print('  SO-SCI: E = %.12f  diff %.1e' % (mc3.e_tot, mc3.e_tot - mc.e_tot))
    check_symmetry(mc3, mf)
    assert abs(mc3.e_tot - mc.e_tot) < 1e-8

    print('O2 CAHF(2e, 4o), cahf_diis')
    res = {}
    for symm in (True, False):
        mfx = mf if symm else scf(mol, symm=False)
        mc = zmcscf.CASSCF(mfx, 4, 2)
        mc.fcisolver = zcahf.CAHF(mol)
        mc.conv_tol = 1e-11
        mc.max_cycle_macro = 60
        mc.verbose = 0
        cahf_diis.mc_diis(mc, mo0.copy(), verbose=0)
        res[symm] = mc
    mc = res[True]
    assert mc.orbsym is not None and res[False].orbsym is None
    check_symmetry(mc, mf)
    print('  E(D∞h) = %.12f  E(no symmetry) = %.12f  diff %.1e'
          % (mc.e_tot, res[False].e_tot, mc.e_tot - res[False].e_tot))
    assert abs(mc.e_tot - res[False].e_tot) < 1e-8


def test_h2o():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', symmetry=True, verbose=0)
    mol = to_symm_frame(mol)
    mf = scf(mol)
    print('H2O CASSCF(4e, 6o), %s -> %s' % (mol.groupname, mf.doublegroupname))
    mc = casscf(mf, 6, 4, mo=mf.mo_coeff)
    check_symmetry(mc, mf)
    mc_c1 = casscf(mf, 6, 4, symmetry=False, mo=mf.mo_coeff)
    print('  E(C2*) = %.12f  E(C1) = %.12f  diff %.1e' % (mc.e_tot, mc_c1.e_tot, mc.e_tot - mc_c1.e_tot))
    assert mc.converged and mc_c1.converged
    assert abs(mc.e_tot - mc_c1.e_tot) < 1e-8


if __name__ == '__main__':
    test_h2o()
    test_o2()
    print('ok')
