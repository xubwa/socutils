'''The density inside get_fock follows casdm1[p,q] = <p^dagger q>.

The AO density of a CASCI/CASSCF is C casdm1^T C^dagger (see
CASBase.make_rdm1).  With randomly rotated complex orbitals the active
density matrix is complex and its transpose matters: the density that
get_fock of zcasci and of zmcscf passes to get_jk must equal mc.make_rdm1,
and Tr(h D) must equal the one-electron energy in the MO basis.

    python mcscf/test/test_get_fock_density.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
import scipy.linalg
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zcasci, zmc_ao2mo
from socutils.fci import zfci


def check(cls, module, mf, mo, ncas, nelecas):
    mc = cls(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mf.mol)
    mc.mo_coeff = mo
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    ci = zmcscf._fake_h_for_fast_casci(mc, mo, eris).kernel(mo, verbose=0)[2]
    dm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)

    seen = {}
    get_jk = mc._scf.get_jk
    def spy(mol, dm, *args, **kwargs):
        seen['dm'] = dm
        return get_jk(mol, dm, *args, **kwargs)
    mc._scf.get_jk = spy
    module.get_fock(mc, mo, ci, None, dm1)
    mc._scf.get_jk = get_jk

    h = mf.get_hcore()
    core, cas = mo[:, :mc.ncore], mo[:, mc.ncore:mc.ncore + ncas]
    e1_mo = (np.trace(core.conj().T @ h @ core)
             + np.einsum('pq,pq', cas.conj().T @ h @ cas, dm1))
    e1_ao = np.einsum('uv,vu', h, seen['dm'])
    ddm = abs(seen['dm'] - mc.make_rdm1(mo, ci)).max()
    print('%-6s max|Im casdm1| = %.1e  E1 MO - AO = %.1e  density vs make_rdm1 = %.1e'
          % (cls.__name__, abs(dm1.imag).max(), abs(e1_mo - e1_ao), ddm))
    assert abs(dm1.imag).max() > 1e-2
    assert abs(e1_mo - e1_ao) < 1e-10
    assert ddm < 1e-10


if __name__ == '__main__':
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g', verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    nmo = mf.mo_coeff.shape[1]
    rng = np.random.default_rng(4)
    k0 = 0.1 * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k0 - k0.conj().T)
    check(zmcscf.CASSCF, zmcscf, mf, mo, 6, 4)
    check(zcasci.CASCI, zcasci, mf, mo, 6, 4)
    print('ok')
