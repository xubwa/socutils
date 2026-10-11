'''Checks of the SO-SCI Hessian (zmc_sosci).

1. J^tu_pq = (pq|tu) in the MO basis, from the AO pair operators, against the
   product of the full MO Cholesky vectors.
2. The A-A block (rotations with an active index), which zmc_sosci builds from
   J^tu and K^tu alone, against the same block of the exact orbital Hessian of
   zmc_twostep (itself checked by finite differences in
   test_exact_hessian_fd.py):  P_A H_exact P_A x.
3. The C-C block and the A-C coupling are those of the super-CI Hessian.
4. The assembled SO-SCI Hessian is symmetric.

Randomly rotated orbitals and a complex active density matrix.

    python mcscf/test/test_sosci_hessian.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
import numpy as np
import scipy.linalg
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_ao2mo, zmc_superci, zmc_twostep, zmc_sosci
from socutils.fci import zfci


class _AllActive:      # a CAS spanning every orbital: _CDERIS then returns the full L_pq
    def __init__(self, mc, nmo):
        self.mol, self._scf, self.verbose, self.ncore, self.ncas, self.max_memory = mc.mol, mc._scf, 0, 0, nmo, mc.max_memory


def run_check(atom, basis, ncas, nelecas, seed=4):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    ncore = mc.ncore
    nmo = mf.mo_coeff.shape[1]
    rng = np.random.default_rng(seed)
    k0 = 0.1 * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k0 - k0.conj().T)
    mc.mo_coeff = mo
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    ci = mci.kernel(mo, verbose=0)[2]
    dm1, dm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

    lfull = zmc_ao2mo._CDERIS(_AllActive(mc, nmo), mo, level=2).cd_pa
    jref = np.einsum('Ppq,Ptu->tupq', lfull, lfull[:, ncore:ncore + ncas, ncore:ncore + ncas])
    err_j = abs(zmc_sosci.pair_coulomb_mo(eris) - jref).max()

    g_exact, hop_exact = zmc_twostep.gen_g_hop_exact(mc, mo, dm1, dm2, eris)[1:]
    g_sci, _, hop_sci = zmc_superci.gen_g_hop(mc, mo.copy(), dm1, dm2, eris)[:3]
    g, hop_aa, pack_a = zmc_sosci.gen_hop_active(mc, mo, dm1, dm2, eris)
    hop = zmc_sosci.gen_g_hop(mc, mo.copy(), dm1, dm2, eris)[2]
    err_g = np.linalg.norm(g - g_exact)
    err_aa = err_rest = 0.0
    for trial in range(3):
        x = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
        xa = pack_a(x)
        err_aa = max(err_aa, np.linalg.norm(pack_a(hop_exact(xa)) - hop_aa(x)))
        # everything but the A-A block is the super-CI Hessian
        rest = hop(x) - pack_a(hop(xa))
        rest_sci = hop_sci(x) - pack_a(hop_sci(xa))
        err_rest = max(err_rest, np.linalg.norm(rest - rest_sci))
    u = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
    v = rng.standard_normal(g.size) + 1j * rng.standard_normal(g.size)
    sym = abs(np.vdot(u, hop(v)).real - np.vdot(v, hop(u)).real)
    print('%-2s max|Im casdm1| %.1e : J^tu_pq %.1e, gradient %.1e, A-A block vs exact %.1e, other blocks vs super-CI %.1e, symmetry %.1e'
          % (atom[:2], abs(dm1.imag).max(), err_j, err_g, err_aa, err_rest, sym))
    return max(err_j, err_g, err_aa, err_rest, sym)


if __name__ == '__main__':
    for args in (('N 0 0 0; N 0 0 1.6', 'sto-3g', 8, 6), ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', '6-31g', 8, 4)):
        err = run_check(*args)
        assert err < 1e-10, f'SO-SCI Hessian check FAILS: {err:.3e}'
    print('SO-SCI Hessian PASSES: exact active block, super-CI elsewhere, symmetric')
