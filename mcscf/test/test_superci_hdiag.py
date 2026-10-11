#!/usr/bin/env python
'''The diagonal handed to the Davidson solver by zmc_superci.gen_g_hop is the diagonal of its Hessian.

For every packed rotation k the element (H e_k)_k of the super-CI Hessian-vector product is compared with
h_diag[k]; complex orbitals and a complex density matrix, away from the SCF point (N2, sto-3g, CAS(6,8)).
The same diagonal is the preconditioner of the SO-SCI and exact-Hessian drivers; the super-CI driver
uses the diagonal in the active natural-orbital basis, which is checked in the same way.

  python test_superci_hdiag.py
      (SOCUTILS_ENV selects the wrapper directory of the checkout under test)
'''
import os, sys
import numpy as np
import scipy.linalg
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))
from pyscf import gto
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_superci, zmc_ao2mo
from socutils.fci import zfci


def run_check(verbose=True):
    mol = gto.M(atom='N 0 0 0; N 0 0 1.6', basis='sto-3g', verbose=0)
    mf = spinor_hf.KRHF(mol).density_fit()
    mf.kernel()
    ncas, nelecas = 8, 6
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.canonicalize_ = False
    mc.natorb = False
    nmo = mf.mo_coeff.shape[1]
    rng = np.random.default_rng(4)
    k = 0.1 * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k - k.conj().T)
    mc.mo_coeff = mo
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    e, _, ci = mci.kernel(mo, verbose=0)
    d1, d2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
    g, h_diag, hop, sop, _, _ = zmc_superci.gen_g_hop(mc, mo.copy(), d1, d2, eris)
    n = g.size
    diag = np.empty(n, dtype=complex)
    for i in range(n):
        ei = np.zeros(n, dtype=complex)
        ei[i] = 1.0
        diag[i] = hop(ei)[i]
    err = abs(diag.real - np.asarray(h_diag).real).max()
    if verbose:
        print('%d rotations, max |Im (H e_k)_k| = %.2e, diagonal between %.3f and %.3f' % (n, abs(diag.imag).max(), diag.real.min(), diag.real.max()))
        print('max |(H e_k)_k - h_diag_k| = %.2e' % err)
    assert err < 1e-10, err
    # the natural-orbital preconditioner of the super-CI driver: x^dagger H x and x^dagger S x for unit
    # rotations in the active natural-orbital basis against the diagonals precond_shift divides by
    u_no, hd_vc, hd_tc, hd_vt, sd_tc, sd_vt = hop.no_diag
    ncore, nocc = mc.ncore, mc.ncore + ncas
    err_no = err_s = 0.0
    for blk, hd, sd in (('tc', hd_tc, sd_tc), ('vt', hd_vt, sd_vt), ('vc', hd_vc, None)):
        for p in range(hd.shape[0]):
            for q in range(hd.shape[1]):
                x1 = np.zeros((nmo, nmo), dtype=complex)
                if blk == 'tc':
                    x1[ncore:nocc, q] = u_no[:, p]
                elif blk == 'vt':
                    x1[nocc + p, ncore:nocc] = u_no[:, q].conj()
                else:
                    x1[nocc + p, q] = 1.0
                x = mc.pack_uniq_var(x1 - x1.T.conj())
                nrm = np.vdot(x, x).real
                err_no = max(err_no, abs(np.vdot(x, hop(x)).real / nrm - hd[p, q]))
                if sd is not None:
                    err_s = max(err_s, abs(np.vdot(x, sop(x)).real / nrm - sd[p, q]))
    if verbose:
        print('natural-orbital basis: max |x^+ H x - hd| = %.2e, max |x^+ S x - sd| = %.2e' % (err_no, err_s))
    assert err_no < 1e-10 and err_s < 1e-10, (err_no, err_s)
    if verbose:
        print('the preconditioner diagonals are the diagonals of the super-CI Hessian and metric')
    return max(err, err_no, err_s)


if __name__ == '__main__':
    run_check()
