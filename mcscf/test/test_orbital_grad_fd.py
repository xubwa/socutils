#!/usr/bin/env python
'''Finite-difference validation of the SA-CASSCF orbital gradient (spinor).

The orbital gradient produced by zmc_superci.gen_g_hop must equal the
finite-difference derivative of the state-averaged CASCI energy with respect
to an orbital rotation kappa (CI re-solved at every displacement -- for a
variational CI the fixed-CI gradient is the exact gradient by
Hellmann-Feynman, so the two must agree to FD accuracy).

This is the first block of the validation scaffolding for the exact
orbital-Hessian Newton solver (zmc_newton): every analytic derivative gets a
finite-difference cross-check on a system small enough to make the check
exact-arithmetic clean.

Run:  python mcscf/test/test_orbital_grad_fd.py
'''
import os, sys
import numpy as np

# SOCUTILS_ENV: the wrapper dir whose socutils/ is the checkout under test
sys.path.insert(0, os.environ.get('SOCUTILS_ENV', os.path.expanduser('~/apps/socutils-git-env')))

from pyscf import gto, lib
from socutils.scf import spinor_hf
from socutils.mcscf import zmcscf, zmc_superci, zmc_ao2mo
from socutils.fci import zfci


def sa_energy(mc, mo, weights):
    '''State-averaged CASCI energy at fixed orbitals (CI solved exactly).'''
    ncas, ncore = mc.ncas, mc.ncore
    nelecas = mc.nelecas if isinstance(mc.nelecas, (int, np.integer)) \
        else sum(mc.nelecas)
    h1eff, ecore = mc.get_h1eff(mo)
    eri = np.asarray(mc.get_h2eff(mo)).reshape(ncas, ncas, ncas, ncas)
    e, _ = zfci.kernel(h1eff, eri, ncas, nelecas, ecore=ecore.real,
                       nroots=len(weights))
    return float(np.dot(weights, np.asarray(e).real))


def run_check(verbose=True, complex_active=False):
    mol = gto.M(atom='Be 0 0 0; H 0 0 1.3', basis='sto-3g', charge=0,
                spin=1, verbose=0)
    mf = spinor_hf.SpinorSCF(mol).density_fit()
    mf.kernel()

    ncas, nelecas = 4, 3          # 4 spinors, 3 electrons -> 4 dets
    nroots = 2
    weights = np.ones(nroots) / nroots

    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver = zfci.FCISolver(mol)
    mc.canonicalize_ = False
    mc.natorb = False
    mo = mf.mo_coeff.copy()
    nmo = mo.shape[1]
    ncore = mc.ncore
    if complex_active:
        # a random complex rotation inside the active space leaves the CASCI
        # energy unchanged but makes the rdms complex -- with natural (real)
        # rdms a transposed casdm1 convention in gen_g_hop goes unnoticed
        import scipy.linalg
        rng0 = np.random.default_rng(7)
        a = rng0.standard_normal((ncas, ncas)) + 1j * rng0.standard_normal((ncas, ncas))
        mo[:, ncore:ncore + ncas] = mo[:, ncore:ncore + ncas] @ scipy.linalg.expm(0.4 * (a - a.conj().T))

    # analytic gradient at mo: solve the CI once, build SA rdms, call gen_g_hop
    h1eff, ecore = mc.get_h1eff(mo)
    eri = np.asarray(mc.get_h2eff(mo)).reshape(ncas, ncas, ncas, ncas)
    es, cis = zfci.kernel(h1eff, eri, ncas, nelecas, ecore=ecore.real,
                          nroots=nroots)
    solver = zfci.FCISolver(mol)
    dm1 = 0; dm2 = 0
    for w, c in zip(weights, cis):
        d1, d2 = solver.make_rdm12(c, ncas, nelecas)
        dm1 = dm1 + w * d1
        dm2 = dm2 + w * d2
    eris = zmc_ao2mo._CDERIS(mc, mo, level=2)
    g_orb, h_diag, hop, sop, precond, mo_ = zmc_superci.gen_g_hop(
        mc, mo, dm1, dm2, eris)
    g_mat = mc.unpack_uniq_var(g_orb)

    # finite-difference reference: dE/dkappa_pq along random unique rotations
    rng = np.random.default_rng(0)
    npair_checked = 0
    max_err = 0.0
    eps = 1e-5
    # sample a handful of (occupied->virtual / core->active) rotations
    pairs = [(p, q) for p in range(nmo) for q in range(p)
             if not (p < ncore and q < ncore)
             and not (ncore <= p < ncore + ncas and ncore <= q < ncore + ncas)]
    rng.shuffle(pairs)
    for (p, q) in pairs[:12]:
        for phase in (1.0, 1.0j):
            dk = np.zeros((nmo, nmo), dtype=complex)
            dk[p, q] = eps * phase
            dk[q, p] = -np.conj(dk[p, q])
            ep = sa_energy(mc, mo @ zmc_superci.expmat(dk), weights)
            em = sa_energy(mc, mo @ zmc_superci.expmat(-dk), weights)
            fd = (ep - em) / (2 * eps)
            # E(kappa) = E0 + Re<g, kappa> in the packed convention:
            # derivative along dk with dk[p,q] = eps*phase
            # convention (calibrated by this harness): dE = 2 Re<g, kappa>
            an = 2 * np.real(np.conj(g_mat[p, q]) * phase)
            err = abs(fd - an)
            max_err = max(max_err, err)
            npair_checked += 1
            if verbose:
                print(f'rot ({p:2d},{q:2d}) phase {phase}: '
                      f'FD = {fd:+.10f}  analytic = {an:+.10f}  '
                      f'err = {err:.2e}')
    print(f'\nchecked {npair_checked} rotations, max |FD - analytic| = '
          f'{max_err:.3e}  (FD eps = {eps:.0e}), max |Im casdm1| = {abs(dm1.imag).max():.3e}')
    return max_err


if __name__ == '__main__':
    for cplx in (False, True):
        err = run_check(verbose=False, complex_active=cplx)
        assert err < 5e-7, f'orbital gradient FAILS FD check (complex_active={cplx}): {err:.3e}'
        print(f'orbital gradient PASSES finite-difference check (complex_active={cplx})')
