#!/usr/bin/env python
'''Basis covariance of the super-CI building blocks (zmc_superci.gen_g_hop).

A unitary rotation W of the active orbitals among themselves (real or complex)
or of the core / virtual orbitals changes nothing physical, so with
mo' = mo W every operator-like quantity must transform as O' = W^+ O W: the
orbital gradient, the super-CI Hessian-vector product H x, the metric S x and
the preconditioner.  The diagonal-occupation metric and the missing complex
conjugates of the pre-2026-10 code only satisfied this in natural orbitals /
with a real active 1-rdm.

Also checks that the full metric reduces to the diagonal-occupation form in
natural orbitals and that H and S are Hermitian.

Run:  python mcscf/test/test_superci_covariance.py
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
    ncore = mc.ncore
    nocc = ncore + ncas
    rng = np.random.default_rng(4)
    # start away from the SCF point, with complex orbitals and a complex rdm
    k = 0.1 * (rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo)))
    mo = mf.mo_coeff @ scipy.linalg.expm(k - k.conj().T)
    mc.mo_coeff = mo

    def build(m):
        eris = zmc_ao2mo._CDERIS(mc, m, level=2)
        mci = zmcscf._fake_h_for_fast_casci(mc, m, eris)
        e, _, ci = mci.kernel(m, verbose=0)
        d1, d2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
        return zmc_superci.gen_g_hop(mc, m.copy(), d1, d2, eris), d1

    (g0, hd0, hop0, sop0, _, _), d1 = build(mo)
    x0 = rng.standard_normal(g0.size) + 1j * rng.standard_normal(g0.size)
    unp, pk = mc.unpack_uniq_var, mc.pack_uniq_var
    ref = {'g': unp(g0), 'Hx': unp(hop0(x0)), 'Sx': unp(sop0(x0)),
           'precond': unp(hop0.precond_shift(x0, -0.3))}
    worst = 0.0

    def rotation(block, cplx):
        n = block.stop - block.start
        a = rng.standard_normal((n, n)) + (1j * rng.standard_normal((n, n)) if cplx else 0)
        w = np.eye(nmo, dtype=complex)
        w[block, block] = scipy.linalg.expm(0.3 * (a - a.conj().T))
        return w

    cases = [('active, real', slice(ncore, nocc), False), ('active, complex', slice(ncore, nocc), True),
             ('core, complex', slice(0, ncore), True), ('virtual, complex', slice(nocc, nmo), True)]
    for label, block, cplx in cases:
        w = rotation(block, cplx)
        tr = lambda m: w.conj().T @ m @ w
        (g1, hd1, hop1, sop1, _, _), _ = build(mo @ w)
        x1 = pk(tr(unp(x0)))
        new = {'g': g1, 'Hx': hop1(x1), 'Sx': sop1(x1), 'precond': hop1.precond_shift(x1, -0.3)}
        for name in ('g', 'Hx', 'Sx', 'precond'):
            if name == 'precond' and not label.startswith('active'):
                continue      # the preconditioner uses the Fock diagonal of the core / virtual orbitals
            err = np.linalg.norm(new[name] - pk(tr(ref[name]))) / np.linalg.norm(new[name])
            worst = max(worst, err)
            if verbose:
                print(f'{label:18s} {name:8s} relative deviation from W^+ O W: {err:.2e}')

    # natural orbitals: the full metric is the diagonal-occupation one
    # casdm1[t,u] = <t^dagger u> becomes c^T D c^* under mo -> mo c
    occ, c = np.linalg.eigh(d1.T)
    wn = np.eye(nmo, dtype=complex)
    wn[ncore:nocc, ncore:nocc] = c
    (gn, hdn, hopn, sopn, _, _), dn = build(mo @ wn)
    xn = unp(x0)
    n = dn.diagonal().real
    diag = np.zeros_like(xn)
    diag[nocc:, :ncore] = xn[nocc:, :ncore]
    diag[nocc:, ncore:nocc] = xn[nocc:, ncore:nocc] * n[None, :]
    diag[ncore:nocc, :ncore] = xn[ncore:nocc, :ncore] * (1 - n)[:, None]
    err_no = np.linalg.norm(sopn(x0) - pk(diag)) / np.linalg.norm(sopn(x0))
    u = rng.standard_normal(g0.size) + 1j * rng.standard_normal(g0.size)
    err_h = abs(np.vdot(u, hop0(x0)) - np.vdot(hop0(u), x0)) / abs(np.vdot(u, hop0(x0)))
    err_s = abs(np.vdot(u, sop0(x0)) - np.vdot(sop0(u), x0)) / abs(np.vdot(u, sop0(x0)))
    print(f'max |Im casdm1| = {abs(d1.imag).max():.2e}')
    offd = abs(dn - np.diag(dn.diagonal())).max()
    print(f'casdm1 in natural orbitals: max off-diagonal {offd:.2e}')
    print(f'metric vs diagonal-occupation form in natural orbitals: {err_no:.2e}')
    print(f'Hermiticity of H: {err_h:.2e}   of S: {err_s:.2e}')
    return max(worst, err_no, err_h, err_s, offd)


if __name__ == '__main__':
    err = run_check()
    assert err < 1e-9, f'super-CI operators are NOT basis covariant: {err:.3e}'
    print(f'super-CI gradient, Hessian, metric and preconditioner are basis covariant (max deviation {err:.1e})')
