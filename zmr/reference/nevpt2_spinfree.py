'''
Second, independent brute-force UC-NEVPT2 for non-relativistic singlet
states, written in spatial orbitals on top of pyscf.fci.direct_spin1.

It shares nothing with the spin-orbital code except the PySCF integrals: the
spin-free Fock matrix, the canonicalization, the Dyall Hamiltonian and the
first-order equation (solved iteratively in the complete FCI space of all
correlated orbitals) are all written separately here.  It serves as the
external reference that UC-NEVPT2 otherwise lacks.
'''

import numpy
from scipy.sparse import linalg as splinalg
from pyscf import ao2mo, fci
from pyscf.fci import cistring, direct_spin1

CLASS_OF_EXCITATION = {
    (2, 2): 'ccvv', (2, 1): 'ccav', (1, 2): 'cavv', (2, 0): 'ccaa',
    (0, 2): 'aavv', (1, 1): 'cava', (1, 0): 'caaa', (0, 1): 'aaav',
}


def _spatial_integrals(mol, hcore_ao, mo_coeff, nfrozen):
    nmo = mo_coeff.shape[1]
    h = mo_coeff.T @ hcore_ao @ mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), nmo)
    f = slice(0, nfrozen)
    r = slice(nfrozen, nmo)
    ecore = (mol.energy_nuc() + 2 * numpy.trace(h[f, f])
             + 2 * numpy.einsum('iijj->', eri[f, f, f, f])
             - numpy.einsum('ijji->', eri[f, f, f, f]))
    h1 = h[r, r] + 2 * numpy.einsum('pqff->pq', eri[r, r, f, f]) \
        - numpy.einsum('pffq->pq', eri[r, f, f, r])
    return ecore, h1, eri[r, r, r, r].copy()


def _active_problem(h1, eri, nc, na):
    c = slice(0, nc)
    a = slice(nc, nc + na)
    e_core = 2 * numpy.trace(h1[c, c]) + 2 * numpy.einsum('iijj->', eri[c, c, c, c]) \
        - numpy.einsum('ijji->', eri[c, c, c, c])
    heff = h1[a, a] + 2 * numpy.einsum('tuii->tu', eri[a, a, c, c]) \
        - numpy.einsum('tiiu->tu', eri[a, c, c, a])
    return e_core, heff, eri[a, a, a, a].copy()


def _fock(h1, eri, nc, na, dm1_act):
    '''Spin-free generalized Fock matrix with the spin-summed active 1-RDM.'''
    c = slice(0, nc)
    a = slice(nc, nc + na)
    f = h1 + 2 * numpy.einsum('pqii->pq', eri[:, :, c, c]) - numpy.einsum('piiq->pq', eri[:, c, c, :])
    f += numpy.einsum('pqtu,tu->pq', eri[:, :, a, a], dm1_act) \
        - 0.5 * numpy.einsum('putq,tu->pq', eri[:, a, a, :], dm1_act)
    return f


def _rotate(h1, eri, u):
    return u.T @ h1 @ u, numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, u, u, u, u, optimize=True)


def spinfree_nevpt2(mol, hcore_ao, mo_coeff, nfrozen, ncore, ncas, nelecas, tol=1e-12):
    '''UC-NEVPT2 energy (single state, lowest CASCI root in the M_S = 0
    sector) by brute force in the FCI space of all correlated orbitals.

    ncore / ncas are numbers of correlated core and active spatial orbitals;
    nelecas = (nalpha, nbeta) in the active space.

    Returns (e_cas, e2_by_class dict).
    '''
    ecore, h1, eri = _spatial_integrals(mol, hcore_ao, mo_coeff, nfrozen)
    nmo = h1.shape[0]
    nc, na = ncore, ncas
    nv = nmo - nc - na

    e_core, heff, aaaa = _active_problem(h1, eri, nc, na)
    solver = direct_spin1.FCI()
    solver.conv_tol = 1e-14
    e_act, ci_act = solver.kernel(heff, aaaa, na, nelecas)
    dm1 = solver.make_rdm1(ci_act, na, nelecas)

    # canonicalize core and virtual blocks of the spin-free Fock matrix
    f = _fock(h1, eri, nc, na, dm1)
    u = numpy.eye(nmo)
    eps = numpy.zeros(nmo)
    for s in (slice(0, nc), slice(nc + na, nmo)):
        e, v = numpy.linalg.eigh(f[s, s])
        u[s, s] = v
        eps[s] = e
    h1, eri = _rotate(h1, eri, u)
    e_core, heff, aaaa = _active_problem(h1, eri, nc, na)

    # Dyall ERIs: eps on core / virtual, heff + aaaa on the active block
    c = slice(0, nc)
    a = slice(nc, nc + na)
    v = slice(nc + na, nmo)
    h_d = numpy.zeros((nmo, nmo))
    h_d[c, c] = numpy.diag(eps[c])
    h_d[v, v] = numpy.diag(eps[v])
    h_d[a, a] = heff
    eri_d = numpy.zeros((nmo,) * 4)
    eri_d[a, a, a, a] = aaaa
    c_d = e_core - 2 * eps[c].sum()

    nelec = (nelecas[0] + nc, nelecas[1] + nc)
    link = (cistring.gen_linkstr_index_trilidx(range(nmo), nelec[0]),
            cistring.gen_linkstr_index_trilidx(range(nmo), nelec[1]))
    h2_full = direct_spin1.absorb_h1e(h1, eri, nmo, nelec, 0.5)
    h2_dyall = direct_spin1.absorb_h1e(h_d, eri_d, nmo, nelec, 0.5)

    def apply_h(x):
        return direct_spin1.contract_2e(h2_full, x, nmo, nelec, link) + ecore * x

    def apply_hd(x):
        return direct_spin1.contract_2e(h2_dyall, x, nmo, nelec, link) + (ecore + c_d) * x

    # embed the CAS vector: core fully occupied, virtuals empty
    stra = cistring.gen_strings4orblist(range(nmo), nelec[0])
    strb = cistring.gen_strings4orblist(range(nmo), nelec[1])
    core_mask = (1 << nc) - 1
    virt_mask = ((1 << nv) - 1) << (nc + na)
    act_a = cistring.gen_strings4orblist(range(na), nelecas[0])
    act_b = cistring.gen_strings4orblist(range(na), nelecas[1])
    ia = cistring.strs2addr(nmo, nelec[0], [core_mask | (s << nc) for s in act_a])
    ib = cistring.strs2addr(nmo, nelec[1], [core_mask | (s << nc) for s in act_b])
    psi = numpy.zeros((len(stra), len(strb)))
    psi[numpy.ix_(ia, ib)] = ci_act.reshape(len(act_a), len(act_b))

    e_cas = float(psi.ravel() @ apply_h(psi).ravel())
    in_cas = numpy.zeros(psi.shape, dtype=bool)
    in_cas[numpy.ix_(ia, ib)] = True

    rhs = apply_h(psi) - apply_hd(psi)
    cas_leak = abs(rhs[in_cas]).max()
    rhs[in_cas] = 0

    hdiag = direct_spin1.make_hdiag(h_d, eri_d, nmo, nelec) + ecore + c_d
    shape = psi.shape
    op = splinalg.LinearOperator((psi.size, psi.size), dtype=float,
                                 matvec=lambda x: (apply_hd(x.reshape(shape)) - e_cas * x.reshape(shape)).ravel())
    denom = hdiag.ravel() - e_cas
    denom[abs(denom) < 1e-8] = 1e-8
    prec = splinalg.LinearOperator(op.shape, dtype=float, matvec=lambda x: x / denom)
    x, info = splinalg.gmres(op, rhs.ravel(), M=prec, rtol=tol, atol=0, restart=200, maxiter=2000)
    if info != 0:
        raise RuntimeError('GMRES did not converge (info=%d)' % info)
    x = x.reshape(shape)

    def count(strs, mask):
        return numpy.array([bin(int(s) & mask).count('1') for s in strs])

    holes = (2 * nc - count(stra, core_mask)[:, None] - count(strb, core_mask)[None, :])
    parts = count(stra, virt_mask)[:, None] + count(strb, virt_mask)[None, :]
    e2 = {}
    for key, name in CLASS_OF_EXCITATION.items():
        m = (holes == key[0]) & (parts == key[1])
        e2[name] = float(-(rhs[m] @ x[m]))
    return e_cas, e2, cas_leak
