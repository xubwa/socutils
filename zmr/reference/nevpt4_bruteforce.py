'''
Brute-force multi-state NEVPT4(SD) in the determinant basis, for validation
of zmr.pt4.

The space is the determinant MRCISD space (contraction 'none') or the span of
the internally contracted functions (zmr.reference.icmrci_bruteforce: of all
model states for 'union', of state J alone for column J with 'state'), with
every CAS determinant.  H and the Dyall H_D (nevpt2_bruteforce.dyall_operator)
are projected onto it as dense matrices, the resolvent is the inverse of
E0_J - H_D on the complement of the model space, and the Bloch columns are
formed as in docs/derivations/nevpt4-sd.md.  Nothing of the product basis is
used.
'''

import numpy

from socutils.zmr import fock
from socutils.zmr.active.sectors import ActiveSpace
from socutils.zmr.reference import dets, icmrci_bruteforce, mrci_bruteforce
from socutils.zmr.reference.nevpt2_bruteforce import dyall_operator


def _space(eris, nelec, civecs_basis, contraction):
    sp = eris.spaces
    if contraction == 'none':
        strs = numpy.unique(mrci_bruteforce.mrci_strings(sp, nelec)[0])
        basis = numpy.eye(len(strs), dtype=numpy.complex128)
    else:
        strs, basis, _ = icmrci_bruteforce.contracted_basis(sp, nelec, civecs_basis, 'cas')
    return strs, basis


def kernel(eris, nelec, civecs, e0='state', contraction='state', renormalization=False,
           weights=None):
    '''Returns (e2, e3, e4): eigenvalues of the Hermitized effective
    Hamiltonian to second, third and fourth order.'''
    civecs = numpy.asarray(civecs, dtype=numpy.complex128)
    civecs = civecs[:, None] if civecs.ndim == 1 else civecs
    n = civecs.shape[1]
    weights = numpy.full(n, 1.0 / n) if weights is None else numpy.asarray(weights, float)
    act = ActiveSpace.from_eris(eris)
    qc = fock.quasi_canonicalize(eris, act.averaged_rdm1(civecs, nelec, weights))
    er = qc.eris
    sp = er.spaces
    h_op = dets.hamiltonian_operator(er.h1, er.eri_block('pppp'), er.ecore)
    hd_op = dyall_operator(er, qc.eps_core, qc.eps_virt)
    cas = dets.cas_strings(sp.ncore, sp.nact, nelec)

    def project(strs, basis):
        h = basis.conj().T @ (dets.operator_matrix(h_op, strs) @ basis)
        hd = basis.conj().T @ (dets.operator_matrix(hd_op, strs) @ basis)
        p = numpy.stack([basis.conj().T @ dets.SparseVector(cas, c).on(strs) for c in civecs.T])
        return h, hd, p

    if contraction == 'state':
        spaces = [project(*_space(er, nelec, civecs[:, [j]], 'state')) for j in range(n)]
    else:
        spaces = [project(*_space(er, nelec, civecs, contraction))] * n
    h, hd, p = spaces[0]
    hmodel = p.conj() @ h @ p.T
    hdmodel = p.conj() @ hd @ p.T
    e_d = hdmodel.diagonal().real
    e0s = e_d.copy() if e0 == 'state' else numpy.full(n, weights @ e_d)
    h1 = hmodel - numpy.diag(e0s)      # <K|V|J>, H0 = sum_J E0_J |J><J| on the model space

    def parts(j):
        h, hd, p = spaces[j]
        # orthonormal basis of the complement of the model space
        u, s, _ = numpy.linalg.svd(numpy.eye(h.shape[0]) - p.T @ p.conj())
        qb = u[:, s > 0.5]
        r = qb @ numpy.linalg.inv(e0s[j] * numpy.eye(qb.shape[1]) - qb.conj().T @ hd @ qb) @ qb.conj().T
        return h, hd, p, r

    sp_parts = [parts(j) for j in range(n)]
    v = [hh - hdd for hh, hdd, _, _ in sp_parts]
    x = [sp_parts[j][3] @ (v[j] @ sp_parts[j][2][j]) for j in range(n)]
    h2 = numpy.stack([sp_parts[j][2].conj() @ (v[j] @ x[j]) for j in range(n)], axis=1)
    shared = contraction != 'state'

    def renorm(vecs, hm, j):
        if shared:
            return sum(vecs[k] * hm[k, j] for k in range(n))
        return vecs[j] * hm[j, j]

    y = [sp_parts[j][3] @ (v[j] @ x[j] - renorm(x, h1, j)) for j in range(n)]
    h3 = numpy.stack([sp_parts[j][2].conj() @ (v[j] @ y[j]) for j in range(n)], axis=1)
    z = []
    for j in range(n):
        rhs = v[j] @ y[j] - renorm(y, h1, j)
        if renormalization:
            rhs = rhs - sum(x[k] * h2[k, j] for k in range(n))
        z.append(sp_parts[j][3] @ rhs)
    h4 = numpy.stack([sp_parts[j][2].conj() @ (v[j] @ z[j]) for j in range(n)], axis=1)

    def herm(m):
        return 0.5 * (m + m.conj().T)
    return (numpy.linalg.eigvalsh(herm(hmodel + h2)), numpy.linalg.eigvalsh(herm(hmodel + h2 + h3)),
            numpy.linalg.eigvalsh(herm(hmodel + h2 + h3 + h4)))
