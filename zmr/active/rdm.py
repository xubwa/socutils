'''
Active-space (transition) reduced density matrices of any rank.

Layout (conventions.md section 5, PySCF style): creators p_1 ... p_k and
annihilators q_1 ... q_k are nested,

    dm_k[p1, q1, p2, q2, ..., pk, qk] = <I| p1^+ p2^+ ... pk^+ qk ... q2 q1 |J>

so dm1[p, q] = <p^+ q> and dm2[p, q, r, s] = <p^+ r^+ s q>, the layout of
pyscf.fci.fci_dhf_slow and socutils.fci.zfci.

With D^J[q1 ... qk, K] = <K| qk ... q1 |J> (q1 applied first) over the
strings K of the N-k sector,

    dm_k^{IJ}[P, Q] = sum_K conj(D^I[P, K]) D^J[Q, K].
'''

import numpy


def annihilation_amplitudes(act, civecs, nelec, k):
    '''D[J, q1, ..., qk, K] = <K| qk ... q1 |Psi_J> for CI vectors given as
    columns of civecs.'''
    x = numpy.asarray(civecs).T                    # (J, dim_N)
    for m in range(k):
        x = act.apply_des(x, nelec - m)            # appends one index q_{m+1}
    return x


def make_rdm(act, civecs, nelec, k, bra=None, ket=None):
    '''Transition k-RDMs between all pairs of model states.

    Returns dm[I, J, p1, q1, ..., pk, qk]; if bra/ket indices are given,
    only that pair, dm[p1, q1, ..., pk, qk].'''
    n = act.norb
    civecs = numpy.asarray(civecs)
    if civecs.ndim == 1:
        civecs = civecs[:, None]
    nstate = civecs.shape[1]
    if k == 0:
        dm = civecs.conj().T @ civecs
    elif nelec < k:
        dm = numpy.zeros((nstate, nstate) + (n,) * (2 * k), dtype=numpy.complex128)
    else:
        d = annihilation_amplitudes(act, civecs, nelec, k)
        d = d.reshape(nstate, n**k, -1)
        dm = numpy.einsum('ipk,jqk->ijpq', d.conj(), d, optimize=True)
        dm = dm.reshape((nstate, nstate) + (n,) * (2 * k))
        # (p1..pk, q1..qk) -> (p1, q1, ..., pk, qk)
        order = [0, 1] + [2 + x for m in range(k) for x in (m, k + m)]
        dm = dm.transpose(order)
    if bra is not None:
        return dm[bra, ket]
    return dm
