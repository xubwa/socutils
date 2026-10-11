'''Kramers (time-reversal) structure of j-spinor molecular orbitals.

The spinor CASSCF driver (mcscf.zmc_superci) keeps the orbitals Kramers
restricted by projecting every rotation with ensure_kramers, which assumes the
layout produced by zquatev: columns 2k and 2k+1 are Kramers partners with
phase +1, T c_2k = c_2k+1 (hence T c_2k+1 = -c_2k).  The helpers here measure
how well a set of orbitals follows that layout and restore it.
'''
import numpy


def time_reversal(mol, c):
    '''T applied to AO-coefficient vector(s) in the j-spinor AO basis.'''
    trmap = mol.time_reversal_map()
    idx = abs(trmap) - 1
    sgn = numpy.sign(trmap)
    tc = numpy.empty_like(c)
    tc[idx] = (sgn[:, None] if c.ndim == 2 else sgn) * c.conj()
    return tc


def pairing_deviation(mol, mo, s=None):
    '''max |C^+ S T(C) - P_ideal| for the interleaved, phase +1 layout
    (0 = perfect, 2 = a pair with the opposite phase).'''
    if s is None:
        s = mol.intor('int1e_ovlp_spinor')
    n = mo.shape[1]
    p = mo.conj().T @ s @ time_reversal(mol, mo)
    ideal = numpy.zeros((n, n))
    for k in range(n // 2):
        ideal[2*k+1, 2*k] = 1
        ideal[2*k, 2*k+1] = -1
    return abs(p - ideal).max()


def subspace_leak(mol, mo_sub, s=None):
    '''S-norm of the part of T(C_sub) outside span(C_sub): 0 when the
    subspace is time-reversal invariant.'''
    if s is None:
        s = mol.intor('int1e_ovlp_spinor')
    tc = time_reversal(mol, mo_sub)
    r = tc - mo_sub @ (mo_sub.conj().T @ s @ tc)
    return numpy.sqrt(abs(numpy.trace(r.conj().T @ s @ r)))


def pair_subspace(mol, mo_sub, op, s=None):
    '''Interleaved, phase +1 Kramers pairs spanning a time-reversal invariant
    subspace: the levels of the projected Hermitian, time-reversal symmetric
    operator `op` (AO basis), each taken together with its partner T v.'''
    if s is None:
        s = mol.intor('int1e_ovlp_spinor')
    m = mo_sub.shape[1]
    e, u = numpy.linalg.eigh(mo_sub.conj().T @ op @ mo_sub)
    out = []
    for k in range(m):
        v = mo_sub @ u[:, k]
        if out:
            q = numpy.array(out).T
            v = v - q @ (q.conj().T @ s @ v)
        nv = numpy.sqrt((v.conj() @ s @ v).real)
        if nv < 1e-3:
            continue
        v = v / nv
        w = time_reversal(mol, v)
        if out:
            w = w - q @ (q.conj().T @ s @ w)
        w = w / numpy.sqrt((w.conj() @ s @ w).real)
        out += [v, w]
        if len(out) == m:
            break
    if len(out) != m:
        raise RuntimeError('could not pair the subspace: is it time-reversal invariant?')
    return numpy.array(out).T


def symmetrize(mol, mo, ncore, ncas, pair_op=None, s=None):
    '''Nearest Kramers-restricted orbitals to `mo`, in the layout the CASSCF
    driver needs.

    The core space is replaced by the dominant ncore eigenvectors of
    (P_core + T P_core T^-1)/2, the active space likewise inside the
    complement, the virtual space is the rest; each space is then paired with
    pair_subspace.  pair_op is the operator whose levels define the orbitals
    inside each space; use the effective Fock matrix (AO basis) so that the
    core and virtual blocks start diagonal, which the Davidson preconditioner
    relies on.  Default: kinetic + nuclear attraction.
    '''
    if s is None:
        s = mol.intor('int1e_ovlp_spinor')
    if pair_op is None:
        pair_op = mol.intor('int1e_kin_spinor') + mol.intor('int1e_nuc_spinor')
    n = mo.shape[1]
    a = mo.conj().T @ s @ time_reversal(mol, mo)      # T(C x) = C a x*

    def sym_top(p, k):
        m = 0.5 * (p + a @ p.conj() @ a.conj().T)
        w, u = numpy.linalg.eigh(0.5 * (m + m.conj().T))
        return u[:, ::-1][:, :k]

    pc = numpy.zeros((n, n))
    pc[:ncore, :ncore] = numpy.eye(ncore)
    uc = sym_top(pc, ncore)
    qc = numpy.eye(n) - uc @ uc.conj().T
    pa = numpy.zeros((n, n))
    pa[ncore:ncore+ncas, ncore:ncore+ncas] = numpy.eye(ncas)
    ua = sym_top(qc @ pa @ qc, ncas)
    qv = qc - ua @ ua.conj().T
    w, u = numpy.linalg.eigh(qv)
    uv = u[:, w > 0.5]
    assert uv.shape[1] == n - ncore - ncas
    return numpy.hstack([pair_subspace(mol, mo @ uc, pair_op, s),
                         pair_subspace(mol, mo @ ua, pair_op, s),
                         pair_subspace(mol, mo @ uv, pair_op, s)])
