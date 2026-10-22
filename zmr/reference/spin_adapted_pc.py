'''
Spin-adapted PC-NEVPT2 functions of a non-relativistic singlet reference,
with an explicit threshold on the eigenvalues of their metric.

Angeli's PC-NEVPT2 (ORCA: FIC-NEVPT2) builds the first-order space of every
external label from products of spin-free excitation operators
E_pq = sum_s p_s^+ q_s acting on the reference, e.g.
E_ai E_tu |0> and E_ti E_au |0> for the cava class.  Their span in the
singlet sector equals the spin-orbital span used by zmr, so with a threshold
that removes only the exact null space both give the same energy; a larger
threshold removes physical functions with small norm.  ORCA 6.0.1 removes
spin-free functions with metric eigenvalues below 1e-6 (not affected by the
NThresh or DThresh keywords), and this module reproduces its numbers with
thresh=1e-6.

Brute force in the determinant space of one class's occupation patterns; the
spin-orbitals must be p = 2P + s within each space (no alpha/beta mixing),
in quasi-canonical orbitals.  Validation only.
'''

import itertools

import numpy

from socutils.zmr.reference import dets
from socutils.zmr.reference.nevpt2_bruteforce import dyall_operator, excitation_class, first_order_space

CLASS_PATTERN = {'cava': (1, 1), 'aavv': (0, 2), 'aaav': (0, 1), 'caaa': (1, 0)}


def _blocks(cls, nc, na, nv):
    '''Independent blocks of spin-free products E_pq E_rs, each given as a
    list of ((p, q), (r, s)) with space-tagged spatial indices, and the number
    of (ordered) external labels the block represents.'''
    act = range(na)
    if cls == 'cava':
        for i in range(nc):
            for a in range(nv):
                yield ([((('v', a), ('c', i)), (('a', t), ('a', u))) for t in act for u in act]
                       + [((('a', t), ('c', i)), (('v', a), ('a', u))) for t in act for u in act]), 1
    elif cls == 'aavv':
        for a in range(nv):
            for b in range(a, nv):
                yield [((('v', a), ('a', t)), (('v', b), ('a', u))) for t in act for u in act], \
                    (1 if a == b else 2)
    elif cls == 'aaav':
        for a in range(nv):
            yield [((('v', a), ('a', t)), (('a', u), ('a', w))) for t in act for u in act for w in act], 1
    elif cls == 'caaa':
        for i in range(nc):
            yield [((('a', t), ('c', i)), (('a', u), ('a', w))) for t in act for u in act for w in act], 1
    else:
        raise KeyError(cls)


def class_energy(eris, civec, nelec, eps_core, eps_virt, cls, thresholds=(1e-12,)):
    '''Second-order energy of one class with spin-adapted PC functions.

    Returns a list of (threshold, energy, skipped, total) where skipped and
    total count the functions per ordered external label as ORCA prints them.'''
    sp = eris.spaces
    offset = {'c': 0, 'a': sp.ncore, 'v': sp.nocc}
    nc, na, nv = sp.ncore // 2, sp.nact // 2, sp.nvirt // 2

    hop = dets.hamiltonian_operator(eris.h1, eris.eri_block('pppp'), eris.ecore)
    hdop = dyall_operator(eris, eps_core, eps_virt)
    psi = dets.SparseVector(dets.cas_strings(sp.ncore, sp.nact, nelec), civec)
    e0 = psi.vdot(dets.apply_operator(hop, psi)).real
    v = dets.apply_operator(hop, psi) - dets.apply_operator(hdop, psi)
    fois = first_order_space(v.strs, sp)
    holes, parts = excitation_class(fois, sp)
    want = CLASS_PATTERN[cls]
    sel = fois[(holes == want[0]) & (parts == want[1])]
    hd = dets.operator_matrix(hdop, sel).tocsr()
    rhs = v.on(sel)

    def so(idx, s):
        space, p = idx
        return offset[space] + 2 * p + s

    def functions(block):
        '''columns E_pq E_rs |psi> on sel, all products of the block at once'''
        orbs, func = [], []
        for f, ((p, q), (r, s_)) in enumerate(block):
            for s1, s2 in itertools.product(range(2), repeat=2):
                orbs.append([so(p, s1), so(q, s1), so(r, s2), so(s_, s2)])
                func.append(f)
        new, sign = dets.apply_opstring((True, False, True, False), orbs, psi.strs)
        pos = dets._lookup(sel, new)
        keep = (sign != 0) & (pos >= 0)
        out = numpy.zeros((len(sel), len(block)), dtype=numpy.complex128)
        cols = numpy.broadcast_to(numpy.asarray(func)[:, None], new.shape)
        numpy.add.at(out, (pos[keep], cols[keep]), (sign * psi.coeffs[None, :])[keep])
        return out

    data = [(functions(block), mult) for block, mult in _blocks(cls, nc, na, nv)]
    results = []
    for thresh in thresholds:
        e, nskip, ntot = 0.0, 0, 0
        for f, mult in data:
            w, x = numpy.linalg.eigh(f.conj().T @ f)
            keep = w > thresh
            nskip += int((~keep).sum()) * mult
            ntot += f.shape[1] * mult
            q = f @ (x[:, keep] / numpy.sqrt(w[keep]))
            k = q.conj().T @ (hd @ q) - e0 * numpy.eye(q.shape[1])
            r = q.conj().T @ rhs
            e -= (r.conj() @ numpy.linalg.solve(k, r)).real
        results.append((thresh, e, nskip, ntot))
    return results
