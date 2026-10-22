'''
Brute-force internally contracted MRCI in the determinant basis.

The variational space is, by definition,

    internal:  the CAS determinants (internal='cas', the reference
               coefficients relax) or the model states (internal='states')
    external:  the span of Q tau |Psi_J> over all elementary one- and
               two-body strings tau of the correlated orbitals and all model
               states J (the union of their first-order interacting spaces),

every function expanded in the determinants of the MRCISD space
(zmr.reference.mrci_bruteforce).  A function Q tau |Psi_J> lies in a single
core/virtual occupation pattern; the functions of one pattern are
orthonormalized (metric eigenvalues below lindep dropped), the full
Hamiltonian is projected onto all of them and diagonalized densely.

Nothing of zmr.uc.classes or zmr.pc is used: the active operator sets
follow from the definition.  Validation only.
'''

import numpy

from socutils.zmr.reference import dets, mrci_bruteforce
from socutils.zmr.reference.nevpt2_bruteforce import CLASS_OF_EXCITATION, _masks, excitation_class
from socutils.zmr.reference.pc_bruteforce import ic_functions


class Result:
    '''e: energies; c: coefficients in the contracted basis (nbasis, nroots);
    basis: (ndet, nbasis) the contracted functions on the determinants strs;
    nbasis_by_class: functions per class ('cas' for the internal space);
    ndet: determinants of the uncontracted space.'''

    def __init__(self, e, c, basis, strs, nbasis_by_class):
        self.e, self.c, self.basis, self.strs = e, c, basis, strs
        self.nbasis_by_class = nbasis_by_class
        self.ndet = len(strs)


def contracted_basis(sp, nelec, civecs, internal='cas', lindep=1e-10):
    '''(strs, basis (ndet, nbasis), {class: number of functions})'''
    civecs = numpy.asarray(civecs)
    civecs = civecs[:, None] if civecs.ndim == 1 else civecs
    strs = numpy.unique(mrci_bruteforce.mrci_strings(sp, nelec)[0])
    cas = dets.cas_strings(sp.ncore, sp.nact, nelec)
    in_cas = dets._lookup(numpy.sort(cas), strs) >= 0
    fois = strs[~in_cas]
    _, act_mask, _ = _masks(sp)
    patterns, pattern_of = numpy.unique(fois & ~act_mask, return_inverse=True)
    holes, parts = excitation_class(fois, sp)
    psi = [dets.SparseVector(cas, c) for c in civecs.T]
    funcs = [ic_functions(p, fois, pattern_of, sp.nmo) for p in psi]

    columns = []
    count = {}
    cas_pos = dets._lookup(strs, numpy.sort(cas))
    if internal == 'cas':
        block = numpy.zeros((len(strs), len(cas)), dtype=numpy.complex128)
        block[cas_pos, numpy.arange(len(cas))] = 1.0
    elif internal == 'states':
        block = numpy.zeros((len(strs), len(psi)), dtype=numpy.complex128)
        for j, p in enumerate(psi):
            block[:, j] = p.on(strs)
        block = numpy.linalg.qr(block)[0]
    else:
        raise ValueError('internal must be cas or states')
    columns.append(block)
    count['cas'] = block.shape[1]
    fois_pos = dets._lookup(strs, fois)
    for ip in range(len(patterns)):
        idx = numpy.nonzero(pattern_of == ip)[0]
        span = [f[ip] for f in funcs if ip in f]
        if not span:
            continue
        span = numpy.concatenate(span, axis=1)
        s, q = numpy.linalg.eigh(span @ span.conj().T)
        q = q[:, s > lindep]
        if q.shape[1] == 0:
            continue
        block = numpy.zeros((len(strs), q.shape[1]), dtype=numpy.complex128)
        block[fois_pos[idx]] = q
        columns.append(block)
        name = CLASS_OF_EXCITATION[(holes[idx[0]], parts[idx[0]])]
        count[name] = count.get(name, 0) + q.shape[1]
    return strs, numpy.concatenate(columns, axis=1), count


def kernel(eris, nelec, civecs, nroots=1, internal='cas', lindep=1e-10):
    '''Lowest nroots internally contracted MRCI energies: Result.'''
    sp = eris.spaces
    strs, basis, count = contracted_basis(sp, nelec, civecs, internal, lindep)
    h_op = dets.hamiltonian_operator(eris.h1, eris.eri_block('pppp'), eris.ecore)
    hmat = dets.operator_matrix(h_op, strs)
    hc = basis.conj().T @ (hmat @ basis)
    e, c = numpy.linalg.eigh(0.5 * (hc + hc.conj().T))
    return Result(e[:nroots], c[:, :nroots], basis, strs, count)
