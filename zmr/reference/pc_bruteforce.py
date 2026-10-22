'''
Brute-force partially contracted (internally contracted) NEVPT2.

The first-order space of model state J is, by definition, the span of
Q tau |Psi_J> over all elementary one- and two-body operator strings
tau = p^+ q and p^+ r^+ s q of the correlated orbitals.  Each such function
lies in a single core/virtual occupation pattern, H_D does not couple
different patterns, so the Hylleraas problem is solved pattern by pattern:
the functions of a pattern are orthonormalized (metric eigenvalues below
lindep are dropped), H_D (the generic sparse operator of
nevpt2_bruteforce.dyall_operator) is projected onto them and the linear
equation is solved directly.

Nothing of zmr.uc.classes or zmr.pc is used: the active operator sets of
the perturber classes are not assumed, they follow from the definition.

contraction='union' uses the functions of all model states for every ket,
'state' only those of the ket state; the second-order matrix is Hermitized
like in zmr.pc.nevpt2.
'''

import numpy

from socutils.zmr.reference import dets
from socutils.zmr.reference.nevpt2_bruteforce import (BruteForceNEVPT2, CLASS_OF_EXCITATION,
                                             _masks, excitation_class, first_order_space)


def _elementary_strings(nmo):
    one = numpy.indices((nmo, nmo)).reshape(2, -1).T
    two = numpy.indices((nmo,) * 4).reshape(4, -1).T
    return [((True, False), one), ((True, True, False, False), two)]


def ic_functions(psi, fois, pattern_of, nmo):
    '''The functions Q tau |psi> for every elementary string tau, grouped by
    the core/virtual occupation pattern they lie in (each lies in exactly
    one).  Returns {pattern index: dense (n_rows_of_pattern, n_functions)}
    with rows in the order of numpy.nonzero(pattern_of == pattern).'''
    rows, cols, vals = [], [], []
    ncol = 0
    for daggers, orbs in _elementary_strings(nmo):
        for t0 in range(0, len(orbs), 4096):
            o = orbs[t0:t0 + 4096]
            new, sign = dets.apply_opstring(daggers, o, psi.strs)
            pos = dets._lookup(fois, new)
            keep = (sign != 0) & (pos >= 0)
            term = numpy.broadcast_to(numpy.arange(len(o))[:, None], new.shape)
            rows.append(pos[keep])
            cols.append(term[keep] + ncol)
            vals.append((sign * psi.coeffs[None, :])[keep])
            ncol += len(o)
    rows = numpy.concatenate(rows)
    cols = numpy.concatenate(cols)
    vals = numpy.concatenate(vals)
    pat = pattern_of[rows]
    # every column must stay inside one pattern
    first = numpy.full(ncol, -1)
    first[cols] = pat
    if (first[cols] != pat).any():
        raise RuntimeError('an operator string mixes occupation patterns')
    local_row = numpy.empty(len(pattern_of), dtype=numpy.int64)
    out = {}
    order = numpy.argsort(pat, kind='stable')
    bounds = numpy.searchsorted(pat[order], numpy.arange(pattern_of.max() + 2))
    for ip in range(pattern_of.max() + 1):
        sel = order[bounds[ip]:bounds[ip + 1]]
        if len(sel) == 0:
            continue
        idx = numpy.nonzero(pattern_of == ip)[0]
        local_row[idx] = numpy.arange(len(idx))
        ucol, lcol = numpy.unique(cols[sel], return_inverse=True)
        block = numpy.zeros((len(idx), len(ucol)), dtype=numpy.complex128)
        numpy.add.at(block, (local_row[rows[sel]], lcol), vals[sel])
        out[ip] = block
    return out


class BruteForcePCNEVPT2(BruteForceNEVPT2):
    '''Brute-force PC-NEVPT2; arguments as BruteForceNEVPT2 plus
    contraction ('union' or 'state') and lindep.'''

    def __init__(self, eris, eps_core, eps_virt, nelec, civecs, e0='state', weights=None,
                 contraction='union', lindep=1e-10):
        BruteForceNEVPT2.__init__(self, eris, eps_core, eps_virt, nelec, civecs, e0, weights)
        self.contraction = contraction
        self.lindep = lindep

    def kernel(self):
        sp = self.sp
        psi = self.model_vectors()
        nstate = len(psi)
        cas_strs = psi[0].strs
        hmodel = numpy.array([[pi.vdot(dets.apply_operator(self.h_op, pj)) for pj in psi]
                              for pi in psi])
        e_cas = hmodel.diagonal().real

        vpsi = []
        for pj in psi:
            v = dets.apply_operator(self.h_op, pj) - dets.apply_operator(self.hd_op, pj)
            pos = v.lookup(cas_strs)
            keep = numpy.ones(len(v), dtype=bool)
            keep[pos[pos >= 0]] = False
            vpsi.append(dets.SparseVector(v.strs[keep], v.coeffs[keep]))
        fois = first_order_space(numpy.concatenate([v.strs for v in vpsi]), sp)
        hd = dets.operator_matrix(self.hd_op, fois).tocsr()
        rhs = numpy.stack([v.on(fois) for v in vpsi], axis=1)

        if self.e0 == 'state':
            e0_list = e_cas
        else:
            e0_list = numpy.full(nstate, numpy.dot(self.weights, e_cas))

        _, act_mask, _ = _masks(sp)
        outer = fois & ~act_mask
        patterns, pattern_of = numpy.unique(outer, return_inverse=True)
        holes, parts = excitation_class(fois, sp)
        funcs = [ic_functions(p, fois, pattern_of, sp.nmo) for p in psi]

        a_by_class = {name: numpy.zeros((nstate, nstate), dtype=numpy.complex128)
                      for name in CLASS_OF_EXCITATION.values()}
        self.rhs_outside = 0.0
        for ip in range(len(patterns)):
            idx = numpy.nonzero(pattern_of == ip)[0]
            name = CLASS_OF_EXCITATION[(holes[idx[0]], parts[idx[0]])]
            hp = hd[idx][:, idx].toarray()
            rp = rhs[idx]
            blocks = [f.get(ip, numpy.zeros((len(idx), 0))) for f in funcs]
            if self.contraction == 'union':
                spans = [numpy.concatenate(blocks, axis=1)] * nstate
            else:
                spans = blocks
            for j in range(nstate):
                g = spans[j] @ spans[j].conj().T
                s, q = numpy.linalg.eigh(g)
                q = q[:, s > self.lindep]
                if q.shape[1] == 0:
                    continue
                # V|J> lies in the span of its own contracted functions
                rj = rp[:, j]
                self.rhs_outside = max(self.rhs_outside,
                                       numpy.linalg.norm(rj - q @ (q.conj().T @ rj)))
                k = q.conj().T @ hp @ q - e0_list[j] * numpy.eye(q.shape[1])
                y = numpy.linalg.solve(k, q.conj().T @ rj)
                a_by_class[name][:, j] -= rp.conj().T @ (q @ y)

        hermitize = self.e0 == 'state' or self.contraction == 'state'
        h2_by_class = {}
        for name, a in a_by_class.items():
            h2_by_class[name] = 0.5 * (a + a.conj().T) if hermitize else a
        self.e_cas = e_cas
        self.hmodel = hmodel
        self.h2_by_class = h2_by_class
        self.h2 = sum(h2_by_class.values())
        self.heff = hmodel + self.h2
        self.energies = numpy.linalg.eigvalsh(self.heff)
        return self.heff, h2_by_class
