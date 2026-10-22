'''
Brute-force fully uncontracted (multi-state) NEVPT2.

The Dyall Hamiltonian is itself a second-quantized Hamiltonian with modified
integrals (conventions.md section 8), so H, H_D and V = H - H_D are built as
generic operators over explicit determinants of the whole correlated orbital
space.  The first-order equation

    (H_D - E0) psi1_J = Q V |Psi_J>

is solved directly with a sparse LU factorization in a determinant space
that contains every core/virtual occupation pattern reached by Q V |Psi_J>,
each combined with the complete active space of matching electron number
(the smallest space closed under H_D).  Nothing of the blocked production
algorithm is used.

Contributions are split into the eight perturber classes by the number of
core holes and virtual particles, which H_D conserves.
'''

import numpy
from scipy.sparse import linalg as splinalg

from socutils.zmr.reference import dets

# (core holes, virtual particles) -> class name
CLASS_OF_EXCITATION = {
    (2, 2): 'ccvv', (2, 1): 'ccav', (1, 2): 'cavv', (2, 0): 'ccaa',
    (0, 2): 'aavv', (1, 1): 'cava', (1, 0): 'caaa', (0, 1): 'aaav',
}
CLASSES = ('ccvv', 'ccav', 'cavv', 'ccaa', 'aavv', 'cava', 'caaa', 'aaav')


def dyall_operator(eris, eps_core, eps_virt):
    '''H_D = C + sum_i eps_i i^+ i + sum_a eps_a a^+ a + H_act as a generic
    operator (see dets.hamiltonian_operator) on the correlated orbitals.
    eris must be in quasi-canonical orbitals.'''
    sp = eris.spaces
    c, a, v = sp.slice('c'), sp.slice('a'), sp.slice('v')
    heff, aaaa, e_core = eris.active_hamiltonian()
    h_d = numpy.zeros((sp.nmo, sp.nmo), dtype=numpy.complex128)
    h_d[c, c] = numpy.diag(eps_core)
    h_d[v, v] = numpy.diag(eps_virt)
    h_d[a, a] = heff
    eri_d = numpy.zeros((sp.nmo,) * 4, dtype=numpy.complex128)
    eri_d[a, a, a, a] = aaaa
    const = e_core - numpy.sum(eps_core)
    return dets.hamiltonian_operator(h_d, eri_d, ecore=const)


def _masks(sp):
    core = (1 << sp.ncore) - 1
    act = ((1 << sp.nact) - 1) << sp.ncore
    virt = ((1 << sp.nvirt) - 1) << sp.nocc
    return numpy.uint64(core), numpy.uint64(act), numpy.uint64(virt)


def excitation_class(strs, sp):
    '''(core holes, virtual particles) of each determinant.'''
    core, _, virt = _masks(sp)
    strs = numpy.asarray(strs, dtype=numpy.uint64)
    holes = sp.ncore - numpy.bitwise_count(strs & core).astype(int)
    parts = numpy.bitwise_count(strs & virt).astype(int)
    return holes, parts


def first_order_space(rhs_strs, sp):
    '''Every core/virtual pattern in rhs_strs times the complete active space
    with the matching number of electrons.'''
    _, act, _ = _masks(sp)
    rhs_strs = numpy.asarray(rhs_strs, dtype=numpy.uint64)
    nelec = int(numpy.bitwise_count(rhs_strs[0]))
    outer = numpy.unique(rhs_strs & ~act)
    out = []
    for o in outer:
        na = nelec - int(numpy.bitwise_count(o))
        act_strs = dets.gen_strings(range(sp.ncore, sp.nocc), na)
        out.append(o | act_strs)
    return numpy.unique(numpy.concatenate(out))


class BruteForceNEVPT2:
    '''Brute-force UC-NEVPT2 for model states given as active-space CI vectors
    (columns, cistring order of the active orbitals).

    Args:
        eris: Hamiltonian in quasi-canonical orbitals
        eps_core, eps_virt: diagonal of the generalized Fock matrix
        nelec: number of active electrons
        civecs: (dim, nstate) CI vectors of the model states
        e0: 'state' or 'average' (conventions.md section 9)
        weights: state weights for e0='average' (default equal)
    '''

    def __init__(self, eris, eps_core, eps_virt, nelec, civecs, e0='state', weights=None):
        self.eris = eris
        self.sp = eris.spaces
        self.nelec = nelec
        civecs = numpy.asarray(civecs)
        self.civecs = civecs[:, None] if civecs.ndim == 1 else civecs
        self.e0 = e0
        nstate = self.civecs.shape[1]
        self.weights = numpy.full(nstate, 1.0 / nstate) if weights is None else numpy.asarray(weights)
        self.h_op = dets.hamiltonian_operator(eris.h1, eris.eri_block('pppp'), eris.ecore)
        self.hd_op = dyall_operator(eris, eps_core, eps_virt)

    def model_vectors(self):
        cas = dets.cas_strings(self.sp.ncore, self.sp.nact, self.nelec)
        return [dets.SparseVector(cas, c) for c in self.civecs.T]

    def kernel(self):
        '''Returns (heff, h2_by_class):
            heff: (nstate, nstate) second-order effective Hamiltonian
            h2_by_class: dict class -> (nstate, nstate) contribution to H2
        '''
        sp = self.sp
        psi = self.model_vectors()
        nstate = len(psi)
        cas_strs = psi[0].strs
        hmodel = numpy.array([[pi.vdot(dets.apply_operator(self.h_op, pj)) for pj in psi]
                              for pi in psi])
        self.hmodel = hmodel
        e_cas = hmodel.diagonal().real

        # V |J>, projected onto the complement of the CAS space
        vpsi = []
        for pj in psi:
            v = dets.apply_operator(self.h_op, pj) - dets.apply_operator(self.hd_op, pj)
            outside = v.lookup(cas_strs)
            self.cas_leak = max(getattr(self, 'cas_leak', 0.0),
                                float(abs(v.on(cas_strs)).max(initial=0)))
            keep = numpy.ones(len(v), dtype=bool)
            keep[outside[outside >= 0]] = False
            vpsi.append(dets.SparseVector(v.strs[keep], v.coeffs[keep]))

        fois = first_order_space(numpy.concatenate([v.strs for v in vpsi]), sp)
        self.nfois = len(fois)
        hd = dets.operator_matrix(self.hd_op, fois)
        eye = _sparse_eye(len(fois))
        rhs = numpy.stack([v.on(fois) for v in vpsi], axis=1)     # (nfois, nstate)

        if self.e0 == 'state':
            e0_list = e_cas
        elif self.e0 == 'average':
            e0_list = numpy.full(nstate, numpy.dot(self.weights, e_cas))
        else:
            raise ValueError('e0 must be "state" or "average"')

        # x_J = (H_D - E0_J)^-1 Q V |J>,  <I|V R(E0_J) V|J> = -rhs_I^+ x_J
        x = numpy.empty_like(rhs)
        for j in range(nstate):
            lu = splinalg.splu((hd - e0_list[j] * eye).tocsc())
            x[:, j] = lu.solve(rhs[:, j])

        holes, parts = excitation_class(fois, sp)
        h2_by_class = {}
        for key, name in CLASS_OF_EXCITATION.items():
            m = (holes == key[0]) & (parts == key[1])
            a = -rhs[m].conj().T @ x[m]              # a[I, J] = <I|V R(E0_J) V|J>
            if self.e0 == 'state':
                a = 0.5 * (a + a.conj().T)           # 1/2 [<I|VR(E_J)V|J> + <I|VR(E_I)V|J>]
            h2_by_class[name] = a
        classified = sum(((holes == k[0]) & (parts == k[1])).sum() for k in CLASS_OF_EXCITATION)
        if classified != len(fois):
            raise RuntimeError('first-order space contains unclassified determinants')

        h2 = sum(h2_by_class.values())
        self.e_cas = e_cas
        self.h2 = h2
        self.h2_by_class = h2_by_class
        self.heff = hmodel + h2
        self.energies = numpy.linalg.eigvalsh(self.heff)
        return self.heff, h2_by_class


def _sparse_eye(n):
    from scipy import sparse
    return sparse.identity(n, dtype=numpy.complex128, format='csr')
