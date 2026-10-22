'''
Active space of spin-restricted orbitals.

Sectors are named by (n_alpha, n_beta).  A determinant is the bit string
(alpha string << norb) | beta string: orbital p has the spin-orbital index p
for beta and norb + p for alpha, and the address of a determinant is
rank(alpha) * dim(beta) + rank(beta).  Order and phases are those of
pyscf.fci, so its CI vectors (reshaped to one dimension) can be used
directly.  Operator strings are given in spin-orbital indices (so()).

    H_act = sum_tu heff_tu E_tu + 1/2 sum_tuvw (tu|vw) (E_tu E_vw - delta_uv E_tw)
'''

import math

import numpy

from socutils.zmr.active import sectors
from socutils.zmr.active import strings as st
from socutils.zmr.lib import kernels

ALPHA, BETA = 0, 1


def so(orbital, spin, norb):
    '''Spin-orbital index of a spatial orbital (array) with the given spin.'''
    return numpy.asarray(orbital) + (norb if spin == ALPHA else 0)


def sector_index(norb, nelec):
    '''Positions of the (n_alpha, n_beta) determinants among all strings with
    n_alpha + n_beta electrons in the 2 norb spin-orbitals (ascending).'''
    na, nb = nelec
    strs = st.gen_strings(2 * norb, na + nb)
    return numpy.nonzero(numpy.bitwise_count(strs >> numpy.uint64(norb)) == na)[0]


def spin_orbital_hamiltonian(heff, aaaa):
    '''(h, eri) over the spin-orbitals [beta block, alpha block].'''
    n = heff.shape[0]
    h = numpy.zeros((2 * n, 2 * n))
    eri = numpy.zeros((2 * n,) * 4)
    for a in (slice(0, n), slice(n, 2 * n)):
        h[a, a] = heff
        for b in (slice(0, n), slice(n, 2 * n)):
            eri[a, a, b, b] = aaaa
    return h, eri


class ActiveSpace(sectors.SectorSpace):
    '''Active-space Hamiltonian (heff, aaaa) over spatial orbitals plus the
    constant E_core.

    Sectors of dimension up to dense_limit keep a dense Hamiltonian.  Larger
    ones are matrix free: H_act is only applied to vectors (apply_h), which
    is all PC-NEVPT2 with given model states needs; the full spectrum (UC)
    and the built-in solver for the model states are not available there.'''

    dense_limit = 8000

    def __init__(self, heff, aaaa, e_core=0.0):
        sectors.SectorSpace.__init__(self)
        self.norb = heff.shape[0]
        if 2 * self.norb > st.MAX_ORB:
            raise NotImplementedError('more than %d active orbitals' % (st.MAX_ORB // 2))
        self.heff = numpy.asarray(heff, dtype=float)
        self.aaaa = numpy.asarray(aaaa, dtype=float).reshape((self.norb,) * 4)
        self.e_core = float(e_core)
        self._spinor = None

    @classmethod
    def from_eris(cls, eris):
        return cls(*eris.active_hamiltonian())

    def dim(self, nelec):
        na, nb = nelec
        if min(na, nb) < 0 or max(na, nb) > self.norb:
            return 0
        return math.comb(self.norb, na) * math.comb(self.norb, nb)

    def target(self, nelec, dn):
        return (nelec[0] + dn[0], nelec[1] + dn[1])

    def blocks(self, nelec):
        '''A single block: no structure is looked for inside a sector
        (n_alpha, n_beta).'''
        if nelec not in self._blocks:
            self._blocks[nelec] = (numpy.zeros(self.dim(nelec), dtype=numpy.int64), 1)
        return self._blocks[nelec]

    def spin_change(self, daggers, ops):
        '''(d n_alpha, d n_beta) of the operator strings; all rows of ops must
        agree.'''
        ops = numpy.asarray(ops).reshape(len(ops), len(daggers))
        sign = numpy.where(numpy.asarray(daggers, dtype=bool), 1, -1)
        alpha = ops >= self.norb
        da = (sign * alpha).sum(axis=1)
        db = (sign * ~alpha).sum(axis=1)
        if len(ops) == 0:
            return 0, 0
        if (da != da[0]).any() or (db != db[0]).any():
            raise ValueError('operator strings of one block lead to different sectors')
        return int(da[0]), int(db[0])

    # ------------------------------------------------------------------
    # reference implementation through the spinor code (no libzmr)

    def _embedded(self):
        if self._spinor is None:
            self._spinor = sectors.ActiveSpace(*spin_orbital_hamiltonian(self.heff, self.aaaa))
        return self._spinor

    def make_hmat_reference(self, nelec):
        idx = sector_index(self.norb, nelec)
        h = self._embedded().make_hmat_numpy(sum(nelec))
        return numpy.ascontiguousarray(h[numpy.ix_(idx, idx)].real)

    def _apply_reference(self, vecs, nelec, ntarget, daggers, ops, out, x0):
        n = self.norb
        src = sector_index(n, nelec)
        dst = sector_index(n, ntarget)
        full = numpy.zeros((vecs.shape[0], len(st.gen_strings(2 * n, sum(nelec)))),
                           dtype=numpy.complex128)
        full[:, src] = vecs
        res = kernels.apply_strings(2 * n, sum(nelec), daggers, ops, full)
        out[:, x0:x0 + len(ops)] = res[:, :, dst].real

    # ------------------------------------------------------------------

    def is_dense(self, nelec):
        return self.dim(nelec) <= self.dense_limit

    def make_hmat(self, nelec):
        '''Dense H_act in the (n_alpha, n_beta) sector.'''
        if self.dim(nelec) == 0:
            return numpy.zeros((0, 0))
        if not self.is_dense(nelec):
            raise NotImplementedError(
                'sector %s has %d determinants (dense_limit = %d): its Hamiltonian is not '
                'stored.  PC-NEVPT2 with given model states (civecs) works matrix free; '
                'UC-NEVPT2 and nstate need the dense sector.'
                % (nelec, self.dim(nelec), self.dense_limit))
        hmat = kernels.build_hmat_ab(self.norb, nelec, self.heff, self.aaaa)
        if hmat is None:
            hmat = self.make_hmat_reference(nelec)
        return hmat

    def apply_h(self, nelec, q):
        '''H_act q for the columns of q (dim, m); matrix free above
        dense_limit.'''
        if self.is_dense(nelec):
            return sectors.SectorSpace.apply_h(self, nelec, q)
        q = numpy.asarray(q)
        parts = [q.real] + ([q.imag] if numpy.iscomplexobj(q) else [])
        out = [kernels.sigma_ab(self.norb, nelec, self.heff, self.aaaa, x) for x in parts]
        if out[0] is None:
            # without libzmr: pyscf.fci, vector by vector
            out = [kernels.sigma_ab_pyscf(self.norb, nelec, self.heff, self.aaaa, x)
                   for x in parts]
        return out[0] if len(out) == 1 else out[0] + 1j * out[1]

    def apply_strings(self, vecs, nelec, blocks):
        '''Operator strings on CI vectors: vecs (..., dim) -> out (..., nX, dim'),
        out[..., X, :] = O_X vecs.  blocks: list of (daggers, ops) with ops in
        spin-orbital indices (so()); all blocks must lead to the same sector.'''
        vecs = numpy.asarray(vecs)
        lead = vecs.shape[:-1]
        ops = [numpy.asarray(o, dtype=numpy.int64).reshape(len(o), len(dag)) for dag, o in blocks]
        changes = {self.spin_change(dag, o) for (dag, _), o in zip(blocks, ops) if len(o)}
        if len(changes) > 1:
            raise ValueError('operator blocks lead to different sectors')
        ntarget = self.target(nelec, changes.pop() if changes else (0, 0))
        nx = sum(len(o) for o in ops)
        v2 = vecs.reshape(-1, vecs.shape[-1])
        parts = [numpy.ascontiguousarray(v2.real)]
        if numpy.iscomplexobj(v2) and abs(v2.imag).max(initial=0) > 0:
            parts.append(numpy.ascontiguousarray(v2.imag))
        outs = []
        for v in parts:
            out = numpy.zeros((v.shape[0], nx, self.dim(ntarget)))
            x0 = 0
            for (dag, _), o in zip(blocks, ops):
                if out.size and len(o):
                    done = kernels.apply_strings_ab(self.norb, nelec, ntarget, dag, o, v, out, x0)
                    if not done:
                        self._apply_reference(v, nelec, ntarget, dag, o, out, x0)
                x0 += len(o)
            outs.append(out)
        out = outs[0] if len(outs) == 1 else outs[0] + 1j * outs[1]
        return out.reshape(lead + out.shape[1:])

    # ------------------------------------------------------------------
    # model space helpers

    def trans_rdm1(self, bra, ket, nelec):
        '''Spin-summed t[p, q] = <bra| E_pq |ket>'''
        n = self.norb
        d = 0
        for spin in (ALPHA, BETA):
            ops = so(numpy.arange(n), spin, n)[:, None]
            db = self.apply_strings(bra, nelec, [((False,), ops)])
            dk = self.apply_strings(ket, nelec, [((False,), ops)])
            d = d + db.conj() @ dk.T
        return d

    def averaged_rdm1(self, civecs, nelec, weights=None):
        '''sum_I w_I <Psi_I| E_pq |Psi_I> for CI vectors given as columns.'''
        civecs = numpy.asarray(civecs)
        if civecs.ndim == 1:
            civecs = civecs[:, None]
        nstate = civecs.shape[1]
        if weights is None:
            weights = numpy.full(nstate, 1.0 / nstate)
        g = 0
        for w, c in zip(weights, civecs.T):
            g = g + w * self.trans_rdm1(c, c, nelec)
        return g.real if numpy.iscomplexobj(g) and abs(g.imag).max(initial=0) < 1e-14 else g
