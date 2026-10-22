'''
Electron-repulsion integrals and the one-electron part of the Hamiltonian in
a correlated spin-orbital / spinor basis ("eris", as in PySCF cc/mcscf):

    H = ecore + sum_pq h_pq p^+ q + 1/2 sum_pqrs (pq|rs) p^+ r^+ s q

Blocks of (pq|rs) are read as attributes named by four orbital-space letters,
as in PySCF mcscf (eris.ppaa): c = core, a = active, v = virtual, p = all
correlated orbitals, e.g. eris.cvcv or eris.aaaa; eris.eri_block(labels) is
the same for labels built at run time.  Mean-field quantities (generalized
Fock matrix, core energy) are methods of the eris object, so that the
integral source can change without touching any caller (conventions.md
section 2).  Two implementations:

    ERIs     stored four-index array (model Hamiltonians, small tests)
    MOERIs   blocks transformed on demand from MO coefficients and cached;
             mean-field terms built in the AO basis.  Density fitting will be
             another source of blocks.
'''

import numpy

from socutils.zmr.lib import linalg as la
from socutils.zmr.spaces import OrbitalSpaces


class ERIsBase:
    '''Operations shared by every integral source.

    Subclasses provide ecore, h1, spaces, eri_block(labels), rotate(u),
    with_virtuals(u), generalized_fock(gamma_act) and core_energy().

    active_charges: symmetry charges of the active orbitals, set by
    zmr.active.symmetry.adapt (None: no symmetry is used).'''

    active_charges = None
    dtype = numpy.complex128

    def __getattr__(self, name):
        # eris.cvcv etc.; only called for attributes that do not exist
        if len(name) == 4 and set(name) <= set('cavp'):
            return self.eri_block(name)
        raise AttributeError(name)

    @property
    def nmo(self):
        return self.spaces.nmo

    def h1_block(self, labels):
        return self.h1[self.spaces.slice(labels[0]), self.spaces.slice(labels[1])]

    def active_hamiltonian(self):
        '''(h_eff, aaaa, E_core) of the active-space Hamiltonian, with
        h_eff_tu = h_tu + sum_i [(tu|ii) - (ti|iu)] (conventions.md section 8).'''
        a = self.spaces.slice('a')
        na = self.spaces.nact
        heff = self.generalized_fock(numpy.zeros((na, na)))[a, a]
        return heff, self.eri_block('aaaa'), self.core_energy()

    def symmetry_error(self):
        '''Largest violation of h = h^+, (pq|rs) = (rs|pq),
        (pq|rs)^* = (qp|sr).  Diagnostic only: builds the full (pp|pp) block.'''
        eri = self.eri_block('pppp')
        return max(abs(self.h1 - self.h1.conj().T).max(),
                   abs(eri - eri.transpose(2, 3, 0, 1)).max(),
                   abs(eri.conj() - eri.transpose(1, 0, 3, 2)).max())

    def _keep_charges(self, new, u):
        '''Carry the active charges over to the rotated integrals if the
        rotation leaves the active orbitals alone.'''
        a = self.spaces.slice('a')
        unchanged = numpy.array_equal(u[a, a], numpy.eye(self.spaces.nact))
        if self.active_charges is not None and unchanged:
            new.active_charges = self.active_charges
        return new

    def _virtual_map(self, u):
        '''(nmo, nocc + n) map that keeps core and active orbitals and takes
        the columns of u (nvirt, n) as the new virtual orbitals, and the
        corresponding OrbitalSpaces.'''
        sp = self.spaces
        u = numpy.asarray(u, dtype=self.dtype)
        if u.ndim != 2 or u.shape[0] != sp.nvirt or u.shape[1] > sp.nvirt:
            raise ValueError('u must have shape (nvirt, n <= nvirt)')
        if abs(u.conj().T @ u - numpy.eye(u.shape[1])).max(initial=0) > 1e-10:
            raise ValueError('the columns of u are not orthonormal')
        full = numpy.zeros((sp.nmo, sp.nocc + u.shape[1]), dtype=self.dtype)
        full[:sp.nocc, :sp.nocc] = numpy.eye(sp.nocc)
        full[sp.nocc:, sp.nocc:] = u
        return full, OrbitalSpaces(sp.ncore, sp.nact, u.shape[1])

    def _check_block_diagonal(self, u):
        sp = self.spaces
        mask = numpy.zeros((sp.nmo, sp.nmo), dtype=bool)
        for label in 'cav':
            s = sp.slice(label)
            mask[s, s] = True
        if abs(u[~mask]).max(initial=0) > 1e-12:
            raise ValueError('orbital rotation mixes core, active and virtual spaces')


class ERIs(ERIsBase):
    '''Hamiltonian with stored four-index integrals.

    Attributes:
        ecore: scalar energy (nuclear repulsion plus frozen core)
        h1: (nmo, nmo) complex, frozen-core dressed one-electron matrix
        spaces: OrbitalSpaces
    '''

    def __init__(self, ecore, h1, eri, spaces):
        nmo = spaces.nmo
        self.ecore = float(numpy.real(ecore))
        self.h1 = numpy.asarray(h1, dtype=numpy.complex128).reshape(nmo, nmo)
        self._eri = numpy.asarray(eri, dtype=numpy.complex128).reshape((nmo,) * 4)
        self.spaces = spaces

    def eri_block(self, labels):
        '''(pq|rs) with p, q, r, s restricted to the spaces named by the four
        characters of labels (c, a, v or p), e.g. eri_block('cvcv').'''
        if len(labels) != 4:
            raise ValueError('eri_block needs four space labels')
        sl = tuple(self.spaces.slice(x) for x in labels)
        return self._eri[sl]

    def rotate(self, u):
        '''Hamiltonian in the orbitals phi'_p = sum_q phi_q u_qp for a unitary
        u.  The caller is responsible for u respecting the space partition.'''
        u = numpy.asarray(u, dtype=numpy.complex128)
        h1 = u.conj().T @ self.h1 @ u
        eri = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', self._eri, u.conj(), u, u.conj(), u,
                           optimize=True)
        return self._keep_charges(ERIs(self.ecore, h1, eri, self.spaces), u)

    def with_virtuals(self, u):
        '''Hamiltonian with the virtual orbitals phi'_a = sum_b phi_b u_ba for
        u of shape (nvirt, n) with orthonormal columns; virtual orbitals
        outside their span leave the correlated space.'''
        full, spaces = self._virtual_map(u)
        h1 = full.conj().T @ self.h1 @ full
        eri = numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', self._eri, full.conj(), full, full.conj(),
                           full, optimize=True)
        new = ERIs(self.ecore, h1, eri, spaces)
        new.active_charges = self.active_charges
        return new

    def core_energy(self):
        '''E_core = ecore + sum_i h_ii + 1/2 sum_ij [(ii|jj) - (ij|ji)]'''
        c = self.spaces.slice('c')
        cccc = self.eri_block('cccc')
        e = (self.ecore + numpy.trace(self.h1[c, c])
             + 0.5 * (numpy.einsum('iijj->', cccc) - numpy.einsum('ijji->', cccc)))
        return float(e.real)

    def generalized_fock(self, gamma_act):
        '''f_pq = h_pq + sum_i [(pq|ii) - (pi|iq)] + sum_tu [(pq|tu) - (pu|tq)] gamma_tu'''
        sp = self.spaces
        f = self.h1.copy()
        if sp.ncore:
            f += (numpy.einsum('pqii->pq', self.eri_block('ppcc'))
                  - numpy.einsum('piiq->pq', self.eri_block('pccp')))
        if sp.nact:
            f += (numpy.einsum('pqtu,tu->pq', self.eri_block('ppaa'), gamma_act)
                  - numpy.einsum('putq,tu->pq', self.eri_block('paap'), gamma_act))
        return f


class MOERIs(ERIsBase):
    '''Integrals generated on demand from MO coefficients.

    Args:
        ecore: nuclear repulsion plus frozen-core energy
        hcore_ao: one-electron operator in the AO basis, frozen-core dressed
        mo_coeff: (nao, nmo) coefficients of the correlated orbitals
        spaces: OrbitalSpaces
        eri_fn: eri_fn(c1, c2, c3, c4) -> (pq|rs) array for four coefficient
            blocks (chemists' notation, p and q on electron 1)
        veff_fn: veff_fn(dm_ao) -> J[dm] - K[dm] in the AO basis, with
            vj[m, n] = sum_ls (mn|ls) dm[s, l]

    eri blocks are cached; a block whose transpose partner is cached is taken
    from it via (pq|rs) = (rs|pq) or (pq|rs)^* = (qp|sr), and a block inside a
    prefetched superblock is sliced from it.  The transformed blocks are
    recorded in self.computed_blocks.

    Factorized sources (eri_fn.pair, zmr.interface.cholesky) keep three-index
    integrals instead, laid out as in socutils.mcscf.zmc_ao2mo._CDERIS:

        cd_pa[L, p, t]   all correlated orbitals x active   (naux, nmo, nact)
        cd_pc[L, p, i]   all correlated orbitals x core     (naux, nmo, ncore)

    each one half transformation with the few active or core orbitals;
    cd_vc, cd_block('va'), ('ca'), ('aa'), ... are slices of them, the
    reversed pairs their conjugate transposes.

    The generalized Fock matrix and the core energy do not need them: one
    streamed pass over the vectors with the occupied orbitals gives the
    Coulomb and exchange matrices of the core and the half-transformed
    active vectors (eri_fn.occupied, as socutils' _CDERIS builds vk_core).
    These do not change under rotations among the core or among the virtual
    orbitals, so the quasi-canonical orbitals are found first and cd_pc is
    built once, for them; cd_pa is the second half transformation of the
    stored active vectors.  Nothing is rotated or transposed afterwards.  A
    four-index block is the product of two of them, formed when it is asked
    for; prefetch builds only cd_pa and cd_vc, and release_blocks() drops
    the four-index blocks (the NEVPT2 driver calls it after every class: no
    class needs the blocks of another one).  The doubly external classes do
    not form their blocks at all but multiply the slices of cd_vc for one
    core orbital (pair) at a time (zmr.uc.classes).
    '''

    def __init__(self, ecore, hcore_ao, mo_coeff, spaces, eri_fn, veff_fn):
        if mo_coeff.shape[1] != spaces.nmo:
            raise ValueError('mo_coeff does not match the orbital spaces')
        self.ecore = float(numpy.real(ecore))
        self.hcore_ao = hcore_ao
        self.mo_coeff = numpy.asarray(mo_coeff)
        self.spaces = spaces
        self.eri_fn = eri_fn
        self.veff_fn = veff_fn
        self.h1 = self.mo_coeff.conj().T @ hcore_ao @ self.mo_coeff
        self._cache = {}
        self._super = []
        self.computed_blocks = []
        self.cd = {}
        self._cd_pa = None
        self._cd_pc = None
        self._occ = None

    @property
    def factorized(self):
        return hasattr(self.eri_fn, 'pair')

    # occupation of a core orbital and weight of the active exchange
    # (spinors; zmr.nr: 2 and 1/2)
    _docc = 1
    _kact = 1.0

    def occupied(self):
        '''Core Coulomb / exchange matrices and half-transformed active
        vectors (cached; shared with objects that differ by rotations among
        the core or the virtual orbitals).'''
        if self._occ is None:
            self._occ = self.eri_fn.occupied(self.coeff('c'), self.coeff('a'), self._docc)
        return self._occ

    @property
    def cd_pa(self):
        if self._cd_pa is None:
            if hasattr(self.eri_fn, 'occupied'):
                self._cd_pa = self.eri_fn.cd_from_half(self.occupied().x_act, self.mo_coeff)
            else:
                ca = self.coeff('a')
                b = self.eri_fn.pair(self.mo_coeff, ca)
                self._cd_pa = b.reshape(b.shape[0], self.spaces.nmo, ca.shape[1])
        return self._cd_pa

    @property
    def cd_pc(self):
        if self._cd_pc is None:
            cc = self.coeff('c')
            b = self.eri_fn.pair(self.mo_coeff, cc)
            self._cd_pc = b.reshape(b.shape[0], self.spaces.nmo, cc.shape[1])
        return self._cd_pc

    @property
    def cd_vc(self):
        return self.cd_pc[:, self.spaces.slice('v'), :]

    def cd_block(self, pq):
        '''Three-index integrals B[L, p, q] of the spaces pq (two letters),
        (naux, n_p, n_q): a view of cd_pa or cd_pc where possible.'''
        p, q = pq
        if q == 'a':
            return self.cd_pa[:, self.spaces.slice(p), :]
        if q == 'c':
            return self.cd_pc[:, self.spaces.slice(p), :]
        if pq not in self.cd:
            if p in 'ac':
                self.cd[pq] = numpy.ascontiguousarray(
                    self.cd_block(q + p).transpose(0, 2, 1).conj())
            else:
                cp, cq = self.coeff(p), self.coeff(q)
                b = self.eri_fn.pair(cp, cq)
                self.cd[pq] = b.reshape(b.shape[0], cp.shape[1], cq.shape[1])
        return self.cd[pq]

    def _share_occupied(self, new, left):
        '''Hand the occupied-orbital quantities to new (orbitals phi' =
        phi left) if left keeps the active orbitals and the core space.'''
        sp = self.spaces
        a = sp.slice('a')
        if self._occ is not None and numpy.array_equal(left[a, a], numpy.eye(sp.nact)):
            new._occ = self._occ

    def _fock_factorized(self, gamma_act):
        f = self.h1 + self.eri_fn.fock(self.occupied(), gamma_act, self.mo_coeff,
                                       self.coeff('a'), self._kact)
        return numpy.asarray(f.real, dtype=self.dtype) if self.dtype == numpy.float64 else f

    def release_blocks(self):
        '''Forget the four-index blocks if they can be formed again from
        three-index integrals.'''
        if self.factorized:
            self._cache.clear()

    def coeff(self, label):
        return self.mo_coeff[:, self.spaces.slice(label)]

    def eri_block(self, labels):
        if len(labels) != 4:
            raise ValueError('eri_block needs four space labels')
        if labels in self._cache:
            return self._cache[labels]
        p, q, r, s = labels
        sup = self._from_superblock(labels)
        if sup is not None:
            out = sup
        elif r + s + p + q in self._cache:
            out = self._cache[r + s + p + q].transpose(2, 3, 0, 1)
        elif q + p + s + r in self._cache:
            out = self._cache[q + p + s + r].transpose(1, 0, 3, 2).conj()
        else:
            cs = [self.coeff(x) for x in labels]
            shape = tuple(c.shape[1] for c in cs)
            if 0 in shape:
                out = numpy.zeros(shape, dtype=self.dtype)
            elif self.factorized:
                bra, ket = self.cd_block(p + q), self.cd_block(r + s)
                naux = bra.shape[0]
                out = la.gemm(numpy.ascontiguousarray(bra).reshape(naux, -1),
                              numpy.ascontiguousarray(ket).reshape(naux, -1), 'T', 'N')
                out = numpy.asarray(out, dtype=self.dtype).reshape(shape)
            else:
                out = numpy.asarray(self.eri_fn(*cs), dtype=self.dtype).reshape(shape)
            self.computed_blocks.append(labels)
        self._cache[labels] = out
        return out

    def prefetch(self, groups):
        '''Transform one superblock (P1 P2|P3 P4) in a single pass, where each
        Pk is a string of space letters, e.g. ('av', 'ca', 'av', 'ca'), which
        contains every block UC-/PC-NEVPT2 needs.  Later requests inside it
        are sliced.'''
        if self.factorized:
            if self.spaces.nact:
                self.cd_pa
            if self.spaces.ncore:
                self.cd_pc
            return
        cs, slices = [], []
        for g in groups:
            parts = [self.coeff(x) for x in g]
            cs.append(numpy.concatenate(parts, axis=1))
            sl, off = {}, 0
            for x, c in zip(g, parts):
                sl[x] = slice(off, off + c.shape[1])
                off += c.shape[1]
            slices.append(sl)
        shape = tuple(c.shape[1] for c in cs)
        if 0 in shape:
            return
        big = numpy.asarray(self.eri_fn(*cs), dtype=self.dtype).reshape(shape)
        self.computed_blocks.append('(' + ','.join(groups) + ')')
        self._super.append((tuple(groups), slices, big))

    def _from_superblock(self, labels):
        for groups, slices, big in self._super:
            if all(x in g for x, g in zip(labels, groups)):
                return big[tuple(sl[x] for x, sl in zip(labels, slices))]
        return None

    def rotate(self, u):
        u = numpy.asarray(u, dtype=self.dtype)
        self._check_block_diagonal(u)
        new = type(self)(self.ecore, self.hcore_ao, self.mo_coeff @ u, self.spaces,
                         self.eri_fn, self.veff_fn)
        if self.factorized:
            self._share_occupied(new, u)
        return self._keep_charges(new, u)

    def with_virtuals(self, u):
        '''Hamiltonian with the virtual orbitals phi'_a = sum_b phi_b u_ba for
        u of shape (nvirt, n) with orthonormal columns; virtual orbitals
        outside their span leave the correlated space.'''
        full, spaces = self._virtual_map(u)
        new = type(self)(self.ecore, self.hcore_ao, self.mo_coeff @ full, spaces,
                         self.eri_fn, self.veff_fn)
        if self.factorized:
            self._share_occupied(new, full)
        new.active_charges = self.active_charges
        return new

    def _dm(self, gamma_act):
        '''AO density of the filled core plus the active density gamma
        (gamma_tu = <t^+ u>): dm = C_c C_c^+ + C_a gamma^T C_a^+.'''
        cc = self.coeff('c')
        ca = self.coeff('a')
        return cc @ cc.conj().T + ca @ numpy.asarray(gamma_act).T @ ca.conj().T

    def generalized_fock(self, gamma_act):
        if self.factorized and hasattr(self.eri_fn, 'occupied'):
            return self._fock_factorized(gamma_act)
        f_ao = self.hcore_ao + self.veff_fn(self._dm(gamma_act))
        return self.mo_coeff.conj().T @ f_ao @ self.mo_coeff

    def core_energy(self):
        cc = self.coeff('c')
        if cc.shape[1] == 0:
            return self.ecore
        if self.factorized and hasattr(self.eri_fn, 'occupied'):
            c = self.spaces.slice('c')
            f0 = self._fock_factorized(numpy.zeros((self.spaces.nact,) * 2))
            e = self.ecore + 0.5 * self._docc * (numpy.trace(self.h1[c, c]) + numpy.trace(f0[c, c]))
            return float(numpy.real(e))
        dm = cc @ cc.conj().T
        e = (self.ecore + numpy.einsum('ij,ji->', dm, self.hcore_ao)
             + 0.5 * numpy.einsum('ij,ji->', dm, self.veff_fn(dm)))
        return float(e.real)

    def to_dense(self):
        '''Stored-integral Hamiltonian (computes the full (pp|pp) block).'''
        return ERIs(self.ecore, self.h1, self.eri_block('pppp'), self.spaces)
