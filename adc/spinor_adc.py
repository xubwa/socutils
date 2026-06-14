#
# Author: Xubo Wang <wangxubo0201@outlook.com>
#
'''
Spinor ADC(2) (two-component, j-adapted Kramers-unrestricted) on top of a
:class:`socutils.scf.spinor_hf.SpinorSCF` reference.

Every spinor orbital is a full complex two-component spin-orbital, so the ADC
working equations are the bare spin-orbital expressions written with the
antisymmetrised physicist integrals ``<pq||rs> = <pq|rs> - <pq|sr>`` and the
MP1 doubles amplitudes ``t_ij^ab = <ij||ab> / (e_i + e_j - e_a - e_b)``.

Implemented strict ADC(2):

  IP  (electron detachment):  1h   + 2h1p space
  EA  (electron attachment):  1p   + 2p1h space
  EE  (neutral excitation):   1p1h + 2p2h space

IP/EA secular matrices are Hermitian and diagonalised directly for the lowest
roots (the satellite 2h1p/2p1h block is diagonal at strict ADC(2)).  EE uses a
Davidson matvec (the 2p2h block is too large to store densely); its 1p1h block
is CIS plus the second-order self-energies and a non-separable ph-ph term, the
1p1h<->2p2h coupling is first order and the 2p2h block is diagonal.  Orbital
energies are taken from the mean field (canonical); under a legal canonical
rotation (per-orbital phase or a unitary mix within an exactly degenerate /
Kramers block) the orbital energies are unchanged and the eigenvalues are
invariant while the eigenvectors become complex.

Two-electron integrals are the spinor Coulomb integrals ``int2e_spinor``,
matching the picture-change convention used across socutils.  For a
non-relativistic :class:`SpinorSCF` reference the IP/EA roots reproduce the
PySCF RADC IP/EA values (each spatial root appearing with its Kramers
multiplicity).
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger


def _davidson(matvec, diag, nroots, max_space=None, tol=1e-9, max_cycle=200):
    '''Lowest ``nroots`` eigenpairs of a complex Hermitian operator given as a
    matvec (acting on a flat complex vector) and its real diagonal.'''
    dim = diag.size
    if dim <= max(400, 6 * nroots):
        # small: just build the dense matrix by probing the matvec
        H = np.empty((dim, dim), dtype=complex)
        e = np.zeros(dim, dtype=complex)
        for k in range(dim):
            e[k] = 1.0
            H[:, k] = matvec(e)
            e[k] = 0.0
        w = scipy.linalg.eigh(H, eigvals_only=True,
                              subset_by_index=[0, min(dim - 1, nroots - 1)])
        return np.sort(w.real)

    if max_space is None:
        max_space = min(dim, max(2 * nroots + 20, 8 * nroots))
    order = np.argsort(diag.real)
    V = np.zeros((dim, nroots), dtype=complex)
    for i in range(nroots):
        V[order[i], i] = 1.0
    V, _ = np.linalg.qr(V)
    AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
    theta_old = np.zeros(nroots)
    for _ in range(max_cycle):
        S = V.conj().T @ AV
        S = (S + S.conj().T) * 0.5
        w, y = np.linalg.eigh(S)
        w = w[:nroots]
        y = y[:, :nroots]
        X = V @ y
        AX = AV @ y
        res = AX - X * w[None, :]
        rnorm = np.linalg.norm(res, axis=0)
        if np.max(np.abs(w - theta_old)) < tol and np.max(rnorm) < np.sqrt(tol):
            return np.sort(w)
        theta_old = w
        # preconditioned new directions
        add = []
        for i in range(nroots):
            if rnorm[i] < np.sqrt(tol):
                continue
            d = diag.real - w[i]
            d[np.abs(d) < 1e-8] = 1e-8
            t = res[:, i] / d
            add.append(t)
        if not add:
            return np.sort(w)
        Vnew = np.column_stack(add)
        # orthonormalise against current space
        Vnew -= V @ (V.conj().T @ Vnew)
        Vnew -= V @ (V.conj().T @ Vnew)
        q, r = np.linalg.qr(Vnew)
        keep = np.abs(np.diag(r)) > 1e-7
        if not np.any(keep):
            return np.sort(w)
        q = q[:, keep]
        if V.shape[1] + q.shape[1] > max_space:
            V = X
            AV = AX
        Aq = np.column_stack([matvec(q[:, i]) for i in range(q.shape[1])])
        V = np.column_stack([V, q])
        AV = np.column_stack([AV, Aq])
    return np.sort(w)


def _antisym_mo_eri(mol, mo_coeff):
    '''Full antisymmetrised physicist integrals ``W[p,q,r,s] = <pq||rs>`` in
    the spinor MO basis (all spin-orbitals).'''
    eri_ao = mol.intor('int2e_spinor')           # chemist (pq|rs)
    c = mo_coeff
    g = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, c.conj(), c, c.conj(), c)
    w = g.transpose(0, 2, 1, 3)                   # <ij|kl> = (ik|jl)
    w = w - w.transpose(0, 1, 3, 2)               # antisymmetrise
    return w


class SpinorADC(lib.StreamObject):
    '''Spinor ADC(2).

    Args:
        mf : a converged :class:`SpinorSCF` (or subclass) mean field.

    The MO coefficients/occupations are read from ``mf`` (or supplied), so
    injecting rotated orbitals (mf.copy() + overwrite mo_coeff) works without
    re-diagonalisation: the same canonical mo_energy is reused.
    '''

    def __init__(self, mf, mo_coeff=None, mo_occ=None):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        self.mo_occ = mf.mo_occ if mo_occ is None else mo_occ
        self.mo_energy = mf.mo_energy

        self._W = None
        self._t2 = None
        self._sig_ip = None
        self._sig_ea = None

    @property
    def nocc(self):
        return int(np.count_nonzero(self.mo_occ > 0))

    @property
    def nmo(self):
        return self.mo_coeff.shape[1]

    # -- shared intermediates ------------------------------------------------
    def _build(self):
        if self._W is not None:
            return
        no = self.nocc
        W = _antisym_mo_eri(self.mol, self.mo_coeff)
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        o = slice(0, no)
        v = slice(no, self.nmo)
        Voovv = W[o, o, v, v]                      # <ij||ab>
        d = (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])
        t2 = Voovv / d
        # Static second-order self-energies (same as the IP/EA quasiparticle
        # corrections).  The t2.conj() placement makes the blocks transform
        # correctly under a complex orbital-phase gauge (holes ~ ph_i*).
        self._sig_ip = 0.25 * (lib.einsum('ikab,jkab->ij', t2.conj(), Voovv)
                               + lib.einsum('jkab,ikab->ij', t2, Voovv.conj()))
        self._sig_ea = 0.25 * (lib.einsum('ijac,ijbc->ab', t2.conj(), Voovv)
                               + lib.einsum('ijbc,ijac->ab', t2, Voovv.conj()))
        self._W = W
        self._t2 = t2

    def _solve(self, M, nroots):
        '''Lowest ``nroots`` eigenvalues of a Hermitian matrix M.'''
        n = M.shape[0]
        if nroots >= n:
            return np.sort(scipy.linalg.eigh(M, eigvals_only=True))
        hi = min(n - 1, 4 * nroots + 8)
        w = scipy.linalg.eigh(M, eigvals_only=True, subset_by_index=[0, hi])
        return np.sort(w)

    # -- IP ------------------------------------------------------------------
    def ip_adc2(self, nroots=6):
        '''Spinor IP-ADC(2) ionization energies (lowest ``nroots``).'''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        W = self._W
        o = slice(0, no)
        v = slice(no, nmo)

        # 1h block: -e_i delta_ij - static self-energy
        Mss = -np.diag(eo).astype(complex) - self._sig_ip

        # 2h1p configurations (k<l occupied pair, a virtual)
        pairs = [(k, l) for k in range(no) for l in range(k + 1, no)]
        npair = len(pairs)
        ndd = npair * nv
        Vooov = W[o, o, o, v]                      # <kl||ia>
        Csd = np.zeros((no, ndd), dtype=complex)
        Ddd = np.empty(ndd)
        for p, (k, l) in enumerate(pairs):
            Csd[:, p * nv:(p + 1) * nv] = Vooov[k, l, :, :]
            Ddd[p * nv:(p + 1) * nv] = ev - eo[k] - eo[l]

        dim = no + ndd
        M = np.zeros((dim, dim), dtype=complex)
        M[:no, :no] = Mss
        M[:no, no:] = Csd
        M[no:, :no] = Csd.conj().T
        M[no:, no:] = np.diag(Ddd).astype(complex)
        return self._solve(M, nroots)

    # -- EA ------------------------------------------------------------------
    def ea_adc2(self, nroots=6):
        '''Spinor EA-ADC(2) electron affinities (lowest ``nroots``).'''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        W = self._W
        o = slice(0, no)
        v = slice(no, nmo)

        # 1p block: +e_a delta_ab - static self-energy
        Mpp = np.diag(ev).astype(complex) - self._sig_ea

        # 2p1h configurations (c<d virtual pair, i hole). Coupling <ai||cd>
        # keeps the external particle 'a' in the bra (correct phase gauge).
        Vovvv = W[o, v, v, v]                      # <ia||cd>; <ai||cd> = -<ia||cd>
        pairs = [(c, d) for c in range(nv) for d in range(c + 1, nv)]
        npair = len(pairs)
        ndd = npair * no
        Cps = np.zeros((nv, ndd), dtype=complex)
        Ddd = np.empty(ndd)
        for p, (c, d) in enumerate(pairs):
            # <ai||cd> = -<ia||cd> = -Vovvv[i,a,c,d]
            Cps[:, p * no:(p + 1) * no] = -Vovvv[:, :, c, d].T
            Ddd[p * no:(p + 1) * no] = ev[c] + ev[d] - eo

        dim = nv + ndd
        M = np.zeros((dim, dim), dtype=complex)
        M[:nv, :nv] = Mpp
        M[:nv, nv:] = Cps
        M[nv:, :nv] = Cps.conj().T
        M[nv:, nv:] = np.diag(Ddd).astype(complex)
        return self._solve(M, nroots)


    # -- EE ------------------------------------------------------------------
    def ee_adc2(self, nroots=6):
        '''Spinor EE-ADC(2) excitation energies (lowest ``nroots``).

        The 1p1h block is CIS + the second-order self-energies and the
        non-separable ph-ph term; the 1p1h<->2p2h coupling is first order and
        the 2p2h block is diagonal (strict ADC(2)).  Solved by a Davidson
        matvec (the 2p2h space is too large to store densely).
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        W = self._W
        o = slice(0, no)
        v = slice(no, nmo)
        t2 = self._t2
        Voovv = W[o, o, v, v]

        # ph-ph second-order non-separable term
        Lam = 0.5 * (lib.einsum('imae,jmbe->iajb', t2.conj(), Voovv)
                     + lib.einsum('jmbe,imae->iajb', t2, Voovv.conj()))
        Wajib = W[v, o, o, v]                      # <aj||ib>
        Wvovv = W[v, o, v, v]                      # <ak||bc>
        Wooov = W[o, o, o, v]                      # <jk||ib>
        Wvvvo = W[v, v, v, o]                      # <bc||ak>
        Wovoo = W[o, v, o, o]                      # <ib||jk>
        sig_ip, sig_ea = self._sig_ip, self._sig_ea
        Dijab = (-eo[:, None, None, None] - eo[None, :, None, None]
                 + ev[None, None, :, None] + ev[None, None, None, :])

        # restricted antisymmetric pair indices for the 2p2h vector
        oo = np.array([(j, k) for j in range(no) for k in range(j + 1, no)],
                      dtype=int).reshape(-1, 2)
        vv = np.array([(b, c) for b in range(nv) for c in range(b + 1, nv)],
                      dtype=int).reshape(-1, 2)
        nop, nvp = len(oo), len(vv)
        ns = no * nv
        ndp = nop * nvp
        dim = ns + ndp

        oj, ok = oo[:, 0], oo[:, 1]
        vb, vc = vv[:, 0], vv[:, 1]

        def unpack(rp):
            r2 = np.zeros((no, no, nv, nv), dtype=complex)
            blk = rp.reshape(nop, nvp)
            r2[oj[:, None], ok[:, None], vb[None, :], vc[None, :]] = blk
            r2[ok[:, None], oj[:, None], vb[None, :], vc[None, :]] = -blk
            r2[oj[:, None], ok[:, None], vc[None, :], vb[None, :]] = -blk
            r2[ok[:, None], oj[:, None], vc[None, :], vb[None, :]] = blk
            return r2

        def pack(s2):
            return s2[oj[:, None], ok[:, None], vb[None, :], vc[None, :]].reshape(-1)

        def matvec(x):
            r1 = x[:ns].reshape(no, nv)
            r2 = unpack(x[ns:])
            s1 = (ev[None, :] - eo[:, None]) * r1
            s1 += lib.einsum('ajib,jb->ia', Wajib, r1)
            s1 -= lib.einsum('ij,ja->ia', sig_ip, r1)
            s1 -= lib.einsum('ab,ib->ia', sig_ea, r1)
            s1 += lib.einsum('iajb,jb->ia', Lam, r1)
            s1 += 0.5 * lib.einsum('akbc,ikbc->ia', Wvovv, r2)
            s1 -= 0.5 * lib.einsum('jkib,jkab->ia', Wooov, r2)
            s2 = Dijab * r2
            tA = lib.einsum('bcak,ja->jkbc', Wvvvo, r1)
            s2 += tA - tA.transpose(1, 0, 2, 3)
            tB = lib.einsum('ibjk,ic->jkbc', Wovoo, r1)
            s2 += tB - tB.transpose(0, 1, 3, 2)
            return np.concatenate([s1.ravel(), pack(s2)])

        diag = np.empty(dim)
        diag[:ns] = (ev[None, :] - eo[:, None]).ravel().real
        dd = Dijab[oj[:, None], ok[:, None], vb[None, :], vc[None, :]].real
        diag[ns:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)


if __name__ == '__main__':
    from pyscf import gto, scf, adc
    from socutils.scf import spinor_hf
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = spinor_hf.SpinorSCF(mol); mf.verbose = 0; mf.kernel()
    myadc = SpinorADC(mf)
    print('spinor IP', np.round(np.unique(np.round(myadc.ip_adc2(8), 6)), 5)[:3])
    print('spinor EA', np.round(np.unique(np.round(myadc.ea_adc2(8), 6)), 5)[:3])
    print('spinor EE', np.round(np.unique(np.round(myadc.ee_adc2(8), 6)), 5)[:3])
    rmf = scf.RHF(mol).run()
    for t in ('ip', 'ea'):
        a = adc.ADC(rmf); a.method = 'adc(2)'; a.method_type = t; a.verbose = 0
        print('pyscf ', t, np.round(np.sort(np.asarray(a.kernel(nroots=4)[0])), 5))
    umf = scf.UHF(mol).run()
    a = adc.ADC(umf); a.method = 'adc(2)'; a.method_type = 'ee'; a.verbose = 0
    print('pyscf  ee', np.round(np.unique(np.round(a.kernel(nroots=8)[0], 6)), 5)[:3])
