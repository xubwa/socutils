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

  IP  (electron detachment): 1h + 2h1p space
  EA  (electron attachment):  1p + 2p1h space

The secular matrix is Hermitian and is diagonalised directly for the lowest
roots (the satellite/2h1p (2p1h) block is diagonal at strict ADC(2)).  Orbital
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


if __name__ == '__main__':
    from pyscf import gto, scf, adc
    from socutils.scf import spinor_hf
    mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
    mf = spinor_hf.SpinorSCF(mol); mf.verbose = 0; mf.kernel()
    myadc = SpinorADC(mf)
    print('spinor IP', np.round(np.unique(np.round(myadc.ip_adc2(8), 6)), 5)[:3])
    print('spinor EA', np.round(np.unique(np.round(myadc.ea_adc2(8), 6)), 5)[:3])
    rmf = scf.RHF(mol).run()
    for t in ('ip', 'ea'):
        a = adc.ADC(rmf); a.method = 'adc(2)'; a.method_type = t; a.verbose = 0
        print('pyscf ', t, np.round(np.sort(np.asarray(a.kernel(nroots=4)[0])), 5))
