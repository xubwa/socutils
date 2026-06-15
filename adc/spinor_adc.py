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

IP additionally supports ADC(2)-x (``method='adc(2)-x'``), which adds the
first-order interaction in the 2h1p block (hole-hole ladder 0.5<mn||ij> plus
the particle-hole ring <ma||ei>, i.e. the EOM-IP Hbar(2h1p,2h1p) with t=0).
Because ADC(2)-x splits the doublet/quartet satellites, the spinor IP-ADC(2)-x
roots match PySCF UADC (spin-orbital), not RADC.

All three are solved by a single complex-Hermitian Davidson matvec over the
restricted (i<j / a<b) antisymmetric satellite space, so the cost is O(N^5)
per iteration rather than the O(N^9) of an explicit dense diagonalisation
(small satellite spaces fall back to a dense solve for robustness).  The EE
1p1h block is CIS plus the second-order self-energies and a non-separable
ph-ph term, the 1p1h<->2p2h coupling is first order and the 2p2h block is
diagonal (strict ADC(2)).  Orbital
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
    if dim <= max(2000, 8 * nroots):
        # small (incl. the typical IP/EA satellite spaces): build the dense
        # matrix by probing the matvec and diagonalise directly -- robust and
        # avoids Davidson's missing-root failure mode for dense, near-
        # degenerate satellite spectra.
        H = np.empty((dim, dim), dtype=complex)
        e = np.zeros(dim, dtype=complex)
        for k in range(dim):
            e[k] = 1.0
            H[:, k] = matvec(e)
            e[k] = 0.0
        w = scipy.linalg.eigh(H, eigvals_only=True,
                              subset_by_index=[0, min(dim - 1, nroots - 1)])
        return np.sort(w.real)

    # iterative block Davidson; start from a generous set of guess vectors
    # (2*nroots) so near-degenerate low roots are not skipped.
    nguess = min(dim, max(2 * nroots, nroots + 12))
    if max_space is None:
        max_space = min(dim, max(nguess + 2 * nroots + 20, 10 * nroots))
    order = np.argsort(diag.real)
    V = np.zeros((dim, nguess), dtype=complex)
    for i in range(nguess):
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


class _SpinorADCERIs:
    '''Antisymmetrised physicist MO integral blocks ``<pq||rs>`` needed by
    spinor MP2/ADC(2)/ADC(2)-x (no vvvv: those methods never need it).

    Built block-wise from the real spherical AO integrals (int2e_sph),
    recombined into the j-spinor MO basis by pyscf's C ``nrr_outcore`` driver
    -- the same path socutils' ZCCSD uses -- so the full complex ``int2e_spinor``
    AO tensor and the all-virtual block are never formed.
    '''

    def __init__(self, mol, mo_coeff, nocc):
        from pyscf.ao2mo import nrr_outcore
        o = mo_coeff[:, :nocc]
        v = mo_coeff[:, nocc:]

        def blk(P, Q, R, S):
            # <PQ||RS> = (PR|QS) - (PS|QR)  (chemist -> antisym physicist)
            nP, nQ = P.shape[1], Q.shape[1]
            nR, nS = R.shape[1], S.shape[1]
            a = np.asarray(nrr_outcore.general_iofree(
                mol, (P, R, Q, S), intor='int2e_sph', motype='j-spinor'))
            a = a.reshape(nP, nR, nQ, nS).transpose(0, 2, 1, 3)
            b = np.asarray(nrr_outcore.general_iofree(
                mol, (P, S, Q, R), intor='int2e_sph', motype='j-spinor'))
            b = b.reshape(nP, nS, nQ, nR).transpose(0, 2, 3, 1)
            return a - b

        self.oovv = blk(o, o, v, v)               # <ij||ab>
        self.oooo = blk(o, o, o, o)               # <ij||kl>
        self.ooov = blk(o, o, o, v)               # <ij||ka>
        self.voov = blk(v, o, o, v)               # <ai||jb>
        self.vovv = blk(v, o, v, v)               # <ai||bc>
        # permutations / Hermitian conjugates of the above (no new transforms)
        self.ovvo = self.voov.transpose(1, 0, 3, 2)            # <ia||bj>
        self.vvvo = self.vovv.conj().transpose(2, 3, 0, 1)     # <bc||ak>
        self.ovoo = self.ooov.conj().transpose(2, 3, 0, 1)     # <ia||jk>


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

        self._eris = None
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
        if self._eris is not None:
            return
        no = self.nocc
        eris = _SpinorADCERIs(self.mol, np.asarray(self.mo_coeff), no)
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        Voovv = eris.oovv                          # <ij||ab>
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
        self._eris = eris
        self._t2 = t2

    # -- IP ------------------------------------------------------------------
    def ip_adc2(self, nroots=6, method='adc(2)'):
        '''Spinor IP ionization energies (lowest ``nroots``).

        ``method`` selects strict ADC(2) (default) or ADC(2)-x, which adds the
        first-order interaction in the 2h1p block (hole-hole ladder 0.5<mn||ij>
        plus the particle-hole ring <ma||ei>).  Solved by a complex-Hermitian
        Davidson matvec over the restricted (i<j) 2h1p space, so the cost is
        O(N^5) per iteration rather than the O(N^9) of a dense diagonalisation.
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]
        eris = self._eris

        Mss = -np.diag(eo).astype(complex) - self._sig_ip
        Wooov = eris.ooov                          # <kl||ia>
        adc2x = (method == 'adc(2)-x')
        if not adc2x and method != 'adc(2)':
            raise NotImplementedError(method)
        if adc2x:
            Woooo = eris.oooo                      # <mn||ij>
            Wovvo = eris.ovvo                      # <ma||ei>

        oo = np.array([(k, l) for k in range(no) for l in range(k + 1, no)],
                      dtype=int).reshape(-1, 2)
        nop = len(oo)
        ok, ol = oo[:, 0], oo[:, 1]
        dim = no + nop * nv

        def unpack(rp):
            r2 = np.zeros((no, no, nv), dtype=complex)
            blk = rp.reshape(nop, nv)
            r2[ok, ol, :] = blk
            r2[ol, ok, :] = -blk
            return r2

        def pack(s2):
            return s2[ok, ol, :].reshape(-1)

        def matvec(x):
            r1 = x[:no]
            r2 = unpack(x[no:])
            s1 = Mss @ r1 + 0.5 * lib.einsum('klia,kla->i', Wooov, r2)
            s2 = (ev[None, None, :] - eo[:, None, None] - eo[None, :, None]) * r2
            s2 += lib.einsum('klia,i->kla', Wooov.conj(), r1)
            if adc2x:
                s2 += 0.5 * lib.einsum('mnij,mna->ija', Woooo, r2)
                t = lib.einsum('maei,mje->ija', Wovvo, r2)
                s2 += t - t.transpose(1, 0, 2)
            return np.concatenate([s1, pack(s2)])

        diag = np.empty(dim)
        diag[:no] = Mss.diagonal().real
        dd = ev[None, :] - (eo[ok] + eo[ol])[:, None]      # (nop, nv)
        if adc2x:
            # exact 2h1p diagonal: + <ij||ij> - <ia||ia> - <ja||ja>
            oooo_d = np.einsum('ijij->ij', eris.oooo).real     # <ij||ij>
            ovov_d = -np.einsum('iaai->ia', eris.ovvo).real    # <ia||ia>
            dd = (dd + oooo_d[ok, ol][:, None]
                  - ovov_d[ok, :] - ovov_d[ol, :])
        diag[no:] = dd.reshape(-1)
        return _davidson(matvec, diag, nroots)

    def ip_adc2x(self, nroots=6):
        '''Spinor IP-ADC(2)-x ionization energies.'''
        return self.ip_adc2(nroots, method='adc(2)-x')

    # -- EA ------------------------------------------------------------------
    def ea_adc2(self, nroots=6):
        '''Spinor EA-ADC(2) electron affinities (lowest ``nroots``).

        Davidson matvec over the restricted (a<b) 2p1h space.
        '''
        self._build()
        no, nmo = self.nocc, self.nmo
        nv = nmo - no
        e = np.asarray(self.mo_energy)
        eo, ev = e[:no], e[no:]

        Mpp = np.diag(ev).astype(complex) - self._sig_ea
        Wvovv = self._eris.vovv                    # <ai||cd>

        vv = np.array([(c, d) for c in range(nv) for d in range(c + 1, nv)],
                      dtype=int).reshape(-1, 2)
        nvp = len(vv)
        vc, vd = vv[:, 0], vv[:, 1]
        dim = nv + nvp * no

        def unpack(rp):
            r2 = np.zeros((nv, nv, no), dtype=complex)
            blk = rp.reshape(nvp, no)
            r2[vc, vd, :] = blk
            r2[vd, vc, :] = -blk
            return r2

        def pack(s2):
            return s2[vc, vd, :].reshape(-1)

        def matvec(x):
            r1 = x[:nv]
            r2 = unpack(x[nv:])
            s1 = Mpp @ r1 + 0.5 * lib.einsum('aicd,cdi->a', Wvovv, r2)
            s2 = (ev[:, None, None] + ev[None, :, None]
                  - eo[None, None, :]) * r2
            s2 += lib.einsum('aicd,a->cdi', Wvovv.conj(), r1)
            return np.concatenate([s1, pack(s2)])

        diag = np.empty(dim)
        diag[:nv] = Mpp.diagonal().real
        diag[nv:] = ((ev[vc] + ev[vd])[:, None] - eo[None, :]).reshape(-1)
        return _davidson(matvec, diag, nroots)


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
        eris = self._eris
        t2 = self._t2
        Voovv = eris.oovv

        # ph-ph second-order non-separable term
        Lam = 0.5 * (lib.einsum('imae,jmbe->iajb', t2.conj(), Voovv)
                     + lib.einsum('jmbe,imae->iajb', t2, Voovv.conj()))
        Wajib = eris.voov                          # <aj||ib>
        Wvovv = eris.vovv                          # <ak||bc>
        Wooov = eris.ooov                          # <jk||ib>
        Wvvvo = eris.vvvo                          # <bc||ak>
        Wovoo = eris.ovoo                          # <ib||jk>
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
