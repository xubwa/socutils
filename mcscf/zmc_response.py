'''Gradient and exact Hessian-vector product of the state-averaged spinor CASSCF energy
in orbitals and CI vectors together (the Jacobian of the Z-vector equations of relaxed
properties).

Parameters (complex):

    orbitals   C' = C exp(K),  K = unpack(x_orb)            (zmcscf.CASSCF.unpack_uniq_var)
    CI vectors c_I' = (c_I + x_I) / |c_I + x_I|

E = sum_I w_I <c_I'|H(C')|c_I'>.  With the real variables X = (Re z, Im z) of the
complex vector z = (x_orb, x_1, x_2, ...), the gradient and the Hessian act as

    dE = Re<G, dz>,   d^2E = Re<dz, J dz>,

G and J dz complex vectors laid out as z ('realified' they are the gradient and the
Hessian product in X).  The blocks (zmc_twostep conventions: E(eps) = E0 + 2 eps Re<g, x>
+ eps^2 <x, H x> for orbital rotations at fixed density matrices):

    G_orb  = 2 g(D_SA),                     G_I = 2 w_I r_I,   r_I = (H - E_I) c_I
    J_oo x = 2 H x                          (zmc_twostep.gen_g_hop_exact, D_SA)
    J_oI y = 2 g_lin(w_I dD_I(y))           dD_I(y) = S_I(y) - 2 Re<c_I, y> D_I,
                                            S_I(y) = [D(c_I + y) - D(c_I - y)] / 2
    J_Io x = 2 w_I (H1(x) c_I - <c_I|H1(x)|c_I> c_I)
    J_II y = 2 w_I ((H - E_I) y - 2 Re<r_I, y> c_I - 2 Re<c_I, y> r_I)

g_lin the orbital gradient of density matrices without the core part, H1(x) the
first-order change of the active Hamiltonian (h_eff and (tu|vw)) under the rotation.
At a stationary point the zero modes are the norm (c_I) and the phase (i c_I) of every
CI vector (elsewhere J c_I = -G_I, J i c_I = i G_I: E is invariant, its gradient is not);
the rotations among the averaged states are not zero modes of J.
'''
import numpy
from pyscf import lib
from pyscf.fci import fci_dhf_slow
from socutils.mcscf import zmc_ao2mo, zmc_twostep


def realify(z):
    z = numpy.asarray(z)
    return numpy.concatenate([z.real, z.imag])


def complexify(v):
    v = numpy.asarray(v)
    n = len(v) // 2
    return v[:n] + 1j * v[n:]


class SAResponse:
    '''The state-averaged spinor CASSCF at orbitals mo and CI vectors ci (list of complex
    vectors, normalized) with weights.  Attributes: n_orb (complex orbital parameters),
    sizes (CI vectors), gradient() (complex G), hop(z) (complex J z), split / join.'''

    def __init__(self, mc, mo, ci, weights, eris=None, cderi=None):
        self.mc = mc
        mc.mo_coeff = mo
        self.mo = mo
        self.ci = [numpy.asarray(c, dtype=complex).ravel() for c in ci]
        self.weights = numpy.asarray(weights, dtype=float)
        self.eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2) if eris is None else eris
        ncore, ncas = mc.ncore, mc.ncas
        self.ncore, self.ncas, self.nelec = ncore, ncas, mc.nelecas
        self.act = slice(ncore, ncore + ncas)
        eris = self.eris
        h = lib.dot(mo.conj().T, lib.dot(mc.get_hcore(), mo))
        vc = lib.dot(mo.conj().T, lib.dot(eris.vj_core - eris.vk_core, mo))
        self.hvc = h + vc
        self.h1eff = self.hvc[self.act, self.act]
        self.eri = numpy.asarray(eris.aaaa).reshape((ncas,) * 4)
        self.h2e = fci_dhf_slow.absorb_h1e(self.h1eff, self.eri, ncas, self.nelec, .5)
        self.hc = [self._h(self.h2e, c) for c in self.ci]
        self.e = numpy.array([numpy.vdot(c, hc).real for c, hc in zip(self.ci, self.hc)])
        self.r = [hc - e * c for c, hc, e in zip(self.ci, self.hc, self.e)]
        self.dms = [self._rdm12(c) for c in self.ci]
        dm1 = sum(w * d[0] for w, d in zip(self.weights, self.dms))
        dm2 = sum(w * d[1] for w, d in zip(self.weights, self.dms))
        self.dm1, self.dm2 = dm1, dm2
        self.e_orb, self.g_orb, self._hop_orb = zmc_twostep.gen_g_hop_exact(mc, mo, dm1, dm2, eris)
        self.n_orb = len(self.g_orb)
        self.sizes = [len(c) for c in self.ci]
        cd_pa = eris.cd_pa
        self.naux = cd_pa.shape[0]
        cd_up = getattr(eris, 'cd_up', None)
        if cd_up is None:
            cd_up = numpy.ascontiguousarray(cd_pa.transpose(0, 2, 1))
        self.cd_up2 = cd_up.reshape(self.naux * ncas, -1)
        self.cd_aa = numpy.asarray(eris.cd_aa).reshape(self.naux, ncas, ncas)

    # -- helpers -------------------------------------------------------------------
    def _h(self, h2e, c):
        return fci_dhf_slow.contract_2e(h2e, c, self.ncas, self.nelec)

    def _rdm12(self, c):
        return fci_dhf_slow.make_rdm12(c, self.ncas, self.nelec)

    def split(self, z):
        out = [z[:self.n_orb]]
        p = self.n_orb
        for n in self.sizes:
            out.append(z[p:p + n])
            p += n
        return out

    def join(self, parts):
        return numpy.concatenate([numpy.asarray(x, dtype=complex) for x in parts])

    def g_lin(self, dm1, dm2):
        '''The orbital gradient (packed, zmc_twostep convention g) of density matrices
        (dm1, dm2) without its core part: the change of g(D) under D -> D + (dm1, dm2).'''
        mc, ncore, ncas, act = self.mc, self.ncore, self.ncas, self.act
        nmo = self.mo.shape[1]
        dm = numpy.asarray(dm1).T
        vja, vka = self.eris.get_jk_active_mo(dm)
        a = numpy.zeros((nmo, nmo), dtype=complex)
        a[:, :ncore] = (vja - vka)[:, :ncore]
        a[:, act] = self.hvc[:, act] @ dm
        t = lib.dot(self.cd_aa.reshape(self.naux, ncas * ncas),
                    numpy.asarray(dm2).reshape(ncas * ncas, ncas * ncas).T).reshape(self.naux, ncas, ncas)
        a[:, act] += lib.dot(self.cd_up2.T,
                             numpy.ascontiguousarray(t.transpose(0, 2, 1)).reshape(self.naux * ncas, ncas))
        return mc.pack_uniq_var(a - a.conj().T)

    def gradient(self):
        return self.join([2 * self.g_orb] + [2 * w * r for w, r in zip(self.weights, self.r)])

    def hop(self, z):
        '''J z (complex layout of z).'''
        x, *ys = self.split(numpy.asarray(z, dtype=complex))
        ncas, nelec = self.ncas, self.nelec
        hx, parts = self._hop_orb(x, parts=True)
        out_o = 2 * hx
        # orbital part from the CI directions: the change of the averaged densities
        ddm1 = numpy.zeros_like(self.dm1)
        ddm2 = numpy.zeros_like(self.dm2)
        for w, c, y, (d1, d2) in zip(self.weights, self.ci, ys, self.dms):
            if not numpy.any(y):
                continue
            p1, p2 = self._rdm12(c + y)
            m1, m2 = self._rdm12(c - y)
            s = 2 * numpy.vdot(c, y).real
            ddm1 += w * (0.5 * (p1 - m1) - s * d1)
            ddm2 += w * (0.5 * (p2 - m2) - s * d2)
        if numpy.any(ddm1) or numpy.any(ddm2):
            out_o = out_o + 2 * self.g_lin(ddm1, ddm2)
        # first-order active Hamiltonian of the rotation
        act = self.act
        h1eff1 = (parts['h1'] + parts['vc1'])[act, act]
        l1 = parts['l1_aa'].reshape(self.naux, ncas, ncas)
        eri1 = (lib.einsum('ptu,pvw->tuvw', l1, self.cd_aa)
                + lib.einsum('ptu,pvw->tuvw', self.cd_aa, l1))
        h2e1 = fci_dhf_slow.absorb_h1e(h1eff1, eri1, ncas, nelec, .5)
        out_c = []
        for w, c, y, r, e in zip(self.weights, self.ci, ys, self.r, self.e):
            h1c = self._h(h2e1, c)
            v = h1c - numpy.vdot(c, h1c) * c
            v = v + (self._h(self.h2e, y) - e * y - 2 * numpy.vdot(r, y).real * c
                     - 2 * numpy.vdot(c, y).real * r)
            out_c.append(2 * w * v)
        return self.join([out_o] + out_c)

    def null_space(self):
        '''Columns (complex layout) of the zero modes at a stationary point: norm and phase
        of every CI vector.'''
        n = self.n_orb + sum(self.sizes)
        cols = []
        p = self.n_orb
        for c in self.ci:
            for f in (1.0, 1j):
                v = numpy.zeros(n, dtype=complex)
                v[p:p + len(c)] = f * c
                cols.append(v)
            p += len(c)
        return numpy.array(cols).T


def sa_energy(mc, mo, ci, weights, cderi=None):
    '''sum_I w_I <c_I|H(mo)|c_I> / <c_I|c_I> (the CI vectors not solved again).'''
    eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
    ncore, ncas = mc.ncore, mc.ncas
    act = slice(ncore, ncore + ncas)
    h = lib.dot(mo.conj().T, lib.dot(mc.get_hcore(), mo))
    vc = lib.dot(mo.conj().T, lib.dot(eris.vj_core - eris.vk_core, mo))
    ecore = (mc.energy_nuc() + numpy.trace(h[:ncore, :ncore])
             + 0.5 * numpy.trace(vc[:ncore, :ncore])).real
    eri = numpy.asarray(eris.aaaa).reshape((ncas,) * 4)
    h2e = fci_dhf_slow.absorb_h1e((h + vc)[act, act], eri, ncas, mc.nelecas, .5)
    out = 0.0
    for w, c in zip(weights, ci):
        c = numpy.asarray(c, dtype=complex).ravel()
        hc = fci_dhf_slow.contract_2e(h2e, c, ncas, mc.nelecas)
        out += w * (numpy.vdot(c, hc).real / numpy.vdot(c, c).real + ecore)
    return out


def move(mc, mo, ci, z):
    '''(mo', [c_I']) at the parameters z (complex layout of SAResponse).'''
    import scipy.linalg
    n_orb = numpy.count_nonzero(mc.uniq_var_indices(mo.shape[1], mc.ncore, mc.ncas, mc.frozen))
    mo1 = mo @ scipy.linalg.expm(mc.unpack_uniq_var(z[:n_orb]))
    out, p = [], n_orb
    for c in ci:
        c = numpy.asarray(c, dtype=complex).ravel()
        d = c + z[p:p + len(c)]
        out.append(d / numpy.linalg.norm(d))
        p += len(c)
    return mo1, out
