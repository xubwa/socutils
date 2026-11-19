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

The CI-space work (H_act and its first-order change applied to vectors, transition density
matrices) runs on the native kernels of zmr (zmr.active.sectors.ActiveSpace.sigma,
zmr.active.rdm.make_rdm), in the determinant order and density layout of socutils.fci.zfci.
'''
import numpy
from pyscf import lib
from socutils.mcscf import zmc_ao2mo, zmc_twostep


def _active_space(h1, eri):
    from socutils.zmr.active.sectors import ActiveSpace
    n = h1.shape[0]
    return ActiveSpace(h1, numpy.asarray(eri).reshape((n,) * 4), 0.0)


def _sigma(act, nelec, vecs):
    '''H_act vecs (a vector or columns) by the native sigma kernel.'''
    v = numpy.asarray(vecs, dtype=complex)
    if v.ndim == 1:
        return act.sigma(nelec, v[:, None])[:, 0]
    return act.sigma(nelec, numpy.ascontiguousarray(v))


def _rdm(act, nelec, vecs, k, bra=None, ket=None):
    from socutils.zmr.active import rdm
    return rdm.make_rdm(act, numpy.asarray(vecs, dtype=complex), nelec, k, bra=bra, ket=ket)


def _sym_rdm12(act, nelec, a, b):
    '''B(a, b) + B(b, a) of the 1- and 2-RDMs (B(x, y) = <x|...|y>).'''
    v = numpy.array([a, b]).T
    d1 = _rdm(act, nelec, v, 1)
    d2 = _rdm(act, nelec, v, 2)
    return d1[0, 1] + d1[1, 0], d2[0, 1] + d2[1, 0]


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
        self.cas = _active_space(self.h1eff, self.eri)
        cmat = numpy.array(self.ci).T
        self.hc = list(_sigma(self.cas, self.nelec, cmat).T)
        self.e = numpy.array([numpy.vdot(c, hc).real for c, hc in zip(self.ci, self.hc)])
        self.r = [hc - e * c for c, hc, e in zip(self.ci, self.hc, self.e)]
        # all (transition) density matrices among the states at once
        self.tdm1 = _rdm(self.cas, self.nelec, cmat, 1)
        self.tdm2 = _rdm(self.cas, self.nelec, cmat, 2)
        self.dms = [(self.tdm1[i, i], self.tdm2[i, i]) for i in range(len(self.ci))]
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
    def apply(self, h1, eri, vecs):
        '''(sum h1 t^+ u + 1/2 sum eri t^+ v^+ w u) vecs in the active space.'''
        return _sigma(_active_space(h1, eri), self.nelec, vecs)

    def apply_h1(self, h1, vecs):
        return self.apply(h1, numpy.zeros((self.ncas,) * 4, dtype=complex), vecs)

    def _rdm12(self, c):
        c = numpy.asarray(c, dtype=complex)
        return _rdm(self.cas, self.nelec, c, 1)[0, 0], _rdm(self.cas, self.nelec, c, 2)[0, 0]

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
            s1, s2 = _sym_rdm12(self.cas, self.nelec, c, y)
            s = 2 * numpy.vdot(c, y).real
            ddm1 += w * (s1 - s * d1)
            ddm2 += w * (s2 - s * d2)
        if numpy.any(ddm1) or numpy.any(ddm2):
            out_o = out_o + 2 * self.g_lin(ddm1, ddm2)
        # first-order active Hamiltonian of the rotation
        act = self.act
        h1eff1 = (parts['h1'] + parts['vc1'])[act, act]
        l1 = parts['l1_aa'].reshape(self.naux, ncas, ncas)
        eri1 = (lib.einsum('ptu,pvw->tuvw', l1, self.cd_aa)
                + lib.einsum('ptu,pvw->tuvw', self.cd_aa, l1))
        h1c_all = self.apply(h1eff1, eri1, numpy.array(self.ci).T).T
        hy_all = _sigma(self.cas, nelec, numpy.array(ys).T).T
        out_c = []
        for w, c, y, r, e, h1c, hy in zip(self.weights, self.ci, ys, self.r, self.e,
                                           h1c_all, hy_all):
            v = h1c - numpy.vdot(c, h1c) * c
            v = v + (hy - e * y - 2 * numpy.vdot(r, y).real * c
                     - 2 * numpy.vdot(c, y).real * r)
            out_c.append(2 * w * v)
        return self.join([out_o] + out_c)

    def ci_diagonal(self):
        '''<D|H_act|D> over the determinants D of the CI space (string order of zfci).'''
        from socutils.zmr.active import strings
        strs = numpy.asarray(strings.gen_strings(self.ncas, self.nelec), dtype=numpy.uint64)
        occ = ((strs[:, None] >> numpy.arange(self.ncas, dtype=numpy.uint64)[None, :]) & 1).astype(float)
        eri = self.eri
        jk = (numpy.einsum('iijj->ij', eri) - numpy.einsum('ijji->ij', eri)).real
        return occ @ self.h1eff.diagonal().real + 0.5 * numpy.einsum('di,ij,dj->d', occ, jk, occ)

    def precond_diagonal(self, floor=1e-3):
        '''A positive diagonal approximation of J (complex layout, real values): twice the
        super-CI orbital diagonal (zmc_superci) and 2 w_I |<D|H|D> - E_I| for the CI vectors,
        floored.'''
        from socutils.mcscf import zmc_superci
        h_orb = numpy.asarray(zmc_superci.gen_g_hop(self.mc, numpy.array(self.mo, copy=True),
                                                     self.dm1, self.dm2, self.eris)[1]).real
        hdd = self.ci_diagonal()
        parts = [2 * abs(h_orb)] + [2 * w * abs(hdd - e) for w, e in zip(self.weights, self.e)]
        return numpy.maximum(numpy.concatenate(parts), floor)

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
    cas = _active_space((h + vc)[act, act], eris.aaaa)
    out = 0.0
    for w, c in zip(weights, ci):
        c = numpy.asarray(c, dtype=complex).ravel()
        hc = _sigma(cas, mc.nelecas, c)
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


# ----------------------------------------------------------------------
# relaxed densities: target gradients, Z-vector, Lagrangian density

def _orbital_gradient(resp, dm1, dm2):
    return zmc_twostep.gen_g_hop_exact(resp.mc, resp.mo, dm1, dm2, resp.eris)[1]


def state_gradient(resp, k):
    '''G of the energy of state k alone (complex layout).'''
    d1, d2 = resp.dms[k]
    parts = [2 * _orbital_gradient(resp, d1, d2)]
    for i, (r, n) in enumerate(zip(resp.r, resp.sizes)):
        parts.append(2 * r if i == k else numpy.zeros(n, dtype=complex))
    return resp.join(parts)


def _projector(cols):
    '''Orthonormal real columns spanning the realified complex columns (by SVD: dependent
    columns, e.g. a direction given twice, count once).'''
    if cols.shape[1] == 0:
        return numpy.zeros((2 * cols.shape[0], 0))
    u, sv, _ = numpy.linalg.svd(numpy.array([realify(c) for c in cols.T]).T, full_matrices=False)
    return u[:, sv > 1e-10 * sv.max()]


def zvector(resp, rhs, tol=1e-10, maxiter=1000, extra_null=None, precond=True, info=None):
    '''z with J z = -rhs on the complement of the zero modes (norm and phase of the CI
    vectors, rotations among degenerate averaged states; extra_null: more complex columns
    to project out), by MINRES in the real
    variables, preconditioned with resp.precond_diagonal (precond=False: none).  info: a
    dict that receives the number of iterations.  Returns z in the complex layout.'''
    import scipy.sparse.linalg
    # the norm and phase of every CI vector, and the rotations and mixings among degenerate
    # averaged states of equal weight: zero modes of J that would change the averaged
    # density (the targets do not depend on them, degenerate_rotations)
    cols = numpy.hstack([resp.null_space(), degenerate_rotations(resp)])
    if extra_null is not None:
        cols = numpy.hstack([cols, extra_null])
    q = _projector(cols)

    def proj(v):
        return v - q @ (q.T @ v)
    n = 2 * len(rhs)
    op = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=lambda v: proj(realify(resp.hop(complexify(proj(v))))))
    m = None
    if precond:
        d = resp.precond_diagonal()
        dinv = 1.0 / numpy.concatenate([d, d])
        m = scipy.sparse.linalg.LinearOperator((n, n), matvec=lambda v: proj(dinv * proj(v)))
    b = proj(-realify(rhs))
    count = [0]

    def callback(xk):
        count[0] += 1
    z, status = scipy.sparse.linalg.minres(op, b, M=m, rtol=tol, maxiter=maxiter,
                                           callback=callback)
    if info is not None:
        info['iterations'] = count[0]
    if status != 0:
        raise RuntimeError('Z-vector equations not converged (minres info %d)' % status)
    return complexify(proj(z))


def mo_density(resp, dm1):
    '''MO density (core 1, active gamma^T) of an active 1-RDM gamma.'''
    nmo = resp.mo.shape[1]
    d = numpy.zeros((nmo, nmo), dtype=complex)
    d[:resp.ncore, :resp.ncore] = numpy.eye(resp.ncore)
    d[resp.act, resp.act] = numpy.asarray(dm1).T
    return d


def ao_density(resp, d_mo):
    '''C D C^+: <O> = tr(O_AO D_AO) for a spinor AO operator O.'''
    return resp.mo @ d_mo @ resp.mo.conj().T


def state_density(resp, k):
    '''AO density of state k.'''
    return ao_density(resp, mo_density(resp, resp.dms[k][0]))


def lagrangian_density(resp, z):
    '''D_z (AO): the derivative of the averaged density along z, so that
    Re<z, dG_SA/dlambda> = tr(O D_z) for h -> h + lambda O.'''
    x, *ys = resp.split(numpy.asarray(z, dtype=complex))
    k = resp.mc.unpack_uniq_var(x)
    d = mo_density(resp, resp.dm1)
    dd = k @ d - d @ k
    for w, c, y, (d1, _) in zip(resp.weights, resp.ci, ys, resp.dms):
        v = numpy.array([c, y]).T
        t1 = _rdm(resp.cas, resp.nelec, v, 1)
        g = t1[0, 1] + t1[1, 0] - 2 * numpy.vdot(c, y).real * d1
        dd[resp.act, resp.act] += w * numpy.asarray(g).T
    return ao_density(resp, dd)


def interstate_directions(resp):
    '''Rotations c_I -> c_I + t c_J, c_J -> c_J - t conj(t) c_I and the symmetric mixings
    among the states (real and imaginary t): complex columns.'''
    n = resp.n_orb + sum(resp.sizes)
    offs = numpy.concatenate([[resp.n_orb], resp.n_orb + numpy.cumsum(resp.sizes)])
    cols = []
    for i in range(len(resp.ci)):
        for j in range(i):
            for f in (1.0, 1j):
                for s in (1.0, -1.0):
                    v = numpy.zeros(n, dtype=complex)
                    v[offs[i]:offs[i + 1]] = f * resp.ci[j]
                    v[offs[j]:offs[j + 1]] = s * numpy.conj(f) * resp.ci[i]
                    cols.append(v)
    return numpy.array(cols).T if cols else numpy.zeros((n, 0), dtype=complex)


def model_eigenvectors(mc, mo, ci, cderi=None):
    '''The CI vectors replaced by the eigenvectors of H_act in their span (Loewdin
    orthonormalized; each with the phase of the old vector it is closest to).'''
    if len(ci) < 2:
        return ci
    eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
    ncore, ncas = mc.ncore, mc.ncas
    act = slice(ncore, ncore + ncas)
    h = lib.dot(mo.conj().T, lib.dot(mc.get_hcore(), mo))
    vc = lib.dot(mo.conj().T, lib.dot(eris.vj_core - eris.vk_core, mo))
    cas = _active_space((h + vc)[act, act], eris.aaaa)
    c = numpy.array([numpy.asarray(x, dtype=complex).ravel() for x in ci])
    w, v = numpy.linalg.eigh(c.conj() @ c.T)
    c = (v / numpy.sqrt(w)) @ v.conj().T @ c
    hc = _sigma(cas, mc.nelecas, c.T).T
    hm = c.conj() @ hc.T
    _, u = numpy.linalg.eigh(0.5 * (hm + hm.conj().T))
    new = u.T @ c
    out = []
    for x in new:
        ov = [numpy.vdot(numpy.asarray(y).ravel(), x) for y in ci]
        kk = int(numpy.argmax(numpy.abs(ov)))
        out.append(x * numpy.conj(ov[kk]) / abs(ov[kk]))
    return out


def polish(mc, mo, ci, weights, tol=1e-10, max_cycle=15, cderi=None, verbose=True):
    '''Newton steps on the exact orbital + CI Hessian from a converged SA-CASSCF (step
    length bounded), the CI vectors kept eigenvectors of H_act in their span; the
    rotations among the states are left to that diagonalization.  Returns (mo, ci, |G|).'''
    ci = model_eigenvectors(mc, mo, ci, cderi)
    gnorm = None
    for it in range(max_cycle):
        resp = SAResponse(mc, mo, ci, weights, cderi=cderi)
        g = resp.gradient()
        gnorm = numpy.linalg.norm(g)
        if verbose:
            print('polish %d: |G| = %.2e, E_SA = %.12f' % (it, gnorm, resp.e_orb))
        if gnorm < tol:
            break
        z = zvector(resp, g, tol=1e-6 * min(1.0, gnorm), extra_null=interstate_directions(resp))
        if numpy.linalg.norm(z) > 0.1:
            z *= 0.1 / numpy.linalg.norm(z)
        mo, ci = move(mc, mo, ci, z)
        ci = model_eigenvectors(mc, mo, ci, cderi)
    return mo, ci, gnorm


# ----------------------------------------------------------------------
# targets in the model space of a multi-state method: sum_IJ rho_IJ hmodel_IJ

def transition_rdm12(resp, bra, ket):
    '''(dm1, dm2) of <bra|...|ket> (the layout of zfci make_rdm12, conjugate-linear in
    bra).'''
    v = numpy.array([bra, ket]).T
    return (_rdm(resp.cas, resp.nelec, v, 1, bra=0, ket=1),
            _rdm(resp.cas, resp.nelec, v, 2, bra=0, ket=1))


def model_rdm12(resp, rho):
    '''sum_IJ rho_IJ <c_I|...|c_J> (rho Hermitian), from the transition density matrices
    of resp.'''
    rho = numpy.asarray(rho)
    return (numpy.einsum('ij,ijpq->pq', rho, resp.tdm1),
            numpy.einsum('ij,ijpqrs->pqrs', rho, resp.tdm2))


def model_gradient(resp, rho):
    '''G of T = sum_IJ rho_IJ hmodel_IJ, hmodel_IJ = <c_I|H_act|c_J> + E_core delta_IJ
    (rho Hermitian): orbitals 2 g(D_rho) (its core part times tr(rho)), CI 2 (g_I - Re<c_I,
    g_I> c_I) with
    g_I = sum_J rho_IJ H_act c_J (dT = 2 Re sum_I <dc_I|g_I>).'''
    dm1, dm2 = model_rdm12(resp, rho)
    # the core part of g(D) (the energy of the core) comes with tr(rho)
    tr = numpy.trace(rho).real
    g_core = _orbital_gradient(resp, numpy.zeros_like(dm1), numpy.zeros_like(dm2))
    parts = [2 * (_orbital_gradient(resp, dm1, dm2) - (1 - tr) * g_core)]
    for i, c in enumerate(resp.ci):
        g = sum(rho[i, j] * resp.hc[j] for j in range(len(resp.ci)))
        parts.append(2 * (g - numpy.vdot(c, g).real * c))
    return resp.join(parts)


def model_density(resp, rho):
    '''AO density of sum_IJ rho_IJ hmodel_IJ (core occupied times tr(rho), sum rho_IJ
    gamma^IJ).'''
    d = mo_density(resp, model_rdm12(resp, rho)[0])
    d[:resp.ncore, :resp.ncore] *= numpy.trace(rho).real
    return ao_density(resp, d)


def degenerate_rotations(resp, tol=1e-9):
    '''Complex columns of the rotations and symmetric mixings among averaged states of
    equal energy and weight: zero modes of J (their coupling ~ w (E_J - E_I) vanishes).'''
    n = resp.n_orb + sum(resp.sizes)
    offs = numpy.concatenate([[resp.n_orb], resp.n_orb + numpy.cumsum(resp.sizes)])
    cols = []
    for i in range(len(resp.ci)):
        for j in range(i):
            if abs(resp.e[i] - resp.e[j]) > tol or abs(resp.weights[i] - resp.weights[j]) > 1e-12:
                continue
            for f in (1.0, 1j):
                for s in (1.0, -1.0):
                    v = numpy.zeros(n, dtype=complex)
                    v[offs[i]:offs[i + 1]] = f * resp.ci[j]
                    v[offs[j]:offs[j + 1]] = s * numpy.conj(f) * resp.ci[i]
                    cols.append(v)
    return numpy.array(cols).T if cols else numpy.zeros((n, 0), dtype=complex)
