'''
Quadruple precision two-component Hamiltonians decoupled from the full
(spin-orbit coupled) four-component Dirac-Coulomb Hamiltonian:

    approx = '1e'     X2C-1e: X, R from the one-electron Dirac matrix h4c
    approx = 'x2cmp'  model potential (socutils.somf.x2cmp, amf_type 'x2cmp',
                      Dirac-Coulomb): X, R from the 4c Fock matrix of the
                      superposition of atomic DHF densities,
                          h = to_2c(F4c[D4c_atoms]) - G2c[D2c_atoms],
                      G = J - K, D2c_atoms the 2c (picture-change transformed)
                      atomic densities.  This is the HF variant of the
                      eamfX2C model of Knecht, Repisky, Jensen and Saue,
                      JCP 157, 114106 (2022).
    approx = 'amfx2c' amfX2C of the same paper: the atomic 4c two-electron
                      Fock matrices G4c_K[D4c_K] enter only the atomic
                      diagonal blocks,
                          X, R from h4c + sum_K G4c_K,
                          h = to_2c(X, R, h4c) + sum_K dF_K,
                          dF_K = to_2c_K(G4c_K) - G2c_K[D2c_K]
                      (X_K, R_K of the atomic Fock matrix).

All matrices are in the decontracted basis (x2c.uncontract_mol) and in the
spin-orbital layout of dhf.build_rkb_hamiltonian:
    4c: [L alpha, L beta, S alpha, S beta],  2c: [alpha, beta].
Atomic densities are computed by quad precision DHF of the free atoms
(closed-shell atoms only), so every atom is exactly neutral:
Tr D4c S4c = Tr D2c S2c = Z to quad precision.
'''

import numpy
from pyscf import gto
from socutils.qdhf import dhf, x2c, qarray
from socutils.qdhf.qarray import QArray, qmp, hmp


def csolve(a, b):
    '''Solve a x = b for complex quad matrices via the real 2n embedding.'''
    a = a.as_complex()
    b = b.as_complex()
    n = a.shape[0]
    big = QArray.zeros((2*n, 2*n))
    big[:n, :n] = QArray(a.re)
    big[:n, n:] = QArray(a.im) * -1
    big[n:, :n] = QArray(a.im)
    big[n:, n:] = QArray(a.re)
    rhs = qarray.concatenate([QArray(b.re), QArray(b.im)], axis=0)
    x = qarray.solve(big, rhs)
    return QArray(x.re[:n], x.re[n:])


def hpow(a, p):
    '''a^p for a Hermitian positive definite quad matrix.'''
    w, v = qarray.eigh(a)
    wp = numpy.array([qmp.mpf(x)**p for x in w.to_mpmath()], dtype=object)
    d = qarray.diag(QArray.from_mpf(wp))
    return v @ d @ v.conj().T


def get_r(s, st):
    '''R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2,  R^H S~ R = S'''
    si = hpow(s, qmp.mpf(-0.5))
    sp = hpow(s, qmp.mpf(0.5))
    return si @ hpow(si @ st @ si, qmp.mpf(-0.5)) @ sp


def decouple(h4c, s4c, return_vectors=False):
    '''X and R from the positive-energy eigenvectors of (h4c, s4c).'''
    n4 = h4c.shape[0]
    n2 = n4 // 2
    x = hpow(s4c, qmp.mpf(-0.5))
    e, c = qarray.eigh(x @ h4c @ x)
    c = x @ c
    cl = c[:n2, n2:]
    cs = c[n2:, n2:]
    # X C_L = C_S  <=>  C_L^T X^T = C_S^T
    xm = csolve(cl.T, cs.T).T
    st = s4c[:n2, :n2] + xm.conj().T @ s4c[n2:, n2:] @ xm
    r = get_r(s4c[:n2, :n2], st)
    if return_vectors:
        return xm, r, c[:, n2:]
    return xm, r


def to_2c(xm, r, m4c):
    n2 = m4c.shape[0] // 2
    L, S = slice(0, n2), slice(n2, 2*n2)
    mlsx = m4c[L, S] @ xm
    l = m4c[L, L] + mlsx + mlsx.conj().T + xm.conj().T @ m4c[S, S] @ xm
    return r.conj().T @ l @ r


def _atom_mol(xmol, ia):
    symb = xmol.atom_symbol(ia)
    return gto.M(atom=[[symb, (0., 0., 0.)]], basis={symb: xmol._basis[symb]},
                 cart=xmol.cart, spin=None, verbose=0)


def atomic_densities(xmol, light_speed=None, cache=None):
    '''Superposition of atomic 4c DHF densities and of their 2c images
    D2c = R^-1 D_LL R^-H (X, R of the atomic Fock matrix).'''
    if cache is None:
        cache = {}
    n = xmol.nao_nr(xmol.cart)
    D4 = QArray.zeros((4*n, 4*n), iscomplex=True)
    D2 = QArray.zeros((2*n, 2*n), iscomplex=True)
    aoslices = xmol.aoslice_by_atom() if not xmol.cart else xmol.aoslice_nr_by_atom()
    for ia in range(xmol.natm):
        p0, p1 = aoslices[ia][2:]
        symb = xmol.atom_symbol(ia)
        if symb not in cache:
            amol = _atom_mol(xmol, ia)
            if amol.nelectron % 2:
                raise NotImplementedError('open-shell atom %s' % symb)
            mf = dhf.DHF(amol, light_speed=light_speed)
            mf.verbose = 0
            mf.conv_tol = 1e-32
            mf.conv_tol_grad = 1e-17
            mf.max_cycle = 100
            mf.kernel()
            if not mf.converged:
                raise RuntimeError('atomic DHF of %s not converged' % symb)
            ham = mf.ham
            vj, vk = ham.get_jk(mf.make_rdm1().as_complex())
            f4 = ham.hcore + vj - vk
            # D4c and D2c from the same occupied eigenvectors of the converged
            # Fock matrix, so that Tr D4c S4c = Tr D2c S2c = N exactly
            xa, ra, cpes = decouple(f4, ham.ovlp, return_vectors=True)
            na = amol.nao_nr(amol.cart)
            cocc = cpes[:, :amol.nelectron]
            d4 = cocc @ cocc.conj().T
            rinv = csolve(ra, QArray.eye(2*na).as_complex())
            c2 = rinv @ cocc[:2*na]
            d2 = c2 @ c2.conj().T
            # atomic two-electron Fock matrices and the amfX2C correction
            vj, vk = ham.get_jk(d4)
            g4 = vj - vk
            ham2 = dhf.build_nr_hamiltonian(amol, 'GHF', verbose=0)
            vj2, vk2 = ham2.get_jk(d2)
            df2 = to_2c(xa, ra, g4) - (vj2 - vk2)
            cache[symb] = (d4, d2, na, g4, df2)
        d4, d2, na = cache[symb][:3]
        assert na == p1 - p0
        _embed(D4, d4, n, na, p0, 4)
        _embed(D2, d2, n, na, p0, 2)
    return D4, D2


def _embed(big, small, n, na, p0, ncomp):
    '''Copy the atomic matrix (ncomp x ncomp blocks of size na) into the
    molecular matrix (blocks of size n) at AO offset p0.'''
    p1 = p0 + na
    for ci in range(ncomp):
        for cj in range(ncomp):
            big[ci*n+p0:ci*n+p1, cj*n+p0:cj*n+p1] = small[ci*na:(ci+1)*na, cj*na:(cj+1)*na]


def atomic_blocks(xmol, light_speed=None, cache=None):
    '''Block-diagonal superpositions of the atomic G4c_K (4c) and dF_K (2c).'''
    if cache is None:
        cache = {}
    atomic_densities(xmol, light_speed, cache)
    n = xmol.nao_nr(xmol.cart)
    G4 = QArray.zeros((4*n, 4*n), iscomplex=True)
    DF = QArray.zeros((2*n, 2*n), iscomplex=True)
    aoslices = xmol.aoslice_by_atom() if not xmol.cart else xmol.aoslice_nr_by_atom()
    for ia in range(xmol.natm):
        p0 = aoslices[ia][2]
        d4, d2, na, g4, df2 = cache[xmol.atom_symbol(ia)]
        _embed(G4, g4, n, na, p0, 4)
        _embed(DF, df2, n, na, p0, 2)
    return G4, DF


def get_hcore(xmol, approx='1e', light_speed=None, ham4c=None, ham2c=None, cache=None):
    '''2c core Hamiltonian (decontracted spin-orbital basis of xmol).'''
    if ham4c is None:
        ham4c = dhf.build_rkb_hamiltonian(xmol, light_speed, verbose=0)
    h4 = ham4c.hcore.as_complex()
    s4 = ham4c.ovlp
    if approx.upper() == '1E':
        xm, r = decouple(h4, s4)
        return to_2c(xm, r, h4)
    elif approx.upper() == 'X2CMP':
        if ham2c is None:
            ham2c = dhf.build_nr_hamiltonian(xmol, 'GHF', verbose=0)
        d4, d2 = atomic_densities(xmol, light_speed, cache)
        vj, vk = ham4c.get_jk(d4)
        f4 = h4 + vj - vk
        xm, r = decouple(f4, s4)
        vj2, vk2 = ham2c.get_jk(d2)
        return to_2c(xm, r, f4) - (vj2 - vk2)
    elif approx.upper() == 'AMFX2C':
        g4, df = atomic_blocks(xmol, light_speed, cache)
        xm, r = decouple(h4 + g4, s4)
        return to_2c(xm, r, h4) + df
    raise ValueError('Unknown approx %s' % approx)


class GHF2c(dhf.GHF):
    '''Quad precision 2c GHF in the decontracted basis of mol with a 2c core
    Hamiltonian from the 4c Dirac-Coulomb Hamiltonian ('1e', 'x2cmp' or 'amfx2c');
    approx = None gives the nonrelativistic GHF in the same basis.'''
    approx = '1e'
    light_speed = None
    cache = None

    def __init__(self, mol, approx='1e'):
        xmol, _ = x2c.uncontract_mol(mol)
        dhf.GHF.__init__(self, xmol)
        self.approx = approx
        self.init_guess = 'double'

    def build_hamiltonian(self):
        ham = dhf.build_nr_hamiltonian(self.mol, 'GHF', self.verbose)
        if self.approx is not None:
            ham.hcore = get_hcore(self.mol, self.approx, self.light_speed,
                                  ham2c=ham, cache=self.cache)
        self.ham = ham
        return ham
