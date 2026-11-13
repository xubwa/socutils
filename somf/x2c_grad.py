from functools import reduce
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.x2c import x2c
import numpy,math
import scipy.linalg
from socutils.grad.x2c_grad_g import _block_diag_xyz
from pyscf.x2c.sfx2c1e_grad import _gen_h1_s1
from numpy.linalg import norm

'''
Analytical energy gradients for x2c1e method

Ref.
JCP 135, 084114 (2011); DOI:10.1063/1.3624397
'''

def hcore_grad_generator_spin_free(x2cobj, mol=None):
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff = x2cobj.get_xmol(mol)

    get_h1_xmol = _gen_h1_s1(xmol)
    t = xmol.intor_symmetric('int1e_kin')
    v = xmol.intor_symmetric('int1e_nuc')
    s = xmol.intor_symmetric('int1e_ovlp')
    w = xmol.intor_symmetric('int1e_pnucp')
    def hcore_deriv(atm_id):
        h1, s1 = get_h1_xmol(atm_id)
        hfw1 = numpy.asarray([x2c1e_hfw1(xmol, h1[i], s1[i], t, v, s, w) for i in range(3)])   
        if contr_coeff is not None:
            hfw1_ctr = lib.einsum('pi,xpq,qj->xij', contr_coeff.conj(), hfw1, contr_coeff)
        return numpy.asarray(hfw1_ctr.real)
    return hcore_deriv

def hcore_grad_generator_spinor(x2cobj, mol=None):
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff_nr = x2cobj.get_xmol(mol)

    get_h1_xmol = get_h1nuc_s1(xmol)
    def hcore_deriv(atm_id):
        h1, s1 = get_h1_xmol(atm_id)
        hfw1 = numpy.asarray([x2c1e_hfw1(xmol, h1[i], s1[i]) for i in range(3)])   
        if contr_coeff_nr is not None:
            np, nc = contr_coeff_nr.shape
            contr_coeff = numpy.zeros((np*2,nc*2))
            contr_coeff[0::2,0::2] = contr_coeff_nr
            contr_coeff[1::2,1::2] = contr_coeff_nr
            hfw1_ctr = lib.einsum('pi,xpq,qj->xij', contr_coeff.conj(), hfw1, contr_coeff)
        return numpy.asarray(hfw1_ctr)
    return hcore_deriv

def hcore_grad_generator_spin_orbital(x2cobj, mol=None):
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff_nr = x2cobj.get_xmol(mol)

    get_h1_xmol = get_h1nuc_s1(xmol)
    ca, cb = mol.sph2spinor_coeff()
    c2 = numpy.vstack((ca, cb))
    def hcore_deriv(atm_id):
        h1, s1 = get_h1_xmol(atm_id)
        hfw1 = numpy.asarray([x2c1e_hfw1(xmol, h1[i], s1[i]) for i in range(3)])   
        if contr_coeff_nr is not None:
            np, nc = contr_coeff_nr.shape
            contr_coeff = numpy.zeros((np*2,nc*2))
            contr_coeff[0::2,0::2] = contr_coeff_nr
            contr_coeff[1::2,1::2] = contr_coeff_nr
            hfw1_ctr = lib.einsum('ai,pi,xpq,qj,bj->xab', c2, contr_coeff, hfw1, contr_coeff, c2.conj())
        return numpy.asarray(hfw1_ctr)
    return hcore_deriv

def hcore_deriv_generator_spin_orbital(self, mol=None, deriv=1):
    if deriv == 1:
        return hcore_grad_generator_spin_orbital(self, mol)
    elif deriv == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError

def hcore_deriv_generator_spinor(self, mol=None, deriv=1):
    if deriv == 1:
        return hcore_grad_generator_spinor(self, mol)
    elif deriv == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError
    
x2c.SpinorX2CHelper.hcore_deriv_generator = hcore_deriv_generator_spinor # type: ignore
x2c.SpinOrbitalX2CHelper.hcore_deriv_generator = hcore_deriv_generator_spin_orbital # type: ignore

# ---------------------------------------------------------------------------
# The X2C hcore term of the nuclear gradient from the flavour's own 4c matrices.
#
# Every X2C flavour here is hcore = C^+ [R^+ L(h4c) R + h2c_extra] C (decontracted spinor basis of xmol,
# C its contraction), the decoupling (X, R) of the 4c pair (h4c0, s4c0) the energy used: the bare Dirac
# one-electron matrices ('x2camf', '1e', 'x2camf_sd', pyscf's helpers), the spin-free ones ('sf1e'), the
# molecular-mean-field 4c Fock h1e + J-K[D4c] ('x2cmp'), with -J-K[D2c] (2c) and the atomic spin-orbit
# matrices (geometry independent) as h2c_extra.  For a density D' = C D C^+,
#     d/dR tr(D' hfw) = tr(Gh dh4c/dR) + tr(GS ds4c/dR) + tr(D' dh2c_extra/dR),
# (Gh, GS) of hfw1_adjoint from one decoupling, dh4c/dR, ds4c/dR per atom (get_h1nuc_s1, plus the 4c J-K
# derivative at fixed D4c for 'x2cmp').  nuc_grad_hcore is the helpers' interface (attached to the pyscf /
# socutils helper classes below); ghf_grad.grad_elec calls it.  HCORE_GRAD = 'perturb': the per-atom
# generators above instead (get_hfw1 of the bare one-electron 4c matrices per atom and direction, the
# reference; X2C1e of the bare integrals whatever the flavour).
# ---------------------------------------------------------------------------
HCORE_GRAD = 'adjoint'


def _contraction(x2cobj, mol):
    '''xmol of the helper and its contraction in the spinor basis (interleaved), or None'''
    xmol, contr_coeff_nr = x2cobj.get_xmol(mol)
    if contr_coeff_nr is None:
        return xmol, None
    np_, nc = contr_coeff_nr.shape
    contr = numpy.zeros((np_ * 2, nc * 2))
    contr[0::2, 0::2] = contr_coeff_nr
    contr[1::2, 1::2] = contr_coeff_nr
    return xmol, contr


def _split4c(h4c0, s4c0):
    n = h4c0.shape[0] // 2
    return x2c1e_hfw0_block(h4c0[:n, :n], h4c0[n:, :n], h4c0[:n, n:], h4c0[n:, n:], s4c0[:n, :n], s4c0[n:, n:])


def hfw_adjoint_4c(h4c0, s4c0, dm2c):
    '''(Gh, GS): tr(dm2c hfw1) = tr(Gh h4c1) + tr(GS s4c1) for the X2C decoupling of (h4c0, s4c0)'''
    a, e, x, st, r, l, h4c, m4c = _split4c(h4c0, s4c0)
    return hfw1_adjoint(a, x, st, m4c, h4c, e, r, l, dm2c)


def hfw1_4c(h4c0, s4c0, h4c1, s4c1=None):
    '''hfw1 of one perturbation (h4c1, s4c1) of the decoupling of (h4c0, s4c0) (get_hfw1, the reference)'''
    a, e, x, st, r, l, h4c, m4c = _split4c(h4c0, s4c0)
    return get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)


def _bare_4c(xmol, spin_free=False):
    '''the bare Dirac one-electron 4c pair (h4c, s4c) of xmol (pyscf.scf.dhf / the 'sf1e' flavour)'''
    c = LIGHT_SPEED
    t = xmol.intor('int1e_spsp_spinor') * .5
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_pnucp_spinor' if spin_free else 'int1e_spnucsp_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    n = s.shape[0]
    h4c = numpy.zeros((2 * n, 2 * n), dtype=complex)
    m4c = numpy.zeros((2 * n, 2 * n), dtype=complex)
    h4c[:n, :n] = v
    h4c[:n, n:] = h4c[n:, :n] = t
    h4c[n:, n:] = w * (.25 / c**2) - t
    m4c[:n, :n] = s
    m4c[n:, n:] = t * (.5 / c**2)
    return h4c, m4c


def flavour_4c(x2cobj, mol=None):
    '''(h4c0, s4c0, spin_free, extra) of the helper's X2C Hamiltonian, the 4c matrices its energy used; extra:
    None or ('x2cmp', D4c, D2c) for the molecular mean field (h4c0 = h1e + J-K[D4c], h2c_extra = -J-K[D2c]).
    Raises NotImplementedError for flavours without an analytic gradient here.'''
    if mol is None:
        mol = x2cobj.mol
    src = getattr(x2cobj, '_grad_src', None) or x2cobj
    flavour = getattr(src, 'amf_type', None)
    if flavour is None:
        # pyscf's X2C helpers: X2C1e of the bare integrals
        if getattr(src, 'approx', '1e').lower() not in ('1e', 'x2c1e'):
            raise NotImplementedError('X2C gradient for approx %s' % src.approx)
        h4c, m4c = _bare_4c(x2cobj.get_xmol(mol)[0])
        return h4c, m4c, False, None
    if flavour not in ('x2camf', 'x2camf_sd', '1e', 'sf1e', 'x2cmp'):
        raise NotImplementedError('analytic nuclear gradient of the X2C flavour %r' % flavour)
    if src.h4c is None or (flavour == 'x2cmp' and getattr(src, 'density_4c', None) is None):
        src.get_hcore(mol)
    if flavour == 'x2cmp':
        if src.gaunt or src.breit:
            raise NotImplementedError('x2cmp gradient with Gaunt / Breit (the 4c Gaunt / Breit derivative '
                                      'integrals of J-K[D4c])')
        if getattr(src, 'h4c_ext', None) is not None:
            raise NotImplementedError('x2cmp gradient with h4c_ext')
        return src.h4c, src.m4c, False, ('x2cmp', src.density_4c, src.density_2c)
    return src.h4c, src.m4c, flavour == 'sf1e', None


def _vhf1_4c(xmol, dm4c, level='SSSS'):
    from pyscf.grad import dhf as dhf_grad
    return dhf_grad.get_coulomb_hf(xmol, numpy.ascontiguousarray(dm4c, dtype=complex), level=level)


def nuc_grad_hcore(x2cobj, dm, mol=None, basis='spin_orbital'):
    '''sum_ij dm_ji d hcore_ij / dR_A (natm, 3) of the helper's X2C hcore, dm a GHF density (2nao, 2nao) for
    basis 'spin_orbital' or a j-adapted spinor density for 'spinor': one hfw1_adjoint of the flavour's 4c pair,
    one contraction with dh4c/dR, ds4c/dR per atom (see above)'''
    if mol is None:
        mol = x2cobj.mol
    h4c0, s4c0, spin_free, extra = flavour_4c(x2cobj, mol)
    xmol, contr = _contraction(x2cobj, mol)
    m = numpy.asarray(dm)
    if basis == 'spin_orbital':
        ca, cb = mol.sph2spinor_coeff()
        c2 = numpy.vstack((ca, cb))
        m = c2.conj().T @ m @ c2
    if contr is not None:
        m = contr @ m @ contr.T
    m = _herm(m)
    gh, gs = hfw_adjoint_4c(h4c0, s4c0, m)
    get_h1_s1 = get_h1nuc_s1(xmol, spin_free=spin_free)
    de = numpy.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        h1, s1 = get_h1_s1(ia)
        de[ia] = (numpy.einsum('pq,xqp->x', gh, h1) + numpy.einsum('pq,xqp->x', gs, s1)).real
    if extra is not None and extra[0] == 'x2cmp':
        # d/dR tr(Gh J-K[D4c]) at fixed Gh, D4c and -d/dR tr(D' J-K[D2c]) (2c, LL): for the bilinear
        # B(X, Y) = tr(X G[Y]), dB/dR_A = 2 Re sum_{i on A} (G1[Y]_ij X_ji + G1[X]_ij Y_ji) (pyscf.grad.dhf)
        d4c, d2c = extra[1], extra[2]
        n2c = d2c.shape[0]
        v_d = _vhf1_4c(xmol, d4c)
        v_g = _vhf1_4c(xmol, gh)
        pad = numpy.zeros((2 * n2c, 2 * n2c), dtype=complex)
        pad[:n2c, :n2c] = d2c
        w_d = _vhf1_4c(xmol, pad, 'LLLL')
        pad[:n2c, :n2c] = m
        w_m = _vhf1_4c(xmol, pad, 'LLLL')
        for ia, (p0, p1) in enumerate(xmol.aoslice_2c_by_atom()[:, 2:]):
            for q0, q1 in ((p0, p1), (p0 + n2c, p1 + n2c)):
                de[ia] += 2 * (numpy.einsum('xij,ji->x', v_d[:, q0:q1], gh[:, q0:q1])
                               + numpy.einsum('xij,ji->x', v_g[:, q0:q1], d4c[:, q0:q1])).real
            de[ia] -= 2 * (numpy.einsum('xij,ji->x', w_d[:, p0:p1, :n2c], m[:, p0:p1])
                           + numpy.einsum('xij,ji->x', w_m[:, p0:p1, :n2c], d2c[:, p0:p1])).real
    return de


def _nuc_grad_hcore_spin_orbital(self, dm, mol=None):
    return nuc_grad_hcore(self, dm, mol, basis='spin_orbital')


def _nuc_grad_hcore_spinor(self, dm, mol=None):
    return nuc_grad_hcore(self, dm, mol, basis='spinor')


x2c.SpinOrbitalX2CHelper.nuc_grad_hcore = _nuc_grad_hcore_spin_orbital    # type: ignore
x2c.SpinorX2CHelper.nuc_grad_hcore = _nuc_grad_hcore_spinor                # type: ignore


def hcore_grad(x2cobj, dm, mol=None, basis='spin_orbital'):
    '''alias of nuc_grad_hcore'''
    return nuc_grad_hcore(x2cobj, dm, mol, basis)

def get_h1nuc_s1(mol, spin_free=False):
    '''per atom the nuclear derivatives (3, n4c, n4c) of the bare Dirac one-electron h4c and s4c (spin_free:
    with p.V p in place of sigma.p V sigma.p, the 'sf1e' flavour); the nuclear attraction of the atom by
    with_rinv_at_nucleus (its nuclear model)'''
    c = LIGHT_SPEED
    s1 = mol.intor('int1e_ipovlp_spinor', comp=3)
    t1 = mol.intor('int1e_ipkin_spinor', comp=3)
    v1 = mol.intor('int1e_ipnuc_spinor', comp=3)
    w1 = mol.intor('int1e_ippnucp_spinor' if spin_free else 'int1e_ipspnucsp_spinor', comp=3)
    aoslices = mol.aoslice_2c_by_atom()
    n2c = s1.shape[1]
    n4c = n2c * 2

    def get_h1_s1(ia):
        h1 = numpy.zeros((3,n4c,n4c), dtype=complex)
        m1 = numpy.zeros((3,n4c,n4c), dtype=complex)
        ish0, ish1, i0, i1 = aoslices[ia]
        with mol.with_rinv_at_nucleus(ia):
            z = mol.atom_charge(ia)
            rinv1 = -z*mol.intor('int1e_iprinv_spinor', comp=3)
            sprinvsp1 = -z*mol.intor('int1e_ipprinvp_spinor' if spin_free else 'int1e_ipsprinvsp_spinor', comp=3)
        rinv1[:,i0:i1,:] -= v1[:,i0:i1]
        sprinvsp1[:,i0:i1,:] -= w1[:,i0:i1]
        for i in range(3):
            s1cc = numpy.zeros((n2c, n2c), dtype=complex)
            t1cc = numpy.zeros((n2c, n2c), dtype=complex)
            s1cc[i0:i1,:] = -s1[i,i0:i1]
            s1cc[:,i0:i1]-= s1[i,i0:i1].T.conj()
            t1cc[i0:i1,:] =-t1[i,i0:i1]
            t1cc[:,i0:i1]-= t1[i,i0:i1].T.conj()
            v1cc = rinv1[i]   + rinv1[i].T.conj()
            w1cc = sprinvsp1[i] + sprinvsp1[i].T.conj()

            h1[i,:n2c,:n2c] = v1cc
            h1[i,:n2c,n2c:] = t1cc
            h1[i,n2c:,:n2c] = t1cc
            h1[i,n2c:,n2c:] = w1cc * (.25/c**2) - t1cc
            m1[i,:n2c,:n2c] = s1cc
            m1[i,n2c:,n2c:] = t1cc * (.5/c**2)
        return h1, m1
    return get_h1_s1

# ---------------------------------------------------------------------------
# The per-perturbation derivative of the X2C Hamiltonian (get_hfw1), JCP 135,
# 084114 (2011).  Matrix square roots and inverse square roots from one eigh
# each, eigenvalues below EIG_TOL dropped (pyscf.x2c.x2c._get_r): S^-1/2,
# S^1/2 from the same eigenpairs, (S^-1/2 S~ S^-1/2)^-1/2; their derivatives
# in the eigenbasis (vectorised); X1 = (CS1 - X0 CL1) CL^-1 by a solve.
# HFW1_IMPL = 'sqrtm' runs the original steps (scipy sqrtm / inv, Python loops;
# the _*_sqrtm functions below) as the reference.
# ---------------------------------------------------------------------------
EIG_TOL = 1e-14
HFW1_IMPL = 'eigh'      # 'eigh' | 'sqrtm' (the original implementation)


def _eigh_pos(a, tol=None):
    '''eigenpairs of the Hermitian a with eigenvalues above tol (EIG_TOL)'''
    e, v = numpy.linalg.eigh(a)
    idx = e > (EIG_TOL if tol is None else tol)
    return e[idx], v[:, idx]


def _from_eig(e, v, f):
    return (v * f(e)) @ v.conj().T


def get_Asq1(A, A1, eig=None):
    '''(A^1/2)^lambda, eig: the eigenpairs of A (_eigh_pos)'''
    e, v = _eigh_pos(A) if eig is None else eig
    return _sqrt_deriv(e, v, A1)


def get_Asqi1(A, A1, eig=None):
    '''(A^-1/2)^lambda = -A^-1/2 (A^1/2)^lambda A^-1/2'''
    e, v = _eigh_pos(A) if eig is None else eig
    return _invsqrt_deriv(e, v, A1)


def get_C1(C0, mo_ene, D1, S4c1=None, cols=None):
    '''C1 = C0 U1 (Ref. (32)-(33)); cols: the columns of C1 wanted (default all).  U1_ij = (D1_ij - S1_ij e_j) /
    (e_j - e_i), 0 for |e_j - e_i| < 1e-5, U1_jj = -S1_jj / 2'''
    N = C0.shape[1]
    cols = numpy.arange(N) if cols is None else numpy.arange(N)[cols]
    Cc = C0[:, cols]
    D1mo = C0.conj().T @ D1 @ Cc
    S1mo = None if S4c1 is None else C0.conj().T @ S4c1 @ Cc
    e = mo_ene
    den = e[None, cols] - e[:, None]
    fac = numpy.where(abs(den) < 1e-5, 0, 1 / numpy.where(abs(den) < 1e-5, 1, den))
    rows = cols
    fac[rows, numpy.arange(len(cols))] = 0
    U1 = D1mo * fac if S1mo is None else (D1mo - S1mo * e[None, cols]) * fac
    if S1mo is not None:
        U1[rows, numpy.arange(len(cols))] = -.5 * S1mo[rows, numpy.arange(len(cols))]
    return C0 @ U1


def get_X1(C0, C1, X0):
    '''X1 = (CS1 - X0 CL1) CL^-1 (Ref. (31)); C1: all columns or the positive-energy ones'''
    size = C0.shape[0] // 2
    CL = C0[:size, size:]
    if C1.shape[1] != size:
        C1 = C1[:, size:]
    m = C1[size:] - X0 @ C1[:size]
    return numpy.linalg.solve(CL.T, m.T).T


def get_R1(S0, ST, ST1, S1=None, seig=None, aeig=None):
    '''R1 for R = S^-1/2 (S^-1/2 ST S^-1/2)^-1/2 S^1/2 (Ref. (34)), from the eigenpairs of S (seig) and
    A = S^-1/2 ST S^-1/2 (aeig)'''
    se, sv = _eigh_pos(S0) if seig is None else seig
    Sq = _from_eig(se, sv, numpy.sqrt)
    Si = _from_eig(se, sv, lambda x: 1 / numpy.sqrt(x))
    if aeig is None:
        aeig = _eigh_pos(Si @ ST @ Si)
    ae, av = aeig
    A1 = Si @ ST1 @ Si
    if S1 is not None:
        Si1 = _invsqrt_deriv(se, sv, S1)
        A1 = A1 + Si1 @ ST @ Si + Si @ ST @ Si1
    R1 = Si @ _invsqrt_deriv(ae, av, A1) @ Sq
    if S1 is not None:
        Ai = _from_eig(ae, av, lambda x: 1 / numpy.sqrt(x))
        R1 = R1 + Si1 @ Ai @ Sq + Si @ Ai @ _sqrt_deriv(se, sv, S1)
    return R1


def _get_Asq1_sqrtm(A, A1):
    # A = B^2
    # B = A^{1/2}
    # return B^lambda
    # Ref (A3)-(A7)
    size = A.shape[0]
    Aeig, Avec = scipy.linalg.eigh(A)
    YA1Y = reduce(numpy.dot, (Avec.T.conj(), A1, Avec))
    YB1Y = numpy.zeros((size,size), dtype=YA1Y.dtype)
    for ii in range(size):
        for jj in range(size):
            dominator = math.sqrt(Aeig[ii]) + math.sqrt(Aeig[jj])
            if abs(dominator) < 1e-10:
                YB1Y[ii,jj] = 0.0
            else:
                YB1Y[ii,jj] = YA1Y[ii,jj] / dominator
    return reduce(numpy.dot, (Avec, YB1Y, Avec.T.conj()))

def _get_Asqi1_sqrtm(A, A1):
    # A = B^2
    # B = A^{1/2}
    # return (B^{-1})^lambda
    # Ref (A3)-(A7)
    size = A.shape[0]
    Aeig, Avec = scipy.linalg.eigh(A)
    YA1Y = reduce(numpy.dot, (Avec.T.conj(), A1, Avec))
    YB1Y = numpy.zeros((size,size), dtype=YA1Y.dtype)
    Asqieig = numpy.zeros((size,size))
    for ii in range(size):
        if(Aeig[ii] > 0.0):
            Asqieig[ii,ii] = 1.0/math.sqrt(Aeig[ii])
        for jj in range(size):
            dominator = math.sqrt(Aeig[ii]) + math.sqrt(Aeig[jj])
            if abs(dominator) < 1e-10:
                YB1Y[ii,jj] = 0.0
            else:
                YB1Y[ii,jj] = YA1Y[ii,jj] / dominator
    B1 = reduce(numpy.dot, (Avec, YB1Y, Avec.T.conj()))
    Binv = reduce(numpy.dot, (Avec, Asqieig, Avec.T.conj()))
    # Ref (A1)
    return -1.0 * reduce(numpy.dot, (Binv, B1, Binv))

def _get_C1_sqrtm(C0, mo_ene, D1, S4c1=None):
    # Ref (32)-(33)
    D1_mo = reduce(numpy.dot,(C0.T.conj(), D1, C0))
    size = D1_mo.shape[0]
    U1 = numpy.zeros((size,size), dtype=D1_mo.dtype)
    if(S4c1 is None):
        S1_mo = numpy.zeros((size,size), dtype=D1_mo.dtype)
    else:
        S1_mo = reduce(numpy.dot,(C0.T.conj(), S4c1, C0))
    for ii in range(size):
        for jj in range(size):
            if(ii == jj):
                U1[ii,ii] = -0.5*S1_mo[ii,ii]
            else:
                denominator = mo_ene[jj]-mo_ene[ii]
                if(abs(denominator) < 1e-5):
                    U1[ii,jj] = 0.0
                else:
                    U1[ii,jj] = (D1_mo[ii,jj] - S1_mo[ii,jj]*mo_ene[jj])/denominator
    return numpy.dot(C0,U1)

def _get_X1_sqrtm(C0, C1, X0):
    # Ref (31)
    size = C0.shape[0]//2
    CL = C0[:size, size:]
    CL1 = C1[:size, size:]
    CS1 = C1[size:, size:]
    CS1 = CS1 - numpy.dot(X0,CL1)
    tmp = numpy.dot(CL, CL.T.conj())
    return reduce(numpy.dot, (CS1, CL.T.conj(), numpy.linalg.inv(tmp)))   # numerically a little bit more stable
    # return numpy.dot(CS1,scipy.linalg.inv(CL))
    # return numpy.linalg.solve(CL.T, CS1.T).T

def get_ST1(S4c0, X0, X1=None, S4c1=None):
    size = S4c0.shape[0]//2
    if(S4c1 is None):
        ST1 = numpy.zeros((size,size),dtype=S4c0.dtype)
    else:
        ST1 = S4c1[:size, :size]
    if(X1 is not None):
        X1TX = reduce(numpy.dot, (X1.T.conj(), S4c0[size:, size:], X0))
        ST1 = ST1 + X1TX + X1TX.T.conj()
    if(S4c1 is not None):
        ST1 = ST1 + reduce(numpy.dot, (X0.T.conj(), S4c1[size:, size:], X0))
    return ST1

def _get_R1_sqrtm(S0, ST, ST1, S1=None):
    # Ref (34)
    # A = S^{-1/2}*ST*S^{-1/2}
    Ssq0 = scipy.linalg.sqrtm(S0)
    Ssqinv0 = scipy.linalg.inv(Ssq0)
    A = reduce(numpy.dot, (Ssqinv0, ST, Ssqinv0))

    # If S1 = 0, R1 is the middle term of (34)
    if S1 is None:
        A1 = reduce(numpy.dot, (Ssqinv0, ST1, Ssqinv0))
    else:
        Ssqinv1 = _get_Asqi1_sqrtm(S0, S1)
        Ssq1 = _get_Asq1_sqrtm(S0, S1)
        A1 = reduce(numpy.dot, (Ssqinv0, ST1, Ssqinv0))\
           + reduce(numpy.dot, (Ssqinv1, ST, Ssqinv0))\
           + reduce(numpy.dot, (Ssqinv0, ST, Ssqinv1))
    R1 = reduce(numpy.dot, (Ssqinv0, _get_Asqi1_sqrtm(A, A1), Ssq0))
    if(S1 is not None):
        Asqinv = scipy.linalg.inv(scipy.linalg.sqrtm(A))
        R1 = R1 + reduce(numpy.dot, (Ssqinv1, Asqinv, Ssq0))\
                + reduce(numpy.dot, (Ssqinv0, Asqinv, Ssq1))
    # sqrtm will return complex256 when the input is complex128
    # it will break the other parts of the code
    if R1.dtype == numpy.complex256:
        R1 = R1.astype(numpy.complex128)
    return R1

def get_L1(h4c0, h4c1, X0, X1):
    # Ref (30)
    size = h4c0.shape[0]//2
    hLS = h4c0[:size,size:]
    hSS = h4c0[size:,size:]
    hLL1 = h4c1[:size,:size]
    hLS1 = h4c1[:size,size:]
    hSS1 = h4c1[size:,size:]

    TX1 = numpy.dot(hLS,X1)
    T1X = numpy.dot(hLS1,X0)
    X1TX = reduce(numpy.dot, (X1.T.conj(), hSS, X0))
    XT1X = reduce(numpy.dot, (X0.T.conj(), hSS1, X0))
    return hLL1 + TX1 + TX1.T.conj() + T1X + T1X.T.conj() + X1TX + X1TX.T.conj() + XT1X


def get_hfw1(C4c0, X0, ST, S4c0, h4c0, mo_ene, R0, L0, h4c1, S4c1=None):
    size2c = C4c0.shape[0]//2
    S2c0 = S4c0[:size2c,:size2c]
    if(S4c1 is None):
        S2c1 = None
    else:
        S2c1 = S4c1[:size2c,:size2c]
    if HFW1_IMPL == 'sqrtm':
        C4c1 = _get_C1_sqrtm(C4c0, mo_ene, h4c1, S4c1)
        X1 = _get_X1_sqrtm(C4c0, C4c1, X0)
        ST1 = get_ST1(S4c0, X0, X1, S4c1)
        R1 = _get_R1_sqrtm(S2c0, ST, ST1, S2c1)
    else:
        C4c1 = get_C1(C4c0, mo_ene, h4c1, S4c1, cols=slice(size2c, None))
        X1 = get_X1(C4c0, C4c1, X0)
        ST1 = get_ST1(S4c0, X0, X1, S4c1)
        R1 = get_R1(S2c0, ST, ST1, S2c1)
    L1 = get_L1(h4c0, h4c1, X0, X1)
    R1LR = reduce(numpy.dot, (R1.T.conj(), L0, R0))
    RL1R = reduce(numpy.dot, (R0.T.conj(), L1, R0))
    hfw1 = RL1R + R1LR + R1LR.T.conj()

    return hfw1

# ---------------------------------------------------------------------------
# Effective 4c densities: the adjoint of get_hfw1.
#
# get_hfw1 is (real-)linear in (h4c1, S4c1).  For a fixed 2c density D the
# property tr(D hfw1(h4c1, S4c1)) is therefore tr(Gh h4c1) + tr(GS S4c1) with
# Hermitian 4c matrices Gh, GS that depend on D only: the derivatives of X and
# R are transformed away (cf. Zou, Guo, Suo, Liu, JCTC 16, 1541 (2020) for the
# nuclear gradients).  One evaluation per density, then every one-electron
# perturbation is a contraction.  hfw1_adjoint applies the adjoints of the
# steps of get_hfw1 (C1, X1, ST1, R1, L1) in reverse order, with the inner
# product <A, B> = Re tr(A^+ B).
# ---------------------------------------------------------------------------

def _herm(a):
    return (a + a.conj().T) * .5


def _sqrt_deriv(aeig, avec, a1):
    '''B^lambda for B = A^1/2 (get_Asq1): self-adjoint.'''
    s = numpy.sqrt(aeig)
    den = s[:, None] + s[None, :]
    y = avec.conj().T @ a1 @ avec
    y = numpy.where(abs(den) < 1e-10, 0, y / numpy.where(abs(den) < 1e-10, 1, den))
    return avec @ y @ avec.conj().T


def _invsqrt_deriv(aeig, avec, a1):
    '''(B^-1)^lambda (get_Asqi1): -B^-1 B^lambda B^-1, self-adjoint.'''
    binv = (avec / numpy.sqrt(aeig)) @ avec.conj().T
    return -binv @ _sqrt_deriv(aeig, avec, a1) @ binv


def hfw1_adjoint(C4c0, X0, ST, S4c0, h4c0, mo_ene, R0, L0, D):
    '''Effective 4c densities (Gh, GS) of the 2c density D (Hermitian, in the
    basis of X0/R0): tr(D get_hfw1(..., h4c1, S4c1)) = tr(Gh h4c1) + tr(GS S4c1)
    for all Hermitian h4c1, S4c1.'''
    N = C4c0.shape[0]
    n = N // 2
    S2c0 = S4c0[:n, :n]
    hSL = h4c0[n:, :n]
    hSS = h4c0[n:, n:]
    sSS = S4c0[n:, n:]
    Gh = numpy.zeros((N, N), dtype=complex)
    GS = numpy.zeros((N, N), dtype=complex)

    # hfw1 = R0^+ L1 R0 + (R1^+ L0 R0 + h.c.)
    DL = R0 @ D @ R0.conj().T
    GR1 = 2 * L0 @ R0 @ D

    # L1 = hLL1 + (hLS X1 + h.c.) + (hLS1 X0 + h.c.) + (X1^+ hSS X0 + h.c.) + X0^+ hSS1 X0
    Gh[:n, :n] += DL
    Gh[:n, n:] += 2 * DL @ X0.conj().T
    Gh[n:, n:] += X0 @ DL @ X0.conj().T
    GX1 = 2 * hSL @ DL + 2 * hSS @ X0 @ DL

    # R1 (get_R1): A = Si ST Si; A1 = Si ST1 Si + Si1 ST Si + Si ST Si1;
    # R1 = Si Asqi1(A, A1) Sq + Si1 Ai Sq + Si Ai Sq1
    seig, svec = _eigh_pos(S2c0)
    Sq = (svec * numpy.sqrt(seig)) @ svec.conj().T
    Si = (svec / numpy.sqrt(seig)) @ svec.conj().T
    A = Si @ ST @ Si
    aeig, avec = _eigh_pos(A)
    Ai = (avec / numpy.sqrt(aeig)) @ avec.conj().T
    GZ = Si @ GR1 @ Sq
    GA1 = _invsqrt_deriv(aeig, avec, GZ)
    GST1 = Si @ GA1 @ Si
    GSi1 = GA1 @ Si @ ST + ST @ Si @ GA1 + GR1 @ Sq @ Ai
    GSq1 = Ai @ Si @ GR1
    GS[:n, :n] += _invsqrt_deriv(seig, svec, GSi1) + _sqrt_deriv(seig, svec, GSq1)

    # ST1 = S1LL + (X1^+ sSS X0 + h.c.) + X0^+ S1SS X0
    H = _herm(GST1)
    GS[:n, :n] += H
    GS[n:, n:] += X0 @ H @ X0.conj().T
    GX1 += 2 * sSS @ X0 @ H

    # X1 = [-X0, I] C0 U1[:, n:] P,  P = CL^+ (CL CL^+)^-1
    CL = C4c0[:n, n:]
    P = numpy.linalg.inv(CL)          # = CL^+ (CL CL^+)^-1, without squaring the condition number
    B = numpy.hstack((-X0, numpy.eye(n)))
    GU1p = C4c0.conj().T @ B.conj().T @ GX1 @ P.conj().T          # (N, n)

    # U1[i, j] = (D1mo - S1mo e_j)/(e_j - e_i), U1[j, j] = -S1mo[j, j]/2, j >= n
    e = mo_ene
    den = e[None, n:] - e[:, None]                                # (N, n)
    mask = abs(den) >= 1e-5
    offdiag = numpy.ones((N, n), dtype=bool)
    offdiag[numpy.arange(n, N), numpy.arange(n)] = False
    fac = numpy.where(mask & offdiag, 1 / numpy.where(mask, den, 1), 0)
    GD1mo = numpy.zeros((N, N), dtype=complex)
    GS1mo = numpy.zeros((N, N), dtype=complex)
    GD1mo[:, n:] = GU1p * fac
    GS1mo[:, n:] = -GU1p * fac * e[None, n:]
    GS1mo[numpy.arange(n, N), numpy.arange(n, N)] = -.5 * GU1p[numpy.arange(n, N), numpy.arange(n)]
    Gh += C4c0 @ GD1mo @ C4c0.conj().T
    GS += C4c0 @ GS1mo @ C4c0.conj().T
    return _herm(Gh), _herm(GS)


def x2c1e_hfw0(t, v, w, s):
    c = LIGHT_SPEED
    return x2c1e_hfw0_block(v, t, t, w * (.25 / c**2) - t, s, t * (.5 / c**2))

def x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS):
    c = LIGHT_SPEED
    nao = hLL.shape[0]
    n2 = nao * 2
    h4c = numpy.zeros((n2, n2), dtype=numpy.cdouble)
    m4c = numpy.zeros((n2, n2), dtype=numpy.cdouble)
    h4c[:nao, :nao] = hLL
    h4c[:nao, nao:] = hLS
    h4c[nao:, :nao] = hSL
    h4c[nao:, nao:] = hSS
    m4c[:nao, :nao] = sLL
    m4c[nao:, nao:] = sSS
    
    e, a = scipy.linalg.eigh(h4c, m4c)
    cl = a[:nao, nao:]
    cs = a[nao:, nao:]

    # b = numpy.dot(cl, cl.T.conj())
    # x = reduce(numpy.dot, (cs, cl.T.conj(), numpy.linalg.inv(b)))
    x = numpy.linalg.solve(cl.T, cs.T).T

    st = sLL + reduce(numpy.dot, (x.T.conj(), sSS, x))
    tx = reduce(numpy.dot, (hLS, x))
    l = h4c[:nao, :nao] + tx + tx.T.conj() + reduce(numpy.dot, (x.T.conj(), h4c[nao:, nao:], x))

    if HFW1_IMPL == 'sqrtm':
        sa = x2c._invsqrt(sLL)
        sb = x2c._invsqrt(reduce(numpy.dot, (sa, st, sa)))
        r = reduce(numpy.dot, (sa, sb, sa, sLL))
    else:
        r = x2c._get_r(sLL, st)
    # hfw = reduce(numpy.dot, (r.T.conj(), l, r))
    return a, e, x, st, r, l, h4c, m4c

def x2c1e_hfw1(mol, h4c1, s4c1=None, t=None, v=None, s=None, w=None):
    #assert (h4c1.shape[0] == 2*t.shape[0]), "The size of h4c1 does match the size of two-component integrals."
    if(t is None):
        t = mol.intor('int1e_spsp_spinor') * .5
    if(v is None):
        v = mol.intor('int1e_nuc_spinor')
    if(w is None):
        w = mol.intor('int1e_spnucsp_spinor')
    if(s is None):
        s = mol.intor('int1e_ovlp_spinor')
    a, e, x, st, r, l, h4c, m4c = x2c1e_hfw0(t, v, w, s)
    return get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)
