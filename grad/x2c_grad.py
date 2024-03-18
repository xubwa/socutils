from functools import reduce
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.x2c import x2c
import numpy,math
import scipy.linalg
from socutils.grad.x2c_grad_g import _block_diag_xyz
from pyscf.x2c.sfx2c1e_grad import _gen_h1_s1

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
        hfw1 = x2c1e_hfw1(xmol, h1, s1)
        if contr_coeff_nr is not None:
            contr_coeff = x2c._block_diag(contr_coeff_nr)
            h1 = lib.einsum('ai,pi,xpq,qj,bj->xab', c2, contr_coeff, h1, contr_coeff, c2.conj())
        return numpy.asarray(h1)
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
#x2c.SpinOrbitalX2CHelper.hcore_deriv_generator = hcore_deriv_generator_spin_orbital # type: ignore

def get_h1nuc_s1(mol):
    c = LIGHT_SPEED
    s1 = mol.intor('int1e_ipovlp_spinor', comp=3)
    t1 = mol.intor('int1e_ipkin_spinor', comp=3)
    v1 = mol.intor('int1e_ipnuc_spinor', comp=3)
    w1 = mol.intor('int1e_ipspnucsp_spinor', comp=3)
    aoslices = mol.aoslice_2c_by_atom()
    n2c = s1.shape[1]
    n4c = n2c * 2

    def get_h1_s1(ia):
        h1 = numpy.zeros((3,n4c,n4c), dtype=complex)
        m1 = numpy.zeros((3,n4c,n4c), dtype=complex)
        ish0, ish1, i0, i1 = aoslices[ia]
        with mol.with_rinv_origin(mol.atom_coord(ia)):
            z = mol.atom_charge(ia)
            rinv1 = -z*mol.intor('int1e_iprinv_spinor', comp=3)
            sprinvsp1 = -z*mol.intor('int1e_ipsprinvsp_spinor', comp=3)
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

def get_Asq1(A, A1):
    # A = B^2
    # B = A^{1/2}
    # return B^lambda
    # Ref (A3)-(A7)
    size = A.shape[0]
    assert(scipy.linalg.ishermitian(A,atol=1e-10))
    Aeig, Avec = scipy.linalg.eigh(A)
    YA1Y = reduce(numpy.dot, (Avec.T.conj(), A1, Avec))
    YB1Y = numpy.zeros((size,size), dtype=YA1Y.dtype)
    for ii in range(size):
        for jj in range(size):
            YB1Y[ii,jj] = YA1Y[ii,jj] / (math.sqrt(Aeig[ii]) + math.sqrt(Aeig[jj]))
    return reduce(numpy.dot, (Avec, YB1Y, Avec.T.conj()))

def get_Asqi1(A, A1):
    # A = B^2
    # B = A^{1/2}
    # return (B^{-1})^lambda
    # Ref (A3)-(A7)
    size = A.shape[0]
    assert(scipy.linalg.ishermitian(A,atol=1e-10))
    Aeig, Avec = scipy.linalg.eigh(A)
    YA1Y = reduce(numpy.dot, (Avec.T.conj(), A1, Avec))
    YB1Y = numpy.zeros((size,size), dtype=YA1Y.dtype)
    Asqieig = numpy.zeros((size,size))
    for ii in range(size):
        if(Aeig[ii] > 0.0):
            Asqieig[ii,ii] = 1.0/math.sqrt(Aeig[ii])
        for jj in range(size):
            YB1Y[ii,jj] = YA1Y[ii,jj] / (math.sqrt(Aeig[ii]) + math.sqrt(Aeig[jj]))
    B1 = reduce(numpy.dot, (Avec, YB1Y, Avec.T.conj()))
    Binv = reduce(numpy.dot, (Avec, Asqieig, Avec.T.conj()))
    # Ref (A1)
    return -1.0 * reduce(numpy.dot, (Binv, B1, Binv))

def get_C1(C0, mo_ene, D1, S4c1=None):
    # Ref (32)-(33)
    D1_mo = reduce(numpy.dot,(C0.T.conj(), D1, C0))
    size = D1_mo.shape[0]
    U1 = numpy.zeros((size,size), dtype=D1_mo.dtype)
    if(S4c1 is None):
        S1_mo = numpy.zeros((size,size))
    else:
        S1_mo = reduce(numpy.dot,(C0.T.conj(), S4c1, C0))
    for ii in range(size):
        for jj in range(size):
            if(ii == jj):
                U1[ii,ii] = -0.5*S1_mo[ii,ii]
            else:
                denominator = mo_ene[jj]-mo_ene[ii]
                if(abs(denominator) < 1e-4):
                    U1[ii,jj] = 0.0
                else:
                    U1[ii,jj] = (D1_mo[ii,jj] - S1_mo[ii,jj]*mo_ene[jj])/denominator
    return numpy.dot(C0,U1)

def get_X1(C0, C1, X0):
    # Ref (31)
    size = C0.shape[0]//2
    CL = C0[:size, size:]
    CL1 = C1[:size, size:]
    CS1 = C1[size:, size:]
    CS1 = CS1 - numpy.dot(X0,CL1)
    # tmp = numpy.dot(CL, CL.T.conj())
    # return reduce(numpy.dot, (CS1, CL.T.conj(), numpy.linalg.inv(tmp)))   # numerically a little bit more stable
    return numpy.dot(CS1,scipy.linalg.inv(CL))

def get_ST1(S4c0, X0, X1=None, S4c1=None):
    size = S4c0.shape[0]//2
    if(S4c1 is None):
        ST1 = numpy.zeros((size,size))
    else:
        ST1 = S4c1[:size, :size]
    if(X1 is not None):
        X1TX = reduce(numpy.dot, (X1.T.conj(), S4c0[size:, size:], X0))
        ST1 = ST1 + X1TX + X1TX.T.conj()
    if(S4c1 is not None):
        ST1 = ST1 + reduce(numpy.dot, (X0.T.conj(), S4c1[size:, size:], X0))
    return ST1

def get_R1(S0, ST, ST1, S1=None):
    # Ref (34)
    # A = S^{-1/2}*ST*S^{-1/2}
    Ssqinv0 = scipy.linalg.inv(scipy.linalg.sqrtm(S0))
    print('ST hermicity', numpy.allclose(ST, ST.T.conj()))
    A = reduce(numpy.dot, (Ssqinv0, ST, Ssqinv0))
    print('A hermicity', scipy.linalg.ishermitian(A,atol=1e-10), numpy.allclose(A, A.T.conj()))
    Ssq0 = scipy.linalg.inv(Ssqinv0)
    if S1 is None:
        Ssqinv1 = numpy.zeros((S0.shape[0],S0.shape[0]))
        Ssq1 = numpy.zeros((S0.shape[0],S0.shape[0]))
    else:
        Ssqinv1 = get_Asqi1(S0, S1)
        Ssq1 = get_Asq1(S0, S1)
    # If S1 = 0, R1 is the middle term of (34)
    A1 = reduce(numpy.dot, (Ssqinv0, ST1, Ssqinv0))\
       + reduce(numpy.dot, (Ssqinv1, ST, Ssqinv0))\
       + reduce(numpy.dot, (Ssqinv0, ST, Ssqinv1))

    R1 = reduce(numpy.dot, (Ssqinv0, get_Asqi1(A, A1), Ssq0))
    if(S1 is not None):
        Asqinv = scipy.linalg.inv(scipy.linalg.sqrtm(A))
        R1 = R1 + reduce(numpy.dot, (Ssqinv1, Asqinv, Ssq0))\
                + reduce(numpy.dot, (Ssqinv0, Asqinv, Ssq1))
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
    C4c1 = get_C1(C4c0, mo_ene, h4c1, S4c1)
    X1 = get_X1(C4c0, C4c1, X0)
    ST1 = get_ST1(S4c0, X0, X1, S4c1)
    R1 = get_R1(S2c0, ST, ST1, S2c1)
    L1 = get_L1(h4c0, h4c1, X0, X1)
    print('r1 norm', numpy.linalg.norm(R1.imag))
    print('r1 hermicity', numpy.allclose(R1, R1.T.conj()))
    print('l1 norm', numpy.linalg.norm(L1.imag))
    print('l1 hermicity', numpy.allclose(L1, L1.T.conj()))
    R1LR = reduce(numpy.dot, (R1.T.conj(), L0, R0))
    RLR1 = reduce(numpy.dot, (R0.T.conj(), L0, R1))
    RL1R = reduce(numpy.dot, (R0.T.conj(), L1, R0))
    hfw1 = RL1R + R1LR + RLR1
    print('hfw1 hermicity', numpy.allclose(hfw1, hfw1.T.conj()))
    return hfw1

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

    b = numpy.dot(cl, cl.T.conj())
    x = reduce(numpy.dot, (cs, cl.T.conj(), numpy.linalg.inv(b)))

    st = sLL + reduce(numpy.dot, (x.T.conj(), sSS, x))
    tx = reduce(numpy.dot, (hLS, x))
    l = h4c[:nao, :nao] + tx + tx.T.conj() + reduce(numpy.dot, (x.T.conj(), h4c[nao:, nao:], x))

    sa = x2c._invsqrt(sLL)
    sb = x2c._invsqrt(reduce(numpy.dot, (sa, st, sa)))
    r = reduce(numpy.dot, (sa, sb, sa, sLL))
    # hfw = reduce(numpy.dot, (r.T.conj(), l, r))
    return a, e, x, st, r, l, h4c, m4c

def x2c1e_hfw1(mol, h4c1, s4c1=None, t=None, v=None, s=None, w=None):
    #assert (h4c1.shape[0] == 2*t.shape[0]), "The size of h4c1 does match the size of two-component integrals."
    print('general x2c gradient driver')
    if(t is None):
        t = mol.intor('int1e_spsp_spinor') * .5
    if(v is None):
        v = mol.intor('int1e_rinv_spinor')
    if(w is None):
        w = mol.intor('int1e_spnucsp_spinor')
    if(s is None):
        s = mol.intor('int1e_ovlp_spinor')
    a, e, x, st, r, l, h4c, m4c = x2c1e_hfw0(t, v, w, s)
    return get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)
