from functools import reduce
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.gto import moleintor
from pyscf.x2c import x2c
import numpy,math
import scipy.linalg
'''
Analytical energy gradients for x2c1e method

Ref.
JCP 135, 084114 (2011); DOI:10.1063/1.3624397
'''

def get_Asq1(A, A1):
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
            YB1Y[ii,jj] = YA1Y[ii,jj] / (math.sqrt(Aeig[ii]) + math.sqrt(Aeig[jj]))
    return reduce(numpy.dot, (Avec, YB1Y, Avec.T.conj()))

def get_Asqi1(A, A1):
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
    A = reduce(numpy.dot, (Ssqinv0, ST, Ssqinv0))
    Ssq0 = scipy.linalg.inv(Ssqinv0)
    # If S1 = 0, R1 is the middle term of (34)
    A1 = reduce(numpy.dot, (Ssqinv0, ST1, Ssqinv0))
    R1 = reduce(numpy.dot, (Ssqinv0, get_Asqi1(A, A1), Ssq0))
    if(S1 is not None):
        Asqinv = scipy.linalg.inv(scipy.linalg.sqrtm(A))
        R1 = R1 + reduce(numpy.dot, (get_Asqi1(S0,S1), Asqinv, Ssq0)) + reduce(numpy.dot, (Ssqinv0, Asqinv, get_Asq1(S0,S1)))
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


def get_hfw1(C4c0, X0, S4c0, h4c0, mo_ene, R0, L0, h4c1, S4c1=None):
    size2c = C4c0.shape[0]//2
    S2c0 = S4c0[:size2c,:size2c]
    ST = S2c0 + reduce(numpy.dot, (X0.T.conj(), S4c0[size2c:,size2c:], X0))
    if(S4c1 is None):
        S2c1 = None
    else:
        S2c1 = S4c1[:size2c,:size2c]
    C4c1 = get_C1(C4c0, mo_ene, h4c1, S4c1)
    X1 = get_X1(C4c0, C4c1, X0)
    ST1 = get_ST1(S4c0, X0, X1, S4c1)
    R1 = get_R1(S2c0, ST, ST1, S2c1)
    L1 = get_L1(h4c0, h4c1, X0, X1)
    R1LR = reduce(numpy.dot, (R1.T.conj(), L0, R0))
    return reduce(numpy.dot, (R0.T.conj(), L1, R0)) + R1LR + R1LR.T.conj()



def x2c1e_hfw1(t, v, w, s, h4c1):
    assert (h4c1.shape[0] == 2*t.shape[0]), "The size of h4c1 does match the size of two-component integrals."

    c = LIGHT_SPEED
    nao = s.shape[0]
    n2 = nao * 2
    h4c = numpy.zeros((n2, n2), dtype=v.dtype)
    m4c = numpy.zeros((n2, n2), dtype=v.dtype)
    h4c[:nao, :nao] = v
    h4c[:nao, nao:] = t
    h4c[nao:, :nao] = t
    h4c[nao:, nao:] = w * (.25 / c**2) - t
    m4c[:nao, :nao] = s
    m4c[nao:, nao:] = t * (.5 / c**2)

    e, a = scipy.linalg.eigh(h4c, m4c)
    cl = a[:nao, nao:]
    cs = a[nao:, nao:]

    b = numpy.dot(cl, cl.T.conj())
    x = reduce(numpy.dot, (cs, cl.T.conj(), numpy.linalg.inv(b)))

    st = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5 / c**2)
    tx = reduce(numpy.dot, (t, x))
    l = h4c[:nao, :nao] + tx + tx.T.conj() + reduce(numpy.dot, (x.T.conj(), h4c[nao:, nao:], x))

    sa = x2c._invsqrt(s)
    sb = x2c._invsqrt(reduce(numpy.dot, (sa, st, sa)))
    r = reduce(numpy.dot, (sa, sb, sa, s))
    # hfw = reduce(numpy.dot, (r.T.conj(), l, r))
    return get_hfw1(a, x, m4c, h4c, e, r, l, h4c1)