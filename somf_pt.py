import numpy, scipy, x2c_grad
import time
from functools import reduce
from pyscf import gto, scf, lib
from pyscf.gto import moleintor
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e
from spinor2sph import spinor2sph_soc
x2camf  = None
try:
    import x2camf
except ImportError:
    pass
'''
Perturbative 

Ref.
JCP 135, 084114 (2011); DOI:10.1063/1.3624397
'''

def x2camf_pt(mol, unc = True):
    if unc:
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
    else:
        xmol, contr_coeff = mol, numpy.eye(mol.nao_nr())
        
    c = LIGHT_SPEED
    t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
    v = xmol.intor_symmetric('int1e_nuc_spinor')
    s = xmol.intor_symmetric('int1e_ovlp_spinor')
    w = xmol.intor_symmetric('int1e_spnucsp_spinor')
    wsf = xmol.intor_symmetric('int1e_pnucp_spinor')
    wso = w - wsf

    n2c = s.shape[0]
    n4c = n2c * 2
    h4c0 = numpy.zeros((n4c, n4c), dtype=v.dtype)
    h4c1 = numpy.zeros((n4c, n4c), dtype=v.dtype)
    s4c0 = numpy.zeros((n4c, n4c), dtype=v.dtype)
    h4c0[:n2c, :n2c] = v
    h4c0[:n2c, n2c:] = t
    h4c0[n2c:, :n2c] = t
    h4c0[n2c:, n2c:] = wsf * (.25 / c**2) - t
    # for ii in range(h4c0.shape[0]):
    #     for jj in range(h4c0.shape[1]):
    #         print(h4c0[ii,jj])

    s4c0[:n2c, :n2c] = s
    s4c0[n2c:, n2c:] = t * (.5 / c**2)

    h4c1[n2c:, n2c:] = wso * (.25 / c**2)
    # Get x2camf 4c integrals and add to h4c1
    x2cobj = x2c.X2C(mol)
    spinor = x2camf.amfi(x2cobj, spin_free=True, two_c=True, with_gaunt=True, with_gauge=True)
    h4c1 += spinor
    

    ene_4c, c4c0 = scipy.linalg.eigh(h4c0, s4c0)
    cl = c4c0[:n2c, n2c:]
    cs = c4c0[n2c:, n2c:]

    b = numpy.dot(cl, cl.T.conj())
    x = reduce(numpy.dot, (cs, cl.T.conj(), numpy.linalg.inv(b)))

    st = s + reduce(numpy.dot, (x.T.conj(), t, x)) * (.5 / c**2)
    tx = reduce(numpy.dot, (t, x))
    l0 = (h4c0[:n2c, :n2c] + tx + tx.T.conj() + reduce(numpy.dot, (x.T.conj(), h4c0[n2c:, n2c:], x)))

    ssqinv0 = x2c._invsqrt(s)
    sb = x2c._invsqrt(reduce(numpy.dot, (ssqinv0, st, ssqinv0)))
    r0 = reduce(numpy.dot, (ssqinv0, sb, ssqinv0, s))
    hfw1 = x2c_grad.get_hfw1(c4c0, x, s4c0, h4c0, ene_4c, r0, l0, h4c1)
    hfw1_sph = spinor2sph_soc(xmol, hfw1)[1:]
    # convert back to contracted basis
    result = numpy.zeros((3, mol.nao_nr(), mol.nao_nr()))
    for ic in range(3):
        result[ic] = reduce(numpy.dot, (contr_coeff.T, hfw1_sph[ic], contr_coeff))
    
    return result
    