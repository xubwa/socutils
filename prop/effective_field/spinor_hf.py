#

'''
Contact density for relativistic 2-component JHF methods.
(In testing)
'''

import warnings
from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.x2c.x2c import _decontract_spinor

warnings.warn('Module contact density is under testing')

Prop = 'Effective electric field'
prop = 'effective electric field'
def kernel(method, dm=None, Xresp=True):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info(f'\n******** {Prop} for 2-component SCF methods (In testing) ********')
    if Xresp:
        log.info('Include the response of X2C transformation')
    else:
        log.info('Ignore the response of X2C transformation')

    xmol, contr_coeff = _decontract_spinor(method.mol, method.with_x2c.xuncontract)

    c = lib.param.LIGHT_SPEED
    n2c = xmol.nao_2c()
    
    if dm is None:
        dm = method.make_rdm1()

    log.info(f'\n{Prop} results')
    
    int1e = xmol.intor('int1e_spspsp_spinor')
    int_4c = np.zeros((n2c*2, n2c*2), dtype=dm.dtype)
    int_4c[:n2c,n2c:] = -1.j * int1e
    int_4c[n2c:,:n2c] = 1.j * int1e

    t = xmol.intor('int1e_kin_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_spnucsp_spinor')
    from socutils.somf import x2c_grad
    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)

    if Xresp:
        int_2c = method.with_x2c.get_hfw1(int_4c)
        #int_2c = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, int_4c)
    else:
        from socutils.somf.eamf import to_2c
        int_2c = to_2c(x, r, int_4c)

    int_2c = reduce(np.dot, (contr_coeff.T.conj(),int_2c, contr_coeff))
    e_eff = np.einsum('ij,ji->', int_2c, dm)
    if e_eff.imag > 1e-10:
        log.warn('Significant imaginary part found in effective electric field')
    log.info(f'Effective electric field: {e_eff}')

    return e_eff

EffectiveField = kernel

from socutils.scf import spinor_hf
spinor_hf.SpinorSCF.EffectiveField = lib.class_as_method(EffectiveField)

if __name__ == '__main__':
    from pyscf import gto
    from socutils.somf import amf, eamf

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = '''H 0.0 0.0 1.973983529195167\nF 0.0 0.0 -0.104715643539254'''
    mol.basis = 'dyallv3z'
    #mol.nucmod='G'
    mol.charge = 1
    mol.unit='B'
    mol.spin = 1
    mol.build()

    mf = spinor_hf.SymmJHF(mol, symmetry='linear', occup={'1/2':[4],'-1/2':[4],'3/2':[1]})
    #mf.with_x2c = amf.SpinorX2CAMFHelper(mol,with_gaunt=False,with_breit=False,with_aoc=False)
    mf.with_x2c = eamf.SpinorEAMFX2CHelper(mol, eamf='x2camf', with_gaunt=False, with_breit=False, with_aoc=True)
    #mf.with_x2c = eamf.SpinorEAMFX2CHelper(mol, eamf='x2camf', with_gaunt=False, with_breit=False, with_aoc=True)
    mf.kernel()
    mf.EffectiveField(Xresp=False)
