#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
#

'''
Contact density for relativistic 2-component JHF methods.
(In testing)
'''

import warnings
from functools import reduce
import numpy as np
from pyscf import lib

warnings.warn('Module contact density is under testing')


def kernel(method, cd_nuc=None, dm=None, Xresp=False):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** Contact density for 2-component SCF methods (In testing) ********')
    if Xresp:
        log.info('Include the response of X2C transformation')
    else:
        log.info('Ignore the response of X2C transformation')

    xmol, contr_coeff_nr = method.with_x2c.get_xmol(method.mol)
    npri, ncon = contr_coeff_nr.shape
    contr_coeff = np.zeros((npri*2,ncon*2))
    contr_coeff[0::2,0::2] = contr_coeff_nr
    contr_coeff[1::2,1::2] = contr_coeff_nr

    c = lib.param.LIGHT_SPEED
    n2c = xmol.nao_2c()
    if cd_nuc is None:
        cd_nuc = range(xmol.natm)
    if dm is None:
        dm = method.make_rdm1()

    coords = []
    log.info('\nContact Density Results')
    for atm_id in cd_nuc:
        coords.append(xmol.atom_coord(atm_id))
    
    aoLa, aoLb = xmol.eval_gto('GTOval_spinor', coords)
    aoSa, aoSb = xmol.eval_gto('GTOval_sp_spinor', coords)

    t = xmol.intor('int1e_kin_spinor')
    s = xmol.intor('int1e_ovlp_spinor')
    v = xmol.intor('int1e_nuc_spinor')
    w = xmol.intor('int1e_spnucsp_spinor')
    from socutils.somf import x2c_grad
    a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0(t, v, w, s)

    cont_den = []
    for atm_id in range(len(coords)):
        int_4c = np.zeros((n2c*2, n2c*2), dtype=dm.dtype)
        int_4c[:n2c,:n2c] = np.einsum('p,q->pq', aoLa[atm_id].conj(), aoLa[atm_id])
        int_4c[:n2c,:n2c]+= np.einsum('p,q->pq', aoLb[atm_id].conj(), aoLb[atm_id])
        int_4c[n2c:,n2c:] = np.einsum('p,q->pq', aoSa[atm_id].conj(), aoSa[atm_id]) / 4.0 / c**2
        int_4c[n2c:,n2c:]+= np.einsum('p,q->pq', aoSb[atm_id].conj(), aoSb[atm_id]) / 4.0 / c**2
        if Xresp:
            int_2c = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, int_4c)
        else:
            from socutils.somf.eamf import to_2c
            int_2c = to_2c(x, r, int_4c)

        int_2c = reduce(np.dot, (contr_coeff.T.conj(), int_2c, contr_coeff))
        cont_den.append(np.einsum('ij,ji->', int_2c, dm))
        if cont_den[-1].imag > 1e-10:
            log.warn('Significant imaginary part found in contact density')
        log.info('\nAtom %d' % atm_id)
        log.info('Contact Density: %f' % cont_den[-1].real)

    return cont_den

ContDen = kernel

from socutils.scf import spinor_hf
spinor_hf.JHF.ContDen = lib.class_as_method(ContDen)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = '''
Au 0.0 0.0 0.0
'''
    mol.unit = 'Bohr'
    mol.spin = 0
    mol.charge = 1
    mol.nucmod = "g"
    mol.basis = {'Au': gto.basis.parse('''
Au   S
62072218.0   0.00053207 -0.00020848  0.00009996  0.00004994 -0.00002154 -0.00000589  0.00000000  0.00000000
16520359.0   0.00124369 -0.00048775  0.00023390  0.00011687 -0.00005041 -0.00001377  0.00000000  0.00000000
5657915.40   0.00213195 -0.00083723  0.00040160  0.00020067 -0.00008656 -0.00002365  0.00000000  0.00000000
2156230.30   0.00362252 -0.00142609  0.00068433  0.00034198 -0.00014753 -0.00004031  0.00000000  0.00000000
899306.480   0.00579819 -0.00229135  0.00110028  0.00054991 -0.00023721 -0.00006482  0.00000000  0.00000000
396414.960   0.00941714 -0.00374286  0.00179895  0.00089937 -0.00038804 -0.00010602  0.00000000  0.00000000
183059.020   0.01508177 -0.00604524  0.00290986  0.00145511 -0.00062770 -0.00017152  0.00000000  0.00000000
87361.6210   0.02450654 -0.00993978  0.00479354  0.00239862 -0.00103529 -0.00028288  0.00000000  0.00000000
42855.5450   0.03981314 -0.01642592  0.00794541  0.00397756 -0.00171591 -0.00046889  0.00000000  0.00000000
21486.5000   0.06481539 -0.02739475  0.01330320  0.00666874 -0.00288046 -0.00078713  0.00000000  0.00000000
10981.3560   0.10362804 -0.04543220  0.02220827  0.01114466 -0.00480873 -0.00131403  0.00000000  0.00000000
5707.96180   0.15862839 -0.07355236  0.03630090  0.01827326 -0.00790499 -0.00216089  0.00000000  0.00000000
3013.43250   0.22039434 -0.11172557  0.05607878  0.02831741 -0.01222719 -0.00334109  0.00000000  0.00000000
1614.04250   0.25390427 -0.14607423  0.07510095  0.03821278 -0.01659850 -0.00454171  0.00000000  0.00000000
876.047500   0.20731856 -0.13003492  0.06831466  0.03483010 -0.01500613 -0.00409537  0.00000000  0.00000000
481.220080   0.09295331  0.01788911 -0.02127273 -0.01231480  0.00530109  0.00143784  0.00000000  0.00000000
265.695650   0.01481640  0.31151509 -0.25125961 -0.14344583  0.06433582  0.01771686  0.00000000  0.00000000
149.025870   0.00057146  0.47617915 -0.49045228 -0.29611979  0.13285338  0.03648194  0.00000000  0.00000000
84.1074450  -0.00006155  0.28594955 -0.31122471 -0.20415764  0.09674200  0.02696818  0.00000000  0.00000000
46.5028220   0.00001866  0.05162061  0.42597978  0.40411951 -0.20906755 -0.05915711  0.00000000  0.00000000
27.0322630  -0.00006763  0.00014218  0.70229077  0.90480710 -0.48531176 -0.13687602  0.00000000  0.00000000
15.4290090   0.00003813  0.00067136  0.26972723  0.19506408 -0.09577753 -0.02824825  0.00000000  0.00000000
8.33768990  -0.00001719 -0.00045024  0.02362522 -0.83268359  0.67431901  0.20733251  0.00000000  0.00000000
4.60569280   0.00001303  0.00012216  0.00090871 -0.62145877  0.78091543  0.26008979  0.00000000  0.00000000
2.32465200  -0.00000666 -0.00006465  0.00002558 -0.07681777 -0.31189439 -0.13965214  0.00000000  0.00000000
1.23675190   0.00000314  0.00004019 -0.00010227  0.00193232 -0.84088875 -0.39848678  0.00000000  0.00000000
0.61137785  -0.00000125 -0.00001135 -0.00001736 -0.00138806 -0.34595248 -0.24143269  0.00000000  0.00000000
0.21209493   0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  1.00000000  0.00000000
0.09190149  -0.00000020 -0.00000208  0.00000105 -0.00009703  0.00217307  0.63102308  0.00000000  0.00000000
0.03929969   0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  1.00000000
Au   P
26507286.0  -0.00001020  0.00000526  0.00000261  0.00000104 -0.00000008  0.00000000  0.00000000
5441469.20  -0.00003600  0.00001858  0.00000923  0.00000367 -0.00000030  0.00000000  0.00000000
1355031.00  -0.00010603  0.00005478  0.00002722  0.00001083 -0.00000088  0.00000000  0.00000000
382714.630  -0.00029700  0.00015370  0.00007639  0.00003039 -0.00000248  0.00000000  0.00000000
119172.560  -0.00082580  0.00042844  0.00021308  0.00008478 -0.00000687  0.00000000  0.00000000
40323.5180  -0.00231939  0.00120791  0.00060127  0.00023923 -0.00001961  0.00000000  0.00000000
14744.4090  -0.00660764  0.00346230  0.00172591  0.00068712 -0.00005548  0.00000000  0.00000000
5810.92180  -0.01873871  0.00991994  0.00495759  0.00197407 -0.00016249  0.00000000  0.00000000
2456.06150  -0.05116496  0.02757432  0.01383715  0.00551749 -0.00044377  0.00000000  0.00000000
1103.73340  -0.12596920  0.07004838  0.03543074  0.01414454 -0.00117011  0.00000000  0.00000000
521.538260  -0.25448743  0.14815404  0.07578646  0.03035589 -0.00242245  0.00000000  0.00000000
256.384820  -0.36165477  0.21930210  0.11351628  0.04552711 -0.00385177  0.00000000  0.00000000
130.289570  -0.28395175  0.11055561  0.04538261  0.01718372 -0.00103477  0.00000000  0.00000000
67.5225360  -0.08965972 -0.27720765 -0.21714182 -0.09533477  0.00699928  0.00000000  0.00000000
36.0892670  -0.00572938 -0.54034709 -0.45359962 -0.20196433  0.01849198  0.00000000  0.00000000
19.3652210  -0.00071839 -0.28760116 -0.07729149 -0.01124489 -0.00278731  0.00000000  0.00000000
10.1043050   0.00044690 -0.04045248  0.57006579  0.35399935 -0.02399973  0.00000000  0.00000000
5.25279110  -0.00015137 -0.00122354  0.53047836  0.37615631 -0.04367379  0.00000000  0.00000000
2.56950900   0.00008833 -0.00013216  0.10314364 -0.25532318  0.04674024  0.00000000  0.00000000
1.26038790  -0.00004699  0.00007664  0.00011525 -0.63366560  0.03594719  0.00000000  0.00000000
0.57610947   0.00001581  0.00001179  0.00137968 -0.33429507  0.08443646  0.00000000  0.00000000
0.19786585  -0.00000615  0.00000399 -0.00030416 -0.03498836 -0.17574375  0.00000000  0.00000000
0.07551192   0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  1.00000000  0.00000000
0.02821419   0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  1.00000000
Au   D
14478.1020   -0.00017767  0.00009208 -0.00002714  0.00000000  0.00000000  0.00000000
3677.16020   -0.00138285  0.00071642 -0.00021025  0.00000000  0.00000000  0.00000000
1291.06680   -0.00776157  0.00405863 -0.00119854  0.00000000  0.00000000  0.00000000
535.609420   -0.03322231  0.01747817 -0.00514372  0.00000000  0.00000000  0.00000000
245.192760   -0.10715911  0.05759270 -0.01709340  0.00000000  0.00000000  0.00000000
119.794850   -0.24797268  0.13424764 -0.03966969  0.00000000  0.00000000  0.00000000
60.7825790   -0.37750981  0.19731868 -0.05847280  0.00000000  0.00000000  0.00000000
31.6631230   -0.31716133  0.08421762 -0.01782774  0.00000000  0.00000000  0.00000000
16.4540820   -0.11442919 -0.26783907  0.10048684  0.00000000  0.00000000  0.00000000
8.38347560   -0.01202610 -0.49908915  0.18510188  0.00000000  0.00000000  0.00000000
4.20264260   -0.00085077 -0.32133067  0.05944718  0.00000000  0.00000000  0.00000000
1.96488350    0.00004763 -0.06160572 -0.27235803  0.00000000  0.00000000  0.00000000
0.88234297    0.00000000  0.00000000  0.00000000  1.00000000  0.00000000  0.00000000
0.36673205    0.00000000  0.00000000  0.00000000  0.00000000  1.00000000  0.00000000
0.13681075    0.00000000  0.00000000  0.00000000  0.00000000  0.00000000  1.00000000
Au   F
804.603730   0.00053587 0.00000000 0.00000000 0.00000000
274.042220   0.00483662 0.00000000 0.00000000 0.00000000
116.667490   0.02526469 0.00000000 0.00000000 0.00000000
55.0923690   0.08435996 0.00000000 0.00000000 0.00000000
27.4952890   0.19722577 0.00000000 0.00000000 0.00000000
14.0612580   0.31151020 0.00000000 0.00000000 0.00000000
7.14450480   0.34677203 0.00000000 0.00000000 0.00000000
3.52275990   0.25684445 0.00000000 0.00000000 0.00000000
1.59267640   0.00000000 1.00000000 0.00000000 0.00000000
0.53758925   0.00000000 0.00000000 1.00000000 0.00000000
0.18145695   0.00000000 0.00000000 0.00000000 1.00000000
Au   G
1.20070770   1.0 0.0
0.40528481   0.0 1.0
''')}
    mol.basis = {'Au':gto.uncontracted_basis(mol.basis['Au'])} 
    mol.build()
    from socutils.scf import x2camf_hf
    mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
    mf.conv_tol = 1e-8
    mf.kernel()
    mf.ContDen()