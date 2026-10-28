import sys, time, numpy, scipy.linalg
sys.argv = ['x', sys.argv[1] if len(sys.argv) > 1 else 'small']
sys.path.insert(0, '/home/user/socutils/examples/qdhf/ocj')
src = open('/home/user/socutils/examples/qdhf/40-analytic_onecenter_j.py').read().split("\nlarge = len(sys.argv)")[0]
g = {'__name__': 'x'}; exec(src, g)
from ocj import j_kernel
from pyscf import gto
from pyscf.lib import param
c = param.LIGHT_SPEED
EPS = float(__import__('os').environ.get('EPS', '1e-13'))
STD = __import__('os').environ.get('STD', '0') == '1'
for name, geom, base in g['systems']:
    mol = gto.M(atom=geom, basis=base[4:], verbose=0)
    mol = gto.M(atom=geom, basis={e: gto.mole.uncontracted_basis(gto.basis.load(base[4:], e)) for e in set(a[0] for a in mol._atom)}, verbose=0)
    xmol, contr, d4, d2 = g['atomic_densities'](mol)
    n2c = xmol.nao_2c()
    ae = g['element_expansions'](xmol, d4, d2)
    t = time.perf_counter(); j4p, j2p = g['j_neutral_atom'](xmol, ae, 1e-13); tp = time.perf_counter() - t
    terms = [(g['potential_terms'](e4), g['potential_terms'](e2)) for e4, e2 in ae]
    tt = []; t = time.perf_counter(); vll4, vss4, vll2 = j_kernel(xmol, terms, EPS, tt, std=STD); tk = time.perf_counter() - t
    vnll = xmol.intor('int1e_nuc_spinor'); vnss = xmol.intor('int1e_spnucsp_spinor')
    j4k = scipy.linalg.block_diag(vll4 - vnll, (vss4 - vnss) * (.25 / c**2)); j2k = vll2 - vnll
    print('%-7s n2c %4d  python %.2f s   kernel %.2f s (3c %.2f, rinv %.2f, spinor %.2f)  |dJ| LL %.1e SS %.1e 2c %.1e' % (
        name, n2c, tp, tk, tt[0], tt[1], tt[2], abs(j4k - j4p)[:n2c, :n2c].max(), abs(j4k - j4p)[n2c:, n2c:].max(), abs(j2k - j2p).max()), flush=True)
