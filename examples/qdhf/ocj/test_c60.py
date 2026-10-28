import sys, time, numpy, scipy.linalg
sys.argv = ['x', 'c60']
src = open('/home/user/socutils/examples/qdhf/40-analytic_onecenter_j.py').read().split("\nlarge = len(sys.argv)")[0]
g = {'__name__': 'x'}; exec(src, g)
from ocj import j_kernel
from pyscf import gto
from pyscf.lib import param
c = param.LIGHT_SPEED
name, geom, base = g['systems'][0]
mol = gto.M(atom=geom, basis={'C': gto.mole.uncontracted_basis(gto.basis.load('cc-pvdz', 'C'))}, verbose=0)
xmol, contr, d4, d2 = g['atomic_densities'](mol)
n2c = xmol.nao_2c()
t = time.perf_counter(); ae = g['element_expansions'](xmol, d4, d2); print('expansion %.2f s' % (time.perf_counter() - t), flush=True)
terms = [(g['potential_terms'](e4), g['potential_terms'](e2)) for e4, e2 in ae]
t = time.perf_counter(); vnll = xmol.intor('int1e_nuc_spinor'); vnss = xmol.intor('int1e_spnucsp_spinor'); tvn = time.perf_counter() - t
res = {}
for eps in (1e-20,):
    tt = []; t = time.perf_counter(); l4, s4, l2 = j_kernel(xmol, terms, eps, tt); tk = time.perf_counter() - t
    res[eps] = (l4, s4, l2)
    npair = xmol.nbas * (xmol.nbas + 1) // 2
    print('kernel eps %.0e  %.2f s (3c %.2f, rinv %.2f, spinor %.2f)  (ij,k) triples kept per k-entry: %.1f%% of pairs' % (
        eps, tk, tt[0], tt[1], tt[2], 100 * tt[3] / npair / 1), flush=True)
t = time.perf_counter(); j4p, j2p = g['j_analytic_exact'](xmol, ae); tp = time.perf_counter() - t
j4k = scipy.linalg.block_diag(res[1e-20][0] - vnll, (res[1e-20][1] - vnss) * (.25 / c**2)); j2k = res[1e-20][2] - vnll
print('python unscreened analytic %.2f s; V_nuc (both blocks) %.2f s' % (tp, tvn))
print('kernel 1e-20 vs python: LL %.1e SS %.1e 2c %.1e' % (abs(j4k - j4p)[:n2c, :n2c].max(), abs(j4k - j4p)[n2c:, n2c:].max(), abs(j2k - j2p).max()))
for eps in ():
    print('kernel %.0e vs 1e-20: LL %.1e SS %.1e 2c %.1e' % ((eps,) + tuple(abs(a - b).max() for a, b in zip(res[eps], res[1e-20]))))
