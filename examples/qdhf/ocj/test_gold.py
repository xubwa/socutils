'''Au4 (tetrahedron) and Au8 (cube), uncontracted dyall-v2z: the kernel J against one scalar
NR J / JK build and the X2C decoupling; accuracy from the exact 2c J (one scalar J build).'''
import sys, os, time, numpy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.argv = ['x', 'gold']
src = open(os.path.join(here, '../40-analytic_onecenter_j.py')).read().split("\nlarge = len(sys.argv)")[0]
g = {'__name__': 'x'}; exec(src, g)
from ocj import j_kernel
from pyscf import gto, scf
from socutils.somf import x2cmp as X
print('%-4s %5s | %7s %7s %7s | %7s %7s %7s %7s %7s | %7s | %8s' % (
    'mol', 'n2c', 'NR J', 'NR JK', 'X2C', 'expand', '3c', 'rinv', 'spinor', 'total', '/NR JK', 'err 2c'))
only = os.environ.get('ONLY')
for name, geom, base in g['systems']:
    if only and name != only:
        continue
    mol = gto.M(atom=geom, basis={'Au': gto.mole.uncontracted_basis(gto.basis.load(base[4:], 'Au'))}, verbose=0)
    t0 = time.perf_counter(); xmol, contr, d4, d2 = g['atomic_densities'](mol); t_atm = time.perf_counter() - t0
    n2c = xmol.nao_2c()
    mfnr = scf.RHF(mol); dmnr = mfnr.get_init_guess(); opt = mfnr.init_direct_scf(mol)
    t0 = time.perf_counter(); scf.hf.get_jk(mol, dmnr, vhfopt=opt, with_k=False); t_nrj = time.perf_counter() - t0
    t0 = time.perf_counter(); scf.hf.get_jk(mol, dmnr, vhfopt=opt); t_nr = time.perf_counter() - t0
    m4 = scf.DHF(xmol); m4.verbose = 0
    t0 = time.perf_counter(); X.x2c1e_hfw0_4cmat(m4.get_hcore(), m4.get_ovlp(), mol=xmol); t_x2c = time.perf_counter() - t0
    t0 = time.perf_counter(); ae = g['element_expansions'](xmol, d4, d2)
    terms = [(g['potential_terms'](e4), g['potential_terms'](e2)) for e4, e2 in ae]; t_exp = time.perf_counter() - t0
    tt = []; t0 = time.perf_counter(); l4, s4, l2 = j_kernel(xmol, terms, 1e-20, tt); t_k = time.perf_counter() - t0
    j2 = l2 - xmol.intor('int1e_nuc_spinor')
    ca, cb = xmol.sph2spinor_coeff()
    p2 = sum((c.conj() @ d2 @ c.T) for c in (ca, cb)).real
    vjs = scf.hf.get_jk(xmol, p2, with_k=False, vhfopt=opt)[0]
    vj2 = ca.conj().T @ vjs @ ca + cb.conj().T @ vjs @ cb
    print('%-4s %5d | %7.1f %7.1f %7.1f | %7.2f %7.1f %7.1f %7.1f %7.1f | %7.2f | %8.1e   (atomic x2camf %.1f s)' % (
        name, n2c, t_nrj, t_nr, t_x2c, t_exp, tt[0], tt[1], tt[2], t_exp + t_k, (t_exp + t_k) / t_nr,
        abs(j2 - vj2).max(), t_atm), flush=True)
