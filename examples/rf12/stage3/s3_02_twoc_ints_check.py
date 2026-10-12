import numpy
from pyscf import gto
from socutils.rf12 import ints4c, twoc_ints
mol = gto.M(atom='Ne 0 0 0; H 0 0.4 1.6', unit='bohr', basis={'Ne': 'cc-pvdz', 'H': 'sto-3g'}, charge=1, verbose=0)
nao = mol.nao
rng = numpy.random.default_rng(3)
def rs(n): return rng.random((n, 2, nao)) - .5 + 1j * (rng.random((n, 2, nao)) - .5)
# reference: brute force from full derivative tensors
I = {c: ints4c.full(mol, 'F', c, 1.1) for c in ('plain', 'ip1', 'ipip1', 'ip1ip2', 'ipip1ipip2')}
def ref(b1, b2, k1, k2):
    # put op functions explicitly: build the full sph tensor with operators applied via the full classes
    def E(s, conj): e = twoc_ints._expand(*s); return e.conj() if conj else e
    ops = (b1[1], k1[1], b2[1], k2[1])
    out = 0
    # generic: compute (op b1 * k1 | op b2 * k2) using classes with operator at i/k after permutation
    return None
sets = [rs(2), rs(3), rs(2), rs(3)]
cases = [('L', 'L', 'L', 'L'), ('lap', 'L', 'L', 'L'), ('L', 'L', 'lap', 'L'), ('L', 'lap', 'L', 'L'),
         ('L', 'L', 'L', 'lap'), ('lap', 'lap', 'L', 'L'), ('lap', 'L', 'L', 'lap'), ('sp', 'L', 'L', 'L'),
         ('L', 'L', 'sp', 'L'), ('sp', 'sp', 'L', 'L'), ('L', 'sp', 'sp', 'L')]
for ops in cases:
    b1, b2, k1, k2 = [(s, o) for s, o in zip(sets, ops)]
    vals = []
    for contract in (('b', 'b'), ('k', 'k'), ('b', 'k'), ('k', 'b')):
        vals.append(twoc_ints.eri2(mol, 'F', b1, b2, k1, k2, 1.1, contract))
    # independent reference: hermiticity  <b1 b2|K|k1 k2> = conj <k1 k2|K|b1 b2>
    h = twoc_ints.eri2(mol, 'F', k1, k2, b1, b2, 1.1).conj().transpose(2, 3, 0, 1)
    print(ops, 'modes agree %.1e' % max(abs(v - vals[0]).max() for v in vals),
          ' hermiticity %.1e' % abs(h - vals[0]).max(), ' max %.2e' % abs(vals[0]).max())
# operator check against finite basis: lap on a function = sum over full ipip1 trace, compare to explicit plain ints
# of an analytically differentiated Gaussian is hard; instead: sp applied twice = lap  (sigma.p sigma.p = p^2)
s = sets[0]
a = twoc_ints.eri2(mol, 'F', (s, 'lap'), (sets[1], 'L'), (sets[2], 'L'), (sets[3], 'L'), 1.1)
print('lap vs ip-ip identity: skipped (needs second derivative class)')
