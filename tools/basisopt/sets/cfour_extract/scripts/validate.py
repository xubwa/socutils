"""Load the staged modules through socutils' register() and build molecules: every element of
every module (scalar AO count for plain sets, spinor count for j-adapted sets)."""
import glob, os, sys, warnings
warnings.simplefilter('ignore')
from pyscf import gto
from socutils.gto import basis as sb
names = sb.register(os.path.abspath('modules'))
bad = 0
for f in sorted(glob.glob('modules/*.py')):
    stem = os.path.basename(f)[:-3]
    d = {}
    exec(open(f).read(), d)
    els = [k for k in d if k[:1].isupper() and isinstance(d[k], list)]
    for el in els:
        try:
            mol = gto.M(atom='%s 0 0 0' % el, basis={el: stem}, spin=None, verbose=0)
        except Exception as e:
            try:
                mol = gto.M(atom='%s 0 0 0' % el, basis={el: stem}, spin=1, verbose=0)
            except Exception as e2:
                print('FAIL', stem, el, str(e2)[:80]); bad += 1; continue
        if mol.nao_2c() <= 0:
            print('EMPTY', stem, el); bad += 1
print('registered %d names; failures %d' % (len(names), bad))
for stem in ('cc-pvtz-so', 'dy-vtz-so', 'cc-pvtz-de4', 'aug-cc-pvtz-x2c'):
    d = {}; exec(open('modules/%s.py' % stem).read(), d)
    el = 'O' if 'O' in d else [k for k in d if k[:1].isupper()][0]
    mol = gto.M(atom='%s 0 0 0' % el, basis={el: stem}, spin=None if el != 'O' else 2, verbose=0)
    print('%-16s %-2s nao_nr %4d  nao_2c %4d' % (stem, el, mol.nao_nr(), mol.nao_2c()))
