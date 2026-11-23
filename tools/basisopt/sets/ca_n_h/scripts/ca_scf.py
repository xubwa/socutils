"""SF-X2C-1e atomic HF energy of Ca (4s2, Gaussian nucleus) in the uncontracted primitive sets
of the candidate sources, to see whether the non-relativistic cc-pV5Z primitives can carry a
5Z-X2C set."""
import basis_set_exchange as bse
from socutils.tools.basisopt import engine
def prims(name, el='Ca'):
    b = bse.get_basis(name, elements=[el])['elements'][str(bse.lut.element_Z_from_sym(el))]
    p = {}
    for s in b['electron_shells']:
        for l in s['angular_momentum']:
            p.setdefault(l, set()).update(float(x) for x in s['exponents'])
    return {l: sorted(v, reverse=True) for l, v in p.items()}
for name in ['cc-pVTZ', 'cc-pVQZ', 'cc-pV5Z', 'cc-pVTZ-X2C', 'cc-pVQZ-X2C', 'cc-pwCVQZ-X2C', 'dyall-v4z']:
    try: p = prims(name)
    except Exception as e: print(name, 'missing'); continue
    r = engine.atom('Ca', p, 'sfx2c1e', gau_nuc=True)
    print('%-14s %s  E = %.8f' % (name, ' '.join('%d%s' % (len(p[l]), 'spdfghi'[l]) for l in sorted(p)), r.e_scf), "conv" if r.converged else "NOT CONVERGED")
