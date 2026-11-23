"""Reader of the DIRAC basis-library format (dyall.* files): {Z: {l: [exponents]}}."""
import re
def read(path):
    out, z, l = {}, None, None
    todo = 0
    for line in open(path):
        s = line.strip()
        m = re.match(r'^a\s+(\d+)\s*$', s, re.I)
        if m:
            z = int(m.group(1)); out[z] = {}; l = -1; todo = 0
            continue
        if z is None or not s or s.startswith('$'):
            continue
        t = s.split()
        if todo == 0:
            if len(t) >= 2 and all(re.fullmatch(r'\d+', x) for x in t):
                l += 1; todo = int(t[0]); out[z][l] = []
            continue
        out[z][l].append(float(t[0].replace('D', 'E')))
        todo -= 1
    return {z: {l: sorted(v, reverse=True) for l, v in d.items() if v} for z, d in out.items()}
if __name__ == '__main__':
    import sys
    D = '/home/xwang405/apps/DIRAC-23.0-Source/basis/'
    for name in ('dyall.v2z', 'dyall.v3z', 'dyall.v4z'):
        b = read(D + name)
        print(name, len(b), 'elements')
        for z, sym in ((78, 'Pt'), (79, 'Au'), (80, 'Hg'), (48, 'Cd'), (30, 'Zn'), (81, 'Tl')):
            if z in b:
                print('  %s  %s' % (sym, '  '.join('%s:%d' % ('spdfghi'[l], len(v)) for l, v in sorted(b[z].items()))))
                for l in sorted(b[z]):
                    if l >= 3:
                        v = b[z][l]
                        print('      %s  %s   ratios %s' % ('spdfghi'[l], ' '.join('%.5g' % x for x in v), ' '.join('%.2f' % (a / c) for a, c in zip(v, v[1:]))))
