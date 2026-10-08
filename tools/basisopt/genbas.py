"""Reading and writing basis sets in the CFOUR GENBAS format.

An entry is
    <ATOM>:<NAME>
    <comment line; for the j-adapted "-SO" sets: "nl  l1 n1  l2 n2 ..." = number of
     contracted functions per kappa for the listed l>
    <blank>
    <number of l shells>
    <l values> / <contracted functions per l> / <primitives per l>
    then for each l: exponents, blank line, coefficient matrix (primitives x contracted).
"""
import numpy as np


class Entry:
    def __init__(self, name, comment, shells):
        self.name = name          # e.g. 'O:PVTZ-X2C'
        self.comment = comment
        self.shells = shells      # {l: (exponents[nprim], coefficients[nprim, ncont])}

    def so_info(self):
        """{l: contracted functions per kappa} from the comment line of an -SO set, or {}."""
        try:
            v = [int(x) for x in self.comment.split()]
        except ValueError:
            return {}
        if not v or len(v) != 2 * v[0] + 1:
            return {}
        return {v[1 + 2 * i]: v[2 + 2 * i] for i in range(v[0])}


def read(path, name, occurrence=-1):
    """Entry `name` ('O:PVTZ-X2C') of a GENBAS file; the last occurrence by default (several
    entries of one name exist in some files)."""
    with open(path) as f:
        lines = f.read().splitlines()
    hits = [i for i, line in enumerate(lines) if line.split() and line.split()[0] == name
            and line.startswith(name)]
    if not hits:
        raise KeyError('%s not found in %s' % (name, path))
    i = hits[occurrence]
    comment = lines[i + 1].strip()
    # some entries pad the three header lines with zeros beyond the number of shells
    nl = int(lines[i + 3].split()[0])
    ls, ncont, nprim = ([int(x) for x in lines[i + 4 + k].split()[:nl]] for k in range(3))
    tokens = iter(' '.join(lines[i + 7:i + 3 + _span(lines, i + 3)]).split())
    shells = {}
    for l, nc, npr in zip(ls, ncont, nprim):
        exps = np.array([float(next(tokens)) for _ in range(npr)])
        coef = np.array([float(next(tokens)) for _ in range(npr * nc)]).reshape(npr, nc)
        if npr:
            shells[l] = (exps, coef)
    return Entry(name, comment, shells)


def _span(lines, start):
    """Number of lines from `start` to the next entry header (a line containing ':')."""
    n = 0
    while start + n < len(lines) and ':' not in lines[start + n]:
        n += 1
    return n


def format_entry(entry):
    ls = sorted(entry.shells)
    out = [entry.name, entry.comment, '', '%3d' % len(ls)]
    out.append(''.join('%5d' % l for l in ls))
    out.append(''.join('%5d' % entry.shells[l][1].shape[1] for l in ls))
    out.append(''.join('%5d' % len(entry.shells[l][0]) for l in ls))
    for l in ls:
        exps, coef = entry.shells[l]
        out.append('')
        for i in range(0, len(exps), 5):
            out.append(''.join('%20.8f' % e if e >= 1e-3 else '%20.10f' % e for e in exps[i:i + 5]))
        out.append('')
        for row in coef:
            out.append(' '.join('%13.8f' % c for c in row))
    out.append('')
    return '\n'.join(out) + '\n'


def append(path, entry):
    with open(path, 'a') as f:
        f.write(format_entry(entry))


def to_pyscf(entry, kappa=False):
    """PySCF basis list.  kappa=False: [[l, [exp, c1, c2, ...], ...], ...], one shell per l with
    all columns (for a j-adapted entry that is the union of both kappas, e.g. for scalar
    checks).  kappa=True: the j-adapted form [[l, kappa, [exp, c...], ...], ...] understood by
    PySCF's spinor code: kappa = l for j = l-1/2, -l-1 for j = l+1/2, 0 for the functions
    shared by both (the uncontracted primitives, and all functions of the l that are not
    j-adapted); entries without the "-SO" line are returned
    in the plain form."""
    so = entry.so_info() if kappa else {}
    out = []
    for l, (exps, coef) in sorted(entry.shells.items()):
        n = so.get(l, 0)
        blocks = [(0 if so else None, coef)] if n == 0 else [(l, coef[:, :n]), (-l - 1, coef[:, n:2 * n]),
                                                             (0, coef[:, 2 * n:])]
        for k, c in blocks:
            if c.shape[1] == 0:
                continue
            rows = [[float(e)] + [float(x) for x in row] for e, row in zip(exps, c)]
            out.append(([l] if k is None else [l, k]) + rows)
    return out


def from_pyscf(name, basis, comment=None, rtol=1e-9):
    """genbas.Entry from a PySCF basis list (the internal format: [[l, [exp, c, ...], ...], ...]
    or with kappa, [[l, kappa, [exp, c, ...], ...], ...]).  Shells of one l are merged over
    the union of their exponents (zero coefficients where a shell lacks a primitive).
    Columns are ordered as the GENBAS "-SO" convention requires: kappa = l (j = l-1/2),
    kappa = -l-1 (j = l+1/2), then kappa = 0; the two kappas must have the same number of
    functions.  For s shells kappa = -1 and 0 are the same thing."""
    per_l = {}
    for sh in basis:
        l = sh[0]
        kap, rows = (sh[1], sh[2:]) if isinstance(sh[1], (int, np.integer)) else (0, sh[1:])
        if kap not in (0, l, -l - 1):
            raise ValueError('l = %d: kappa = %d is not possible' % (l, kap))
        cls = 'minus' if (kap == l and l > 0) else 'plus' if (kap == -l - 1 and l > 0) else 'shared'
        rows = [[float(x) for x in r] for r in rows]
        per_l.setdefault(l, []).append((cls, [r[0] for r in rows], np.array([r[1:] for r in rows])))
    shells, so = {}, []
    for l in sorted(per_l):
        exps = []
        for _, e, _ in per_l[l]:
            for x in e:
                if not any(abs(x - y) <= rtol * y for y in exps):
                    exps.append(x)
        exps.sort(reverse=True)
        cols = {'minus': [], 'plus': [], 'shared': []}
        for cls, e, c in per_l[l]:
            full = np.zeros((len(exps), c.shape[1]))
            for i, x in enumerate(e):
                full[[j for j, y in enumerate(exps) if abs(x - y) <= rtol * y][0]] += c[i]
            cols[cls].append(full)
        n = {k: sum(c.shape[1] for c in v) for k, v in cols.items()}
        if n['minus'] != n['plus']:
            raise ValueError('l = %d: %d functions for j = l-1/2 but %d for j = l+1/2; the GENBAS '
                             'j-adapted format needs equal numbers' % (l, n['minus'], n['plus']))
        if n['minus']:
            so.append((l, n['minus']))
        shells[l] = (np.array(exps), np.hstack(cols['minus'] + cols['plus'] + cols['shared']))
    if comment is None:
        comment = ('%d  ' % len(so) + ' '.join('%d %d' % x for x in so)) if so else 'converted from the PySCF format'
    return Entry(name, comment, shells)


def format_pyscf(basis, indent='    '):
    """Text of a PySCF basis list as a Python literal, one primitive per line."""
    out = ['[']
    for sh in basis:
        head = sh[:2] if isinstance(sh[1], (int, np.integer)) else sh[:1]
        out.append(indent + '[' + ', '.join(str(int(x)) for x in head) + ',')
        for row in sh[len(head):]:
            out.append(indent * 2 + '[' + ', '.join(repr(float(x)) for x in row) + '],')
        out.append(indent + '],')
    out.append(']')
    return '\n'.join(out)


def write_pyscf(path, basis_by_element, header=''):
    """Python file defining `basis = {'O': [...], ...}` (PySCF internal format; with kappa for
    j-adapted sets).  Use: `from <file> import basis; gto.M(..., basis=basis)`, or
    read_pyscf(path)."""
    with open(path, 'w') as f:
        if header:
            f.write(''.join('# %s\n' % line for line in header.splitlines()))
        f.write('basis = {\n')
        for el, b in basis_by_element.items():
            f.write('%r: %s,\n' % (el, format_pyscf(b)))
        f.write('}\n')


def read_pyscf(path):
    """{element: basis list} from a file written by write_pyscf (or any Python file whose only
    assignment is a literal `basis = {...}`); the file is parsed, not executed."""
    import ast
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'basis':
            return ast.literal_eval(node.value)
    raise ValueError('no literal `basis = {...}` in %s' % path)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Convert basis sets between the CFOUR GENBAS format '
                                 'and the PySCF internal format (plain or j-adapted with kappa).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('to-genbas', help='PySCF format -> GENBAS entry (appended to --out)')
    a.add_argument('source', help='Python file with `basis = {element: [...]}` (as written by '
                   'to-pyscf), or a basis-set name known to PySCF')
    a.add_argument('elements', nargs='+', help='elements to convert')
    a.add_argument('--name', required=True, help='name of the entries (without the element)')
    a.add_argument('--out', default='GENBAS')
    b = sub.add_parser('to-pyscf', help='GENBAS entries -> Python file with `basis = {...}`')
    b.add_argument('genbas')
    b.add_argument('entries', nargs='+', help="entries, e.g. 'O:PVTZ-SO' 'H:PVTZ-SO'")
    b.add_argument('--out', required=True, help='Python file to write')
    b.add_argument('--plain', action='store_true',
                   help='no kappa: one shell per l with all columns (also for "-SO" entries)')
    args = ap.parse_args()
    if args.cmd == 'to-genbas':
        import os
        if os.path.exists(args.source):
            src = read_pyscf(args.source)
        else:
            from pyscf import gto
            src = {el: gto.basis.load(args.source, el) for el in args.elements}
        for el in args.elements:
            entry = from_pyscf('%s:%s' % (el.upper(), args.name), src[el])
            append(args.out, entry)
            print('%s appended to %s  (%s)' % (entry.name, args.out, ' '.join(
                '%d%s->%d' % (len(e), 'spdfghik'[l], c.shape[1]) for l, (e, c) in sorted(entry.shells.items()))))
    else:
        out = {}
        for name in args.entries:
            entry = read(args.genbas, name)
            el = name.split(':')[0].capitalize()
            out[el] = to_pyscf(entry, kappa=not args.plain)
        write_pyscf(args.out, out, header='from %s: %s' % (args.genbas, ' '.join(args.entries)))
        print('%s written (%s)' % (args.out, ', '.join(out)))


if __name__ == '__main__':
    main()
