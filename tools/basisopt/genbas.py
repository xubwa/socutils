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


def to_pyscf(entry):
    """PySCF basis list [[l, [exp, c1, c2, ...], ...], ...] (one shell per l)."""
    return [[l] + [[float(e)] + [float(c) for c in row] for e, row in zip(exps, coef)]
            for l, (exps, coef) in sorted(entry.shells.items())]
