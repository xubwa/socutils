'''
Brute-force uncontracted MRCISD in the determinant basis.

The MRCISD space of a complete active space reference is spanned by all
determinants with at most two core holes and at most two virtual particles
(the CAS determinants plus their first-order interacting space).  The
Hamiltonian is built with Slater-Condon rules (socutils.fci.zfci) on that
explicit determinant list and diagonalized densely.  Validation only.
'''

import itertools

import numpy
from socutils.fci import zfci

from socutils.zmr.reference import dets

# (core holes, virtual particles) -> block name; the internal block is 'cas'
BLOCK_NAME = {(0, 0): 'cas', (1, 0): 'caaa', (2, 0): 'ccaa', (0, 1): 'aaav', (0, 2): 'aavv',
              (1, 1): 'cava', (2, 1): 'ccav', (1, 2): 'cavv', (2, 2): 'ccvv'}


def mrci_strings(sp, nelec, blocks=None):
    '''Determinants (bit strings) of the MRCISD space, and their block
    (core holes, virtual particles).  blocks restricts the list of
    (holes, particles) pairs.'''
    core = list(range(sp.ncore))
    act = list(range(sp.ncore, sp.nocc))
    virt = list(range(sp.nocc, sp.nmo))
    out, labels = [], []
    for h in range(3):
        for p in range(3):
            if blocks is not None and (h, p) not in blocks:
                continue
            na = nelec + h - p
            if na < 0 or na > sp.nact or h > sp.ncore or p > sp.nvirt:
                continue
            act_strs = dets.gen_strings(act, na)
            for holes in itertools.combinations(core, h):
                cmask = sum(1 << c for c in core if c not in holes)
                for parts in itertools.combinations(virt, p):
                    vmask = sum(1 << v for v in parts)
                    out.append(act_strs | numpy.uint64(cmask | vmask))
                    labels += [(h, p)] * len(act_strs)
    return numpy.concatenate(out), labels


def kernel(eris, nelec, nroots=1, blocks=None):
    '''Lowest nroots MRCISD energies and vectors.

    Returns (energies, vectors (ndet, nroots), determinant strings).'''
    sp = eris.spaces
    strs, _ = mrci_strings(sp, nelec, blocks)
    occs = [dets.str_to_occ(s) for s in strs]
    hmat = zfci.make_hmat_dets(eris.h1, eris.eri_block('pppp'), sp.nmo, occs)
    hmat = hmat + eris.ecore * numpy.eye(len(strs))
    e, c = numpy.linalg.eigh(hmat)
    return e[:nroots], c[:, :nroots], strs
