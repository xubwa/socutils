'''
The spin-orbital form of a spin-restricted Hamiltonian, for cross-checks of
zmr.nr against the spinor code: in every orbital space the beta spin-orbitals
are followed by the alpha ones, which is the spin-orbital order of
zmr.nr.active, so determinants and phases coincide and a CI vector of the
sector (n_alpha, n_beta) is embedded by copying it to the positions
sector_index(norb, nelec).
'''

import numpy

from socutils.zmr import integrals
from socutils.zmr.active import strings as st
from socutils.zmr.nr.active import ALPHA, BETA, sector_index
from socutils.zmr.spaces import OrbitalSpaces


def spinor_eris(eris):
    '''zmr.integrals.ERIs for the spatial integrals eris (zmr.nr.integrals).'''
    sp = eris.spaces
    orb, spin = [], []
    for label in 'cav':
        sl = sp.slice(label)
        idx = list(range(sl.start, sl.stop))
        orb += idx + idx
        spin += [BETA] * len(idx) + [ALPHA] * len(idx)
    orb = numpy.array(orb, dtype=int)
    same = numpy.equal.outer(spin, spin)
    h = eris.h1[numpy.ix_(orb, orb)] * same
    g = eris.eri_block('pppp')[numpy.ix_(orb, orb, orb, orb)]
    g = g * same[:, :, None, None] * same[None, None, :, :]
    return integrals.ERIs(eris.ecore, h, g, OrbitalSpaces(2 * sp.ncore, 2 * sp.nact, 2 * sp.nvirt))


def embed(civecs, norb, nelec):
    '''CI vectors (columns) of the sector nelec = (n_alpha, n_beta) as
    vectors of the spinor sector with n_alpha + n_beta electrons.'''
    civecs = numpy.asarray(civecs)
    if civecs.ndim == 1:
        civecs = civecs[:, None]
    idx = sector_index(norb, nelec)
    out = numpy.zeros((len(st.gen_strings(2 * norb, sum(nelec))), civecs.shape[1]),
                      dtype=numpy.complex128)
    out[idx] = civecs
    return out
