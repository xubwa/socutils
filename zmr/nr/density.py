'''
Unrelaxed (response) one-particle density of the spin-free PC-NEVPT2 (zmr.nr) for a
singlet model state, through the spinor code on the spin-orbital form of the same
Hamiltonian: for singlets the spin-free contracted functions and the spin-orbital ones
span the same first-order spaces (zmr/tests/test_nr_nevpt2.py), so the second-order
energy and its derivatives coincide.

The spin-orbitals are in the order of zmr.nr.spinor (in every space the beta ones, then
the alpha ones), in which a CI vector of PySCF (sector (n_alpha, n_beta)) is a vector of
the spinor sector with the same coefficients (zmr.nr.spinor.embed): the model state is the
one of the CASSCF, not solved again.

    route = spin_orbital_route(mc, cderi)
    ref, d2 = make_rdm1(mc, cderi)     spatial, D[P, Q] = sum_sigma <P_sigma^+ Q_sigma>
    dE2 / dlambda = sum O[P, Q] d2[P, Q]   for h -> h + lambda O at fixed orbitals and CI
'''

import numpy

from socutils.zmr.interface import pyscf_nr
from socutils.zmr.nr.spinor import embed
from socutils.zmr.pc.nevpt2 import PCNEVPT2
from socutils.zmr.uc import density


class Route:
    '''The spin-orbital Hamiltonian of a PySCF CASSCF/CASCI and the map back to its
    spatial orbitals: spatial[j], spin[j] (0 alpha, 1 beta) of spin-orbital j.'''

    def __init__(self, mc, cderi, mo_coeff=None, ci=None):
        if mc.mol.spin != 0:
            raise NotImplementedError('the spin-orbital route equals the spin-free PC-NEVPT2 '
                                      'for singlets only')
        mo = numpy.asarray(mc.mo_coeff if mo_coeff is None else mo_coeff)
        nao, nmo = mo.shape
        ncore, ncas = mc.ncore, mc.ncas
        cols, spatial, spin = [], [], []
        for lo, hi in ((0, ncore), (ncore, ncore + ncas), (ncore + ncas, nmo)):
            for sigma in (1, 0):                 # beta block, then alpha block
                for p in range(lo, hi):
                    c = numpy.zeros(2 * nao)
                    if sigma == 0:
                        c[:nao] = mo[:, p]
                    else:
                        c[nao:] = mo[:, p]
                    cols.append(c)
                    spatial.append(p)
                    spin.append(sigma)
        self.c_ghf = numpy.array(cols).T
        self.spatial = numpy.array(spatial)
        self.spin = numpy.array(spin)
        self.nmo = nmo
        self.eris = pyscf_nr.ghf_eris(mc.mol, mc.get_hcore(), self.c_ghf, 0, 2 * ncore,
                                      2 * ncas, cderi=cderi)
        nelecas = mc.nelecas
        self.nelec_ab = (int(nelecas[0]), int(nelecas[1]))
        self.nelec = sum(self.nelec_ab)
        ci = mc.ci if ci is None else ci
        self.ci = numpy.asarray(ci).ravel()
        self.civecs = embed(self.ci, ncas, self.nelec_ab)

    def spin_sum(self, d):
        '''Spatial matrix sum_sigma d[(P sigma), (Q sigma)] of a spin-orbital one.'''
        out = numpy.zeros((self.nmo, self.nmo), dtype=numpy.result_type(d, float))
        same = self.spin[:, None] == self.spin[None, :]
        numpy.add.at(out, (self.spatial[:, None].repeat(len(self.spin), 1)[same],
                           self.spatial[None, :].repeat(len(self.spin), 0)[same]), d[same])
        return out

    def pt(self, max_memory=4000):
        '''The PC-NEVPT2 of the spinor code for the model state.'''
        pt = PCNEVPT2(self.eris, self.nelec, civecs=self.civecs)
        pt.verbose = 0
        pt.max_memory = max(pt.max_memory, max_memory)
        pt.kernel()
        return pt


def spin_orbital_route(mc, cderi, mo_coeff=None, ci=None):
    return Route(mc, cderi, mo_coeff, ci)


def spin_sum(d):
    '''Spatial density from a spin-orbital one in the interleaved order p = 2P + sigma.'''
    return d[0::2, 0::2] + d[1::2, 1::2]


def make_rdm1(mc, cderi, max_memory=4000, timings=None):
    '''(D_ref, D_pt2, pt): the reference and unrelaxed second-order 1-RDMs of the singlet
    model state of the PySCF CASSCF/CASCI mc (its CI vector), in its orbitals (all
    orbitals correlated), and the spin-orbital PC-NEVPT2 they come from.  cderi:
    three-index vectors of the AO integrals (zmr.interface.cholesky).'''
    route = Route(mc, cderi)
    pt = route.pt(max_memory)
    ref, d2 = density.make_rdm1(pt, max_memory=max_memory, timings=timings)
    return route.spin_sum(ref), route.spin_sum(d2), pt
