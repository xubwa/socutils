# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2023/12/7

from pyscf import gto, scf, lib, mcscf, ao2mo
import numpy as np


class CAHF(lib.StreamObject):

    def __init__(self, mol, **kwargs):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        rdm1, rdm2 = self.make_rdm12(None, norb, nelec, **kwargs)
        #eri = ao2mo.restore(1, h2.real, norb) + 1.j * ao2mo.restore(1, h2.imag, norb)
        eri = h2
        e1 = np.einsum('ij,ji', h1, rdm1)
        e2 = 0.5 * np.einsum('ijkl,ijkl->', eri, rdm2)
        return (e1 + e2 + ecore).real, rdm2

    def make_rdm1(self, fake_civec_asNone, ncas, nelec, **kwargs):
        if not isinstance(nelec, (int, np.integer)):
            nelec = sum(nelec)

        return np.diag(np.ones(ncas)) * (nelec / ncas)

    def make_rdm2(self, fake_civec, ncas, nelec, **kwargs):
        if not isinstance(nelec, (int, np.integer)):
            nelec = sum(nelec)
        if fake_civec is None:
            rdm2 = np.zeros((ncas, ncas, ncas, ncas))
            # 2*delta(tu)delta(vw)-(delta(tv)delta(uw))
            # the symmetrized form in molpro is not adopted here
            for i in range(ncas):
                for j in range(ncas):
                    rdm2[i, i, j, j] += 1.
                    rdm2[i, j, j, i] -= 1.

            rdm2 *= nelec * (nelec - 1) / (ncas * (ncas - 1))
            return rdm2
        else:
            return fake_civec

    def make_rdm12(self, fake_civec, ncas, nelec, **kwargs):
        if not isinstance(nelec, (int, np.integer)):
            nelec = sum(nelec)

        rdm1 = self.make_rdm1(fake_civec, ncas, nelec, **kwargs)
        rdm2 = self.make_rdm2(fake_civec, ncas, nelec, **kwargs)

        return rdm1, rdm2

def make_rdm1_det(det, norb):
    rdm1 = np.zeros((norb, norb))
    for occi in det:
        rdm1[occi, occi] = 1.0
    return

def make_rdm2_det(det, norb):
    rdm2 = np.zeros((norb, norb))
    return
# multi determinant average
class MultiSlater(CAHF):
    def __init__(self, mol, det_list, weight_list=None, **kwargs):
        super().__init__(mol)
        self.dets = det_list
        if weight_list is None:
            self.weights = np.ones(len(dets))/len(dets)
        else:
            assert len(det_list) == len(weight_list)


# multiple open shell CAHF
class MultiZCAHF(CAHF):
    def __init__(self, mol, orb_open, elec_open, **kwargs):
        super().__init__(mol)
        self.orb_open = orb_open
        self.elec_open = elec_open
        assert len(orb_open) == len(elec_open), "Open electron and open orbital mismatch"

    def make_rdm1(self, fake_civec, ncas, nelec, **kwargs):
        orb_open = self.orb_open
        elec_open = self.elec_open
        if not isinstance(nelec, (int, np.integer)):
            nelec = sum(nelec)
        assert sum(elec_open) == nelec
        assert sum(orb_open) == ncas

        dm1 = np.diag(np.ones(ncas))
        orb_open_cumu = [sum(orb_open[:i+1]) for i in range(len(orb_open))]
        
        start = 0
        for i, orb in enumerate(orb_open):
            end = start + orb
            elec = elec_open[i]
            dm1[start:end, start:end] *= elec/orb
            start = end
        return dm1

    def make_rdm2(self, fake_civec, ncas, nelec, **kwargs):
        orb_open = self.orb_open
        elec_open = self.elec_open
        if not isinstance(nelec, (int, np.integer)):
            nelec = sum(nelec)
        if fake_civec is None:
            rdm2 = np.zeros((ncas, ncas, ncas, ncas))
            nopen = len(self.orb_open)

            start_i = 0
            for i in range(nopen):
                orb_i = orb_open[i]
                elec_i = elec_open[i]
                alpha_i = elec_i/orb_i
                alpha_ii = alpha_i * (elec_i-1)/(orb_i-1)
                end_i = start_i + orb_i
                for t in range(start_i, end_i):
                    start_j = 0
                    for j in range(nopen):
                        orb_j = orb_open[j]
                        elec_j = elec_open[j]
                        alpha_j = elec_j/orb_j
                        end_j = start_j + orb_j
                        for u in range(start_j, end_j):
                            if i == j:
                                rdm2[t,t,u,u] += alpha_ii
                                rdm2[t,u,u,t] -= alpha_ii
                            else:
                                rdm2[t,t,u,u] += alpha_i * alpha_j
                                rdm2[t,u,u,t] -= alpha_i * alpha_j
                        start_j = end_j
                    start_i = end_i
            return rdm2
        else:
            return fake_civec


if __name__ == '__main__':
    mol = gto.M(atom='F 0 0 0', basis='ccpvdz', verbose=4, spin=1)
    mf = scf.RHF(mol).x2c()
    mf.max_cycle = 0
    mf.run()
    mc = mcscf.CASSCF(mf, 3, 5)
    mc.fcisolver = CAHF(mol)
    mc.mc2step()
