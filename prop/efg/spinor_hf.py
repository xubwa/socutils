#!/usr/bin/env python
#
# Author: Chaoqun Zhang <bbzhchq@gmail.com>
# Modified from pyscf/prop/efg/dhf.py
#

'''
Electric field gradients for relativistic 2-component JHF methods.
(In testing)
'''

import warnings
import numpy as np
from pyscf import lib

warnings.warn('Module EFG is under testing')

def _get_quad_nuc(mol, atm_id):
    mask = np.ones(mol.natm, dtype=bool)
    mask[atm_id] = False  # exclude the contribution from atm_id
    dr = mol.atom_coords()[mask] - mol.atom_coord(atm_id)
    d = np.linalg.norm(dr, axis=1)
    rr = 3*np.einsum('ix,iy->ixy', dr, dr)
    for i in range(3):
        rr[:,i,i] -= d**2
    z = mol.atom_charges()[mask]
    efg_nuc = np.einsum('i,ixy->xy', z/d**5, rr)
    return efg_nuc


def kernel(method, efg_nuc=None, dm=None):
    log = lib.logger.Logger(method.stdout, method.verbose)
    log.info('\n******** EFG for 2-component SCF methods (In testing) ********')
    mol = method.mol
    if efg_nuc is None:
        efg_nuc = range(mol.natm)

    c = lib.param.LIGHT_SPEED
    if dm is None:
        dm = method.make_rdm1()

    log.info('\nElectric Field Gradient Tensor Results')
    n2c = mol.nao_2c()
    coords = mol.atom_coords()
    aoLa, aoLb = mol.eval_gto('GTOval_spinor', coords)
    aoSa, aoSb = mol.eval_gto('GTOval_sp_spinor', coords)
    efg = []
    for i, atm_id in enumerate(efg_nuc):
        # The electronic quadrupole operator (3 \vec{r} \vec{r} - r^2) / r^5
        with mol.with_rinv_origin(coords[atm_id]):
            ipipv = mol.intor('int1e_ipiprinv_spinor', 9).reshape(3,3,n2c,n2c)
            ipvip = mol.intor('int1e_iprinvip_spinor', 9).reshape(3,3,n2c,n2c)
            h1LL = ipipv + ipvip  # (nabla i | r/r^3 | j)
            h1LL = h1LL + h1LL.conj().transpose(0,1,3,2)
            trace = h1LL[0,0] + h1LL[1,1] + h1LL[2,2]
            h1LL[0,0] -= trace
            h1LL[1,1] -= trace
            h1LL[2,2] -= trace

            ipipv = mol.intor('int1e_ipipsprinvsp_spinor', 9).reshape(3,3,n2c,n2c)
            ipvip = mol.intor('int1e_ipsprinvspip_spinor', 9).reshape(3,3,n2c,n2c)
            h1SS = ipipv + ipvip  # (nabla i | r/r^3 | j)
            h1SS = h1SS + h1SS.conj().transpose(0,1,3,2)
            trace = h1SS[0,0] + h1SS[1,1] + h1SS[2,2]
            h1SS[0,0] -= trace
            h1SS[1,1] -= trace
            h1SS[2,2] -= trace
        
        fcLL = np.einsum('p,q->pq', aoLa[atm_id].conj(), aoLa[atm_id])
        fcLL+= np.einsum('p,q->pq', aoLb[atm_id].conj(), aoLb[atm_id])
        fcSS = np.einsum('p,q->pq', aoSa[atm_id].conj(), aoSa[atm_id])
        fcSS+= np.einsum('p,q->pq', aoSb[atm_id].conj(), aoSb[atm_id])

        efg_e = np.zeros((3,3), dtype=dm.dtype)
        for x in range(3):
            for y in range(x+1):
                int_4c = np.zeros((n2c*2, n2c*2), dtype=dm.dtype)
                int_4c[:n2c,:n2c] = h1LL[x,y]
                int_4c[n2c:,n2c:] = h1SS[x,y] / 4.0 / c**2
                if x == y:
                    int_4c[:n2c,:n2c] -= 8*np.pi/3 * fcLL
                    int_4c[n2c:,n2c:] -= 8*np.pi/3 * fcSS / 4.0 / c**2
                from pyscf.socutils.somf import x2c_grad
                int_2c = x2c_grad.x2c1e_hfw1(mol, int_4c)
                efg_e[x,y] = np.einsum('ij,ji->', int_2c, dm)
                efg_e[y,x] = efg_e[x,y]

        efg_nuc = _get_quad_nuc(mol, atm_id)
        v = efg_nuc - efg_e
        efg.append(v)
        log.info('\nAtom %d' % atm_id)
        log.info('Electronic EFG:')
        print(efg_e.real)
        log.info('Nuclear EFG:')
        print(efg_nuc.real)

        eigs, _ = np.linalg.eigh(v)
        log.info('Total EFG on main axis (Eigenvalues of total EFG tensors):')
        print(eigs)

    return np.asarray(efg).real

EFG = kernel

from pyscf.socutils.scf import spinor_hf
spinor_hf.JHF.EFG = lib.class_as_method(EFG)


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = '''
O 0.          0. -0.12390941
H 0. -1.42993701  0.98326612
H 0.  1.42993701  0.98326612
'''
    mol.basis = 'uncccpvdz'
    mol.unit = 'Bohr'
    mol.build()
    from pyscf.socutils.scf import x2camf_hf
    mf = x2camf_hf.X2CAMF_RHF(mol, with_gaunt=False, with_breit=False)
    mf.conv_tol = 1e-12
    mf.kernel()
    mf.EFG()