'''
    Convert libcint integrals to cfour format

    In libcint, the AO integrals within a single shell 
    are ordered as m = -l, -l+1, ..., l-1, l 
    except for p functions, which are ordered as px, py, pz.

    In CFOUR (spinor calculations), the AO integrals within a single shell
    are ordered as m = l, -l, l-1, -l+1, ..., 0.
'''
import struct
from functools import reduce
import numpy as np

def cint2cfour_l(lshell):
    dim = lshell*2+1
    if lshell == 0 or lshell == 1:
        transMat = np.eye(dim)
    else:
        transMat = np.zeros((dim,dim))
        transMat[lshell,-1] = 1.0
        for l in range(1,lshell+1):
            transMat[lshell-l,  -2*l] = 1
            transMat[lshell+l,-1-2*l] = 1
    return transMat

def spinor2sph(mol, spinor):
    assert (spinor.shape[0] == mol.nao_2c()), "spinor integral must be of shape (nao_2c, nao_2c)"
    c = mol.sph2spinor_coeff()
    c2 = np.vstack(c)
    ints_sph = np.einsum('ip,pq,qj->ij', c2, spinor, c2.T.conj())
    return ints_sph
    
def write_cfour_integrals(ints, filename):
    with open(filename, 'wb') as f:
        for j in range(ints.shape[1]):
            for i in range(ints.shape[0]):
                value = ints[i,j]
                f.write(struct.pack('d', value.real))
                f.write(struct.pack('d', value.imag))

def write_cfour_basis(mol, basisName = "PYTEST", filename = "GENBAS"):
    from pyscf.gto import mole
    uniq_atoms = set([a[0] for a in mol._atom])

    with open(filename, 'w') as f:
        for atom in uniq_atoms:
            symbol = mole._std_symbol(atom)
            f.write(symbol.upper() + ":" + basisName.upper() + "\n")
            f.write("generated from pyscf interface\n")
            f.write("\n")
            raw_bas = mole.uncontracted_basis(mol._basis[atom])
            angs = []
            exp_a = []
            for bas in raw_bas:
                angs.append(bas[0])
                exp_a.append(bas[-1][0])
            nshells = len(angs)
            f.write(str(nshells) + "\n")
            for ishel in range(nshells):
                f.write(str(angs[ishel]) + " ")
            f.write("\n")
            for _ in range(2):
                for ishel in range(nshells):
                    f.write("1 ")
                f.write("\n")
            f.write("\n")
            for ishel in range(nshells):
                f.write(str(exp_a[ishel]) + "\n\n1.0\n\n")

    return

def write_cfour_input(mol, basisName = "PYTEST", filename = "ZMAT", job = "SCF", memGB = 4):
    unit = mol.unit.upper()
    cfourinput = '''
*CFOUR(CALC=%s
UNIT=%s
MEMORY=%d
COORD=CARTESIAN
SYM=OFF
FIXGEO=ON
MEM_UNIT=GB
BASIS=SPECIAL)

'''
    with open(filename, "w") as f:
        f.write("input generated from pyscf interface\n")
        for ia in range(mol.natm):
            coord = mol.atom_coord(ia)
            f.write("%s %f %f %f\n" % (mol.atom_symbol(ia), coord[0], coord[1], coord[2]))
        f.write(cfourinput % (job.upper(), unit, memGB))
        for ia in range(mol.natm):
            f.write(mol.atom_symbol(ia) + ":" + basisName + "\n")
        f.write("\n")

def cint2cfour(ints_sph, mol):
    nshells = mol.nbas
    basInfo = mol._bas
    nao_2c = mol.nao_2c()
    transMat = np.zeros((nao_2c,nao_2c))
    offset = 0
    for i in range(nshells):
        angl = basInfo[i][1]
        transMat[offset:offset+2*angl+1, offset:offset+2*angl+1] = cint2cfour_l(angl)
        transMat[nao_2c//2+offset:nao_2c//2+offset+2*angl+1, nao_2c//2+offset:nao_2c//2+offset+2*angl+1] = cint2cfour_l(angl)
        offset += 2*angl+1
    assert (offset == nao_2c//2), "Error in constructing transformation matrix"
    return reduce(np.dot, (transMat.T.conj(), ints_sph, transMat))

if __name__ == '__main__':
    import os
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom = '''
H        0.0 0.0 -2.0
S        0.0 0.0 0.0
H        0.0 0.0 2.0
'''
    mol.unit = 'Bohr'
    mol.basis = 'unc-cc-pvtz'
    mol.build()

    write_cfour_input(mol)
    write_cfour_basis(mol)
    ints = scf.X2C(mol).get_hcore()
    ints = cint2cfour(spinor2sph(mol, ints), mol)
    print(ints.shape)
    write_cfour_integrals(ints, "PYSCFINT")