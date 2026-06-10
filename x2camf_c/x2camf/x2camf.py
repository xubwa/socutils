from x2camf import libx2camf
from pyscf import gto, lib
from pyscf.data import elements
from pyscf.x2c import x2c
from pyscf.gto import mole
import numpy

def construct_molecular_matrix(atm_blocks, atom_slices, xmol, n2c, four_component):
    if four_component:
        mol_matrix = numpy.zeros((n2c*2, n2c*2))
        for ia in range(xmol.natm):
            n2ca = atm_blocks[xmol.elements[ia]].shape[0]//2
            _, _, c0, c1 = atom_slices[ia]
            mol_matrix[c0:c1,c0:c1] = atm_blocks[xmol.elements[ia]][:n2ca,:n2ca]
            mol_matrix[n2c+c0:n2c+c1,c0:c1] = atm_blocks[xmol.elements[ia]][n2ca:,:n2ca]
            mol_matrix[c0:c1,n2c+c0:n2c+c1] = atm_blocks[xmol.elements[ia]][:n2ca,n2ca:]
            mol_matrix[n2c+c0:n2c+c1,n2c+c0:n2c+c1] = atm_blocks[xmol.elements[ia]][n2ca:,n2ca:]
    else:
        mol_matrix = numpy.zeros((n2c, n2c))
        for ia in range(xmol.natm):
            _, _, c0, c1 = atom_slices[ia]
            mol_matrix[c0:c1,c0:c1] = atm_blocks[xmol.elements[ia]]
    return mol_matrix

def amfi(x2cobj, printLevel = 0, with_gaunt = True, with_gauge = True, with_gaunt_sd=False, gaussian_nuclear = False, aoc = False, pt = False, pcc = False, int4c = False):
    mol = x2cobj.mol
    #computes the internal integer for soc integral flavor.
    soc_int_flavor = 0
    soc_int_flavor += with_gaunt << 0
    soc_int_flavor += with_gauge << 1
    soc_int_flavor += gaussian_nuclear << 2
    soc_int_flavor += aoc << 3
    soc_int_flavor += pt << 4
    soc_int_flavor += pcc << 5
    soc_int_flavor += int4c << 6
    soc_int_flavor += with_gaunt_sd << 7

    uniq_atoms = set([a[0] for a in mol._atom])
    amf_int = {}
    den_4c = {}
    for atom in uniq_atoms:
        symbol = mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = mole.uncontracted_basis(mol._basis[atom])
        #amf_internal_basis
        shell = []
        exp_a = []
        for bas in raw_bas:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = numpy.asarray(shell)
        exp_a = numpy.asarray(exp_a)
        amf_int[atom] = _amf(atom_number, shell, exp_a, soc_int_flavor, printLevel)

    xmol, _ = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    
    amf_matrix = construct_molecular_matrix(amf_int, atom_slices, xmol, n2c, int4c)
    return amf_matrix

def pcc_k(x2cobj, printLevel = 0, with_gaunt = True, with_gauge = True, with_gaunt_sd=False, gaussian_nuclear = False, aoc = False, pt = False, pcc = False, int4c = False):
    mol = x2cobj.mol
    #computes the internal integer for soc integral flavor.
    soc_int_flavor = 0
    soc_int_flavor += with_gaunt << 0
    soc_int_flavor += with_gauge << 1
    soc_int_flavor += gaussian_nuclear << 2
    soc_int_flavor += aoc << 3
    soc_int_flavor += pt << 4
    soc_int_flavor += pcc << 5
    soc_int_flavor += int4c << 6
    soc_int_flavor += with_gaunt_sd << 7

    uniq_atoms = set([a[0] for a in mol._atom])
    pcc_int = {}
    for atom in uniq_atoms:
        symbol = mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = mole.uncontracted_basis(mol._basis[atom])
        #amf_internal_basis
        shell = []
        exp_a = []
        for bas in raw_bas:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = numpy.asarray(shell)
        exp_a = numpy.asarray(exp_a)
        pcc_int[atom] = _pcc_k(atom_number, shell, exp_a, soc_int_flavor, printLevel)

    xmol, _ = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    
    pcc_matrix = construct_molecular_matrix(pcc_int, atom_slices, xmol, n2c, int4c)
    return pcc_matrix


# takes an atom number basis and flavor of soc integral and returns the amf matrix
# basis should be like [[shell,primitive],[shell,primitive],...]
# only works for uncontracted basis
# this function serves as a raw interface.
def _amf(atom_number, shell, exp_a, soc_int_flavor, printLevel):
    if atom_number > 118 or atom_number < 1:
        raise ValueError("atom number must be between 1 and 118")

    nbas = shell.shape[0]
    nshell = shell[-1]+1
    results = libx2camf.amfi(soc_int_flavor, atom_number, nshell, nbas, printLevel, shell, exp_a,
                             speed_of_light=lib.param.LIGHT_SPEED)
    amf_mat = results[0]
    return amf_mat

def _pcc_k(atom_number, shell, exp_a, soc_int_flavor, printLevel):
    if atom_number > 118 or atom_number < 1:
        raise ValueError("atom number must be between 1 and 118")

    nbas = shell.shape[0]
    nshell = shell[-1]+1
    results = libx2camf.pcc_K(soc_int_flavor, atom_number, nshell, nbas, printLevel, shell, exp_a,
                              speed_of_light=lib.param.LIGHT_SPEED)
    pcc_mat = results[0]
    return pcc_mat

if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = '''
S        0.000000    0.000000    0.117790
'''
    mol.basis = 'unc-cc-pvtz'
    mol.symmetry = 0
    mol.build()

    x2cobj = x2c.X2C(mol)
    amfint = x2camf(x2cobj, spin_free=True, two_c=True, with_gaunt=True, with_gauge=True)

    import oldamf
    mf = oldamf.X2CAMF_RHF(mol, with_gaunt=True, with_breit=True, prog='sph_atm')
    mf.max_cycle = 1
    mf.kernel()
    hcore = numpy.zeros((mol.nao_2c(), mol.nao_2c()), dtype=complex)
    with open("amf_int", "r") as ifs:
        lines = ifs.readlines()
        if(len(lines) != hcore.shape[0]**2):
            print("Something went wrong. The dimension of hcore and amfi calculations do NOT match.")
        else:
            for ii in range(hcore.shape[0]):
                for jj in range(hcore.shape[1]):
                    hcore[ii][jj] = hcore[ii][jj] + complex(lines[ii*hcore.shape[0]+jj])

    print(amfint - hcore)
