import pyscf
import x2camf
from pyscf import gto,x2c
# mol = gto.M(
#     atom=[["O", (0., 0., 0.)], 
#     [1, (0., -0.757, 0.587)],
#     [1, (0., 0.757, 0.587)]],
#     basis={"O":"unc-ccpvdz","H":"unc-ccpvdz"})
# atomUnique = []
# indexList = []
# for aa in range(len(mol.elements)):
#     atom = mol.elements[aa]
#     found = False
#     for ii in range(len(atomUnique)):
#         if atom == atomUnique[ii]:
#             indexList.append(ii)
#             found = True
#             breakfor atom in mol.elements:
#         atomUnique.append(atom)
#         indexList.append(len(atomUnique)-1)        

def write_input(mol, with_gaunt=False, with_breit=False, with_aoc=False):
    with open("amf_input","w") as ofs:
        ofs.write(str(len(mol.elements))+"\n")
        for atom in mol.elements:
            ATOM = atom.upper()
            if(type(mol.basis) is str):
                ofs.write(ATOM+"\t"+ATOM+":"+mol.basis+"\n")
            #elif(type(mol.basis) is list):
            else:
                ofs.write(ATOM+"\t"+ATOM+":"+mol.basis[atom]+"\n")
        ofs.write('%amfiMethod*\n')
        method = str(int(with_aoc))+"\n"+str(int(with_gaunt))+"\n"+str(int(with_breit))
        ofs.write(method)
    with open("GENBAS","w") as ofs:
        def load(basis_name, symb):
            if basis_name.lower().startswith('unc'):
                return gto.uncontract(gto.basis.load(basis_name[3:], symb))
            else:
                return gto.basis.load(basis_name, symb)

        for atom in mol.elements:
            ATOM = atom.upper()
            if(type(mol.basis) is str):
                basisAtom = mol.basis
            else:
                basisAtom = mol.basis[atom]
            basisUNC = gto.uncontract(load(basisAtom,atom))
            maxL = basisUNC[len(basisUNC)-1][0]
            ofs.write(atom.upper()+":"+basisAtom+"\n"+"\n"+str(maxL+1)+"\n")
            basisList = []
            for ll in range(maxL+1):
                basisList.append([])
            for ii in range(len(basisUNC)):
                basisList[basisUNC[ii][0]].append(basisUNC[ii][1][0])
            for ll in range(maxL+1):
                ofs.write(str(ll)+"\t")
            ofs.write("\n")
            for ll in range(maxL+1):
                ofs.write(str(len(basisList[ll]))+"\t")
            ofs.write("\n")
            for ll in range(maxL+1):
                ofs.write(str(len(basisList[ll]))+"\t")
            ofs.write("\n")
            for ll in range(maxL+1):
                for ii in range(len(basisList[ll])):
                    ofs.write(str(basisList[ll][ii])+"\t")
                ofs.write("\n")
                for ii in range(len(basisList[ll])):
                    for jj in range(len(basisList[ll])):
                        if(ii == jj):
                            ofs.write("1.0\t")
                        else:
                            ofs.write("0.0\t")
                    ofs.write("\n")
            ofs.write("\n")


