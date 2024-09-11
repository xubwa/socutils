import sys
import numpy as np

def read_genbas(basis_name, filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    found = False
    for line in lines:
        if len(line.split()) > 0 and line.split()[0] == basis_name:
            index = lines.index(line)
            found = True
            break
    if not found:
        print(f"Could not find basis set for {basis_name} in {filename}.")
        sys.exit(1)
    maxL = int(lines[index+3].split()[0]) - 1
    basinfo = []
    exps = []
    coeffs = []
    for ii in range(maxL+1):
        basinfo.append([])
        basinfo[ii].append(int(lines[index+4].split()[ii]))
        basinfo[ii].append(int(lines[index+5].split()[ii]))
        basinfo[ii].append(int(lines[index+6].split()[ii]))
        exps.append(np.zeros(basinfo[ii][2]))
        coeffs.append(np.zeros(basinfo[ii][2]*basinfo[ii][1]))
    assert len(exps) == maxL+1
    
    ll = 0
    nexp = 0
    ncoeff = 0
    for line in lines[index+8:]:
        line = line.split()
        if len(line) == 0:
            continue
        else:
            for ii in range(len(line)):
                d_tmp = float(line[ii])
                if nexp < basinfo[ll][2]:
                    exps[ll][nexp] = d_tmp
                    nexp += 1
                elif ncoeff < basinfo[ll][2]*basinfo[ll][1]:
                    coeffs[ll][ncoeff] = d_tmp
                    ncoeff += 1
        if nexp == basinfo[ll][2] and ncoeff == basinfo[ll][2]*basinfo[ll][1]:
            nexp = 0
            ncoeff = 0
            ll += 1
            if ll > maxL:
                break
    for ll in range(maxL+1):
        coeffs[ll] = coeffs[ll].reshape((basinfo[ll][2],basinfo[ll][1]))
    return exps, coeffs, basinfo

def parse_genbas(basis, filename="GENBAS", uncontract=False):
    exps, coeffs, basinfo = read_genbas(basis, filename)
    basis_pyscf = []
    for ii in range(len(basinfo)):
        basis_pyscf.append([basinfo[ii][0]])
        if uncontract:
            for jj in range(basinfo[ii][2]):
                basis_pyscf[ii].append([exps[ii][jj],1.0])
        else:
            for jj in range(basinfo[ii][2]):
                basis_pyscf[ii].append([exps[ii][jj]])
                for kk in range(basinfo[ii][1]):
                    basis_pyscf[ii][jj+1].append(coeffs[ii][jj][kk])
    return basis_pyscf

def genbas_parser(basis, filename="GENBAS"):
    """
    Generate PySCF basis set from CFOUR basis file
    """
    basis_pyscf = {}
    for key, value in basis.items():
        if "-SO" in value:
            raise NotImplementedError("Spin-orbit basis sets are not supported.")
        exps, coeffs, basinfo = read_genbas(value, filename)
        basis_pyscf[key] = []
        for ii in range(len(basinfo)):
            basis_pyscf[key].append([basinfo[ii][0]])
            for jj in range(basinfo[ii][2]):
                basis_pyscf[key][ii].append([exps[ii][jj]])
                for kk in range(basinfo[ii][1]):
                    basis_pyscf[key][ii][jj+1].append(coeffs[ii][jj][kk])
    return basis_pyscf
