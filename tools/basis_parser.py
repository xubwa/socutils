import sys
import numpy as np

def read_genbas(basis_name, filename, so_basis=False):
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
    max_ang = int(lines[index+3].split()[0]) - 1
    basinfo = []
    exps = []
    coeffs = []
    for ii in range(max_ang+1):
        basinfo.append([])
        basinfo[ii].append(int(lines[index+4].split()[ii]))
        basinfo[ii].append(int(lines[index+5].split()[ii]))
        basinfo[ii].append(int(lines[index+6].split()[ii]))
        exps.append(np.zeros(basinfo[ii][2]))
        coeffs.append(np.zeros(basinfo[ii][2]*basinfo[ii][1]))
    assert len(exps) == max_ang+1
    print(basinfo)
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
            if ll > max_ang:
                break
    for ll in range(max_ang+1):
        coeffs[ll] = coeffs[ll].reshape((basinfo[ll][2],basinfo[ll][1]))
    if so_basis:
        # read so_basis info from the line after title line
        # which is index+1 line
        raw_info = lines[index+1].split()

        if len(raw_info) == 0:
            raise ValueError("No spin-orbit basis information found.")

        n_info = int(raw_info[0])
        if len(raw_info) != 1 + 2 * n_info:
            raise ValueError("Invalid spin-orbit basis information.")
        so_basis_info = []
        for l in range(max_ang+1):
            so_basis_info.append([l,0,0, basinfo[l][1]])
        for ii in range(n_info):
            l = int(raw_info[2*ii+1])
            nso = int(raw_info[2*ii+2])
            nshared = basinfo[l][1] - nso*2
            so_basis_info[l]=[l, nso, nso, nshared]
        return exps, coeffs, basinfo, so_basis_info
    return exps, coeffs, basinfo

def to_pyscf_kappa_basis(bas, so_info):
    so_basis = []
    for bas_n, so_n in zip(bas, so_info):
        assert(sum(so_n[1:]) == len(bas_n[1])-1)
        bas_np = np.asarray(bas_n[1:])
        for ikappa, nkappa in enumerate(so_n[1:]):
            offset = 1+sum(so_n[1:ikappa+1])
            if nkappa == 0:
                continue
            if ikappa == 0:
                kappa = so_n[0]
            elif ikappa == 1:
                kappa = -so_n[0]-1
            elif ikappa == 2:
                kappa = 0
            else:
                raise Exception("so basis information is problematic")
            kappa_basis = np.hstack((bas_np[:,[0]],bas_np[:,offset:offset+nkappa])).tolist()
            so_basis.append([so_n[0], kappa] + kappa_basis)
    return so_basis

def parse_genbas(basis, filename="GENBAS", uncontract=False, so_basis=False):
    if so_basis:
        exps, coeffs, basinfo, so_basis_info = read_genbas(basis, filename, so_basis)
    else:
        exps, coeffs, basinfo = read_genbas(basis, filename)
    basis_pyscf = []
    for ii in range(len(basinfo)):
        if uncontract:
            for jj in range(basinfo[ii][2]):
                basis_pyscf.append([ii,[exps[ii][jj],1.0]])
        else:
            basis_pyscf.append([basinfo[ii][0]])
            for jj in range(basinfo[ii][2]):
                basis_pyscf[ii].append([exps[ii][jj]])
                for kk in range(basinfo[ii][1]):
                    basis_pyscf[ii][jj+1].append(coeffs[ii][jj][kk])
    if so_basis:
        basis_pyscf = to_pyscf_kappa_basis(basis_pyscf, so_basis_info)
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
