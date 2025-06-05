import x2camf
import numpy as np
import scipy
from functools import reduce
from pyscf import x2c, lib
from pyscf.x2c.x2c import _decontract_spinor
from pyscf.lib import chkfile
from pyscf.data import elements
from socutils import somf
from socutils.somf import x2c_grad
from socutils.scf import spinor_hf
from socutils.tools import spinor2sph
from pyscf import gto, scf
from x2camf.x2camf import construct_molecular_matrix, pcc_k, _amf
from x2camf import libx2camf

LIGHT_SPEED = lib.param.LIGHT_SPEED

def build_prim(mol):
    bas = mol._bas
    env = mol._env
    prim_vec = np.zeros(mol.nao_2c())
    aoloc = mol.ao_loc_2c()
    for i, ibas in enumerate(bas):
        assert ibas[2] == 1
        prim_vec[aoloc[i]:aoloc[i+1]] = env[ibas[5]]
    return prim_vec

THRESHOLD=1.0
CAP = 1e14
THRESH_PROD=0.
def screen_amf_matrix(xmol, soc_matrix):
    prim = build_prim(xmol)
    n2c = xmol.nao_2c()
    for i in range(n2c):
        for j in range(n2c):
            if prim[i] * prim[j] < THRESH_PROD or prim[j] * prim[i] < THRESH_PROD:
                soc_matrix[i,j] = 0.0
            if (prim[i] < THRESHOLD and prim[j] < CAP) or (prim[i] < CAP and prim[j] < THRESHOLD):
                soc_matrix[i,j] = 0.0
    return soc_matrix

def screen_amf4c_matrix(xmol, matrix_4c):
    prim = build_prim(xmol)
    n2c = xmol.nao_2c()
    for i in range(n2c):
        for j in range(n2c):
            #if prim[i] * prim[j] < THRESH_PROD:
            if prim[i] * prim[j] < THRESH_PROD or prim[j] * prim[i] < THRESH_PROD:
                matrix_4c[i,j] = 0.0
                matrix_4c[i+n2c,j] = 0.0
                matrix_4c[i,j+n2c] = 0.0
                matrix_4c[i+n2c, j+n2c] = 0.0
                if i==j:
                    matrix_4c[i,j] = 1.0
                    matrix_4c[i+n2c, j+n2c] = 1.0
            #if (prim[i] < THRESHOLD or prim[j] < THRESHOLD) and (prim[i] < CAP or prim[j] < CAP):
            if (prim[i] < THRESHOLD and prim[j] < CAP) or (prim[i] < CAP and prim[j] < THRESHOLD):
                matrix_4c[i,j] = 0.0
                matrix_4c[i+n2c,j] = 0.0
                matrix_4c[i,j+n2c] = 0.0
                matrix_4c[i+n2c, j+n2c] = 0.0
                if i==j:
                    matrix_4c[i,j] = 1.0
                    matrix_4c[i+n2c, j+n2c] = 1.0
                
    return matrix_4c

def x2c1e_hfw0_4cmat(h4c, m4c, mol=None):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]
    sLL = m4c[:n2c,:n2c]
    sSS = m4c[n2c:,n2c:]

    a, e, x, st, r, h2c, _, _ = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
    return x, st, r, h2c

def to_2c(x, r, h4c):
    n4c = h4c.shape[0]
    n2c = n4c//2
    hLL = h4c[:n2c,:n2c]
    hLS = h4c[:n2c,n2c:]
    hSL = h4c[n2c:,:n2c]
    hSS = h4c[n2c:,n2c:]

    l = hLL + np.dot(hLS, x) + np.dot(hLS, x).T.conj() + reduce(np.dot, (x.T.conj(), hSS, x))
    h2c = reduce(np.dot, (r.T.conj(), l, r))
    return h2c

def extract_ith_integral(atm_integrals, idx):
    out = {}
    for key, value in atm_integrals.items():
        out[key] = value[idx]
    return out

def x2cmp_screen(x2cobj, verbose=None, gaunt=False, breit=False, pcc=True, aoc=False, nucmod=None):
    mol = x2cobj.mol
    if nucmod is None:
        nucmod = mol.nucmod
    soc_int_flavor = 0
    print(gaunt, breit,aoc,pcc,nucmod)
    soc_int_flavor += gaunt << 0
    soc_int_flavor += breit << 1
    soc_int_flavor += nucmod << 2
    soc_int_flavor += aoc << 3
    soc_int_flavor += False << 4 # this parameter for spin dependant gaunt

    pcc_int_flavor = soc_int_flavor + 1 << 5

    uniq_atoms = set([a[0] for a in mol._atom])
    atm_ints = {}

    mol_ref = mol.copy()
    mol_ref.basis='dyallv2z'
    mol_ref.build()

    for atom in uniq_atoms:
        symbol = gto.mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = gto.mole.uncontracted_basis(mol._basis[atom])
        raw_bas_ref = gto.mole.uncontracted_basis(mol_ref._basis[atom])

        min_bas_ref = [0.05,0.05,0.1,0.2,0.4,0.6,0.8]
        #min_bas_ref = [-10.0,-10.0,-10.0,0.2,0.4,0.6,0.8]
        #min_bas_ref = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        for bas in raw_bas_ref:
            ang = bas[0]
            if min_bas_ref[ang] < 0.0:
                min_bas_ref[ang] = 1.*bas[-1][0]
            if bas[-1][0] < min_bas_ref[ang]:
                min_bas_ref[ang] = 1.*bas[-1][0]
        print(min_bas_ref)
        # screen raw_bas based on smallest primitive in each ang mom
        def to_prim(bas):
            prim = []
            for ibas in bas:
                ang_mom = ibas[0]
                prim = ibas[-1][0]
                prim.extend([ibas] * (ang_mom * 2 + 1))
            return np.array(prim)
        
        raw_bas_screened = []
        screen_or_not = []
        for bas in raw_bas:
            ang = bas[0]
            prim = bas[-1][0]
            if prim > min_bas_ref[ang] - 1e-5:
                raw_bas_screened.append(bas)
                screen_or_not.append([ang, 1.0])
            else:
                screen_or_not.append([ang, 0.0])

        screen_mask = []
        for iscreen in screen_or_not:
            ang = iscreen[0]
            screen_mask.extend([iscreen[1]] * 2 * (ang * 2 + 1))
        nonzero = np.count_nonzero(screen_mask)
        screen = np.ones(nonzero)
        n2c_screen = nonzero
        n2c = len(screen_mask)
        refill = np.zeros((n2c_screen, n2c))
        j = 0
        i = 0
        while i < n2c_screen:
            if screen_mask[j] == 1.0:
                refill[i,j] = 1.0
                j += 1
                i += 1
            else:
                j += 1
        print(f'Atom {atom} n2c {n2c} n2c_screen {n2c_screen}')
        shell = []
        exp_a = []
        for bas in raw_bas_screened:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = np.asarray(shell)
        exp_a = np.asarray(exp_a)
        nbas = shell.shape[0]
        nshell = shell[-1] + 1
        integrals = x2camf.libx2camf.atm_integrals(soc_int_flavor, atom_number, nshell, nbas, verbose, shell, exp_a)
        integrals.append(_amf(atom_number, shell, exp_a, pcc_int_flavor, 4))
        for i, int_i in enumerate(integrals):
            if int_i.shape[-1] == n2c_screen:
                int_i_refill = np.zeros((n2c, n2c))
                int_i_refill = reduce(np.dot, (refill.T, int_i, refill))
            elif int_i.shape[-1] == n2c_screen*2:
                int_i_refill = np.zeros((n2c*2, n2c*2))
                int_i_refill[:n2c,:n2c] = reduce(np.dot, (refill.T, int_i[:n2c_screen,:n2c_screen], refill))
                int_i_refill[:n2c,n2c:] = reduce(np.dot, (refill.T, int_i[:n2c_screen,n2c_screen:], refill))
                int_i_refill[n2c:,:n2c] = reduce(np.dot, (refill.T, int_i[n2c_screen:,:n2c_screen], refill))
                int_i_refill[n2c:,n2c:] = reduce(np.dot, (refill.T, int_i[n2c_screen:,n2c_screen:], refill))
            else:
                raise ValueError('Matrix neither 2c nor 4c')
            #if atom_number == 1:
            #    int_i_refill *= 0.0
            integrals[i] = int_i_refill
        atm_ints[atom] = integrals
    x2cobj.atomic_integrals = atm_ints

    xmol, _  = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    mf_4c = scf.DHF(xmol)
    mf_4c.with_gaunt = gaunt
    mf_4c.with_breit = breit
    h1e_4c = mf_4c.get_hcore()
    s4c = mf_4c.get_ovlp()
    print('amf_type', x2cobj.amf_type)
    x2cobj.h4c = h1e_4c
    x2cobj.m4c = s4c
    x2cobj.soc_matrix = np.zeros((n2c, n2c))
    density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
    x2cobj.density_2c = density_2c

    if x2cobj.amf_type == 'dirac_amf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_fock_4c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_h1e  = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        atm_so2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        atm_so4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 9), atom_slices, xmol, n2c, True)
        atm_X = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        n2c = atm_fock_4c2e.shape[0]//2
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c+atm_fock_4c2e, s4c, mol=xmol)
        h_x2c = to_2c(x, r, h1e_4c)
        soc_matrix = construct_molecular_matrix(extract_ith_integral(atm_ints, 13), atom_slices, xmol, n2c, False)
        h_x2c += soc_matrix
        x2cobj.soc_matrix = soc_matrix
    elif 'screenH' in x2cobj.amf_type:
        '''
        screen the matrix element of H4c.
        {i} denotes the set of basis functions that are not diffuse.
        {j} denotes the set of basis functions that are diffuse.
        keep H_ii block and H_jj block of the H4c.
        H_ij block is set to 0.
        '''
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_h4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        n2c = atm_h4c.shape[0]//2

        atom_slices = xmol.offset_2c_by_atom()
        n2c = xmol.nao_2c()
        x = np.zeros((n2c,n2c), dtype=complex)
        r = np.zeros((n2c,n2c), dtype=complex)
        atm_h4c = np.zeros_like(atm_h4c, dtype=complex)
        atm_s4c = np.zeros_like(atm_h4c, dtype=complex)

        for ia in range(xmol.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
            t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
            with xmol.with_rinv_at_nucleus(ia):
                z = -xmol.atom_charge(ia)
                v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                if 'sf' in x2cobj.amf_type:
                    w1 = z*xmol.intor('int1e_prinvp_spinor', shls_slice=shls_slice)
                else:
                    w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
            atm_h4c[p0:p1, n2c+p0:n2c+p1] = t1
            atm_h4c[n2c+p0:n2c+p1, p0:p1] = t1
            atm_h4c[p0:p1, p0:p1] = v1
            atm_h4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1] = w1*(0.5/LIGHT_SPEED)**2 - t1
            atm_s4c[p0:p1,p0:p1] = s1
            atm_s4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1] = t1*0.5/(LIGHT_SPEED)**2

        ao_loc = xmol.ao_loc_2c()
        nbas = xmol.nbas
        exps = xmol.bas_exps()
        atm_h4c_bkp = atm_h4c.copy()
        atm_s4c_bkp = atm_s4c.copy()
        for ishl in range(nbas):
            p0, p1 = ao_loc[ishl], ao_loc[ishl+1]
            angular = xmol.bas_angular(ishl)
            exp = exps[ishl][0]
            # since xmol is decontracted, exp[0] is its only exponent
            if exp < min_bas_ref[angular]:
                atm_h4c[p0:p1,:] *= 0.0j
                atm_h4c[:,p0:p1] *= 0.0j
                atm_h4c[p0+n2c:p1+n2c,:]*=0.0j
                atm_h4c[:,p0+n2c:p1+n2c]*=0.0j
                atm_s4c[p0:p1,:] *= 0.0j
                atm_s4c[:,p0:p1] *= 0.0j
                atm_s4c[p0+n2c:p1+n2c,:]*=0.0j
                atm_s4c[:,p0+n2c:p1+n2c]*=0.0j
                #atm_h4c[p0+n2c:p1+n2c,p0+n2c:p1+n2c] = np.diag(np.arange(p0,p1))*10.0-2.*LIGHT_SPEED**2+0.j
                #atm_h4c[p0:p1,p0:p1] = np.diag(np.arange(p0,p1))*10.0+0.j
                #atm_s4c[p0:p1,p0:p1] = np.diag(np.ones(p1-p0))+0.j
                #atm_s4c[p0+n2c:p1+n2c,p0+n2c:p1+n2c] = atm_s4c[p0:p1,p0:p1]
        for ishl in range(nbas):
            p0, p1 = ao_loc[ishl], ao_loc[ishl+1]
            angular = xmol.bas_angular(ishl)
            exp = exps[ishl][0] # since xmol is decontracted, exp[0] is its only exponent
            if exp < min_bas_ref[angular]:
                atm_hll_bkp = atm_h4c_bkp[p0:p1,p0:p1]
                atm_hss_bkp = atm_h4c_bkp[p0+n2c:p1+n2c,p0+n2c:p1+n2c]
                atm_hls_bkp = atm_h4c_bkp[p0:p1,p0+n2c:p1+n2c]
                atm_hsl_bkp = atm_h4c_bkp[p0+n2c:p1+n2c,p0:p1]
                atm_sll_bkp = atm_s4c_bkp[p0:p1,p0:p1]
                atm_sss_bkp = atm_s4c_bkp[p0+n2c:p1+n2c,p0+n2c:p1+n2c]
                atm_h4c[p0:p1,p0:p1] = atm_hll_bkp
                atm_h4c[p0+n2c:p1+n2c,p0+n2c:p1+n2c] = atm_hss_bkp
                atm_h4c[p0:p1,p0+n2c:p1+n2c] = atm_hls_bkp
                atm_h4c[p0+n2c:p1+n2c,p0:p1] = atm_hsl_bkp
                atm_s4c[p0:p1,p0:p1] = atm_sll_bkp
                atm_s4c[p0+n2c:p1+n2c,p0+n2c:p1+n2c] = atm_sss_bkp
            
        for ia in range(xmol.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            hls = atm_h4c[p0:p1, n2c+p0:n2c+p1]
            hsl = atm_h4c[n2c+p0:n2c+p1, p0:p1]
            hll = atm_h4c[p0:p1, p0:p1]
            hss = atm_h4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1]
            sll = atm_s4c[p0:p1,p0:p1]
            sss = atm_s4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1]
            n2c_atm = p1-p0
            n4c_atm = 2*n2c_atm
            h4c_atm = np.zeros((n4c_atm,n4c_atm), dtype=complex)
            h4c_atm[:n2c_atm,:n2c_atm] = hll
            h4c_atm[n2c_atm:,:n2c_atm] = hsl
            h4c_atm[:n2c_atm,n2c_atm:] = hls
            h4c_atm[n2c_atm:,n2c_atm:] = hss
            s4c_atm = np.zeros_like(h4c_atm)
            s4c_atm[:n2c_atm,:n2c_atm] = sll
            s4c_atm[n2c_atm:,n2c_atm:] = sss
            x1, st1, r1, h2c1 = x2c1e_hfw0_4cmat(h4c_atm, s4c_atm, xmol)
            x[p0:p1,p0:p1] = x1 
            r[p0:p1,p0:p1] = r1

        for ishl in range(nbas):
            p0, p1 = ao_loc[ishl], ao_loc[ishl+1]
            angular = xmol.bas_angular(ishl)
            exp = exps[ishl][0] # since xmol is decontracted, exp[0] is its only exponent
            if exp < min_bas_ref[angular]:
               print(x[p0:p1,p0:p1])
               print(atm_h4c[p0:p1,p0:p1])
               print(atm_h4c[p0+n2c:p1+n2c,p0+n2c:p1+n2c])
               print(atm_s4c[p0:p1,p0:p1])
               x[p0:p1,:] *= 0.0
               x[:,p0:p1] *= 0.0
               x[p0:p1,p0:p1] = np.eye(p1-p0)+0.j
               r[p0:p1,p0:p1] = np.eye(p1-p0)+0.j

        if 'sf' in x2cobj.amf_type:
            t = mol.intor('int1e_spsp_spinor') * 0.5
            vn = mol.intor('int1e_nuc_spinor')
            wn = mol.intor('int1e_pnucp_spinor')
            n2c = mol.nao_2c()
            n4c = n2c * 2
            h1e_4c = np.empty((n4c, n4c), np.complex128)
            h1e_4c[:n2c,:n2c] = vn
            h1e_4c[n2c:,:n2c] = t
            h1e_4c[:n2c,n2c:] = t
            h1e_4c[n2c:,n2c:] = wn * (.25/LIGHT_SPEED**2) - t
        if 'ax' in x2cobj.amf_type:
            r = x2cobj._get_rmat(x)
        h_x2c = to_2c(x, r, h1e_4c)
        
    elif 'screenX' in x2cobj.amf_type:
        '''
        In this scheme, X matrix is screened base on exponentials,
        atomic X approximation is employed.
        '''
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_h4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        n2c = atm_h4c.shape[0]//2

        atom_slices = xmol.offset_2c_by_atom()
        n2c = xmol.nao_2c()
        x = np.zeros((n2c,n2c), dtype=np.complex128)
        atm_h4c = np.zeros_like(atm_h4c, dtype=complex)
        atm_s4c = np.zeros_like(atm_h4c, dtype=complex)

        t = mol.intor('int1e_spsp_spinor') * 0.5
        vn = mol.intor('int1e_nuc_spinor')
        if 'sf' in x2cobj.amf_type:
            wn = mol.intor('int1e_pnucp_spinor')
        else:
            wn = mol.intor('int1e_sprinvsp_spinor')
        n2c = mol.nao_2c()
        n4c = n2c * 2
        h1e_4c = np.empty((n4c, n4c), np.complex128)
        h1e_4c[:n2c,:n2c] = vn
        h1e_4c[n2c:,:n2c] = t
        h1e_4c[:n2c,n2c:] = t
        h1e_4c[n2c:,n2c:] = wn * (.25/LIGHT_SPEED**2) - t

        for ia in range(xmol.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
            t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
            with xmol.with_rinv_at_nucleus(ia):
                z = -xmol.atom_charge(ia)
                v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                if 'sf' in x2cobj.amf_type:
                    w1 = z*xmol.intor('int1e_prinvp_spinor', shls_slice=shls_slice)
                else:
                    w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
            atm_h4c[p0:p1, n2c+p0:n2c+p1] = t1
            atm_h4c[n2c+p0:n2c+p1, p0:p1] = t1
            atm_h4c[p0:p1, p0:p1] = v1
            atm_h4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1] = w1*(0.5/LIGHT_SPEED)**2 - t1
            atm_s4c[p0:p1,p0:p1] = s1
            atm_s4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1] = t1*0.5/(LIGHT_SPEED)**2

        ao_loc = xmol.ao_loc_2c()
        nbas = xmol.nbas
        exps = xmol.bas_exps()
            
        for ia in range(xmol.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            hls = atm_h4c[p0:p1, n2c+p0:n2c+p1]
            hsl = atm_h4c[n2c+p0:n2c+p1, p0:p1]
            hll = atm_h4c[p0:p1, p0:p1]
            hss = atm_h4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1]
            sll = atm_s4c[p0:p1,p0:p1]
            sss = atm_s4c[n2c+p0:n2c+p1,n2c+p0:n2c+p1]
            n2c_atm = p1-p0
            n4c_atm = 2*n2c_atm
            h4c_atm = np.zeros((n4c_atm,n4c_atm), dtype=complex)
            h4c_atm[:n2c_atm,:n2c_atm] = hll
            h4c_atm[n2c_atm:,:n2c_atm] = hsl
            h4c_atm[:n2c_atm,n2c_atm:] = hls
            h4c_atm[n2c_atm:,n2c_atm:] = hss
            s4c_atm = np.zeros_like(h4c_atm)
            s4c_atm[:n2c_atm,:n2c_atm] = sll
            s4c_atm[n2c_atm:,n2c_atm:] = sss
            x1, st1, r1, h2c1 = x2c1e_hfw0_4cmat(h4c_atm, s4c_atm, xmol)
            x[p0:p1,p0:p1] = x1 

        for ishl in range(nbas):
            p0, p1 = ao_loc[ishl], ao_loc[ishl+1]
            angular = xmol.bas_angular(ishl)
            exp = exps[ishl][0] # since xmol is decontracted, exp[0] is its only exponent
            if exp < min_bas_ref[angular]:
               #x[p0:p1,:] *= 0.0
               x[:,p0:p1] *= 0.0
               x[p0:p1,p0:p1] = np.eye(p1-p0)+0.j
        r = x2cobj._get_rmat(x)
        h_x2c = to_2c(x, r, h1e_4c)
    return h_x2c

def x2cmp(x2cobj, verbose=None, gaunt=False, breit=False, pcc=True, aoc=False, nucmod=None):
    mol = x2cobj.mol
    if nucmod is None:
        nucmod = mol.nucmod
    soc_int_flavor = 0
    print(gaunt, breit,aoc,pcc,nucmod)
    soc_int_flavor += gaunt << 0
    soc_int_flavor += breit << 1
    soc_int_flavor += nucmod << 2
    soc_int_flavor += aoc << 3
    soc_int_flavor += False << 4 # this parameter for spin dependant gaunt

    uniq_atoms = set([a[0] for a in mol._atom])
    atm_ints = {}

    for atom in uniq_atoms:
        symbol = gto.mole._std_symbol(atom)
        atom_number = elements.charge(symbol)
        raw_bas = gto.mole.uncontracted_basis(mol._basis[atom])
        shell = []
        exp_a = []
        for bas in raw_bas:
            shell.append(bas[0])
            exp_a.append(bas[-1][0])
        shell = np.array(shell)
        exp_a = np.array(exp_a)
        nbas = shell.shape[0]
        nshell = shell[-1] + 1
        atm_ints[atom] = x2camf.libx2camf.atm_integrals(soc_int_flavor, atom_number, nshell, nbas, verbose, shell, exp_a)
    x2cobj.atomic_integrals = atm_ints
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};

    xmol, _  = x2cobj.get_xmol()
    n2c = xmol.nao_2c()
    atom_slices = xmol.aoslice_2c_by_atom()
    mf_4c = scf.DHF(xmol)
    mf_4c.with_gaunt = gaunt
    mf_4c.with_breit = breit
    h1e_4c = mf_4c.get_hcore()
    s4c = mf_4c.get_ovlp()
    print('amf_type', x2cobj.amf_type)
    x2cobj.h4c = h1e_4c
    x2cobj.m4c = s4c
    x2cobj.soc_matrix = np.zeros((n2c, n2c))
    density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
    x2cobj.density_2c = density_2c
    if x2cobj.amf_type == 'x2cmp':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        atomic_h1e_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        atomic_fock_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 3), atom_slices, xmol, n2c, True)
        atomic_fock_4c_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atomic_fock_2c_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        atomic_fock_2c2e_fw = to_2c(atm_x, atm_r, atomic_fock_4c_2e)
        density_4c_ss = density_4c.copy()
        ene_corr = np.einsum('ij,ji->', atomic_fock_4c_2e, density_4c).real-np.einsum('ij,ji->',atomic_fock_2c_2e, density_2c).real
        ene_corr2 = np.einsum('ij,ji->',atomic_fock_2c2e_fw - atomic_fock_2c_2e, density_2c).real
        print(ene_corr*0.5, ene_corr2*0.5)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c - vk_4c
        fock_4c = h1e_4c + veff_4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, s4c, xmol)
        mf_2c = spinor_hf.SCF(xmol)
        veff_2c = mf_2c.get_veff(dm=density_2c)
        x2cobj.veff_2c=veff_2c
        h_x2c = to_2c(x, r, fock_4c) - veff_2c
        x2cobj.h4c = h1e_4c + veff_4c
        x2cobj.soc_matrix = h_x2c - to_2c(x, r, h1e_4c)
    elif x2cobj.amf_type == 'x2cmp_x1e':
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c - vk_4c
        fock_4c = h1e_4c + veff_4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, xmol)
        mf_2c = scf.X2C(xmol)
        veff_2c = mf_2c.get_veff(dm=density_2c)
        #x2cobj.veff_2c=veff_2c
        heff = to_2c(x, r, h1e_4c) - veff_2c
        h1e_x2c = to_2c(x, r, h1e_4c)
        x2cobj.h4c = h1e_4c
        x2cobj.soc_matrix = to_2c(x, r, veff_4c) - veff_2c
        h_x2c = h1e_x2c + x2cobj.soc_matrix
    elif 'aimp_1x' in x2cobj.amf_type:
        print('aimp_1x')
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        density_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 12), atom_slices, xmol, n2c, False)
        k1c_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 7), atom_slices, xmol, n2c, True)
        k1c_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 8), atom_slices, xmol, n2c, False)
        atm_X = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        vj_4c, vk_4c= mf_4c.get_jk(dm=density_4c)
        veff_4c = vj_4c
        fock_4c = h1e_4c + veff_4c
        mf_2c = scf.X2C(xmol)
        vj_2c, vk_2c = mf_2c.get_jk(dm=density_2c)
        veff_2c = vj_2c
        x, st, r, h2c = x2c1e_hfw0_4cmat(fock_4c, s4c, xmol)
        if x2cobj.amf_type=='aimp_1x_0':
            h_x2c = to_2c(x, r, fock_4c + k1c_4c) - veff_2c - k1c_2c #+ x2camf.x2camf.pcc_k(x2cobj, with_gaunt=False, with_gauge=False)
        elif x2cobj.amf_type=='aimp_1x_1':
            r = x2cobj._get_rmat(atm_X)
            h_x2c = to_2c(atm_X, r, fock_4c + k1c_4c) - veff_2c - k1c_2c
        elif x2cobj.amf_type=='aimp_1x_2':
            heff = to_2c(x, r, fock_4c) - veff_2c - k1c_2c
            atm_r = x2cobj._get_rmat(atm_X)
            h_x2c = heff + to_2c(atm_X, atm_r, k1c_4c)
        elif x2cobj.amf_type=='aimp_1x_3':
            h_x2c = to_2c(x, r, fock_4c) - veff_2c + pcc_k(x2cobj, with_gaunt=gaunt, with_gauge=breit)
        print('return here')
    elif x2cobj.amf_type == 'dirac_amf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_fock_4c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_h1e  = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        atm_so2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        atm_so4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 9), atom_slices, xmol, n2c, True)
        atm_X = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        density_4c = construct_molecular_matrix(extract_ith_integral(atm_ints, 11), atom_slices, xmol, n2c, True)
        n2c = atm_fock_4c2e.shape[0]//2
        atm_fock_4c2e_screen = screen_amf4c_matrix(xmol, atm_fock_4c2e)
        atm_fock_2c2e_screen = screen_amf_matrix(xmol, atm_fock_2c2e)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c+atm_fock_4c2e_screen, s4c, mol=xmol)
        h_x2c = to_2c(x, r, h1e_4c)
        soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=pcc, gaussian_nuclear=x2cobj.gau_nuc)
        h_x2c += soc_matrix
        x2cobj.soc_matrix = soc_matrix
    elif x2cobj.amf_type == 'cq_amf':
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=pcc, gaussian_nuclear=x2cobj.gau_nuc)
        soc_matrix = screen_amf_matrix(xmol, soc_matrix)
        x2cobj.soc_matrix = soc_matrix
        h_x2c = to_2c(x, r, h1e_4c) + soc_matrix
    elif x2cobj.amf_type == '1e_amf':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_fock_4c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        atm_h1e  = construct_molecular_matrix(extract_ith_integral(atm_ints, 2), atom_slices, xmol, n2c, True)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        h_x2c = to_2c(x, r, h1e_4c + atm_fock_4c2e) - atm_fock_2c2e
    elif x2cobj.amf_type == 'atm_fock':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        atm_fock_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        from pyscf.x2c.x2c import _get_hcore_fw
        from pyscf.lib.parameters import LIGHT_SPEED
        print('atm_fock')
        mf_4c = scf.DHF(xmol)
        r = x2cobj._get_rmat(atm_x)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        hx2c_1e = to_2c(atm_x, atm_r, h1e_4c)
        h_x2c = hx2c_1e + to_2c(atm_x, r, atm_fock_2e) - atm_fock_2c2e
    elif x2cobj.amf_type == 'atm_x2e':
        # results{0 atm_X, 1 atm_R, 2 h1e_4c, 3 fock_4c, 4 fock_2c, 5 fock_4c_2e, 
        #         6 fock_2c_2e, 7 fock_4c_K, 8 fock_2c_K, 9 so_4c, 10 so_2c, 11 den_4c, 12 den_2c};
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        atm_fock_2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 5), atom_slices, xmol, n2c, True)
        atm_fock_2c2e = construct_molecular_matrix(extract_ith_integral(atm_ints, 6), atom_slices, xmol, n2c, False)
        #soc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=x2cobj.gaunt, with_gauge=x2cobj.breit,
        #             with_gaunt_sd=x2cobj.gaunt_sd, aoc=True, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc)
        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        from pyscf.x2c.x2c import _get_hcore_fw
        from pyscf.lib.parameters import LIGHT_SPEED
        print('atm_x1e')
        mf_4c = scf.DHF(xmol)
        r = x2cobj._get_rmat(atm_x)
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        hx2c_1e = to_2c(atm_x, r, h1e_4c)
        pcc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=x2cobj.pcc, gaussian_nuclear=x2cobj.gau_nuc)
        x2cobj.soc_matrix = pcc_matrix
        h_x2c = hx2c_1e + pcc_matrix
    elif x2cobj.amf_type == 'amfx2c_a1e':
        atom_slices = xmol.offset_2c_by_atom()
        n2c = xmol.nao_2c()
        x = np.zeros((n2c,n2c), dtype=np.complex128)
        from pyscf.lib.parameters import LIGHT_SPEED as c
        for ia in range(xmol.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            s1 = xmol.intor('int1e_ovlp_spinor', shls_slice=shls_slice)
            t1 = xmol.intor('int1e_spsp_spinor', shls_slice=shls_slice) * .5
            with xmol.with_rinv_at_nucleus(ia):
                z = -xmol.atom_charge(ia)
                v1 = z*xmol.intor('int1e_rinv_spinor', shls_slice=shls_slice)
                w1 = z*xmol.intor('int1e_sprinvsp_spinor', shls_slice=shls_slice)
            x[p0:p1,p0:p1] = x2c.x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
        r = x2cobj._get_rmat(x)
        h1 = to_2c(x, r, h1e_4c)
        pcc_matrix = x2camf.amfi(x2cobj, printLevel=x2cobj.verbose, with_gaunt=gaunt, with_gauge=breit,
                     with_gaunt_sd=x2cobj.gaunt_sd, aoc=aoc, pcc=True, gaussian_nuclear=x2cobj.gau_nuc)
        return h1 + pcc_matrix
    elif x2cobj.amf_type == 'x2camf':
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        prim = build_prim(xmol)
        x2cobj.soc_matrix = so_2c
        h_x2c = to_2c(x, r, h1e_4c) + so_2c
    elif x2cobj.amf_type == 'x2camf_sd':
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        so_2c = spinor2sph.spinor2spinor_sd(xmol, so_2c)
        x2cobj.soc_matrix = so_2c
        h_x2c = to_2c(x, r, h1e_4c) + so_2c
    elif x2cobj.amf_type == '1e':
        x2cobj.h4c = h1e_4c
        x2cobj.m4c = s4c
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4c, s4c, mol=xmol)
        h_x2c = to_2c(x, r, h1e_4c)
    elif x2cobj.amf_type == 'sf1e':
        from pyscf.lib.parameters import LIGHT_SPEED as c
        t = xmol.intor('int1e_spsp_spinor') * 0.5
        vn = xmol.intor('int1e_nuc_spinor')
        wn = xmol.intor('int1e_pnucp_spinor')
        n2c = xmol.nao_2c()
        n4c = n2c * 2
        h1e_4csf = np.empty((n4c, n4c), np.complex128)
        h1e_4csf[:n2c,:n2c] = vn
        h1e_4csf[n2c:,:n2c] = t
        h1e_4csf[:n2c,n2c:] = t
        h1e_4csf[n2c:,n2c:] = wn * (.25/c**2) - t
        x, st, r, h2c = x2c1e_hfw0_4cmat(h1e_4csf, s4c, mol=xmol)
        x2cobj.h4c = h1e_4csf
        x2cobj.m4c = s4c
        h_x2c = to_2c(x, r, h1e_4csf)
    elif x2cobj.amf_type == 'x2camf_axr':
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        r = x2cobj._get_rmat(atm_x)
        x2cobj.soc_matrix = so_2c
        h_x2c = to_2c(atm_x, r, h1e_4c) + so_2c
    elif x2cobj.amf_type == 'x2camf_au':
        atm_x = construct_molecular_matrix(extract_ith_integral(atm_ints, 0), atom_slices, xmol, n2c, False)
        atm_r = construct_molecular_matrix(extract_ith_integral(atm_ints, 1), atom_slices, xmol, n2c, False)
        so_2c = construct_molecular_matrix(extract_ith_integral(atm_ints, 10), atom_slices, xmol, n2c, False)
        x2cobj.soc_matrix = so_2c
        h_x2c = to_2c(atm_x, atm_r, h1e_4c) + so_2c
    
    # convert to contracted basis
    # x2cobj.soc_matrix and h_x2c
    # x2cobj.h4c and x2cobj.m4c shall remain in uncontracted form

    xmol, contr_coeff_nr = x2cobj.get_xmol()
    xmol, contr_coeff = _decontract_spinor(mol, x2cobj.xuncontract)
    #nprim, ncontr = contr_coeff_nr.shape
    #contr_coeff = np.zeros((nprim * 2, ncontr * 2))
    #contr_coeff[0::2, 0::2] = contr_coeff_nr
    #contr_coeff[1::2, 1::2] = contr_coeff_nr
    x2cobj.soc_matrix = reduce(np.dot, (contr_coeff.T.conj(), x2cobj.soc_matrix, contr_coeff))
    h_x2c = reduce(np.dot, (contr_coeff.T.conj(), h_x2c, contr_coeff))
    return h_x2c

class SpinorX2CMPHelper(x2c.x2c.SpinorX2CHelper):
    hcore = None
    def __init__(self, mol, x2cmp='x2cmp', with_gaunt=False, with_breit=False, with_pcc=True, with_aoc=False, screen=False):
        super().__init__(mol)
        self.x2c_intermediate = None
        self.gaunt = with_gaunt
        self.gaunt_sd = False        
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        self.amf_type = x2cmp # use this class to implement various flavors of approximations.
        self.nucmod = mol.nucmod
        self.h4c = None
        self.m4c = None
        self.veff_2c = None
        self.soc_matrix = None
        self.screen = screen
        if self.nucmod != {}:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        
    def x2cmp(self):
        print(self.gaunt,self.breit,self.pcc,self.aoc)
        return x2cmp(self, self.mol.verbose, self.gaunt, self.breit, self.pcc, self.aoc, self.gau_nuc)
    
    def x2cmp_screen(self):
        print(self.gaunt,self.breit,self.pcc,self.aoc)
        xmol, contr_coeff_nr = self.get_xmol()
        npri, ncon = contr_coeff_nr.shape
        contr_coeff = np.zeros((npri*2,ncon*2))
        contr_coeff[0::2,0::2] = contr_coeff_nr
        contr_coeff[1::2,1::2] = contr_coeff_nr
        x2cmp_unc = x2cmp_screen(self, self.mol.verbose, self.gaunt, self.breit, self.pcc, self.aoc, self.gau_nuc)
        return reduce(np.dot, (contr_coeff.T, x2cmp_unc, contr_coeff))
    
    def get_hcore(self, mol):
        if self.hcore is None:
            if self.screen is False:
                self.hcore = self.x2cmp()
            else:
                self.hcore = self.x2cmp_screen()
        return self.hcore

    def save_hcore(self, filename='x2cmp.chk'):
        if self.hcore is None:
            chkfile.dump(filename, 'x2cmp_integral', self.x2cmp())
        else:
            chkfile.dump(filename, 'x2cmp_integral', self.hcore())
        if self.h4c is not None:
            chkfile.dump(filename, 'h4c', self.h4c)
        if self.m4c is not None:
            chkfile.dump(filename, 'm4c', self.m4c)
        if self.soc_matrix is not None:
            chkfile.dump(filename, 'soc_integral', self.soc_matrix)

    def load_hcore(self, filename='x2cmp.chk'):
        try:
            self.hcore = chkfile.load(filename, 'x2cmp_integral')
        except:
            raise ValueError('No x2cmp integral found in the chkfile')
        try:
            self.h4c = chkfile.load(filename, 'h4c')
        except:
            self.h4c = None
        try:
            self.m4c = chkfile.load(filename, 'm4c')
        except:
            self.m4c = None
        try:
            self.soc_matrix = chkfile.load(filename, 'soc_integral')
        except:
            self.soc_matrix = None

    def get_soc_integrals(self):
        return self.soc_matrix

    def get_hfw1(self, h4c1, s4c1=None, x_response=True):
        if self.h4c is None:
            self.get_hcore(self.mol)
        n4c = self.h4c.shape[0]
        n2c = n4c//2
        hLL = self.h4c[:n2c,:n2c]
        hLS = self.h4c[:n2c,n2c:]
        hSL = self.h4c[n2c:,:n2c]
        hSS = self.h4c[n2c:,n2c:]
        sLL = self.m4c[:n2c,:n2c]
        sSS = self.m4c[n2c:,n2c:]

        if self.x2c_intermediate is None:
            a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
            self.x2c_intermediate={'a':a,'e':e,'x':x,'st':st,'r':r,'l':l}
        else:
            a = self.x2c_intermediate['a']
            e = self.x2c_intermediate['e']
            x = self.x2c_intermediate['x']
            st = self.x2c_intermediate['st']
            r = self.x2c_intermediate['r']
            l = self.x2c_intermediate['l']
            h4c = self.h4c
            m4c = self.m4c
        if x_response is True:
            hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)
        else:
            hfw1 = to_2c(x, r, h4c1)
        return hfw1
    
    def get_xr(self):
        if self.h4c is None:
            self.get_hcore(self.mol)
        h4c = self.h4c
        s4c = self.m4c
        xmol = self.get_xmol()
        x, st, r, h2c = x2c1e_hfw0_4cmat(h4c, s4c, xmol)
        return x, r

    def to_4c_coeff(self, c2c):
        x, r = self.get_xr()
        cl = np.dot(r, c2c)
        cs = np.dot(x, cl)
        n2c = c2c.shape[0]
        dtype = c2c.dtype
        c4c = np.zeros((n2c*2,n2c*2), dtype=dtype)
        c4c[:n2c, n2c:] = cl
        c4c[n2c:, n2c:] = cs
        return c4c

class SpinOrbitalX2CMPHelper(x2c.x2c.SpinOrbitalX2CHelper):
    hcore = None
    def __init__(self, mol, x2cmp='x2cmp', with_gaunt=False, with_breit=False, with_pcc=True, with_aoc=False):
        super().__init__(mol)
        self.x2c_intermediate = None
        self.gaunt = with_gaunt
        self.gaunt_sd = False        
        self.breit = with_breit
        self.pcc = with_pcc
        self.aoc = with_aoc
        self.amf_type = x2cmp # use this class to implement various flavors of approximations.
        self.nucmod = mol.nucmod
        self.h4c = None
        self.m4c = None
        self.veff_2c = None
        self.soc_matrix = None
        if self.nucmod != {}:
            self.gau_nuc = True
        else:
            self.gau_nuc = False
        
    def x2cmp(self):
        x2cmp_spinor = x2cmp(self, self.mol.verbose, self.gaunt, self.breit, self.pcc, self.aoc, self.gau_nuc)
        return spinor2sph.spinor2sph(self.mol, x2cmp_spinor)
    
    def get_hcore(self, mol):
        if self.hcore is None:
            self.hcore = self.x2cmp()
        return self.hcore

    def save_hcore(self, filename='x2cmp.chk'):
        if self.hcore is None:
            chkfile.dump(filename, 'x2cmp_integral', self.x2cmp())
        else:
            chkfile.dump(filename, 'x2cmp_integral', self.hcore())
        if self.h4c is not None:
            chkfile.dump(filename, 'h4c', self.h4c)
        if self.m4c is not None:
            chkfile.dump(filename, 'm4c', self.m4c)
        if self.soc_matrix is not None:
            chkfile.dump(filename, 'soc_integral', self.soc_matrix)


    def load_hcore(self, filename='x2cmp.chk'):
        self.hcore = chkfile.load(filename, 'x2cmp_integral')

    def get_hfw1(self, h4c1, s4c1=None, x_response=True):
        if self.h4c is None:
            self.get_hcore(self.mol)
        n4c = self.h4c.shape[0]
        n2c = n4c//2
        hLL = self.h4c[:n2c,:n2c]
        hLS = self.h4c[:n2c,n2c:]
        hSL = self.h4c[n2c:,:n2c]
        hSS = self.h4c[n2c:,n2c:]
        sLL = self.m4c[:n2c,:n2c]
        sSS = self.m4c[n2c:,n2c:]

        a, e, x, st, r, l, h4c, m4c = x2c_grad.x2c1e_hfw0_block(hLL, hSL, hLS, hSS, sLL, sSS)
        if x_response is True:
            hfw1 = x2c_grad.get_hfw1(a, x, st, m4c, h4c, e, r, l, h4c1, s4c1)
        else:
            hfw1 = to_2c(x, r, h4c1)
        return hfw1

X2CMP = SpinorX2CMPHelper

if __name__ == '__main__':
    mol = gto.M(atom = 'Ne 0 0 0; Ar 0 0 1.8', basis='uncccpvdz', verbose=5)
    x2cobj = SpinorEAMFX2CHelper(mol, with_gaunt=True, with_breit=True)
    from socutils.scf import x2camf_hf
    mf = x2camf_hf.SCF(mol)
    mf.with_x2c=x2cobj
    mf.kernel()
    print(mf.mo_energy)
