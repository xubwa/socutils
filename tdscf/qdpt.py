import numpy, scipy
from pyscf import lib

def transition_spin_density_t_0(xy_t, sz_t):
    assert sz_t in [0, 1, -1]
    norm = numpy.sum(xy_t**2)
    xy_t_normalized = xy_t / numpy.sqrt(norm)
    no, nv = xy_t.shape
    nmo = nv + no
    tdm = numpy.zeros((nmo, nmo))
    tdm[no:, :no] = xy_t_normalized.T
    if sz_t == 0:
        tdm *= numpy.sqrt(2)

    return tdm

def transition_spin_density_t_s(xy_t, xy_s, sz_t):
    assert sz_t in [0, 1, -1]
    norm = numpy.sum(xy_t**2)
    xy_t_normalized = xy_t / numpy.sqrt(norm)
    norm = numpy.sum(xy_s**2)
    xy_s_normalized = xy_s / numpy.sqrt(norm)
    no, nv = xy_t.shape
    nmo = nv + no
    tdm = numpy.zeros((nmo, nmo))
    
    sum_i = lib.einsum("ib,ia->ba", xy_t_normalized, xy_s_normalized)
    sum_a = lib.einsum("ja,ia->ij", xy_t_normalized, xy_s_normalized)
    tdm[no:, no:] = sum_i / numpy.sqrt(2)
    tdm[:no, :no] = -sum_a / numpy.sqrt(2)
    if sz_t == 0:
        tdm *= numpy.sqrt(2)

    return tdm

def transition_spin_density_t_t(xy_t1, xy_t2, sz_t1, sz_t2):
    if (sz_t1, sz_t2) in [(1, -1), (-1, 1), (0, 0)]:
        return numpy.zeros((xy_t1.shape[0] + xy_t1.shape[1],
                            xy_t1.shape[0] + xy_t1.shape[1]))
    elif (sz_t1, sz_t2) in [(0,1), (0,-1)]:
        return transition_spin_density_t_t(xy_t2, xy_t1, sz_t2, sz_t1).T
    assert (sz_t1, sz_t2) in [(1,1), (-1,-1), (1,0), (-1,0)]
    norm = numpy.sum(xy_t1**2)
    xy_t1_normalized = xy_t1 / numpy.sqrt(norm)
    norm = numpy.sum(xy_t2**2)
    xy_t2_normalized = xy_t2 / numpy.sqrt(norm)
    no, nv = xy_t1.shape
    nmo = nv + no
    tdm = numpy.zeros((nmo, nmo))

    sum_i = lib.einsum("ib,ia->ba", xy_t1_normalized, xy_t2_normalized)
    sum_a = lib.einsum("ja,ia->ij", xy_t1_normalized, xy_t2_normalized)
    if sz_t1 == 1 and sz_t2 == 1:
        tdm[no:, no:] = sum_i
        tdm[:no, :no] = sum_a
    elif sz_t1 == -1 and sz_t2 == -1:
        tdm[no:, no:] = -sum_i
        tdm[:no, :no] = -sum_a
    elif sz_t1 == 1 and sz_t2 == 0:
        tdm[no:, no:] = -sum_i / numpy.sqrt(2)
        tdm[:no, :no] = -sum_a / numpy.sqrt(2)
    elif sz_t1 == -1 and sz_t2 == 0:
        tdm[no:, no:] = sum_i / numpy.sqrt(2)
        tdm[:no, :no] = sum_a / numpy.sqrt(2)
        
    return tdm

def contract_soc_tdm(socints_mo, tdm, sz_diff):
    assert socints_mo.shape[0] == 3
    assert sz_diff in [0, 1, -1]
    if sz_diff == 0:
        res = numpy.trace(tdm.T @ socints_mo[2])
    elif sz_diff == 1:
        res = numpy.trace(tdm.T @ socints_mo[0] - 1j * tdm.T @ socints_mo[1])
    elif sz_diff == -1:
        res = numpy.trace(tdm.T @ socints_mo[0] + 1j * tdm.T @ socints_mo[1])
    return 1.0j * res

def qdpt(tda_singlet, tda_triplet, socints_mo):
    nroots_s = tda_singlet.nroots
    nroots_t = tda_triplet.nroots
    nroots_total = 1 + nroots_t*3 + nroots_s
    mat_qdpt = numpy.zeros((nroots_total, nroots_total), dtype=numpy.complex128)
    
    for i in range(nroots_t):
        for sz in range(-1, 2):
            pos_t = 1 + nroots_s + i*3 + (sz + 1)
            # <T|H_soc|0>
            tdm = transition_spin_density_t_0(tda_triplet.xy[i][0], sz)
            soc_me = contract_soc_tdm(socints_mo, tdm, sz)
            mat_qdpt[pos_t, 0] = soc_me
            mat_qdpt[0, pos_t] = numpy.conj(soc_me)

            # <T|H_soc|S>
            for j in range(nroots_s):
                pos_s = 1 + j
                tdm_ts = transition_spin_density_t_s(tda_triplet.xy[i][0], tda_singlet.xy[j][0], sz)
                soc_me_ts = contract_soc_tdm(socints_mo, tdm_ts, sz)
                mat_qdpt[pos_t, pos_s] = soc_me_ts
                mat_qdpt[pos_s, pos_t] = numpy.conj(soc_me_ts)

            # <T|H_soc|T>
            for j in range(i + 1):
                for sz2 in range(-1, 2):
                    if abs(sz - sz2) > 1:
                        continue
                    pos_t2 = 1 + nroots_s + j*3 + (sz2 + 1)
                    tdm_tt = transition_spin_density_t_t(tda_triplet.xy[i][0], tda_triplet.xy[j][0], sz, sz2)
                    soc_me_tt = contract_soc_tdm(socints_mo, tdm_tt, sz - sz2)
                    mat_qdpt[pos_t, pos_t2] = soc_me_tt
                    mat_qdpt[pos_t2, pos_t] = numpy.conj(soc_me_tt)
    
    for i in range(nroots_s):
        mat_qdpt[1+i, 1+i] += tda_singlet.e[i]
    for i in range(nroots_t):
        for sz in range(-1, 2):
            pos_t = 1 + nroots_s + i*3 + (sz + 1)
            mat_qdpt[pos_t, pos_t] += tda_triplet.e[i]
    assert numpy.allclose(mat_qdpt, mat_qdpt.conj().T)
    return scipy.linalg.eigh(mat_qdpt)
        
    


if __name__ == "__main__":
    from pyscf import gto, scf, dft
    from pyscf.tdscf.rks import TDA
    mol = gto.Mole()
    mol.atom = """
O      0.000000    0.000000    0.601105
C     -0.000000    0.000000   -0.598757
H      0.000000   -0.944973   -1.202781
H      0.000000    0.944973   -1.202781
    """
    mol.basis = "def2svp"
    mol.verbose = 4
    mol.build()

    from socutils.somf import somf_pt
    mf = dft.RKS(mol,xc="pbe,pbe")
    mf.kernel()

    # X2CAMF SOC
    # x2cints = somf_pt.get_psoc_x2camf(mol)
    # socints_mo = numpy.array([mf.mo_coeff.T @ x2cints[i] @ mf.mo_coeff for i in range(3)])

    # Mean-field Breit-Pauli SOC
    x2cints = somf_pt.get_soc_mf_bp(mf,mol)
    socints_mo = [numpy.zeros_like(mf.mo_coeff.T @ x2cints[0] @ mf.mo_coeff)]
    socints_mo = numpy.array([mf.mo_coeff.T @ x2cints[i] @ mf.mo_coeff for i in range(3)])
    

    s_tda = TDA(mf)
    s_tda.singlet = True
    s_tda.nroots = 5
    s_tda.kernel()
    
    t_tda = TDA(mf)
    t_tda.singlet = False
    t_tda.nroots = 5
    t_tda.kernel()

    e, c = qdpt(s_tda, t_tda, socints_mo)
    print("QDPT instability of ground state (cm-1):")
    print(e[0] * 219474.63)
    print("QDPT energies (cm-1):")
    print((e - e[0]) * 219474.63)
    