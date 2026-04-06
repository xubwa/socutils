import numpy

def fragmo_guess(mol, mol_metal, mol_ligand,
                 mo_metal, mo_ligand,
                 nocc_metal, nocc_ligand,
                 nact_metal=0):
    """Construct a FragMO initial guess for a metal-ligand complex.

    Combines MO coefficients from a metal fragment and a ligand fragment
    into a full MO guess for the complex. The AO space of the complex is
    assumed to be the direct sum of the two fragments (metal AOs first,
    ligand AOs second).

    MO ordering in the output:
        metal_occ | ligand_occ | metal_act | metal_virt | ligand_virt

    Args:
        mol: the full complex molecule (pyscf.gto.Mole)
        mol_metal: the metal fragment molecule
        mol_ligand: the ligand fragment molecule
        mo_metal: MO coefficients of the metal fragment (nao_metal x nmo_metal)
        mo_ligand: MO coefficients of the ligand fragment (nao_ligand x nmo_ligand)
        nocc_metal: number of occupied orbitals in the metal fragment
        nocc_ligand: number of occupied orbitals in the ligand fragment
        nact_metal: number of active orbitals in the metal fragment (default 0)

    Returns:
        mo_guess: MO coefficient matrix for the complex (nao x nao, complex128)
    """
    nao = mol.nao_2c()
    ao_metal = mol_metal.nao_2c()

    mo_guess = numpy.zeros((nao, nao), dtype=complex)

    col = 0
    # metal occupied
    mo_guess[:ao_metal, col:col+nocc_metal] = mo_metal[:, :nocc_metal]
    col += nocc_metal

    # ligand occupied
    mo_guess[ao_metal:, col:col+nocc_ligand] = mo_ligand[:, :nocc_ligand]
    col += nocc_ligand

    # metal active
    mo_guess[:ao_metal, col:col+nact_metal] = mo_metal[:, nocc_metal:nocc_metal+nact_metal]
    col += nact_metal

    # metal virtual
    nvirt_metal = mo_metal.shape[1] - nocc_metal - nact_metal
    mo_guess[:ao_metal, col:col+nvirt_metal] = mo_metal[:, nocc_metal+nact_metal:]
    col += nvirt_metal

    # ligand virtual
    nvirt_ligand = mo_ligand.shape[1] - nocc_ligand
    mo_guess[ao_metal:, col:col+nvirt_ligand] = mo_ligand[:, nocc_ligand:]

    return mo_guess
