import numpy

def _lowdin(s, c, label='orbitals'):
    """C -> C (C^H S C)^{-1/2}: minimal-change orthonormalization of a block."""
    if c.shape[1] == 0:
        return c
    m = c.conj().T @ s @ c
    w, v = numpy.linalg.eigh(m)
    if w[0] < 1e-8:
        raise numpy.linalg.LinAlgError(
            f'{label} are (near-)linearly dependent, '
            f'smallest metric eigenvalue {w[0]:.3e}')
    return c @ (v * w**-0.5) @ v.conj().T

def lowdin_ortho(mol, mo):
    """Symmetrically (Lowdin) orthonormalize the whole MO set at once."""
    return _lowdin(mol.intor('int1e_ovlp_spinor'), mo, 'fragment orbitals')

def blockwise_ortho(mol, mo, ncore, nact):
    """Orthonormalize core/active/virtual blocks in priority order.

    The active block is Lowdin-orthonormalized first and therefore keeps its
    shape exactly (no core/virtual pollution).  The core block is then
    projected against the active space and Lowdin-orthonormalized; the
    virtual block absorbs all remaining non-orthogonality last.  Within each
    block the symmetric (Lowdin) scheme keeps the orbitals maximally close
    to their input.
    """
    s = mol.intor('int1e_ovlp_spinor')
    out = mo.copy()
    slices = {'active': slice(ncore, ncore + nact),
              'core': slice(0, ncore),
              'virtual': slice(ncore + nact, mo.shape[1])}
    done = []
    for label in ('active', 'core', 'virtual'):
        c = out[:, slices[label]]
        for d in done:
            c = c - d @ (d.conj().T @ s @ c)
        c = _lowdin(s, c, label + ' orbitals')
        out[:, slices[label]] = c
        done.append(c)
    return out

def fragmo_guess(mol, mol_metal, mol_ligand,
                 mo_metal, mo_ligand,
                 nocc_metal, nocc_ligand,
                 nact_metal=0, ortho=True):
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
        ortho: orthonormalize the assembled orbitals w.r.t. the overlap of
            the complex (default True). The raw fragment orbitals are not
            mutually orthogonal; SCF tolerates this since diagonalizing the
            Fock matrix re-orthonormalizes, but MCSCF orbital rotations
            (exp of an antihermitian matrix) require an orthonormal basis.
            True or 'block': block-wise scheme, active first (kept free of
            core/virtual pollution), then core, then virtual.
            'lowdin': one global Lowdin over all orbitals at once.

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

    if ortho == 'lowdin':
        mo_guess = lowdin_ortho(mol, mo_guess)
    elif ortho:
        mo_guess = blockwise_ortho(mol, mo_guess,
                                   nocc_metal + nocc_ligand, nact_metal)

    return mo_guess
