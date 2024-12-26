from pyscf import scf, gto
from socutils.grad import ghf_grad # for GHF.Gradients

def mfobj(dx):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [["O", (0.0, 0.0, 0.0)],
                ["H", (0.0, 1.0, 1.0+dx)],
                ["H", (0.0, -1.0, 1.0)]]
    mol.basis = "cc-pvdz"
    mol.charge = 0
    mol.spin = 0
    mol.unit = "Bohr"
    mol.build()

    mf = scf.GHF(mol).x2c1e()
    mf.conv_tol = 1e-12

    return mf

mf0 = mfobj(0.0)
e0 = mf0.kernel()
mfg = mf0.Gradients()
g0 = mfg.kernel()

dx = 1e-4
mfp = mfobj(dx)
ep = mfp.kernel()
mfm = mfobj(-dx)
em = mfm.kernel()
g_num = (ep - em) / (2 * dx)

print(g0)
print("Analytical gradient: ", g0[1,2])
print("Numerical gradient: ", g_num)