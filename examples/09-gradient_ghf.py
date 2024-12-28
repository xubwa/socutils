from pyscf import scf, gto
from socutils.grad import ghf_grad # for GHF.Gradients

def mfobj(dx):
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [["I", (0.0, 0.0, 0.0)],
                ["H", (0.0, 1.0, 1.0+dx)],
                ["H", (0.0, -1.0, 2.0)]]
    mol.basis = {"H": 'cc-pvdz', "I": 'cc-pvtz-dk'}
    mol.charge = 0
    mol.spin = 1
    mol.unit = "Bohr"
    mol.build()

    mf = scf.GHF(mol).x2c1e() #.denisty_fit() # enable density fitting
    # overwrite with_x2c to enable x2camf
    # from socutils.somf import amf
    # mf.with_x2c = amf.SpinOrbitalX2CAMFHelper(mol, with_gaunt=True, with_breit=True)
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