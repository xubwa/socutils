from pyscf import gto, scf, mcscf
import numpy
from pyscf.shciscf import shci

# molecule object
# to carry out ecp calculation, spin-orbit pseudo potential has to be provided
# crenbl and crenbs ecp are currently the only available in pyscf, for more ecps
# visit http://www.tc.uni-koeln.de/PP/index.en.html it is only available in molpro format.
# or http://www.cosmologic-services.de/basis-sets/basissets.php the dhf-()-2c basis sets also have spin-orbit ecps.
mol = gto.M(
    verbose = 4,
    atom = '''Br 0 0 0
    ''',
    basis = {'Br':'unc-cc-pvtz-dk'},
    spin = 1)

# mean-field object
mf_x2c = scf.RHF(mol).x2c()
mf_x2c.kernel()
from pyscf import mcscf
import somf
# scalar x2c casscf calculation
mc_x2c = mcscf.CASSCF(mf_x2c, 4, 7).state_average_(numpy.ones(3)/3.0)
mc_x2c.mc2step()

# writes soc integrals, only necessary function related to soc calculaion.
somf.write_soc_integrals(mc_x2c, pc1e = 'x2c1', pc2e = 'x2c')

# generates FCIDUMP for the scalar relativistic system
mch_x2c = shci.SHCISCF(mf_x2c, 4, 7).state_average_(numpy.ones(6)/6.0)
mch_x2c.fcisolver.DoSOC = True
mch_x2c.fcisolver.DoRDM = False
mch_x2c.fcisolver.nroots = 6
mch_x2c.fcisolver.sweep_iter = [0]
mch_x2c.fcisolver.sweep_epsilon = [1e-5]
shci.dryrun(mch_x2c, mc_x2c.mo_coeff)
