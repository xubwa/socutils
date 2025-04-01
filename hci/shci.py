#!/usr/bin/env python
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Xubo Wang <wangxubo0201@outlook.com>
# Modified from original shci.py by
#         Sandeep Sharma <sanshar@gmail.com>
#         James Smith <james.smith9113@gmail.com>
#
'''
SHCI solver for CASCI and CASSCF.
'''

from functools import reduce
import os
import sys
import re
import struct
import time
import tempfile
import warnings

from subprocess import check_call, CalledProcessError

import numpy, h5py
import pyscf.tools
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf import mcscf

from socutils.tools import fcidump_rel

# Settings
import os
ZFCIEXE = os.popen("which ZFCI").read().strip()
ZSHCIEXE = os.popen("which ZSHCI").read().strip()
SHCISCRATCHDIR = os.path.join(os.environ.get('TMPDIR', '.'), str(os.getpid()))
SHCIRUNTIMEDIR = '.'
MPIPREFIX = '' #'mpirun -np 2'  # change to srun for SLURM job system

# remove all libraries, python or hdf5 based io is enough.

def read_rdm2(filename, norb):
    f = h5py.File(filename, 'r')
    rdm2_real = numpy.array(f['rdm2_real'])
    rdm2_imag = numpy.array(f['rdm2_imag'])
    rdm2 = rdm2_real - 1j * rdm2_imag
    rdm2 = rdm2.reshape(norb, norb, norb, norb)
    rdm2 = rdm2.transpose(0,2,1,3)
    return rdm2

class SHCI(pyscf.lib.StreamObject):
    r'''SHCI program interface and object to hold SHCI program input parameters.

    Attributes:
        davidsonTol: double
        epsilon2: double
        epsilon2Large: double
        targetError: double
        sampleN: int
        epsilon1: vector<double>
        onlyperturbative: bool
        restart: bool
        fullrestart: bool
        dE: double
        eps: double
        prefix: str
        stochastic: bool
        nblocks: int
        excitation: int
        nvirt: int
        singleList: bool
        io: bool
        nroots: int
        nPTiter: int
        DoRDM: bool
        sweep_iter: [int]
        sweep_epsilon: [float]
        initialStates: [[int]]
        groupname : str
            groupname, orbsym together can control whether to employ symmetry in
            the calculation.  "groupname = None and orbsym = []" requires the
            SHCI program using C1 symmetry.
    '''

    def __init__(self, mol=None, maxM=None, tol=None):
        self.mol = mol
        self.wfnsym = None
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.fci_exe = ZFCIEXE
        self.hci_exe = ZSHCIEXE
        self.scratchDirectory = SHCISCRATCHDIR
        self.mpiprefix = MPIPREFIX

        self.integralFile = "FCIDUMP"
        self.configFile = "input.dat"
        self.outputFile = "output.dat"
        self.runtimeDir = SHCIRUNTIMEDIR
        self.extraline = []

        # TODO: Organize into pyscf and SHCI parameters
        # Standard SHCI Input parameters
        self.dets = None
        self.davidsonTol = 1.e-6
        self.epsilon2 = 1.e-7
        self.epsilon2Large = 1000.
        self.targetError = 1.e-4
        self.sampleN = 200
        self.epsilon1 = None
        self.onlyperturbative = False
        self.fullrestart = False
        self.dE = 1.e-8
        self.eps = None
        self.stochastic = True
        self.singleList = True
        self.io = True
        self.nroots = 1
        self.nPTiter = 0
        self.DoRDM = True
        self.DoTRDM = False
        self.sweep_iter = []
        self.sweep_epsilon = []
        self.maxIter = 6
        self.restart = False
        self.orbsym = []
        self.onlywriteIntegral = False
        self.spin = None
        self.orbsym = []
        if mol is None:
            self.groupname = None
        else:
            if mol.symmetry:
                self.groupname = mol.groupname
            else:
                self.groupname = None
        ##################################################
        # don't modify the following attributes, if you do not finish part of calculation, which can be reused.
        #DO NOT CHANGE these parameters, unless you know the code in details
        self.twopdm = True  #By default, 2rdm is calculated after the calculations of wave function.
        self.shci_extra_keyword = []  #For shci advanced user only.
        # This flag _restart is set by the program internally, to control when to make
        # SHCI restart calculation.
        self._restart = False
        self.generate_schedule()
        self.returnInt = False
        self._keys = set(self.__dict__.keys())
        self.initialStates = None

    def generate_schedule(self):
        return self

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** SHCI flags ********')
        log.info('FCI executable                = %s', self.fci_exe)
        log.info('SHCI executable                = %s', self.hci_exe)
        log.info('mpiprefix              = %s', self.mpiprefix)
        log.info('scratchDirectory       = %s', self.scratchDirectory)
        log.info('integralFile           = %s',
                 os.path.join(self.runtimeDir, self.integralFile))
        log.info('configFile             = %s',
                 os.path.join(self.runtimeDir, self.configFile))
        log.info('outputFile             = %s',
                 os.path.join(self.runtimeDir, self.outputFile))
        log.info('maxIter                = %d', self.maxIter)
        log.info(
            'sweep_iter             = %s',
            '[' + ','.join(['{:>5}' for item in self.sweep_iter
                            ]).format(*self.sweep_iter) + ']')
        log.info(
            'sweep_epsilon          = %s',
            '[' + ','.join(['{:>5}' for item in self.sweep_epsilon
                            ]).format(*self.sweep_epsilon) + ']')
        log.info('nPTiter                = %i', self.nPTiter)
        log.info('Stochastic             = %r', self.stochastic)
        log.info('restart                = %s',
                 str(self.restart or self._restart))
        log.info('fullrestart            = %s', str(self.fullrestart))
        log.info('')
        return self

    # ABOUT RDMs AND INDEXES: -----------------------------------------------------------------------
    #   There is two ways to stored an RDM
    #   (the numbers help keep track of creation/annihilation that go together):
    #     E3[i1,j2,k3,l3,m2,n1] is the way DICE outputs text and bin files
    #     E3[i1,j2,k3,l1,m2,n3] is the way the tensors need to be written for SQA and ICPT
    #
    #   --> See various remarks in the pertinent functions below.
    # -----------------------------------------------------------------------------------------------
        
    def trans_rdm1(self, state_i, state_j, norb, nelec, link_index=None, **kwargs):
        trdm1 = numpy.zeros((norb, norb), dtype=complex)
        # assume Dice prints only i < j transition rdm.
        if state_i > state_j:
            tmp = state_i
            state_i = state_j
            state_j = tmp

        filetrdm1 = os.path.join(self.scratchDirectory, "transition1RDM.%d.%d.txt" % (state_i, state_j))
        with open(filetrdm1) as f:
            line = f.readline()
            file_orb = int(line.split()[0])
            for line in f:
                orb1 = int(line.split()[0])
                orb2 = int(line.split()[1])
                val = re.split("[(,)]", line.split()[2])
                val = complex(float(val[1]), float(val[2]))
                trdm1[orb1][orb2] = val
        return trdm1
        
    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
        # Avoid calling self.make_rdm12 because it may be overloaded
        return self.make_rdm12(state, norb, nelec, link_index, **kwargs)[0]
    
    def make_rdm2(self, state, norb, nelec, link_index=None, **kwargs):
        # Avoid calling self.make_rdm12 because it may be overloaded
        return self.make_rdm12(state, norb, nelec, link_index, **kwargs)[1]

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        twopdm = read_rdm2(os.path.join(SHCISCRATCHDIR, f'Dice_{state}_{state}.rdm2.h5'), norb)
        onepdm = numpy.einsum('ijkk->ji', twopdm) / (nelectrons - 1)
        return onepdm, twopdm

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, wfnsym=None,**kwargs):
        """
        Approximately solve CI problem for the specified active space.
        """

        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)

        if fciRestart is None:
            fciRestart = self.restart or self._restart

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']

        #writeIntegralFile(self, h1e, eri.reshape(norb*norb, norb*norb), norb, nelec, ecore)
        writeSHCIConfFile(self, nelec, fciRestart)

        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
            logger.debug1(self, open(inFile, 'r').read())

        if self.onlywriteIntegral:
            logger.info(self, 'Only write integral')
            try:
                calc_e = read_energy(self)
            except IOError:
                if self.nroots == 1:
                    calc_e = 0.0
                else:
                    calc_e = [0.0] * self.nroots
            return calc_e, roots

        if self.returnInt:
            return h1e, eri

        executeSHCI(self)

        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())

        calc_e = read_energy(self)

        return calc_e, roots

    # comment out approx stuff until I understand them.
    #def approx_kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
    #    fciRestart = True

    #    if 'orbsym' in kwargs:
    #        self.orbsym = kwargs['orbsym']
    #    writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
    #    writeSHCIConfFile(self, nelec, fciRestart)
    #    if self.verbose >= logger.DEBUG1:
    #        inFile = os.path.join(self.runtimeDir, self.configFile)
    #        logger.debug1(self, 'SHCI Input conf')
    #        logger.debug1(self, open(inFile, 'r').read())
    #    executeSHCI(self)
    #    if self.verbose >= logger.DEBUG1:
    #        outFile = os.path.join(self.runtimeDir, self.outputFile)
    #        logger.debug1(self, open(outFile).read())
    #    calc_e = read_energy(self)

    #    if self.nroots == 1:
    #        roots = 0
    #    else:
    #        roots = range(self.nroots)
    #    return calc_e, roots
    
    #def restart_scheduler_(self):
    #    def callback(envs):
    #        if (envs['norm_gorb'] < self.shci_switch_tol
    #                or ('norm_ddm' in envs
    #                    and envs['norm_ddm'] < self.shci_switch_tol * 10)):
    #            self._restart = True
    #        else:
    #            self._restart = False
#
    #    return callback

    def cleanup_dice_files(self):
        """
        Remove the files used for Dice communication.
        """
        
        os.remove(os.path.join(self.runtimeDir, self.configFile))
        os.remove(os.path.join(self.runtimeDir, self.outputFile))
        os.remove(os.path.join(self.runtimeDir, self.integralFile))

'''
def transition_dipole(mc, state_i, state_j):
    t_dm1 = mc.fcisolver.trans_rdm1(state_i, state_j, mc.ncas, mc.nelecas)
    ncore = mc.ncore
    ncasorb = mc.ncas
    mol = mc.mol
    mo_cas = mc.mo_coeff[:, ncore:ncore+ncasorb]
    t_dm1 = pyscf.lib.einsum('pi, ij, qj->pq', mo_cas, t_dm1, mo_cas)
    #print(t_dm1)
    charge_center = (numpy.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords()) / mol.atom_charges().sum())
    with mol.with_common_origin(charge_center):
        t_dip = numpy.einsum('xij,ji->x', mol.intor('int1e_r'), t_dm1)
    return t_dip

def oscillator_strength(mc, state_i, state_j):
    t_dip = transition_dipole(mc, state_i, state_j)
    calc_e = read_energy(mc.fcisolver)
    delta_e = abs(calc_e[state_i] - calc_e[state_j])
    return 2./3.*delta_e*sum(abs(t_dip)**2), 2./3.*delta_e*abs(t_dip)**2

def phospherescence_lifetime(mc, s0=0, triplet_list=[1,2,3]):
    au2wavenumber = 219470.
    tau = numpy.zeros(len(triplet_list))
    calc_e = read_energy(mc.fcisolver)
    for state in triplet_list:
        delta_e = calc_e[state] - calc_e[s0]
        fosc = oscillator_strength(mc, s0, state)
        tau[triplet_list.index(state)] = 1.5/(fosc*(delta_e*au2wavenumber)**2)
    tau_av = 3./(sum(1/tau))
    return tau_av, tau
'''
    
def make_sched(SHCI):

    niter = len(SHCI.sweep_iter)
    # Check that the number of different epsilons match the number of iter steps.
    assert (niter == len(SHCI.sweep_epsilon))

    if (niter == 0):
        SHCI.sweep_iter = [0]
        SHCI.sweep_epsilon = [1.e-3]

    sched_string = 'schedule '
    for it, eps in zip(SHCI.sweep_iter, SHCI.sweep_epsilon):
        sched_string += '\n' + str(it) + '\t' + str(eps)
    sched_string += '\nend\n'

    return sched_string

def writeSHCIConfFile(SHCI, nelec, Restart):
    conf_file = os.path.join(SHCI.runtimeDir, SHCI.configFile)

    f = open(conf_file, 'w')

    # Reference determinant section
    f.write('#system\n')
    f.write(f'nocc {nelec}\n')
    if SHCI.initialStates is not None:
        print("write determinants")
        for i, state in enumerate(SHCI.initialStates):
            for j in state:
                f.write(f'{j} ')
            if i != len(SHCI.initialStates) - 1:
                f.write('\n')
    else:
        for i in range(nelec):
            f.write(f'{i} ')

    f.write('\nend\n')

    # Handle different cases for FCIDUMP file names/paths
    f.write(f'orbitals {os.path.join(SHCI.runtimeDir, SHCI.integralFile)}\n')
    f.write(f'nroots {SHCI.nroots}\n')
    # Variational Keyword Section
    f.write('\n#variational\n')
    if not Restart:
        schedStr = make_sched(SHCI)
        f.write(schedStr)
    else:
        f.write('schedule\n')
        f.write(f'{0}  {SHCI.sweep_epsilon[-1]}\n')
        f.write('end\n')

    f.write(f'davidsonTol {SHCI.davidsonTol}\n')
    f.write(f'dE {SHCI.dE}\n')

    # Sets maxiter to 6 more than the last iter in sweep_iter[] if restarted.
    if not Restart:
        f.write(f'maxiter {SHCI.sweep_iter[-1] + 6}\n')
    else:
        f.write('maxiter 10\n')
        f.write('fullrestart\n')

    # Perturbative Keyword Section
    f.write('\n#pt\n')
    if not SHCI.stochastic:
        f.write('deterministic \n')
        f.write(f'epsilon2 {SHCI.epsilon2}\n')
    else:
        f.write(f'nPTiter {SHCI.nPTiter}\n')
        f.write(f'epsilon2 {SHCI.epsilon2}\n')
        f.write(f'epsilon2Large {SHCI.epsilon2Large}\n')
        f.write(f'targetError {SHCI.targetError}\n')
        f.write(f'sampleN {SHCI.sampleN}\n')

    # Miscellaneous Keywords
    f.write('\n#misc\n')
    f.write('readText\n')
    if Restart:
        f.write('noio \n')
    if SHCI.scratchDirectory != "":
        if not os.path.exists(SHCI.scratchDirectory):
            os.makedirs(SHCI.scratchDirectory)
        f.write(f'prefix {SHCI.scratchDirectory}\n')
    if SHCI.DoRDM:
        f.write('DoRDM\n')
    for line in SHCI.extraline:
        f.write(f'{line}\n')
    f.write('\n')  # SHCI requires that there is an extra line.
    f.close()

def writeIntegralFile(shci_obj, h1eff, eri_cas, norb, nelec, ecore=0.0):
    fname = shci_obj.integralFile
    return fcidump_rel.from_integrals(fname, h1eff, eri_cas, norb, nelec, ecore)

def executeSHCI(SHCI):
    #file1 = os.path.join(SHCI.runtimeDir, "%s/shci.e" % (SHCI.scratchDirectory))
    #if os.path.exists(file1):
    #    os.remove(file1)
    inFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)
    outFile = os.path.join(SHCI.runtimeDir, SHCI.outputFile)
    try:
        cmd = ' '.join((SHCI.mpiprefix, SHCI.fci_exe, inFile))
        cmd = "%s > %s 2>&1" % (cmd, outFile)
        check_call(cmd, shell=True)
    except CalledProcessError as err:
        logger.error(SHCI, cmd)
        raise err


def read_energy(SHCI):
    file1 = open(
        os.path.join(SHCI.runtimeDir, "%s/shci.e" % (SHCI.scratchDirectory)),
        "rb")
    fmt = ['d'] * SHCI.nroots
    fmt = ''.join(fmt)
    calc_e = struct.unpack(fmt, file1.read())
    file1.close()
    if SHCI.nroots == 1:
        return calc_e[0]
    else:
        return list(calc_e)


def dryrun(mc, mo_coeff=None):
    '''
    Generate FCIDUMP and SHCI input.dat file for a give multiconfigurational 
    object. This method DOES NOT run a single iteration using Dice it just 
    generates all the inputs you'd need to do so.

    Args:
        mc: a CASSCF/CASCI object (or RHF object)

    Kwargs: 
        mo_coeff: `np.ndarray` of the molecular orbital coefficients. 
            Default is None.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.shciscf import shci
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 2; O 0 1 1;", symmetry=True)
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASCI(mf, 3, 4)
    >>> mc.fcisolver = shci.SHCI(mol)
    >>> shci.dryrun(mc)

    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    
    # Set orbsym b/c it's needed in `writeIntegralFile()`
    if mc.fcisolver.orbsym is not []:
        mc.fcisolver.orbsym = getattr(mo_coeff, "orbsym", [])

    mc.kernel(mo_coeff) # Works, but runs full CASCI/CASSCF
    h1e, ecore = mc.get_h1eff(mo_coeff)
    h2e = mc.get_h2eff(mo_coeff)

    #writeIntegralFile(mc.fcisolver, h1e, h2e, mc.ncas, mc.nelecas, ecore)
    writeSHCIConfFile(mc.fcisolver, mc.nelecas, False)
