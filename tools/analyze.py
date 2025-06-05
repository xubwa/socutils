import pyscf
from pyscf import lib
import numpy as np
import scipy
import re

def analyze_from_chk(chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mo_energy = lib.chkfile.load(chkfile, 'scf/mo_energy')
    mo_coeff = lib.chkfile.load(chkfile,'scf/mo_coeff')
    analyze(mol, mo_coeff, mo_energy)

def analyze_mc_from_chk(chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mo_energy = lib.chkfile.load(chkfile, 'mcscf/mo_energy')
    mo_coeff = lib.chkfile.load(chkfile,'mcscf/mo_coeff')
    analyze(mol, mo_coeff, mo_energy)

def analyze_ghf(mol, mo_coeff, mo_energy):
    sph2spinor = np.vstack(mol.sph2spinor_coeff())
    mo_spinor = np.dot(sph2spinor.T, mo_coeff)
    return analyze(mol, mo_spinor, mo_energy)

def analyze_spinor_hf(mf):
    return analyze(mf.mol, mf.mo_coeff, mf.mo_energy)
def analyze(mol, mo_coeff, mo_energy):
    s = mol.intor('int1e_ovlp_spinor')
    s_sqrt = scipy.linalg.sqrtm(s)
    mo_normalized = np.dot(s_sqrt, mo_coeff)
    labels = np.array(mol.spinor_labels())
    
    label_list = dict()
    
    for idx, label in enumerate(labels):
        label = label.split()
        match = re.match(r'(\d+)([a-zA-Z].*)', label[2])
        spinor_label = f'{label[0]} {label[1]} {match.group(2):9s}'
        if spinor_label in label_list:
            label_list[spinor_label].append(idx)
        else:
            label_list[spinor_label]=[idx]
    
    for idx in range(mo_normalized.shape[1]):
        mo_i = abs(mo_normalized[:,idx])**2
        threshold = 0.01
        print(f'\nMO #{idx+1} Energy={mo_energy[idx]:16.8f}\nSpinor AO with contribution greater than {threshold:.2e} ')
        for label in label_list:
            contribution = sum(mo_i[label_list[label]])
            if contribution > threshold:
                print(f'{label}, {contribution:8.4f}')
        print(sum(mo_i))
        sort_idx = np.argsort(mo_i)
        print(f'Top contributing AOs')
        for i in range(5):
            print(f'{labels[sort_idx[-1-i]]:20s} {mo_i[sort_idx[-1-i]]:.4e}')

def analyze_nr(mol, mo_coeff, mo_energy):
    s = mol.intor('int1e_ovlp')
    s_sqrt = scipy.linalg.sqrtm(s)
    mo_normalized = np.dot(s_sqrt, mo_coeff)
    labels = np.array(mol.ao_labels())
    
    label_list = dict()
    
    for idx, label in enumerate(labels):
        label = label.split()
        match = re.match(r'(\d+)([a-zA-Z].*)', label[2])
        spinor_label = f'{label[0]} {label[1]} {match.group(2):9s}'
        if spinor_label in label_list:
            label_list[spinor_label].append(idx)
        else:
            label_list[spinor_label]=[idx]
    
    for idx in range(mo_normalized.shape[1]):
        mo_i = abs(mo_normalized[:,idx])**2
        threshold = 0.01
        print(f'\nMO #{idx+1} Energy={mo_energy[idx]:16.8f}\nAO with contribution greater than {threshold:.2e} ')
        for label in label_list:
            contribution = sum(mo_i[label_list[label]])
            if contribution > threshold:
                print(f'{label}, {contribution:8.4f}')
                
        sort_idx = np.argsort(mo_i)
        print(f'Top contributing AOs')
        for i in range(5):
            print(f'{labels[sort_idx[-1-i]]:20s} {mo_i[sort_idx[-1-i]]:.4e}')
