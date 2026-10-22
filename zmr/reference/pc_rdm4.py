'''
PC-NEVPT2 commutator matrices from explicit RDMs up to rank four (the terms
of zmr.reference.pc_rdm4_terms).  Reference for zmr.pc.rdm_terms, which
replaces the 4-RDM by intermediate states; small active spaces only.
'''

from socutils.zmr.active import rdm as active_rdm
from socutils.zmr.pc import rdm_terms
from socutils.zmr.reference.pc_rdm4_terms import TERMS


def class_matrices(class_name, act, civecs, nelec, subset=None):
    '''(S, Kc) of zmr.pc.rdm_terms.class_matrices with the 4-RDM built explicitly.'''
    dms = {k: active_rdm.make_rdm(act, civecs, nelec, k)
           for k in range(rdm_terms.max_rank(TERMS) + 1)}
    return rdm_terms.class_matrices(class_name, dms, act.heff, act.aaaa, subset, terms=TERMS)
