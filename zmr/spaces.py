'''Partition of the correlated orbitals into core / active / virtual.'''


class OrbitalSpaces:
    '''Correlated orbitals ordered as core, active, virtual (conventions.md
    section 1).  Frozen orbitals are not part of the partition.'''

    LABELS = 'cavp'

    def __init__(self, ncore, nact, nvirt):
        self.ncore = int(ncore)
        self.nact = int(nact)
        self.nvirt = int(nvirt)

    @property
    def nmo(self):
        return self.ncore + self.nact + self.nvirt

    @property
    def nocc(self):
        return self.ncore + self.nact

    def slice(self, label):
        '''c = core, a = active, v = virtual, p = all orbitals'''
        if label == 'c':
            return slice(0, self.ncore)
        if label == 'a':
            return slice(self.ncore, self.nocc)
        if label == 'v':
            return slice(self.nocc, self.nmo)
        if label == 'p':
            return slice(0, self.nmo)
        raise KeyError('unknown orbital space label %r' % label)

    def size(self, label):
        s = self.slice(label)
        return s.stop - s.start

    def __repr__(self):
        return 'OrbitalSpaces(ncore=%d, nact=%d, nvirt=%d)' % (
            self.ncore, self.nact, self.nvirt)
