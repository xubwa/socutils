"""PySCF modules of the version-2 core-valence entries (GENBAS.thuof_cv2); the version-1 modules
(Dyall's prescription, GENBAS.thuof_cv) are kept as *_v1.py."""
import os
from socutils.tools.basisopt import genbas
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, 'pyscf')
for z in 'DTQ':
    for a in ('', 'A'):
        name = '%sDYCV%sZ-SOG' % (a, z)
        fn = name.lower().replace('-', '_')
        old = os.path.join(P, fn + '.py')
        if os.path.exists(old) and not os.path.exists(os.path.join(P, fn + '_v1.py')):
            os.rename(old, os.path.join(P, fn + '_v1.py'))
        ent = genbas.read(os.path.join(HERE, 'GENBAS.thuof_cv2'), 'U:' + name)
        genbas.write_pyscf(old, {'U': genbas.to_pyscf(ent, kappa=True)},
                           header='%s: core-valence functions at Peterson cc-pwCVnZ positions on Dyall primitives (version 2, draft of 2026-10-15)' % name)
        print('written', old)
