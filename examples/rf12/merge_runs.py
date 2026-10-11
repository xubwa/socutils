#!/usr/bin/env python
'''merge stage1_pw.py outputs of the same system (different l ranges)'''
import sys
import json
out = None
for fn in sys.argv[2:]:
    d = json.load(open(fn))
    if out is None:
        out = d
        continue
    have = {(k['l'], k['kappa']) for k in out['kappas']}
    out['kappas'] += [k for k in d['kappas'] if (k['l'], k['kappa']) not in have]
out['kappas'].sort(key=lambda k: (k['l'], k['kappa']))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
