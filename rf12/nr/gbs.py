#!/usr/bin/env python
# Copyright 2026 The socutils developers. All Rights Reserved.
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

'''Reader for Psi4 .gbs (Gaussian94-style) basis files -> pyscf basis dicts,
so that the reference program and this code use byte-identical bases.'''

ANG = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6}


def load(path, element):
    lines = open(path).read().splitlines()
    out = []
    i = 0
    el = element.upper()
    while i < len(lines):
        tok = lines[i].split()
        if len(tok) == 2 and tok[0].upper() == el and tok[1] == '0':
            i += 1
            while i < len(lines) and not lines[i].startswith('****'):
                t = lines[i].split()
                if t and t[0].upper() in ANG or (t and t[0].upper() == 'SP'):
                    kind, nprim = t[0].upper(), int(t[1])
                    prims = [[float(x.replace('D', 'E')) for x in lines[i + 1 + k].split()] for k in range(nprim)]
                    if kind == 'SP':
                        out.append([0] + [[p[0], p[1]] for p in prims])
                        out.append([1] + [[p[0], p[2]] for p in prims])
                    else:
                        out.append([ANG[kind]] + [[p[0], p[1]] for p in prims])
                    i += 1 + nprim
                else:
                    i += 1
            return out
        i += 1
    raise KeyError('%s not in %s' % (element, path))


def basis(name, elements, basdir='/home/user/psi4env/share/psi4/basis'):
    import os
    path = os.path.join(basdir, name.lower() + '.gbs')
    return {el: load(path, el) for el in elements}
