import numpy
from pyscf import gto, scf
from socutils.rf12 import fourc, dc_mp2f12_kb
from socutils.rf12.nr import gbs, mp2f12
c0 = 137.035999084
mol = gto.M(atom='He 0 0 0', unit='bohr', basis=gbs.basis('aug-cc-pvdz', ['He']), verbose=0)
rhf = scf.RHF(mol); rhf.conv_tol = 1e-12; rhf.kernel()
cb = gbs.basis('aug-cc-pvdz-optri', ['He'])
nr = mp2f12.MP2F12(rhf, cb); nr.kernel()
I = nr.ints
nocc, nmo = 1, I.dims['nmo']
F = lambda l: I('F', 'aa' + l)[0, 0]
FG = I('FG', 'aaaa')[0, 0, 0, 0]; F2 = I('F2', 'aaaa')[0, 0, 0, 0]; U = I('U', 'aaaa')[0, 0, 0, 0]
fk = (I.fock + I.K)[0]
B2 = 2 * numpy.dot(I('F2', 'aaa1')[0, 0, 0], fk)
Faa11 = F('11')
B3 = 2 * numpy.einsum('PC,CA,PA', Faa11, I.K, Faa11)
Vp = numpy.einsum('pq,pq', F('PP'), I('G', 'aaPP')[0, 0]) + 2 * numpy.einsum('mc,mc', F('oc'), I('G', 'aaoc')[0, 0])
Xp = numpy.einsum('pq,pq', F('PP'), F('PP')) + 2 * numpy.einsum('mc,mc', F('oc'), F('oc'))
S2 = dict(VFG=FG, Xf2=F2, Uf=U, Tm2=B2, fKf=B3, Vproj=Vp, Xproj=Xp, V=nr.V[0,0,0,0], X=nr.X[0,0,0,0], B=nr.B[0,0,0,0],
          Bproj=B2 + U - B3 - nr.B[0,0,0,0])
with fourc.light_speed(c0 * 20):
    mf = scf.DHF(mol); mf.conv_tol = 1e-11; mf.kernel()
m = dc_mp2f12_kb.DCMP2F12KB(mf, cb, c0 * 20); m.kernel()
P = {k: v[0, 0] for k, v in m.parts.items()}
P['Tm2'] = 2 * P['Tm']
P['V'], P['X'], P['B'] = m.V[0, 0], m.X[0, 0], m.B[0, 0]
P['Bproj'] = P['PFf'] * 2 - P['PFPf']
fac = dict(VFG=.5, Vproj=.5, V=.5)
for k in S2:
    f = fac.get(k, .25)
    print('%-6s stage2*%.2f = %16.10f   KB = %16.10f   diff = %10.3f uEh' % (k, f, S2[k] * f, P[k].real, (P[k].real - f * S2[k]) * 1e6))
print('e_f12 diff', (m.e_f12 - nr.e_f12) * 1e6)
