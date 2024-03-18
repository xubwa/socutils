import re
# assumes only primitives
def add_diffuse(bas, angular=[0,1,2], expand=1):
    for ang in angular:
        prim_list = []
        for prim in bas:
            if prim[0] == ang:
                prim_list.append(prim[-1][0])
        prim_min = min(prim_list)
        for iexpand in range(expand):
            bas.append([ang,[prim_min*(1./2.5)**(iexpand+1),1.0]])
    return bas
def construct_configuration(sph_irrep,configuration='2s1s'):
    angulars = ['s','p','d','f','g','h','i']
    ang_shift = [1,2,3,4,5,6,7]
    shell_size = [2,6,10,14,18,22,26]
    reorder = []
    offset = []
    for ang in angulars:
        offset.append(len(reorder))
        for i, sph in enumerate(sph_irrep):
            if ang in sph:
                reorder.append(i)
    config = re.findall(r'\d+[a-z]', configuration)
    result = []
    print(reorder)
    for iconfig in config:
        n = int(iconfig[0:-1])
        ang = angulars.index(iconfig[-1])
        for i in range(shell_size[ang]):
            result.append(reorder[offset[ang]+(n-ang_shift[ang])*shell_size[ang]+i])
    for idx in reorder:
        if idx not in result:
            result.append(idx)
    print(result)
    return result
