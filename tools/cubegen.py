import time
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyscf import __config__
from pyscf.tools.cubegen import Cube

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)
ORIGIN = getattr(__config__, 'cubegen_box_origin', None)
# If given, EXTENT should be a 3-element ndarray/list/tuple to represent the
# extension in x, y, z
EXTENT = getattr(__config__, 'cubegen_box_extent', None)

def orbital(mol, coeff, outfile_amplitude='orbValue.cub', outfile_angle='orbPhase.cub',
            nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN):
    """Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        outfile_amplitude : str
            Name of Cube file to be written.
        outfile_angle : str
            Name of Cube file to be written.
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval_spinor'

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        orb_on_grid[ip0:ip1] = numpy.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)
    amp_on_grid = numpy.abs(orb_on_grid)
    ang_on_grid = numpy.angle(orb_on_grid)
    # Write out orbital to the .cube file
    cc.write(amp_on_grid, outfile_amplitude, comment='Amplitude of orbital value in real space (1/Bohr^3)')
    cc.write(ang_on_grid, outfile_angle, comment='Phase angle from -pi to pi of complex orbital vallues')
    return orb_on_grid