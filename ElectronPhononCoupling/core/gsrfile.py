from __future__ import print_function

__author__ = "Gabriel Antonius"

from copy import copy

import numpy as np
import netCDF4 as nc

from . import EpcFile, EigFile

from .mpi import MPI, comm, size, rank, mpi_watch

from ..util.ncutil import get_kpt_map

__all__ = ['GsrFile']


class GsrFile(EigFile):

    def __init__(self, *args, **kwargs):
        self.max_nband = kwargs.pop('max_nband', None)
        self.kpts_inp = kwargs.pop('kpts_inp', None)
        super(GsrFile, self).__init__(*args, **kwargs)
        self.degen = None

    def read_nc(self, fname=None):
        """Open the Eig.nc file and read it."""
        fname = fname if fname else self.fname

        #super(GsrFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:
            keyword = 'reduced_coordinates_of_kpoints'
            kpt_idx = get_kpt_map(root, keyword, self.kpts_inp)
            self.Kptns = root.variables[keyword][kpt_idx,:]

            self.EIG = root.variables['eigenvalues'][:,kpt_idx,:self.max_nband] 
            self.occ = root.variables['occupations'][:,kpt_idx,:]
            self.nspin, self.nkpt, self.nband = self.EIG.shape

    @mpi_watch
    def broadcast(self):
        """Broadcast the data from master to all workers."""
        comm.Barrier()

        if rank == 0:
            nspin, nkpt, nband = self.EIG.shape
            dim = np.array([nspin, nkpt, nband], dtype=np.int)
        else:
            dim = np.empty(3, dtype=np.int)
            self.nspin, self.nkpt, self.nband = dim

        comm.Bcast([dim, MPI.INT])

        if rank != 0:
            self.EIG = np.empty(dim, dtype=np.float64)
            self.Kptns = np.empty((dim[1],3), dtype=np.float64)
            self.occ = np.empty(dim, dtype=np.float64)

        comm.Bcast([self.EIG, MPI.DOUBLE])
        comm.Bcast([self.Kptns, MPI.DOUBLE])
        comm.Bcast([self.occ, MPI.DOUBLE])

