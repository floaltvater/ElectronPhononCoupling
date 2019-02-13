from __future__ import print_function

__author__ = "Gabriel Antonius"

from copy import copy

import numpy as np
import netCDF4 as nc

from . import EpcFile, EigFile

from .mpi import MPI, comm, size, rank, mpi_watch

__all__ = ['GsrFile']


class GsrFile(EigFile):

    def __init__(self, *args, **kwargs):
        super(GsrFile, self).__init__(*args, **kwargs)
        self.degen = None

    def read_nc(self, fname=None):
        """Open the Eig.nc file and read it."""
        fname = fname if fname else self.fname

        #super(GsrFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.EIG = root.variables['eigenvalues'][:,:,:] 
            self.Kptns = root.variables['reduced_coordinates_of_kpoints'][:,:]
            self.occ = root.variables['occupations'][:,:,:]

    @property
    def nspin(self):
        return self.EIG.shape[0] if self.EIG is not None else None

    @property
    def nkpt(self):
        return self.EIG.shape[1] if self.EIG is not None else None

    @property
    def nband(self):
        return self.EIG.shape[2] if self.EIG is not None else None

    def trim_nband(self, nband_max):
        """
        Limit the number of eigenvalues to nband_max bands.
        """
        self.EIG = self.EIG[:,:,:nband_max]
        self.occ = self.occ[:,:,:nband_max]

    def trim_nkpt(self, idx_kpt):
        """
        Limit the number of k-points.
        """
        self.EIG = self.EIG[:,idx_kpt,:]
        self.Kptns = self.Kptns[idx_kpt,:]
        self.occ = self.occ[:,idx_kpt,:]

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

