
from __future__ import print_function

__author__ = "Gabriel Antonius"

import os

import numpy as np
from numpy import zeros
import netCDF4 as nc

from .mpi import MPI, comm, size, rank, mpi_watch

from . import EpcFile

__all__ = ['Eigr2dFile']


class Eigr2dFile(EpcFile):

    def read_nc(self, fname=None):
        """Open the EIG2D.nc file and read it."""
        fname = fname if fname else self.fname

        super(Eigr2dFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.natom = len(root.dimensions['number_of_atoms'])
            self.nsppol = len(root.dimensions['number_of_spins'])
            nkpt = len(root.dimensions['number_of_kpoints'])
            nband = len(root.dimensions['max_number_of_states'])
            # number_of_spins, number_of_kpoints, max_number_of_states
            self.occ = root.variables['occupations'][:,:,:]

            self.EIG2D = zeros((nkpt, nband, 3, self.natom, 3, self.natom), dtype=np.complex)

            # number_of_atoms, number_of_cartesian_directions, number_of_atoms, number_of_cartesian_directions,
            # number_of_kpoints, product_mband_nsppol, cplex
            EIG2Dtmp = root.variables['second_derivative_eigenenergies'][:,:,:,:,:,:,:]

            EIG2Dtmp2 = np.einsum('ijklmno->mnlkjio', EIG2Dtmp)

            self.EIG2D.real[...] = EIG2Dtmp2[...,0]
            self.EIG2D.imag[...] = EIG2Dtmp2[...,1]

            del EIG2Dtmp, EIG2Dtmp2

            # number_of_spins, number_of_kpoints, max_number_of_states   
            self.eigenvalues = root.variables['eigenvalues'][:,:,:]

            # number_of_kpoints, 3
            self.kpt = root.variables['reduced_coordinates_of_kpoints'][:,:]

            self.qred = root.variables['current_q_point'][:]
            self.wtq = root.variables['current_q_point_weight'][:]
            self.rprimd = root.variables['primitive_vectors'][:,:]
    
    @property
    def nkpt(self):
        return self.EIG2D.shape[0] if self.EIG2D is not None else None

    @property
    def nband(self):
        return self.EIG2D.shape[1] if self.EIG2D is not None else None

    def trim_nband(self, nband_max):
        """
        Limit the number of eigenvalues to nband_max bands.
        """
        self.EIG2D = self.EIG2D[:,:nband_max].copy()
        self.occ = self.occ[:,:,:nband_max].copy()
        self.eigenvalues = self.eigenvalues[:,:,:nband_max].copy()

    def trim_nkpt(self, idx_kpt):
        """
        Limit the number of k-points.
        """
        self.EIG2D = self.EIG2D[idx_kpt].copy()
        self.occ = self.occ[:,idx_kpt].copy()
        self.eigenvalues = self.eigenvalues[:,idx_kpt].copy()
        self.kpt = self.kpt[idx_kpt].copy()

    @mpi_watch
    def broadcast(self):
        """Broadcast the data from master to all workers."""
    
        comm.Barrier()

        if rank == 0:
            dim = np.array([self.natom, self.nkpt, self.nband, self.nsppol], dtype=np.int)
        else:
            dim = np.empty(4, dtype=np.int)

        comm.Bcast([dim, MPI.INT])

        if rank != 0:

            self.natom, nkpt, nband, self.nsppol = dim[:]

            self.occ = np.empty((self.nsppol, nkpt, nband), dtype=np.float)

            self.EIG2D = np.empty((nkpt, nband, 3, self.natom,
                                   3, self.natom), dtype=np.complex)

            self.eigenvalues = np.empty((self.nsppol, nkpt, nband), dtype=np.float)

            # number_of_kpoints, 3
            self.kpt = np.empty((nkpt, 3), dtype=np.float)
            self.qred = np.empty(3, dtype=np.float)
            self.wtq = np.empty(1, dtype=np.float)
            self.rprimd = np.empty((3, 3), dtype=np.float)

        comm.Bcast([self.occ, MPI.DOUBLE])
        comm.Bcast([self.EIG2D, MPI.COMPLEX])
        comm.Bcast([self.eigenvalues, MPI.DOUBLE])
        comm.Bcast([self.kpt, MPI.DOUBLE])
        comm.Bcast([self.qred, MPI.DOUBLE])
        comm.Bcast([self.wtq, MPI.DOUBLE])
        comm.Bcast([self.rprimd, MPI.DOUBLE])
