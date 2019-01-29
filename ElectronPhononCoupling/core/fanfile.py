
from __future__ import print_function

__author__ = "Gabriel Antonius"

import os

import numpy as np
from numpy import zeros
import netCDF4 as nc

from .mpi import MPI, comm, size, rank, mpi_watch

from . import EpcFile

__all__ = ['FanFile']


class FanFile(EpcFile):
    
    def __init__(self, *args, **kwargs):
        self.nbands_only = kwargs.pop('nbands_only', None)
        self.kpt_idx = kwargs.pop('kpt_idx', None)
        super(FanFile, self).__init__(*args, **kwargs)
    
    def read_nc(self, fname=None):
        """Open the FAN.nc file and read it."""
        fname = fname if fname else self.fname

        super(FanFile, self).read_nc(fname)

        with nc.Dataset(fname, 'r') as root:

            self.natom = len(root.dimensions['number_of_atoms'])
            if self.kpt_idx is None:
                kpt_idx = range(len(root.dimensions['number_of_kpoints']))
            else:
                kpt_idx = self.kpt_idx
            self.kpt = root.variables['reduced_coordinates_of_kpoints'][kpt_idx,:]
            self.nkpt = self.kpt.shape[0]
            
            if self.nbands_only:
                self.nband = len(self.nbands_only)
            else:
                self.nband = len(root.dimensions['max_number_of_states'])
            nbd_idx = self.nbands_only or range(self.nband)
            max_nband = len(root.dimensions['max_number_of_states'])
            self.nsppol = len(root.dimensions['number_of_spins'])
            
            # number_of_spins, number_of_kpoints, max_number_of_states
            self.occ = root.variables['occupations'][:,kpt_idx,nbd_idx]

            self.FAN = zeros((self.nkpt, self.nband, 3, self.natom,
                              3, self.natom, self.nband), dtype=np.complex)

            # product_mband_nsppol, number_of_atoms,  number_of_cartesian_directions,
            # number_of_atoms, number_of_cartesian_directions,
            # number_of_kpoints, product_mband_nsppol*2
            # I believe product_mband_nsppol means that the values are ordered according to
            # iband+mband*isppol for isppol in range(nsppol) for iband in range(mband)
            nband_nsppol = [ib+max_nband*ip for ip in range(self.nsppol) for ib in nbd_idx] 
            nband_nsppol_cplx = [c+ib*2+max_nband*ip*2 for ip in range(self.nsppol) for ib in nbd_idx for c in range(2)] 
            FANtmp = root.variables['second_derivative_eigenenergies_actif'][nband_nsppol,:,:,:,:,kpt_idx,nband_nsppol_cplx]
            #FANtmp2 = zeros((self.nkpt,2*self.nband,3,self.natom,3,self.natom,self.nband))
            FANtmp2 = np.einsum('ijklmno->nomlkji', FANtmp)
            self.FAN.real[...] = FANtmp2[:, ::2, ...]
            self.FAN.imag[...] = FANtmp2[:, 1::2, ...]
            del FANtmp, FANtmp2

            # number_of_spins, number_of_kpoints, max_number_of_states   
            self.eigenvalues = root.variables['eigenvalues'][:,kpt_idx,nbd_idx]

            self.qred = root.variables['current_q_point'][:]
            self.wtq = root.variables['current_q_point_weight'][:]
            self.rprimd = root.variables['primitive_vectors'][:,:]

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

            self.natom, self.nkpt, self.nband, self.nsppol = dim[:]

            self.occ = np.empty((self.nsppol, self.nkpt, self.nband), dtype=np.float)

            self.FAN = np.empty((self.nkpt, self.nband, 3, self.natom,
                                 3, self.natom, self.nband), dtype=np.complex)

            self.eigenvalues = np.empty((self.nsppol, self.nkpt, self.nband), dtype=np.float)

            # number_of_kpoints, 3
            self.kpt = np.empty((self.nkpt, 3), dtype=np.float)
            self.qred = np.empty(3, dtype=np.float)
            self.wtq = np.empty(1, dtype=np.float)
            self.rprimd = np.empty((3, 3), dtype=np.float)

        comm.Bcast([self.occ, MPI.DOUBLE])
        comm.Bcast([self.FAN, MPI.COMPLEX])
        comm.Bcast([self.eigenvalues, MPI.DOUBLE])
        comm.Bcast([self.kpt, MPI.DOUBLE])
        comm.Bcast([self.qred, MPI.DOUBLE])
        comm.Bcast([self.wtq, MPI.DOUBLE])
        comm.Bcast([self.rprimd, MPI.DOUBLE])
