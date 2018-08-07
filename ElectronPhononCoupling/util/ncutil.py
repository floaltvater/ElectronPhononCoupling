
import netCDF4 as nc
import numpy as np

def nc_copy(dsin, dsout, except_dimensions=None, except_variables=None):
    """
    Copy all dimensions and variable of one nc.Dataset instance into another.
    """

    #Copy dimensions
    for dname, dim in dsin.dimensions.iteritems():
        if except_dimensions and dname in except_dimensions:
            continue
        dsout.createDimension(dname, len(dim))

    #Copy variables
    for vname, varin in dsin.variables.iteritems():
        if except_variables and vname in except_variables:
            continue
        outVar = dsout.createVariable(vname, varin.datatype, varin.dimensions)
        outVar[...] = varin[...]

def get_kpt_map(dsin, keyword, kpts_inp):
    """
    Take a list of kpts (kpts_inp), compare it to the list of kpts from an nc
    file (ds_kpts), and return a list of indeces for array indexing, so that
    kpts_inp corresponds to ds_kpts[kpt_map].
    
    Raises IOError if a kpt from kpts_inp is not found in ds_kpts.
    """
    ds_kpts = dsin.variables[keyword][:]
    if kpts_inp is None:
        return np.array(len(ds_kpts))
    kpts_inp = np.array(kpts_inp)
    eps = 1e-10
    diff = ds_kpts[np.newaxis,:,:] - kpts_inp[:,np.newaxis,:]
    norm = np.linalg.norm(diff, axis=2)
    kpt_idx = np.argwhere(norm < eps)[:,1]
    if not len(kpt_idx) == len(kpts_inp):
        missing = np.argwhere(np.product(norm >= eps, axis=1) == 1.)
        kpts_missing = kpts_inp[missing[:,0]]
        msg = "kpt(s) not found in {}: {}\n\n{}".format(dsin.filepath(), kpts_missing, ds_kpts)
        raise IOError(msg)
    return kpt_idx

