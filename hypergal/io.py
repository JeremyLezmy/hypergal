""" io tools """

import numpy as np
import warnings
from pysedm.io import parse_filename

# ============ #
#  FILE I/O    #
# ============ #

def e3dfilename_to_wcscalcube(filename):
    """ """
    return filename.replace(".fits",".h5").replace("e3d","wcube")

def e3dfilename_to_hgspec(filename, which, extension='.txt'):
    """ """
    if which =='host':        
        return filename.replace(".fits",extension).replace("e3d","spec_hghost")
    
    if which in ['target','sn']:        
        return filename.replace(".fits",extension).replace("e3d","spec_hgtarget")

    raise ValueError(f"which can be host, sn or target ; {which} given")

def e3dfilename_to_hgout(filename):
    """ """
    return filename.replace(".fits",".h5").replace("e3d","hgout")

def e3dfilename_to_cubeint(filename):
    """ """
    return filename.replace(".fits",".h5").replace("e3d","intcube")

def e3dfilename_to_hgcubes(filename, which):
    """ """
    if which in ["int", "intrinsic", "intcube", "cubeint"]:
        return e3dfilename_to_cubeint(filename)
    
    if which == "fitted":
        return filename.replace(".fits",".h5").replace("e3d","hgfitted")
    
    if which == "cutout":
        return filename.replace(".fits",".h5").replace("e3d","hgcutout")
    
    if which == "model":
        return filename.replace(".fits",".h5").replace("e3d","hgmodel")

    if which == "hostmodel":
        return filename.replace(".fits",".h5").replace("e3d","hghostmodel")

    if which == "snmodel":
        return filename.replace(".fits",".h5").replace("e3d","hgsnmodel")

    if which == "bkgdmodel":
        return filename.replace(".fits",".h5").replace("e3d","hgbkgdmodel")
    
    if which in ["residual", "res"]:
        return filename.replace(".fits",".h5").replace("e3d","hgres")
    
    if which in ["psfresidual", "host"]:
        return filename.replace(".fits",".h5").replace("e3d","hgpsfres")

    raise ValueError(f"which can be int, fitted, model, residual or host ; {which} given")
    
def get_slicefit_datafile(filename, which=None):
    """ """
    if which is None:
        return e3dfilename_to_hgout(filename)
    
    if which in ["cutouts", "cutout"]:
        return e3dfilename_to_hgout(filename), "cout_slicefit"
    
    if which == "meta":
        return e3dfilename_to_hgout(filename), "meta_slicefit"
    
    if which == "full":
        return e3dfilename_to_hgout(filename), "full_slicefit"
    
    raise ValueError(f"which can be cutout, meta or full, {which} given")

# ============ #
# Data Access  #
# ============ #
def get_target_info(name, verbose=False, client=None):
    """ """
    from ztfquery import sedm, fritz

    fsource  = fritz.FritzSource.from_name(name)
    radec    = fsource.get_coordinates()            
    redshift = fsource.get_redshift(False)
                
    if verbose:
        print(f"Target {name} located at {radec} and redshift {redshift}")

    squery = sedm.SEDMQuery()
    cubefiles  = squery.get_target_cubes(name, client=client)
    astrmfiles = squery.get_target_astrom(name, client=client)
    # Check if all cubes have an astrom
    cubeid = [parse_filename(cube_)["sedmid"] for cube_ in cubefiles]
    astrid = [parse_filename(astr_)["sedmid"] for astr_ in astrmfiles]
    flagok = np.in1d(cubeid, astrid)
    if not np.all(flagok):
        cubefiles, discarded_cubefiles  = list(np.asarray(cubefiles)[flagok]), np.asarray(cubefiles)[~flagok]
        #discarded_cubefiles = np.asarray(cubefiles)[~flagok]
        warnings.warn(f"the following file(s) are discarded for this were not able to find corresponding astrometry {discarded_cubefiles}")

    return np.unique(cubefiles), radec, redshift


def get_calibrated_cube(filename):
    """ """
    # tmp
    from pysedm.dask.base import DaskCube
    return DaskCube.get_calibrated_cube(filename, as_dask=False)
