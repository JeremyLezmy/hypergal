""" io tools """

from pysedm.io import parse_filename


def e3dfilename_to_wcube(filename):
    """ """
    return filename.replace(".fits",".h5").replace("e3d","wcube")

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
    
    if which == "model":
        return filename.replace(".fits",".h5").replace("e3d","hgmodel")
    
    if which in ["residual", "res"]:
        return filename.replace(".fits",".h5").replace("e3d","hgres")
    
    if which in ["psfresidual", "host"]:
        return filename.replace(".fits",".h5").replace("e3d","hgpsfres")

    raise ValueError(f"which can be int, fitted, model, residual or host ; {which} given")
    
# ============ #
#  GETTER      #
# ============ #
def get_calibrated_cube(filename):
    """ """
    # tmp
    from pysedm.dask.base import DaskCube
    return DaskCube.get_calibrated_cube(filename, as_dask=False)

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

