""" io tools """

import numpy as np
import warnings
from pysedm.io import parse_filename

# ============ #
#  FILE I/O    #
# ============ #


def e3dfilename_to_wcscalcube(filename, suffix=''):
    """ """
    return filename.replace(".fits", ".h5").replace("e3d", "wcube"+suffix)


def e3dfilename_to_hgspec(filename, which, extension='.txt', suffix=''):
    """ """
    if which == 'host':
        return filename.replace(".fits", extension).replace("e3d", "hgspec_host"+suffix)

    if which in ['target', 'sn']:
        return filename.replace(".fits", extension).replace("e3d", "hgspec_target"+suffix)

    raise ValueError(f"which can be host, sn or target ; {which} given")


def e3dfilename_to_hgout(filename, suffix=''):
    """ """
    return filename.replace(".fits", ".h5").replace("e3d", "hgout"+suffix)


def e3dfilename_to_cubeint(filename, suffix=''):
    """ """
    return filename.replace(".fits", ".h5").replace("e3d", "intcube"+suffix)


def e3dfilename_to_hgcubes(filename, which, suffix=''):
    """ """
    if which in ["int", "intrinsic", "intcube", "cubeint"]:
        return e3dfilename_to_cubeint(filename)

    if which == "fitted":
        return filename.replace(".fits", ".h5").replace("e3d", "hgfitted"+suffix)

    if which == "cutout":
        return filename.replace(".fits", ".h5").replace("e3d", "hgcutout"+suffix)

    if which == "model":
        return filename.replace(".fits", ".h5").replace("e3d", "hgmodel"+suffix)

    if which == "hostmodel":
        return filename.replace(".fits", ".h5").replace("e3d", "hghostmodel"+suffix)

    if which == "snmodel":
        return filename.replace(".fits", ".h5").replace("e3d", "hgsnmodel"+suffix)

    if which == "bkgdmodel":
        return filename.replace(".fits", ".h5").replace("e3d", "hgbkgdmodel"+suffix)

    if which in ["residual", "res"]:
        return filename.replace(".fits", ".h5").replace("e3d", "hgres"+suffix)

    if which in ["psfresidual", "host"]:
        return filename.replace(".fits", ".h5").replace("e3d", "hgpsfres"+suffix)

    raise ValueError(
        f"which can be int, fitted, model, residual or host ; {which} given")


def get_slicefit_datafile(filename, which=None, suffix=''):
    """ """
    if which is None:
        return e3dfilename_to_hgout(filename, suffix=suffix)

    if which in ["cutouts", "cutout"]:
        return e3dfilename_to_hgout(filename, suffix=suffix), "cout_slicefit"

    if which == "meta":
        return e3dfilename_to_hgout(filename, suffix=suffix), "meta_slicefit"

    if which == "full":
        return e3dfilename_to_hgout(filename, suffix=suffix), "full_slicefit"

    raise ValueError(f"which can be cutout, meta or full, {which} given")

# ============ #
# Data Access  #
# ============ #


def get_target_info(name, contains=None, date_range=None, ignore_astrom=True, verbose=False, client=None):
    """ """
    from ztfquery import sedm, fritz
    try:
        fsource = fritz.FritzSource.from_name(name)
    except OSError:
        warnings.warn(
            f"The target {name} doesn't exist in Fritz!")
        return [], None, None
    radec = fsource.get_coordinates()
    redshift = fsource.get_redshift(False)

    if verbose:
        print(f"Target {name} located at {radec} and redshift {redshift}")

    squery = sedm.SEDMQuery()
    squery.update_pharosio()
    df = squery.get_whatdata(targets=name, date_range=date_range)
    cubefiles = sedm.download_from_whatdata(
        df, 'cube', contains=contains, client=client, return_filename=True)
    astrmfiles = sedm.download_from_whatdata(
        df, 'astrom', contains=contains, client=client, return_filename=True)
    hexagrid = sedm.download_from_whatdata(
        df, 'HexaGrid', contains=None, client=client, return_filename=True)
    #cubefiles  = squery.get_target_cubes(name, contains=contains, client=client)
    #astrmfiles = squery.get_target_astrom(name, contains=contains, client=client)

    if ignore_astrom:
        return np.unique(cubefiles), radec, redshift
    # Check if all cubes have an astrom
    cubeid = [parse_filename(cube_)["sedmid"] for cube_ in cubefiles]
    astrid = [parse_filename(astr_)["sedmid"] for astr_ in astrmfiles]
    flagok = np.in1d(cubeid, astrid)
    if not np.all(flagok):
        cubefiles, discarded_cubefiles = list(np.asarray(
            cubefiles)[flagok]), np.asarray(cubefiles)[~flagok]
        #discarded_cubefiles = np.asarray(cubefiles)[~flagok]
        warnings.warn(
            f"the following file(s) are discarded for this were not able to find corresponding astrometry {discarded_cubefiles}")

    return np.unique(cubefiles), radec, redshift


def get_calibrated_cube(filename):
    """ """
    # tmp
    from pysedm.dask.base import DaskCube
    return DaskCube.get_calibrated_cube(filename, as_dask=False)
