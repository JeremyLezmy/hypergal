import os
import warnings
import numpy as np
import pandas

from pysedm.dask import base
from dask import delayed

from ..photometry import panstarrs
from ..spectroscopy import basics as spectrobasics
from ..spectroscopy import sedfitting
from .. import __version__ as hgvs

SEDM_SCALE = 0.558
PS_SCALE = 0.25
DEFAULT_SCALE_RATIO = SEDM_SCALE/PS_SCALE


def remove_out_spaxels(cube, overwrite=False):

    spx_map = cube.spaxel_mapping
    ill_spx = np.argwhere(np.isnan(list(spx_map.values()))).T[0]
    if len(ill_spx) > 0:
        cube_fix = cube.get_partial_cube(
            [i for i in cube.indexes if i not in cube.indexes[ill_spx]],
            np.arange(len(cube.lbda)))
        if overwrite:
            cube_fix.writeto(cube.filename)
        return cube_fix
    else:
        return cube


class DaskHyperGal(base.DaskCube):

    @classmethod
    def get_sourcecubes(cls, cubefile, radec, spxy=None, binfactor=2,
                        filters=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                        source_filter="ps1.r", source_thres=2, hgfirst=True,
                        scale_cout=15, scale_sedm=10, use_extsource=True,
                        rmtarget=2, target_radius=10, sn_only=False):
        """ """
        #
        # Cubes
        sedm_cube = cls.get_calibrated_cube(
            cubefile, hgfirst=hgfirst, as_wcscube=True, radec=radec, spxy=spxy, apply_byecr=True)
        cutouts = cls.get_cutout(radec=radec, binfactor=2, filters=filters)
        #
        # cout_cube->Source & cube
        sources = cutouts.extract_sources(filter_=source_filter, thres=source_thres,
                                          savefile=None)
        cout_cube = cutouts.to_cube(binfactor=binfactor)
        #
        # get sources cube
        wcsin = cout_cube.wcs
        if sn_only:
            target_radius = 12
        if use_extsource:
            source_coutcube = cout_cube.get_extsource_cube(sourcedf=sources, wcsin=wcsin, radec=radec,
                                                           sourcescale=scale_cout, radius=target_radius*DEFAULT_SCALE_RATIO, boundingrect=True, sn_only=sn_only)

            source_sedmcube = sedm_cube.get_extsource_cube(sourcedf=sources, wcsin=wcsin, radec=radec,
                                                           sourcescale=scale_sedm, radius=target_radius, boundingrect=False, sn_only=sn_only)
        else:
            source_coutcube = cout_cube.copy()
            source_sedmcube = sedm_cube.copy()

        if rmtarget is not None:
            rmradius = 2
            target_pos = source_sedmcube.radec_to_xy(*radec).flatten()

            source_sedmcube_notarget = source_sedmcube.get_target_removed(target_pos=target_pos,
                                                                          radius=rmradius,
                                                                          store=False, get_filename=False)
            return source_coutcube, source_sedmcube_notarget

        return source_coutcube, source_sedmcube

    # =============== #
    #   INTERNAL      #
    # =============== #

    @classmethod
    def get_calibrated_cube(cls, cubefile, fluxcalfile=None, hgfirst=True, apply_byecr=True,
                            store_data=False, get_filename=False, as_wcscube=True, radec=None, spxy=None, **kwargs):
        """ """
        cube = delayed(remove_out_spaxels)(super().get_calibrated_cube(cubefile, fluxcalfile=fluxcalfile, hgfirst=hgfirst,
                                           apply_byecr=apply_byecr,
                                           get_filename=False, **kwargs))
        if not as_wcscube:
            return cube

        #header = {**dict(cube.header), **dict({'Hypergal_version': f'{hgvs}'})}
        # cube.set_header(header)
        cube.header.update(dict({'HYPERGAL': f'{hgvs}'}))

        if get_filename and not store_data:
            warnings.warn(
                "you requested get_filename without storing the data (store_data=False)")

        return delayed(spectrobasics.sedmcube_to_wcscube)(cube, radec=radec, spxy=spxy,
                                                          store_data=store_data,
                                                          get_filename=get_filename)

    @staticmethod
    def get_cutout(radec=None, cubefile=None, client_dl=None, filters=None, **kwargs):
        """ """
        prop_cutout = dict(filters=filters, client=client_dl)
        if cubefile is not None:
            return delayed(panstarrs.PS1CutOuts.from_sedmfile)(cubefile, **prop_cutout)
        if radec is not None:
            return delayed(panstarrs.PS1CutOuts.from_radec)(*radec, **prop_cutout)

        raise ValueError("cubefile or radec must be given. Both are None")

    @staticmethod
    def run_sedfitter(cube_cutout, redshift, working_dir, sedfitter="cigale", ncores=1, lbda=None,
                      saveplot_rmspull=None, saveplot_intcube=None, sn_only=False, **kwargs):
        """ """
        if lbda is None:
            from pysedm.sedm import SEDM_LBDA
            lbda = SEDM_LBDA

        tmp_inputpath = os.path.join(working_dir, "input_sedfitting.txt")

        if sedfitter == "cigale":
            sfitter = delayed(sedfitting.Cigale.from_cube_cutouts)(cube_cutout, redshift,
                                                                   tmp_inputpath=tmp_inputpath,
                                                                   initiate=True,
                                                                   working_dir=working_dir,
                                                                   ncores=ncores, sn_only=sn_only, **kwargs)
        else:
            raise NotImplementedError(
                f"Only the cigale sed fitted has been implemented. {sedfitter} given")

        # run sedfit
        bestmodel_dir = sfitter.run()  # bestmodel_dir trick is for dask

        # get the results
        spectra_lbda = sfitter.get_sample_spectra(bestmodel_dir=bestmodel_dir,
                                                  lbda_sample=lbda,
                                                  saveplot_rmspull=saveplot_rmspull,
                                                  saveplot_intcube=saveplot_intcube)
        specdata = spectra_lbda[0]
        lbda = spectra_lbda[1]
        return cube_cutout.get_new(newdata=specdata, newlbda=lbda, newvariance="None")
