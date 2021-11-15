import os
import warnings
import numpy as np
import pandas

from pysedm.dask import base
from dask import delayed

from ..photometry import panstarrs
from ..spectroscopy import basics as spectrobasics
from ..spectroscopy import sedfitting


class DaskHyperGal(base.DaskCube):

    @classmethod
    def get_sourcecubes(cls, cubefile, radec, spxy=None, binfactor=2,
                        filters=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                        source_filter="ps1.r", source_thres=2, hgfirst=False,
                        scale_cout=15, scale_sedm=10, use_extsource=True,
                        rmtarget=2):
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

        if use_extsource:
            source_coutcube = cout_cube.get_extsource_cube(sourcedf=sources, wcsin=wcsin,
                                                           sourcescale=scale_cout, boundingrect=True)

            source_sedmcube = sedm_cube.get_extsource_cube(sourcedf=sources, wcsin=wcsin,
                                                           sourcescale=scale_sedm, boundingrect=False)
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
    def get_calibrated_cube(cls, cubefile, fluxcalfile=None, hgfirst=False, apply_byecr=True,
                            store_data=False, get_filename=False, as_wcscube=True, radec=None, spxy=None, **kwargs):
        """ """
        cube = super().get_calibrated_cube(cubefile, fluxcalfile=fluxcalfile, hgfirst=hgfirst,
                                           apply_byecr=apply_byecr,
                                           get_filename=False, **kwargs)
        if not as_wcscube:
            return cube

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
                      saveplot_rmspull=None, saveplot_intcube=None, **kwargs):
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
                                                                   ncores=ncores, **kwargs)
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
