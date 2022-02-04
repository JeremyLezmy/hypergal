import os
import numpy as np
from .daskbasics import DaskHyperGal
from pysedm.sedm import SEDM_LBDA
from ztfquery.sedm import SEDMLOCAL_BASESOURCE
from shapely.geometry import Point
#import shapely
from ..photometry import basics as photobasics
from ..spectroscopy import adr as spectroadr
import pandas
from ..fit import SceneFitter, MultiSliceParameters, Priors
from ..utils.cubes import CubeModelBuilder
from .. import psf
from ..scene.basics import PointSource
from .. import io
from dask import delayed
from astropy.io import fits
import pyifu
import dask

# GAL EMISSION LINES

Mg = 5177
Na = 5896

Hbeta = 4861.333
Halpha = 6562.819
S2_2 = 6730.810
N1 = 7468.310

O3_a = 4932.603
O3_b = 4958.911
O3_c = 5006.843

all_em = np.array([Hbeta, Halpha, S2_2, N1])
all_em_names = [r'$H_{\beta}$', r'$H_{\alpha}$', r'$S[II]$', r'$N[I]$']

O3li = np.array([O3_a, O3_b, O3_c])

all_ab = np.array([Mg, Na])
all_ab_names = [r'$M_g$', r'$N_a$']


def em_lines(z):
    return(z*all_em + all_em)


def o3_lines(z):
    return(z*O3li + O3li)


def ab_lines(z):
    return(z*all_ab + all_ab)


class DaskScene(DaskHyperGal):

    @classmethod
    def compute_targetcubes(cls, name, client, contains=None, date_range=None, verbose=False, ignore_astrom=True, manual_radec=None,
                            return_cubefile=False, manual_z=None, **kwargs):
        """ """
        cubefiles, radec, redshift = io.get_target_info(
            name, contains=contains, date_range=date_range, ignore_astrom=ignore_astrom, verbose=True)
        if len(cubefiles) == 0:
            if return_cubefile:
                return None, []
            return None
        this = cls(client=client)
        if manual_z != None:
            redshift = manual_z
        elif redshift == None:
            redshift = 0.07

        if manual_radec != None:
            radec = radec
        elif radec == None:
            raise ValueError(
                'No available radec from datas, you must manually set it to rebuild astrometry')

        storings = [this.compute_single(cubefile_, radec, redshift, **kwargs)
                    for cubefile_ in cubefiles]
        if return_cubefile:
            return storings, cubefiles
        return storings

    @staticmethod
    def remove_out_spaxels(cube, overwrite=False):

        spx_map = cube.spaxel_mapping
        ill_spx = np.argwhere(np.isnan(list(spx_map.values()))).T[0]
        if len(ill_spx) > 0:
            cube_fix = cube.get_partial_cube(
                [i for i in cube.indexes if i not in cube.indexes[ill_spx]], np.arange(len(cube.lbda)))
            if overwrite:
                cube_fix.writeto(cube.filename)
            return cube_fix
        else:
            return cube

    def compute_single(self, cubefile, radec, redshift,
                       hgfirst=True, binfactor=2,
                       filters=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                       source_filter="ps1.r", source_thres=2,
                       scale_cout=15, scale_sedm=10, rmtarget=2,
                       lbda_range=[5000, 8500], nslices=6,
                       filters_fit=["ps1.g", "ps1.r", "ps1.i", "ps1.z"],
                       psfmodel="Gauss2D", pointsourcemodel="GaussMoffat2D", ncores=1, testmode=True, xy_ifu_guess=None,
                       prefit_photo=True, use_exist_intcube=True, use_extsource=True,
                       split=True, curved_bkgd=True, build_astro=True,
                       host_only=False, sn_only=False, suffix_plot=None):
        """ """
        info = io.parse_filename(cubefile)
        cubeid = info["sedmid"]
        name = info["name"]
        filedir = os.path.dirname(cubefile)
        # SED
        working_dir = os.path.join(os.path.dirname(cubefile), f"tmp_{cubeid}")

        # PLOTS
        plotbase = os.path.join(filedir, "hypergal",
                                info["name"], info["sedmid"])
        if suffix_plot is not None:
            plotbase = os.path.join(filedir, "hypergal",
                                    info["name"], suffix_plot + info["sedmid"])
        dirplotbase = os.path.dirname(plotbase)

        #dirspec = os.path.join( SEDMLOCAL_BASESOURCE, "hypergal_output", "target_spec", name)
        #dirhost = os.path.join( SEDMLOCAL_BASESOURCE, "hypergal_output", "host_spec", name)
        #dirplot = os.path.join( SEDMLOCAL_BASESOURCE, "hypergal_output", "plots", name)
        # for dirs in [dirspec, dirhost, dirplot]:
        #    if not os.path.isdir(dirs):
        #        os.makedirs(dirs, exist_ok=True)

        if xy_ifu_guess is not None:
            initguess = dict(
                {k: v for k, v in zip(['xoff', 'yoff'], xy_ifu_guess)})
        else:
            initguess = None

        if not os.path.isdir(dirplotbase):
            os.makedirs(dirplotbase, exist_ok=True)

        # Fileout
        stored = []

        # ------------ #
        #    STEP 1    #
        # ------------ #
        # ---> Build the cutouts, and the calibrated data cube

        if build_astro and xy_ifu_guess != None:
            spxy = xy_ifu_guess
        else:
            spxy = None

        calcube = delayed(self.remove_out_spaxels)(self.get_calibrated_cube(
            cubefile, hgfirst=hgfirst, apply_byecr=True, radec=radec, spxy=spxy))

        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec, spxy=spxy,
                                                                binfactor=binfactor,
                                                                filters=filters,
                                                                source_filter=source_filter,
                                                                source_thres=source_thres, hgfirst=hgfirst, scale_cout=scale_cout,
                                                                scale_sedm=scale_sedm, use_extsource=use_extsource,
                                                                rmtarget=rmtarget, sn_only=sn_only)

        source_coutcube = source_coutcube__source_sedmcube[0]
        source_sedmcube = source_coutcube__source_sedmcube[1]

        # ---> Storing <--- # 0
        stored.append(source_sedmcube.to_hdf(
            io.e3dfilename_to_hgcubes(cubefile, "fitted")))
        stored.append(source_coutcube.to_hdf(
            io.e3dfilename_to_hgcubes(cubefile, "cutout")))
        # ---> Storing <--- # 1
        stored.append(calcube.to_hdf(io.e3dfilename_to_wcscalcube(cubefile)))

        #
        #   Step 1.1 Cutouts
        #
        # ---> fit position and PSF parameters from the cutouts

        if prefit_photo and not sn_only:

            saveplot_structure = plotbase + '_' + name + '_cout_fit_'
            bestfit_cout = self.fit_cout_slices(source_coutcube, source_sedmcube, radec,
                                                saveplot_structure=saveplot_structure,
                                                filterin=filters, filters_to_use=filters_fit,
                                                psfmodel=psfmodel, pointsourcemodel=pointsourcemodel, guess=initguess, host_only=host_only, kind='metaslices', onlyvalid=True)

            # ---> Storing <--- # 2
            # stored.append(bestfit_cout.to_hdf(
            #    *io.get_slicefit_datafile(cubefile, "cutout")))

            # ---> Get the object for future guesses || Guesser

            cout_ms_param = delayed(MultiSliceParameters)(bestfit_cout, cubefile=cubefile,
                                                          psfmodel=psfmodel.replace("2D", "3D"), pointsourcemodel='GaussMoffat3D',
                                                          load_adr=True, load_psf=True, load_pointsource=True)
        else:
            cout_ms_param = None
        #
        #   Step 1.2 Intrinsic Cube
        #
        # ---> SED fitting of the cutouts
        if use_exist_intcube and os.path.exists(io.e3dfilename_to_hgcubes(cubefile, "intcube")):
            from ..spectroscopy import WCSCube
            int_cube = delayed(WCSCube.read_hdf)(
                io.e3dfilename_to_hgcubes(cubefile, "intcube"))

        else:
            saveplot_rmspull = plotbase + '_' + name + '_cigale_pullrms.png'
            saveplot_intcube = plotbase + '_' + name + '_intcube.png'
            int_cube = self.run_sedfitter(source_coutcube,
                                          redshift=redshift, working_dir=working_dir,
                                          sedfitter="cigale", ncores=ncores, lbda=SEDM_LBDA,
                                          testmode=testmode,
                                          saveplot_rmspull=saveplot_rmspull,
                                          saveplot_intcube=saveplot_intcube, sn_only=sn_only)

            # ---> Storing <--- # 3
            stored.append(int_cube.to_hdf(
                io.e3dfilename_to_hgcubes(cubefile, "intcube")))

        # ------------ #
        #    STEP 2    #
        # ------------ #
        #
        #   ADR and PSF
        # from metaslices
        # --> get the metaslices
        mcube_sedm = source_sedmcube.to_metacube(lbda_range, nslices=nslices)
        mcube_intr = int_cube.to_metacube(lbda_range, nslices=nslices)

        # ---> fit position and PSF parameters from the cutouts

        saveplot_structure = plotbase + '_' + name + '_metaslice_fit_'
        bestfit_mfit = self.fit_cube(mcube_sedm, mcube_intr, radec, nslices=nslices,
                                     saveplot_structure=saveplot_structure,
                                     mslice_param=cout_ms_param, psfmodel=psfmodel, pointsourcemodel=pointsourcemodel, jointfit=False, curved_bkgd=curved_bkgd,
                                     fix_params=['scale', 'rotation'], host_only=host_only, sn_only=sn_only, kind='metaslices', onlyvalid=True)

        # ---> Storing <--- # 4
        stored.append(bestfit_mfit.to_hdf(
            *io.get_slicefit_datafile(cubefile, "meta")))

        #badfit=bestfit_mfit[ np.logical_or(bestfit_mfit.xs('errors', axis=1)==0 , bestfit_mfit.xs('errors', axis=1)<1e-10)]
        #bestfit_mfit = bestfit_mfit.drop(np.unique(badfit.index.codes[0].values()))

        # ---> Get the object for future guesses || Guesser

        saveplot_adr = plotbase + '_' + name + '_adr_fit.png'
        saveplot_psf = plotbase + '_' + name + '_psf3d_fit.png'
        meta_ms_param = delayed(MultiSliceParameters)(bestfit_mfit, cubefile=cubefile,
                                                      psfmodel=psfmodel.replace("2D", "3D"), pointsourcemodel='GaussMoffat3D',
                                                      load_adr=True, load_psf=True, load_pointsource=True, saveplot_adr=saveplot_adr, saveplot_pointsource=saveplot_psf)

        # ------------ #
        #    STEP 3    #
        # ------------ #
        #  Ampl Fit

        bestfit_completfit = self.fit_cube(source_sedmcube, int_cube, radec, nslices=len(SEDM_LBDA),
                                           mslice_param=meta_ms_param, psfmodel=psfmodel, pointsourcemodel=pointsourcemodel, jointfit=False, curved_bkgd=curved_bkgd,
                                           saveplot_structure=None,  # plotbase+"full_fit_",
                                           fix_params=['scale', 'rotation',
                                                       "xoff", "yoff",
                                                       "a", "b", "sigma", 'a_ps', 'b_ps', 'alpha_ps', 'eta_ps'],
                                           host_only=host_only, sn_only=sn_only, kind='slices')
        # ---> Storing <--- # 5
        stored.append(bestfit_completfit.to_hdf(
            *io.get_slicefit_datafile(cubefile, "full")))

        target_specfile = io.e3dfilename_to_hgspec(cubefile, 'target')
        stored.append(self.get_target_spec(bestfit_completfit,
                      calcube.header, savefile=target_specfile))

        # ---> Get the object for future guesses || Guesser
        full_ms_param = delayed(MultiSliceParameters)(bestfit_completfit, psfmodel=psfmodel.replace("2D", "3D"), pointsourcemodel='GaussMoffat3D',
                                                      load_adr=False, load_psf=False, load_pointsource=False)

        # ------------ #
        #    STEP 4    #
        # ------------ #
        # Cube Building

        if split:

            host_sn_bkgd = self.build_cubes(int_cube, calcube, radec,
                                            meta_ms_param, full_ms_param,
                                            psfmodel=psfmodel, pointsourcemodel=pointsourcemodel, curved_bkgd=curved_bkgd, split=True)
            hostmodel = host_sn_bkgd[0]
            snmodel = host_sn_bkgd[1]
            bkgdmodel = host_sn_bkgd[2]

            stored.append(hostmodel.to_hdf(
                io.e3dfilename_to_hgcubes(cubefile, "hostmodel")))

            stored.append(snmodel.to_hdf(
                io.e3dfilename_to_hgcubes(cubefile, "snmodel")))

            stored.append(bkgdmodel.to_hdf(
                io.e3dfilename_to_hgcubes(cubefile, "bkgdmodel")))

            host_specfile = io.e3dfilename_to_hgspec(cubefile, 'host')

            stored.append(self.get_host_spec(self.get_sourcedf(radec, cubefile), source_coutcube,
                          snmodel, bkgdmodel, calcube, sourcescale=5, savefile=host_specfile))

            saveplot_coeff = plotbase + '_' + name + '_all_comp_fit.png'
            stored.append(self.show_host_ampl(bestfit_completfit, delayed(fits.getheader)(cubefile), calcube,
                          snmodel, bkgdmodel, hostmodel, self.get_sourcedf(radec, cubefile), source_coutcube, saveplot_coeff))

            saveplot_report = plotbase + '_' + name + '_global_report.png'
            stored.append(self.global_report(calcube, hostmodel, snmodel, bkgdmodel, source_coutcube, self.get_sourcedf(
                radec, cubefile), bestfit_completfit, bestfit_mfit, radec, redshift, cubefile, lbda_range, nslices, saveplot=saveplot_report))

            return stored

        cubemodel_cuberes = self.build_cubes(int_cube, calcube, radec,
                                             meta_ms_param, full_ms_param,
                                             psfmodel=psfmodel, pointsourcemodel=pointsourcemodel)
        cubemodel = cubemodel_cuberes[0]
        cuberes = cubemodel_cuberes[1]
        # ---> Storing <--- # 6
        stored.append(cubemodel.to_hdf(
            io.e3dfilename_to_hgcubes(cubefile, "model")))
        # ---> Storing <--- # 7
        stored.append(cuberes.to_hdf(
            io.e3dfilename_to_hgcubes(cubefile, "res")))

        return stored
    #
    #
    #

    @staticmethod
    def build_cubes(cube_int, cube_sedm, radec, mslice_meta, mslice_final,
                    psfmodel='Gauss2D', pointsourcemodel="GaussMoffat2D", scenemodel="SceneSlice", curved_bkgd=False, nslices=len(SEDM_LBDA), split=False, sn_only=False, host_only=False):
        """ """
        xy_in = cube_int.radec_to_xy(*radec).flatten()
        cubebuilder = delayed(CubeModelBuilder)(cube_in=cube_int, cube_comp=cube_sedm,
                                                mslice_meta=mslice_meta, mslice_final=mslice_final,
                                                xy_in=xy_in, pointsourcemodel=pointsourcemodel,
                                                scenemodel=scenemodel,
                                                sn_only=sn_only, host_only=host_only, curved_bkgd=curved_bkgd)
        if split:
            hm = []
            psm = []
            bkgm = []
            for index_ in range(nslices):
                dat = cubebuilder.get_modelslice(
                    index_, as_slice=False, split=True)
                hm.append(dat[0])
                psm.append(dat[1])
                bkgm.append(dat[2])

            hostmodel = cube_sedm.get_new(newdata=hm, newvariance="None")
            psmodel = cube_sedm.get_new(newdata=psm, newvariance="None")
            bkgdmodel = cube_sedm.get_new(newdata=bkgm, newvariance="None")

            return hostmodel, psmodel, bkgdmodel
        # Getting the data,  slice at the time
        datamodel = [cubebuilder.get_modelslice(index_, as_slice=False)
                     for index_ in range(nslices)]

        # Build the model cube
        cubemodel = cube_sedm.get_new(newdata=datamodel, newvariance="None")
        # Build the residual cubt
        cuberes = cube_sedm.get_new(
            newdata=cube_sedm.data-datamodel, newvariance="None")

        return cubemodel, cuberes

    #
    #
    #

    @classmethod
    def fit_cube(cls, cube_sedm, cube_intr, radec, nslices,
                 saveplot_structure=None,
                 mslice_param=None, initguess=None,
                 psfmodel="Gauss2D", pointsourcemodel="GaussMoffat2D",
                 curved_bkgd=True,
                 jointfit=False,
                 fix_pos=False, fix_psf=False,
                 fix_params=['scale', 'rotation'], host_only=False, sn_only=False, kind=None, onlyvalid=False):
        """ """
        if jointfit:
            raise NotImplementedError(
                "Joint fit of meta slices has not been implemented.")

        #
        # And the default positions
        xy_in = cube_intr.radec_to_xy(*radec).flatten()
        xy_comp = cube_sedm.radec_to_xy(*radec).flatten()

        mpoly = delayed(cube_sedm.get_spaxel_polygon)(format='multipolygon')

        # --------------------- #
        # Loop over the slices  #
        # --------------------- #
        best_fits = {}

        for i_ in range(nslices):
            # the slices
            slice_in = cube_intr.get_slice(index=i_, slice_object=True)
            slice_comp = cube_sedm.get_slice(index=i_, slice_object=True)
            # mpoly = delayed(slice_comp.get_spaxel_polygon)(
            #    format='multipolygon')
            gm = psf.gaussmoffat.GaussMoffat2D(**{'alpha': 2, 'eta': 1})
            ps = delayed(PointSource)(gm, mpoly)

            if mslice_param is not None:
                guess = mslice_param.get_guess(slice_in.lbda, squeeze=True)
            else:
                guess = {}

            if initguess is not None:
                guess.update(initguess)

            savefile = None if saveplot_structure is None else (
                saveplot_structure + f"{i_}.pdf")

            #
            # Update the guess
            if fix_pos:
                fix_params += ["xoff", "yoff"]

            if fix_psf:
                if psfmodel == "Gauss2D":
                    fix_params += ["a", "b", "sigma"]
                else:
                    raise NotImplementedError(
                        "Only Gauss2D psf implemented for fixing parameters")

            #mpoly = delayed(slice_comp.get_spaxel_polygon)( format='multipolygon')
            #gm = psf.gaussmoffat.GaussMoffat2D()
            #ps = delayed(PointSource)(gm, mpoly)
            #
            # fit the slices
            best_fits[i_] = delayed(SceneFitter.fit_slices_projection)(slice_in, slice_comp,
                                                                       psf=getattr(
                                                                           psf, psfmodel)(),
                                                                       savefile=savefile,
                                                                       whichscene='SceneSlice',
                                                                       pointsource=ps,
                                                                       curved_bkgd=curved_bkgd,
                                                                       xy_in=xy_in,
                                                                       xy_comp=xy_comp,
                                                                       guess=guess,
                                                                       fix_params=fix_params,
                                                                       add_lbda=True, priors=Priors(),
                                                                       host_only=host_only, sn_only=sn_only,
                                                                       kind=kind, onlyvalid=onlyvalid)

        # ------------------------ #
        # Returns the new bestfit  #
        # ------------------------ #
        return delayed(pandas.concat)(best_fits)

    #
    #
    #

    @classmethod
    def fit_metasclices(cls, cube_sedm, cube_intr, radec,
                        adr=None, adr_ref=None, psf3d=None, guess=None,
                        fix_pos=False, fix_psf=False,
                        lbda_range=[5000, 9000], nslices=6,
                        psfmodel="Gauss2D", jointfit=False,
                        fix_params=['scale', 'rotation']):
        """ """
        if jointfit:
            raise NotImplementedError(
                "Joint fit of meta slices has not been implemented.")

        if guess is None:
            guess = {}
        #
        # Get the metaslices
        mcube_sedm = cube_sedm.to_metacube(lbda_range, nslices=nslices)
        mcube_intr = cube_intr.to_metacube(lbda_range, nslices=nslices)
        #
        # And the default positions
        xy_in = mcube_intr.radec_to_xy(*radec).flatten()
        xy_comp = mcube_sedm.radec_to_xy(*radec).flatten()

        # --------------------- #
        # Loop over the slices  #
        # --------------------- #
        best_fits = {}
        for i_ in range(nslices):
            # the slices
            slice_in = mcube_intr.get_slice(index=i_, slice_object=True)
            slice_comp = mcube_sedm.get_slice(index=i_, slice_object=True)
            savefile = f"/Users/lezmy/Libraries/hypergal/notebooks/data/daskout/fit_metaslices_{i_}.pdf"

            #
            # Update the guess
            guess = delayed(cls.update_slicefit_guess)(guess, slice_comp.lbda,
                                                       adr=adr, adr_ref=adr_ref, psf3d=psf3d)
            if fix_pos and (adr is not None and adr_ref is not None):
                fix_params += ["xoff", "yoff"]

            if fix_psf and (psf3d is not None):
                if psfmodel == "Gauss2D":
                    fix_params += ["a", "b", "sigma"]
                else:
                    raise NotImplementedError(
                        "Only Gauss2D psf implemented for fixing parameters")
            #
            # fit the slices
            best_fits[i_] = delayed(SceneFitter.fit_slices_projection)(slice_in, slice_comp,
                                                                       psf=getattr(
                                                                           psf, psfmodel)(),
                                                                       savefile=savefile,
                                                                       xy_in=xy_in,
                                                                       xy_comp=xy_comp,
                                                                       guess=guess,
                                                                       fix_params=fix_params,
                                                                       add_lbda=True)
        # ------------------------ #
        # Returns the new bestfit  #
        # ------------------------ #
        return delayed(pandas.concat)(best_fits)

    @staticmethod
    def get_target_spec(fullfit, header, savefile=None):
        specval = delayed(fullfit.xs)('ampl_ps', level=1)['values'].values
        specerr = delayed(fullfit.xs)('ampl_ps', level=1)['errors'].values
        speclbda = delayed(fullfit.xs)('lbda', level=1)['values'].values
        speccoef = delayed(fullfit.xs)('norm_comp', level=1)['values'].values
        specfi = delayed(pyifu.spectroscopy.get_spectrum)(
            speclbda, specval*speccoef/header['EXPTIME'], (specerr*speccoef/header['EXPTIME'])**2, header)

        if savefile != None:
            if savefile.rsplit('.')[-1] == 'txt':
                asci = True
            elif savefile.rsplit('.')[-1] == 'fits':
                asci = False
            specfi = delayed(specfi.writeto)(savefile,  ascii=asci)

        return(specfi)

    @staticmethod
    def fit_cout_slices(source_coutcube, source_sedmcube, radec,
                        filterin=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                        filters_to_use=["ps1.r", "ps1.i", "ps1.z"],
                        saveplot_structure=None,
                        psfmodel="Gauss2D", pointsourcemodel="GaussMoffat2D",
                        host_only=False, guess=None, kind=None, onlyvalid=False):
        """ """
        #
        # Get the slices
        cout_filter_slices = {f_: source_coutcube.get_slice(index=filterin.index(f_), slice_object=True)
                              for f_ in filters_to_use}
        source_sedmcube_sub = source_sedmcube.get_partial_cube(delayed(source_sedmcube.indexes), np.argwhere(
            (SEDM_LBDA > 4500) & (SEDM_LBDA < 8800)).squeeze())

        sedm_filter_slices = {f_: source_sedmcube_sub.get_slice(lbda_max=np.max(photobasics.get_filter(f_, as_dataframe=False)[0]), lbda_min=np.min(photobasics.get_filter(f_, as_dataframe=False)[0]),
                                                                slice_object=True) for f_ in filters_to_use}

        # sedm_filter_slices = {f_: source_sedmcube.get_slice(lbda_trans=photobasics.get_filter(f_, as_dataframe=False),
        #                                                    slice_object=True)
        #                      for f_ in filters_to_use}

        xy_in = source_coutcube.radec_to_xy(*radec).flatten()
        xy_comp = source_sedmcube_sub.radec_to_xy(*radec).flatten()
        mpoly = delayed(source_sedmcube_sub.get_spaxel_polygon)(
            format='multipolygon')
        #
        # Get the slices
        best_fits = {}
        for f_ in filters_to_use:
            if saveplot_structure is not None:
                savefile = saveplot_structure + f"{f_}.pdf"
            else:
                savefile = None

            # mpoly = delayed(sedm_filter_slices[f_].get_spaxel_polygon)(
            #    format='multipolygon')
            gm = psf.gaussmoffat.GaussMoffat2D(**{'alpha': 2.5, 'eta': 1})
            ps = delayed(PointSource)(gm, mpoly)
            best_fits[f_] = delayed(SceneFitter.fit_slices_projection)(cout_filter_slices[f_],
                                                                       sedm_filter_slices[f_],
                                                                       psf=getattr(
                                                                           psf, psfmodel)(),
                                                                       savefile=savefile,
                                                                       whichscene='SceneSlice',
                                                                       pointsource=ps,
                                                                       xy_in=xy_in,
                                                                       xy_comp=xy_comp,
                                                                       guess=guess, add_lbda=True, priors=Priors(),
                                                                       host_only=host_only,
                                                                       kind=kind, onlyvalid=onlyvalid)

        # return delayed(pandas.concat)(best_fits)
        return best_fits

    def build_intcube(self, cubefile, radec, redshift,
                      savefile=None,
                      binfactor=2,
                      filters=["ps1.g", "ps1.r", "ps1.i", "ps1.z", "ps1.y"],
                      source_filter="ps1.r", source_thres=2,
                      scale_cout=15, scale_sedm=10, rmtarget=2,
                      ncores=1, testmode=True):
        """ This method enables to run only the intrinsic cube generation. """
        cubeid = io.parse_filename(cubefile)["sedmid"]
        working_dir = f"tmp_{cubeid}"
        if savefile is None:
            savefile = cubefile.replace(
                ".fits", ".h5").replace("e3d", "intcube")

        prop_sourcecube = dict(binfactor=binfactor,
                               filters=filters,
                               source_filter=source_filter,
                               source_thres=source_thres, scale_cout=scale_cout,
                               scale_sedm=scale_sedm, rmtarget=rmtarget)

        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec,
                                                                **prop_sourcecube)

        source_coutcube = source_coutcube__source_sedmcube[0]

        int_cube = self.run_sedfitter(source_coutcube,
                                      redshift=redshift, working_dir=working_dir,
                                      sedfitter="cigale", ncores=ncores, lbda=SEDM_LBDA,
                                      testmode=testmode)

        fileout = int_cube.to_hdf(savefile)

        return fileout

    def build_cout_bestfit(self, cubefile, radec,
                           binfactor=2,
                           filters=["ps1.g", "ps1.r",
                                    "ps1.i", "ps1.z", "ps1.y"],
                           source_filter="ps1.r", source_thres=2,
                           scale_cout=15, scale_sedm=10, rmtarget=2,
                           filters_fit=["ps1.r", "ps1.i", "ps1.z"],
                           psfmodel="Gauss2D"):
        """ """
        cubeid = io.parse_filename(cubefile)["sedmid"]
        working_dir = f"tmp_{cubeid}"
        savefile = io.e3dfilename_to_hgcubes(cubefile, "int")
        stored = []

        prop_sourcecube = dict(binfactor=binfactor,
                               filters=filters,
                               source_filter=source_filter,
                               source_thres=source_thres, scale_cout=scale_cout,
                               scale_sedm=scale_sedm, rmtarget=rmtarget)

        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec,
                                                                **prop_sourcecube)

        source_coutcube = source_coutcube__source_sedmcube[0]
        source_sedmcube = source_coutcube__source_sedmcube[1]
        return self.fit_cout_slices(source_coutcube, source_sedmcube, radec,
                                    filterin=filters, filters_to_use=filters_fit,
                                    psfmodel=psfmodel)

    @staticmethod
    def get_sourcedf(radec, cubefile, client=None):
        """ """
        cutout = DaskHyperGal.get_cutout(radec, cubefile, None, ['ps1.r'])
        sources = cutout.extract_sources(filter_='ps1.r', thres=20,
                                         savefile=None)

        if client is None:
            return sources
        return client.compute(sources).result()

    @staticmethod
    def get_host_spec(sourcedf, coutcube, snmodel, bkgdmodel, datacube, sourcescale=5, lbdarange=[4000, 9300], savefile=None):

        hostiso = datacube.get_new(
            newdata=datacube.data - snmodel.data - bkgdmodel.data)

        hostobsonly = hostiso.get_extsource_cube(
            sourcedf, wcsin=coutcube.wcs, wcsout=hostiso.wcs, sourcescale=sourcescale, )

        flagin = (hostobsonly.lbda > lbdarange[0]) & (
            hostobsonly.lbda < lbdarange[1])

        spec = delayed(np.nanmean)(hostobsonly.data[flagin].T, axis=0)
        spec_var = delayed(np.nanmean)(hostobsonly.variance[flagin].T, axis=0)
        spec_var_ = delayed(np.divide)(spec_var, hostobsonly.lbda[flagin].shape)
        spec_lbda = hostobsonly.lbda[flagin]

        specfi = delayed(pyifu.spectroscopy.get_spectrum)(
            spec_lbda, spec, spec_var_, hostobsonly.header)

        if savefile != None:
            if savefile.rsplit('.')[-1] == 'txt':
                asci = True
            elif savefile.rsplit('.')[-1] == 'fits':
                asci = False
            specfi = delayed(specfi.writeto)(savefile,  ascii=asci)

        return(specfi)

    @staticmethod
    @dask.delayed
    def show_host_ampl(fullparam, header, datacub, snmod, bkgdmod, hostmod, df, coutcube, saveplot=None):

        import pandas as pd
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, figsize=(15, 8))

        ax.scatter(fullparam.xs('lbda', level=1)['values'].values,
                   fullparam.xs('ampl', level=1)['values'].values*fullparam.xs('norm_comp', level=1)['values'].values/fullparam.xs('norm_in', level=1)['values'].values/header['EXPTIME'], s=4, c='b')
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.errorbar(x=fullparam.xs('lbda', level=1)['values'].values, y=fullparam.xs('ampl', level=1)['values'].values*fullparam.xs('norm_comp', level=1)['values'].values/fullparam.xs('norm_in', level=1)['values'].values / header['EXPTIME'],
                    yerr=fullparam.xs('ampl', level=1)['errors'].values*fullparam.xs('norm_comp', level=1)[
            'values'].values/fullparam.xs('norm_in', level=1)['values'].values / header['EXPTIME'],
            fmt='none', c='b', zorder=0)

        ax.set_ylabel(r'$\alpha$ (coefficent for Host Model)', color='b')
        ax.tick_params(axis='y', colors='blue')
        ax.set_xlim(4000, None)

        if len(df) > 0:

            ax2 = ax.twinx()
            hostiso = datacub.get_new(
                newdata=datacub.data - snmod.data - bkgdmod.data)
            hostobsonly = hostiso.get_extsource_cube(
                df, wcsin=coutcube.wcs, wcsout=hostiso.wcs, sourcescale=5, )
            hostmodonly = hostmod.get_extsource_cube(
                df, wcsin=coutcube.wcs, wcsout=hostiso.wcs, sourcescale=5, )
            x, y = np.transpose(hostobsonly.index_to_xy(hostobsonly.indexes))

            # ax2.plot(hostmodonly.lbda, np.nanmean(hostmodonly.data.T, axis=0)*1e15,
            #         c='forestgreen', lw=2, linestyle=(0, (3, 1, 1, 1)), label='Host Model')
            ax2.plot(hostobsonly.lbda, np.nanmean(hostobsonly.data.T,
                                                  axis=0)*1e15, label='Host Isolated', color='g')
            ax2.fill_between(hostobsonly.lbda, (np.nanmean(hostobsonly.data.T, axis=0) - (np.nanmean(hostobsonly.variance.T, axis=0)/len(hostobsonly.lbda))**0.5)
                             * 1e15, (np.nanmean(hostobsonly.data.T, axis=0) + (np.nanmean(hostobsonly.variance.T, axis=0)/len(hostobsonly.lbda))**0.5)*1e15, color='g', alpha=0.3)
            ax2.set_ylabel(
                r' Host Model ( Femto-erg Flux Unit) ($ferg.cm^{-2}.s^{-1}.\AA^{-1}$)', color='g')
            ax2.tick_params(axis='y', colors='green')

        ax3 = ax.twinx()
        bkdat = fullparam.xs('background', level=1)['values'].values
        bkerr = fullparam.xs('background', level=1)['errors'].values
        norm_comp = fullparam.xs('norm_comp', level=1)['values'].values
        bk_comp = fullparam.xs('bkgd_comp', level=1)['values'].values
        ax3.plot(fullparam.xs('lbda', level=1)[
                 'values'].values, (bkdat*norm_comp+bk_comp)*1e15, color='r', label='Sky Model')
        ax3.fill_between(x=fullparam.xs('lbda', level=1)['values'].values, y1=(bkdat*norm_comp+bk_comp)*1e15 - (bkerr*norm_comp)*1e15,
                         y2=(bkdat*norm_comp+bk_comp) *
                         1e15 + (bkerr*norm_comp)*1e15,
                         alpha=0.3, color='red')

        ax3.set_ylabel(
            r' Sky Model ( Femto-erg Flux Unit) ($ferg.cm^{-2}.s^{-1}.\AA^{-1}$)', color='r')
        ax3.tick_params(axis='y', colors='red')
        ax3.spines['right'].set_position(('outward', 60))

        ax4 = ax.twinx()

        speccoef = fullparam.xs('norm_comp', level=1)['values'].values
        specval = fullparam.xs('ampl_ps', level=1)[
            'values'].values * speccoef/header['EXPTIME']
        specerr = fullparam.xs('ampl_ps', level=1)[
            'errors'].values*speccoef/header['EXPTIME']
        speclbda = fullparam.xs('lbda', level=1)['values'].values

        ax4.plot(speclbda, specval*1e15, color='k', label=f'SN Spectrum')
        ax4.fill_between(speclbda, specval*1e15 + (specerr)*1e15,
                         specval*1e15 - (specerr)*1e15, color='k', alpha=0.3)
        ax4.set_ylabel(
            r' Target spectra ( Femto-erg Flux Unit) ($ferg.cm^{-2}.s^{-1}.\AA^{-1}$)', color='k')
        ax4.tick_params(axis='y', colors='black', )
        ax4.yaxis.tick_left()
        ax4.yaxis.set_label_position("left")
        ax4.spines['left'].set_position(('outward', 60))
        ax4.set_ylim(None, np.max(specval*1e15))
        fig.legend(loc="upper right", bbox_to_anchor=(
            1, 1), bbox_transform=ax.transAxes)
        fig.suptitle(f'Fitted Host model coefficient (blue), mean Host isolated spectra (green), Target spectrum (black) and Sky component (red)' +
                     '\n' + f'{header["NAME"]} ({header["OBSDATE"]}, ID: {header["OBSTIME"].rsplit(".")[0].replace(":","-")})', fontsize=11, fontweight="bold")
        if saveplot is not None:
            fig.savefig(saveplot)

    @staticmethod
    @dask.delayed
    def global_report(datacub, hostmod, snmod, bkgdmod, coutcube, df,
                      fullparam, metaparam, radec, redshift, cubefile,
                      lbda_range, nslices, saveplot=None):

        from matplotlib.pyplot import cm
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mainlbdarange = lbda_range
        fullmod = datacub.get_new(
            newdata=hostmod.data + snmod.data + bkgdmod.data)
        fullres = datacub.get_new(newdata=datacub.data - fullmod.data)
        hostiso = datacub.get_new(
            newdata=datacub.data - snmod.data - bkgdmod.data)
        sniso = datacub.get_new(
            newdata=datacub.data - hostmod.data - bkgdmod.data, newvariance=datacub.variance)

        mslices = sniso.to_metaslices(
            lbda_ranges=mainlbdarange, nslices=nslices, as_slice=True)
        mslicesmod = snmod.to_metaslices(
            lbda_ranges=mainlbdarange, nslices=nslices, as_slice=True)
        mslicesres = fullres.to_metaslices(
            lbda_ranges=mainlbdarange, nslices=nslices, as_slice=True)

        slid = (abs(metaparam.xs('lbda', level=1)['values']-6500)).idxmin()

        gmsn = psf.GaussMoffat2D()

        alpha = metaparam.loc[slid].xs('alpha_ps')['values']
        eta = metaparam.loc[slid].xs('eta_ps')['values']
        a_ell = metaparam.loc[slid].xs('a_ps')['values']
        b_ell = metaparam.loc[slid].xs('b_ps')['values']
        xoff = metaparam.loc[slid].xs('xoff')['values']
        yoff = metaparam.loc[slid].xs('yoff')['values']
        ampl_ps = metaparam.loc[slid].xs('ampl_ps')['values']
        norm_comp = metaparam.loc[slid].xs('norm_comp')['values']

        gmsn.update_parameters(alpha=alpha, eta=eta, a=a_ell, b=b_ell)

        p = Point(xoff, yoff)
        circle = p.buffer(8)
        idx = mslices[slid].get_spaxels_within_polygon(circle)
        idxmod = mslicesmod[slid].get_spaxels_within_polygon(circle)
        idxsn = idxmod.copy()

        mslice = mslices[slid].get_subslice(idx)
        mslicemod = mslicesmod[slid].get_subslice(idxmod)
        msliceres = mslicesres[slid].get_subslice(idx)

        xsn, ysn = np.transpose(mslice.index_to_xy(mslice.indexes))
        xcsn, ycsn = xoff, yoff
        dxsn = xsn-xcsn
        dysn = ysn-ycsn
        rsn = np.sqrt(dxsn**2 + gmsn.a_ell*dysn**2 + 2*gmsn.b_ell * (dxsn*dysn))

        sndat = mslice.data
        snerr = mslice.variance**0.5

        xgrid, ygrid = np.meshgrid(np.linspace(
            0.2, 10, 100), np.linspace(0.2, 10, 100))
        xc, yc = 0, 0
        dx = xgrid-xc
        dy = ygrid-yc
        rgrid = np.sqrt(dx**2 + gmsn.a_ell*dy**2 + 2*gmsn.b_ell * (dx*dy))

        radiusrav = rgrid.ravel()
        radiusrav.sort()
        radiusrav = radiusrav[::-1]
        profil = gmsn.get_radial_profile(radiusrav)

        x0 = metaparam.xs('xoff', level=1)['values'].values
        x0err = metaparam.xs('xoff', level=1)['errors'].values
        y0 = metaparam.xs('yoff', level=1)['values'].values
        y0err = metaparam.xs('yoff', level=1)['errors'].values
        lbda = metaparam.xs('lbda', level=1)['values'].values

        ADRFitter = spectroadr.ADRFitter(xpos=x0, ypos=y0, xpos_err=x0err, ypos_err=y0err,
                                         lbda=lbda, init_adr=spectroadr.ADR.from_header(datacub.header))

        ADRFitter.fit_adr()

        datacube = datacub.copy()
        modelcube = fullmod.copy()
        lbdamin, lbdamax = (3700, 9300)

        fig = plt.figure(figsize=(30, 20), dpi=75)
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 0.01, 1, 0.25, 1])

        gs0 = gs[0, 0].subgridspec(1, 3)
        gs0bl = gs[1, 0].subgridspec(1, 1)
        gs1 = gs[2, 0].subgridspec(1, 2)
        gs1bl = gs[3, 0].subgridspec(1, 1)
        gs2 = gs[4, 0].subgridspec(1, 3, width_ratios=[1, 0.8, 0.8], wspace=0.5)
        gs3 = gs[0, 1].subgridspec(
            1, 4, width_ratios=[1, 0.6667, 0.6667, 0.6667])
        gs3bl = gs[1, 1].subgridspec(1, 1)
        gs4 = gs[2, 1].subgridspec(1, 1)
        gs4bl = gs[3, 1].subgridspec(1, 1)
        gs5 = gs[4, 1].subgridspec(1, 1)

        axdat = fig.add_subplot(gs0[0, 0])
        axmod = fig.add_subplot(gs0[0, 1])
        axpull = fig.add_subplot(gs0[0, 2])
        axadr = fig.add_subplot(gs1[0, :])

        axhostiso = fig.add_subplot(gs2[0, 0])
        axhostisospec = fig.add_subplot(gs2[0, 1:])

        axsniso = fig.add_subplot(gs3[0, 0])
        axsnisozoom = fig.add_subplot(gs3[0, 1])
        axsnmodzoom = fig.add_subplot(gs3[0, 2])
        axsnreszoom = fig.add_subplot(gs3[0, 3])
        axsnprof = fig.add_subplot(gs4[0, 0])
        axsnspec = fig.add_subplot(gs5[0, 0])

        cmap = cm.viridis
        cmapres = cm.coolwarm
        cmaprms = cm.cividis

        lbdacond = (datacube.lbda > lbdamin) & (datacube.lbda < lbdamax)
        lbdacondrms = (datacube.lbda > 4500) & (datacube.lbda < 8500)

        slicerms = pyifu.spectroscopy.Slice.from_data(data=np.sqrt(len(datacube.data[lbdacondrms])**-1 * np.nansum(((datacube.data[lbdacondrms]-modelcube.data[lbdacondrms])/modelcube.data[lbdacondrms])**2, axis=0)),
                                                      spaxel_mapping=datacube.spaxel_mapping, spaxel_vertices=datacube.spaxel_vertices)

        slicemodel = np.array([np.nanmean(np.delete(modelcube.data[lbdacond].T[i], np.isnan(datacube.data[lbdacond].T[i])))
                               for i in range(len(modelcube.data[lbdacond].T))])

        slicedat = np.array([np.nanmean(np.delete(datacube.data[lbdacond].T[i], np.isnan(datacube.data[lbdacond].T[i])))
                             for i in range(len(datacube.data[lbdacond].T))])

        slice_err = np.sqrt(
            np.nansum(datacube.variance[lbdacond].T, axis=1))/len(datacube.data[lbdacond])
        slicepull = pyifu.spectroscopy.Slice.from_data(data=(slicedat - slicemodel)/(np.sqrt(2)*(
            slice_err)), spaxel_mapping=datacube.spaxel_mapping, spaxel_vertices=datacube.spaxel_vertices, lbda=np.mean(datacube.lbda[lbdacond]))  # PULL
        # slicepull = slicerms##RMS

        vmin = np.nanpercentile(datacube.get_slice(
            lbda_min=mainlbdarange[0], lbda_max=mainlbdarange[1]), 0.5)
        vmax = np.nanpercentile(datacube.get_slice(
            lbda_min=mainlbdarange[0], lbda_max=mainlbdarange[1]), 99.5)

        datacube._display_im_(axim=axdat, vmin=vmin, vmax=vmax,
                              lbdalim=mainlbdarange, cmap=cmap, rasterized=False)
        axdat.scatter(*ADRFitter.refract(ADRFitter.fitted_xref, ADRFitter.fitted_yref,
                      6000), marker='x', color='r', s=32, zorder=10, label='Target')
        if len(df) > 0:
            for n_ in range(len(df)):
                if np.logical_and(*abs(hostiso.wcs.all_world2pix(coutcube.wcs.all_pix2world(np.array([df.x[n_], df.y[n_]])[:, None].T, 0), 0)[0]) < (22, 22)):
                    axdat.scatter(*hostiso.wcs.all_world2pix(coutcube.wcs.all_pix2world(np.array([df.x[n_], df.y[n_]])[
                        :, None].T, 0), 0)[0], marker='x', color='k', s=32, zorder=10, label='Host')
        axdat.legend(
            *[*zip(*{l: h for h, l in zip(*axdat.get_legend_handles_labels())}.items())][::-1], loc='lower left')

        modelcube._display_im_(axim=axmod, vmin=vmin, vmax=vmax,
                               lbdalim=mainlbdarange, cmap=cmap, rasterized=False)
        axmod.scatter(*ADRFitter.refract(ADRFitter.fitted_xref,
                      ADRFitter.fitted_yref, 6000), marker='x', color='r', s=32, zorder=10)
        if len(df) > 0:
            for n_ in range(len(df)):
                if np.logical_and(*abs(hostiso.wcs.all_world2pix(coutcube.wcs.all_pix2world(np.array([df.x[n_], df.y[n_]])[:, None].T, 0), 0)[0]) < (22, 22)):
                    axmod.scatter(*hostiso.wcs.all_world2pix(coutcube.wcs.all_pix2world(np.array(
                        [df.x[n_], df.y[n_]])[:, None].T, 0), 0)[0], marker='x', color='k', s=32, zorder=10)

        slicepull.show(ax=axpull, show_colorbar=False, vmin=-6,
                       vmax=6, cmap=cmapres, rasterized=False)  # PULL
        # slicepull.show(ax=axpull, show_colorbar=False,vmin=0,vmax=0.15, cmap=cmaprms,rasterized=False); ##RMS

        axdat.set_axis_off()
        axmod.set_axis_off()
        axpull.set_axis_off()

        fraction = 0.05
        norm = mpl.colors.Normalize(vmin=-6, vmax=6)  # PULL
        # norm = mpl.colors.Normalize(vmin=0, vmax=0.15)##RMS
        cbar = axpull.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmapres),  # PULL
            # mpl.cm.ScalarMappable(norm=norm, cmap=cmaprms), ##RMS
            ax=axpull, pad=.05, extend='both', fraction=fraction)

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = axdat.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axdat, pad=.05, extend='both', fraction=fraction)

        cbar = axmod.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axmod, pad=.05, extend='both', fraction=fraction)

        cbar.set_label(r'Flux (erg unit)', color="0.5",
                       fontsize="medium", labelpad=10)

        prop = dict(loc="center", color="0.5", fontsize="medium")
        axdat.set_title('SEDM Data', **prop)
        axmod.set_title('Scene Model', **prop)
        axmod.set_aspect('equal')
        axdat.set_aspect('equal')

        axpull.set_title(r'Pull of spectraly integrated spaxels', **prop)
        axpull.set_aspect('equal')

        ADRFitter.show(ax=axadr)
        axadr.set_title(fr'$x_{{ref}}= {np.round(ADRFitter._fit_xref,2)},y_{{ref}}= {np.round(ADRFitter._fit_yref,2)}, \lambda_{{ref}}= {ADRFitter.lbdaref}\AA  $' + '\n' +
                        fr'$Airmass= {np.round( ADRFitter._fit_airmass,2)},Parangle= {np.round( ADRFitter._fit_parangle,0)}  $'
                        + '\n' + f'Input Airmass = {datacub.header["AIRMASS"]}, Input Parangle = {datacub.header["TEL_PA"]}', **prop)

        mu = 0
        sigma = 1
        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)

        if len(df) > 0:
            hostmodonly = hostmod.get_extsource_cube(
                df, wcsin=coutcube.wcs, wcsout=hostiso.wcs, sourcescale=5, )
            hostobsonly = hostiso.get_extsource_cube(
                df, wcsin=coutcube.wcs, wcsout=hostiso.wcs, sourcescale=5, )
            x, y = np.transpose(hostobsonly.index_to_xy(hostobsonly.indexes))

            flagin = (hostobsonly.lbda > 4000) & (hostobsonly.lbda < 9300)

            # axhostisospec.plot(hostmodonly.lbda[flagin], np.nanmean(
            #    hostmodonly.data[flagin].T, axis=0), c='r', label='Host Model')
            axhostisospec.plot(hostobsonly.lbda[flagin], np.nanmean(
                hostobsonly.data[flagin].T, axis=0), label='Host isolated')
            axhostisospec.fill_between(hostobsonly.lbda[flagin], np.nanmean(hostobsonly.data[flagin].T, axis=0) - (np.nanmean(hostobsonly.variance[flagin].T, axis=0)/len(hostobsonly.lbda[flagin]))**0.5,
                                       np.nanmean(hostobsonly.data[flagin].T, axis=0) + (np.nanmean(hostobsonly.variance[flagin].T, axis=0)/len(hostobsonly.lbda[flagin]))**0.5, alpha=0.5)

        vmin = np.nanpercentile(hostiso.get_slice(
            lbda_min=mainlbdarange[0], lbda_max=mainlbdarange[1]), 0.5)
        vmax = np.nanpercentile(hostiso.get_slice(
            lbda_min=mainlbdarange[0], lbda_max=mainlbdarange[1]), 99.5)

        hostiso._display_im_(
            axim=axhostiso, lbdalim=mainlbdarange, vmin=vmin, vmax=vmax)
        axhostiso.set_aspect('equal')
        if len(df) > 0:
            axhostiso.scatter(x, y, c='k', marker='D', s=4)
            prop = dict(loc="center", color="0.5", fontsize="medium")
            axhostiso.set_title(
                'Host isolated : Data - SNmodel - BKGmodel', **prop)
            axhostisospec.set_xlabel(r'Wavelength($\AA$)')
            axhostisospec.set_ylabel(r'Flux ($erg.s^{-1}.cm^{-2}.\AA^{-1}$)')

            xemlines_gal = em_lines(redshift)
            idx = [(np.abs(hostobsonly.lbda[flagin]-xl)).argmin()
                   for xl in xemlines_gal]
            yemlines_gal = np.nanmean(hostobsonly.data[flagin].T, axis=0)[idx]

            xo3lines_gal = o3_lines(redshift)
            idx = [(np.abs(hostobsonly.lbda[flagin]-xl)).argmin()
                   for xl in xo3lines_gal]
            yo3lines_gal = np.nanmean(hostobsonly.data[flagin].T, axis=0)[idx]
            xo3lines_gal = xo3lines_gal[yo3lines_gal.argmax()]
            yo3lines_gal = np.max(yo3lines_gal)

            xablines_gal = ab_lines(redshift)
            idx = [(np.abs(hostobsonly.lbda[flagin]-xl)).argmin()
                   for xl in xablines_gal]
            yablines_gal = np.nanmean(hostobsonly.data[flagin].T, axis=0)[idx]

            axhostisospec.vlines(em_lines(redshift), ymin=yemlines_gal + 0.1*np.median(np.nanmean(hostmodonly.data[flagin].T, axis=0)), ymax=yemlines_gal + 0.15*np.median(
                np.nanmean(hostmodonly.data[flagin].T, axis=0)), color='k', alpha=0.5, ls='--', label='EM/AB lines' + '\n' + f'Input z={np.round(redshift,4)}')
            axhostisospec.vlines(ab_lines(redshift), ymin=yablines_gal - 0.1*np.median(np.nanmean(
                hostmodonly.data[flagin].T, axis=0)), ymax=yablines_gal - 0.15*np.median(np.nanmean(hostmodonly.data[flagin].T, axis=0)), color='k', alpha=0.5, ls='--')
            axhostisospec.vlines(xo3lines_gal, ymin=yo3lines_gal + 0.1*np.median(np.nanmean(hostmodonly.data[flagin].T, axis=0)), ymax=yo3lines_gal + 0.15*np.median(
                np.nanmean(hostmodonly.data[flagin].T, axis=0)), color='k', alpha=0.5, ls='--')
            for l in range(len(all_em_names)):
                axhostisospec.text(em_lines(redshift)[l], yemlines_gal[l] + 0.17*np.median(np.nanmean(
                    hostmodonly.data[flagin].T, axis=0)), all_em_names[l], ha='center', va='center')

            axhostisospec.text(xo3lines_gal, yo3lines_gal + 0.17*np.median(np.nanmean(
                hostmodonly.data[flagin].T, axis=0)), r'$O[III]$', ha='center', va='center')

            for l in range(len(all_ab_names)):
                axhostisospec.text(ab_lines(redshift)[l], yablines_gal[l] - 0.17*np.median(np.nanmean(
                    hostmodonly.data[flagin].T, axis=0)), all_ab_names[l], ha='center', va='center')
            axhostisospec.legend()
            axhostiso.set_axis_off()

        rms_cub = snmod.get_new(newdata=fullres.data / fullmod.data)
        rms_subcub = rms_cub.get_partial_cube(rms_cub.indexes, np.argwhere(
            (datacub.lbda > 5000) & (datacub.lbda < 8500)).squeeze())
        rms_slice = pyifu.spectroscopy.Slice.from_data(data=np.sqrt(len(rms_subcub.data)**-1 * np.nansum((rms_subcub.data)**2, axis=0)),
                                                       spaxel_mapping=rms_subcub.spaxel_mapping, spaxel_vertices=rms_subcub.spaxel_vertices, lbda=np.mean(datacub.lbda[(datacub.lbda > 5000) & (datacub.lbda < 8500)]))

        p = Point(xoff, yoff)
        circle = p.buffer(8)
        idx = rms_slice.get_spaxels_within_polygon(circle)
        rms_slice_sub = rms_slice.get_subslice(idx)

        sl_subpull = slicepull.get_subslice(idx)

        sniso._display_im_(axim=axsniso, lbdalim=mainlbdarange)
        axsniso.set_aspect('equal')
        axsniso.scatter(xsn, ysn, c='k', marker='D', s=4)
        axsniso.set_title('SN isolated : Data - Hostmodel - BKGmodel', **prop)
        axsniso.set_axis_off()

        sniso.get_partial_cube(idx, np.arange(len(sniso.lbda)))._display_im_(
            axim=axsnisozoom, vmax='95', vmin='10', lbdalim=mainlbdarange, rasterized=False)
        axsnisozoom.set_aspect('equal')
        axsnisozoom.set_title('SN isolated', **prop)
        axsnisozoom.set_axis_off()

        snmod.get_partial_cube(idx, np.arange(len(snmod.lbda)))._display_im_(
            axim=axsnmodzoom, vmax='95', vmin='10', lbdalim=mainlbdarange, rasterized=False)
        axsnmodzoom.set_aspect('equal')
        axsnmodzoom.set_title('Model', **prop)
        axsnmodzoom.set_axis_off()

        # rms_slice_sub.show(ax=axsnreszoom, cmap=cmaprms, vmin=0, vmax=0.15, show_colorbar=False); ##RMS
        sl_subpull.show(ax=axsnreszoom, cmap=cmapres,
                        vmin=-6, vmax=6, show_colorbar=False)
        axsnreszoom.set_axis_off()
        axsnreszoom.set_aspect('equal')
        axsnreszoom.set_title(r'Pull', **prop)

        # norm = mpl.colors.Normalize(vmin=0, vmax=0.15)##RMS
        norm = mpl.colors.Normalize(vmin=-6, vmax=6)  # PULL
        cbar = axsnreszoom.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmapres),  # PULL
            # mpl.cm.ScalarMappable(norm=norm, cmap=cmaprms), ##RMS
            ax=axsnreszoom, pad=.05, extend='both', fraction=fraction)

        axsnprof.plot(radiusrav, profil*ampl_ps*norm_comp /
                      np.max(profil*ampl_ps*norm_comp), label="Gaussian + Moffat model", c='r')
        axsnprof.scatter(rsn, sndat/np.max(profil*ampl_ps *
                         norm_comp), c='k', label="Datas", s=16)
        axsnprof.errorbar(rsn, sndat/np.max(profil*ampl_ps*norm_comp), snerr /
                          np.max(profil*ampl_ps*norm_comp), fmt='none', c='k')

        axsnprof.set_xlim(np.min(rsn)-0.1, 8)
        axsnprof.set_ylim(-0.5, 1.5)
        axsnprof.set_xlabel('Elliptical Radius (spx)')
        axsnprof.set_ylabel(' Flux (normalized) ')
        axsnprof.set_title(
            fr'SN profile with Gaussian + Moffat model (from Metaslice at {np.round(mslice.lbda,0)} $\AA$)', **prop)
        axsnprof.vlines(radiusrav[np.where(abs(profil*ampl_ps*norm_comp/np.max(profil*ampl_ps*norm_comp) - 0.5) == np.min(abs(profil*ampl_ps*norm_comp/np.max(profil*ampl_ps*norm_comp) - 0.5)))[0]], -0.5, 1.5, ls='--', color='b', alpha=0.5,
                        label=fr'fitted FWHM={np.round(radiusrav[np.where(abs(profil*ampl_ps*norm_comp/np.max(sndat) -0.5) == np.min(abs(profil*ampl_ps*norm_comp/np.max(sndat) - 0.5)))[0]][0]*2*0.558,2)} " (1spx = 0.558")')
        axsnprof.legend()

        speccoef = fullparam.xs('norm_comp', level=1)['values'].values
        specval = fullparam.xs('ampl_ps', level=1)[
            'values'].values * speccoef/datacub.header['EXPTIME']
        specerr = fullparam.xs('ampl_ps', level=1)[
            'errors'].values*speccoef/datacub.header['EXPTIME']
        speclbda = fullparam.xs('lbda', level=1)['values'].values

        axsnspec.plot(speclbda, specval, label='Target model spectra ')
        axsnspec.fill_between(speclbda, specval+specerr,
                              specval-specerr, alpha=0.3)
        axsnspec.set_xlim(3800, 9300)
        axsnspec.set_ylim(
            0, np.max(specval[(speclbda > 3800) & (speclbda < 9300)]))
        axsnspec.set_xlabel(r'Wavelength($\AA$)')
        axsnspec.set_ylabel(r'Flux($ erg.s^{1}.cm^{-2}.\AA^{-1}$)')
        axsnspec.set_title('Target model spectra', **prop)

        l1t, l1b = gs1bl.get_grid_positions(
            fig)[0]+0.01, gs1bl.get_grid_positions(fig)[1]-0.01
        line = plt.Line2D((.1, .48), (l1b, l1b), color="k", linewidth=1.5)
        fig.add_artist(line)
        fig.text(0.29, (2*l1t+l1b)/3, 'Host Spectrum', fontsize=14, ha='center')
        line2 = plt.Line2D((.1, .48), (l1t, l1t), color="k", linewidth=1.5)
        fig.add_artist(line2)

        lgt, lgb = gs0.get_grid_positions(
            fig)[1]+0.05, gs0.get_grid_positions(fig)[1]+0.03
        lineg = plt.Line2D((.1, .48), (lgb, lgb), color="k", linewidth=1.5)
        fig.add_artist(lineg)
        fig.text(0.29, (lgt+2*lgb)/3, 'Global view', fontsize=14, ha='center')
        line2g = plt.Line2D((.1, .48), (lgt, lgt), color="k", linewidth=1.5)
        fig.add_artist(line2g)

        l3t, l3b = lgt, lgb
        line3 = plt.Line2D((.52, .9), (l3b, l3b), color="k", linewidth=1.5)
        fig.add_artist(line3)
        fig.text(0.71, (lgt+2*lgb)/3, 'SN Spectrum', fontsize=14, ha='center')
        line4 = plt.Line2D((.52, .9), (l3t, l3t), color="k", linewidth=1.5)
        fig.add_artist(line4)

        linesep = plt.Line2D((.5, .5), (lgb, gs2.get_grid_positions(fig)[
                             0]), color="k", linewidth=1.5)
        fig.add_artist(linesep)

        import datetime
        import hypergal
        fig.text(
            0.5, 0.01, f"hypergal version {hypergal.__version__} | made the {datetime.datetime.now().date().isoformat()} | J.Lezmy (lezmy@ipnl.in2p3.fr)", ha='center', color='grey', fontsize=10)
        fig.suptitle(
            datacube.header['NAME'] + fr' ({datacube.header["OBSDATE"]} , ID: {datacube.header["OBSTIME"].rsplit(".")[0].replace(":","-")})', fontsize=16, fontweight="bold", y=0.97)

        if saveplot is not None:
            fig.savefig(saveplot)
        else:
            return fig
