import os

from .daskbasics import DaskHyperGal
from pysedm.sedm import SEDM_LBDA

from ..photometry import basics as photobasics
from ..spectroscopy import adr as spectroadr
import pandas
from ..fit import SceneFitter, MultiSliceParameters
from ..utils.cubes import CubeModelBuilder
from .. import psf

from .. import io

from dask import delayed


class DaskHost( DaskHyperGal ):



    @classmethod
    def compute_targetcubes(cls, name, client, verbose=False, **kwargs):
        """ """
        cubefiles, radec, redshift = io.get_target_info(name, verbose=True)
        this  = cls(client=client)
        storings = [this.compute_single(cubefile_, radec, redshift)
                        for cubefile_ in cubefiles]
        return storings

    
    def compute_single(self, cubefile, radec, redshift,
                           binfactor=2,
                           filters=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                           source_filter="ps1.r", source_thres=2,
                           scale_cout=15, scale_sedm=10, rmtarget=2,
                           lbda_range=[5000, 9000], nslices=6,
                           filters_fit=["ps1.r", "ps1.i","ps1.z"],
                           psfmodel="Gauss2D", ncores=1, testmode=True):
        """ """
        info        = io.parse_filename(cubefile)
        cubeid      = info["sedmid"]
        name        = info["name"]
        filedir     = os.path.dirname(cubefile)
        # SED 
        working_dir = f"tmp_{cubeid}"
        
        # PLOTS
        plotbase    = os.path.join(filedir, "hypergal", info["name"], info["sedmid"])
        dirplotbase = os.path.dirname(plotbase)
        if not os.path.isdir(dirplotbase):
            os.makedirs(dirplotbase, exist_ok=True)
            
        # Fileout 
        stored = []

        # ------------ #
        #    STEP 1    #
        # ------------ #
        # ---> Build the cutouts, and the calibrated data cube
        calcube = self.get_calibrated_cube(cubefile, apply_byecr=True)
        
        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec,
                                                                binfactor=binfactor,
                                                                filters=filters,
                                                                source_filter=source_filter,
                                                                source_thres=source_thres, scale_cout=scale_cout,
                                                                scale_sedm=scale_sedm, rmtarget=rmtarget)
                                                        
        
        source_coutcube  = source_coutcube__source_sedmcube[0]
        source_sedmcube  = source_coutcube__source_sedmcube[1]
        
        # ---> Storing <--- # 0
        stored.append( source_sedmcube.to_hdf( io.e3dfilename_to_hgcubes(cubefile,"fitted") ))
        # ---> Storing <--- # 1        
        stored.append( calcube.to_hdf( io.e3dfilename_to_wcscalcube(cubefile) ))
        
        #
        #   Step 1.1 Cutouts
        #
        # ---> fit position and PSF parameters from the cutouts
        bestfit_cout =  self.fit_cout_slices(source_coutcube, source_sedmcube, radec,
                                                saveplot_structure = plotbase+"cout_fit_",
                                                filterin=filters, filters_to_use=filters_fit,
                                                 psfmodel=psfmodel)

        # ---> Storing <--- # 2
        stored.append( bestfit_cout.to_hdf(*io.get_slicefit_datafile(cubefile, "cutout")) )
        
        # ---> Get the object for future guesses || Guesser
        cout_ms_param = delayed(MultiSliceParameters)(bestfit_cout, cubefile=cubefile, 
                                                          psfmodel=psfmodel.replace("2D","3D"), 
                                                          load_adr=True, load_psf=True)
        
        #
        #   Step 1.2 Intrinsic Cube
        #
        # ---> SED fitting of the cutouts
        int_cube = self.run_sedfitter(source_coutcube,
                                          redshift=redshift, working_dir=working_dir,
                                          sedfitter="cigale", ncores=ncores, lbda=SEDM_LBDA,
                                          testmode=testmode)

        # ---> Storing <--- # 3
        stored.append( int_cube.to_hdf( io.e3dfilename_to_hgcubes(cubefile,"intcube") ) )

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
        bestfit_mfit = self.fit_cube(mcube_sedm, mcube_intr, radec, nslices=nslices,
                                         saveplot_structure = plotbase+"metaslice_fit_",
                                        mslice_param=cout_ms_param, psfmodel=psfmodel, jointfit=False,
                                        fix_params=['scale', 'rotation'])
        
        # ---> Storing <--- # 4
        stored.append( bestfit_mfit.to_hdf(*io.get_slicefit_datafile(cubefile, "meta")) )

        # ---> Get the object for future guesses || Guesser        
        meta_ms_param = delayed(MultiSliceParameters)(bestfit_mfit, cubefile=cubefile, 
                                                        psfmodel=psfmodel.replace("2D","3D"), 
                                                      load_adr=True, load_psf=True)

                
        # ------------ #
        #    STEP 3    #
        # ------------ #
        #  Ampl Fit
        bestfit_completfit = self.fit_cube(source_sedmcube, int_cube, radec, nslices=len(SEDM_LBDA),
                                        mslice_param=meta_ms_param, psfmodel=psfmodel, jointfit=False,
                                        saveplot_structure = plotbase+"full_fit_",
                                        fix_params=['scale', 'rotation',
                                                        "xoff", "yoff",
                                                        "a","b","sigma"])
        # ---> Storing <--- # 5
        stored.append( bestfit_completfit.to_hdf(*io.get_slicefit_datafile(cubefile, "full")) )

        # ---> Get the object for future guesses || Guesser        
        full_ms_param = delayed(MultiSliceParameters)(bestfit_completfit, psfmodel=psfmodel.replace("2D","3D") )

        # ------------ #
        #    STEP 4    #
        # ------------ #
        # Cube Building

        cubemodel_cuberes = self.build_cubes(int_cube, calcube, radec,
                                                 meta_ms_param, full_ms_param,
                                                psfmodel=psfmodel)
        cubemodel = cubemodel_cuberes[0]
        cuberes   = cubemodel_cuberes[1]
        # ---> Storing <--- # 6
        stored.append(cubemodel.to_hdf(  io.e3dfilename_to_hgcubes(cubefile,"model") ))
        # ---> Storing <--- # 7
        stored.append(cuberes.to_hdf(  io.e3dfilename_to_hgcubes(cubefile,"res") ))

        return stored
    #
    #
    #    
    @staticmethod
    def build_cubes(cube_int, cube_sedm, radec, mslice_meta, mslice_final,
                    psfmodel='Gauss2D', scenemodel="HostSlice", nslices=len(SEDM_LBDA)):
        """ """
        xy_in   = cube_int.radec_to_xy(*radec).flatten()
        cubebuilder = delayed(CubeModelBuilder)(cube_in=cube_int, cube_comp=cube_sedm,
                                                mslice_meta=mslice_meta, mslice_final=mslice_final, 
                                                xy_in=xy_in)
        # Getting the data,  slice at the time
        datamodel = [cubebuilder.get_modelslice(index_, as_slice=False)
                         for index_ in range( nslices )]
        
        # Build the model cube
        cubemodel = cube_sedm.get_new(newdata=datamodel, newvariance="None")
        # Build the residual cubt
        cuberes   = cube_sedm.get_new(newdata=cube_sedm.data-datamodel, newvariance="None")
        
        return cubemodel, cuberes
        
        
    #
    #
    #    
    @classmethod
    def fit_cube(cls, cube_sedm, cube_intr, radec, nslices,
                     saveplot_structure=None,
                     mslice_param=None,
                     psfmodel="Gauss2D",
                     jointfit=False,
                     fix_pos=False, fix_psf=False,
                     fix_params=['scale', 'rotation']):
        """ """
        if jointfit:
            raise NotImplementedError("Joint fit of meta slices has not been implemented.")

            
        #
        # And the default positions        
        xy_in   = cube_intr.radec_to_xy(*radec).flatten()
        xy_comp = cube_sedm.radec_to_xy(*radec).flatten()
        
        # --------------------- #
        # Loop over the slices  #
        # --------------------- #
        best_fits = {}
        for i_ in range(nslices):
            # the slices
            slice_in   = cube_intr.get_slice(index=i_, slice_object=True)
            slice_comp = cube_sedm.get_slice(index=i_, slice_object=True)
            if mslice_param is not None:
                guess = mslice_param.get_guess(slice_in.lbda, squeeze=True)

            savefile  = None if saveplot_structure is None else (saveplot_structure+ f"{i_}.pdf")
                
            #
            # Update the guess
            if fix_pos:
                fix_params += ["xoff","yoff"]
                
            if fix_psf:
                if psfmodel == "Gauss2D":
                    fix_params += ["a","b", "sigma"]
                else:
                    raise NotImplementedError("Only Gauss2D psf implemented for fixing parameters")
            
            #
            # fit the slices
            best_fits[i_] = delayed(SceneFitter.fit_slices_projection)(slice_in, slice_comp, 
                                                                        psf=getattr(psf,psfmodel)(), 
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
            raise NotImplementedError("Joint fit of meta slices has not been implemented.")
        

        if guess is None:
            guess = {}
        #
        # Get the metaslices
        mcube_sedm = cube_sedm.to_metacube(lbda_range, nslices=nslices)
        mcube_intr = cube_intr.to_metacube(lbda_range, nslices=nslices)
        #
        # And the default positions        
        xy_in   = mcube_intr.radec_to_xy(*radec).flatten()
        xy_comp = mcube_sedm.radec_to_xy(*radec).flatten()
        
        # --------------------- #
        # Loop over the slices  #
        # --------------------- #
        best_fits = {}
        for i_ in range(nslices):
            # the slices
            slice_in   = mcube_intr.get_slice(index=i_, slice_object=True)
            slice_comp = mcube_sedm.get_slice(index=i_, slice_object=True)
            savefile   = f"/Users/mrigault/Libraries/hypergal/notebooks/data/daskout/fit_metaslices_{i_}.pdf"

            #
            # Update the guess
            guess = delayed(cls.update_slicefit_guess)(guess, slice_comp.lbda,
                                                        adr=adr, adr_ref=adr_ref, psf3d=psf3d)
            if fix_pos and (adr is not None and adr_ref is not None):
                fix_params += ["xoff","yoff"]
                
            if fix_psf and (psf3d is not None):
                if psfmodel == "Gauss2D":
                    fix_params += ["a","b", "sigma"]
                else:
                    raise NotImplementedError("Only Gauss2D psf implemented for fixing parameters")
            #
            # fit the slices
            best_fits[i_] = delayed(SceneFitter.fit_slices_projection)(slice_in, slice_comp, 
                                                                        psf=getattr(psf,psfmodel)(), 
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
    def fit_cout_slices(source_coutcube, source_sedmcube, radec,
                          filterin=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                          filters_to_use=["ps1.r", "ps1.i","ps1.z"],
                          saveplot_structure=None,
                          psfmodel="Gauss2D", guess=None):
        """ """
        #
        # Get the slices
        cout_filter_slices = {f_: source_coutcube.get_slice(index=filterin.index(f_), slice_object=True) 
                                  for f_ in filters_to_use}

        sedm_filter_slices = {f_: source_sedmcube.get_slice(lbda_trans=photobasics.get_filter(f_, as_dataframe=False), 
                                                            slice_object=True)
                                  for f_ in filters_to_use}
        xy_in   = source_coutcube.radec_to_xy(*radec).flatten()
        xy_comp = source_sedmcube.radec_to_xy(*radec).flatten()
        #
        # Get the slices
        best_fits = {}
        for f_ in filters_to_use:
            if saveplot_structure is not None:
                savefile = saveplot_structure+"{f_}.pdf"
            else:
                savefile = None
                
            best_fits[f_] = delayed(SceneFitter.fit_slices_projection)(cout_filter_slices[f_], 
                                                                            sedm_filter_slices[f_], 
                                                                            psf=getattr(psf,psfmodel)(), 
                                                                            savefile=savefile, 
                                                                            xy_in=xy_in, 
                                                                            xy_comp=xy_comp,
                                                                            guess=guess, add_lbda=True)
            
        return delayed(pandas.concat)(best_fits)

    
    def build_intcube(self, cubefile, radec, redshift,
                          savefile=None,
                    binfactor=2,
                    filters=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                    source_filter="ps1.r", source_thres=2,
                    scale_cout=15, scale_sedm=10, rmtarget=2,
                    ncores=1, testmode=True):
        """ This method enables to run only the intrinsic cube generation. """
        cubeid      = io.parse_filename(cubefile)["sedmid"]
        working_dir = f"tmp_{cubeid}"
        if savefile is None:
            savefile = cubefile.replace(".fits",".h5").replace("e3d","intcube") 
        
        prop_sourcecube = dict(binfactor=binfactor,
                               filters=filters,
                               source_filter=source_filter,
                               source_thres=source_thres, scale_cout=scale_cout,
                               scale_sedm=scale_sedm, rmtarget=rmtarget)
        
        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec,
                                                                **prop_sourcecube)
                                                        
        source_coutcube  = source_coutcube__source_sedmcube[0]
        
        int_cube = self.run_sedfitter(source_coutcube,
                                          redshift=redshift, working_dir=working_dir,
                                          sedfitter="cigale", ncores=ncores, lbda=SEDM_LBDA,
                                          testmode=testmode)
        
        fileout = int_cube.to_hdf( savefile )
        
        return fileout

    def build_cout_bestfit(self, cubefile, radec,
                               binfactor=2,
                               filters=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                               source_filter="ps1.r", source_thres=2,
                               scale_cout=15, scale_sedm=10, rmtarget=2,
                               filters_fit=["ps1.r", "ps1.i","ps1.z"],
                               psfmodel="Gauss2D"):
        """ """
        cubeid      = io.parse_filename(cubefile)["sedmid"]
        working_dir = f"tmp_{cubeid}"
        savefile = io.e3dfilename_to_hgcubes(cubefile,"int")
        stored   = []
        
        prop_sourcecube = dict(binfactor=binfactor,
                               filters=filters,
                               source_filter=source_filter,
                               source_thres=source_thres, scale_cout=scale_cout,
                               scale_sedm=scale_sedm, rmtarget=rmtarget)
        
        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec,
                                                                **prop_sourcecube)
                                                        
        
        source_coutcube  = source_coutcube__source_sedmcube[0]
        source_sedmcube  = source_coutcube__source_sedmcube[1]
        return  self.fit_cout_slices(source_coutcube, source_sedmcube, radec,
                                     filterin=filters, filters_to_use=filters_fit,
                                     psfmodel=psfmodel)
