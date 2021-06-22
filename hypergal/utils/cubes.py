
import os
import numpy as np
from .. import spectroscopy, fit, psf, io
from ..scene import basics
from ..scene.basics import PointSource, BackgroundCurved
from pysedm.dask.base import DaskCube

class CubeModelBuilder( object ):
    """ """
    def __init__(self, cube_in, cube_comp,
                     mslice_meta, mslice_final, 
                     xy_in=None, 
                     psfmodel='Gauss2D', pointsourcemodel='GaussMoffat2D',
                     scenemodel="HostSlice", curved_bkgd=False):
        """ """
        # - Cubes        
        self.set_cube(cube_in, "in")
        self.set_cube(cube_comp, "comp")
        
        # - Meta Slices
        self.set_mslice_parameters(mslice_meta, which="meta")
        self.set_mslice_parameters(mslice_final, which="full")
        if xy_in is not None:
            self.set_xyin(xy_in)
        
        self._psfmodel = psfmodel
        self._scenemodel = scenemodel
        self._pointsourcemodel = pointsourcemodel if scenemodel in ['SceneSlice', 'SceneCube'] else None
        self._curved_bkgd = curved_bkgd
        
        
    @classmethod
    def from_filename(cls, filename, radec, scenemodel='HostSlice'):
        """ """
        dirout = os.path.dirname(filename)
        intcube = io.e3dfilename_to_hgcubes(cubefile,"intcube")
        
        if not os.path.isfile(intcube):
            raise IOError(f"No intrinsic cube file: {intcube}")
                        
        cube_sedm = io.get_calibrated_cube(filename)
        cube_intr = spectroscopy.WCSCube(intcube)
        
        xy_in = cube_intr.radec_to_xy(*radec).flatten()
        
        load_pointsource = True if scenemodel in ['SceneSlice', 'SceneCube'] else False
        
        mslice_meta = fit.MultiSliceParameters.read_hdf(*io.get_slicefit_datafile(filename, "meta"), 
                                                        cubefile=filename, 
                                                        load_adr=True, load_psf=True, load_pointsource = load_pointsource)

        mslice_final = fit.MultiSliceParameters.read_hdf(*io.get_slicefit_datafile(filename, "full"))
        
        return cls(cube_in=cube_intr, cube_comp=cube_sedm, 
                   mslice_meta=mslice_meta, mslice_final=mslice_final,
                  xy_in=xy_in, scenemodel=scenemodel)

    # ================ #
    #   Methods        #
    # ================ #
    # --------- #
    #  SETTER   #
    # --------- #
    def set_xyin(self, xy_in):
        """ """
        self._xyin = xy_in
        
    def set_cube(self, cube, which):
        """ """
        if which == "in":
            self._cube_in = cube
        elif which == "comp":
            self._cube_comp = cube
        else:
            raise ValueError(f"which must be in or comp, {which} given")
    
    def set_mslice_parameters(self, mslice, which):
        """ """
        if which == "meta":
            self._msliceparam_meta = mslice
        elif which == "full":
            self._msliceparam_full = mslice
        else:
            raise ValueError(f"which must be meta or full, {which} given")
    
    # --------- #
    #  GETTER   #
    # --------- #
    def get_lbda(self, index):
        """ """
        return self.lbda[index]
    
    def get_parameters(self, index):
        """ """
        return {**self.slice_parameters_meta.get_guess( self.get_lbda(index) ), 
                **dict(self.slice_parameters_full.values.loc[index])}
    
    def get_scene(self, index, adapt_flux=False, **kwargs):
        """ """
        slice_param = self.get_parameters(index)
        slice_in    = self.cube_in.get_slice(index=index, slice_object=True)
        slice_comp  = self.cube_comp.get_slice(index=index, slice_object=True)
        
        psfmodel = getattr(psf, self._psfmodel)()
        xy_comp = [slice_param["xoff"],slice_param["yoff"]]
        
        if self._scenemodel == "HostSlice":
            from hypergal.scene import host
            scene = host.HostSlice.from_slices(slice_in, slice_comp, 
                                               xy_in=self.xy_in, xy_comp=xy_comp, 
                                               psf=psfmodel, adapt_flux=adapt_flux, 
                                               **kwargs)
            
        elif self._scenemodel == "SceneSlice":
            
            from hypergal.scene import host
            pointsourcemodel = getattr(psf, self._pointsourcemodel)()
            pointsource = PointSource( pointsourcemodel, self.mpoly_comp )
            scene = host.SceneSlice.from_slices(slice_in, slice_comp, 
                                               xy_in=self.xy_in, xy_comp=xy_comp, 
                                                psfgal=psfmodel, pointsource=pointsource, adapt_flux=adapt_flux, curved_bkgd=self._curved_bkgd,
                                               **kwargs)
        else:
            raise NotImplementedError("Only HostSlice and SceneSlice scenes have been implemented.")
        
        scene.update(ignore_extra=True, **slice_param)
        return scene
        
    def get_modelslice(self, index, as_slice=True, split=False, **kwargs):
        """ """
        slice_param = self.get_parameters(index)
        scene = self.get_scene(index, **kwargs)
        
        
        flux_conv      = scene.get_convolved_flux_in()
        modelflux_base = scene.overlay.get_projected_flux(flux_conv, fill_comp=True)
        
        if hasattr(scene, 'pointsource'):
            psmod=scene.pointsource.get_model(**{k.replace('_ps',''):v for k,v in slice_param.items() if k in scene.pointsource.BASE_PS_PARAMETERS + scene.pointsource.PSF_PARAMETERS + scene.GEOMETRY_PARAMETERS and '_err' not in k})
        else:
            psmod=0                                                      

        if (self._scenemodel=="SceneSlice" and  not scene.has_curved_bkgd) or self._scenemodel!="SceneSlice":
           bkgdmod = np.repeat(slice_param["background"],len(modelflux_base))
        elif self._scenemodel=="SceneSlice" and scene.has_curved_bkgd:

            x,y = scene.slice_comp_xy
            coeffs = dict({k:v for k,v in slice_param.items() if k in BackgroundCurved.BACKGROUND_PARAMETERS})
            bkgdmod =  BackgroundCurved.get_background(x, y, coeffs )
                
        
        modelflux_in = (modelflux_base - slice_param["bkgd_in"])/slice_param["norm_in"]
        model = (slice_param["ampl"] * modelflux_in + bkgdmod + psmod) * slice_param["norm_comp"] + slice_param["bkgd_comp"]

        if split and hasattr(scene, 'pointsource'):
            
            modelhost = (slice_param["ampl"] * modelflux_in ) * slice_param["norm_comp"] #+ slice_param["bkgd_comp"]
            modelps = (psmod) * slice_param["norm_comp"] #+ slice_param["bkgd_comp"]
            modelbkgd = bkgdmod * slice_param["norm_comp"] + slice_param["bkgd_comp"]
            
            return modelhost, modelps, modelbkgd
        
        if as_slice:
            return scene.slice_comp.get_new(newdata=model, newvariance="None")
        
        return model
        
    def get_modelcube(self, client=None):
        """ """
        if client is not None:
            import dask
            d_modeldata = [dask.delayed(self.get_modelslice)(index_, as_slice=False)
                           for index_ in range( len(cmodel.cube_comp.lbda)) ]
            return d_modeldata
        else:
            modeldata = [self.get_modelslice(index_, as_slice=False)
                             for index_ in range( len(cmodel.cube_comp.lbda)) ]
        return modeldata
            
    def get_slices(self, index):
        """ return the data, model and residual slices """
        sldata = self.cube_comp.get_slice(index=index, slice_object=True)
        data   = np.asarray(sldata.data)
        
        model   = self.get_modelslice(index, as_slice=False)
        slmodel = sldata.get_new(newdata=model, newvariance="None")
        slres   = sldata.get_new(newdata=data-model)
        
        return sldata, slmodel, slres

    def show_slice(self, index):
        """ """
        import matplotlib.pyplot as mpl
        sldata, slmodel, slres = self.get_slices(index)

        fig = mpl.figure(figsize=[9,3])
        axd  = fig.add_axes([0.1,0.1,0.25, 0.8])
        axm  = fig.add_axes([0.4,0.1,0.25, 0.8])
        axr  = fig.add_axes([0.7,0.1,0.25, 0.8])

        sldata.show(ax=axd, show_colorbar=False)
        slmodel.show(ax=axm, show_colorbar=False)
        slres.show(ax=axr, show_colorbar=False)

        return fig
    # --------- #
    #  PLOTTER  #
    # --------- #
    
    def has_pointsource(self):
        """ """
        return True if self._scenemodel in ['SceneSlice', 'SceneCube'] else False
        
    # ================ #
    #   Properties     #
    # ================ #
    # - Cubes
    @property
    def cube_in(self):
        """ """
        return self._cube_in
    
    @property
    def cube_comp(self):
        """ """
        return self._cube_comp
    
    @property
    def mpoly_comp(self):
        """ """
        return self.cube_comp.get_spaxel_polygon( format='multipolygon')
    
    @property
    def lbda(self):
        """ """
        return self.cube_in.lbda
    
    # - Paramerters
    @property
    def slice_parameters_meta(self):
        """ """
        return self._msliceparam_meta
    
    @property
    def slice_parameters_full(self):
        """ """
        return self._msliceparam_full
    
    @property
    def xy_in(self):
        """ """
        return self._xyin


