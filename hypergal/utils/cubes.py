
import os
import numpy as np
from .. import spectroscopy, fit, psf, io
from ..scene import basics

from pysedm.dask.base import DaskCube

class CubeModelBuilder( object ):
    """ """
    def __init__(self, cube_in, cube_comp,
                     mslice_meta, mslice_final, 
                     xy_in=None, 
                     psfmodel='Gauss2D',
                     scenemodel="HostSlice"):
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
        
    @classmethod
    def from_filename(cls, filename, radec):
        """ """
        dirout = os.path.dirname(filename)
        intcube = io.e3dfilename_to_hgcubes(cubefile,"intcube")
        
        if not os.path.isfile(intcube):
            raise IOError(f"No intrinsic cube file: {intcube}")
                        
        cube_sedm = io.get_calibrated_cube(filename)
        cube_intr = spectroscopy.WCSCube(intcube)
        
        xy_in = cube_intr.radec_to_xy(*radec).flatten()
        
        
        mslice_meta = fit.MultiSliceParameters.read_hdf(*io.get_slicefit_datafile(filename, "meta"), 
                                                        cubefile=filename, 
                                                        load_adr=True, load_psf=True)

        mslice_final = fit.MultiSliceParameters.read_hdf(*io.get_slicefit_datafile(filename, "full"))
        
        return cls(cube_in=cube_intr, cube_comp=cube_sedm, 
                   mslice_meta=mslice_meta, mslice_final=mslice_final,
                  xy_in=xy_in)

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
                **dict(self.slice_parameters_full.loc[index])}
    
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
        else:
            raise NotImplementedError("Only HostSlice scene has been implemented.")
        
        scene.update(ignore_extra=True, **slice_param)
        return scene
        
    def get_modelslice(self, index, as_slice=True, **kwargs):
        """ """
        slice_param = self.get_parameters(index)
        scene = self.get_scene(index, **kwargs)
        
        
        flux_conv      = scene.get_convolved_flux_in()
        modelflux_base = scene.overlay.get_projected_flux(flux_conv, fill_comp=True)
        modelflux_in = (modelflux_base - slice_param["bkgd_in"])/slice_param["norm_in"]
        model = (slice_param["ampl"] * modelflux_in + slice_param["background"]) * slice_param["norm_comp"] + slice_param["bkgd_comp"]
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
