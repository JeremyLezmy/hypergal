
from ..utils import geometry
import numpy as np
#import warnings

SEDM_SCALE = 0.558
PS_SCALE = 0.25
DEFAULT_SCALE_RATIO = SEDM_SCALE/PS_SCALE


# ================= #
#                   #
#    BASE SCENE     # 
#                   #
# ================= #
class _BaseScene_( object ):
    BASE_PARAMETERS = ["ampl", "background"]

    # ============= #
    #  Methods      #
    # ============= #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_overlay(self, overlay):
        """ Provide the hypergal.utils.geometry.Overlay object """
        self._overlay = overlay

    def set_psf(self, psf):
        """ Provide the hypergal.psf object. Might be PSF_2D or PSF_3D """
        self._psf = psf

    def set_fitted_data(self, which, flux, variance=None, background=0, norm=1):
        """ """
        if which == "in":
            self._bkgd_in = background
            self._norm_in = norm
            self._flux_in = flux
            self._shape_in, self._binfactor_in = self._guess_in_shape_()
            self._flux_in2d = self._flux_in.reshape(self._shape_in)
            self._variance_in = variance
            
                
        elif which == "comp":
            self._bkgd_comp = background            
            self._norm_comp = norm
            self._flux_comp = flux
            self._variance_comp = variance
        else:
            raise ValueError(f"which can only be in or comp, {which} given")
        
    def update_baseparams(self, **kwargs):
        """ 
        Set parameters from self.BASE_PARAMETERS (amplitude and background)
        """
        for k,v in kwargs.items():
            if k in self.BASE_PARAMETERS:
                self.baseparams[k] = v
            else:
                warnings.warn(f"{k} is not a base parameters, ignored")
                continue
            
    def update(self, ignore_extra=False, **kwargs):
        """ 
        Can update any parameter through kwarg option.\n
        Might be self.BASE_PARAMETER, self.PSF_PARAMETERS or self.GEOMETRY_PARAMETERS
        """
        baseparams = {}
        psfparams = {}
        geometryparams = {}
        for k,v in kwargs.items():
            # Change the baseline scene
            if k in self.BASE_PARAMETERS:
                baseparams[k] = v
                
            # Change the scene PSF
            elif k in self.PSF_PARAMETERS:
                psfparams[k] = v
                
            # Change the scene geometry                
            elif k in self.GEOMETRY_PARAMETERS:
                geometryparams[k] = v
                
            # or crash
            elif not ignore_extra:
                raise ValueError(f"Unknow input parameter {k}={v}")
            
        self.update_baseparams(**baseparams)
        if len(geometryparams)>0:
            self.overlay.change_comp(**geometryparams)
        if len(psfparams)>0:
            self.psf.update_parameters(**psfparams)
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_model(self, ampl=None, background=None,
                      overlayparam=None,
                      psfparam=None, fill_comp=False):
        """Convolves and project flux_in into the 

        Parameters
        ----------
        overlayparam: dict or None
            If dict, this is passed as kwargs to overlay

        psfparam: dict or None
            If dict, this is passed as a kwargs to self.psf.update_parameters(psfparam)

        Returns
        -------
        Array
        """
        if ampl is None:
            ampl = self.baseparams["ampl"]
            
        if background is None:
            background = self.baseparams["background"]
            
        # 1.
        # Change position of the comp grid if needed
        #   - if the overlayparam are the same as already know, no update made.
        if overlayparam is not None and len(overlayparam)>0:
            self.overlay.change_comp(**{k:v for k,v in overlayparam.items() if v is not None})

        # 2.            
        # Change values of flux and variances of _in by convolving the image
        if psfparam is not None:
            psfparam = {k:v for k,v in psfparam.items() if v is not None}
        flux_in = self.get_convolved_flux_in(psfparam)

        # 3. (overlaydf calculated only if needed)
        # Get the new projected flux and variance (_in->_comp grid)
        modelflux = self.overlay.get_projected_flux(flux_in, fill_comp=fill_comp)

        # 4. Out
        return ampl*modelflux + background
    
    def get_convolved_flux_in(self, psfconv=None):
        """ 
        Compute and return the slice_in data convolved with the setted psf object.

        Parameters
        ----------
        psfconf: dict -optional-
             Goes to self.psf.update_parameters() to update the psf parameters.\n
             Default is None.
        
        Returns
        -------
        Convolved data 2D-array
        """
        if psfconv is not None:
            self.psf.update_parameters(**psfconv)
            
        return self.psf.convolve(self._flux_in2d).flatten()

    def guess_parameters(self):
        """ 
        Return guessed parameters for all the parameters.\n
        Include BASE_PARAMETERS (amplitude and background), 
        geometrical parameters (scale, xy_in etc) 
        and psf parameters (shape and ellipticity)       
        """
        ampl = 1
        bkgd = 0
        base_guess = {**{k:None for k in self.BASE_PARAMETERS},
                      **{"ampl":ampl, "background": bkgd}
                      }
        geom_guess = self.overlay.geoparam_comp
        psf_guess  = self.psf.guess_parameters()
        guess_step1 =  {**base_guess, **geom_guess, **psf_guess}
        self.update(**guess_step1)
        
        model_comp = self.get_model()
        bkgd = np.median(self.flux_comp)-np.median(model_comp)
        ampl = np.sum(self.flux_comp)/np.sum(model_comp)
        
        return {**guess_step1, **{"ampl":ampl, "background":bkgd}}

    def get_parameters(self):
        """ """
        return {**self.baseparams,**self.overlay.geoparam_comp,**self.psf.parameters}

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def overlay(self):
        """  Overlay object between slice_in and slice_comp """
        if not hasattr(self,"_overlay"):
            raise AttributeError("No Overlay set ; see self.set/load_overlay()")
        return self._overlay

    @property
    def psf(self):
        """  PSF object to convolve on slice_in. """
        if not hasattr(self, "_psf"):
            raise AttributeError("No PSF set ; see self.set_psf()")
        
        return self._psf

    @property
    def baseparams(self):
        """  Base parameters (e.g. amplitude and background) """
        if not hasattr(self, "_baseparams"):
            self._baseparams = {k:1. if "ampl" in k else 0. for k in self.BASE_PARAMETERS}
        return self._baseparams

    # // _in prop
    @property
    def norm_in(self):
        """  Norm used to scale _in data """ # No test to gain time
        return self._norm_in
    
    @property
    def bkgd_in(self):
        """  Estimated _in background. """ # No test to gain time
        return self._bkgd_in

    @property
    def flux_in(self):
        """  Original _in data minus bkgd_in and divided by norm_in. """ # No test to gain time
        return self._flux_in

    @property
    def variance_in(self):
        """  Variance of _in scaled by norm_in. """ # No test to gain time
        return self._variance_in

    # // _comp prop
    @property
    def norm_comp(self):
        """  Norm used to scale _comp data. """ # No test to gain time
        return self._norm_comp
    
    @property
    def bkgd_comp(self):
        """ Estimated _comp background. """ # No test to gain time
        return self._bkgd_comp

    @property
    def flux_comp(self):
        """ Original _comp data minus bkgd_comp and divided by norm_comp. """ # No test to gain time
        return self._flux_comp

    @property
    def variance_comp(self):
        """  Variance of _comp scaled by norm_comp. """ # No test to gain time
        return self._variance_comp

    # ----------- #
    #  Parameters #
    # ----------- #
    @property
    def PSF_PARAMETERS(self):
        """  PSF parameters (profile + ellipticity). """
        return self.psf.PARAMETER_NAMES
    
    @property
    def GEOMETRY_PARAMETERS(self):
        """  Geometry parameters of slices (xy_in/out, scale_in/out, rotation). """
        return self.overlay.GEOMETRY_PARAMETERS
    
    @property
    def PARAMETER_NAMES(self):
        """ All parameters names """
        return self.BASE_PARAMETERS + self.GEOMETRY_PARAMETERS + self.PSF_PARAMETERS

# ================= #
#                   #
#   SLICE SCENE     # 
#                   #
# ================= #
class SliceScene( _BaseScene_ ):

    BASE_PARAMETERS = ["ampl", "background"]
    
    def __init__(self, slice_in, slice_comp, xy_in=None, xy_comp=None, load_overlay=True,
                     psf=None, adapt_flux=True, **kwargs):
        """ 
        Take a 2D (slice) as an input data + geometry, and adapt it to an output data + geometry.
        Many transformations might be done to simulate the output scene, such as psf convolution, 
        geometrical projection, background addition and amplitude factor application.
        
        Parameters
        ---------
        slice_in: pyifu.Slice
            Slice that you want to project in the out geometry.

        slice_comp: pyifu.Slice
            Slice on which you want your new scene.

        xy_in,xy_comp: 2d-array (float) or None
            Reference coordinates (target position) for the _in and _comp geometries\n
            e.g. xy_comp = [3.1,-1.3]
        
        load_overlay: bool
            Set overlay information for an exact projection between the pixels of each slice (purely geometric)\n
            Default is True.

        psf: hypergal.psf
            Set a psf object, with which slice_in will be convolve before the projection in slice_comp geometry.\n

        kwargs:
            Go to self.load_overlay
        
        """
        self.set_slice(slice_in, "in", adapt_flux=adapt_flux)
        self.set_slice(slice_comp, "comp", adapt_flux=adapt_flux)
        
        if load_overlay:
            self.load_overlay(xy_in=xy_in, xy_comp=xy_comp, **kwargs)
            
        if psf is not None:
            self.set_psf(psf)
            
    @classmethod
    def from_slices(cls, slice_in, slice_comp, xy_in=None, xy_comp=None, psf=None, **kwargs):
        """ 
        Take a 2D (slice) as an input data + geometry, and adapt it to an output data + geometry.
        Many transformations might be done to simulate the output scene, such as psf convolution, 
        geometrical projection, background addition and amplitude factor application.
        
        Parameters
        ---------
        slice_in: pyifu.Slice
            Slice that you want to project in the out geometry.

        slice_comp: pyifu.Slice
            Slice on which you want your new scene.
        
        xy_in,xy_comp: 2d-array (float) or None
            Reference coordinates (target position) for the _in and _comp geometries\n
            e.g. xy_comp = [3.1,-1.3]
       
        load_overlay: bool
            Set overlay information for an exact projection between the pixels of each slice (purely geometric) \n
            Default is True.

        psf: hypergal.psf
            Set a psf object, with which slice_in will be convolve before the projection in slice_comp geometry.

        kwargs 
            Go to self.load_overlay
        
        """
        return cls(slice_in, slice_comp,
                       xy_in=xy_in, xy_comp=xy_comp, psf=psf,
                       **kwargs)

    # ============= #
    #  Methods      #
    # ============= #
    def load_overlay(self,  xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=1/DEFAULT_SCALE_RATIO, scale_comp=1):
        """ 
        Load and set the overlay object from slice_in and slice_out (see hypergal/utils/geometry.Overlay() ).

        Parameters
        ----------
        xy_in,xy_comp: 2d-array (float) or None
            Reference coordinates (target position) for the _in and _comp geometries\n
            e.g. xy_comp = [3.1,-1.3]

        rotation_in,rotation_comp: float or None
            Rotation (in degree) or the _in and _comp geomtries
        
        scale_in,scale_comp:  float or None
            Scale of the _in and _comp geometries\n
            The scale at 1 defines the units for the offset.\n
            Since we are moving _comp, it makes sense to have this one at 1.

        Returns
        -------
        
        """
        # get all the inputs and drop the self.
        klocal = locals()
        _ = klocal.pop("self")
        overlay = geometry.Overlay.from_slices(self.slice_in, self.slice_comp,
                                                **klocal)
        self.set_overlay(overlay)
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_slice(self, slice_, which, adapt_flux=True, norm="99", bkgd="50"):
        """ Set the 'in' or 'comp' geometry
        
        Parameters
        ----------
        slice: pyifu.Slice
            Spaxelhandler object (Slice)

        which: string
            Which geometry are you providing (in or comp)?\n
            A ValueError is raise if which is not 'in' or 'comp'

        Returns
        -------
        None
        """
        data = slice_.data.copy()
        if not adapt_flux:
            norm = 1
            bkgd = 0
            
        if type(bkgd) is str:
            bkgd  = np.percentile(data[data==data], float(bkgd))
            
        data -= bkgd
        if type(norm) is str:
            norm = np.percentile(data[data==data], float(norm))
            
        data /= norm
        if slice_.has_variance():
            variance = slice_.variance/norm**2
        else:
            variance = None

        if which == "in":
            self._slice_in = slice_                
        elif which == "comp":
            self._slice_comp = slice_
        else:
            raise ValueError(f"which can be 'in' or 'comp', {which} given")

        self.set_fitted_data(which=which, flux=data, variance=variance, norm=norm, background=bkgd)
        
        
    def show(self, savefile=None, titles=True,
                 res_as_ratio=False, cutout_convolved=True,
                 vmin="1", vmax="99", cmap="cividis", cmapproj=None,
                 fill_comp=True):
        """
        General plot of the process. Will show 4 Axes. \n
        First one is slice_in + empty geometry of slice_out.\n
        Second one is the new scene (after projection + if choosen convolution).\n
        Third one is the slice_comp.\n
        Fourth one is residual between new scene and slice_comp.
        
        Parameters
        ----------
        savefile: string
            Where to save if not None.\n
            Default is None.
        
        titles: bool
            Display title for each Axes?
        
        res_as_ratio: bool
            If True, divide the residual by the model values.

        cutout_convolved: bool
            If True, display the convolved slice_in instead of the original 
            slice_in in the first Axes.
        
        vmin, vmax: string or float/int
            For the colormap scale.\n
            If string, the corresponding percentile is computed\n
            Otherwise, directly use the input values.
       
        Returns
        -------
        Figure
        
        """
        
        # Convolved image flux
        flux_in    = self.get_convolved_flux_in() if cutout_convolved else self.flux_in
        flux_model = self.get_model(fill_comp=fill_comp)
        flux_comp  = self.flux_comp
    
        return self._show_slice_scene_(self.overlay, flux_in, flux_model, flux_comp,
                                           axes=None,
                                           savefile=savefile, vmin=vmin, vmax=vmax,
                                           cmap=cmap, cmapproj=cmapproj,
                                           cutout_convolved=cutout_convolved,
                                           res_as_ratio=res_as_ratio, titles=titles)

    @staticmethod
    def _show_slice_scene_(overlay,  flux_in, flux_model, flux_comp,
                               axes=None, savefile=None,
                               cutout_convolved=True, 
                               vmin="1", vmax="99", cmap="viridis", cmapproj=None,
                               res_as_ratio=False, titles=True, index=None):
        """ """
        from ..utils import tools
        if cmapproj is None:
            cmapproj = cmap

        if axes is not None:
            ax, axm, axd, axr = axes
            fig = ax.figure
        else:
            import matplotlib.pyplot as mpl
            fig = mpl.figure(figsize=[9,2.7])
            left, witdth = 0.05, 0.21
            height = 0.8
            spanx, extraspanx =0.018, 0.03
            ax = fig.add_axes([left+0*(witdth+spanx)            ,0.12,witdth*1.1, height])
            axm = fig.add_axes([left+1*(witdth+spanx)+extraspanx,0.12,witdth, height])
            axd = fig.add_axes([left+2*(witdth+spanx)+extraspanx,0.12,witdth, height])
            axr = fig.add_axes([left+3*(witdth+spanx)+extraspanx,0.12,witdth, height])
            
        axifu = [axm, axd, axr]
        
        
        overlay.show(ax=ax, comp_index=index, flux_in = flux_in, lw_in=0,
                         adjust=True, cmap=cmapproj)
        vmin, vmax = tools.parse_vmin_vmax(flux_model, vmin, vmax)
        
        prop = {"cmap":cmap, "vmin":vmin, "vmax":vmax, "lw":0.1, "edgecolor":"0.7",
                    "adjust":True, "index":index}
        overlay.show_mpoly("comp", ax=axm, flux=flux_model, **prop)
        overlay.show_mpoly("comp", ax=axd, flux=flux_comp, **prop)

        res = (flux_comp-flux_model)
        if res_as_ratio:
            res /=  flux_model
            prop = {**prop,**{"vmin":-0.5, "vmax":0.5, "cmap":"coolwarm"}}
            title_res = "Residual (data-scene)/scene [±50%]"
        else:
            title_res = "Residual (data-scene)"
            
        overlay.show_mpoly("comp", ax=axr, flux=res, **prop)
        
        clearwhich = ["left","right","top","bottom"]
        for ax_ in axifu:
            ax_.set_yticks([])
            ax_.set_xticks([])        
            [ax_.spines[which].set_visible(False) for which in clearwhich]

        if titles:
            prop = dict(loc="left", color="0.5", fontsize="small")
            ax.set_title("Projection"+ " (convolved)" if cutout_convolved else "", **prop)
            axm.set_title("Projected Scene", **prop)
            axd.set_title("Data", **prop)
            axr.set_title(title_res, **prop)

        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
            
    # ============= #
    #  Internal     #
    # ============= #
    def _guess_in_shape_(self):
        """ """
        slicexy  = self.slice_in.xy
        [xmin,ymin], [xmax, ymax] = np.percentile(slicexy, [0,100], axis=1)
        
        steps = np.unique(np.diff(slicexy, axis=1))
        steps = steps[steps>0]
        if len(steps)!=1:
            raise ValueError(f"Cannot guess the binfactor. 1 entry expected, {steps} obtained.")
        else:
            binfactor = steps[0]

        shape = int((xmax-xmin)/binfactor)+1, int((ymax-ymin)/binfactor)+1
        return shape, binfactor
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def slice_in(self):
        """  Slice_in object. """
        return self._slice_in
    
    @property    
    def slice_comp(self):
        """ Slice_comp object. """
        return self._slice_comp

    
# ================= #
#                   #
#   CUBE SCENE      # 
#                   #
# ================= #
class CubeScene( SliceScene ):
    
    BASE_SLICE_PARAMETERS = ["ampl", "background"]
    
    def __init__(self, cube_in, cube_comp, 
                 xy_in=None, xy_comp=None, 
                 load_overlay=True,
                 psf=None, **kwargs):
        """ 
        Take a 2D (slice) as an input data + geometry, and adapt it to an output data + geometry.
        Many transformations might be done to simulate the output scene, such as psf convolution, 
        geometrical projection, background addition and amplitude factor application.
        
        Parameters
        ---------
        slice_in: pyifu.Slice
            Slice that you want to project in the out geometry.

        slice_comp: pyifu.Slice
            Slice on which you want your new scene.

        xy_in,xy_comp: 2d-array (float) or None
            Reference coordinates (target position) for the _in and _comp geometries\n
            e.g. xy_comp = [3.1,-1.3]
        
        load_overlay: bool
            Set overlay information for an exact projection between the pixels of each slice (purely geometric)\n
            Default is True.

        psf: hypergal.psf
            Set a psf object, with which slice_in will be convolve before the projection in slice_comp geometry.\n

        kwargs:
            Go to self.load_overlay
        
        """
        self.set_cube(cube_in,  "in")
        self.set_cube(cube_comp,"comp")
        
        if load_overlay:
            self.load_overlay(xy_in=xy_in, xy_comp=xy_comp, **kwargs)
            
        if psf is not None:
            self.set_psf(psf)
            
    @classmethod
    def from_cubes(cls, cube_in, cube_comp, xy_in=None, xy_comp=None, psf=None, **kwargs):
        """ 
        Take a 2D (slice) as an input data + geometry, and adapt it to an output data + geometry.
        Many transformations might be done to simulate the output scene, such as psf convolution, 
        geometrical projection, background addition and amplitude factor application.
        
        Parameters
        ---------
        slice_in: pyifu.Slice
            Slice that you want to project in the out geometry.

        slice_comp: pyifu.Slice
            Slice on which you want your new scene.
        
        xy_in,xy_comp: 2d-array (float) or None
            Reference coordinates (target position) for the _in and _comp geometries\n
            e.g. xy_comp = [3.1,-1.3]
       
        load_overlay: bool
            Set overlay information for an exact projection between the pixels of each slice (purely geometric) \n
            Default is True.

        psf: hypergal.psf
            Set a psf object, with which slice_in will be convolve before the projection in slice_comp geometry.

        kwargs 
            Go to self.load_overlay
        
        """
        return cls(cube_in, cube_comp,
                    xy_in=xy_in, xy_comp=xy_comp, psf=psf,
                    **kwargs)

    # ============= #
    #  Methods      #
    # ============= #
    def load_overlay(self,  xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=1/DEFAULT_SCALE_RATIO, scale_comp=1):
        """ 
        Load and set the overlay object from slice_in and slice_out (see hypergal/utils/geometry.Overlay() ).

        Parameters
        ----------
        xy_in,xy_comp: 2d-array (float) or None
            Reference coordinates (target position) for the _in and _comp geometries\n
            e.g. xy_comp = [3.1,-1.3]

        rotation_in,rotation_comp: float or None
            Rotation (in degree) or the _in and _comp geomtries
        
        scale_in,scale_comp:  float or None
            Scale of the _in and _comp geometries\n
            The scale at 1 defines the units for the offset.\n
            Since we are moving _comp, it makes sense to have this one at 1.

        Returns
        -------
        
        """
        # get all the inputs and drop the self.
        klocal = locals()
        _ = klocal.pop("self")
        overlay = geometry.OverlayADR.from_cubes(self.cube_in, self.cube_comp,
                                                  spaxel_comp_unit=SEDM_SCALE,
                                                    **klocal)
        self.set_overlay(overlay)
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_cube(self, cube_, which, norm="99", bkgd="50"):
        """ Set the 'in' or 'comp' geometry
        
        Parameters
        ----------
        slice: pyifu.Slice
            Spaxelhandler object (Slice)

        which: string
            Which geometry are you providing (in or comp)?\n
            A ValueError is raise if which is not 'in' or 'comp'

        Returns
        -------
        None
        """
        data = cube_.data.copy()
        if type(bkgd) is str:
            bkgd  = np.percentile(data[data==data], float(bkgd), axis=1)[:,None]
            
        data -= bkgd
        if type(norm) is str:
            norm = np.percentile(data[data==data], float(norm), axis=1)[:,None]
            
        data /= norm
        if cube_.has_variance():
            variance = cube_.variance/norm**2
        else:
            variance = None

        if which == "in":
            self._cube_in = cube_                
        elif which == "comp":
            self._cube_comp = cube_
        else:
            raise ValueError(f"which can be 'in' or 'comp', {which} given")

        self.set_fitted_data(which=which, flux=data, variance=variance, 
                             norm=norm, background=bkgd)
        
    # ============= #
    #  Internal     #
    # ============= #
    def get_amplitudes(self):
        """ """
        return np.asarray([self.baseparams[f"ampl{i}"] for i in range(self.nslices)])
    
    def get_backgrounds(self):
        """ """
        return np.asarray([self.baseparams[f"background{i}"] for i in range(self.nslices)])
    
    def get_model(self, ampl=None, background=None,
                        overlayparam=None,
                        psfparam=None, **kwargs):
        """Convolves and project flux_in into the 

        Parameters
        ----------
        overlayparam: dict or None
            If dict, this is passed as kwargs to overlay

        psfparam: dict or None
            If dict, this is passed as a kwargs to self.psf.update_parameters(psfparam)

        kwargs enables you to pass ampl and background as k-arguments
        Returns
        -------
        Array
        """
        if len(kwargs)>0:
            try:
                ampl = np.asarray([kwargs[f"ampl{i}"] for i in range(self.nslices)])
                background = np.asarray([kwargs[f"background{i}"] for i in range(self.nslices)])
            except:
                warnings.warn("Cannot use the kwargs for ampl and background definition")
                    
        if ampl is None:
            ampl = self.get_amplitudes()
            
        if background is None:
            background = self.get_backgrounds()
            
        # 1.
        # Change position of the comp grid if needed
        #   - if the overlayparam are the same as already know, no update made.
        if overlayparam is not None and len(overlayparam)>0:
            self.overlay.change_comp(**{k:v for k,v in overlayparam.items() if v is not None})

        # 2.            
        # Change values of flux and variances of _in by convolving the image
        if psfparam is not None:
            psfparam = {k:v for k,v in psfparam.items() if v is not None}
        flux_in = self.get_convolved_flux_in(psfparam)
        # 3. (overlaydf calculated only if needed)
        # Get the new projected flux and variance (_in->_comp grid)
        modelflux = self.overlay.get_projected_flux(flux_in)

        # 4. Out
        return ampl[:,None]*modelflux + background[:,None]
    
    def get_convolved_flux_in(self, psfconv=None):
        """ 
        Compute and return the slice_in data convolved with the setted psf object.

        Parameters
        ----------
        psfconf: dict -optional-
             Goes to self.psf.update_parameters() to update the psf parameters.\n
             Default is None.
        
        Returns
        -------
        Convolved data 2D-array
        """
        if psfconv is not None:
            self.psf.update_parameters(**psfconv)

        return np.asarray(self.psf.convolve(self._flux_in2d, lbda=self.cube_in.lbda)).reshape(self.flux_in.shape)

    def guess_parameters(self):
        """ 
        Return guessed parameters for all the parameters.\n
        Include BASE_PARAMETERS (amplitude and background), 
        geometrical parameters (scale, xy_in etc) 
        and psf parameters (shape and ellipticity)       
        """
        base_guess = self.baseparams.copy()
        geom_guess = self.overlay.geoparam_comp
        psf_guess  = self.psf.guess_parameters()
        guess_step1 =  {**base_guess, **geom_guess, **psf_guess}
        
        self.update(**guess_step1)
        
        model_comp = self.get_model()
        bkgd = np.median(self.flux_comp, axis=1)-np.median(model_comp, axis=1)
        ampl = np.percentile(self.flux_comp, 95, axis=1) / np.percentile(model_comp, 95, axis=1)
        baseparams = {**{f"ampl{i}":ampl[i]       for i in range(self.nslices)},
                      **{f"background{i}":bkgd[i] for i in range(self.nslices)} }
        return {**guess_step1, **baseparams}



    def show(self, index=None, savefile=None, titles=True, 
                 res_as_ratio=False, cutout_convolved=True,
                 vmin="1", vmax="99", cmap="cividis", cmapproj=None):
        """
        General plot of the process. Will show 4 Axes. \n
        First one is slice_in + empty geometry of slice_out.\n
        Second one is the new scene (after projection + if choosen convolution).\n
        Third one is the slice_comp.\n
        Fourth one is residual between new scene and slice_comp.
        
        Parameters
        ----------
        savefile: string
            Where to save if not None.\n
            Default is None.
        
        titles: bool
            Display title for each Axes?
        
        res_as_ratio: bool
            If True, divide the residual by the model values.

        cutout_convolved: bool
            If True, display the convolved slice_in instead of the original slice_in in the first Axes.
        
        vmin, vmax: string or float/int
            For the colormap scale.\n
            If string, the corresponding percentile is computed\n
            Otherwise, directly use the input values.
       
        Returns
        -------
        Figure
        
        """
        import matplotlib.pyplot as mpl
        
        if index is None:
            index = np.arange(self.nslices)
        else:
            index = np.atleas_1d(index)
            
        nindex = len(index)
        
        # Figure Definition
        fig = mpl.figure(figsize=[9,2.7*nindex])
        left, witdth = 0.05, 0.21
        height = 0.8/nindex
        spanx, extraspanx = 0.018, 0.03
        for i,index_ in enumerate(index):
            bottom = (0.12 + i)/nindex
            ax = fig.add_axes([left+0*(witdth+spanx)            , bottom, witdth*1.1, height])
            axm = fig.add_axes([left+1*(witdth+spanx)+extraspanx, bottom, witdth, height])
            axd = fig.add_axes([left+2*(witdth+spanx)+extraspanx, bottom, witdth, height])
            axr = fig.add_axes([left+3*(witdth+spanx)+extraspanx, bottom, witdth, height])

            flux_in = (self.get_convolved_flux_in() if cutout_convolved else self.flux_in)[index_]
            flux_model = self.get_model()[index_]
            flux_comp = self.flux_comp[index_]
            _ = self._show_slice_scene_(self.overlay, flux_in, flux_model, flux_comp,
                                           axes=[ax, axm,axd,axr], savefile=None,
                                           cutout_convolved=cutout_convolved, 
                                           vmin=vmin, vmax=vmax, cmap=cmap, cmapproj=cmapproj,
                                           res_as_ratio=res_as_ratio, titles=titles, 
                                           index=index_)
        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
    

    # ============= #
    #  Internal     #
    # ============= #
    def _guess_in_shape_(self):
        """ """
        slicexy  = self.cube_in.xy
        [xmin,ymin], [xmax, ymax] = np.percentile(slicexy, [0,100], axis=1)

        steps = np.unique(np.diff(slicexy, axis=1))
        steps = steps[steps>0]
        if len(steps)!=1:
            raise ValueError(f"Cannot guess the binfactor. 1 entry expected, {steps} obtained.")
        else:
            binfactor = steps[0]

        shape = int((xmax-xmin)/binfactor)+1, int((ymax-ymin)/binfactor)+1
        return [len(self.cube_in.data)]+list(shape), binfactor
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def cube_in(self):
        """  Slice_in object. """
        return self._cube_in
    
    @property    
    def cube_comp(self):
        """ Slice_comp object. """
        return self._cube_comp

    @property
    def nslices(self):
        """ """
        return self.overlay.nslices

    @property
    def BASE_PARAMETERS(self):
        """ """
        return [f"{k}{i}" for k in self.BASE_SLICE_PARAMETERS for i in range(self.nslices)]

    
