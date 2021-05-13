
from ..utils import geometry
import numpy as np
#import warnings

DEFAULT_SCALE_RATIO = 0.558/0.25 #SEDm/PS1


# ================= #
#                   #
#   SLICE SCENE     # 
#                   #
# ================= #
class SliceScene( object ):

    BASE_PARAMETERS = ["ampl", "background"]
    
    def __init__(self, slice_in, slice_comp, xy_in=None, xy_comp=None, load_overlay=True,
                     psf=None, **kwargs):
        """ """
        self.set_slice(slice_in, "in")
        self.set_slice(slice_comp, "comp")
        
        if load_overlay:
            self.load_overlay(xy_in=xy_in, xy_comp=xy_comp, **kwargs)
            
        if psf is not None:
            self.set_psf(psf)
            
    @classmethod
    def from_slices(cls, slice_in, slice_comp, xy_in=None, xy_comp=None, psf=None, **kwargs):
        """ """
        return cls(slice_in, slice_comp,
                       xy_in=xy_in, xy_comp=xy_comp, psf=psf,
                       **kwargs)

    # ============= #
    #  Methods      #
    # ============= #
    def load_overlay(self,  xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=None, scale_comp=DEFAULT_SCALE_RATIO):
        """ """
        # get all the inputs and drop the self.
        klocal = locals()
        _ = klocal.pop("self")
        overlay = geometry.Overlay.from_slices(self.slice_in, self.slice_comp,
                                                **klocal)
        self.set_overlay(overlay)
        
    # --------- #
    #  SETTER   #
    # --------- #
    def set_slice(self, slice_, which, norm="99", bkgd="50"):
        """ set the 'in' or 'comp' geometry
        
        Parameters
        ----------
        slice: [pyifu.Slice]
            spaxelhandler object (Slice)

        which: [string]
            Which geometry are you providing ?
            - in or which.
            a ValueError is raise if which is not 'in' or 'comp'

        Returns
        -------
        None
        """
        data = slice_.data.copy()
        if type(bkgd) is str:
            bkgd  = np.percentile(data, float(bkgd))
            
        data -= bkgd
        if type(norm) is str:
            norm = np.percentile(data, float(norm))
            
        data /= norm
        
        if which == "in":
            self._slice_in = slice_
            self._bkgd_in = bkgd
            self._norm_in = norm
            self._flux_in = data
            nshape = int(np.sqrt(len(self._flux_in)))
            self._flux_in2d = self._flux_in.reshape(nshape,nshape)
            if slice_.has_variance():
                self._variance_in = slice_.variance/norm**2
            else:
                self._variance_in = None
                
        elif which == "comp":
            self._slice_comp = slice_
            self._bkgd_comp = norm            
            self._norm_comp = norm
            self._flux_comp = data
            if slice_.has_variance():
                self._variance_comp = slice_.variance/norm**2
            else:
                self._variance_comp = None
            
        else:
            raise ValueError(f"which can be 'in' or 'comp', {which} given")

    def set_baseparams(self, parameters):
        """ """
        self._baseparams = {k:v for k,v in parameters.items() if k in self.BASE_PARAMETERS}
        
    def set_overlay(self, overlay):
        """ Provide the hypergal.utils.geometry.Overlay object """
        self._overlay = overlay

    def set_psf(self, psf):
        """ """
        self._psf = psf

    def update(self, **kwargs):
        """ """
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
            else:
                raise ValueError(f"Unknow input parameter {k}={v}")

        self.set_baseparams(baseparams)
        if len(geometryparams)>0:
            self.overlay.change_comp(**geometryparams)
        if len(psfparams)>0:
            self.psf.update_parameters(**psfparams)
        
    # --------- #
    #  GETTER   #
    # --------- #
    def get_model(self, ampl=None, background=None,
                      overlayparam=None,
                      psfparam=None):
        """ Convolves and project flux_in into the 


        Parameters
        ----------
        overlayparam: [dict or None]
            if dict, this is passed as kwargs to overlay
        psfparam: [dict or None]
            if dict, this is passed as a kwargs to self.psf.update_parameters(**psfparam)

        Returns
        -------
        flux
        """
        if ampl is None:
            ampl = self.baseparams["ampl"]
            
        if background is None:
            background = self.baseparams["background"]
            
        # 1.
        # Change position of the comp grid if needed
        #   - if the overlayparam are the same as already know, no update made.
        if overlayparam is not None and len(overlayparam)>0:
            self.overlay.change_comp(**overlayparam)

        # 2.            
        # Change values of flux and variances of _in by convolving the image
        flux_in = self.get_convolved_flux_in(psfparam)

        # 3. (overlaydf calculated only if needed)
        # Get the new projected flux and variance (_in->_comp grid)
        modelflux = self.overlay.get_projected_flux(flux_in)

        # 4. Out
        return ampl*modelflux + background
    
    def get_convolved_flux_in(self, psfconv=None):
        """ """
        if psfconv is not None:
            self.psf.update_parameters(**psfconv)
            
        return self.psf.convolve(self._flux_in2d).flatten()

    def guess_parameters(self):
        """ """
        ampl = 1
        bkgd = 0
        base_guess = {**{k:None for k in self.BASE_PARAMETERS},
                      **{"ampl":ampl, "background": bkgd}
                      }
        geom_guess = self.overlay.geoparam_comp
        psf_guess  = self.psf.guess_parameters()
        return {**base_guess, **geom_guess, **psf_guess}


    def show(self, savefile=None, titles=True, res_as_ratio=True, cutout_convolved=True,
                 vmin="1", vmax="99"):
        """ """
        import matplotlib.pyplot as mpl
        from ..utils import tools
        fig = mpl.figure(figsize=[9,2.7])
        left, witdth = 0.05, 0.21
        height = 0.8
        spanx, extraspanx =0.018, 0.03
        ax = fig.add_axes([left+0*(witdth+spanx)            ,0.12,witdth*1.1, height])
        axm = fig.add_axes([left+1*(witdth+spanx)+extraspanx,0.12,witdth, height])
        axd = fig.add_axes([left+2*(witdth+spanx)+extraspanx,0.12,witdth, height])
        axr = fig.add_axes([left+3*(witdth+spanx)+extraspanx,0.12,witdth, height])
        axifu = [axm, axd, axr]

        # Convolved image flux
        flux_in = self.get_convolved_flux_in() if cutout_convolved else self.flux_in
        self.overlay.show(ax=ax, flux_in = flux_in, lw_in=0, adjust=True)

        # Model flux (=convolved *ampl + background)
        flux_model = self.get_model().values
        vmin, vmax = tools.parse_vmin_vmax(flux_model, vmin, vmax)
        
        prop = {"cmap":"cividis", "vmin":vmin, "vmax":vmax, "lw":0.1, "edgecolor":"0.7","adjust":True}
        self.overlay.show_mpoly("comp", ax=axm, flux=flux_model, **prop)
        self.overlay.show_mpoly("comp", ax=axd, flux=self.flux_comp, **prop)

        res = (self.flux_comp-flux_model)
        if res_as_ratio:
            res /=  flux_model
            prop = {**prop,**{"vmin":-0.5, "vmax":0.5, "cmap":"coolwarm"}}
            title_res = "Residual (data-scene)/scene [±50%]"
        else:
            title_res = "Residual (data-scene)"
            
        self.overlay.show_mpoly("comp", ax=axr, flux=res, **prop)
        
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
    #  Properties   #
    # ============= #
    @property
    def slice_in(self):
        """ """
        return self._slice_in
    
    @property    
    def slice_comp(self):
        """ """
        return self._slice_comp

    @property
    def overlay(self):
        """ """
        if not hasattr(self,"_overlay"):
            raise AttributeError("No Overlay set ; see self.set/load_overlay()")
        return self._overlay

    @property
    def psf(self):
        """ """
        if not hasattr(self, "_psf"):
            raise AttributeError("No PSF set ; see self.set_psf()")
        
        return self._psf

    @property
    def baseparams(self):
        """ """
        if not hasattr(self, "_baseparams"):
            self._baseparams = {}
        return self._baseparams

    # // _in prop
    @property
    def norm_in(self):
        """ original _in data""" # No test to gain time
        return self._norm_in
    
    @property
    def bkgd_in(self):
        """ """ # No test to gain time
        return self._bkgd_in

    @property
    def flux_in(self):
        """ """ # No test to gain time
        return self._flux_in

    @property
    def variance_in(self):
        """ """ # No test to gain time
        return self._variance_in

    # // _comp prop
    @property
    def norm_comp(self):
        """ given data = data*norm + back""" # No test to gain time
        return self._norm_comp
    
    @property
    def bkgd_comp(self):
        """ """ # No test to gain time
        return self._bkgd_comp

    @property
    def flux_comp(self):
        """ """ # No test to gain time
        return self._flux_comp

    @property
    def variance_comp(self):
        """ """ # No test to gain time
        return self._variance_comp

    # ----------- #
    #  Parameters #
    # ----------- #
    @property
    def PSF_PARAMETERS(self):
        """ """
        return self.psf.PARAMETER_NAMES
    
    @property
    def GEOMETRY_PARAMETERS(self):
        """ """
        return self.overlay.PARAMETER_NAMES
    
    @property
    def PARAMETER_NAMES(self):
        """ """
        return self.BASE_PARAMETERS + self.GEOMETRY_PARAMETERS + self.PSF_PARAMETERS
    
# ================= #
#                   #
#   CUBE SCENE      # 
#                   #
# ================= #

class CubeScene( SliceScene ):
    def __init__(self, *args, **kwargs):
        """ """
        _ = super().__init__(*args, **kwargs)
        print("CubeScene has not been implemented yet")
        
