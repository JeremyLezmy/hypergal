
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
class _BaseScene_(object):
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
        for k, v in kwargs.items():
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
        for k, v in kwargs.items():
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
        if len(geometryparams) > 0:
            self.overlay.change_comp(**geometryparams)
        if len(psfparams) > 0:
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
        if overlayparam is not None and len(overlayparam) > 0:
            self.overlay.change_comp(
                **{k: v for k, v in overlayparam.items() if v is not None})

        # 2.
        # Change values of flux and variances of _in by convolving the image
        if psfparam is not None:
            psfparam = {k: v for k, v in psfparam.items() if v is not None}
        flux_in = self.get_convolved_flux_in(psfparam)

        # 3. (overlaydf calculated only if needed)
        # Get the new projected flux and variance (_in->_comp grid)
        modelflux = self.overlay.get_projected_flux(
            flux_in, fill_comp=fill_comp)

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
        base_guess = {**{k: None for k in self.BASE_PARAMETERS},
                      **{"ampl": ampl, "background": bkgd}
                      }
        geom_guess = self.overlay.geoparam_comp
        psf_guess = self.psf.guess_parameters()
        guess_step1 = {**base_guess, **geom_guess, **psf_guess}
        self.update(**guess_step1)

        model_comp = self.get_model()
        bkgd = np.median(self.flux_comp)-np.median(model_comp)
        ampl = np.sum(self.flux_comp)/np.sum(model_comp)

        return {**guess_step1, **{"ampl": ampl, "background": bkgd}}

    def get_parameters(self):
        """ """
        return {**self.baseparams, **self.overlay.geoparam_comp, **self.psf.parameters}

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def overlay(self):
        """  Overlay object between slice_in and slice_comp """
        if not hasattr(self, "_overlay"):
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
            self._baseparams = {
                k: 1. if "ampl" in k else 0. for k in self.BASE_PARAMETERS}
        return self._baseparams

    # // _in prop
    @property
    def norm_in(self):
        """  Norm used to scale _in data """  # No test to gain time
        return self._norm_in

    @property
    def bkgd_in(self):
        """  Estimated _in background. """  # No test to gain time
        return self._bkgd_in

    @property
    def flux_in(self):
        """  Original _in data minus bkgd_in and divided by norm_in. """  # No test to gain time
        return self._flux_in

    @property
    def variance_in(self):
        """  Variance of _in scaled by norm_in. """  # No test to gain time
        return self._variance_in

    # // _comp prop
    @property
    def norm_comp(self):
        """  Norm used to scale _comp data. """  # No test to gain time
        return self._norm_comp

    @property
    def bkgd_comp(self):
        """ Estimated _comp background. """  # No test to gain time
        return self._bkgd_comp

    @property
    def flux_comp(self):
        """ Original _comp data minus bkgd_comp and divided by norm_comp. """  # No test to gain time
        return self._flux_comp

    @property
    def variance_comp(self):
        """  Variance of _comp scaled by norm_comp. """  # No test to gain time
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


class SliceScene(_BaseScene_):

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
    def set_slice(self, slice_, which, adapt_flux=True, norm="99", bkgd="5"):
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

        if type(bkgd) is str and which == "comp":
            bkgd = np.percentile(data[data == data], float(bkgd))
        elif which == 'in':
            bkgd = 0

        data -= bkgd
        if type(norm) is str:
            norm = np.percentile(data[data == data], float(norm))
            if norm == 0:
                norm = np.percentile(data[data == data], float('99.9')) if np.percentile(
                    data[data == data], float('99.9')) > 0 else np.max(data)  # fix in case only few pixels has been fitted by the sedfitter

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

        self.set_fitted_data(which=which, flux=data,
                             variance=variance, norm=norm, background=bkgd)

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
        flux_in = self.get_convolved_flux_in() if cutout_convolved else self.flux_in
        flux_model = self.get_model(fill_comp=fill_comp)
        flux_comp = self.flux_comp

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
            fig = mpl.figure(figsize=[9, 2.7])
            left, witdth = 0.05, 0.21
            height = 0.8
            spanx, extraspanx = 0.018, 0.03
            ax = fig.add_axes([left+0*(witdth+spanx), 0.12, witdth*1.1, height])
            axm = fig.add_axes(
                [left+1*(witdth+spanx)+extraspanx, 0.12, witdth, height])
            axd = fig.add_axes(
                [left+2*(witdth+spanx)+extraspanx, 0.12, witdth, height])
            axr = fig.add_axes(
                [left+3*(witdth+spanx)+extraspanx, 0.12, witdth, height])

        axifu = [axm, axd, axr]

        overlay.show(ax=ax, comp_index=index, flux_in=flux_in, lw_in=0,
                     adjust=True, cmap=cmapproj)
        vmin, vmax = tools.parse_vmin_vmax(flux_model, vmin, vmax)

        prop = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "lw": 0.1, "edgecolor": "0.7",
                "adjust": True, "index": index}
        overlay.show_mpoly("comp", ax=axm, flux=flux_model, **prop)
        overlay.show_mpoly("comp", ax=axd, flux=flux_comp, **prop)

        res = (flux_comp-flux_model)
        if res_as_ratio:
            res /= flux_model
            prop = {**prop, **{"vmin": -0.5, "vmax": 0.5, "cmap": "coolwarm"}}
            title_res = "Residual (data-scene)/scene [±50%]"
        else:
            title_res = "Residual (data-scene)"

        overlay.show_mpoly("comp", ax=axr, flux=res, **prop)

        clearwhich = ["left", "right", "top", "bottom"]
        for ax_ in axifu:
            ax_.set_yticks([])
            ax_.set_xticks([])
            [ax_.spines[which].set_visible(False) for which in clearwhich]

        if titles:
            prop = dict(loc="left", color="0.5", fontsize="small")
            ax.set_title("Projection" +
                         " (convolved)" if cutout_convolved else "", **prop)
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
        slicexy = self.slice_in.xy
        [xmin, ymin], [xmax, ymax] = np.percentile(slicexy, [0, 100], axis=1)

        steps = np.unique(np.diff(slicexy, axis=1))
        steps = steps[steps > 0]
        if len(steps) != 1:
            raise ValueError(
                f"Cannot guess the binfactor. 1 entry expected, {steps} obtained.")
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
class CubeScene(SliceScene):

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
        self.set_cube(cube_comp, "comp")

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
            bkgd = np.percentile(data[data == data],
                                 float(bkgd), axis=1)[:, None]

        data -= bkgd
        if type(norm) is str:
            norm = np.percentile(data[data == data],
                                 float(norm), axis=1)[:, None]

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
        if len(kwargs) > 0:
            try:
                ampl = np.asarray([kwargs[f"ampl{i}"]
                                  for i in range(self.nslices)])
                background = np.asarray(
                    [kwargs[f"background{i}"] for i in range(self.nslices)])
            except:
                warnings.warn(
                    "Cannot use the kwargs for ampl and background definition")

        if ampl is None:
            ampl = self.get_amplitudes()

        if background is None:
            background = self.get_backgrounds()

        # 1.
        # Change position of the comp grid if needed
        #   - if the overlayparam are the same as already know, no update made.
        if overlayparam is not None and len(overlayparam) > 0:
            self.overlay.change_comp(
                **{k: v for k, v in overlayparam.items() if v is not None})

        # 2.
        # Change values of flux and variances of _in by convolving the image
        if psfparam is not None:
            psfparam = {k: v for k, v in psfparam.items() if v is not None}
        flux_in = self.get_convolved_flux_in(psfparam)
        # 3. (overlaydf calculated only if needed)
        # Get the new projected flux and variance (_in->_comp grid)
        modelflux = self.overlay.get_projected_flux(flux_in)

        # 4. Out
        return ampl[:, None]*modelflux + background[:, None]

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
        psf_guess = self.psf.guess_parameters()
        guess_step1 = {**base_guess, **geom_guess, **psf_guess}

        self.update(**guess_step1)

        model_comp = self.get_model()
        bkgd = np.median(self.flux_comp, axis=1)-np.median(model_comp, axis=1)
        ampl = np.percentile(self.flux_comp, 95, axis=1) / \
            np.percentile(model_comp, 95, axis=1)
        baseparams = {**{f"ampl{i}": ampl[i] for i in range(self.nslices)},
                      **{f"background{i}": bkgd[i] for i in range(self.nslices)}}
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
        fig = mpl.figure(figsize=[9, 2.7*nindex])
        left, witdth = 0.05, 0.21
        height = 0.8/nindex
        spanx, extraspanx = 0.018, 0.03
        for i, index_ in enumerate(index):
            bottom = (0.12 + i)/nindex
            ax = fig.add_axes(
                [left+0*(witdth+spanx), bottom, witdth*1.1, height])
            axm = fig.add_axes(
                [left+1*(witdth+spanx)+extraspanx, bottom, witdth, height])
            axd = fig.add_axes(
                [left+2*(witdth+spanx)+extraspanx, bottom, witdth, height])
            axr = fig.add_axes(
                [left+3*(witdth+spanx)+extraspanx, bottom, witdth, height])

            flux_in = (self.get_convolved_flux_in()
                       if cutout_convolved else self.flux_in)[index_]
            flux_model = self.get_model()[index_]
            flux_comp = self.flux_comp[index_]
            _ = self._show_slice_scene_(self.overlay, flux_in, flux_model, flux_comp,
                                        axes=[ax, axm, axd, axr], savefile=None,
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
        slicexy = self.cube_in.xy
        [xmin, ymin], [xmax, ymax] = np.percentile(slicexy, [0, 100], axis=1)

        steps = np.unique(np.diff(slicexy, axis=1))
        steps = steps[steps > 0]
        if len(steps) != 1:
            raise ValueError(
                f"Cannot guess the binfactor. 1 entry expected, {steps} obtained.")
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


class FullSliceScene(SliceScene):

    def __init__(self, slice_in, slice_comp, xy_in=None, xy_comp=None, load_overlay=True,
                 psfgal=None, adapt_flux=True, pointsource=None, curved_bkgd=False, **kwargs):

        _ = super().__init__(slice_in=slice_in, slice_comp=slice_comp, xy_in=xy_in, xy_comp=xy_comp, load_overlay=load_overlay,
                             psf=psfgal, adapt_flux=adapt_flux, **kwargs)

        self.set_pointsource(pointsource)

        if load_overlay:
            mpoly = self.overlay.mpoly_comp
        else:
            mpoly = slice_comp.get_spaxel_polygon(format='multipolygon')

        self._has_curved_bkgd = curved_bkgd

    @classmethod
    def from_slices(cls, slice_in, slice_comp, xy_in=None, xy_comp=None, psfgal=None, pointsource=None, curved_bkgd=False, **kwargs):
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
                   xy_in=xy_in, xy_comp=xy_comp, psfgal=psfgal, pointsource=pointsource, curved_bkgd=curved_bkgd,
                   **kwargs)

    def get_model(self, ampl=None, background=None,
                  overlayparam=None,
                  psfparam=None, ampl_ps=None, bkgdx=None, bkgdy=None, bkgdxy=None, bkgdxx=None, bkgdyy=None, fill_comp=False):
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

        if ampl_ps is None:
            ampl_ps = self.baseparams["ampl_ps"]

        if background is None and not self.has_curved_bkgd:
            background = self.baseparams["background"]

        elif background is None and self.has_curved_bkgd:
            x, y = self.slice_comp_xy
            coeffs = dict({k: v for k, v in self.baseparams.items()
                          if k in BackgroundCurved.BACKGROUND_PARAMETERS})
            background = BackgroundCurved.get_background(x, y, coeffs)

        elif background is not None and self.has_curved_bkgd:

            x, y = self.slice_comp_xy
            coeffs = dict({'background': background, 'bkgdx': bkgdx, 'bkgdy': bkgdy,
                          'bkgdxy': bkgdxy, 'bkgdxx': bkgdxx, 'bkgdyy': bkgdyy})
            background = BackgroundCurved.get_background(x, y, coeffs)

        if psfparam is not None:
            psf_hostparam = {k: v for k, v in psfparam.items(
            ) if k in super(FullSliceScene, self).PSF_PARAMETERS}
            psf_psparam = {k.replace('_ps', ''): v for k, v in psfparam.items(
            ) if k in self.pointsource.PSF_PARAMETERS}

        else:
            psf_hostparam = None
            psf_psparam = None
        # 1.
        # Change position of the comp grid if needed
        #   - if the overlayparam are the same as already know, no update made.
        if overlayparam is not None and len(overlayparam) > 0:
            self.overlay.change_comp(
                **{k: v for k, v in overlayparam.items() if v is not None})

        # 2.
        # Change values of flux and variances of _in by convolving the image
        if psf_hostparam is not None:
            psf_hostparam = {k: v for k,
                             v in psf_hostparam.items() if v is not None}
        flux_in = self.get_convolved_flux_in(psf_hostparam)

        # 3. (overlaydf calculated only if needed)
        # Get the new projected flux and variance (_in->_comp grid)
        modelflux = self.overlay.get_projected_flux(
            flux_in, fill_comp=fill_comp)

        # 4. Out
        galmodel = ampl*modelflux + background

        if psf_psparam is not None:
            psf_psparam = {k: v for k, v in psf_psparam.items()
                           if v is not None}

            ps_profile = self.pointsource.get_model(
                xoff=self.overlay.geoparam_comp['xoff'], yoff=self.overlay.geoparam_comp['yoff'], ampl=ampl_ps, bkg=None, **psf_psparam)

        else:
            ps_profile = self.pointsource.get_model(
                xoff=self.overlay.geoparam_comp['xoff'], yoff=self.overlay.geoparam_comp['yoff'], ampl=ampl_ps, bkg=None)

        return galmodel+ps_profile

    def guess_parameters(self):
        """ 
        Return guessed parameters for all the parameters.\n
        Include BASE_PARAMETERS (amplitude and background), 
        geometrical parameters (scale, xy_in etc) 
        and psf parameters (shape and ellipticity)       
        """
        ampl = 1
        bkgd = 0
        ampl_ps = 1
        base_guess = {**{k: None for k in self.BASE_PARAMETERS},
                      **{"ampl": ampl, "background": bkgd, "ampl_ps": ampl_ps}
                      }
        if self.has_curved_bkgd:
            base_guess.update(
                **{k: 0 for k in BackgroundCurved.BACKGROUND_PARAMETERS})

        geom_guess = self.overlay.geoparam_comp
        psf_guess = self.host_psf.guess_parameters()
        psf_guess.update(
            {k+'_ps': v for k, v in self.pointsource_psf.guess_parameters().items()})
        guess_step1 = {**base_guess, **geom_guess, **psf_guess}
        self.update(**guess_step1)

        model_comp = self.get_model()
        bkgd = np.median(self.flux_comp)-np.median(model_comp)
        ampl = np.sum(self.flux_comp)/np.sum(model_comp)

        from shapely.geometry import Point
        x, y = geom_guess['xoff'], geom_guess['yoff']
        p = Point(x, y)
        circle = p.buffer(3)
        idx = self.slice_comp.get_spaxels_within_polygon(circle)
        ampl_ps = np.nansum(self.flux_comp[[self.slice_comp.indexes[i] in np.array(
            idx) for i in range(len(self.slice_comp.indexes))]])
        #ampl_ps = np.nanmax(self.slice_comp.get_subslice([i for i in self.slice_comp.indexes if i in idx]).data)*10

        return {**guess_step1, **{"ampl": ampl, "background": bkgd, "ampl_ps": ampl_ps}}

    def update(self, ignore_extra=False, **kwargs):
        """ 
        Can update any parameter through kwarg option.\n
        Might be self.BASE_PARAMETER, self.PSF_PARAMETERS or self.GEOMETRY_PARAMETERS
        """
        baseparams = {}
        psfparams = {}
        geometryparams = {}
        for k, v in kwargs.items():
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

        if len(geometryparams) > 0:
            self.overlay.change_comp(**geometryparams)

        for k in psfparams.keys():
            if k in super(FullSliceScene, self).PSF_PARAMETERS:
                self.host_psf.update_parameters(**{k: psfparams[k]})
            if k in self.pointsource.PSF_PARAMETERS:
                self.pointsource_psf.update_parameters(
                    **{k.replace('_ps', ''): psfparams[k]})

    def set_host_psf(self, psf):
        """ """
        self.set_psf(psf)

    def set_pointsource_psf(self, psf):
        """ """
        self.pointsource.set_psf(psf)

    def set_pointsource(self, pointsource):
        """ """
        self._pointsource = pointsource

    @property
    def PARAMETER_NAMES(self):
        """ All parameters names """
        return self.BASE_PARAMETERS + self.GEOMETRY_PARAMETERS + self.PSF_PARAMETERS

    @property
    def pointsource(self):
        """ All parameters names """
        return self._pointsource

    @property
    def host_psf(self):
        """ All parameters names """
        return self.psf

    @property
    def pointsource_psf(self):
        """ All parameters names """
        return self.pointsource.psf

    @property
    def PSF_PARAMETERS(self):
        """ All parameters names """
        return super().PSF_PARAMETERS + self.pointsource.PSF_PARAMETERS

    @property
    def has_curved_bkgd(self):
        """ All parameters names """
        return self._has_curved_bkgd

    @property
    def slice_comp_xy(self):
        """ All parameters names """

        return self.pointsource.centroids.T

    @property
    def BASE_PARAMETERS(self):
        """ All parameters names """
        basepar = super().BASE_PARAMETERS + self.pointsource.BASE_PS_PARAMETERS
        basepar.remove('background_ps')
        if self.has_curved_bkgd:
            basepar.remove('background')
            basepar += BackgroundCurved.BACKGROUND_PARAMETERS
        return basepar


class PointSource(object):

    BASE_PS_PARAMETERS = ["ampl_ps", "background_ps"]

    def __init__(self, psf, mpoly, **kwargs):
        """ """
        self.set_psf(psf)
        self.update_psfparams(**kwargs)
        self.set_mpoly(mpoly)
        self._centroids = self.get_centroids(mpoly)

    @staticmethod
    def get_centroids(mpoly):
        """ """
        listpoly = list(mpoly)
        cent = [np.array([listpoly[i].centroid.x, listpoly[i].centroid.y])
                for i in range(len(listpoly))]
        cent = np.asarray(cent).squeeze()

        return cent

    def get_model(self, xoff=0, yoff=0, ampl=None, bkg=None, **kwargs):
        """ """
        x, y = self.centroids.T

        if ampl is None:
            ampl = self.baseparams['ampl_ps']

        if bkg is None:
            bkg = self.baseparams['background_ps']

        psfval = ampl * \
            self.psf.evaluate(x=x, y=y, x0=xoff, y0=yoff, **kwargs) + bkg
        self.set_psfflux(psfval)

        return psfval

    def update_psfparams(self, **kwargs):
        """ """

        for k, v in kwargs.items():
            if '_ps' in k:
                kwargs[k.replace('_ps', '')] = kwargs.pop(k)

        self.psf.update_parameters(**kwargs)

    def update_baseparams(self, **kwargs):
        import warnings
        """ 
        Set parameters from self.BASE_PS_PARAMETERS (amplitude and background)
        """
        for k, v in kwargs.items():
            if k in self.BASE_PS_PARAMETERS:
                self.baseparams[k] = v
            else:
                warnings.warn(f"{k} is not a base parameters, ignored")
                continue

    def update(self, ignore_extra=False, **kwargs):
        """ 
        Can update any parameter through kwarg option.\n
        Might be self.BASE_PS_PARAMETER, self.PSF_PARAMETERS 
        """
        baseparams = {}
        psfparams = {}

        for k, v in kwargs.items():
            # Change the baseline scene
            if k in self.BASE_PS_PARAMETERS:
                baseparams[k] = v

            # Change the scene PSF
            elif k in self.PSF_PARAMETERS:
                psfparams[k] = v

            # or crash
            elif not ignore_extra:
                raise ValueError(f"Unknow input parameter {k}={v}")

        self.update_baseparams(**baseparams)
        if len(psfparams) > 0:
            self.update_psfparams(**psfparams)

    def guess_ps_parameters(self):
        """ 
        Return guessed parameters for all the parameters.\n
        Include BASE_PARAMETERS (amplitude and background), 
        and psf parameters (shape and ellipticity)       
        """
        base_guess = self.baseparams.copy()
        psf_guess = self.psf.guess_parameters()
        guess_step1 = {**base_guess,  **psf_guess}

        self.update(**guess_step1)

        model_comp = self.get_model()
        bkgd = np.nanmedian(self.flux_comp, axis=1) - \
            np.nanmedian(model_comp, axis=1)
        ampl = np.nanpercentile(self.flux_comp, 95, axis=1) / \
            np.nanpercentile(model_comp, 95, axis=1)
        baseparams = {**{f"ampl{i}": ampl[i] for i in range(self.nslices)},
                      **{f"background{i}": bkgd[i] for i in range(self.nslices)}}
        return {**guess_step1, **baseparams}

    def show_psf(self, ax=None, facecolor=None, edgecolor="k", adjust=False,
                 index=None,
                 flux=None, cmap="cividis", vmin=None, vmax=None, **kwargs):
        """ 
        Show multipolygon with its corresponding flux if provided.

        Parameters
        ----------

        ax: Axes -optional-
            You can provide your own ax (one)

        facecolor,edgecolor,cmap: string
            Go to Matplotlib parameters

        flux: array -optional-
            Flux corresponding to self.mpoly_*which* \n
            Default is None.

        vmin,vmax: float, None, string -optional-
            Colorbar limits. 
            - Ignored if flux is None \n
            - String: used as flux percentile\n
            - Float: used value\n
            - None: converted to '1' and '99'

        kwargs
            Goes to geometry.show_polygon()

        Returns
        -------
        Matplotlib.Axes    

        """
        import matplotlib.pyplot as mpl
        from hypergal.utils import tools
        if ax is None:
            fig = mpl.figure(figsize=[6, 6])
            ax = fig.add_subplot(111)
            adjust = True
        else:
            fig = ax.figure

        if flux is not None:
            cmap = mpl.cm.get_cmap(cmap)
            vmin, vmax = tools.parse_vmin_vmax(flux, vmin, vmax)
            colors = cmap((flux-vmin)/(vmax-vmin))
        else:
            colors = [facecolor]*len(self.mpoly)

        for i, poly in enumerate(self.mpoly):
            _ = geometry.show_polygon(
                poly, facecolor=colors[i], edgecolor=edgecolor, ax=ax, **kwargs)

        if adjust:
            verts = np.asarray(self.mpoly.convex_hull.exterior.xy).T
            ax.set_xlim(*np.percentile(verts[:, 0], [0, 100]))
            ax.set_ylim(*np.percentile(verts[:, 1], [0, 100]))

        return ax

    def set_psf(self, psf):
        """ Provide the hypergal.psf object. """
        self._psf = psf

    def set_mpoly(self, mpoly):
        """ """
        self._mpoly = mpoly

    def set_psfflux(self, psfflux):
        """ """
        self._psfflux = psfflux

    @property
    def psf(self):
        """  PSF object to convolve on slice_in. """
        if not hasattr(self, "_psf"):
            raise AttributeError("No PSF set ; see self.set_psf()")

        return self._psf

    @property
    def psfflux(self):
        """  Evaluated PSF flux  """
        if not hasattr(self, "_psfflux"):
            raise AttributeError(
                "No PSF computed ; see self.build_point_source()")

        return self._psfflux

    @property
    def mpoly(self):
        """  """
        if not hasattr(self, "_mpoly"):
            raise AttributeError("No geometry set ")

        return self._mpoly

    @property
    def centroids(self):
        """  """
        if not hasattr(self, "_centroids"):
            raise AttributeError(
                "No centroids computed. See self.get_centroid() ")

        return self._centroids

    @property
    def PSF_PARAMETERS(self):
        """  PSF parameters (profile + ellipticity). """
        return [k + '_ps' for k in self.psf.PARAMETER_NAMES]

    @property
    def baseparams(self):
        """  Base parameters (e.g. amplitude and background) """
        if not hasattr(self, "_baseparams"):
            self._baseparams = {
                k: 1. if "ampl_ps" in k else 0. for k in self.BASE_PS_PARAMETERS}
        return self._baseparams


class PointSource3D(PointSource):

    BASE_PS_SLICE_PARAMETERS = ["ampl_ps", "background_ps"]

    def __init__(self, psf, mpoly, lbdaref, lbda):
        """ """
        _ = super().__init__(psf=psf, mpoly=mpoly)

        self.psf.set_lbdaref(lbdaref)
        self._centroids = self.get_centroids(mpoly)
        self.set_lbda(lbda)

    @classmethod
    def from_adr(cls, psf, mpoly, lbda, adr, xref, yref, spaxel_comp_unit, **kwargs):
        """ """

        this = cls(psf=psf, mpoly=mpoly, lbdaref=adr.lbdaref, lbda=lbda)
        this.set_adr(adr)

        xoff, yoff = adr.refract(xref, yref, lbda, unit=spaxel_comp_unit)

        this.get_model(xoff, yoff, **kwargs)

        return this

    @classmethod
    def from_header(cls, psf, mpoly, lbda, header, xref, yref, spaxel_comp_unit, **kwargs):
        """ """
        from pyifu.adr import ADR

        adr = ADR.from_header(header)

        this = cls(psf=psf, mpoly=mpoly, lbdaref=adr.lbdaref, lbda=lbda)

        this.set_adr(adr)

        xoff, yoff = adr.refract(xref, yref, lbda, unit=spaxel_comp_unit)

        this.get_model(xoff, yoff, **kwargs)

        return this

    def get_model(self, xoff, yoff, ampl=None, bkg=None, **kwargs):
        """ """
        x, y = self.centroids.T

        if ampl is None:
            ampl = self.get_ps_amplitudes()

        if bkg is None:
            bkg = self.get_ps_backgrounds()

        self.psf.update_parameters(**kwargs)
        psfval = ampl * np.asarray([self.psf.evaluate(x=x, y=y, lbda=l, x0=xo, y0=yo)
                                   for l, xo, yo in zip(self.lbda, xoff, yoff)]).T + bkg

        self.set_psfflux(psfval.T)

        return psfval.T

    def get_ps_amplitudes(self):
        """ """
        return np.asarray([self.baseparams[f"ampl_ps{i}"] for i in range(self.nslices)])

    def get_ps_backgrounds(self):
        """ """
        return np.asarray([self.baseparams[f"background_ps{i}"] for i in range(self.nslices)])

    def set_lbda(self, lbda):
        """ """
        self._lbda = lbda

    def set_adr(self, adr):
        """ """
        self._adr = adr

    @property
    def lbda(self):
        return self._lbda

    @property
    def adr(self):
        """  """
        if not hasattr(self, "_adr"):
            raise AttributeError("No adr set ")

        return self._adr

    @property
    def nslices(self):
        """ """
        return len(self.lbda)

    @property
    def BASE_PS_PARAMETERS(self):
        """ """
        return [f"{k}{i}" for k in self.BASE_PS_SLICE_PARAMETERS for i in range(self.nslices)]


class BackgroundCurved():
    """ """
    BACKGROUND_PARAMETERS = ["background", "bkgdx",
                             "bkgdy", "bkgdxy", "bkgdxx", "bkgdyy"]

    def __init__(self):

        pass

    @classmethod
    def get_background(cls, x, y, bkgdprop=None):
        """ The background at the given positions """
        this = cls()
        if bkgdprop is None or len(bkgdprop) < len(this.BACKGROUND_PARAMETERS):

            bkgdprop = this.guess_parameters()

        return this.curved_plane(x, y, [bkgdprop[k] for k in this.BACKGROUND_PARAMETERS])

    @staticmethod
    def curved_plane(x, y,
                     bkg_coefs):
        """ """
        return np.dot(np.asarray([np.ones(x.shape[0]), x, y, x*y, x*x, y*y]).T, bkg_coefs)

    def guess_parameters(self):
        """ 
        Default parameters (init for an eventual fit)
        """
        return {**{"background": 0., "bkgdx": 0., "bkgdy": 0., "bkgdxy": 0., "bkgdxx": 0., "bkgdyy": 0.}
                }
