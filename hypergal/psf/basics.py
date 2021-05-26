""" """

import numpy as np
from astropy.convolution import convolve

class PSF2D( object ):
    
    ELLIPTICITY_PARAMETERS = ["a","b"]
    PROFILE_PARAMETERS = []
    
    def __init__(self, **kwargs):
        """ 
        Create a PSF 2D object.

        Parameters
        ----------
        kwargs : dict -optional-
            Allow to update existings parameters.\n
            Key(s) is/are parameter(s), value(s) the value(s).

        """
        self._ellipticity_params = {k:1 if k=="a" else 0 for k in self.ELLIPTICITY_PARAMETERS}
        self._profile_params = {}
        self.update_parameters(**kwargs)
        
    # ============= #
    #  Methods      #
    # ============= #
    def update_parameters(self, **kwargs):
        """ 
        Update existing parameters.

        Parameters
        ----------
        kwargs : dict -optional-
            Allow to update existings parameters.\n
            Key(s) is/are parameter(s), value(s) the value(s).

        Returns
        -------
        """
        for k,v in kwargs.items():
            # Change the ellipticity
            if k in self.ELLIPTICITY_PARAMETERS:
                self._ellipticity_params[k] = v
            # Change the Profile                
            elif k in self.PROFILE_PARAMETERS:
                self._profile_params[k] = v
            # Crashes                
            else:
                raise ValueError(f"Unknow input parameter {k}={v}")
        
    def evaluate(self, x, y, x0=0, y0=0, **kwargs):
        """ 
        Evaluate the radial profile of the defined psf model with its current parameters.

        Parameters
        ----------
        x: array
             x-axis of the grid where you want to evaluate the psf.

        y: array
             y-axis of the grid where you want to evaluate the psf.

        x0: float
            Centroid position on the x-axis.

        y0: float
            Centroid position on the y-axis.

        kwargs: dict
            Go to self.update_parameters()

        Returns
        -------
        Array of the radial profile.
        """
        self.update_parameters(**kwargs)
        dx = x-x0
        dy = y-y0
        r = np.sqrt( dx**2 + self.a_ell*dy**2 + 2*self.b_ell * (dx*dy) )
        return self.get_radial_profile(r)

    def get_radial_profile(self, r):
        """ 
        Get radial profile according to the choosen psf and its elliptical radius.

        Parameters
        ----------
        r: array
            Elliptical radius

        Returns
        -------
        Array of the radial profile at the *r* position.
        """
        raise NotImplementedError("You must define your PSF self.get_radial_profile()")

    def get_stamp(self, psfwindow=17, **kwargs):
        """ 
        Evaluate psf on a 2D-grid of size *psfwindow*, centered on x0=y0=0.

        Parameters
        ----------
        psfwindow: float
            Size of the stamp.\n
            Default is 17 pix.

        kwargs: dict
            Go to self.update_parameters().

        Returns
        -------
        2D-Array of the evaluate PSF on this stamp.
        """
        x, y = np.mgrid[-psfwindow/2:psfwindow/2, -psfwindow/2:psfwindow/2]+0.5
        return self.evaluate(x, y, **kwargs)
    
    def convolve(self, arr2d, psfwindow=17, **kwargs):
        """ 
        Convolve some 2D-array with the 2D PSF.

        Parameters
        ----------
        arr2d: 2D-array
            Datas on which you want to apply the convolution

        psfwindow: float
            Size on the stamp on which you want to generate the psf (see self.get_stamp() )\n
            Default is 17 pix.

        kwargs: dict
            Go to self.update_parameters()

        Returns
        -------
        The 2D-array datas convolved.
        
        """
        return convolve(arr2d, self.get_stamp(psfwindow=psfwindow, **kwargs), normalize_kernel=False)

    def guess_parameters(self):
        """ 
        Initial parameters for the ellipticity of the psf.\n
        Used as initial condition for an eventual fit.\n
        Guess parameters are a==1 and b=0 (no ellipticity).
        """
        return {**{"a":1,"b":0}, **{k:None for k in self.PROFILE_PARAMETERS}}
    
    # -------- #
    #  PLOTTER #
    # -------- #
    def show(self, ax=None, psfwindow=17, **kwargs):
        """ 
        Show 2D PSF with current parameters.

        Parameters
        ----------
        ax: Matplotlib.Axes -optional-
            Choice to use an existing Axes (dim=1)

        psfwindow: float
            Size of the stamp where the PSF will be display\n
            Default is 17 pix.

        kwargs: dict
            goes to imshow()

        Returns
        -------
        Figure
        """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[5,5])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        prop = dict(origin="lower", cmap="cividis")
        sc = ax.imshow(self.get_stamp(psfwindow),  **{**prop,**kwargs} )
        return fig
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def a_ell(self):
        """ a elliticity parameter (defined as "a+b**2") """
        return self._ellipticity_params["a"]
    
    @property
    def b_ell(self):
        """ b elliticity parameter """
        return self._ellipticity_params["b"]
    
    @property
    def PARAMETER_NAMES(self):
        """ 
        Name of the PSF parameters: Ellipticity + Shape
        """
        return self.ELLIPTICITY_PARAMETERS + self.PROFILE_PARAMETERS

    @property
    def parameters(self):
        """ """
        return {**{k:self._ellipticity_params.get(k,None) for k in self.ELLIPTICITY_PARAMETERS},
                **{k:self._profile_params.get(k,None) for k in self.PROFILE_PARAMETERS}
                }
                    
    
class PSF3D( PSF2D ):

    def __init__(self, lbdaref=6000, **kwargs):
        """ 
        3D PSF. Inherited from 2D PSF, with wavelength parameter as 3rd dimension.

        Parameters
        ----------
        lbdaref: float -optional-
              Reference wavelength for chromatic parameters.\n
              Default is 6000 (AA)

        Results
        -------
        """
        _ = super().__init__(**kwargs)
        self.set_lbdaref(lbdaref)
        return _
        
    # ============= #
    #  Methods      #
    # ============= #
    def evaluate(self, x, y, lbda, x0=0, y0=0, **kwargs):
        """ 
        Evaluate the radial profile of the defined psf model with its current parameters.

        Parameters
        ----------
        x: array
            x-axis of the grid where you want to evaluate the psf.

        y: array
            y-axis of the grid where you want to evaluate the psf.

        lbda: array
            Wavelength 3rd dimension.

        x0: float
           Centroid position on the x-axis.

        y0: float
           Centroid position on the y-axis.

        kwargs: dict
           Go to self.update_parameters()

        Returns
        -------
        Array of the radial profile.
        """
        self.update_parameters(**kwargs)
        dx = x-x0
        dy = y-y0
        r = np.sqrt( dx**2 + self.a_ell*dy**2 + 2*self.b_ell * (dx*dy) )
        return self.get_radial_profile(r, lbda)


    def convolve(self, arr3d, lbda, psfwindow=17, **kwargs):
        """ 
        Convolve some 3D-array with the 2D PSF.

        Parameters
        ----------
        arr3d: 3d-array
            Datas on which you want to apply the convolution

        lbda: 1d-array
            Wavelength [in AA] associated with arr3d

        psfwindow: float -optional-
            Size on the stamp on which you want to generate the psf (see self.get_stamp() )\n
            Default is 17 pix.

        kwargs: dict
            Go to self.update_parameters()

        Returns
        -------
        The 2D-array datas convolved.
        
        """
        if len(lbda) != len(arr3d):
            raise ValueError(f"lbda and arr3d do not have the same size : {len(lbda)} vs. {len(arr3d)}")
        
        stamp3d = self.get_stamp(lbda=lbda, psfwindow=psfwindow, **kwargs)
        # loop as it's esier and not that slow
        return [convolve(arr2d, stamp2d, normalize_kernel=False)
                    for arr2d,stamp2d in zip(arr3d, stamp3d)]

    def get_radial_profile(self, r, lbda):
        """ 
        Get radial profile according to the choosen psf and its elliptical radius.

        Parameters
        ----------
        r: array
            Elliptical radius

        Returns
        -------
        Array of the radial profile at the *r* position.
        """
        raise NotImplementedError("You must define your PSF self.get_radial_profile()")
    
    def set_lbdaref(self, lbda):
        """ 
        Set new reference wavelength
        """
        self._lbdaref = float(lbda)

    def show(self, lbda, ax=None, psfwindow=17, **kwargs):
        """ 
        Show 2D PSF with current parameters.

        Parameters
        ----------
        ax: Matplotlib.Axes -optional-
            Choice to use an existing Axes (dim=1)

        psfwindow: float
            Size of the stamp where the PSF will be display\n
            Default is 17 pix.

        kwargs: dict
            goes to imshow()

        Returns
        -------
        Figure
        """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[5,5])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        prop = dict(origin="lower", cmap="cividis")
        sc = ax.imshow(self.get_stamp(psfwindow, lbda=lbda)[0],  **{**prop,**kwargs} )
        return fig
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def lbdaref(self):
        """ 
        Reference wavelength
        """
        return self._lbdaref
    
