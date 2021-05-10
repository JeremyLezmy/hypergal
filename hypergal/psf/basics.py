""" """

import numpy as np
from astropy.convolution import convolve

class PSF2D( object ):
    
    ELLIPTICITY_PARAMETERS = ["a","b"]
    
    def __init__(self, **kwargs):
        """ """
        self._ellipticity_params = {k:1 for k in self.ELLIPTICITY_PARAMETERS}
        self._profile_params = {}
        self.update_parameters(**kwargs)
        
    # ============= #
    #  Methods      #
    # ============= #
    def update_parameters(self, **kwargs):
        """ """
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
        """ """
        self.update_parameters(**kwargs)
        dx = x-x0
        dy = y-y0
        r = np.sqrt( dx**2 + self.a_ell*dy**2 + 2*self.b_ell * (dx*dy) )
        return self.get_radial_profile(r)

    def get_radial_profile(self, r):
        """ """
        raise NotImplementedError("You must define your PSF self.get_radial_profile()")

    def get_stamp(self, psfwindow=17, **kwargs):
        """ """
        x, y = np.mgrid[-psfwindow/2:psfwindow/2, -psfwindow/2:psfwindow/2]+0.5
        return self.evaluate(x, y, **kwargs)
    
    def convolve(self, arr2d, psfwindow=17, **kwargs):
        """ """
        return convolve(arr2d, self.get_stamp(psfwindow=psfwindow, **kwargs), normalize_kernel=False)


    # -------- #
    #  PLOTTER #
    # -------- #
    def show(self, ax=None, psfwindow=17, **kwargs):
        """ 
        kwargs goes to imshow()
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
        """ a elliticity parameter """
        return self._ellipticity_params["a"]

    @property
    def b_ell(self):
        """ a elliticity parameter """
        return self._ellipticity_params["b"]


class PSF3D( PSF2D ):

    def __init__(self, lbdaref=6000, **kwargs):
        """ """
        _ = super().__init__(**kwargs)
        self.set_lbdaref(lbdaref)
        return _
        
    # ============= #
    #  Methods      #
    # ============= #
    def evaluate(self, x, y, lbda, x0=0, y0=0, **kwargs):
        """ """
        self.update_parameters(**kwargs)
        dx = x-x0
        dy = y-y0
        r = np.sqrt( dx**2 + self.a_ell*dy**2 + 2*self.b_ell * (dx*dy) )
        return self.get_radial_profile(r, lbda)

    def get_radial_profile(self, r, lbda):
        """ """
        raise NotImplementedError("You must define your PSF self.get_radial_profile()")
    
    def set_lbdaref(self, lbda):
        """ """
        self._lbdaref = float(lbda)
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def lbdaref(self):
        """ """
        self._lbdaref
    
