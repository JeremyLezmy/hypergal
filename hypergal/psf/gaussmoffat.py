import numpy as np
from .basics import PSF2D, PSF3D

def get_radial_gaussmoffat(r, alpha, beta, sigma, eta, a_ell=1, b_ell=1):
    """ 
    Get normalized radial profile for a gaussian (core) + Moffat (wings) psf profile.

    Parameters
    ----------
    r: array
        Elliptical radius

    alpha: float
        Moffat radius

    beta: float
        Moffat power

    sigma: float
        Radius of the gaussian

    eta: float
        Weight between Gaussian and Moffat such as PSF = eta*Gauss + Moff

    a_ell: float
        Elliptical parameter

    b_ell: float
        Elliptical parameter

    Returns
    -------
    Array of the normalized radial profile at the *r* position.

    Note
    ---------
    "a" and "b" descirbed simultaneously the orientation (angle) and the ratio of the two axes.
    """
    normalisation = (np.pi / np.sqrt(a_ell - b_ell**2) * \
                 (2 * eta * sigma**2 + alpha**2/ (beta - 1)) )
    gaussian = np.exp(-0.5 * r**2 / sigma**2)
    moffat = ( 1+(r/alpha)**2 )**(-beta)
    return (eta*gaussian + moffat)/normalisation

    

class GaussMoffat2D( PSF2D ):
    
    PROFILE_PARAMETERS = ["eta", "sigma", "alpha"] # beta fixed by alpha

    # ============= #
    #  Methods      #
    # ============= #    
    def get_radial_profile(self, r):
        """ 
        Get gaussian + Moffat radial profile according to its elliptical radius and current parameters.

        Parameters
        ----------
        r: array
            Elliptical radius

        Returns
        -------
        Array of the radial profile at the *r* position.
        """
        alpha = self.get_alpha()
        beta  = self.get_beta()
        sigma = self.get_sigma()
        eta   = self.get_eta()
        return get_radial_gaussmoffat(r, alpha=alpha, beta=beta,
                                         sigma=sigma, eta=eta,
                                         a_ell=self.a_ell, b_ell=self.b_ell)
    
    def get_alpha(self):
        """ 
        Return Moffat radius.
        """
        return self._profile_params["alpha"]

    def get_beta(self, b0=1.35, b1=0.22):
        """ 
        Return Moffat power. Beta is fixed by alpha value such as beta = b0*alpha + b1

        Parameters
        ----------
        b0: float
            Default is 1.35

        b1: float
            Default is 0.22 

        Returns
        -------
        Beta Moffat power
        """
        return b0  + self.get_alpha() * b1 
    
    def get_sigma(self):
        """ 
        Return gaussian radius.
        """
        return self._profile_params["sigma"]
    
    def get_eta(self):
        """ 
        Return weight between gaussian and Moffat such as PSF = eta * Gauss + Moff
        """
        return self._profile_params["eta"]

    def guess_parameters(self):
        """ 
        Default parameters (init for an eventual fit)
        """
        return {**{"a":1.,"b":0.},
                **{"alpha":2., "eta":1., "sigma":2.}
                }
    
    # ============= #
    #  Properties   #
    # ============= #    


class GaussMoffat3D( PSF3D, GaussMoffat2D ):

    # ============= #
    #  Methods      #
    # ============= #        
    def get_radial_profile(self, r, lbda):
        """ 
        Get gaussian + Moffat radial profile according to its elliptical radius and the wavelength (3rd dimension).

        Parameters
        ----------
        r: array
            Elliptical radius

        lbda: float
            Wavelength, used for chromatic parameters.

        Returns
        -------
        Array of the radial profile at the *r* position and the *lbda* wavelength.
        """
        # Most likely r -> r[:,None]
        alpha = self.get_alpha()
        beta  = self.get_beta()
        sigma = self.get_sigma(lbda)
        eta   = self.get_eta()
        return get_radial_gaussmoffat(r, alpha=alpha, beta=beta,
                                         sigma=sigma, eta=eta,
                                         a_ell=self.a_ell, b_ell=self.b_ell)

    # ---------- #
    # Chromatic  #
    # ---------- #
    def get_sigma(self, lbda, rho=-1.5):
        """ 
        Chromatic shape parameter for the gaussian radius.\n
        Power law such as sigma = sigmaref * (lbda/lbdaref)^rho

        Parameters
        ----------
        lbda: float
            Wavelength (should be same unit than self.lbdaref)

        rho: [float]
            Power of the wavelength power law\n
            Default is -1.5
        
        Returns
        -------
        Float
        """
        sigmaref = super().get_sigma()
        return sigmaref * (lbda/self.lbdaref)**rho
    
