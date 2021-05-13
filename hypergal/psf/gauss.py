import numpy as np
from .basics import PSF2D, PSF3D

def get_radial_gauss(r, sigma, a_ell=1, b_ell=1):
    """ """
    normalisation = np.pi / np.sqrt(a_ell - b_ell**2) * (2 * sigma**2 ) 
    gaussian = np.exp(-0.5 * r**2 / sigma**2)
    return gaussian/normalisation

    

class Gauss2D( PSF2D ):
    
    PROFILE_PARAMETERS = ["sigma"] 

    # ============= #
    #  Methods      #
    # ============= #    
    def get_radial_profile(self, r):
        """ """
        sigma = self.get_sigma()
        return get_radial_gauss(r, sigma=sigma,  a_ell=self.a_ell, b_ell=self.b_ell)
    
    def get_sigma(self):
        """ """
        return self._profile_params["sigma"]
    
    def guess_parameters(self):
        """ """
        return {**{"a":1.,"b":0.},
                **{"sigma":1}
                }

    

class Gaus3D( PSF3D, Gauss2D ):

    # ============= #
    #  Methods      #
    # ============= #        
    def get_radial_profile(self, r, lbda):
        """ """
        # Most likely r -> r[:,None]
        sigma = self.get_sigma(lbda)
        return get_radial_gauss(r, sigma=sigma,  a_ell=self.a_ell, b_ell=self.b_ell)

    # ---------- #
    # Chromatic  #
    # ---------- #
    def get_sigma(self, lbda, rho=1/5):
        """ """
        sigmaref = super().get_sigma()
        return sigmaref * (lbda/self.lbdaref)*rho
    
