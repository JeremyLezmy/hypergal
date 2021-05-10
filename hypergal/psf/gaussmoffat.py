import numpy as np
from .basics import PSF2D, PSF3D

def get_radial_gaussmoffat(r, alpha, beta, sigma, eta, a_ell=1, b_ell=1):
    """ """
    normalisation = (np.pi / np.sqrt(a_ell - b_ell**2) * \
                 (2 * eta * sigma**2 + alpha**2/ (beta - 1)) )
    gaussian = np.exp(-0.5 * r**2 / sigma**2)
    moffat = ( 1+(r/alpha)**2 )**(-beta)
    return (eta*gaussian + moffat)/normalisation

    

class GaussMoffat2D( PSF2D ):
    
    PROFILE_PARAMETERS = ["eta", "sigma", "alpha"]

    # ============= #
    #  Methods      #
    # ============= #    
    def get_radial_profile(self, r):
        """ """
        alpha = self.get_alpha()
        beta  = self.get_beta()
        sigma = self.get_sigma()
        eta   = self.get_eta()
        return get_radial_gaussmoffat(r, alpha=alpha, beta=beta,
                                         sigma=sigma, eta=eta,
                                         a_ell=self.a_ell, b_ell=self.b_ell)
    
    def get_alpha(self):
        """ """
        return self._profile_params["alpha"]

    def get_beta(self, b0=1.35, b1=0.22):
        """ """
        return b0  + self.get_alpha() * b1 
    
    def get_sigma(self):
        """ """
        return self._profile_params["sigma"]
    
    def get_eta(self):
        """ """
        return self._profile_params["eta"]

    # ============= #
    #  Properties   #
    # ============= #    


class GaussMoffat3D( PSF3D, GaussMoffat2D ):

    # ============= #
    #  Methods      #
    # ============= #        
    def get_radial_profile(self, r, lbda):
        """ """
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
    def get_sigma(self, lbda, rho=1/5):
        """ """
        sigmaref = super().get_sigma()
        return sigmaref * (lbda/self.lbdaref)*rho
    
