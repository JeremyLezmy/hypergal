import numpy as np
from .basics import PSF2D, PSF3D

__all__ = ["GaussMoffat2D", "GaussMoffat3D"]


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

    CHROMATIC_PARAMETERS = ['sigma']
    PROFILE_PARAMETERS = ['alpha', 'eta', 'sigma', 'rho']
    
    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def fit_from_values(cls, values, lbda, errors=None, **kwargs):
        """ 

        Parameters
        ----------
        values: [dict/serie]
            dictionary or pandas.Series containing the freerameters 
            (a, b | sigma)

        lbda: [array]
            wavelength assiated to the input values

        errors: [dict/serie or None] -optional-
            errors associated to the inpout values, same format.

        Returns
        -------
        Gauss3D
        """
        from scipy.optimize import minimize
        this = cls(**kwargs)
        
        param3d = {}
        # Loop over the PARAMETER_NAMES and given the values, errors and lbda
        #   - get the mean values if the parameter is not chromatic
        #   - fit the instance profile if it is.
        for param in this.PARAMETER_NAMES:
            
            # Non chromatic parameters | for instance a and b
            #   -> Compute weighted mean for non-chromatics parameters
            if param not in this.CHROMATIC_PARAMETERS and param in values.keys(): 
                value = np.asarray(values[param])
                if errors is not None:
                    variance = np.asarray(errors[param])**2
                    param3d[param] = np.average(value, weights=1/variance)
                else:
                    param3d[param] = np.mean(value)
                
            # Non chromatic parameters
            elif param in this.CHROMATIC_PARAMETERS and param in values.keys():   ###If param is chromatic
                # Sigma
                value = np.asarray(values[param])
                variance = np.asarray(errors[param])**2 if errors is not None else np.ones( len(value) )
                       
                def get_chromparam(arr_):
                    """ function to be minimizing """
                    sigma_, rho_ = arr_
                    this.update_parameters(**{"sigma":sigma_, "rho":rho_})
                    model = this.get_sigma(lbda) # rho has been updated already
                    chi2 = np.sum( (value-model)**2/variance )
                    return chi2

                fit_output= minimize( get_chromparam, np.array([1,1]) )
                
                param3d["sigma"] = fit_output.x[0]
                param3d["rho"]   = fit_output.x[1]
        
        this.update_parameters(**param3d)
        return this

    
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
        if rho is None:
            rho = self._profile_params['rho']
            
        return sigmaref * (np.atleast_1d(lbda)/self.lbdaref)**rho

    def guess_parameters(self):
        """ 
        Init parameters (default) for the 2D gaussianMoffat profile.
        Return
        --------
        Elliptical parameters ("a" and "b") and the shape parameter (sigma)
        """
        return {**super().guess_parameters(), **{"rho":-0.5}}
    
