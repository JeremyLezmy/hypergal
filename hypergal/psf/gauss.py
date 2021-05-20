import numpy as np
from .basics import PSF2D, PSF3D


__all__ = ["Gauss2D", "Gauss3D"]

def get_radial_gauss(r, sigma, a_ell=1, b_ell=1):
    """ 
    Get normalized radial profile according for gaussian psf profile.

    Parameters
    ----------
    r: array
        Elliptical radius

    sigma: float
        Radius of the gaussian

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
    normalisation = np.pi / np.sqrt(a_ell - b_ell**2) * (2 * sigma**2 ) 
    gaussian = np.exp(-0.5 * r**2 / sigma**2)
    return gaussian/normalisation

    

class Gauss2D( PSF2D ):
    
    PROFILE_PARAMETERS = ["sigma"] 

    # ============= #
    #  Methods      #
    # ============= #    
    def get_radial_profile(self, r):
        """ 
        Get gaussian radial profile according to its elliptical radius.

        Parameters
        ----------
        r: array
            Elliptical radius

        Returns
        -------
        Array of the radial profile at the *r* position.
        """
        sigma = self.get_sigma()
        return get_radial_gauss(r, sigma=sigma,  a_ell=self.a_ell, b_ell=self.b_ell)
    
    def get_sigma(self):
        """ 
        Return the value of the radius of the gaussian parameter.
        """
        return self._profile_params["sigma"]
    
    def guess_parameters(self):
        """ 
        Init parameters (default) for the 2D gaussian profile.
        Return
        --------
        Elliptical parameters ("a" and "b") and the shape parameter (sigma)
        """
        return {**{"a":1.,"b":0.},
                **{"sigma":1}
                }

    

class Gauss3D( PSF3D, Gauss2D ):

    # ============= #
    #  Methods      #
    # ============= #
    CHROM_PARAM = ['sigma']
    PROFILE_PARAMETERS = ['sigma', 'rho']
    
    
    def fit_from_values(self, values, errors, lbda):
        """ """
        
        param3d={}
        
        for param in self.PARAMETER_NAMES:
            
            if param not in self.CHROM_PARAM and param in values.keys():  ###Compute weighted mean for non-chromatics parameters
                
                val = np.array(values[param ] )   
                err = np.array( errors[param] ) if errors is not None else np.ones(len(val))
                
                param3d[param] = np.sum( np.dot(val, err**-2))/np.sum( err**-2 ) 
                
            elif param in self.CHROM_PARAM and param in values.keys():   ###If param is chromatic 

                    val = np.array(values[param ] )   
                    err = np.array( errors[param ] ) if errors is not None else np.ones(len(val))
                   
                    if self.lbdaref is None:
                        param3d[param] = np.sum( np.dot(val, err**-2))/np.sum( err**-2 )
                        continue
                        
                    def get_chromparam(X):    ####function which goes in minimize (general if power law)
                        paramref=X[0]      
                        power=X[1]
                        locals()[param] = paramref
                        self.update_parameters(**{param :paramref})
                        return( np.sum( (getattr(self, 'get_'+param)(lbda, power) - val)**2/ err**2))

                    import scipy
                    sci=scipy.optimize.minimize(get_chromparam, np.array([1,1]) )
                    param_chrom = sci.x[0]
                    power_chrom = sci.x[1]
                    param3d[param] = param_chrom
                    param3d["rho"] = power_chrom
        
        self.update_parameters(**{k:param3d[k] for k in self.PARAMETER_NAMES})
      
        
    def get_radial_profile(self, r, lbda):
        """ 
        Get gaussian radial profile according to its elliptical radius and the wavelength (3rd dimension).

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
        sigma = self.get_sigma(lbda)
        return get_radial_gauss(r, sigma=sigma,  a_ell=self.a_ell, b_ell=self.b_ell)

    # ---------- #
    # Chromatic  #
    # ---------- #
    def get_sigma(self, lbda, rho=None):
        """ 
        Chromatic shape parameter for the gaussian profile.\n
        Power law such as sigma = sigmaref * (lbda/lbdaref)^rho

        Parameters
        ----------
        lbda: float
            Wavelength (should be same unit than self.lbdaref)

        rho: float
            Power of the wavelength power law\n
            Default is -1.5

        Returns
        -------
        Float
        """
        sigmaref = super().get_sigma()
        if rho is None:
            rho = self._profile_params['rho']
        return sigmaref * (lbda/self.lbdaref)**rho
    
