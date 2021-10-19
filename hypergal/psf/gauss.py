import numpy as np
from .basics import PSF2D, PSF3D


__all__ = ["Gauss2D", "Gauss3D"]


def get_radial_gauss(r, sigma, a_ell=1, b_ell=0):
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
    normalisation = np.pi / np.sqrt(a_ell - b_ell**2) * (2 * sigma**2)
    gaussian = np.exp(-0.5 * r**2 / sigma**2)
    return gaussian/normalisation


class Gauss2D(PSF2D):

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
        return {**{"a": 1., "b": 0.},
                **{"sigma": 1}
                }


class Gauss3D(PSF3D, Gauss2D):

    CHROMATIC_PARAMETERS = ['sigma']
    PROFILE_PARAMETERS = ['sigma', 'rho']

    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def fit_from_values(cls, values, lbda, errors=None, **kwargs):
        """ 

        Parameters
        ----------
        values: [dict/serie]
            dictionary or pandas.Series containing the freepameters 
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
        from iminuit import Minuit
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
            elif param in this.CHROMATIC_PARAMETERS and param in values.keys():  # If param is chromatic
                # Sigma
                if len(values[param]) < 2:
                    param3d["sigma"] = values[param]
                    param3d["rho"] = -0.4
                else:
                    value = np.asarray(values[param])
                    variance = np.asarray(
                        errors[param])**2 if errors is not None else np.ones(len(value))

                    from iminuit import cost

                    def model_sigma(lbda, sigmaref, rho):
                        this.update_parameters(
                            **{"sigma": sigmaref, "rho": rho})
                        return this.get_sigma(lbda)

                    c = cost.LeastSquares(
                        lbda, value, variance**0.5, model_sigma)
                    c.loss = "soft_l1"
                    m = Minuit(c, sigmaref=2, rho=-0.4)
                    migout = m.migrad()

                    # def get_chromparam(arr_):
                    #    """ function to be minimizing """
                    #    sigma_, rho_ = arr_
                    #    this.update_parameters(**{"sigma":sigma_, "rho":rho_})
                    #    model = this.get_sigma(lbda) # rho has been updated already
                    #    chi2 = np.sum( (value-model)**2/variance )
                    #    return chi2

                    #fit_output= minimize( get_chromparam, np.array([1,1]) )

                    #param3d["sigma"] = fit_output.x[0]
                    #param3d["rho"]   = fit_output.x[1]

                    param3d["sigma"] = m.values[0]
                    param3d["rho"] = m.values[1]

        this.update_parameters(**param3d)
        return this

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
        return get_radial_gauss(r, sigma=sigma[:, None, None],
                                a_ell=self.a_ell, b_ell=self.b_ell)

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

        return sigmaref * (np.atleast_1d(lbda)/self.lbdaref)**rho

    def guess_parameters(self):
        """ 
        Init parameters (default) for the 2D gaussian profile.
        Return
        --------
        Elliptical parameters ("a" and "b") and the shape parameter (sigma)
        """
        return {**super().guess_parameters(), **{"rho": -0.5}}
