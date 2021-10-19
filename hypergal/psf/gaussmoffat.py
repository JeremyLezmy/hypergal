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
    normalisation = (np.pi / np.sqrt(a_ell - b_ell**2) *
                     (2 * eta * sigma**2 + alpha**2 / (beta - 1)))
    gaussian = np.exp(-0.5 * r**2 / sigma**2)
    moffat = (1+(r/alpha)**2)**(-beta)
    return (eta*gaussian + moffat)/normalisation


class GaussMoffat2D(PSF2D):

    PROFILE_PARAMETERS = ["eta", "alpha"]  # beta and sigma fixed by alpha

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
        beta = self.get_beta()
        sigma = self.get_sigma()
        eta = self.get_eta()
        return get_radial_gaussmoffat(r, alpha=alpha, beta=beta,
                                      sigma=sigma, eta=eta,
                                      a_ell=self.a_ell, b_ell=self.b_ell)

    def get_alpha(self):
        """ 
        Return Moffat radius.
        """
        return self._profile_params["alpha"]

    def get_beta(self, b0=1.51, b1=0.22):
        """ 
        Return Moffat power. Beta is fixed by alpha value such as beta = b1*alpha + b0

        Parameters
        ----------
        b0: float
            Default is 1.51

        b1: float
            Default is 0.22 

        Returns
        -------
        Beta Moffat power
        """
        return b0 + self.get_alpha() * b1

    def get_sigma(self, sig0=0.38, sig1=0.40):
        """ 
        Return gaussian radius. Sigma is fixed by alpha value such as sigma = sig1*alpha + sig0

        Parameters
        ----------
        sig0: float
            Default is 0.38

        sig1: float
            Default is 0.40

        Returns
        -------
        Sigma Gaussian radius
        """
        return sig0 + self.get_alpha()*sig1

    def get_eta(self):
        """ 
        Return weight between gaussian and Moffat such as PSF = eta * Gauss + Moff
        """
        return self._profile_params["eta"]

    def guess_parameters(self):
        """ 
        Default parameters (init for an eventual fit)
        """
        return {**{"a": 1., "b": 0.},
                **{"alpha": 2, "eta": 0.8}
                }

    # ============= #
    #  Properties   #
    # ============= #


class GaussMoffat3D(PSF3D, GaussMoffat2D):

    CHROMATIC_PARAMETERS = ['alpha']
    PROFILE_PARAMETERS = ['alpha', 'eta', 'rho']

    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def fit_from_values(cls, values, lbda, errors=None, saveplot=None, **kwargs):
        """ 

        Parameters
        ----------
        values: [dict/serie]
            dictionary or pandas.Series containing the free parameters 
            (a, b | alpha)

        lbda: [array]
            wavelength assiated to the input values

        errors: [dict/serie or None] -optional-
            errors associated to the inpout values, same format.

        Returns
        -------
        Gauss3D
        """
        from scipy.optimize import minimize
        from iminuit import Minuit, cost
        from astropy.stats import sigma_clipping
        this = cls(**kwargs)

        param3d = {}
        mainlbda = lbda.copy()
        # Loop over the PARAMETER_NAMES and given the values, errors and lbda
        #   - get the mean values if the parameter is not chromatic
        #   - fit the instance profile if it is.
        for param in this.PARAMETER_NAMES:

            # Non chromatic parameters | for instance a and b
            #   -> Compute weighted mean for non-chromatics parameters
            if param not in this.CHROMATIC_PARAMETERS and param in values.keys():

                if len(values[param]) < 2:
                    param3d[param] = values[param]

                else:
                    if param == 'eta':
                        value_ = np.asarray(values[param])
                        flag = np.logical_or(value_ < 1e-10, value_ > 100)
                        value = value_[~flag].copy()
                        lbda = mainlbda[~flag].copy()
                        variance_ = np.asarray(
                            errors[param])**2 if errors is not None else np.ones(len(value_))
                        variance = variance_[~flag].copy()
                    else:
                        value = np.asarray(values[param])
                        variance = np.asarray(
                            errors[param])**2 if errors is not None else np.ones(len(value_))
                        lbda = mainlbda.copy()

                    def model_cst(lbda, cst):
                        return np.tile(cst, len(lbda))

                    c = cost.LeastSquares(lbda, value, variance**0.5, model_cst)
                    c.loss = "soft_l1"
                    m = Minuit(c, cst=np.average(value, weights=1/variance))
                    migout = m.migrad()
                    param3d[param] = m.values[0]

            # Non chromatic parameters
            elif param in this.CHROMATIC_PARAMETERS and param in values.keys():  # If param is chromatic
                # Alpha
                if len(values[param]) < 2:
                    param3d["alpha"] = values[param]
                    param3d["rho"] = -0.4

                else:
                    value_ = np.asarray(values[param])
                    # alpha > 7 means fwhm>4" alpha<0.9 means fwhm<1"
                    flag = np.logical_or(value_ > 7, value_ < 0.9)
                    value = value_[~flag].copy()
                    lbda = mainlbda[~flag].copy()
                    variance_ = np.asarray(
                        errors[param])**2 if errors is not None else np.ones(len(value_))
                    variance = variance_[~flag].copy()

                    def model_alpha(lbda, alpharef, rho):
                        this.update_parameters(
                            **{"alpha": alpharef, "rho": rho})
                        return this.get_alpha(lbda)

                    c = cost.LeastSquares(
                        lbda, value, variance**0.5, model_alpha)
                    c.loss = "soft_l1"
                    m = Minuit(c, alpharef=2, rho=-0.4)
                    migout = m.migrad()

                    # def get_chromparam(arr_):
                    #    """ function to be minimizing """
                    #    alpha_, rho_ = arr_
                    #    this.update_parameters(**{"alpha":alpha_, "rho":rho_})
                    #    model = this.get_alpha(lbda) # rho has been updated already
                    #    chi2 = np.sum( (value-model)**2/variance )
                    #    return chi2

                    #fit_output= minimize( get_chromparam, np.array([2,-0.4]) )

                    #param3d["alpha"] = fit_output.x[0]
                    #param3d["rho"]   = fit_output.x[1]

                    param3d["alpha"] = m.values[0]
                    param3d["rho"] = m.values[1]

        this.update_parameters(**param3d)
        if saveplot is not None:
            this.show_chromfit(values, mainlbda, errors, saveplot)
        return this

    def get_beta(self, lbda, b0=1.51, b1=0.22, rho=None):
        """ 
        Return Moffat power. Beta is fixed by alpha value such as beta = b1*alpha + b0

        Parameters
        ----------
        b0: float
            Default is 1.51

        b1: float
            Default is 0.22 

        Returns
        -------
        Beta Moffat power
        """
        return b0 + self.get_alpha(lbda=lbda, rho=rho) * b1

    def get_sigma(self, lbda, sig0=0.38, sig1=0.40, rho=None):
        """ 
        Return gaussian radius. Sigma is fixed by alpha value such as sigma = sig1*alpha + sig0

        Parameters
        ----------
        sig0: float
            Default is 0.38

        sig1: float
            Default is 0.40

        Returns
        -------
        Sigma Gaussian radius
        """
        return sig0 + self.get_alpha(lbda=lbda, rho=rho)*sig1

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
        alpha = self.get_alpha(lbda)
        beta = self.get_beta(lbda)
        sigma = self.get_sigma(lbda)
        eta = self.get_eta()
        return get_radial_gaussmoffat(r, alpha=alpha, beta=beta,
                                      sigma=sigma, eta=eta,
                                      a_ell=self.a_ell, b_ell=self.b_ell)

    # ---------- #
    # Chromatic  #
    # ---------- #
    def get_alpha(self, lbda, rho=None):
        """ 
        Chromatic shape parameter for the Moffat radius.\n
        Power law such as alpha = alpharef * (lbda/lbdaref)^rho

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
        alpharef = super().get_alpha()
        if rho is None:
            try:
                rho = self._profile_params['rho']
            except KeyError:
                rho = -0.4

        return alpharef * (np.atleast_1d(lbda)/self.lbdaref)**rho

    def guess_parameters(self):
        """ 
        Init parameters (default) for the 2D gaussianMoffat profile.
        Return
        --------
        Elliptical parameters ("a" and "b") and the shape parameter (sigma)
        """
        return {**super().guess_parameters(), **{"rho": -0.4}}

    def show_chromfit(self, values, lbda, errors=None, saveplot=None):

        import matplotlib.pyplot as plt
        nparm = len([k for k in self.parameters if k in values.keys()])
        if nparm % 2 == 0:
            fig, axs = plt.subplots(
                2, int(nparm/2), figsize=(15, 10), sharex=True)
        else:
            fig, axs = plt.subplots(
                2, int(nparm/2)+1, figsize=(15, 10), sharex=True)

        # for param in self.PARAMETER_NAMES:

        for (ax, param) in zip(axs.flat, self.PARAMETER_NAMES):
            if param in self.CHROMATIC_PARAMETERS and param in values.keys():
                ax.scatter(lbda, values[param], color='k',
                           label='Metaslices fitted values')
                ax.plot(np.linspace(np.min(lbda)-100, np.max(lbda)+100, 100), self.get_alpha(np.linspace(np.min(lbda)-100, np.max(lbda)+100, 100)), color='r',
                        label=fr'Chromatic Fit (power law) $\alpha(\lambda)={self.parameters["alpha"]:.2f}\left(\frac{{\lambda}}{{{self.lbdaref}}}\right)^{{{self.parameters["rho"]:.2f}}}$')
                if errors[param] is not None:
                    ax.errorbar(lbda, values[param],
                                errors[param], fmt='none', color='k')
            elif param not in self.CHROMATIC_PARAMETERS and param in values.keys():
                ax.scatter(lbda, values[param], color='k',
                           label='Metaslices fitted values')
                ax.hlines(self.parameters[param], np.min(lbda)-100, np.max(lbda)+100, color='r',
                          label=fr'Chromatic Fit (constant) {param}={self.parameters[param]:.3f}')
                if errors[param] is not None:
                    ax.errorbar(lbda, values[param],
                                errors[param], fmt='none', color='k')

            ax.set_xlabel(r'$\lambda(\AA)$', fontsize=13)
            if param == 'alpha':
                ax.set_ylabel(r'$\alpha (\lambda)$', fontsize=13)
                ax.set_ylim(0.5, np.min([8, np.max(values[param])]))

            elif param == 'eta':
                ax.set_ylabel(r'$\eta (\lambda)$', fontsize=13)
                ax.set_ylim(-1, np.min([8, np.max(values[param])]))
            elif param == 'sigma':
                ax.set_ylabel(r'$\sigma (\lambda)$', fontsize=13)
            else:
                ax.set_ylabel(param + r' ($\lambda$)', fontsize=13)
            ax.tick_params(axis='both', labelsize=13)
            ax.legend(fontsize=11)

        fig.suptitle('3D fit of Chromatic parameters',
                     fontsize=16, fontweight="bold", y=0.99)
        if saveplot is not None:
            fig.savefig(saveplot)
