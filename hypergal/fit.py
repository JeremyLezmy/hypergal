
import warnings
import pandas
import numpy as np
from scipy import stats
from iminuit import Minuit


# ===================== #
#                       #
#      PRIORS           #
#                       #
# ===================== #
class Priors( object ):
    """ 
    Priors object which can be used for the hypergal fit process.
    """
    
    BOUND_VALUE = 1e-12
    def set_parameters(self, parameters):
        """ 
        Set parameters you want to constraint.

        Parameters
        ----------
        parameters: dict
            Key should be the name of the parameter (value its value)

        Returns
        -------
        """
        self._parameters = parameters
        self._parameter_names = list(self.parameters.keys())
        
    # ============== #
    #  Methods       #
    # ============== #
    def get_product(self):
        """ 
        Return product of priors distribution on the setted parameters in self.parameters.
        """
        priors = []
        if self.has_ellipticity_param():
            a = self.parameters["a"]
            b = self.parameters["b"]
            priors.append(self.get_ellipticity_prior(a,b))
        
        # np.prod([])-> 1
        return np.prod(priors)
    
    
    # -------- #
    #  HAS     #
    # -------- #
    def has_ellipticity_param(self):
        """ 
        Check if ellipticity parameters ('a' and 'b') are in self.parameter_name.
        """
        return "a" in self.parameter_names and \
               "b" in self.parameter_names
    
    # ============== #
    #  Statics       #
    # ============== #
    
    @staticmethod
    def get_ellipticity_prior(a, b, max_ratio=0.9, q_troncnorm={"loc":0, "scale":0.15, "a":0, "b":4 }):
        """ 
        Get prior on ellipticity parameters
        """
        if b>max_ratio * np.sqrt(a):
            return 0

        numerator = np.sqrt( (1-a)**2 + 4*b**2) - (1+a)
        denominator = -np.sqrt( (1-a)**2 + 4*b**2) - (1+a)
        q = (1-numerator/denominator)
        return stats.truncnorm.pdf(q, **q_troncnorm)

    
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def parameters(self):
        """ 
        Parameters value loaded in self.
        """
        return self._parameters
    
    @property
    def parameter_names(self):
        """ 
        Parameters name loaded in self.
        """
        return self._parameter_names 
    
                    

# ===================== #
#                       #
#      FITTER           #
#                       #
# ===================== #

class SceneFitter( object ):
    def __init__(self, scene, fix_params=["scale","rotation"], priors=None, debug=False):
        """ 
        Main Scene Fitter of a given slice/cube in an IFU. 

        Parameters
        ----------
        scene: SliceScene/HostScene
            Scene object instantiates with the slice/cube from photometric source and the data slice/cube from IFU you want to model

        fix_params: list of string -optional-
            Parameters you want to be fixed during the fit.\n
            Default is ["scale","rotation"]

        priors: .Priors() -optional-
            Prior object to constraint some parameter. \n
            Can be instantiate in .Priors()\n
            Default is None

        debug: bool
            If True, will print steps informations  during the fit.\n
            Default is None.
        """
        if scene is not None:
            self.set_scene(scene)
            
        self.set_fixed_params(fix_params)

        if priors is None:
            priors = Priors()
        self.set_priors(priors)
        self._debug = debug
        
    # ============== #
    # Initialisation #
    # ============== #
    @classmethod
    def from_slices(cls, slice_in, slice_comp, psf, whichscene="HostSlice",
                        xy_in=None, xy_comp=None, 
                    fix_params=["scale","rotation"], debug=False, **kwargs):
        """ 
        Main Scene Fitter of a given slice/cube in an IFU. Instantiate from slice datas instead of SceneObject/HostObject.

        Parameters
        ----------
        slice_in: pyifu.Slice
             Slice you want to use to model slice_comp.

        slice_comp: pyifu.Slice
             Slice you want to model. 
        
        psf: hypergal.psf
             Set a psf object, with which slice_in will be convolve before the projection in slice_comp geometry.

        whichscene: string
            Might be 'HostSlice' or 'SliceScene'

        xy_in, xy_comp: 2d-array (float) or None
             Reference coordinates (target position) for the _in and _comp geometries\n
             e.g. xy_comp = [3.1,-1.3]
        
        fix_params: list of string -optional-
            Parameters you want to be fixed during the fit.\n
            Default is ["scale","rotation"]

        debug: bool
            If True, will print steps informations  during the fit.\n
            Default is None.
        """
        if whichscene == "HostSlice":
            from .scene import host
            scene = host.HostSlice.from_slices(slice_in, slice_comp, 
                                               xy_in=xy_in, xy_comp=xy_comp, 
                                               psf=psf, **kwargs)
        else:
            raise NotImplementedError("Only HostSlice scene has been implemented.")
        
        return cls.from_scene(scene, fix_params=fix_params, debug=debug)
    
    @classmethod
    def from_scene(cls, scene, fix_params=["scale","rotation"], debug=False, **kwargs):
        """ 
        Main Scene Fitter of a given slice/cube in an IFU. 

        Parameters
        ----------
        scene: SliceScene/HostScene
            Scene object instantiates with the slice/cube from photometric source and the data slice/cube from IFU you want to model

        fix_params: list of string -optional-
            Parameters you want to be fixed during the fit.\n
            Default is ["scale","rotation"]

        priors: .Priors() -optional-
            Prior object to constraint some parameter. \n
            Can be instantiate in .Priors()\n
            Default is None

        debug: bool
            If True, will print steps informations  during the fit.\n
            Default is None.
        """
        return cls(scene, fix_params=fix_params, debug=debug, **kwargs)

    # ============== #
    # Class Method   #
    # ============== #
    @classmethod
    def fit_slices_projection(cls, slice_in, slice_comp, psf, whichscene="HostSlice",
                                  xy_in=None, xy_comp=None, 
                                  fix_params=["scale","rotation"], debug=False,
                                  guess=None, limit=None, error=None, use_priors=True,
                                  savefile=None, result_as_dataframe=True):
        """ 
        Main Scene Fitter of a given slice/cube in an IFU. Instantiate from slice datas instead of SceneObject/HostObject.
        Directly fit the scene after the instantiation.

        Parameters
        ----------
        slice_in: pyifu.Slice
             Slice you want to use to model slice_comp.

        slice_comp: pyifu.Slice
             Slice you want to model. 
        
        psf: hypergal.psf
             Set a psf object, with which slice_in will be convolve before the projection in slice_comp geometry.

        whichscene: string
            Might be 'HostSlice' or 'SliceScene'

        xy_in,xy_comp: 2d-array (float) or None
             reference coordinates (target position) for the _in and _comp geometries
             e.g. xy_comp = [3.1,-1.3]
        
        fix_params: list of string -optional-
            Parameters you want to be fixed during the fit.\n
            Default is ["scale","rotation"]

        debug: bool -optional-
            If True, will print steps informations  during the fit.\n
            Default is None.

        guess: dict -optional-
            Guess values for the parameters to fit. Keys are names of parameters.\n
            Default is None.

        limit: dict -optional-
            Limit values (bounds) for the parameters to fit.  Keys are names of parameters.\n
            Default is None.

        error: array -optional-
            Access parameter parabolic errors via an array-like view (see Minuit.error)\n
            Defaut is None.

        use_prior: bool -optional-
            If True (Default), will use setted priors and therefore maximize likelihood instead of minimize Chi square.

        savefile: string -optional-
            If not None, will save the plot which shows the model and residual scene after the minimisation.
            
        result_as_dataframe: bool -optional-
            If True (Default), will return the best fitted parameters as a DataFrame

        Returns
        -------
        Dict or DataFrame            
        """
        this = cls.from_slices(slice_in, slice_comp,  psf=psf,
                                   whichscene=whichscene,
                                   xy_in=xy_in, xy_comp=xy_comp, 
                                   fix_params=fix_params, debug=debug)
        migradout = this.fit(guess=guess, limit=limit, error=error, use_priors=use_priors,
                                 runmigrad=True)
        if savefile is not None:
            this.scene.show(savefile=savefile)
            
        return this.get_bestfit_parameters(as_dataframe=result_as_dataframe)
    
        
        
    # ============== #
    #   Methods      #
    # ============== #
    # ------- #
    # SETTER  #
    # ------- #
    def set_scene(self, scene):
        """ 
        Set Scene object (see hypergal.scene).

        Parameters
        ----------
        scene: SliceScene/HostScene
            Scene object instantiates with the slice/cube from photometric source and the data slice/cube from IFU you want to model

        Returns
        -------
        """
        self._scene = scene
        self._base_parameters = {k:None for k in self.scene.BASE_PARAMETERS}
        self._psf_parameters = {k:None for k in self.scene.PSF_PARAMETERS}
        self._geometry_parameters = {k:None for k in self.scene.GEOMETRY_PARAMETERS}
        
    def set_fixed_params(self, list_of_params):
        """ 
        Set parameters you want to be fixed during the fit.

        Parameters
        ----------
        list_of_params: list of string
            List of parameters to be fixed (e.g. ["scale","rotation"] ) 

        """
        if list_of_params is None or len(list_of_params)==0:
            self._fixedparams = []
        else:
            list_of_params = np.atleast_1d(list_of_params)
            for k in list_of_params:
                if k not in self.PARAMETER_NAMES:
                    raise ValueError(f"{k} is not a known parameter")
            
            self._fixedparams = list_of_params
        
        self._freeparams = [k for k in self.PARAMETER_NAMES if k not in self._fixedparams]
        
    def set_freeparameters(self, parameters):
        """ 
        Set free parameters value. 

        Parameters
        ----------
        parameters: dict
             Dict of all free parameters, key being the name, value the value.

        """
        if len(parameters) != self.nfree_parameters:
            raise ValueError(f"you must provide {self.nfree_parameters} parameters, you gave {len(parameters)}")
            
        dfreeparam = {k:v for k, v in zip(self.free_parameters, parameters)}
        if self._debug:
            print(f"setting: {dfreeparam}")
        return self.update_parameters(**dfreeparam)

    def set_priors(self, priors):
        """ 
        Set prior from Priors() object.
        """
        self._priors = priors
            
    def set_bestfit(self, dictparameters):
        """ 
        Set the bestfit parameters values. Automatically call at the end of the fit method, but you can set it if you already know these values.

        Parameters
        ----------
        dictparameters: dict
            Dictionary of parameters. \n
            If keys end with '_err', will update the property self.bestfit_errors, else will update the property self.bestfit_values

        """
        if dictparameters is None:
            self._bestfit = {}
            self._bestfit_values = {}
            self._bestfit_errors = {}
        else:
            self._bestfit = dictparameters
            self._bestfit_values = {k:v for k,v in dictparameters.items() if not k.endswith("_err")}
            self._bestfit_errors = {k:v for k,v in dictparameters.items() if k.endswith("_err")}
        
    def update_parameters(self, **kwargs):
        """ 
        Update parameters which describe the scene, such as BASE_PARAMETERS, PSF_PARAMETERS and GEOMETRY_PARAMETERS from self.scene.
        
        """
        for k,v in kwargs.items():
            # Change the baseline scene
            if k in self.scene.BASE_PARAMETERS:
                self._base_parameters[k] = v
                
            # Change the scene PSF
            elif k in self.scene.PSF_PARAMETERS:
                self._psf_parameters[k] = v
                
            # Change the scene geometry                
            elif k in self.scene.GEOMETRY_PARAMETERS:
                self._geometry_parameters[k] = v
                
            # or crash
            else:
                raise ValueError(f"Unknow input parameter {k}={v}")

    # ------- #
    # GETTER  #
    # ------- #
    def get_guesses(self, free_only=False, as_array=False, **kwargs):
        """ 
        Get guesses values for the availables parameters.

        Parameters
        ----------
        free_only: bool -optional-
            If True, return guesses only for free parameters (setted in self.free_parameters)\n
            Default is False.
        
        as_array: bool -optional-
            If True, return array of guesses values. Else return full dictionnary.\n
            Default is False.
        
        Returns
        -------
        Dict or Array
        """
        dict_guess = {**self.scene.guess_parameters(), **kwargs}
        if free_only:
            dict_guess = {k:dict_guess[k] for k in self.free_parameters}
            
        if as_array:
            return np.array(list(dict_guess.values()))
        
        return dict_guess
        
    def get_limits(self, a_limit=None, pos_limits=4, sigma_limit=[0,5]):
        """ 
        Get limits values (bounds) as list for free parameters.

        Parameters
        ----------
        a_limit: list of 2 floats/None -optional-
            Bounds for ellipticity parameters\n
            Default is None

        pos_limits: float
            Shift (in spx) from guess value for the target position parameters (x0 and y0).\n
            Default is 4.

        sigma_limit: list of 2 floats/None -optional-
            Bounds for the psf shape parameter (radius of the gaussian)\n
            Default is [0,5].
        
        Returns
        -------
        List
        """
        param_names = self.free_parameters
        param_guess = self.get_guesses(free_only=True, as_array=True)
        limits = [None for i in range(self.nfree_parameters)]
        
        if "xoff" in param_names:
            id_ = param_names.index("xoff")
            limits[id_] = [param_guess[id_]-pos_limits, param_guess[id_]+pos_limits]
            
        if "yoff" in param_names:
            id_ = param_names.index("yoff")
            limits[id_] = [param_guess[id_]-pos_limits, param_guess[id_]+pos_limits]
            
        if "a" in param_names:
            id_ = param_names.index("a")
            limits[id_] = a_limit

        if "sigma" in param_names:
            id_ = param_names.index("sigma")
            limits[id_] = sigma_limit
            
        return limits 
        
    def get_parameters(self, free_only=False):
        """ 
        Get all parameters.

        Parameters
        ----------
        free_only: bool -optional-
            If True, return only free parameters\n
            Default is False
        
        Returns
        -------
        Dictionary
        """
        all_params = { **self._base_parameters,
                       **self._psf_parameters,
                       **self._geometry_parameters}
        if free_only:
            return {k:all_params[k] for k in self.free_parameters}
        
        return all_params
    
    def get_model(self, parameters=None):
        """ 
        Get scene model according to the setted parameters.

        Parameters
        ----------
        parameters: dict -optional-
            Allow to update the free parameters currently load in self.\n
            If None (Default), will use parameters in self.free_parameters.

        Returns
        -------
        Array of model flux.

        """
        if parameters is not None:
            self.set_freeparameters(parameters)
    
        return self.scene.get_model(**self._base_parameters,
                                    overlayparam = self._geometry_parameters, 
                                    psfparam = self._psf_parameters)

    def get_bestfit_parameters(self, incl_err=True, as_dataframe=True):
        """ 
        Get bestfit parameters in self.bestfit.

        Parameters
        ----------
        incl_err: bool -optional-
            If True (Default), include errors on the best fit parameters.

        as_dataframe: bool -optional-
            If True (Default), return Dataframe.

        Returns
        -------
        Dict or Dataframe
        """
        if not self.has_bestfit():
            raise AttributeError("No bestfit set. see set_bestfit() or fit()")
        
        if as_dataframe:
            df =  pandas.DataFrame({"values":self._bestfit_values,
                                    "errors":{k.replace("_err",""):v for k,v in self._bestfit_errors.items()}
                                     })
            return df if incl_err else df["values"]
        
        return self._bestfit if incl_err else self._bestfit_values
            
    # ------------ #
    #  FITTING     #
    # ------------ #
    # - chi2 = -2log(Likelihood)
    def get_chi2(self, parameters=None, leastsq=False):
        """ 
        Get Chi square value.

        Parameters
        ----------
        parameters: dict -optional-
             If None (Default) will compute model with current load parameters in self.free_parameters.\n
             Otherwise you can provide new parameters.
        
        leastsq: bool -optional-
             If True, don't consider errors (Least Square).\n
             Default is False.

        Returns
        -------
        Float

        """
        model = self.get_model(parameters).values
        if leastsq:
            return np.nansum( (self.scene.flux_comp - model)**2 )
        
        return np.nansum( (self.scene.flux_comp - model)**2/self.scene.variance_comp )
    
    # - priors = -2log(prod_of_priors)
    def get_prior(self):
        """ Return -2log(prod_of_priors). Only if you've setted a Priors() object. See priors.get_product() """
        self.priors.set_parameters( self.get_parameters(free_only=True) )
        return self.priors.get_product()

    # - prob = Likelihood*prod_of_priors
    #   logprob =-2log(prob) = -2log(Likelihood) + -2log(prod_of_priors) 
    def get_logprob(self, parameters=None, bound_value=1e13, leastsq=False):
        """ 
        Get logprob value ( = -2log(Likelihood) + -2log(prod_of_priors) ).

        Parameters
        ----------
        parameters: dict -optional-
             If None (Default) will compute model with current load parameters in self.free_parameters.\n
             Otherwise you can provide new parameters.
        
        bound_values: float -optional-
             Used to avoid Nan in log(prod_of_priors) if prior=0. (Must be high)\n
             Default is 1e13
        
        leastsq: bool -optional-
             If True, don't consider errors (Least Square).\n
             Default is False.
        
        Returns
        -------
        Float

        """
        if parameters is not None:
            self.set_freeparameters(parameters)
            
        prior = self.get_prior()
        if prior == 0: # this way, avoid the NaN inside get_chi2()
            if self._debug:
                print(f"prior=0, returning {bound_value}")
            return bound_value

        chi2 = self.get_chi2(leastsq=leastsq)
        if self._debug:
            print(f"chi2 = {chi2} (dof={self.dof} | chi2_dof={chi2/self.dof})")
        
        return chi2 - 2*np.log(prior)

    # - fitting over logprob or chi2, see use_priors
    def fit(self, guess=None, limit=None, verbose=False, error=None,
                use_priors=True, runmigrad=True, errordef=0.5, **kwargs):
        """ 
        Fitter.

        Parameters
        ----------
        guess: dict -optional-
            Guess values for the free_parameters.\n
            If None (Default), automatically estimates them (see self.scene.guess_parameters() )

        limit: dict -optional-
            Bounds values for the free_parameters.\n
            If None (Default), use default values in self.get_limits.

        verbose: bool -optional-
            Print parameters name, guess and bounds before the fit.\n
            Default is False

        error: array -optional-
            Access parameter parabolic errors via an array-like view (see Minuit.error)\n
            Defaut is None.

        use_prior: bool -optional-
            If True (Default), will use setted priors and therefore maximize likelihood instead of minimize usual Chi square.

        runmigrad: bool -optional-
            If True (Default), run Minuit.migrad(). If False return Minuit.from_array_func() object without runninf the fit

        errordef: float -optional-
            errordef=1 for least-squares score function \n
            errordef=0.5 for maximum-likelihood score function (Default)

        kwargs:
            Goes to Minuit.from_array_func()

        Returns
        -------
        Minuit.from_array_func() if not runmigrad
        Minuit.from_array_func().migrad() if runmigrad
        """
        if guess is None: guess = {}
        if limit is None: limit = {}
        guess = self.get_guesses(free_only=True, as_array=True, **guess)
        limit = self.get_limits(**limit)
        if verbose or self._debug:
            print(f"param names {self.free_parameters}")
            print(f"guess {guess}")
            print(f"limits {limit}")

            
        m = Minuit.from_array_func(self.get_logprob if use_priors else self.get_chi2, guess, limit=limit,
                                    name= self.free_parameters, error=error,errordef=errordef,
                                   **kwargs)
        if not runmigrad:
            self.set_bestfit(None)
            return m

        migradout = m.migrad()
        if not migradout[0].is_valid:
            warnings.warn("migrad() is not valid.")
            
        self.set_bestfit({**dict(m.values),**{k+"_err":v for k,v in m.errors.items()}})
        # setup the scene at the best values
        self.scene.update(**dict(m.values))
        return migradout
        
    # ============== #
    #  Parameters    #
    # ============== #
    @property
    def scene(self):
        """ 
        Scene object (see self.scene)
        """
        return self._scene
    
    @property
    def free_parameters(self):
        """ 
        Free parameters.
        """
        return self._freeparams
    
    @property
    def nfree_parameters(self):
        """ 
        Number of free parameters.
        """
        return len(self.free_parameters)

    @property
    def dof(self):
        """ 
        Degree of Freedom for the fit. (number of spaxel - self.nfree_parameters)
        """
        return len(self.scene.flux_comp) - self.nfree_parameters
    
    @property
    def fixed_parameters(self):
        """ 
        Fixed parameters
        """
        return self._fixedparams
    
    @property
    def priors(self):
        """ 
        Priors() object (see .Priors() ) 
        """
        return self._priors
    
    @property
    def PARAMETER_NAMES(self):
        """ 
        Name of all parameters name. (see self.scene.PARAMETER_NAMES)
        """
        return self.scene.PARAMETER_NAMES

    # ------- #
    #  Fit    #
    # ------- #
    @property
    def bestfit(self):
        """ 
        Best fit values (manually setted or computed with fit method)
        """
        if not hasattr(self,"_bestfit") or self._bestfit is None or len(self._bestfit)==0:
            return None
        return self._bestfit
    
    def has_bestfit(self):
        """ Test if a bestfit values dictionary has been set. """
        return self.bestfit is not None
        


