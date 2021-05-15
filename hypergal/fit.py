
import numpy as np
from scipy import stats
from iminuit import Minuit


# ===================== #
#                       #
#      PRIORS           #
#                       #
# ===================== #
class Priors( object ):
    """ """
    
    BOUND_VALUE = 1e-12
    def set_parameters(self, parameters):
        """ """
        self._parameters = parameters
        self._parameter_names = list(self.parameters.keys())
        
    # ============== #
    #  Methods       #
    # ============== #
    def get_product(self):
        """ """
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
        """ """
        return "a" in self.parameter_names and \
               "b" in self.parameter_names
    
    # ============== #
    #  Statics       #
    # ============== #
    
    @staticmethod
    def get_ellipticity_prior(a, b, max_ratio=0.9, q_troncnorm={"loc":0, "scale":0.15, "a":0, "b":4 }):
        """ """
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
        """ """
        return self._parameters
    
    @property
    def parameter_names(self):
        """ """
        return self._parameter_names 
    
                    

# ===================== #
#                       #
#      FITTER           #
#                       #
# ===================== #

class SceneFitter( object ):
    def __init__(self, scene, fix_params=["scale","rotation"], priors=None, debug=False):
        """ """
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
        """ """
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
        """ """
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
        """ """
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
        """ """
        self._scene = scene
        self._base_parameters = {k:None for k in self.scene.BASE_PARAMETERS}
        self._psf_parameters = {k:None for k in self.scene.PSF_PARAMETERS}
        self._geometry_parameters = {k:None for k in self.scene.GEOMETRY_PARAMETERS}
        
    def set_fixed_params(self, list_of_params):
        """ """
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
        """ """
        if len(parameters) != self.nfree_parameters:
            raise ValueError(f"you must provide {self.nfree_parameters} parameters, you gave {len(parameters)}")
            
        dfreeparam = {k:v for k, v in zip(self.free_parameters, parameters)}
        if self._debug:
            print(f"setting: {dfreeparam}")
        return self.update_parameters(**dfreeparam)

    def set_priors(self, priors):
        """ """
        self._priors = priors
            
    def set_bestfit(self, dictparameters):
        """ """
        if dictparameters is None:
            self._bestfit = {}
            self._bestfit_values = {}
            self._bestfit_errors = {}
        else:
            self._bestfit = dictparameters
            self._bestfit_values = {k:v for k,v in dictparameters.items() if not k.endswith("_err")}
            self._bestfit_errors = {k:v for k,v in dictparameters.items() if k.endswith("_err")}
        
    def update_parameters(self, **kwargs):
        """ """
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
        """ """
        dict_guess = {**self.scene.guess_parameters(), **kwargs}
        if free_only:
            dict_guess = {k:dict_guess[k] for k in self.free_parameters}
            
        if as_array:
            return list(dict_guess.values())
        
        return dict_guess
        
    def get_limits(self, a_limit=None, pos_limits=4, sigma_limit=[0,5]):
        """ """
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
        """ """
        all_params = {**self._base_parameters,
                    **self._psf_parameters,
                    **self._geometry_parameters}
        if free_only:
            return {k:all_params[k] for k in self.free_parameters}
        return all_params
    
    def get_model(self, parameters=None):
        """ """
        if parameters is not None:
            self.set_freeparameters(parameters)
    
        return self.scene.get_model(**self._base_parameters,
                                    overlayparam = self._geometry_parameters, 
                                    psfparam = self._psf_parameters)

    def get_bestfit_parameters(self, incl_err=True, as_dataframe=True):
        """ """
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
        """ """
        model = self.get_model(parameters).values
        if leastsq:
            return np.nansum( (self.scene.flux_comp - model)**2 )
        
        return np.nansum( (self.scene.flux_comp - model)**2/self.scene.variance_comp )
    
    # - priors = -2log(prod_of_priors)
    def get_prior(self):
        """ priors.get_product() """
        self.priors.set_parameters( self.get_parameters(free_only=True) )
        return self.priors.get_product()

    # - prob = Likelihood*prod_of_priors
    #   logprob =-2log(prob) = -2log(Likelihood) + -2log(prod_of_priors) 
    def get_logprob(self, parameters=None, bound_value=1e13, leastsq=False):
        """ """
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
        """ """
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
        """ """
        return self._scene
    
    @property
    def free_parameters(self):
        """ """
        return self._freeparams
    
    @property
    def nfree_parameters(self):
        """ """
        return len(self.free_parameters)

    @property
    def dof(self):
        """ """
        return len(self.scene.flux_comp) - self.nfree_parameters
    
    @property
    def fixed_parameters(self):
        """ """
        return self._fixedparams
    
    @property
    def priors(self):
        """ """
        return self._priors
    
    @property
    def PARAMETER_NAMES(self):
        """ """
        return self.scene.PARAMETER_NAMES

    # ------- #
    #  Fit    #
    # ------- #
    @property
    def bestfit(self):
        """ """
        if not hasattr(self,"_bestfit") or self._bestfit is None or len(self._bestfit)==0:
            return None
        return self._bestfit
    
    def has_bestfit(self):
        """ Test if a bestfit values dictionary has been set. """
        return self.bestfit is not None
        
