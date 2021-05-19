

import numpy as np
from pyifu.adr import ADR

IFU_SCALE = 0.558

class ADRFitter(ADR):

    def __init__(self, xpos, ypos, lbda,
                     xpos_err=None, ypos_err=None,
                     init_adr=None, **kwargs ):
        """
        Inherits from pyifu.adr.ADR() object. \n
        Fit airmass and parallactic angle using position of an object along wavelength.

        Parameters
        ----------
        xpos,ypos: array
            Postion of the object in the IFU

        xpos_err,ypos_err: array
            Error on xpos and ypos

        lbda: array
            Wavelength corresponding to the previous positions
        
        init_adr: pyifu.adr.ADR() -optional-
            A guess adr object, where the datas will be get back as init parameters
        
        kwargs: Argument
            Go to ADR.set()

        """
        self.set_xpos(xpos)
        self.set_ypos(ypos)
        self.set_xpos_err(xpos_err)
        self.set_ypos_err(ypos_err)
        self.set_lbda(lbda)
        
        if init_adr is not None:
            self.set(**init_adr.data)
            
        self._set_state("Initial")
            
        _ = super().__init__(**kwargs)
        
    @classmethod
    def from_centroids(cls, xpos, ypos, lbda, xpos_err=None, ypos_err=None, init_adr=None, **kwargs):
        """
        Inherits from pyifu.adr.ADR() object.\n 
        Fit airmass and parallactic angle using position of an object along wavelength.

        Parameters
        ----------
        xpos,ypos: array
            Postion of the object in the IFU

        xpos_err,ypos_err: array
            Error on xpos and ypos

        lbda: array
            Wavelength corresponding to the previous positions
        
        init_adr: pyifu.adr.ADR() -optional-
            A guess adr object, where the datas will be get back as init parameters
        
        kwargs: Argument
            Go to ADR.set()
       
        """

        return cls(xpos, ypos, lbda, xpos_err=xpos_err, ypos_err=xpos_err, init_adr=init_adr, **kwargs)

    def fit_adr(self, show=False, **kwargs):
        """
        Fitter. New params will be directly accessible through self.data.

        Parameters
        ----------
        show: bool
            If True, plot of ypos(xpos) with best adr fit.
        
        kwargs: Argument
            Go to ADR.set()
        
        """
        
        for k in self.PROPERTIES:
            if self.data[k] is None:
                raise ValueError(f'{k} must be set with self.set() method')

        from scipy import optimize
        xref_init, yref_init = self.guess_ref_pos()
        datas = np.array([self.xpos, self.ypos])

        if self.xpos_err is not None and self.ypos_err is not None:
            err = np.array([ self.xpos_err, self.ypos_err])
        else:
            err =np.ones((2,len(self.xpos)))

        def minifit(X):
            """ """
            self.set( parangle=X[0] )
            self.set( airmass=X[1] )
            xref=X[2]
            yref=X[3] 
        
            model = self.refract( xref, yref,self.lbda, unit = IFU_SCALE)
               
            return (np.sum((datas-model)**2 / err**2))

        adrfit=optimize.minimize(minifit, np.array([self.parangle, self.airmass, xref_init, yref_init]), bounds=[ (None,None), (1,None), (None,None), (None,None)] )

        if adrfit.success:
            self._set_state("Success Fit")
        if not adrfit.success:
            self._set_state("Reject Fit")
        
        self._fit_airmass = adrfit.x[1]
        self._fit_parangle = adrfit.x[0]
        self._fit_xref = adrfit.x[2]
        self._fit_yref = adrfit.x[3]
        
        if show:
            self.show(**kwargs)
        
    
    def show(self, ax=None, savefile=None):
        """
        Show position and current loaded adr (which is the fitted one if you've run self.fit_adr() ) .

        Parameters
        ----------
        ax: Matplotlib.Axes -optional-
            You can provide your own Axe (one)

        savefile: string
            If not None, fig.figsave(savefile)\n
            Default is None.
        
        Returns
        -------
        Axes
        """
        import matplotlib.pyplot as plt
        
        if ax==None:
            fig,ax=plt.subplots( figsize=(8,8))
        else:
            fig = ax.figure
        
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        
        colormap = cm.jet
        normalize = mcolors.Normalize(vmin=np.min(self.lbda), vmax=np.max(self.lbda))
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        colors = plt.cm.jet((self.lbda-np.min(self.lbda))/(np.max(self.lbda)-np.min(self.lbda)))
        
        ax.scatter( self.xpos, self.ypos, cmap=colormap, c=self.lbda, label='Input position')
        ax.errorbar(self.xpos, self.ypos, self.xpos_err, self.ypos_err, fmt='none', color=colors)
          
        adrfit=ax.scatter(self.refract(self._fit_xref, self._fit_yref, self.lbda , unit = IFU_SCALE)[0], self.refract(self._fit_xref, self._fit_yref, self.lbda , unit = IFU_SCALE)[1], marker='o',cmap=colormap, c=self.lbda, fc='none',edgecolors='k', label='Fitted ADR')
   
        from matplotlib.lines import Line2D
        Line2D([0], [0], marker='o',linestyle='',  markersize=8, fillstyle=Line2D.fillStyles[-1],label=r'Theoretical ADR ')
        Line2D([0], [0],marker='o',linestyle='',markeredgecolor='k', markerfacecolor='k',  markersize=8, fillstyle=Line2D.fillStyles[-1],label=r'Fitted position ')
   
        ax.legend()        
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel(r'x(spx)')
        ax.set_ylabel(r'y(spx)')
        fig.colorbar(s_map, label=r'$\lambda$', ax=ax, use_gridspec=True)
        fig.suptitle(fr'$x_{{ref}}= {np.round(self._fit_xref,2)},y_{{ref}}= {np.round(self._fit_yref,2)}, \lambda_{{ref}}= {self.lbdaref}\AA  $' + '\n' +
                     fr'$Airmass= {np.round( self._fit_airmass,2)},Parangle= {np.round( self._fit_parangle,2)}  $')
       
        ax.set_aspect('equal',adjustable='datalim')       
        if savefile != None:            
            fig.savefig( savefile )
        return ax
        

    def guess_ref_pos(self):
        """ 
        Guess reference x and y positions for the fit, according to self.lbdaref\n
        Just select the xpos and ypos of the closest wavelength of self.lbdaref.
        """
        if self.lbdaref is not None:
            idx = (np.abs(self.lbda-self.lbdaref)).argmin()
            xref_init = self.xpos[idx]
            yref_init = self.ypos[idx]
        return xref_init, yref_init
        
    # --------- #
    #  SETTER   #
    # --------- #
    
    def set_xpos(self, xpos):
        """
        Set array of x position in function of wavelength.
        """
        self._xpos = xpos

    def set_ypos(self, ypos):
        """
        Set array of y position in function of wavelength.
        """
        self._ypos = ypos
        
    def set_xpos_err(self, xpos_err):
        """
        Set array of error on x position.
        """
        self._xpos_err = xpos_err

    def set_ypos_err(self, ypos_err):
        """
        Set array of error on y position.
        """
        self._ypos_err = ypos_err

    def set_lbda(self, lbda):
        """
        Set array of wavelenght for each position.
        """
        self._lbda = lbda

    def _set_state(self, state):
        """
        Set the state of ADR object: Initial of Fitted
        """
        self._state = state

    @property
    def xpos(self):
        """
        Array of x position in function of wavelength.
        """
        return self._xpos
    
    @property
    def ypos(self):
        """
        Array of y position in function of wavelength.
        """
        return self._ypos
        
    @property
    def xpos_err(self):
        """
        Array of error on self.xpos .
        """
        return self._xpos_err
        
    @property
    def ypos_err(self):
        """
        Array of error on self.ypos .
        """
        return self._ypos_err
        
    @property
    def lbda(self):
        """
        Array of wavelenght for each position
        """
        return self._lbda

    @property
    def fitted_airmass(self):
        """
        Fitted airmass.
        """
        if not hasattr(self, '_fit_airmass'):
            return None
        return self._fit_airmass

    @property
    def fitted_parangle(self):
        """
        Fitted parangle.
        """
        if not hasattr(self, '_fit_parangle'):
            return None
        return self._fit_parangle

    @property
    def fitted_xref(self):
        """
        Fitted xref (according to lbdaref).
        """
        if not hasattr(self, '_fit_xref'):
            return None
        return self._fit_xref

    @property
    def fitted_yref(self):
        """
        Fitted yref (according to lbdaref).
        """
        if not hasattr(self, '_fit_yref'):
            return None
        return self._fit_yref

    @property
    def state(self):
        """
        State of the ADR object instance. \n
        Might be Initial, Succes fit or Bad fit.
        """
        return self._state
