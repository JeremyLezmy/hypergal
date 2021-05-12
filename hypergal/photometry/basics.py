""" Basic Photometry tools """

import pandas
import warnings
import numpy as np
import pandas as pd

from .astrometry import WCSHolder


class CutOut( WCSHolder ):

    def __init__(self, instdata=None, radec=None):
        """ 
        Set ra, dec and instrument data

        Parameters
        ----------
        ra, dec: [floats]  -optional-
            position in degrees

        instdata: [list of Instrument object]  -optional-
        
        Returns
        --------
        """
        self.set_instdata(instdata=instdata)
        if radec is not None:
            self.set_radec(*radec)

    # ================ #
    #  Initialisation  #
    # ================ #
    @classmethod
    def from_radec(cls, ra, dec, load_cutout=True, size=140, client=None,  filters=None):
        """ 
        load cutout from ra, dec datas

        Parameters
        ----------
        ra, dec: [floats] 
            position in degrees

        size: [float] 
            image size in pixels
            Default is 140

        filters: [strings]  -optional-
            string with filters to include
            if None, load all the filters

        client: [dask Client]  -optional-
             Provide a dask client for using Dask multiprocessing.
             If so, a list of futures will be returned.
        
        Returns
        --------
        List of cutout
        """
        
        if load_cutout:
            instdata = cls.download_cutouts(ra, dec, size=size, filters=filters, client=client)
        else:
            instdata = None
            
        return cls(instdata=instdata, radec=[ra, dec])

    @classmethod
    def from_sedmfile(cls, filename, load_cutout=True, size=140, client=None,  filters=["g","r","i","z","y"]):
        """ 
        load cutout from SEDM file, according to the ra, dec information in the header

        Parameters
        ----------
        filename: [string] 
            path of the sedm object

        size: [float] 
            image size in pixels
            Default is 140

        filters: [strings]  -optional-
            string with filters to include
            if None, load all the filters

        client: [dask Client]  -optional-
             Provide a dask client for using Dask multiprocessing.
             If so, a list of futures will be returned.
        
        Returns
        --------
        List of cutout
        """
        from astropy.io import fits
        from astropy import coordinates, units
        
        header = fits.getheader(filename)
        coords = coordinates.SkyCoord(header.get("OBJRA"), header.get("OBJDEC"),
                                    frame='icrs', unit=(units.hourangle, units.deg))
        ra, dec = coords.ra.deg, coords.dec.deg
        # - 
        return cls.from_radec(ra, dec, load_cutout=load_cutout, size=size, client=client, filters=filters)

    # ================ #
    #  StaticMethods   #
    # ================ #
    @staticmethod
    def download_cutouts(ra, dec, size=140, filters=None, client=None, ignore_warnings=True):
        """ """
        raise NotImplementedError(" Object inheriting from CutOut must implemente download_cutouts")
    
    # ================ #
    #   Methods        #
    # ================ #
    def to_cube(self, header_id=0, influx=True, binfactor=None, xy_center=None, **kwargs):
        """ 
        Transform CutOuts object into 3d Cube, according to the wavelength of each image.

        Parameters
        ----------
        xy_center: [optional] -optional-
            center coordinates (in pixel) or the returned cube.
            - if None: this is ignored
            - if string: 'target', this will convert the self.ra and self.dec into xy_center and set it
            - else; used as centroid

        influx: [bool]  -optional-
            If True, return data in flux in erg.s-1.cm-2.AA-1
            If False, return data in counts.
        
        binfactor: [int] -optional-
            Apply a binning [binfactor x binfactor] on the images

        Returns
        --------
        WCSCube object
        """
        if xy_center is not None and xy_center=='target':
            xy_center = self.radec_to_xy(self.ra, self.dec).flatten()
            
        from ..spectroscopy import WCSCube
        return WCSCube.from_cutouts(self, header_id=header_id, influx=influx,
                                        binfactor=binfactor, xy_center=xy_center, **kwargs)

    
    def to_dataframe( self, which=['data','err'], filters=None, influx=True):
        """
        Get Panda DataFrame from Cutouts
        Parameters
        ----------
        which: [string/list of string]
            What do you want in your dataframe
            Might be 'data', 'err', 'var'
            Default is ['data', 'err']

        filters: [string/list of string]
            For which filter(s) do you want [which]. 
            If None, '*', 'all', consider all availbales filters
            Default is None

        influx: [bool]
            Do you want [which] in flux (erg/s/cm2/AA) or in counts
            Default is True

        Return
        ----------
        Pandas DataFrame
        """
        df = pd.DataFrame()
        which = which.split() if type(which)==str else which
        filters = filters.split() if type(filters)==str else filters
        if which is None or which in ['*', 'all']:
            which = ['data', 'err']
        if filters is None or filters in ['*', 'all']:
            filters = self.filters

        for w in which:
            to_add = self._get_which_(w, filters, influx)
            to_add = to_add.reshape( (to_add.shape[0], to_add.shape[-1]*to_add.shape[-2]) )

            if w=='data':
                df = df.assign(**dict(zip(filters , to_add)))
            else:
                df = df.assign(**dict(zip([s + '_'+ w for s in filters] , to_add)))

        return df
            
    # -------- #
    #  SETTER  #
    # -------- #
    def set_instdata(self, instdata, load_wcs=True):
        """ 
        Set Instrument Data
 
        instdata: [list of Instrument object] 

        load_wcs: [bool]
             Do you want to load wcs information from Instrument header?
            
        """
        if instdata is None:
            return
        
        self._instdata = np.atleast_1d(instdata)
        if self.ninst>0 and load_wcs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.load_wcs(self.instdata[0].header)
                
    def set_radec(self, ra, dec):
        """ 
        Set Right Ascension and Declination 
        ra, dec: [floats] 
            position in degrees
        """
        self._ra, self._dec = ra, dec

    # -------- #
    #  GETTER  #
    # -------- #
    def get_index(self, filters = None):
        """ 
        Get cutout index from filter name information.

        Parameters
        ----------
        filters: [String or list of strings]
            Name of the filter(s) (for instance 'ps1.r' or ['ps1.g', 'ps1.i']
            If filters is None or in ['*, 'all'], return all the indices.

        Returns
        -------- 
        list of index
        
        """
        if filters is None or filters in ["*","all"]:
            return np.arange(self.ninst)
        
        return [self.filters.index(f_) for f_ in np.atleast_1d(filters)]
        
    def get_data(self, filters=None, influx=False):
        """ 
        Get datas from cutouts 

        Parameters
        ----------
        filters : [String or list of strings]
            Name of the filter(s) (for instance 'ps1.r' or ['ps1.g', 'ps1.i']
            If filters is None or in ['*, 'all'], consider all the available filters.

        influx : [bool] -optional-
            If True, return datas in flux in erg.s-1.cm-2.AA-1
            If False, return datas in counts.
           
        Returns
        -------- 
        List of array of size [image_size]
        """
        return self._get_which_("data", filters=filters, influx=influx)
    
    def get_error(self, filters=None, influx=False):
        """ 
        Get errors from cutouts 

        Parameters
        ----------
        filters : [String or list of strings]
            Name of the filter(s) (for instance 'ps1.r' or ['ps1.g', 'ps1.i']
            If filters is None or in ['*, 'all'], consider all the available filters.

        influx : [bool] -optional-
            If True, return errors in flux in erg.s-1.cm-2.AA-1
            If False, return errors in counts.
           
        Returns
        -------- 
        List of array of size [image_size]
        """
        return self._get_which_("error", filters=filters, influx=influx)
    
    def get_variance(self, filters=None, influx=False):
        """ 
        Get variances from cutouts 

        Parameters
        ----------
        filters : [String or list of strings]
            Name of the filter(s) (for instance 'ps1.r' or ['ps1.g', 'ps1.i']
            If filters is None or in ['*, 'all'], consider all the available filters.

        influx : [bool] -optional-
            If True, return variances in flux in erg.s-1.cm-2.AA-1
            If False, return variances in counts.
           
        Returns
        -------- 
        List of array of size [image_size]
        """
        return self._get_which_("variance", filters=filters, influx=influx)

    def _get_which_(self, which, filters=None, influx=False):
        """ 
        Get [which] from cutouts 

        Parameters
        ----------
        
        which: [String]
            Might be "data", "var"/"variance", "err"/"error"

        filters : [String or list of strings]
            Name of the filter(s) (for instance 'ps1.r' or ['ps1.g', 'ps1.i']
            If filters is None or in ['*, 'all'], consider all the available filters.

        influx : [bool] -optional-
            If True, return [which] in flux in erg.s-1.cm-2.AA-1
            If False, return [which] in counts.
           
        Returns
        -------- 
        List of array of size [image_size]
        """
        
        if which == "data":
            data = self.data.copy()
            coef = 1 if not influx else self.flux_per_count[:,None,None]
        elif which in ["var", "variance"]:
            data = self.variance.copy()
            coef = 1 if not influx else self.flux_per_count[:,None,None]**2
        elif which in ["err", "error"]:
            data = np.sqrt(self.variance.copy())
            coef = 1 if not influx else self.flux_per_count[:,None,None]
        else:
            raise ValueError(f"get_which only implemented for 'data', 'variance' or 'error', {which} given")

        # data in flux or counts        
        data *= coef
        # returns
        if filters is None or filters in ["*","all"]:
            return data
        
        return data[self.get_index(filters)]

    # -------- #
    #  Apply   #
    # -------- #
    # ================ #
    #  Internal        #
    # ================ #
    def _call_down_(self, what, isfunc, index=None, *args, **kwargs):
        """ 
        Get [what] attribut from Instrument object: 

        Parameters
        ----------
        
        what: [String]
            Which attribut do you want to get from the intrument object (for instance 'lbda', 'bandname', 'mab0' ...)

        isfun : [bool]
            Does the atribut you want is callable

        index : [int or list of int] -optional-
            Index of the Instrument object to consider (see self.get_index)
        
        *args, **kwargs go to callable attribut if isfunc is True   

        Returns
        -------- 
        Asked attribut for the Instrument object
        """
        if index is None:
            instrus = self.instdata
        else:
            instrus = self.instdata[index]
            
        if not isfunc:
            return [getattr(s_, what) for s_ in instrus]
        return [getattr(s_,what)(*args, **kwargs) for s_ in instrus]

    def _map_down_(self, method, onwhat, index=None, *args, **kwargs):
        """ call inst_.{method}( inst_.{onwhat}, *args, **kwargs) looping over the instruments """
        if index is None:
            instrus = self.instdata
        else:
            instrus = self.instdata[index]

        return [getattr(s_,method)( getattr(s_, onwhat), *args, **kwargs) for s_ in instrus]
    
    # ================ #
    #  Properties      #
    # ================ #
    @property
    def instdata(self):
        """ 
        (list of) Instrument object
        """
        return self._instdata

    @property
    def ninst(self):
        """ 
        Number of available Instrument object
        """
        return len(self.instdata)
    #
    # calldown
    @property
    def data(self):
        """ 
        Data available (in counts) for all availables Instrument object
        """
        return self._call_down_("data", isfunc=False)
    
    @property
    def variance(self):
        """ 
        Variance available (in counts) for all availables Instrument object
        """
        return self._call_down_("var", isfunc=False)
    
    @property
    def headers(self):
        """ 
        Return DatFrame of the headers for all availables Instrument object
        """
        return pandas.DataFrame([dict(h_) for h_ in self._call_down_("header", isfunc=False)])
    
    @property
    def filters(self):
        """ 
        List of available filters 
        """
        return self._call_down_("bandname", isfunc=False)

    @property
    def lbda(self):
        """ 
        List of images wavelength
        """
        return self._call_down_("lbda", isfunc=False)

    @property
    def mab0(self):
        """ 
        AB zero point for all the images
        """
        return self._call_down_("mab0", isfunc=False)

    @property
    def flux_per_count(self):
        """ 
        Conversion factor from 1 count to flux in erg.s-1.cm-2.AA-1
        """
        mab0 = np.asarray(self.mab0)
        lbda = np.asarray(self.lbda)
        return 10**(-(2.406+mab0) / 2.5 ) / (lbda**2)
    #
    # Coordinates
    @property
    def ra(self):
        """ 
        Right ascension in degrees
        """
        return self._ra

    @property
    def dec(self):
        """ 
        Declination in degrees
        """
        return self._dec
