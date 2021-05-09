""" Basic Photometry tools """

import pandas
import warnings
import numpy as np
from astrobject.instruments import panstarrs


from .astrometry import WCSHolder


class CutOut( WCSHolder ):

    def __init__(self, instdata=None, radec=None):
        """ """
        self.set_instdata(instdata=instdata)
        self.set_radec(*radec)

    # ================ #
    #  Initialisation  #
    # ================ #
    @classmethod
    def from_radec(cls, ra, dec, load_cutout=True, size=140, client=None,  filters=None):
        """ """
        if load_cutout:
            instdata = cls.download_cutouts(ra, dec, size=size, filters=filters, client=client)
        else:
            instdata = None
            
        return cls(instdata=instdata, radec=[ra, dec])

    @classmethod
    def from_sedmfile(cls, filename, load_cutout=True, size=140, client=None,  filters=["g","r","i","z","y"]):
        """ """
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
        xy_center: [optional] -optional-
            center coordinates (in pixel) or the returned cube.
            - if None: this is ignored
            - if string: 'target', this will convert the self.ra and self.dec into xy_center and set it
            - else; used as centroid
        """
        if xy_center is not None and xy_center=='target':
            xy_center = self.radec_to_xy(self.ra, self.dec).flatten()
            
        from ..spectroscopy import WCSCube
        return WCSCube.from_cutouts(self, header_id=header_id, influx=influx,
                                        binfactor=binfactor, xy_center=xy_center, **kwargs)
        
    # -------- #
    #  SETTER  #
    # -------- #
    def set_instdata(self, instdata, load_wcs=True):
        """ """
        if instdata is None:
            return
        
        self._instdata = np.atleast_1d(instdata)
        if self.ninst>0 and load_wcs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.load_wcs(self.instdata[0].header)
                
    def set_radec(self, ra, dec):
        """ """
        self._ra, self._dec = ra, dec

    # -------- #
    #  GETTER  #
    # -------- #
    def get_index(self, filters):
        """ """
        if filters is None or filters in ["*","all"]:
            return np.arange(self.ninst)
        
        return [self.filters.index(f_) for f_ in np.atleast_1d(filters)]
        
    def get_data(self, filters=None, influx=False):
        """ """
        return self._get_which_("data", filters=filters, influx=influx)
    
    def get_error(self, filters=None, influx=False):
        """ """
        return self._get_which_("error", filters=filters, influx=influx)
    
    def get_variance(self, filters=None, influx=False):
        """ """
        return self._get_which_("variance", filters=filters, influx=influx)

    def _get_which_(self, which, filters=None, influx=False):
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
        """ """
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
        """ """
        return self._instdata

    @property
    def ninst(self):
        """ """
        return len(self.instdata)
    #
    # calldown
    @property
    def data(self):
        """ """
        return self._call_down_("data", isfunc=False)
    
    @property
    def variance(self):
        """ """
        return self._call_down_("var", isfunc=False)
    
    @property
    def headers(self):
        """ """
        return pandas.DataFrame([dict(h_) for h_ in self._call_down_("header", isfunc=False)])
    
    @property
    def filters(self):
        """ """
        return self._call_down_("bandname", isfunc=False)

    @property
    def lbda(self):
        """ """
        return self._call_down_("lbda", isfunc=False)

    @property
    def mab0(self):
        """ """
        return self._call_down_("mab0", isfunc=False)

    @property
    def flux_per_count(self):
        """ """
        mab0 = np.asarray(self.mab0)
        lbda = np.asarray(self.lbda)
        return 10**(-(2.406+mab0) / 2.5 ) / (lbda**2)
    #
    # Coordinates
    @property
    def ra(self):
        """ """
        return self._ra

    @property
    def dec(self):
        """ """
        return self._dec
