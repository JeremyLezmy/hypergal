""" Module Containing the basic objects """
import warnings
import numpy as np

from ..photometry.astrometry import WCSHolder
from pyifu.spectroscopy import Cube

class WCSCube( Cube, WCSHolder ):
    """ """

    @classmethod
    def from_cutouts(cls, hgcutout, header_id=0, influx=True, binfactor=None, xy_center=None):
        """ """
        
        lbda = np.asarray(hgcutout.lbda)
        sort_lbda = np.argsort(lbda)

        # Data
        lbda = lbda[sort_lbda]
        data = hgcutout.get_data(influx=influx)[sort_lbda]
        variance = hgcutout.get_variance(influx=influx)[sort_lbda]
        spaxel_vertices = np.asarray([[0,0],[1,0],[1,1],[0,1]])-0.5 # centered
        if binfactor is not None:
            binfactor = int(binfactor)
            if binfactor==1:
                warnings.warn("binfactor=1, this means nothing to do.")
            else:
                from ..utils.array import restride
                data = np.sum(restride(data, (1, binfactor, binfactor)),axis=(-2,-1))
                variance = np.sum(restride(variance, (1, binfactor, binfactor)),axis=(-2,-1))
                spaxel_vertices *=binfactor
        else:
            binfactor=1
            
        # Header
        header = hgcutout.instdata[header_id].header

        # Mapping
        xsize, ysize = np.asarray(data[header_id].shape)
        pixels_ = np.mgrid[0:xsize*binfactor:binfactor,0:ysize*binfactor:binfactor]
        
        init_shape = np.shape(pixels_)
        spaxels_xy = pixels_.reshape(2, init_shape[1]*init_shape[2]).T
        if xy_center is not None:
            spaxels_xy = np.asarray(spaxels_xy, dtype="float")-xy_center
        spaxel_mapping = {i:v for i,v in enumerate(spaxels_xy)}

        #
        # Init
        return cls.from_data(data=np.concatenate(data.T, axis=0).T, 
                              variance=np.concatenate(variance.T, axis=0).T, 
                            lbda=lbda,header=header,
                            spaxel_vertices=spaxel_vertices,
                                spaxel_mapping=spaxel_mapping)
    
    # ================ #
    #   Methods        #
    # ================ #
    def set_header(self, header, *args, **kwargs):
        """ """
        _ = super().set_header(header)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.load_wcs(header)
        
