""" PanStarrs related tools """

import pandas
import warnings
import numpy as np

from ..utils import downloads
from .basics import CutOut

from astrobject.instruments import panstarrs

def query_panstarrs_metadata(ra, dec, size=240, filters="grizy", type="stack"):
    
    """ Query ps1filenames.py service to get a list of images
    
    Parameters
    ----------
    ra, dec: [floats] 
        position in degrees
    size: [float] 
        image size in pixels (0.25 arcsec/pixel)
    filters: [strings]
        string with filters to include
        
    Returns
    --------
    Table (a table with the results)
    """
    

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&type={type}&format=fits"
           "&filters={filters}").format(**locals())
    d_ = pandas.read_csv(url, sep=" ")
    d_["basename"] = d_.pop("shortname")
    d_["baseurl"]  = d_.pop("filename")
    d_["project"]  = "ps1"
    d_["filters"] = d_["filter"]
    return d_
    
def _ps_pstourl_(dataframe, format="fits", size=240, type="stack", output_size=None):
    """ Get URL for images in the table
    
    Parameters
    ----------
    format: [string] 
        data format (options are "jpg", "png" or "fits")
                
    Returns
    -------
    String (a string with the URL)
    """
    if format not in ["jpg","png","fits"]:
        raise ValueError("format must be one of jpg, png, fits (%s given)"%format)
    
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&type={type}&format={format}").format(
               **{"ra":dataframe["ra"].values[0],"dec":dataframe["dec"].values[0],
                   "size":size, "type":type,"format":format}
          )
        
    if output_size:
        url = url + "&output_size={}".format(output_size)
        
    # sort filters from red to blue
    urlbase = url + "&red="
    return [urlbase+filename for filename in dataframe['baseurl'].values]

def get_ps_url(ra, dec, size=240, output_size=None, filters="grizy", type="stack", format="fits"):#, color=False):
    
    """ Get url for images in the table
    
    Parameters
    ----------
    ra, dec: [floats] 
        position in degrees

    size: [float] 
        image size in pixels (0.25 arcsec/pixel)

    filters: [strings]
        string with filters to include

    format: [string] 
        data format (options are "jpg", "png" or "fits")
        
    color: [bool]
        if True, creates a color image (only for jpg or png format).
        Default is return a list of URLs for single-filter grayscale images.
        
    Returns
    -------
    String (a string with the URL)
    """
    df = query_panstarrs_metadata(ra, dec, size=size, type=type, filters=filters)
    return _ps_pstourl_(df, output_size=output_size, size=size, type=type, format=format)
    
    


class PS1CutOuts( CutOut ):
  
    # ================ #
    #  StaticMethods   #
    # ================ #
    @staticmethod
    def download_cutouts(ra, dec, size=140, filters=None, client=None, ignore_warnings=True):
        """ Download Panstarrs cutouts
    
        Parameters
        ----------
        ra, dec: [floats] 
            position in degrees

        size: [float]  -optional-
            image size in pixels (0.25 arcsec/pixel)
            Default is 140

        filters: [strings]  -optional-
             string with filters to include
             if None, load all the filters (g,r,i,z,y)

        client: [dask Client]  -optional-
             Provide a dask client for using Dask multiprocessing.
             If so, a list of futures will be returned.
             
        
    Returns
    -------
    List of cutouts, (Panstarrs Instrument object, see astrobject.instrument)
    """
        if filters is None:
            filters=["g","r","i","z","y"]
            
        nfilters = len(filters)
        imgdata_url = get_ps_url(ra, dec, filters="".join(filters), size=size, type="stack")
        imgwt_url = get_ps_url(ra, dec, filters="".join(filters), size=size, type="stack.wt")

        images = downloads.download_urls(imgdata_url+imgwt_url, fileout=None, client=client)
        data, weights = np.asarray(images).reshape(2,nfilters)
        
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")
            return [panstarrs.PanSTARRS(data_, weightfilename=weights_, background=0)
                        for data_, weights_ in zip(data, weights)]

