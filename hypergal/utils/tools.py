
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'    

def parse_vmin_vmax(data, vmin, vmax):
    """ parse the input vmin vmax given the data.
    
    if float or int given, this does nothing, 
    if string given, this computes the corresponding percentile of the data.

    e.g. 

    vmin_, vmax_ = parse_vmin_vmax(data, 40, '90')
    -> the input vmin is not a string, so it is returned as such
    -> the inout vmax is a string, so the returned vmax_ corresponds to the 
       90-th percent value of data.

    Parameters
    ----------
    data: [array]      
        data (float array)

    vmin, vmax: [string or float/int]
        - if string, the corresponding percentile is computed
        otherwise, nothing happends.

    Return
    ------
    float, float
    """
    if vmax is None: vmax="99"
    if vmin is None: vmin = "1"
                
    if type(vmax) == str:
        vmax=np.nanpercentile(data, float(vmax))
        
    if type(vmin) == str:
        vmin=np.nanpercentile(data, float(vmin))
        
    return vmin, vmax
        
