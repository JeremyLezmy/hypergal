
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
    """ 
    Compute the vmin-th and the vmax-th percentile of the data.
    Parameters
    ----------
    data: [array]      
    vmin: [string]
    vmax: [string]
    Return
    ----------
    2 floats, vmin-th and vmax-th percentile of the given datas.
    """
    if vmax is None: vmax="99"
    if vmin is None: vmin = "1"
                
    if type(vmax) == str:
        vmax=np.nanpercentile(data, float(vmax))
        
    if type(vmin) == str:
        vmin=np.nanpercentile(data, float(vmin))
        
    return vmin, vmax
        
