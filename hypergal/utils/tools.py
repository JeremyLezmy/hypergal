
import numpy as np
def parse_vmin_vmax(data, vmin, vmax):
    """ """
    if vmax is None: vmax="99"
    if vmin is None: vmin = "1"
                
    if type(vmax) == str:
        vmax=np.nanpercentile(data, float(vmax))
    if type(vmin) == str:
        vmin=np.nanpercentile(data, float(vmin))
        
    return vmin, vmax
        
