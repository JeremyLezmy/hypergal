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
    """ Parse the input vmin vmax given the data.\n
    If float or int given, this does nothing \n
    If string given, this computes the corresponding percentile of the data.\n
    e.g. parse_vmin_vmax(data, 40, '90')\n
    -> the input vmin is not a string, so it is returned as such\n
    -> the input vmax is a string, so the returned vmax corresponds to the
       90-th percent value of data.

    Parameters
    ----------
    data: array
        data (float array)

    vmin,vmax: string or float/int
        If string, the corresponding percentile is computed\n
        Otherwise, nothing happends.

    Returns
    -------
    float, float
    """
    if vmax is None: vmax = "99"
    if vmin is None: vmin = "1"

    if type(vmax) == str:
        vmax = np.nanpercentile(data, float(vmax))

    if type(vmin) == str:
        vmin = np.nanpercentile(data, float(vmin))

    return vmin, vmax
