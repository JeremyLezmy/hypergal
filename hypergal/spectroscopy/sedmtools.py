#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ztfquery import sedm
from pysedm import byecr, astrometry
from shapely.geometry.point import Point

# color printing
from ..utils.tools import bcolors

def get_target_position(cube, warn=False, **kwargs):
    """ shortcut to pysedm.astrometry.position_source()
    returns the best guess target location in the ifu.
    """
    return astrometry.position_source(cube, warn=warn,  **kwargs)[0]

def remove_target_spx(cube, xy, radius=3, store=False, get_filename=False):
    
    """ Remove spaxels around the target position

    Parameters
    ----------
    cube: [pyifu.Cube]

    xy: [list, array 2D]
        Target Position (spx coordinate)

    radius: [float]
        radius of the circle centered on xy where you want to remove spaxels (spx coordinates)
        Default is 3.

    store: [bool]
        Do you want to store the new cube?
       
    get_filename: [bool]
        If True, return filename of the new cube
        If False, return the new cube object

    Returns
    -------
    New cube without spaxels
    """
    x, y = xy
    p = Point(x, y)
    circle = p.buffer(radius)
    idx = cube.get_spaxels_within_polygon( circle )
    cube_wotar =cube.get_partial_cube([i for i in cube.indexes if i not in idx],np.arange(len(cube.lbda)) )
    cube_wotar.set_filename( cube.filename.replace('e3d', 'e3dobjrm') )
    
    if store:
        cube_wotar.writeto( cube_wotar.filename )

    if get_filename:
        return cube_wotar.filename
        
    return cube_wotar

