#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          geometry_tool.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: rlezmy $
# Created on:        $Date: 2021/01/18 18:43:04 $
# Modified on:       2021/01/25 17:50:40
# Copyright:         2019, Jeremy Lezmy
# $Id: geometry_tool.py, 2021/01/18 18:43:04  JL $
################################################################################

"""
.. _geometry_tool.py:

geometry_tool.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/18 18:43:04'
__adv__ = 'geometry_tool.py'

import os
import sys
import datetime
import numpy as np


def shapely_grid(bins, size, center):
    
    from shapely import affinity, geometry
    
    xsize=size[0]
    ysize=size[1]
    
    
    x, y = np.meshgrid(np.arange(-xsize/2+center[0], xsize/2+center[0], bins), np.arange(-ysize/2+center[1], ysize/2+center[1], bins))
    x, y = x.flatten(), y.flatten()
    pixels = np.vstack((x,y)).T 

    pixelscont=np.empty(shape=(len(pixels)), dtype=object)
    pixelscontpoint=np.empty(shape=(len(pixels),4), dtype=object)
    pixindex=np.zeros(np.shape(pixels),dtype=int)

  
    for p,pixel in enumerate(pixels):

        #squarecorner = np.array([[pixel[0]-bins/2, pixel[1]-bins/2], [pixel[0]-bins/2,pixel[1]+bins/2], [pixel[0]+bins/2,pixel[1]+bins/2], [pixel[0]+bins/2,pixel[1]-bins/2]])
        squarecorner = np.array([[pixel[0]- 0.5, pixel[1]- 0.5], [pixel[0]- 0.5,pixel[1]+bins -0.5], [pixel[0]+bins - 0.5, pixel[1]+ bins - 0.5], [pixel[0]+bins -0.5,pixel[1]-0.5]])
        
        pixelscontpoint[p] = [geometry.Point( squarecorner[i][0], squarecorner[i][1]) for i in range(4)]
        

        pixelscont[p]=geometry.Polygon([[pi.x, pi.y] for pi in pixelscontpoint[p]])
        pixindex[p]=pixel
        
    Multipol=geometry.MultiPolygon(pixelscont.tolist())
    
    
    #return(np.array(list(affinity.translate(Multipol, xoff=-Multipol.centroid.x+center[0]  - 0.5    , yoff=-Multipol.centroid.y+center[1]  - 0.5))))
    return(Multipol)




def show_Mutipolygon( multipol, ax=None):

    import geopandas

    import shapely.geometry as sg
    import shapely.ops as so
    import matplotlib.pyplot as plt
    
    
    new_shape = so.cascaded_union(multipol)

    if ax ==None:
        fig, ax= plt.subplots()
        
    else:
        fig = ax.figure

    ax.set_aspect('equal', 'datalim')
    for geom in multipol.geoms:    
        xs, ys = geom.exterior.xy    
        ax.fill(xs, ys, alpha=0.5, fc='none', ec='k')
    

    return (fig,ax)






def restride(arr, binfactor, squeezed=True, flattened=False):
    """
    Rebin ND-array `arr` by `binfactor`.
    Let `arr.shape = (s1, s2, ...)` and `binfactor = (b1, b2, ...)` (same
    length), new shape will be `(s1/b1, s2/b2, ... b1, b2, ...)` (squeezed).
    * If `binfactor` is an iterable of length < `arr.ndim`, it is prepended
      with 1's.
    * If `binfactor` is an integer, it is considered as the bin factor for all
      axes.
    If `flattened`, the bin axes are explicitely flattened into a single
    axis. Note that this will probably induce a copy of the array.
    Bin 2D-array by a factor 2:
    >>> restride(np.ones((6, 8)), 2).shape
    (3, 4, 2, 2)
    Bin 2D-array by a factor 2, with flattening of the last 2 bin axes:
    >>> restride(np.ones((6, 8)), 2, flattened=True).shape
    (3, 4, 4)
    Bin 2D-array by uneven factor (3, 2):
    >>> restride(np.ones((6, 8)), (3, 2)).shape
    (2, 4, 3, 2)
    Bin 3D-array by factor 2 over the last 2 axes, and take bin average:
    >>> q = np.arange(2*4*6).reshape(2, 4, 6)
    >>> restride(q, (2, 2)).mean(axis=(-1, -2))
    array([[[ 3.5,  5.5,  7.5],
            [15.5, 17.5, 19.5]],
           [[27.5, 29.5, 31.5],
            [39.5, 41.5, 43.5]]])
    Bin 3D-array by factor 2, and take bin average:
    >>> restride(q, 2).mean(axis=(-1, -2, -3))
    array([[15.5, 17.5, 19.5],
           [27.5, 29.5, 31.5]])
    .. Note:: for a 2D-array, `restride(arr, (3, 2))` is equivalent to::
         np.moveaxis(arr.ravel().reshape(arr.shape[1]/3, arr.shape[0]/2, 3, 2), 1, 2)
    """

    try:                        # binfactor is list-like
        # Convert binfactor to [1, ...] + binfactor
        binshape = [1] * (arr.ndim - len(binfactor)) + list(binfactor)
    except TypeError:           # binfactor is not list-like
        binshape = [binfactor] * arr.ndim

    assert len(binshape) == arr.ndim, "Invalid bin factor (shape)."
    assert (~np.mod(arr.shape, binshape).astype('bool')).all(), \
        "Invalid bin factor (modulo)."

    # New shape
    rshape = [ d // b for d, b in zip(arr.shape, binshape) ] + binshape
    # New stride
    rstride = [ d * b for d, b in zip(arr.strides, binshape) ] + list(arr.strides)

    rarr = np.lib.stride_tricks.as_strided(arr, rshape, rstride)

    if flattened:               # Flatten bin axes, which may induce a costful copy!
        rarr = rarr.reshape(rarr.shape[:-(rarr.ndim - arr.ndim)] + (-1,))

    return rarr.squeeze() if squeezed else rarr  # Remove length-1 axes






def get_cube_grid(cube, scale=1, targShift=(0,0), x0=0, y0=0):  
    
    from shapely import geometry
    
    """
    Return MultiPolygon from Shapely
    """
    
    index=cube.indexes
    
    vertices=cube.get_index_vertices(index)

    #nan_idx=np.where([True in np.isnan(vertices)[i] for i in range(len(vertices))])[0]
    
    #if len(nan_idx)>0:
    #    vertices = np.delete(vertices, nan_idx, axis=0)
    #    index = np.delete(index, nan_idx, axis=0)
        
    
    cell = np.empty(shape=(len(index),), dtype=object)

    for j in range(0,len(index)):

        pointlist = np.empty( shape=( len(vertices[1]) ), dtype=object)
       
        for i in range( len(vertices[1]) ):
                   
            pointlist[i] = geometry.Point( np.array( [ vertices[j].T[0] * scale + x0 - targShift[0] * scale, vertices[j].T[1] * scale + y0 - targShift[1] * scale]).T[i][0],
                                           np.array( [ vertices[j].T[0] * scale + x0 - targShift[0] * scale, vertices[j].T[1] * scale + y0 - targShift[1] * scale]).T[i][1])
               
        cell[j]=geometry.Polygon([[p.x, p.y] for p in pointlist])
  
    return(geometry.MultiPolygon(cell.tolist()))



# End of geometry_tool.py ========================================================
