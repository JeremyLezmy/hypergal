#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          sedmtools.py
# Description:       script description
# Author:            Jeremy Lezmy <lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/05/10 17:27:54 $
# Modified on:       2021/05/10 17:57:20
# Copyright:         2019, Jeremy Lezmy
# $Id: sedmtools.py, 2021/05/10 17:27:54  JL $
################################################################################

"""
.. _sedmtools.py:

sedmtools.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/05/10 17:27:54'
__adv__ = 'sedmtools.py'

import os
import sys
import datetime
from ztfquery import sedm, io
from pysedm import astrometry, byecr
import pysedm
from shapely.geometry.point import Point

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
    

SEDMDIROUT = os.path.join(SEDMLOCAL_BASESOURCE,"redux")
PHAROS_DATALOC = 'http://pharos.caltech.edu/data/'


def remove_targ_spx(cube, xy, radius = 3, store = True, get_filename = True):
    
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
    cube_wotar.set_filename( cube.filename.replace('e3d', 'e3d_spx_removed') )
    
    if store:
        cube_wotar.writeto( cube_wotar.filename )

    if get_filename:
        return cube_wotar.filename
        
    return cube_wotar



def get_byecr_cube(self, cube , cut_critera = 5, store = True, get_filename = True):

        """ Apply byecr method (cosmic ray removing) over sedm cube

        Parameters
        ----------
        cube: [pyifu.Cube]

        cut_criteria: [float] -optional-
            cut criteria values we want to use.
            Default is '5'.

        store: [bool]
               Do you want to store the new cube?
       
        get_filename: [bool]
               If True, return filename of the new cube
               If False, return the new cube object
                
        Returns
        -------
        New cube without cosmic ray
        """

        night = cube.header['OBSDATE'].rsplit('-')
        night = ''.join(night)

        try:
            bycrcl = byecr.SEDM_BYECR( night, cube)
            cr_df = bycrcl.get_cr_spaxel_info(None, False, cut_critera)
            
        except:
            download_hexagrid(night)
            bycrcl = pysedm.byecr.SEDM_BYECR( night, cube)
            cr_df = bycrcl.get_cr_spaxel_info(None, False, cut_critera)

        print( bcolors.OKBLUE + f" Byecr succeeded, {len(cr_df)} detected cosmic-rays removed!" + bcolors.ENDC)

        cube_bycr = cube.copy()
        cube_bycr.data[cr_df["cr_lbda_index"], cr_df["cr_spaxel_index"]] = np.nan
        cube_bycr.header.set("NCR", len(cr_df), "total number of detected cosmic-rays from byecr")
        cube_bycr.header.set("NCRSPX", len(np.unique(cr_df["cr_spaxel_index"])), "total number of cosmic-ray affected spaxels")

        cube_bycr.set_filename( cube.filename.replace('e3d', 'e3d_crr') )
        
        if store:
            cube_bycr.writeto( cube_bycr.filename)

        if get_filename:
            return cube_byecr.filename
             
        return cube_bycr


def download_hexagrid(self, night, dirout = SEDMDIROUT, nodl = False, show_progress = False, overwrite=False):
        
    io.download_single_url( url = os.path.join(PHAROS_DATALOC, night, night +'_HexaGrid.pkl'), fileout =os.path.join( dirout, night, night + '_HexaGrid.pkl'), show_progress=False, overwrite=overwrite )

# End of sedmtools.py ========================================================
