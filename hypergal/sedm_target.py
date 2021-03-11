#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          sedm_target.py
# Description:       script description
# Author:            Jeremy Graziani <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/01/18 10:38:37 $
# Modified on:       2021/03/11 16:39:35
# Copyright:         2021, Jeremy Lezmy
# $Id: sedm_target.py, 2021/01/18 10:38:37  JL $
################################################################################

"""
.. _sedm_target.py:

sedm_target.py
==============


"""
__license__ = "2021, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/18 10:38:37'
__adv__ = 'sedm_target.py'

import os
import sys
import datetime

import pandas as pd
import pyifu
from pysedm import astrometry, byecr

import matplotlib.pyplot as plt
from ztfquery import sedm, io

import pysedm

import matplotlib
import numpy as np
from hypergal import geometry_tool as geotool




PHAROS_DATALOC = 'http://pharos.caltech.edu/data/'
DATA_PREFIX = '/e3d_crr_b_ifu'
FLUXCAL_PREFIX = '/fluxcal_auto_robot_lstep1__crr_b_'

ASTRO_PREFIX = '/guider_crr_b_ifu'
ASTRO_SUFFIX = '_astrom.fits'

LOCALSOURCE   = os.getenv('ZTFDATA',"./Data/")
SEDMLOCAL_BASESOURCE = os.path.join(LOCALSOURCE,"SEDM")
SEDMDIROUT = os.path.join(SEDMLOCAL_BASESOURCE,"redux")

SPEC_CONT_PREFIX = '/spec_auto_contsep_lstep1__crr_b_ifu'
SPEC_ROBOT_PREFIX = '/spec_auto_robot_lstep1__crr_b_ifu'

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



class SEDM_tools():


    def __init__(self, target, night, obs_hour):

        self.target = target
        self.night = night
        self.obs_hour = obs_hour



    def get_cube(self, path = 'default' ):

        
        if path == 'default':
            
            try:
                cube = pysedm.get_sedmcube( SEDMDIROUT + '/' + self.night + DATA_PREFIX + self.night + '_' + self.obs_hour + '_' + self.target   + '.fits')

            except:
                self.download_e3d()
                cube = pysedm.get_sedmcube( SEDMDIROUT + '/' + self.night + DATA_PREFIX + self.night + '_' + self.obs_hour + '_' + self.target   + '.fits')

        else:
            cube =  pysedm.get_sedmcube(path)

        self.cube = self.remove_out_spaxels(cube)

        self.cube.load_adr()

        
        return(self.cube)


    def get_calib_cube(self, cube = None, which_airmass = 'robot', path='default', remove_sky = True ):

        import fnmatch
        import os
        from pysedm import fluxcalibration
        from astropy.io import fits
        
        if cube == None:
            if not hasattr(self,"cube"):
                
                cube = self.get_cube(path).copy()
            else:
                cube = self.cube.copy()
                
        else:
            cube=cube.copy()
      
        try:
            for file in os.listdir(cube.filename.rsplit('/', 1)[0]):
                if fnmatch.fnmatch(file, 'spec_auto_'+ which_airmass + '*.fits'):
                    spec_fits = cube.filename.rsplit('/', 1)[0]+'/'+file
                    
            hdul = fits.open(spec_fits)

        except:

            self.download_spec(which = which_airmass)
            
            for file in os.listdir(cube.filename.rsplit('/', 1)[0]):
                if fnmatch.fnmatch(file, 'spec_auto_'+ which_airmass + '*.fits'):
                    spec_fits = cube.filename.rsplit('/', 1)[0]+'/'+file
                    
            hdul = fits.open(spec_fits)

        
        
        fcal = pysedm.io.fetch_nearest_fluxcal(file=cube.filename)
        
        if fcal==None:
            self.download_fluxcal()
            fcal = pysedm.io.fetch_nearest_fluxcal(file=cube.filename)
            
        fs=fluxcalibration.load_fluxcal_spectrum(fcal)
        
        if remove_sky == True:
            cube.remove_sky(usemean=True)
            
        cube.scale_by(cube.header['EXPTIME'], onraw=False)
        cube.scale_by(fs.get_inversed_sensitivity(hdul[0].header['airmass']), onraw=False)

        cube.load_adr()
        
        self.cube_cal = self.remove_out_spaxels(cube)
        
        return (self.cube_cal)


    def remove_out_spaxels(self, cube, overwrite = True):

        spx_map = cube.spaxel_mapping
        ill_spx = np.argwhere(np.isnan(list( spx_map.values() ))).T[0]

        if len(ill_spx)>0:

            cube_fix = cube.get_partial_cube([i for i in cube.indexes if i not in cube.indexes[ill_spx]],np.arange(len(cube.lbda)) )

            if overwrite:
        
                cube_fix.writeto(cube.filename)

            return cube_fix
        
        else:
            
            return cube


    def get_byecr_cube(self, cube , save=True, cut_critera=7):

        night = cube.header['OBSDATE'].rsplit('-')
        night = ''.join(night)

        try:
            bycrcl = pysedm.byecr.SEDM_BYECR( night, cube)
            cr_df = pysedm.bycrcl.get_cr_spaxel_info(None, False, cut_critera)
            
        except:
            self.download_hexagrid(night)
            bycrcl = pysedm.byecr.SEDM_BYECR( night, cube)
            cr_df = bycrcl.get_cr_spaxel_info(None, False, cut_critera)

        print( bcolors.OKBLUE + f" Byecr succeeded, {len(cr_df)} detected cosmic-rays removed!" + bcolors.ENDC)
        cube.data[cr_df["cr_lbda_index"], cr_df["cr_spaxel_index"]] = np.nan
        cube.header.set("NCR", len(cr_df), "total number of detected cosmic-rays from byecr")
        cube.header.set("NCRSPX", len(np.unique(cr_df["cr_spaxel_index"])), "total number of cosmic-ray affected spaxels")
        
        
        if save:
            cube.writeto(cube.filename.rsplit('crr')[0]+'crr_crr'+cube.filename.rsplit('crr')[1])
            cube_byecr = pysedm.get_sedmcube(cube.filename.rsplit('crr')[0]+'crr_crr'+cube.filename.rsplit('crr')[1])
            self.cube_crr = cube_byecr.copy()
            return(cube_byecr)
        
        else:
            self.cube_crr = cube.copy()
            return(cube)

    def get_hexagrid(self):

        hexagrid = geotool.get_cube_grid(self.cube)

        self.hexagrid = hexagrid

        return(hexagrid)

    
    def show_hexagrid(self, ax=None, slice_overlapped=None, **kwargs):

        if ax==None:
            fig,ax = plt.subplots()
            
        else:
            fig = ax.figure

        if not hasattr(self,'hexagrid'):
            
            self.get_hexagrid();
            
        geotool.show_Mutipolygon( self.hexagrid, ax=ax)

        if slice_overlapped is not None:
            sli = self.cube.get_slice(lbda_min = slice_overlapped[0], lbda_max = slice_overlapped[1], slice_object=True )
            sli.show(ax=ax, **kwargs);

        return fig,ax


    
    def get_estimate_target_coord(self):

        filename = self.cube.filename

        astro = astrometry.Astrometry(filename)

        try:
            self.targetastro = astro.get_target_coordinate()
        except:
            self.download_astrometry()
            self.targetastro = astro.get_target_coordinate()

        return( self.targetastro )

    

    def get_Host_redshift_fritz(self):

        from ztfquery import fritz as ftz

        fritzObj = ftz.FritzAccess()

        source = ftz.download_source(self.target, get_object =True)

        self.redshift = source.redshift
        return(source.redshift)
    

    ####################################################################################
    ########################## DOWNLOAD SECTION ########################################
    ####################################################################################

    def download_e3d(self, dirout = 'default', nodl = False, show_progress = False):

        s=sedm.SEDMQuery()
        s.download_target_data(self.target, nodl=nodl,show_progress = show_progress, download_dir = dirout)

        

    def download_fluxcal(self, dirout = 'default', nodl = False, show_progress = False):
        
        s=sedm.SEDMQuery()
        s.download_night_fluxcal(self.night, nodl=nodl,show_progress = show_progress, download_dir = dirout)


    def download_astrometry(self, dirout = SEDMDIROUT, nodl = False, show_progress = False, overwrite=False):

        io.download_single_url(PHAROS_DATALOC + self.night + ASTRO_PREFIX + self.night + '_' +  self.obs_hour + ASTRO_SUFFIX,
                               dirout + '/' + self.night+ ASTRO_PREFIX + self.night + '_' + self.obs_hour  + ASTRO_SUFFIX , show_progress=show_progress, overwrite = overwrite)
        

    def download_spec(self, dirout = SEDMDIROUT, which = 'contsep', nodl = False, show_progress = False, overwrite=False):

        if which == 'contsep':
            spec_prefix = SPEC_CONT_PREFIX

        elif which == 'robot':
            spec_prefix = SPEC_ROBOT_PREFIX
            
        io.download_single_url(PHAROS_DATALOC + self.night + spec_prefix + self.night + '_' +  self.obs_hour + '_'+ self.target +'.fits',
                               dirout + '/' + self.night+ spec_prefix + self.night + '_' + self.obs_hour  +  '_'+ self.target + '.fits' , show_progress=show_progress, overwrite = overwrite)


    def download_hexagrid(self, night, dirout = SEDMDIROUT, nodl = False, show_progress = False, overwrite=False):
        
        io.download_single_url(PHAROS_DATALOC + night + '/'+ night +'_HexaGrid.pkl', dirout + '/' + night +'/' + night + '_HexaGrid.pkl', show_progress=False, overwrite=overwrite )
       


    








# End of sedm_target.py ========================================================
