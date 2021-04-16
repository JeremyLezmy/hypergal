#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          Panstarrs_target.py
# Description:       script description
# Author:            Jeremy Graziani <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/01/18 10:38:37 $
# Modified on:       2021/04/16 20:23:32
# Copyright:         2021, Jeremy Lezmy
# $Id: Panstarrs_target.py, 2021/01/18 17:23:14  JL $
################################################################################

"""
.. _Panstarrs_target.py:

Panstarrs_target.py
==============


"""
__license__ = "2021, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/18 17:23:14'
__adv__ = 'Panstarrs_target.py'

import os
import sys
import datetime

import numpy as np

from pymage import panstarrs 
from astropy import units as u
from astropy.coordinates import SkyCoord
from pymage import panstarrs
from astropy import constants
from hypergal import geometry_tool as geotool
import geopandas
import matplotlib.pyplot as plt




def HR_to_deg( ra, dec):
    
    co=SkyCoord(ra, dec, frame='icrs',unit=(u.hourangle, u.deg))
    ra=co.ra.deg
    dec=co.dec.deg

    return (ra, dec)


class Panstarrs_target():

    def __init__(self, ra, dec):

        self.ra = ra
        self.dec = dec
        
        self.target = panstarrs.PS1Target.from_coord(ra,dec)
        self.target.download_catalog(update=True)
        #self.target.download_extended_catalog()
        

    def load_cutout(self, size = 240, load_weight = True ):

        try:
            self.target.download_cutout(size = size, load_weight = load_weight)

        except:
            print("No cutout available")

        else:
            self.imgcutout = self.target.imgcutout
        
        
            self.gfilter = self.target.imgcutout["g"]
            self.rfilter = self.target.imgcutout["r"]
            self.ifilter = self.target.imgcutout["i"]
            self.zfilter = self.target.imgcutout["z"]
            self.yfilter = self.target.imgcutout["y"]
            
            self._size = size


    def available_filters(self):

        if not self.has_cutout:
            raise AttributeError("No cutout loaded yet. Run self.load_cutout()")

        else:
            self.available_filters = list(self.imgcutout.keys())

        return(self.available_filters)


    def build_geo_dataframe(self, subsample = 2):

        if not hasattr(self,'_size'):
            raise AttributeError("No cutout loaded yet. Run self.load_cutout()")
        
        size = self._size
        center = self._size/2
        self.full_grid = geotool.shapely_grid( bins = subsample, size = (size,size), center = (center,center) )

        df = geopandas.GeoDataFrame()
        df['geometry'] = np.array(self.full_grid)
        
        for filt in self.available_filters:
            
            if self.imgcutout[filt].data is not None:

                if subsample==1:
                    
                    df['ps1.'+ str(filt)] = self.imgcutout[filt].count_to_flux( self.imgcutout[filt].data.ravel() )
                    df['ps1.'+ str(filt) + '.err'] = self.imgcutout[filt].count_to_flux( self.imgcutout[filt].var.ravel()**0.5 )
                    
                elif subsample > 1:
                    
                    restride_dat, restride_err = geotool.restride(self.imgcutout[filt].data, subsample), geotool.restride(self.imgcutout[filt].var**0.5, subsample)
                    sum_restride_dat, sum_restride_err = np.sum(restride_dat, axis=(2,3)), np.sum(restride_err, axis=(2,3))

                    df['ps1.'+ str(filt)] = self.imgcutout[filt].count_to_flux( sum_restride_dat.ravel() )
                    df['ps1.'+ str(filt) + '.err'] = self.imgcutout[filt].count_to_flux( sum_restride_err.ravel() )

        df = df.assign(**{'centroid_x' :df['geometry'].centroid.x, 'centroid_y':df['geometry'].centroid.y, 'id_pixel':np.arange(0,len(df))})

        self.geo_dataframe = df
        self.subsample = subsample

        return(df)



    def make_cigale_compatible(self,  mJy=True):
    
        cig_df = self.geo_dataframe.copy()
    
        for filt in self.available_filters:
            cig_df['ps1.'+ filt] = flux_aa_to_hz(cig_df['ps1.'+ filt], self.imgcutout[filt].INFO['ps1.'+ filt]['lbda'] ) * 10**26
            cig_df['ps1.'+ filt + '.err'] = flux_aa_to_hz(cig_df['ps1.'+ filt + '.err'], self.imgcutout[filt].INFO['ps1.'+ filt]['lbda'] ) * 10**26
    
        cig_df.columns = cig_df.columns.str.replace(".", "_")
            
        return cig_df

    

    def show(self, with_grid=False, ax=None, filt = 'ps1.r', origin='lower'):


        size = self._size
        subsample = self.subsample
        if ax==None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.imshow(np.reshape(self.geo_dataframe[filt].values, (int(size/subsample), int(size/subsample))), origin=origin, extent=(-0.5, size-0.5, -0.5, size-0.5) )

        if with_grid:
            geotool.show_Mutipolygon( self.full_grid, ax=ax)


        return(fig,ax)
        

    def get_target_coord(self):

        return( self.rfilter.coords_to_pixel( self.ra, self.dec))

    def get_pix_size(self):

        return(self.rfilter.pixel_size_arcsec.value)
    

    @property
    def has_cutout(self):
                       
        try:
            self.target.imgcutout

        except:
            return(False)

        else:
            return(True)

        
    @property
    def available_filters(self):

        if not self.has_cutout:
            raise AttributeError("No cutout loaded yet. Run self.load_cutout()")

        else:
           filt_list = list(self.imgcutout.keys())

        return(filt_list)
    


def flux_aa_to_hz(flux, wavelength):
 
    return flux * (wavelength**2 / constants.c.to("AA/s").value)



def flux_hz_to_aa(flux, wavelength):
   
    return flux / (wavelength**2 / constants.c.to("AA/s").value)





# End of Panstarrs_target.py ========================================================
