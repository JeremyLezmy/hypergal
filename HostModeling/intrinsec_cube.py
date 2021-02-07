#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          intrinsec_cube.py
# Description:       script description
# Author:            Jeremy Lezmy <lezmy@ipnl.in2p3.fr>
# Author:            $Author: rlezmy $
# Created on:        $Date: 2021/01/25 13:28:46 $
# Modified on:       2021/02/01 11:02:29
# Copyright:         2019, Jeremy Lezmy
# $Id: intrinsec_cube.py, 2021/01/25 13:28:46  JL $
################################################################################

"""
.. _intrinsec_cube.py:

intrinsec_cube.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/25 13:28:46'
__adv__ = 'intrinsec_cube.py'

import os
import sys
import datetime


import numpy as np
import pysedm
from pylephare import lephare, io, spectrum
import pandas as pd
from shapely import geometry
import shapely
import time
from shapely import affinity
import geopandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve
import warnings
import pyifu

from HostModeling import SED_Fitting as sedfit
from HostModeling import Panstarrs_target as ps1targ
from HostModeling import sedm_target as sedtarg
from HostModeling import geometry_tool as geotool


from collections import OrderedDict
from HostModeling import PSF_kernel as psfker


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


class Parameter(OrderedDict):
  
    
    def __init__(self):
        
        self.params = dict()
        self.bounds = dict()
    
    def add_param(self, name, value=None, minval=None, maxval=None):
        """
        Parameters
        ----------
        name : str
            Name of the Parameter.
        value : float, optional
            Numerical Parameter value.
       
        min : float, optional
            Lower bound for value (default is ``-numpy.inf``, no lower
            bound).
        max : float, optional
            Upper bound for value (default is ``numpy.inf``, no upper
            bound).
     
        """
        
        self.params.update({name:value})
        self.bounds.update({name:(minval,maxval)})
        
    def valuesdict(self):
        """Return an ordered dictionary of parameter values.
        Returns
        -------
        OrderedDict
            An ordered dictionary of :attr:`name`::attr:`value` pairs for
            each Parameter.
        """
        return OrderedDict(self.params)
    
    def boundsdict(self):
        
        return OrderedDict(self.bounds)
    
    def update_value(self, name, value):
        
        if name in self.params.keys():
            self.params[name]=value
        else:
            pass
        
    def update_bounds(self, name, minval, maxval):
        
        if name in self.params.keys():
            self.bounds[name]=(minval,maxval)
        else:
            pass
        






class Intrinsec_cube():


    def __init__(self, pixelgrid, pixel_size, hexagrid, spec, lbda, psf_model = 'Gauss_Mof_kernel' ):

        self.pixelgrid = pixelgrid
        self.hexagrid = hexagrid
        self.spec = spec
        self.lbda = lbda
        self.pixelsize = pixel_size

        if psf_model is not None:
            self.psfmodel = psfker.PSF_kernel(psf_model)



    def load_adr(self, adr=None, **kwargs):

        if adr==None:
            adr = pyifu.adr.ADR()
        
            for (k,v) in kwargs.items():
                if k not in adr.PROPERTIES:
                    raise ValueError("unknown property %s, it cannot be set. known properties are: "%k,", ".join(adr.PROPERTIES))
                #exec("%s = %d" % (k,v))
                adr._properties[k]=  v
        
            self.adr = adr
        else:
            self.adr = adr


    def get_metaslices_data( self, lbda_ranges, metaslices):

        lbda_stepbin = self.lbda_step_bin(lbda_ranges, metaslices)

        binned_spec = np.zeros( (np.shape(self.spec)[0], metaslices) )
        binned_lbda = np.zeros( metaslices )
        
        for (i,j) in enumerate( lbda_stepbin ):
            
            binned_spec[:,i] = np.mean(  self.spec[:, (self.lbda>=j[0]) & (self.lbda<j[1])], axis=1)
            binned_lbda[i] = np.mean ( self.lbda[  (self.lbda>=j[0]) & (self.lbda<j[1]) ] )

        self.binned_spec , self.binned_lbda = binned_spec, binned_lbda

        return ( binned_spec, binned_lbda ) 
        


        
        
    def run_overlay(self, nb_process = 'auto', apply_psf = True, use_binned_data = False, lbda_ranges=[3700, 9300], metaslices=22, **kwargs):


        
        self.set_nb_process(nb_process)

        
        if use_binned_data :

            spec, lbda = self.get_metaslices_data( lbda_ranges, metaslices)

        else :
            spec, lbda = self.spec, self.lbda


        if not hasattr(self, 'adr'):
            
            print(bcolors.WARNING + " There's not any loaded adr ( self.load_adr() ), there won't be any refraction in the intrinsec cube" + bcolors.ENDC)
            adr = None
            lbdaref = 6500
            
        elif hasattr(self,'adr') and  any(self.adr._properties[k] == None for k in self.adr._fundamental_parameters):
            
            print( bcolors.WARNING + "Some fundamentals parameters are not set in adr, there won't be any refraction in the intrinsec cube" + bcolors.ENDC)
            adr = None
            lbdaref = 6500

        else:
            adr = self.adr
            lbdaref = self.adr.lbdaref
        
        

        if apply_psf and hasattr(self, 'psfmodel'):

            H,W = self.get_shape_pixelgrid()
            
            spec = psf_convolution( np.reshape(spec, ( H, W, len(lbda)) ), lbda, lbdaref, self.psfmodel, self._nb_process)
            spec = spec.reshape(int(H*W), len(lbda))



            
        
        new_spax = measure_overlay(self._nb_process, spec, lbda, self.pixelgrid, self.hexagrid, adr, self.pixelsize  )

        
        self.new_spax = new_spax

        self.lbda_used = lbda
        self.spec_used = spec
        
        return( new_spax )


    def Build_model_cube( self, target_ifu=[0,0] ,target_image=[0,0], ifu_ratio=2.12, corr_factor = 1):

        
        spax_data = self.new_spax

        flux = np.array([spax_data[i]['flux'] for i in range(len(spax_data))])
        
        pixMap = dict(zip(np.arange(0, len(spax_data[0])), np.array( [ (np.array (spax_data[0]['centroid_x']) + target_ifu[0] * ifu_ratio - target_image[0]) / ifu_ratio,
                                                                       (np.array (spax_data[0]['centroid_y']) + target_ifu[1] * ifu_ratio - target_image[1]) / ifu_ratio ]).T ))


        spax_vertices = np.array([[ 0.19491447,  0.6375365 ],[-0.45466557,  0.48756913],[-0.64958004, -0.14996737],[-0.19491447, -0.6375365 ],[ 0.45466557, -0.48756913], [ 0.64958004,  0.14996737]])
        
        Model_cube = pyifu.spectroscopy.get_cube( data = corr_factor * flux,lbda = self.lbda_used, spaxel_mapping = pixMap)

        Model_cube.set_spaxel_vertices( spax_vertices )

        self.Model_cube = Model_cube
        self.pixmapping = pixMap
        
        return (Model_cube)
       




    def lbda_step_bin(self, lbda_ranges, metaslice):


        STEP_LBDA_RANGE = np.linspace(lbda_ranges[0],lbda_ranges[1], metaslice+1)
        return np.asarray([STEP_LBDA_RANGE[:-1], STEP_LBDA_RANGE[1:]]).T

    

    def update_adr(self, **kwargs):

        for (k,v) in kwargs.items():
            if k not in self.adr.PROPERTIES:
                raise ValueError("unknown property %s, it cannot be set. known properties are: "%k,", ".join(self.adr.PROPERTIES))
            #exec("%s = %d" % (k,v))
            self.adr._properties[k]=  v
        
        
        
    
    def update_PSFparameter(self, **kwargs):
            
            self.psfmodel.update_parameter(**kwargs) 
            
      

    def get_shape_pixelgrid(self):

        H,W = (np.array( self.pixelgrid.envelope.exterior.xy).T[0] - np.array( self.pixelgrid.envelope.exterior.xy).T[2] ) / 2
        H = abs(int(H))
        W = abs(int(W))
        return (H,W)




    def set_hexagrid(self, hexagrid):

        self.hexagrid = hexagrid


    def set_nb_process(self, value):

        if value == 'auto':
            
            import multiprocessing
            self._nb_process = multiprocessing.cpu_count() - 2
            
        else:
            self._nb_process = value



            

def measure_overlay(nb_process, spec, lbda, pixelgrid, hexagrid, adr, pixel_size):
    
    import time
    from pathos.multiprocessing import ProcessingPool as Pool
    import pathos
    

    centroid_x = np.array([i.centroid.x for i in list(pixelgrid)])
    centroid_y = np.array([i.centroid.y for i in list(pixelgrid)])
    
    if adr == None:
        
        newcent = np.repeat(np.array([centroid_x,centroid_y]).T[:, :, np.newaxis],len(lbda),axis=2 ).swapaxes(1,2)
        
    else:
        newcent=adr.refract(centroid_x,centroid_y,lbda, unit=pixel_size).T
    
    spax=geopandas.GeoDataFrame({"geometry":list(hexagrid), "id_spaxels":np.arange(0,len( list(hexagrid) ))})
    
    pix=geopandas.GeoDataFrame({"geometry":list(pixelgrid),"id_pixels":np.arange(0,len( list(pixelgrid) ))})
    pix['centroid_x']=pix['geometry'].centroid.x
    pix['centroid_y']=pix['geometry'].centroid.y
    
    def multiprocess(number):

        pixels = pix.copy()
        pixels['flux'] = spec[:,number]
        
        pixel_wadr=pixels.copy()
        pixel_wadr[['centroid_x','centroid_y']]=newcent[:,number,:]
        
        pixel_wadr['geometry']=pixel_wadr['geometry'].translate(xoff=np.array([pixel_wadr['centroid_x'][0]-(centroid_x[0])]),yoff=np.array([pixel_wadr['centroid_y'][0]-(centroid_y[0])])) #*scales

        spaxels=spax.copy()
        
        intersec=geopandas.overlay(pixel_wadr,spaxels,how='intersection')
        
        def localdef_get_area(l):
            return l.geometry.area/pixel_wadr.iloc[l.id_pixels].geometry.area
    
        intersec['area']=intersec.apply(localdef_get_area,axis=1)
    
        intersec["newflux"]=intersec['area']*intersec['flux']
        spaxels['flux']=intersec.groupby('id_spaxels')['newflux'].sum()

        return(spaxels)
    

    start=time.time()
    print(f"Start of the multiprocessing with {nb_process} core")   
    
    with Pool(nb_process) as p:
        new_spax = p.map(multiprocess, range(len(lbda)))
    
     
    end=time.time()   
    print(f"end of the multiprocessing, it took: { end-start} s")
    
    
    for i in range(len(lbda)):
        new_spax[i]['centroid_x'] = new_spax[i]['geometry'].centroid.x
        new_spax[i]['centroid_y'] = new_spax[i]['geometry'].centroid.y

    

    return ( new_spax )


def psf_convolution(data, lbda, lbdaref, psfkernel, nb_process, **kwargs ):

    import pathos
    from pathos.multiprocessing import ProcessingPool as Pool

    new_data = np.atleast_3d(data)
    

    def multipro_psf(number):
       
        new_data[:,:,number] = psfkernel.convolution( new_data[:,:,number],  lbda[number], lbdaref)

        return new_data[:,:,number]

    with Pool(nb_process) as p:
        result = p.map(multipro_psf, np.arange( new_data.shape[-1] ))
    
     
    return(np.asarray(result).T.swapaxes(0,1))




   

# End of intrinsec_cube.py ========================================================
