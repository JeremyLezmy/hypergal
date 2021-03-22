#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          SED_Fitting.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: rlezmy $
# Created on:        $Date: 2021/01/21 14:40:25 $
# Modified on:       2021/03/11 17:26:43
# Copyright:         2019, Jeremy Lezmy
# $Id: SED_Fitting.py, 2021/01/21 14:40:25  JL $
################################################################################

"""
.. _SED_Fitting.py:

SED_Fitting.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/21 14:40:25'
__adv__ = 'SED_Fitting.py'

import os
import sys
import datetime

import numpy as np
import pysedm
from pylephare import lephare, spectrum
from pylephare import io as ioleph
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


lbda_sedm = pysedm.sedm.SEDM_LBDA
PS1_FILTER_LEPHARE=['ps1.g','ps1.r', 'ps1.i', 'ps1.z', 'ps1.y']
PS1_FILTER_LEPHARE_err=['ps1.g.err','ps1.r.err', 'ps1.i.err', 'ps1.z.err', 'ps1.y.err']

ORDER = "ugrizy"
POS = {c:p for (p, c) in enumerate(ORDER)}


class Lephare_SEDfitting():

    def __init__(self, dataframe ):

        if type(dataframe)==geopandas.geodataframe.GeoDataFrame:       
            self.dataframe = pd.DataFrame(dataframe)
        else:
            self.dataframe = dataframe

        


    def Setup_Lephare(self, spec_dirout = 'default', Sig_noise_ratio=0, Sig_noise_filt='all', redshift=0.01, context=None, configfile=None, inhz=False):

        if spec_dirout=='default':
            self.spec_dirout = ioleph.get_default_path()

        else:
            self.spec_dirout = ioleph.get_default_path().rsplit('/',1)[0] + '/' + str(spec_dirout)

       

        
        LePhare_DF = self.dataframe.copy()

        
        for col in LePhare_DF:

            if col not in PS1_FILTER_LEPHARE + PS1_FILTER_LEPHARE_err  :

                LePhare_DF.pop(col)

        lst=list(LePhare_DF.columns)
        lst.sort(key = lambda c: POS[c.split('.')[1][-1]])
        LePhare_DF = LePhare_DF[lst]

        if context is not None:

            LePhare_DF['context']=np.array([CONTEXT]*len(LePhare_DF))

        LePhare_DF['zspec']=np.array([redshift]*len(LePhare_DF))
        

        if Sig_noise_filt == 'all' :
            
            filt = [ele for ele in lst  if ('err' not in ele)]
        else:
            filt = Sig_noise_filt
        
        idx = LePhare_DF.loc[ np.logical_and.reduce([ LePhare_DF[i].values / LePhare_DF[i + '.err'].values > Sig_noise_ratio for i in filt])].index

        Lephare_DF_threshold = LePhare_DF.loc[idx].copy()
        

        lp = lephare.LePhare(data=Lephare_DF_threshold, configfile=configfile, dirout=self.spec_dirout, inhz=inhz)
        
        self.Lephare_DF = LePhare_DF
        self.Lephare_DF_threshold = Lephare_DF_threshold
        self.Lephare_instance = lp
        self._idx_underThreshold = idx


       

    def run_Lephare(self, **kwarg):

        if not hasattr(self, 'Lephare_DF'):
            warnings.warn("You didn't setup lephare with self.Setup_Lephare(). Default parameters will be use")
            self.Setup_Lephare
            
        lp = self.Lephare_instance
        lp_out = lp.run( **kwarg )

        lp_out['spec'].sort()

        self.LephareOut = lp_out

        return(lp_out)


    def get_Sample_spectra(self, lbda_sample = lbda_sedm, lbda_lephare_range = [3000,10000], interp_kind = 'cubic', box_ker_size=10, save_dirout_data = None):

       
        kerbox = Box1DKernel( box_ker_size )       

        if not hasattr(self, 'LephareOut'):
            raise AttributeError("No spectra loaded yet. Run self.run_Lephare()")
        
        lp_out = self.LephareOut
        full_DF = self.Lephare_DF

       
        spec_data_leph=np.empty(shape=(len( full_DF )), dtype='object')
        spec_data_interp=np.zeros(shape=(len( full_DF ),len(lbda_sample)))
        

        fitind=0           
        for i in range(len( full_DF )):
          
            if i in ( self._idx_underThreshold ):
                
                valid_spec = 'yes'
                
                try:
                    spectrum.LePhareSpectrum().load(lp_out["spec"][fitind])
                    
                except:# ValueError: 
                    valid_spec='no'
                    
                if valid_spec=='yes' :
                    
                    lbda=np.array(spectrum.LePhareSpectrum (filename = lp_out["spec"][fitind], lbda_range = lbda_lephare_range ).get_spectral_data()[0] )
                    #spec_data_leph[i] = np.zeros(shape=(len(lbda)))
                    spec_data_leph[i] = np.array(spectrum.LePhareSpectrum(filename=lp_out["spec"][fitind], lbda_range=lbda_lephare_range).get_spectral_data()[1] )        
                    f=interp1d(lbda, spec_data_leph[i], kind = interp_kind)
                    spec_data_interp[i] = convolve( f (np.linspace(lbda_sample[0],lbda_sample[-1],len(lbda_sample)*box_ker_size) ),kerbox,boundary='extend',normalize_kernel=True)[::10]
                    #spec_data_interp[i]=f(lbda_sedm)
                    
                else:
                    
                    spec_data_interp[i] = np.array([np.nan] * len(lbda_sample))
                    #spec_data_fit[i]=np.array([np.nan]*len(lbda))
                    spec_data_leph[i] = np.nan
                    
                fitind+=1
                
            else:
                spec_data_interp[i] = np.array([0] * len(lbda_sample))
                #spec_data_fit[i]=np.array([np.nan]*len(lbda))
                spec_data_leph[i] = np.nan

        self.spec_lbda_leph_range = lbda_lephare_range
        self.spec_sample = spec_data_interp
        self.spec_lbda_sample = lbda_sample
        self.spec_leph = spec_data_leph

        if save_dirout_data is not None:

            np.savez(save_dirout_data, spec=spec_data_interp, lbda=lbda_sample)
        
        return(spec_data_interp,lbda_sample)



    
    def get_3D_cube(self, pixel_bin = 2, origin_shift = -0.5):
        
        
        spec, lbda = self.spec_sample, self.spec_lbda_sample

        
        pixMap=dict(zip(np.arange(0, len(spec)), np.array( [np.array( self.dataframe['centroid_x']), np.array(self.dataframe['centroid_y'])]).T))
        cube=pyifu.spectroscopy.get_cube(data=spec.T,lbda=lbda,spaxel_mapping=pixMap)

        cube.set_spaxel_vertices(xy = [[origin_shift, origin_shift],[origin_shift + pixel_bin, origin_shift],[origin_shift + pixel_bin , origin_shift + pixel_bin],[origin_shift, origin_shift + pixel_bin ]])

        self.cube3D = cube
        
        return(cube)








#class SED_Fitting_to_cube():

#    def __init__( self, spec, lbda, geo):

#        self.spec_data = spec
#        self.lbda = lbda
        




# End of SED_Fitting.py ========================================================
