#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          sedfitting.py
# Description:       script description
# Author:            Jeremy Lezmy <lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/05/11 13:36:18 $
# Modified on:       2021/05/17 18:00:52
# Copyright:         2019, Jeremy Lezmy
# $Id: sedfitting.py, 2021/05/11 13:36:18  JL $
################################################################################

"""
.. _sedfitting.py:

sedfitting.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/05/11 13:36:18'
__adv__ = 'sedfitting.py'

import os
import sys
import datetime
import numpy as np
from .utils import * 
from configobj import ConfigObj
import json
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve
import geopandas
#import pkg_resources

#JSON_PATH = pkg_resources.resource_filename('hypergal', 'spectroscopy/config/')
JSON_PATH = os.path.dirname(os.path.realpath(__file__))

PS1_FILTER=['ps1_g', 'ps1_r', 'ps1_i', 'ps1_z', 'ps1_y']
PS1_FILTER_err=['ps1_g_err','ps1_r_err', 'ps1_i_err', 'ps1_z_err', 'ps1_y_err']

LBDA_SEDM = np.linspace(3700, 9300, 220)
ORDER = "ugrizy"
POS = {c:p for (p, c) in enumerate(ORDER)}

DEFAULT_CIGALE_MODULES = ['sfhdelayed', 'bc03', 'nebular', 'dustatt_modified_CF00', 'dale2014', 'redshifting']


class SEDFitter():

    def __init__(self, dataframe, redshift = 0.001, snr = None):
        """ 
        Initiate SEDFitter with a dataframe.

        Parameters
        ----------
        dataframe: Pandas.Dataframe
            Dataframe with flux data and flux errors for each filter you want to use.

        redshift: float -optional-
            Redshift of the object. Will be the same for each row of the dataframe\n
            Default is 0.001

        snr: float -optional-
            Threshold Signal over Noise ratio for all the bands, for which the pixel won't be selected for the sedfitting. 
        Returns
        --------
        """        
        if type(dataframe)==geopandas.geodataframe.GeoDataFrame:       
            self.set_dataframe(pd.DataFrame(dataframe))            
        else:
            self.set_dataframe(dataframe)

        self.set_snr(snr)
        self.set_redshift(redshift)

        
    def setup_df(self, path_to_save = None, to_mJy = True):
        
        """ 
        Make the input dataframe compatible with the sedfitter.

        Parameters
        ----------
        path_to_save: string
            Where you want to store the dataframe which will be the input of the sedfitter.\n
            For Cigale must be .txt file.

        which: string]
            Which sedfitter are you using. \n
            If Cigale (default), it will reorder the columns, and add one with ID for each pixel.

        to_mJy: bool
            
           
        Returns
        --------
        
        """
        df = self.input_df.copy()
        
        for col in df:
            if col not in PS1_FILTER + PS1_FILTER_err and col not in [k.replace('_','.') for k in PS1_FILTER + PS1_FILTER_err] :
                df.pop(col)
                
        lst=list(df.columns)
        lst.sort(key = lambda c: POS[c.split('_')[1][-1]])
        
        df = df[lst]
        df['redshift']=np.array([self.redshift]*len(df))

        if hasattr(self, '_sedfitter_name') and self._sedfitter_name=='cigale':      
            df['id']=df.index        
            df = df.reindex(columns=(['id','redshift'] + list([a for a in df.columns if a not in ['id','redshift']]) ))

        self.set_clean_df(df)
        filt = [ele for ele in lst  if ('err' not in ele)]
        self._filters = filt
        idx = df.loc[ np.logical_and.reduce([ df[i].values / df[i + '_err'].values > self.snr for i in filt])].index
        df_threshold = df.loc[idx].copy()
        self.set_input_sedfitter(df_threshold)
               
        if path_to_save==None:
            df_threshold.to_csv('in_sedfitter.txt', header=True, index=None, sep='\t', mode='w+')
            self._dfpath = 'in_sedfitter.txt'
        else:
            dirout = os.path.dirname(path_to_save)
            if not os.path.isdir(dirout) :
                os.makedirs(dirout, exist_ok = True)
            df_threshold.to_csv(path_to_save, header=True, index=None, sep='\t', mode='w+')
            self._dfpath = os.path.abspath(path_to_save)
        
        self._idx_used = idx
         
    # -------- #
    #  SETTER  #
    # -------- #
    
    def set_dataframe(self, dataframe):
        """ 
        Set original dataframe before setup
        """
        self._input_df = dataframe

    def set_snr(self, snr):
        """ 
        Set signal over noise ratio as threshold (used for all filters)
        """
        self._snr = snr

    def set_redshift(self, redshift):
        """ 
        Set redshift (will be shared for all the pixel)
        """
        self._redshift = redshift

    def set_clean_df(self, dataframe):
        """ 
        Set dataframe already in the format (column order, columns names etc) asked by the sedfitter. This is before SNR selection applied
        """
        self._clean_df = dataframe

    def set_input_sedfitter(self, dataframe):
        """ 
        Set dataframe which will be directly used by the sedfitter (after setup, SNR selection, and flux compatible)
        """
        self._input_sedfitter = dataframe
        
    # ================ #
    #  Properties      #
    # ================ #
    
    @property
    def input_df(self):
        """ 
        Original dataframe before setup
        """
        return self._input_df

    @property
    def snr(self):
        """
        Signal over noise ratio as threshold (used for all filters)
        """
        return self._snr

    @property
    def redshift(self):
        """ 
        Redshift (shared for all the pixel)
        """
        return self._redshift

    @property
    def clean_df(self):
        """ 
        Dataframe already in the format (column order, columns names etc) asked by the sedfitter. This is before SNR selection applied
        """
        if not hasattr(self, '_clean_df'):
            return None
        return self._clean_df

    @property
    def input_sedfitter(self):
        """
        Dataframe which will be directly used by the sedfitter (after setup, SNR selection, and flux compatible)
        """
        if not hasattr(self, '_input_sedfitter'):
            return None
        return self._input_sedfitter

    @property
    def dfpath(self):
        """ 
        Path of the dataframe read by te sedfitter
        """
        if not hasattr(self, '_dfpath'):
            return None
        return self._dfpath

    @property
    def idx_used(self):
        """ 
        Indices of the object which passed the SNR selection
        """
        if not hasattr(self, '_idx_used'):
            return None
        return self._idx_used

    @property
    def filters(self):
        """ 
        Filters used for the sedfitting
        """
        if not hasattr(self, '_filters'):
            return None
        return self._filters
    

class Cigale(SEDFitter):

    # ============= #
    #  Methods      #
    # ============= #

    def __init__(self, dataframe, redshift = 0.001, snr = None):
        
        super().__init__(dataframe = dataframe, redshift = redshift, snr = snr)
        self._sedfitter_name = 'cigale'

    def initiate_cigale(self, sed_modules='default', cores='auto', working_dir=None ):
        """ 
        Initiate Cigale (not launch yet)

        Parameters
        ---------

        sed_modules: list of string
            Name of each sed module you want to use. See Cigale Doc. \n
            Default is ['sfhdelayed', 'bc03', 'nebular', 'dustatt_modified_CF00', 'dale2014', 'redshifting']

        cores: int
            How many cores do you want to use for the sedfitting?\n
            Default is number available - 2 (1 if you only have 1 or 2 cores)

        working_dir: string
            Where do you want to run cigale?\n
            If None, will be in current pwd.\n
            Default is None.

        """

        self._currentpwd = os.getcwd()
        
        if working_dir is not None:
            if not os.path.isdir(working_dir) :
                os.makedirs(working_dir, exist_ok = True)
            os.chdir(working_dir)
        else :
            working_dir = self.currentpwd

        self._working_dir = os.path.abspath(working_dir)
            
        command_cigale('init')
        config = ConfigObj('pcigale.ini', encoding='utf8', write_empty_values=True)
        
        config['data_file'] = self.dfpath
        
        if sed_modules=='default':
            config['sed_modules'] = DEFAULT_CIGALE_MODULES
        
        elif type(sed_modules)!=list and type(sed_modules[0])!=str:
            print('sed_modules should be a list of string.')
            
        else:
            config['sed_modules'] = sed_modules
            
        self.set_nb_process(cores)
        config['analysis_method'] = 'pdf_analysis'
        config['cores'] = self._nb_process
        
        config.write()
        
        command_cigale('genconf')

        config = ConfigObj('pcigale.ini', encoding='utf8', write_empty_values=True)

        if sed_modules=='default':
            with open(os.path.join(JSON_PATH,'..','config',
                       'cigale.json')) as data_file:
               
                params = json.load(data_file)
                
            config = update(config,params)
        config['sed_modules_params'][[k for k in config['sed_modules_params'].keys() if 'dustatt' in k][0]]['filters'] = ' & '.join(ele for ele in config['bands']  if ('err' not in ele))
        
        config.write()            
        self._config = config


    def run(self, path_result=None, result_dir_name='out/'):
        """
        Run Cigale.\n
        Results will be stored in self._working_dir, and a directory 'out/' will be created.
        """
        
        command_cigale('run')

        #### Should be Removed or at least reworked #####
        
        if path_result is not None:
            actual_path = os.getcwd()+'/'
            if os.path.exists(path_result+ result_dir_name):
                name = path_result+datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '_'+ result_dir_name
                shutil.move(path_result+result_dir_name, name)
                print(f"The {result_dir_name} directory already exists, the old one was renamed to {name}")

            if os.path.exists(actual_path+result_dir_name):
                name = actual_path+datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '_'+ result_dir_name
                shutil.move(actual_path+result_dir_name, name)
                print(f"The {result_dir_name} directory already exists, the old one was renamed to {name}")
            if result_dir_name!='out/':
                shutil.move(actual_path+'out/', actual_path+result_dir_name)
                files = ['pcigale.ini','pcigale.ini.spec','cig_df.txt', result_dir_name]
            else:
                shutil.move(name, actual_path+'out/')
                files = ['pcigale.ini','pcigale.ini.spec','cig_df.txt',result_dir_name]
            
            move_files(actual_path, path_result, files)
            
            self._out_dir = result_dir_name
             
            self._path_result = path_result
            
        else:
            self._path_result = os.path.abspath(self.working_dir)
            
            self._out_dir = 'out/'


    def get_sample_spectra(self, lbda_sample = LBDA_SEDM, interp_kind = 'linear', box_ker_size=10, save_file = None, as_cube = False):
        """
        Get spectra fitted by Cigale in the wavelength space of your choice. 
        
        Parameters
        ----------
        lbda_sample : 1D array
            Wavelength space where you want to get the spectra.\n
            Default is SEDM wavelength space == np.linspace(3700,9300,220)

        interp_kind : string
            Interpolation method to get an analytical function of the cigale fitted spectra.\n
            Goes to scipy.interpolate.interp1d\n
            Default is linear. (Issue with cubic since some fitted points by cigale have same wavelength, which breaks the monotony of the fonction)
        
        box_ker_size : float
            Size in pixel of the door function. Used to convolve the interpolated datas.\n
            Default is 10

        save_file : string
            If not None (Default), where to save the sample spectra? Will save the corresponding wavelength too.
              
        as_cube : bool
            If True, return a pyifu 3D cube. You must have set the geometry of the images first\n
            Default is False

        Returns
        -------
        If as_cube,return a pyif.Cube
        If not as_cube, return 2 arrays 
        """
        kerbox = Box1DKernel( box_ker_size )       
        full_DF = self.clean_df.copy()     
        spec_data_cg=np.empty(shape=(len( full_DF )), dtype='object')
        spec_data_interp=np.zeros(shape=(len( full_DF ),len(lbda_sample)))       
        fitind=0           
        for i in range(len( full_DF )):          
            if i in ( self.idx_used ):               
                valid_spec = 'yes'               
                try:
                    datafile = fits.open( os.path.join(self._path_result,self._out_dir,f'{i}_best_model.fits'))
                    data = Table(datafile[1].data)                                  
                except:
                    valid_spec='no'
                    
                if valid_spec=='yes' :                                      
                    lbda_full = np.array(data['wavelength']) * 10 #nm  ->  AA
                    lbda = lbda_full[(lbda_full<lbda_sample[-1]+2000) & (lbda_full>lbda_sample[0]-2000)]
                    spec_data_cg[i] = flux_hz_to_aa(np.array(data['Fnu'])[(lbda_full<lbda_sample[-1]+2000) & (lbda_full>lbda_sample[0]-2000)] * 10**(-26), lbda) #mJy ->  erg.s^-1.cm^-2.Hz^-1  ->  erg.s^-1.cm^-2.AA^-1                     
                    f=interp1d(lbda, spec_data_cg[i], kind = interp_kind)
                    spec_data_interp[i] = convolve( f (np.linspace(lbda_sample[0],lbda_sample[-1],len(lbda_sample)*box_ker_size) ),kerbox,boundary='extend',normalize_kernel=True)[::10]
                                      
                else:                   
                    spec_data_interp[i] = np.array([np.nan] * len(lbda_sample))                   
                    spec_data_cg[i] = np.nan
                    
                fitind+=1
                
            else:
                spec_data_interp[i] = np.array([0] * len(lbda_sample))                
                spec_data_cg[i] = 0        
        self.spec_sample = spec_data_interp
        self.spec_lbda_sample = lbda_sample
        self.spec_cg = spec_data_cg

        if save_file is not None:
            np.savez(save_file, spec=spec_data_interp, lbda=lbda_sample)
        if as_cube:

            return
                    
        return(spec_data_interp,lbda_sample)


        ####### Miss Plots ######


    def clean_output(self):
        """
        THIS COMMAND WILL DELETE self.working_dir, BE CAREFUL \n
        Be sure you saved the cigale results before using this command.\n
        Go back to the initial path (_currentpwd) before the cigale run, 
        then clean the directory where cigale has been run.
        """
        if self.currentpwd == self.working_dir:
            return
        os.chdir( self._currentpwd)
        shutil.rmtree( self._working_dir)
        self._working_dir = None
        

    # -------- #
    #  SETTER  #
    # -------- #
    
    def set_nb_process(self, value):
        """
        Set quantity of core to use for the multiprocessing.
        """
        if value == 'auto':           
            import multiprocessing
            if multiprocessing.cpu_count()>2:
                self._nb_process = multiprocessing.cpu_count() - 2
            else:
                self._nb_process = 1
        else:
            self._nb_process = value

            
    def set_sedmodule_config(self, sedmodule, param, value):
        """
        Set sed modules parameters in config file
        """
        self.config['sed_modules_params'][sedmodule][param] = value
        self.config.write()

    @property
    def config(self):
        """
        Config file
        """
        if not hasattr(self, '_config'):
            return None
        return self._config

    @property
    def working_dir(self):
        """
        Working directory
        """
        if not hasattr(self, '_working_dir'):
            return None
        return self._working_dir

    @property
    def currentpwd(self):
        """
        Current path 
        """
        if not hasattr(self, '_currentpwd'):
            return None
        return self._currentpwd

   
    
    
# End of sedfitting.py ========================================================
