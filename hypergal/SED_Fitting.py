#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          SED_Fitting.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: rlezmy $
# Created on:        $Date: 2021/01/21 14:40:25 $
# Modified on:       2021/04/16 20:00:48
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
from configobj import ConfigObj
from astropy import constants
import shutil
from datetime import datetime
import multiprocessing as mp
from astropy.io import fits
from astropy.table import Table
import numpy as np
from pcigale import init, genconf, check, run
from pcigale.session.configuration import Configuration



PS1_FILTER_CIGALE=['ps1_g','ps1_r', 'ps1_i', 'ps1_z', 'ps1_y']
PS1_FILTER_CIGALE_err=['ps1_g_err','ps1_r_err', 'ps1_i_err', 'ps1_z_err', 'ps1_y_err']
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



PS1_FILTER_CIGALE=['ps1_g','ps1_r', 'ps1_i', 'ps1_z', 'ps1_y']
PS1_FILTER_CIGALE_err=['ps1_g_err','ps1_r_err', 'ps1_i_err', 'ps1_z_err', 'ps1_y_err']
ORDER = "ugrizy"
POS = {c:p for (p, c) in enumerate(ORDER)}
import geopandas
from configobj import ConfigObj
lbda_sedm = np.linspace(3700,9300,220)
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve

class Cigale_sed():
    
    def __init__(self, dataframe ):
        
        if type(dataframe)==geopandas.geodataframe.GeoDataFrame:       
            self.dataframe = pd.DataFrame(dataframe)
        else:
            self.dataframe = dataframe

    
    def setup_cigale_df(self, SNR=4, SNR_filt='all', redshift=0.02, path_to_save = None):
        
        cig_df = self.dataframe.copy()
        
        for col in cig_df:

            if col not in PS1_FILTER_CIGALE + PS1_FILTER_CIGALE_err  :

                cig_df.pop(col)
                
        lst=list(cig_df.columns)
        lst.sort(key = lambda c: POS[c.split('_')[1][-1]])
        cig_df = cig_df[lst]
            
        cig_df['redshift']=np.array([redshift]*len(cig_df))
        cig_df['id']=cig_df.index
        
        cig_df = cig_df.reindex(columns=(['id','redshift'] + list([a for a in cig_df.columns if a not in ['id','redshift']]) ))

        if SNR_filt == 'all' :
            
            filt = [ele for ele in lst  if ('err' not in ele)]
        else:
            filt = SNR_filt
        
        idx = cig_df.loc[ np.logical_and.reduce([ cig_df[i].values / cig_df[i + '_err'].values > SNR for i in filt])].index

        cig_df_threshold = cig_df.loc[idx].copy()
        
        if path_to_save==None:
            cig_df_threshold.to_csv('cig_df.txt', header=True, index=None, sep='\t', mode='w+')
            self._dfpath = 'cig_df.txt'
        else:
            cig_df_threshold.to_csv(path_to_save, header=True, index=None, sep='\t', mode='w+')
            self._dfpath = path_to_save
        
        self.cig_df = cig_df
        self.cig_df_threshold = cig_df_threshold
        self._idx_underThreshold = idx
        
        
    def initiate_cigale(self, dfpath='default', sed_modules='default', cores='auto' ):
    
        command_cigale('init')
        config = ConfigObj('pcigale.ini', encoding='utf8', write_empty_values=True)
        
        if dfpath=='default':
            config['data_file'] = self._dfpath
        else:
            config['data_file'] = dfpath
        
        if sed_modules=='default':
            config['sed_modules'] = ['sfhdelayed', 'bc03', 'nebular', 'dustatt_powerlaw', 'dale2014', 'redshifting']
        
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
        
        if 'sfhdelayed' in config['sed_modules_params']:
            config['sed_modules_params']['sfhdelayed']['tau_main'] = ['250', '500', '1000', '2000.0', '4000', '6000', '8000']
            config['sed_modules_params']['sfhdelayed']['age_main'] = ['250', '500', '1000', '2000', '4000', '8000', '10000', '12000']
            
        if 'bc03' in config['sed_modules_params']:
        
            config['sed_modules_params']['bc03']['imf'] = '1'
            config['sed_modules_params']['bc03']['metallicity'] = ['0.0001', '0.0004', '0.004', '0.008', '0.02', '0.05']
            
        if 'dustatt_powerlaw' in config['sed_modules_params']:
            
            config['sed_modules_params']['dustatt_powerlaw']['Av_young'] = ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0']
            config['sed_modules_params']['dustatt_powerlaw']['uv_bump_amplitude'] = ['0.0', '1.0', '2.0', '3.0']
            
            config['sed_modules_params']['dustatt_powerlaw']['filters'] = ' & '.join(ele for ele in config['bands']  if ('err' not in ele))
            
        if 'dale2014' in config['sed_modules_params']:
            
            config['sed_modules_params']['dale2014']['alpha'] = ['0.5', '1.0', '1.5', '2.0', '2.5']
        
        config['analysis_params']['redshift_decimals'] = '3'
        config['analysis_params']['save_best_sed'] = 'True'
        
        config.write()
            
        self.config = config
        
        
    def run_cigale(self, path_result=None, result_dir_name='out/'):
        
        
        command_cigale('run')
        
        if path_result is not None:
            actual_path = os.getcwd()+'/'
            if os.path.exists(path_result+ result_dir_name):
                name = path_result+datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '_'+ result_dir_name
                os.rename(path_result+result_dir_name, name)
                print(f"The {result_dir_name} directory already exists, the old one was renamed to {name}")
            os.rename(actual_path+'out/', result_dir_name)
            files = ['pcigale.ini', 'pcigale.ini.spec','cig_df.txt', result_dir_name]
            move_files(actual_path, path_result, files)
            
            self._path_result = path_result
        else:
            self._path_result = ''
        self._out_dir = result_dir_name
        
    def get_Sample_spectra(self, lbda_sample = lbda_sedm, interp_kind = 'linear', box_ker_size=10, save_dirout_data = None):

       
        kerbox = Box1DKernel( box_ker_size )       

        full_DF = self.cig_df

       
        spec_data_cg=np.empty(shape=(len( full_DF )), dtype='object')
        spec_data_interp=np.zeros(shape=(len( full_DF ),len(lbda_sample)))
        

        fitind=0           
        for i in range(len( full_DF )):
          
            if i in ( cg._idx_underThreshold ):
                
                valid_spec = 'yes'
                
                try:
                    datafile = fits.open(self._path_result+self._out_dir+f'{i}'+'_best_model.fits')
                    data = Table(datafile[1].data)
                                   
                except:
                    valid_spec='no'
                
                    
                if valid_spec=='yes' :                   
                    
                    lbda_full = np.array(data['wavelength']) * 10 #nm  ->  AA
                    lbda = lbda_full[(lbda_full<lbda_sample[-1]+2000) & (lbda_full>lbda_sample[0]-2000)]
                    spec_data_cg[i] = flux_hz_to_aa(np.array(data['Fnu'])[(lbda_full<lbda_sample[-1]+2000) & (lbda_full>lbda_sample[0]-2000)] * 10**(-26), lbda) #mJy ->  erg.s^-1.cm^-2.Hz^-1
                      
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
    
    
    
    
    def get_res_rms_df(self):
        
        if self._path_result == '':

            file_in = fits.open(self._out_dir+'observations.fits')
            file_out = fits.open(self._out_dir+'results.fits')
            
        else:
            file_in = fits.open(self._path_result+self._out_dir+'observations.fits')
            file_out = fits.open(self._path_result+self._out_dir+'results.fits')
            
        data_in = Table(file_in[1].data)
        data_out = Table(file_out[1].data)

        #Getting the filter names used with these data
        filterlist = data_in.colnames
        del filterlist[0:2] #Remove id and redshift
        del filterlist[1::2] #Remove *_err

        rms_df = pd.DataFrame( columns= filterlist + ['Total'])

        col_best_fit = [f'best.{i}'  for i in filterlist]
        filtererrlist = [f'{i}_err'  for i in filterlist]

        sni = 0
        for idx in range(len( cg.cig_df)):

            if idx in self.cig_df_threshold.index:
                
                filt_res =     list(  ( np.array(list(data_out[col_best_fit][sni])) - np.array(list(data_in[filterlist][sni])) )/(np.array(list(data_in[filterlist][sni])))  )
                tot_rms = np.sqrt( (1/len(filt_res)) * np.nansum(np.array(filt_res)**2))
                
                rms_df.loc[idx] = filt_res + [tot_rms]
                sni+=1

            else :
                rms_df.loc[idx] = [np.nan]*len(rms_df.columns)

        self.rms_df = rms_df
        
        return rms_df


    def show_rms(self, pixel_bin=2, hist=False, arcsec_unit=True, px_in_asrcsec=0.25, vmin=5, vmax=95, savepath=None):
        '''
        Plot the residual images for each filters and the total RMS

        Parameters
        
        Options
        -------
       

        Returns
        -------
        '''

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8,5), sharex=not hist, sharey=not hist,constrained_layout=True)

        filterlist = self.rms_df[:-2]
        resmin = np.nanpercentile(filterlist , vmin)
        resmax = np.nanpercentile(filterlist, vmax )
        resbound = abs(max(abs(resmin),abs(resmax)))
        
        shape = int(np.max( cg.dataframe['centroid_x'])-np.min( cg.dataframe['centroid_x']) + pixel_bin)
        
        if arcsec_unit:
            extent = [0,shape*px_in_asrcsec, 0,shape*px_in_asrcsec]
            centered_extent = [-shape*px_in_asrcsec/2, shape*px_in_asrcsec/2, -shape*px_in_asrcsec/2 , shape*px_in_asrcsec/2]
            unit = 'arcsec'
        else :
            extent = [0,shape , 0,shape]
            centered_extent = [-shape/2, shape/2, -shape/2, shape/2]
            unit = 'px'
        for ax, filter in zip(axs.flat,filterlist):
            mean_rms = "{:6.4f}".format(np.sqrt( (1/len( cg._idx_underThreshold)) * np.nansum(filterlist[filter].values**2) ) )
            if hist:
                data_to_plot = dict_res[filter].flat
                ax.hist(data_to_plot, bins=30, range=(resmin, resmax))
                ax.set(title=filter+' RMS='+mean_rms, xlabel='Residuals')
            else:
                imres=ax.imshow(np.reshape(self.rms_df[filter].values,(int(shape/pixel_bin), int(shape/pixel_bin))), vmin=-resbound, vmax=resbound, cmap='seismic',origin='lower',extent = extent, aspect = 1)
                ax.set(title=filter+' RMS='+mean_rms, xlabel='x (in '+unit+')', ylabel='y (in '+unit+')')
                ax.label_outer()

        mean_rms = "{:6.4f}".format( np.sqrt( (1/len( self._idx_underThreshold)) * np.nansum(filterlist['Total'].values**2) ) )
        if hist:
            data_to_plot = tot_rms.flat
            axs[-1,-1].hist(data_to_plot, bins=30, range=(rmsmin, rmsmax))
            axs[-1,-1].set(title=r' $ \sum $ $\alpha$filters RMS='+mean_rms, xlabel='Spectral RMS')
        else:
            imrms=axs[-1,-1].imshow(np.reshape(self.rms_df["Total"].values,(int(shape/pixel_bin), int(shape/pixel_bin))),vmin=np.nanpercentile(self.rms_df["Total"].values, vmin), vmax=np.nanpercentile(self.rms_df["Total"].values, vmax),  cmap='inferno_r',origin='lower',extent = extent, aspect = 1)
            axs[-1,-1].set(title=fr' $ \sum $ filters RMS=' + mean_rms, xlabel='x (in '+unit+')')
            cbar_res = fig.colorbar(imres, ax=axs[0].ravel().tolist(),extend='both', label='Residuals')
            cbar_rms = fig.colorbar(imrms, ax=axs[1].ravel().tolist(),extend='max', label='Spectral RMS')
            

        fig.suptitle('Residuals & RMS using CIGALE', fontsize=17)
        #plt.show()
        #if savepath != None:
        #    filename = 'RMS_hist' if hist else 'RMS'
        #    fig.savefig(savepath+filename, facecolor = 'white',transparent=False)
        #    print(savepath+filename+' saved')
        return fig,ax


        
    def set_sedmodule_config(self, sedmodule, param, value):
        
            self.config[sedmodule][param] = value
            self.config.write()
               
        
        
    def set_nb_process(self, value):

        if value == 'auto':
            
            import multiprocessing
            self._nb_process = multiprocessing.cpu_count() - 2
            
        else:
            self._nb_process = value



def command_cigale(command, file_path=None):
    '''
    Call pcigale commands through python function rather than shell commands.
    Note that the run command requires to work in current terminal directory,
       data, config and results files are moved back after the operation.

    Parameters
    ----------
    command : string
        Available pcigale command are 'init', 'genconf', 'check', 'run'

    Options
    -------
    file_path : string
        Path to data, config and result files, if different from the current directory.
        Default is None.
    '''

    configfile=''
    if file_path != None:
        configfile += file_path
    configfile += 'pcigale.ini' #The configfile MUST have this name.

    if sys.version_info[:2] < (3, 6):
        raise Exception(f"Python {sys.version_info[0]}.{sys.version_info[1]} is"
        f" unsupported. Please upgrade to Python 3.6 or later.")

    # We set the sub processes start method to spawn because it solves
    # deadlocks when a library cannot handle being used on two sides of a
    # forked process. This happens on modern Macs with the Accelerate library
    # for instance. On Linux we should be pretty safe with a fork, which allows
    # to start processes much more rapidly.
    
    if sys.platform.startswith('linux'):
        mp.set_start_method('fork')
    else:
        mp.set_start_method('spawn', force=True)

    config = Configuration(configfile)

    if command == 'init':
        init(config)
    elif command == 'genconf':
        genconf(config)
    elif command == 'check':
        check(config)
    elif command == 'run':

        if file_path != None:
            #pcigale run command requires the data and config files to be in the
            # directory where the command is called.
            # We move these files before the run, then move them and results back.
            actual_path = os.getcwd()+'/'
            files = ['pcigale.ini', 'pcigale.ini.spec', 'test.mag']
            move_files(file_path, actual_path, files)

        run(config)

        if file_path != None:
            if os.path.exists(file_path+'out/'):
                name = file_path+datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '_out/'
                os.rename(file_path+'out/', name)
                print(f"The out/ directory already exists, the old one was renamed to {name}")
            files = ['pcigale.ini', 'pcigale.ini.spec', 'test.mag', 'out/']
            move_files(actual_path, file_path, files)

    else :
        print(f'Command \'{command}\' was not recognized. Available commands are'
              f' \'init\', \'genconf\', \'check\' and \'run\'')




#class SED_Fitting_to_cube():

#    def __init__( self, spec, lbda, geo):

#        self.spec_data = spec
#        self.lbda = lbda
        




# End of SED_Fitting.py ========================================================
