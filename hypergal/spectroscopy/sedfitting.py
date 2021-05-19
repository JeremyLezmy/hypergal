#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import warnings
import pandas
import numpy as np
from configobj import ConfigObj


from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve

from glob import glob

from .. import _PACKAGE_ROOT
from . import utils



PS1_FILTER=['ps1_g', 'ps1_r', 'ps1_i', 'ps1_z', 'ps1_y']
PS1_FILTER_err=['ps1_g_err','ps1_r_err', 'ps1_i_err', 'ps1_z_err', 'ps1_y_err']


# = np.linspace(3700, 9300, 220)



ORDER = "ugrizy"
POS = {c:p for (p, c) in enumerate(ORDER)}

DEFAULT_CIGALE_MODULES = ['sfhdelayed', 'bc03', 'nebular', 'dustatt_modified_CF00', 'dale2014', 'redshifting']
CIGALE_CONFIG_PATH  = os.path.join(_PACKAGE_ROOT,'config/cigale.json')


class SEDFitter():
    
    FITTER_NAME = "unknown"
    
    def __init__(self, dataframe, redshift, snr=None, setup=True, tmp_inputpath=None):
        """ 
        Initiate SEDFitter with a dataframe.

        Parameters
        ----------
        dataframe: Pandas.Dataframe
            Dataframe with flux data and flux errors for each filter you want to use.

        redshift: float -optional-
            Redshift of the object. Will be the same for each row of the dataframe

        snr: float -optional-
            Threshold Signal over Noise ratio for all the bands, for which the pixel won't be selected for the sedfitting. 
        Returns
        --------
        """        
        self.set_dataframe(dataframe)
        self.set_snr(snr)
        self.set_redshift(redshift)
        if setup:
            self.setup_df(tmp_inputpath=tmp_inputpath)
        
    def setup_df(self, tmp_inputpath=None):
        
        """ 
        Make the input dataframe compatible with the sedfitter.

        Parameters
        ----------
        tmp_inputpath: string
            Where you want to store the dataframe which will be the input of the sedfitter.\n
            For Cigale must be .txt file.

        which: string]
            Which sedfitter are you using. \n
            If Cigale (default), it will reorder the columns, and add one with ID for each pixel.
           
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

        if self.FITTER_NAME in ['cigale']:
            df['id']=df.index        
            df = df.reindex(columns=(['id','redshift'] + list([a for a in df.columns if a not in ['id','redshift']]) ))

        self.set_clean_df(df)
        filt = [ele for ele in lst  if ('err' not in ele)]
        self._filters = filt
        idx = df.loc[ np.logical_and.reduce([ df[i].values / df[i + '_err'].values > self.snr for i in filt])].index
        df_threshold = df.loc[idx].copy()
        self.set_input_sedfitter(df_threshold)
               
        if tmp_inputpath is None:
            df_threshold.to_csv('in_sedfitter.txt', header=True, index=None, sep='\t', mode='w+')
            self._dfpath = os.path.abspath('in_sedfitter.txt')
        else:
            dirout = os.path.dirname(tmp_inputpath)
            if not os.path.isdir(dirout) :
                os.makedirs(dirout, exist_ok = True)
            df_threshold.to_csv(tmp_inputpath, header=True, index=None, sep='\t', mode='w+')
            self._dfpath = os.path.abspath(tmp_inputpath)
        
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
    

class Cigale( SEDFitter ):

    FITTER_NAME = "cigale"

    def __init__(self, dataframe, redshift, snr=None, setup=True, tmp_inputpath=None,
                     initiate=True, ncores="auto", working_dir=None):
        """ """
        _ = super().__init__(dataframe, redshift, snr=snr, setup=setup, tmp_inputpath=tmp_inputpath)
        if initiate:
            self.initiate_cigale(working_dir=working_dir, cores=ncores)
            
    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def from_cube_cutouts(cls, cubeouts, redshift, snr=3, in_unit="aa",
                              tmp_inputpath=None,
                              initiate=True, ncores="auto", working_dir=None, **kwargs):
        """ 
        **kwargs goes to __init__
           -> setup
        """
        import pandas
        try:
            bands = [cubeouts.header[f"FILTER{i}"] for i in range(len(cubeouts.data))]
        except:
            raise TypeError("the given cube is not a cutout cube, no FILTER{i} entries in the header")
        
        cigale_bands = [b.replace(".","_") for b in bands]
        #
        # Build the input dataframe
        pdict = cubeouts.to_pandas()
        
        df = pandas.concat({"data":pdict["data"], "variance":pdict["variance"]}, )
        
        df = pdict["data"].rename({k:v for k,v in enumerate(cigale_bands)}, axis=1) # correct column names
        df_err = pandas.DataFrame(np.sqrt(pdict["variance"].values), index=df.index, 
                columns=[k+"_err" for k in df.columns]) # errors and not variance
        #
        # good unit if necessary
        if in_unit is not None and in_unit != "mjy":
            if in_unit == 'aa':
                convertion = getattr(utils,f"flux_{in_unit}_to_mjy")(1, cubeouts.lbda)
            else:
                convertion = getattr(utils,f"flux_{in_unit}_to_mjy")(1)
                
            df *= convertion
            df_err *= convertion
        
        df = df.merge(df_err, right_index=True, left_index=True) # combined them
        
        return cls(dataframe=df, redshift=redshift, snr=snr,
                       tmp_inputpath=tmp_inputpath,
                       initiate=initiate, ncores=ncores, working_dir=working_dir,**kwargs)
        
        
    def initiate_cigale(self, sed_modules='default', cores='auto', working_dir=None ):
        """  Initiate Cigale (not launch yet)

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

        self._currentpwd = os.getcwd() # is absolute already
        
        if working_dir is not None:
            working_dir = os.path.abspath(working_dir) # make it absolute
            if not os.path.isdir(working_dir) :
                os.makedirs(working_dir, exist_ok = True)
                
            os.chdir(working_dir)
            
        else:
            working_dir = self._currentpwd

        self._working_dir = working_dir
            
        utils.command_cigale('init')
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
        
        utils.command_cigale('genconf')

        config = ConfigObj('pcigale.ini', encoding='utf8', write_empty_values=True)

        if sed_modules=='default':
            with open( CIGALE_CONFIG_PATH ) as data_file:
                params = json.load(data_file)
                
            config = utils.update_config(config, params)
            
        config['sed_modules_params'][[k for k in config['sed_modules_params'].keys() if 'dustatt' in k][0]]['filters'] = ' & '.join(ele for ele in config['bands']  if ('err' not in ele))
        
        config.write()            
        self._config = config


    def run(self, path_result=None, result_dir_name='out/'):
        """
        Run Cigale.\n
        Results will be stored in self._working_dir, and a directory 'out/' will be created.
        """
        
        utils.command_cigale('run')

        #### Should be Removed or at least reworked #####
        if path_result is not None:
            import datetime
            warnings.warn("path_result is None. DEPRECATED")
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
            
            utils.move_files(actual_path, path_result, files)
            
            self._path_result = path_result            
            self._out_dir = result_dir_name

        else:
            self._path_result = os.path.abspath(self.working_dir)
            self._out_dir = 'out/'

        return os.path.join(self._path_result, self._out_dir)
    
    def read_cigale_specout(self, id_, as_raw=False, lbda_range=[1000, 12000], 
                            columns=None, flux_units="aa"):
        """ """
        try:
            datafile = fits.open( os.path.join(self._path_result,self._out_dir,f'{id_}_best_model.fits'))
            data = Table(datafile[1].data).to_pandas()
        except:
            warnings.warn(f"Cannot read the corresponding output {id_}")
            return None if columns is None else pandas.DataFrame(columns=columns)

        if as_raw:
            return data

        data["wavelength"] *= 10 # nn-> AA
        data = data[data["wavelength"].between(*lbda_range)]
        if flux_units == "aa":
            data["Fnu"] *= utils.flux_mjy_to_aa(1, data["wavelength"])
            data.rename({"Fnu":"flux"}, axis=1,inplace=True)
        elif flux_units == "hz":
            data["Fnu"] *= utils.flux_mjy_to_hz(1)
            data.rename({"Fnu":"flux"}, axis=1,inplace=True)
        elif flux_units is not None and flux_units != "mjy":
            raise ValueError(f"Cannot parse the input flux_units {flux_units} ; aa, hz or mjy implemented")
        else:
            data.rename({"Fnu":"flux"}, axis=1,inplace=True)

        if columns is None:
            return data

        return data[columns]

    @staticmethod
    def cigale_as_lbda(cigale_df, lbda, interp_kind="linear", res=10):
        """ """
        f = interp1d(cigale_df["wavelength"].values,
                     cigale_df["flux"].values, kind=interp_kind)
        
        f_hres  =  f(np.linspace(lbda[0], lbda[-1], len(lbda)*res) )
        kerbox  = Box1DKernel( res )
        newflux = convolve( f_hres, kerbox, boundary='extend', normalize_kernel=True)[::res]
        return pandas.DataFrame({"wavelength":lbda, "flux":newflux})

    def get_sample_spectra(self, bestmodel_dir=None, lbda_sample=None, interp_kind='linear', res=10, client=None):
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
              
        Returns
        -------
        If as_cube,return a pyif.Cube
        If not as_cube, return 2 arrays 
        """
        if lbda_sample is None:
            from pysedm.sedm import SEDM_LBDA 
            lbda_sample = SEDM_LBDA

        # find files
        if bestmodel_dir is None:
            bestmodel_dir =  os.path.join(self._path_result, self._out_dir)
            
        cigale_output = np.sort(glob( os.path.join(bestmodel_dir,"*_best_model.fits") ))
        
        # Build the datafile
        datafile = pandas.DataFrame(cigale_output, columns=["outputfile"])
        datafile["id"] = datafile["outputfile"].astype("str").str.split("/", expand=True
                                                              ).iloc[:,-1].astype("str").str.split("_", expand=True).iloc[:,0].rename({"0":"id"}, axis=1)
        datafile["id"] = datafile["id"].astype("int")
        datafile = datafile.set_index("id").sort_index()

        #
        # Build the Output DataFrame and convolve at the requested lbda_sample
        if client is not None:
            from dask import delayed
            d_dout = {}
            for id_ in datafile.index:
                cigale_dl = delayed(self.read_cigale_specout)(id_, columns=["wavelength", "flux"])
                d_dout[id_] = delayed(self.cigale_as_lbda)(cigale_dl, lbda_sample, interp_kind=interp_kind, res=res)

            dictdout = client.compute(d_dout).result()
        else:
            dictdout = {k: self.cigale_as_lbda( self.read_cigale_specout(k, columns=["wavelength", "flux"]),
                                            lbda_sample, interp_kind=interp_kind, res=res)
                                            for k in datafile.index}
                                                        
         
        dflux = pandas.concat(dictdout)["flux"]
        
        #
        # get the data
        data = [np.zeros( len(lbda_sample) ) if id_ not in datafile.index else dflux.xs(id_).values
                for id_ in self.input_df.index]
            
        return np.asarray(data).T, lbda_sample
        



        ####### Miss Plots ######


    def clean_output(self):
        """
        THIS COMMAND WILL DELETE self.working_dir, BE CAREFUL \n
        Be sure you saved the cigale results before using this command.\n
        Go back to the initial path (_currentpwd) before the cigale run, 
        then clean the directory where cigale has been run.
        """
        if self.currentpwd == self.working_dir:
            warnings.warn("currentpwd is the same as working dir. Cannot remove it. Nothing done")
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

   
