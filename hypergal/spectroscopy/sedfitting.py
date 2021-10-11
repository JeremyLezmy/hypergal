#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import warnings
import pandas
import numpy as np
from configobj import ConfigObj
import shutil

from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve

from glob import glob

from .. import _PACKAGE_ROOT
from . import utils


PS1_FILTER = ['ps1_g', 'ps1_r', 'ps1_i', 'ps1_z', 'ps1_y']
PS1_FILTER_err = ['ps1_g_err', 'ps1_r_err', 'ps1_i_err', 'ps1_z_err', 'ps1_y_err']


# = np.linspace(3700, 9300, 220)


ORDER = "ugrizy"
POS = {c: p for (p, c) in enumerate(ORDER)}

DEFAULT_CIGALE_MODULES = ['sfhdelayed', 'bc03', 'nebular', 'dustatt_modified_CF00', 'dale2014', 'redshifting']
CIGALE_CONFIG_PATH = os.path.join(_PACKAGE_ROOT, 'config/cigale.json')


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
            if col not in PS1_FILTER + PS1_FILTER_err and col not in [k.replace('_', '.') for k in PS1_FILTER + PS1_FILTER_err]:
                df.pop(col)

        lst = list(df.columns)
        lst.sort(key=lambda c: POS[c.split('_')[1][-1]])

        df = df[lst]
        df['redshift'] = np.array([self.redshift]*len(df))

        if self.FITTER_NAME in ['cigale']:
            df['id'] = df.index
            df = df.reindex(columns=(['id', 'redshift'] + list([a for a in df.columns if a not in ['id', 'redshift']])))

        self.set_clean_df(df)
        filt = [ele for ele in lst if ('err' not in ele)]
        self._filters = filt
        idx = df.loc[ np.logical_and.reduce([df[i].values / df[i + '_err'].values > self.snr for i in filt])].index
        df_threshold = df.loc[idx].copy()
        self.set_input_sedfitter(df_threshold)

        if tmp_inputpath is None:
            df_threshold.to_csv('in_sedfitter.txt', header=True, index=None, sep='\t', mode='w+')
            self._dfpath = os.path.abspath('in_sedfitter.txt')
        else:
            dirout = os.path.dirname(tmp_inputpath)
            if not os.path.isdir(dirout):
                os.makedirs(dirout, exist_ok=True)
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


class Cigale(SEDFitter):

    FITTER_NAME = "cigale"

    def __init__(self, dataframe, redshift, snr=None, setup=True, tmp_inputpath=None,
                     initiate=True, ncores="auto", working_dir=None, testmode=False, cubeouts=None):
        """ """
        _ = super().__init__(dataframe, redshift, snr=snr, setup=setup, tmp_inputpath=tmp_inputpath)
        if initiate:
            self.initiate_cigale(working_dir=working_dir, cores=ncores, testmode=testmode)
        self._cubeouts = cubeouts
    # ============= #
    #  Methods      #
    # ============= #
    @classmethod
    def from_cube_cutouts(cls, cubeouts, redshift, snr=3, in_unit="aa",
                              tmp_inputpath=None,
                              initiate=True, ncores="auto", working_dir=None, **kwargs):

        """  Initiate Cigale from cube of cutouts.

        Parameters
        ----------
        cubeouts: WCSCube
            Cube of cutouts.

        redshift: float
            Redshift of the host.

        snr: float -optional-
            Signal over noise ratio which will be use as threshold in pixel selection.

        in_unit: string -optional-
            Unit of given flux in cubeouts. Might be 'aa', 'hz' or 'mjy'.

        tmp_inputpath: string -optional-
            Temporary path where to save dataframe which will go in Cigale process.

        initiate: bool -optional-
            If True, initiate Cigale (file building) without running it.

        ncore: 'auto' or float -optional-
            Number of cores to use for the multiprocessing. \n
            If 'auto', will use number of available core -2 .

        working_dir: string -optional-
            Temporary path where Cigale will run and generate outputs.

        **kwargs -optional-
            Goes to __init__

        Returns
        -------

        """
        import pandas
        try:
            bands = [cubeouts.header[f"FILTER{i}"] for i in range(len(cubeouts.data))]
        except:
            raise TypeError("the given cube is not a cutout cube, no FILTER{i} entries in the header")
        
        cigale_bands = [b.replace(".", "_") for b in bands]
        #
        # Build the input dataframe
        pdict = cubeouts.to_pandas()
        
        df = pandas.concat({"data":pdict["data"], "variance":pdict["variance"]}, )
        
        df = pdict["data"].rename({k: v for k, v in enumerate(cigale_bands)}, axis=1)  # correct column names
        df_err = pandas.DataFrame(np.sqrt(pdict["variance"].values), index=df.index,
                                  columns=[k+"_err" for k in df.columns])  # errors and not variance
        #
        # good unit if necessary
        if in_unit is not None and in_unit != "mjy":
            if in_unit == 'aa':
                convertion = getattr(utils, f"flux_{in_unit}_to_mjy")(1, cubeouts.lbda)
            else:
                convertion = getattr(utils, f"flux_{in_unit}_to_mjy")(1)

            df *= convertion
            df_err *= convertion
        
        df = df.merge(df_err, right_index=True, left_index=True)  # combined them

        return cls(dataframe=df, redshift=redshift, snr=snr,
                   tmp_inputpath=tmp_inputpath,
                   initiate=initiate, ncores=ncores, working_dir=working_dir, cubeouts=cubeouts, **kwargs)

    def initiate_cigale(self, sed_modules='default', cores='auto', working_dir=None, testmode=False):
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

        self._currentpwd = os.getcwd()  # is absolute already

        if working_dir is not None:
            working_dir = os.path.abspath(working_dir)  # make it absolute
            if not os.path.isdir(working_dir):
                os.makedirs(working_dir, exist_ok=True)

            os.chdir(working_dir)

        else:
            working_dir = self._currentpwd

        self._working_dir = working_dir

        utils.command_cigale('init')
        config = ConfigObj('pcigale.ini', encoding='utf8', write_empty_values=True)

        config['data_file'] = self.dfpath

        if sed_modules == 'default':
            config['sed_modules'] = DEFAULT_CIGALE_MODULES

        elif type(sed_modules) != list and type(sed_modules[0]) != str:
            print('sed_modules should be a list of string.')

        else:
            config['sed_modules'] = sed_modules

        self.set_nb_process(cores)
        config['analysis_method'] = 'pdf_analysis'
        config['cores'] = self._nb_process
        config.write()

        utils.command_cigale('genconf')

        config = ConfigObj('pcigale.ini', encoding='utf8', write_empty_values=True)

        if sed_modules == 'default':
            with open(CIGALE_CONFIG_PATH) as data_file:
                params = json.load(data_file)
            if testmode:
                config['analysis_params']['save_best_sed'] = 'True'
            else:
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
            datafile = fits.open( os.path.join(self._path_result, self._out_dir, f'{id_}_best_model.fits'))
            data = Table(datafile[1].data).to_pandas()
        except:
            warnings.warn(f"Cannot read the corresponding output {id_}")
            return None if columns is None else pandas.DataFrame(columns=columns)

        if as_raw:
            return data

        data["wavelength"] *= 10  # nn-> AA
        data = data[data["wavelength"].between(*lbda_range)]
        if flux_units == "aa":
            data["Fnu"] *= utils.flux_mjy_to_aa(1, data["wavelength"])
            data.rename({"Fnu": "flux"}, axis=1, inplace=True)
        elif flux_units == "hz":
            data["Fnu"] *= utils.flux_mjy_to_hz(1)
            data.rename({"Fnu": "flux"}, axis=1, inplace=True)
        elif flux_units is not None and flux_units != "mjy":
            raise ValueError(f"Cannot parse the input flux_units {flux_units} ; aa, hz or mjy implemented")
        else:
            data.rename({"Fnu": "flux"}, axis=1, inplace=True)

        if columns is None:
            return data

        return data[columns]

    @staticmethod
    def cigale_as_lbda(cigale_df, lbda, interp_kind="linear", res=10):
        """ """
        f = interp1d(cigale_df["wavelength"].values,
                     cigale_df["flux"].values, kind=interp_kind)

        f_hres = f(np.linspace(lbda[0], lbda[-1], len(lbda)*res))
        kerbox = Box1DKernel(res)
        newflux = convolve(f_hres, kerbox, boundary='extend', normalize_kernel=True)[::res]
        return pandas.DataFrame({"wavelength": lbda, "flux": newflux})

    def get_sample_spectra(self, bestmodel_dir=None, lbda_sample=None, interp_kind='linear',
                           res=10, apply_sedm_lsf=True, client=None,
                           saveplot_rmspull=None, saveplot_intcube=None):
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
            bestmodel_dir = os.path.join(self._path_result, self._out_dir)

        cigale_output = np.sort(glob(os.path.join(bestmodel_dir, "*_best_model.fits")))

        # Build the datafile
        datafile = pandas.DataFrame(cigale_output, columns=["outputfile"])
        datafile["id"] = datafile["outputfile"].astype("str").str.split("/", expand=True
                                                              ).iloc[:,-1].astype("str").str.split("_", expand=True).iloc[:,0].rename({"0":"id"}).values
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
        data = [np.zeros(len(lbda_sample)) if id_ not in datafile.index else dflux.xs(id_).values
                for id_ in self.input_df.index]

        if apply_sedm_lsf:

            from pysedm.sedm import SEDM_LBDA
            sig = utils.sedm_lsf(SEDM_LBDA)
            newdatas = np.asarray(data)

            if client is not None:
                import dask.array as da
                from dask import delayed
                spd = []
                for id_ in range(newdatas.shape[0]):
                    fulldel = da.from_delayed(delayed(utils.gauss_convolve_variable_width)(newdatas[id_][None, ::], sig=sig, prec=10.), shape=(1, len(sig)), dtype='float')
                    spd.append(fulldel)

                ss = da.stack(spd)
                newd = client.compute(ss).result().squeeze()

                if saveplot_rmspull is not None:
                    data_in, data_out = self.get_data_inout(os.path.join(self.working_dir, 'out'))
                    rms,pull = self.get_rms_pull_df(data_in, data_out, self.input_df.index)
                    self.show_pull_rms_map(rms, pull, saveplot_rmspull)
                    intcube = self.cubeouts.get_new(newdata=newd.T, newlbda=lbda_sample, newvariance="None")
                    self.show_intcube(intcube, np.sort(self.cubeouts.lbda), data_in, data_out, saveplot_intcube)
                return newd.T, lbda_sample

            for id_ in range(newdatas.shape[0]):

                newdatas[id_, :] = utils.gauss_convolve_variable_width(newdatas[id_][None, ::], sig=sig, prec=100.)
                if saveplot_rmspull is not None:
                    data_in, data_out = self.get_data_inout(os.path.join(self.working_dir, 'out'))
                    rms,pull = self.get_rms_pull_df(data_in, data_out, self.input_df.index)
                    self.show_pull_rms_map(rms, pull, saveplot_rmspull)
                    intcube = self.cubeouts.get_new(newdata=newdatas.T, newlbda=lbda_sample, newvariance="None")
                    self.show_intcube(intcube, np.sort(self.cubeouts.lbda), data_in, data_out, saveplot_intcube)
            return newdatas.T, lbda_sample
        
        if saveplot_rmspull is not None:
            data_in, data_out = self.get_data_inout(os.path.join(self.working_dir, 'out'))
            rms,pull = self.get_rms_pull_df(data_in, data_out, self.input_df.index)
            self.show_pull_rms_map(rms, pull, saveplot_rmspull)
            intcube = self.cubeouts.get_new(newdata=np.asarray(data).T, newlbda=lbda_sample, newvariance="None")
            self.show_intcube(intcube, np.sort(self.cubeouts.lbda), data_in, data_out, saveplot_intcube)
        return np.asarray(data).T, lbda_sample

        ####### Miss Plots ######


    @staticmethod
    def get_data_inout(directory):

        file_in = fits.open(os.path.join(directory, 'observations.fits'))
        file_out = fits.open(os.path.join(directory, 'results.fits'))

        data_in = Table(file_in[1].data)
        data_out = Table(file_out[1].data)

        return data_in, data_out

    @staticmethod
    def get_rms_pull_df(data_in, data_out, idx):

        # Getting the filter names used with these data
        filterlist = data_in.colnames
        del filterlist[0:2]  # Remove id and redshift
        del filterlist[1::2]  # Remove *_err

        rms_df = pandas.DataFrame(columns=filterlist + ['Total'])
        pull_df = pandas.DataFrame(columns=filterlist + ['Total'])

        col_best_fit = [f'best.{i}' for i in filterlist]
        filtererrlist = [f'{i}_err' for i in filterlist]

        sni = 0
        for idx in idx:

            if idx in np.array(data_in['id']):

                filt_res = list((np.array(list(data_out[col_best_fit][sni])) - np.array(list(data_in[filterlist][sni])))/(np.array(list(data_in[filterlist][sni]))))
                tot_rms = np.sqrt((1/len(filt_res)) * np.nansum(np.array(filt_res)**2))

                filt_pull = list((np.array(list(data_out[col_best_fit][sni])) - np.array(list(data_in[filterlist][sni])))/(np.array(list(data_in[filtererrlist][sni]))))
                tot_pull = np.sqrt((1/len(filt_pull)) * np.nansum(np.array(filt_pull)**2))

                rms_df.loc[idx] = filt_res + [tot_rms]
                pull_df.loc[idx] = filt_pull + [tot_pull]
                sni += 1

            else :
                rms_df.loc[idx] = [np.nan]*len(rms_df.columns)
                pull_df.loc[idx] = [np.nan]*len(pull_df.columns)
        return rms_df, pull_df


    @staticmethod
    def show_pull_rms_map(rms_df, pull_df, saveplot=None,
                          pixel_bin=2, hist=False, arcsec_unit=False,
                          px_in_asrcsec=0.25, vmin=5, vmax=95, pull=True, dpi=100):

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), dpi=dpi, sharex=not hist, sharey=not hist,constrained_layout=True)

        filterlist = rms_df.iloc[:, 0:-1].copy()
        resmin = np.nanpercentile(filterlist, vmin)
        resmax = np.nanpercentile(filterlist, vmax)
        resbound = abs(max(abs(resmin), abs(resmax)))

        filterpulllist = pull_df.iloc[:, 0:-1].copy()
        pullmin = -5
        pullmax = 5
        pullbound = abs(max(abs(pullmin), abs(pullmax)))


        idx_thre  = rms_df[~np.isnan(rms_df['Total'])].index.values
        shape = int(138 + pixel_bin)

        if arcsec_unit:
            extent = [0,shape*px_in_asrcsec, 0, shape*px_in_asrcsec]
            centered_extent = [-shape*px_in_asrcsec/2, shape*px_in_asrcsec/2, -shape*px_in_asrcsec/2 , shape*px_in_asrcsec/2]
            unit = 'arcsec'
        else :
            extent = [-0.5,shape-0.5, -0.5, shape-0.5]
            centered_extent = [-shape/2, shape/2, -shape/2, shape/2]
            unit = 'px'
        for ax, filter in zip(axs.flat,filterlist):
            mean_rms = "{:6.4f}".format(np.sqrt( (1/len( idx_thre)) * np.nansum(filterlist[filter].values**2) ) )
            mean_pull = "{:6.4f}".format(np.sqrt( (1/len( idx_thre)) * np.nansum(filterpulllist[filter].values**2) ) )
            if hist:
                data_to_plot = filterlist[filter].values
                ax.hist(data_to_plot, bins=30, range=(resmin, resmax))
                ax.set(title=filter+' RMS='+mean_rms, xlabel='Residuals')

            if pull:
                imres=ax.imshow( np.rot90(np.reshape(pull_df[filter].values,(int(shape/pixel_bin), int(shape/pixel_bin)))), vmin=-2, vmax=2, cmap='coolwarm',origin='upper',extent = extent, aspect = 1)
                ax.set(title=filter, xlabel='x ('+unit+')', ylabel='y ( '+unit+')')
                ax.set_xlabel('x ( '+unit+')', fontsize=15)
                ax.set_ylabel('y ( '+unit+')', fontsize=15)
                ax.tick_params(labelsize=15)
                ax.set_xticks( ax.get_yticks())
                ax.set_title(filter, fontsize=15)
                ax.label_outer()
                ax.set_aspect('equal')

            else:
                imres=ax.imshow( np.rot90(np.reshape(rms_df[filter].values,(int(shape/pixel_bin), int(shape/pixel_bin)))), vmin=-resbound, vmax=resbound, cmap='seismic',origin='upper',extent = extent, aspect = 1)
                ax.set(title=filter+' RMS='+mean_rms, xlabel='x ( '+unit+')', ylabel='y ( '+unit+')')
                ax.label_outer()

        mean_rms = "{:6.4f}".format( np.sqrt( (1/len( idx_thre)) * np.nansum(rms_df['Total'].values**2) ) )
        mean_pull = "{:6.4f}".format( np.sqrt( (1/len( idx_thre)) * np.nansum(pull_df['Total'].values**2) ) )
        if hist:
            data_to_plot = rms_df['Total'].values
            axs[-1,-1].hist(data_to_plot, bins=30)
            axs[-1,-1].set(title=r' $ \sum $ filters RMS='+mean_rms, xlabel='Spectral RMS')

        else:
            imrms=axs[-1,-1].imshow(np.rot90(np.reshape(rms_df["Total"].values,(int(shape/pixel_bin), int(shape/pixel_bin)))),vmin=np.nanpercentile(rms_df["Total"].values, vmin), vmax=np.nanpercentile(rms_df["Total"].values, vmax),  cmap='inferno_r',origin='upper',extent = extent, aspect = 1)
            axs[-1,-1].set(title=fr' $ \sum $ filters RMS=' + mean_rms, xlabel='x ('+unit+')')
            axs[-1,-1].set_xlabel('x ('+unit+')', fontsize=15)
            axs[-1,-1].tick_params(labelsize=15)
            axs[-1,-1].set_title(fr' $ \sum $ filters RMS=' + mean_rms, fontsize=15)
            axs[-1,-1].set_aspect('equal')
            cbar_res = fig.colorbar(imres, ax=axs[0].ravel().tolist(), extend='both', label='Pull', aspect=50)
            cbar_rms = fig.colorbar(imrms, ax=axs[1].ravel().tolist(), extend='max', label='Spectral RMS', aspect=50)
            cbar_res.set_label('Pull', fontsize=13)
            cbar_rms.set_label('Spectral RMS', fontsize=13)
            cbar_rms.ax.tick_params(labelsize=13)
            cbar_res.ax.tick_params(labelsize=13)
            cbar_rms.set_label('Spectral RMS', labelpad=10)

        if saveplot is not None:
            fig.savefig(saveplot, dpi='figure', bbox_inches='tight')


    @staticmethod
    def show_intcube( intcube, lbda_bb, data_in, data_out, saveplot=None):

        from hypergal.spectroscopy import utils
        import matplotlib.pyplot as plt 
        filterlist = data_in.colnames

        del filterlist[0:2] #Remove id and redshift
        del filterlist[1::2] #Remove *_err
        filtererrlist = [f'{i}_err'  for i in filterlist]
        col_best_fit = [f'best.{i}'  for i in filterlist]
        filtererrlist = [f'{i}_err'  for i in filterlist]
        bestfit_integrated = utils.flux_mjy_to_aa(data_out[col_best_fit].to_pandas(),lbda_bb).sum().values/len(intcube.indexes)
        val_in_integrated = utils.flux_mjy_to_aa(data_in[filterlist].to_pandas(),lbda_bb).sum().values/len(intcube.indexes)
        err_in_integrated = np.sqrt(np.sum(utils.flux_mjy_to_aa(data_in[filtererrlist].to_pandas(),lbda_bb).values**2, axis=0))/len(intcube.indexes)

        fig = plt.figure(figsize=(12,4), dpi=150)
        gs = fig.add_gridspec(1, 3)
        axim = fig.add_subplot(gs[0, 2])
        axspec = fig.add_subplot(gs[0, 0:2])


        axspec.plot( intcube.lbda, np.mean(intcube.data, axis=1),label="Mean spectrum")
        #ax.plot(cutouts.lbda,np.mean(intcube.data, axis=0)[~ (np.mean(intcube.data, axis=0)==0)])
        axspec.scatter(lbda_bb, bestfit_integrated, color='r',s=49, label='Photo. cigale outputs', zorder=10)
        axspec.scatter(lbda_bb, val_in_integrated, s=49,  facecolors='none',edgecolor='b', label='Photo. PS1 inputs', zorder=11 )
        axspec.errorbar(lbda_bb, val_in_integrated, err_in_integrated, fmt='none', color='b', zorder=12)
        axspec.legend()
        axspec.set_xlabel(r'Wavelength($\AA$)')
        axspec.set_ylabel(r'Flux ($erg.s^{-1}.cm^{-2}.\AA^{-1}$)')

        intcube._display_im_(axim=axim, rasterized=False)
        #ax.scatter(x,y, c='k', marker='D', s=4)
        axim.set_aspect('equal')
        axim.set_xlabel('x(spx)')
        axim.set_ylabel('y(spx)')
        axim.yaxis.tick_right()
        axim.yaxis.set_label_position("right")
        if saveplot is not None:
            fig.savefig(saveplot, dpi='figure', bbox_inches='tight') 

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

        os.chdir(self._currentpwd)
        shutil.rmtree(self._working_dir)
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

    @property
    def cubeouts(self):
        return self._cubeouts
                    
