#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          Host_removing.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: rlezmy $
# Created on:        $Date: 2021/01/28 16:26:31 $
# Modified on:       2021/02/01 11:37:58
# Copyright:         2019, Jeremy Lezmy
# $Id: Host_removing.py, 2021/01/28 16:26:31  JL $
################################################################################

"""
.. _Host_removing.py:

Host_removing.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/28 16:26:31'
__adv__ = 'Host_removing.py'

import os
import sys
import datetime
from scipy import optimize

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

from hypergal import SED_Fitting as sedfit
from hypergal import Panstarrs_target as ps1targ
from hypergal import sedm_target as sedtarg
from hypergal import geometry_tool as geotool


from collections import OrderedDict
from hypergal import PSF_kernel as psfker
from hypergal import intrinsec_cube as hostmodel
from iminuit import Minuit


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
    
    def add_param(self, names, values=None, bounds=(None,None)):
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
        for (name, value, bound) in zip(names, values, bounds):
            
            self.params.update({name:value})
            self.bounds.update({name:bound})
        
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

IFU_ratio = 2.12
default_airmass_bounds = (1,5)


default_fixed_params=['temperature', 'relathumidity', 'pressure', 'lbdaref']
        

class Host_removing():

    
    def __init__(self, sedm_target, scene, target_pixcoord_image ):
        """ 
        Parameters:

        sedm_target : Instance of sedm_target()
        scene :  Instance of intrinsec_cube()
        target_pixcoord_image: Position in pixel of the target in the photometric image. Format list/array with size 2. 
        """

        self.sedm = sedm_target
        self.sedm_cube = sedm_target.cube_cal        
        self.scene = scene       
        self.target_imagecoord =  target_pixcoord_image
        
        self.hexagrid = scene.hexagrid

        self.set_parameters_values_init()
        self.set_parameters_bounds()
        
        

        

    def get_sedm_data(self, lbda_ranges=None, metaslices=None):

        if lbda_ranges is not None or metaslices is not None:
            
            self.set_lbda_ranges(lbda_ranges)
            self.set_metaslices(metaslices)
            
            lbda_stepbin = self.lbda_step_bin(self.lbda_ranges, self.metaslices)
            
            binned_sedm_data = np.zeros( (metaslices, np.shape(self.sedm_cube.data)[-1] ))
            binned_sedm_var = np.zeros( (metaslices, np.shape(self.sedm_cube.variance)[-1] ))
            binned_lbda = np.zeros( metaslices )
        
            for (i,j) in enumerate( lbda_stepbin ):
                        
                slice_obj = self.sedm_cube.get_slice(lbda_min=j[0], lbda_max=j[1], slice_object=True)
            
                binned_sedm_data[i,:] = slice_obj.data
                binned_sedm_var[i,:] = slice_obj.variance
            
                binned_lbda[i] = np.mean ( self.sedm_cube.lbda[  (self.sedm_cube.lbda>=j[0]) & (self.sedm_cube.lbda<j[1]) ] )

            self.binned_sedm_data = binned_sedm_data
            self.binned_sedm_var = binned_sedm_var
            self.binned_lbda = binned_lbda

            return ( binned_sedm_data, binned_sedm_var, binned_lbda )

        else:

            return (self.sedm_cube.data, self.sedm_var, self.sedm_cube.lbda)


    
    def get_residual_cube(self, full_cube = True ):

        if not hasattr(self.scene, 'Model_cube'):
            
            print( bcolors.WARNING + " You first have to build a model with self.scene.Build_model_cube(), see Intrinsec_cube class  " + bcolors.ENDC)

            return
        
        if full_cube == True:

            if np.shape(self.scene.Model_cube.data) != np.shape(self.sedm_cube.data):

                print( bcolors.WARNING + " Shapes don't match between sedm data and model data. You should first build the model with the same wavelength bins" + bcolors.ENDC)

                return
            
            else:
            
                cuberesidu=pyifu.spectroscopy.get_cube(data =(self.sedm_cube.data - self.scene.Model_cube.data), lbda = self.sedm_cube.lbda, spaxel_mapping = self.scene.pixmapping)

        else :
            
            if np.shape(self.scene.Model_cube.data) != np.shape(self.binned_sedm_data):

                print( bcolors.WARNING + " Shapes don't match between sedm data and model data. You should first build the model with the same wavelength bins" + bcolors.ENDC)

                return
            

            cuberesidu=pyifu.spectroscopy.get_cube(data =(self.binned_sedm_data - self.scene.Model_cube.data), lbda = self.binned_lbda, spaxel_mapping = self.scene.pixmapping)

        spax_vertices = np.array([[ 0.19491447,  0.6375365 ], [-0.45466557,  0.48756913], [-0.64958004, -0.14996737], [-0.19491447, -0.6375365 ], [ 0.45466557, -0.48756913], [ 0.64958004,  0.14996737]])
        
        cuberesidu.set_spaxel_vertices( spax_vertices )
        
        
        self.cuberesidu = cuberesidu
        return(cuberesidu)



    

    def evaluate_model(self, parameters, fix_parameters, lbda_ranges, metaslices, use_bin_data, nb_process):

        if ('x0_IFU' in fix_parameters) and ('y0_IFU' in fix_parameters):
            IFU_coord = list(self._init_IFU_target.values())
        else:
            IFU_coord = [parameters[k] for k in ['x0_IFU','y0_IFU']]
        psfparam = {k:parameters[k] for k in (self.scene.psfmodel.params.keys() - fix_parameters) }
        adrparam = {k:parameters[k] for k in (self.scene.adr.data.keys() - fix_parameters) }
        
        
        update_hexagrid = geotool.get_cube_grid( self.sedm_cube, scale = IFU_ratio, targShift=IFU_coord, x0=self.target_imagecoord[0] , y0=self.target_imagecoord[1]  )

        self.scene.set_hexagrid( update_hexagrid )
        
        self.scene.update_PSFparameter(**psfparam)
        self.scene.update_adr(**adrparam)

        flat_model = self.scene.run_overlay( lbda_ranges=lbda_ranges, metaslices=metaslices, use_binned_data=use_bin_data , nb_process = nb_process)

        Model_cube =  self.scene.Build_model_cube( target_ifu=IFU_coord ,target_image = self.target_imagecoord, ifu_ratio=IFU_ratio, corr_factor = parameters['corr_factor'])

        return Model_cube

        


    def chi_square(self, parameters, fix_parameters, sedm_data, sedm_var, lbda, lbda_ranges, metaslices, use_bin_data, nb_process):

        model = self.evaluate_model(parameters=parameters, fix_parameters=fix_parameters, lbda_ranges=lbda_ranges, metaslices=metaslices, use_bin_data=use_bin_data, nb_process=nb_process)

        return np.sum( (sedm_data - model.data)**2 / sedm_var )
        
        
    

    def fit(self, lbda_ranges=None, metaslices=None, fix_parameters = default_fixed_params, default_bounds=True, use_bin_data=True, nb_process='auto'):

        sedm_data, sedm_var, lbda = self.get_sedm_data(lbda_ranges, metaslices)

        parameters = self.init_params_values.copy()

        if fix_parameters is not None:
            self.update_parameter(param_to_pop = fix_parameters)
            parameters = self.current_params.copy()

        fit_params_init = np.array(list(parameters.values()))
        
        fit_params_name = list(parameters.keys())

        if default_bounds:
            self.set_default_parameters_bounds()

        fit_params_bounds = list(self.parameters_bounds.copy().values())


        self.fit_params_init=fit_params_init
        self.fit_params_name=fit_params_name
        self.fit_params_bounds=fit_params_bounds
        


        def chi_squareflat(x, fix_parameters=fix_parameters, sedm_data=sedm_data, sedm_var= sedm_var, lbda=lbda,  metaslices=metaslices, lbda_ranges=lbda_ranges, use_bin_data=use_bin_data, nb_process=nb_process):
            
            map_parameters = {i: j for i, j in zip(fit_params_name, x)}
            print(map_parameters)
            
            return self.chi_square(map_parameters, fix_parameters=fix_parameters, sedm_data=sedm_data, sedm_var= sedm_var, lbda=lbda, metaslices=metaslices, lbda_ranges=lbda_ranges, use_bin_data=use_bin_data, nb_process=nb_process )

        
        res = optimize.minimize(chi_squareflat, fit_params_init, bounds=fit_params_bounds, method="L-BFGS-B", options={'ftol': 1e-03, 'gtol': 1e-02, 'eps': 2e-02}  )
        #m = Minuit.from_array_func(chi_squareflat, fit_params_init, limit=fit_params_bounds, name=fit_params_name)
        #res=m.migrad()

        self.fit_values = dict({k:v for k,v in zip( fit_params_name, res.x)})

        return res
       



    def get_fitted_cubes(self, lbda_ranges=[], metaslices=None, use_binned_data=False, nb_process='auto', return_residual=True, model_save_dirout='default', residu_save_dirout='default'):


        if not hasattr(self,'fit_values'):

            print(bcolors.WARNING + " You have to run the fit first " + bcolors.ENDC)
            return
        
        fit_param = self.init_params_values.copy()
        fit_param.update( self.fit_values)
        
        cubemod = self.evaluate_model(fit_param, [], lbda_ranges, metaslices, use_binned_data, nb_process)
        cubemod.set_header( self.sedm_cube.header)

        self.fitted_cubemodel = cubemod


        if model_save_dirout =='default':

            cubemod.writeto(self.sedm_cube.filename[:-5] + '_HostModel' + self.sedm_cube.filename[-5:])
            self._fit_model_path = self.sedm_cube.filename[:-5] + '_HostModel' + self.sedm_cube.filename[-5:]

        elif model_save_dirout not in ( None, 'default'):
            
            cubemod.writeto( model_save_dirout)
            self._fit_model_path = model_save_dirout
            
            
        if not return_residual:
            return cubemod
        
        
        else:
            
            residu = self.get_residual_cube( full_cube = not use_binned_data)
            residu.set_header( self.sedm_cube.header)
            
            self.fitted_cuberesidu = residu

            if residu_save_dirout =='default':

                residu.writeto(self.sedm_cube.filename[:-5] + '_HostRemoved' + self.sedm_cube.filename[-5:])
                self._fit_residu_path = self.sedm_cube.filename[:-5] + '_HostRemoved' + self.sedm_cube.filename[-5:]

            elif residu_save_dirout not in ( None, 'default'):
            
                residu.writeto( residu_save_dirout)
                self._fit_residu_path = residu_save_dirout

            return (cubemod, residu)


        

    def extract_star_spectra( self, cube_path=None, exptime_correction=True, fwhm_guess=2, step1range=[4500, 9300], 
                              step1bins=8,  psfmodel='NormalMoffatFlat', centroid='fitted', **kwargs):
        
        if cube_path == None and hasattr(self,'_fit_residu_path'):
            cube_path = self._fit_residu_path
            
        cube = pysedm.sedm.load_sedmcube(cube_path)

        if exptime_correction:
            #cube.scale_by((1/self.sedm_cube.header['EXPTIME']))
            cube.header['EXPTIME']=1

        if centroid=='fitted':
            centroid = np.array([self.fit_values[k] for k in ['x0_IFU', 'y0_IFU']])

        cube.extract_pointsource(fwhm_guess=fwhm_guess, step1range=step1range, step1bins=step1bins, psfmodel=psfmodel, centroid=centroid, **kwargs)
        spec = cube.extractstar.get_spectrum(which='raw', persecond=False)
        spec.writeto('spec_'+cube_path.rsplit('/',1)[-1][10:-5]+'.txt', ascii=True)
        self.extracted_spec = spec
        self.cube_star_extracted = cube
        return cube



    
    def show_extracted_psf(self, ax=None, sliceid='auto', **kwargs):

        if not hasattr(self,'cube_star_extracted'):
            
             print(bcolors.WARNING + " You have to run the extractstar first with self.extract_star_spectra() " + bcolors.ENDC)
             return
         
        if slideid=='auto':
        
            sliceid = int(len(self.cube_star_extracted.extractstar.lbdastep1)/2)
   
        if vmin==None:
            ylim_low = np.percentile( self.cube_star_extracted.data[self.cube_star_extracted.data>0],1)
        else:
            ylim_low = np.percentile( self.cube_star_extracted.data[self.cube_star_extracted.data>0],vmin)


        if ax==None:
            self.cube_star_extracted.extractstar.show_psf(sliceid=sliceid, ylim_low=ylim_low, **kwargs)

        else:
            self.cube_star_extracted.extractstar.show_psf(sliceid=sliceid, axes=ax, ylim_low=ylim_low, **kwargs)


            


    def show_full_output(self, lbda_min=5000, lbda_max=8000, savefile_dirout=None, sliceid=2, **kwargs):

        fig10 = plt.figure( figsize=(8,12))
        fig10.subplots_adjust(top=0.95)
        line1 = plt.Line2D((.1,.9),(.91,.91), color="k", linewidth=1)
        line2 = plt.Line2D((.1,.9),(.88,.88), color="k", linewidth=1)
        fig10.add_artist(line1)
        fig10.add_artist(line2)
        fig10.suptitle( self.sedm.target + ' | ' + self.sedm.night + ' | ' + self.sedm.obs_hour, fontsize=15, y=0.9)
        gs0 = fig10.add_gridspec(3, 1, height_ratios=[2,0.5,1])
        
        gs00 = gs0[0].subgridspec(1, 3)
        gs01 = gs0[1].subgridspec(1, 5)
        gs02 = gs0[2].subgridspec(1, 5)
        
        
        ax0=fig10.add_subplot(gs00[0, 0])
        sl=self.sedm_cube.get_slice(lbda_min=lbda_min, lbda_max=lbda_max, slice_object=True )
        sl.show(ax = ax0, vmin=np.percentile(sl.data,1),vmax=np.percentile(sl.data,99.5), show_colorbar=False, rasterized=True )
        ax0.set_xlabel('SEDM data')
        ax0.set(adjustable='box', aspect='equal')
        
        
        ax1=fig10.add_subplot(gs00[0, 1])
        sl = self.fitted_cubemodel.get_slice(lbda_min=lbda_min, lbda_max=lbda_max, slice_object=True )
        sl.show(ax = ax1,vmin=np.percentile(sl.data,1),vmax=np.percentile(sl.data,99.5), show_colorbar=False, rasterized=True )
        ax1.set_xlabel('Host Model')
        ax1.set_yticks([])
        ax1.set(adjustable='box', aspect='equal')
        
        
        ax2=fig10.add_subplot(gs00[0, 2])
        sl=self.fitted_cuberesidu.get_slice(lbda_min=4750, lbda_max=8350, slice_object=True )
        sl.show(ax = ax2,vmin=np.percentile(sl.data,1),vmax=np.percentile(sl.data,99.5), show_colorbar=False, rasterized=True )
        ax2.set_xlabel('Residu')
        ax2.set_yticks([])
        ax2.set(adjustable='box', aspect='equal')
        
        
        ax3 = fig10.add_subplot(gs01[0, 0])
        ax4 = fig10.add_subplot(gs01[0, 1])
        ax5 = fig10.add_subplot(gs01[0, 2])
        ax6 = fig10.add_subplot(gs01[0, 3:])
        
        
        
        #self.cube_star_extracted.extractstar.show_psf( sliceid=2, vmin='1', vmax='98.',axes=[ax3,ax4,ax5,ax6], ylim_low= np.percentile( self.cube_star_extracted.data[self.cube_star_extracted.data>0],1), psf_in_log=True, );
        self.cube_star_extracted.extractstar.show_psf( sliceid=sliceid, vmin='1', vmax='98.',axes=[ax3,ax4,ax5,ax6], psf_in_log=False, logscale=False );

        ax3.set(adjustable='box', aspect='equal')
        ax4.set(adjustable='box', aspect='equal')
        ax5.set(adjustable='box', aspect='equal')
        #ax6.set(adjustable='box',aspect='equal')
        ax3.set_title('Data', fontsize=8)
        ax4.set_title('Model',fontsize=8)
        ax5.set_title('Residual',fontsize=8)
        ax6.set_xlabel("Elliptical distance (spx)", fontsize=7)
        #ax6.set_yticks([])
        ax6.yaxis.tick_right()
        ax6.set_ylim(np.min(np.percentile( self.cube_star_extracted.extractstar.es_products['psffit'].slices[sliceid]['slpsf'].slice.data, 0.03)), 1.1*np.max(self.cube_star_extracted.extractstar.es_products['psffit'].slices[sliceid]['slpsf'].slice.data))
        ax6.set_yticklabels([np.format_float_scientific( ax6.get_yticks()[i], precision=2) for i in range(len(ax6.get_yticks()))])
        
        ax6.legend(fontsize=7)
        
        
        axsp = fig10.add_subplot(gs02[0, 1:])
        self.cube_star_extracted.extractstar.show_extracted_spec(ax=axsp ,add_metaslices=False)
        axsp.set_xlabel(r'Wavelength ($\AA$)')
        axsp.set_ylabel(r'Flux ($ erg \ s^{-1} cm^{-2} \AA^{-1}$)')
        axsp.set_ylim(np.min(np.percentile(self.extracted_spec.data,0.5))*0.8, np.max(np.percentile(self.extracted_spec.data,99))*1.3);
        
        #axsp.set_ylim(np.min(spec.data)*0.7,np.max(spec.data)*1.3)
       
        
        line3 = plt.Line2D((.1,.9),(.53,.53), color="k", linewidth=1)
        fig10.add_artist(line3)
        fig10.text(0.5, 0.537, 'Point Source Extraction from pysedm method', fontsize=12, ha='center')
        line4 = plt.Line2D((.1,.9),(.56,.56), color="k", linewidth=1)
        fig10.add_artist(line4)
        import datetime
        fig10.text( 0.5,0.01,f"hypergal version 0.1 | made the {datetime.datetime.now().date().isoformat()} | J.Lezmy (lezmy@ipnl.in2p3.fr)", ha='center', color='grey', fontsize=7)

        if savefile_dirout is not None:
            fig10.savefig(savefile_dirout, **kwargs) 

        return fig10


            

    def set_parameters_values_init(self, fix_parameter = default_fixed_params, corr_factor=1):


        IFU_target = self.sedm.get_estimate_target_coord()
        
        self._init_IFU_target = dict({'x0_IFU': IFU_target[0], 'y0_IFU':IFU_target[1]})
        self._init_adr_parameter = self.scene.adr.data.copy()
        self._init_PSF_parameter = self.scene.psfmodel.params.copy()
        
        self._corr_factor = dict({"corr_factor":corr_factor})

        self.init_params_values = dict()
        self.init_params_values.update( self._init_adr_parameter)
        self.init_params_values.update( self._init_PSF_parameter)
        self.init_params_values.update( self._init_IFU_target)
        self.init_params_values.update( self._corr_factor)

        


    def set_parameters_bounds(self, names=None, bounds=None ):

            parameters_bounds = self.init_params_values.copy()
    
            for k in parameters_bounds.keys():
                parameters_bounds[k]=(None,None)

            if (names==None or bounds==None) :
                self.parameters_bounds = parameters_bounds

            return
                
    
            for (name, bound) in zip(names,bounds):
        
                if name in parameters_bounds.keys():
                    parameters_bounds[name]=bound
                else:
                    print(bcolors.WARNING + "unknown property %s, it cannot be set. known properties are: "%name,", ".join(parameters_bounds) + bcolors.ENDC)
        
            self.parameters_bounds = parameters_bounds



    def set_default_parameters_bounds(self):

        IFU_target = self.sedm.get_estimate_target_coord()
        self.parameters_bounds['airmass'] = default_airmass_bounds

        for (k,bound) in zip(self.scene.psfmodel.params.keys(), self.scene.psfmodel.psfmodel._bounds) :
            self.parameters_bounds[k] = bound
            
        self.parameters_bounds['corr_factor']=(0.5,2)
        self.parameters_bounds['x0_IFU'] = ( IFU_target[0]-6, IFU_target[0]+6)
        self.parameters_bounds['y0_IFU'] = ( IFU_target[1]-6, IFU_target[1]+6)


    def update_parameter(self, param_to_pop=None, **kwargs):
        
        if not hasattr(self, 'current_params'):
            self.current_params = self.init_params_values.copy()
        
        for (k,v) in kwargs.items():
            
            if k not in self.current_params:
                
                raise ValueError("unknown property %s, it cannot be set. known properties are: "%k,", ".join(self.current_params))
           
            self.current_params[k]=  v        

        if param_to_pop is not None:

            for k in param_to_pop:
                if k in self.current_params.keys():
                
                    self.current_params.pop(k)
                    self.parameters_bounds.pop(k)

        

    def lbda_step_bin(self, lbda_ranges, metaslice):

        STEP_LBDA_RANGE = np.linspace(lbda_ranges[0],lbda_ranges[1], metaslice+1)
        return np.asarray([STEP_LBDA_RANGE[:-1], STEP_LBDA_RANGE[1:]]).T

        
    def set_lbda_ranges(self, lbda_ranges):

        self.lbda_ranges = lbda_ranges

    
    def set_metaslices(self, metaslices):

        self.metaslices = metaslices

    










# End of Host_removing.py ========================================================
