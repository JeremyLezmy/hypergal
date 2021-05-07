#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          fitter.py
# Description:       script description
# Author:            Jeremy Lezmy <lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/04/29 17:01:52 $
# Modified on:       2021/05/07 10:17:43
# Copyright:         2019, Jeremy Lezmy
# $Id: fitter.py, 2021/04/29 17:01:52  JL $
################################################################################

"""
.. _fitter.py:

fitter.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <lezmy@ipnl.in2p3.fr>'
__date__ = '2021/04/29 17:01:52'
__adv__ = 'fitter.py'

import os
import sys
import datetime
from scipy import optimize

import numpy as np
import pysedm
import pandas as pd
from shapely import geometry, affinity
import shapely
import time
import geopandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from scipy.interpolate import interp1d
from astropy.convolution import Box1DKernel, convolve
import warnings
import pyifu

from hypergal import sed_fitting as sedfit
from hypergal import panstarrs_target as ps1targ
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


IFU_ratio = 2.232
IFU_scale = 0.558
default_airmass_bounds = (1,5)


default_fixed_params=['temperature', 'relathumidity', 'pressure', 'lbdaref']
default_fixed_params_slice=['temperature', 'relathumidity', 'pressure', 'lbdaref', 'airmass', 'parangle']
        

class Fitter():

    
    def __init__(self, sedmcalcube, scene, IFU_target=None ):
        """ 
        Parameters:

        sedm_target : Instance of sedm_target()
        scene :  Instance of intrinsec_cube()
        target_pixcoord_image: Position in pixel of the target in the photometric image. Format list/array with size 2. 
        """

        #self.sedm = sedm_target
        self.sedm_cube = sedmcalcube        
        self.scene = scene       
        self.target_imagecoord =  scene.int_targetpos
        
        self.hexagrid = scene.hexagrid
        
        if IFU_target is None:
            IFU_target = scene.sedm_targetpos
            
        self.set_IFU_target_init(IFU_target)
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



    def get_intrinsec_data(self, lbda_ranges=None, metaslices=None):

        spec,_ = self.scene.get_metaslices_data( lbda_ranges, metaslices)
        self.binned_intrinsec_data = spec.T.copy()
        
        return (spec.T)




    def evaluate_slice_model(self, parameters, fix_parameters, intrinsec_data, lbda, bkg):

        if ('x0_IFU' in fix_parameters) and ('y0_IFU' in fix_parameters):
            IFU_coord = list(self._init_IFU_target.values())
        else:
            IFU_coord = [parameters[k] for k in ['x0_IFU','y0_IFU']]
        psfparam = {k:parameters[k] for k in (self.scene.psfmodel.params.keys() - fix_parameters) }
        #adrparam = {k:parameters[k] for k in (self.scene.adr.data.keys() - fix_parameters) }
        
        
        update_hexagrid = geotool.get_cube_grid( self.sedm_cube, scale = IFU_ratio, targShift=IFU_coord, x0=self.target_imagecoord[0] , y0=self.target_imagecoord[1]  )

        self.scene.set_hexagrid( update_hexagrid )
        
        self.scene.update_PSFparameter(**psfparam)
        #self.scene.update_adr(**adrparam)

        H,W = self.scene.get_shape_pixelgrid()
            
        spec = hostmodel.psf_convolution( data = np.reshape(np.atleast_2d(intrinsec_data.copy()), ( H, W, len(np.array([lbda]))) ), lbda = np.array([lbda]), lbdaref = lbda, psfkernel=self.scene.psfmodel, nb_process=1)
        spec = spec.reshape(int(H*W), len(np.array([lbda])))

        new_spax = hostmodel.measure_overlay(nb_process = 1, spec= spec, lbda= np.array([lbda]), pixelgrid=self.scene.pixelgrid, hexagrid=update_hexagrid, adr=None, pixel_size=self.scene.pixelsize )

        flux = np.array([new_spax[i]['flux'] for i in range(len(new_spax))])
        pixMap = dict(zip(np.arange(0, len(new_spax[0])), np.array( [ (np.array (new_spax[0]['centroid_x']) + IFU_coord[0] * IFU_ratio - self.target_imagecoord[0]) / IFU_ratio,
                                                                      (np.array (new_spax[0]['centroid_y']) + IFU_coord[1] * IFU_ratio - self.target_imagecoord[1]) / IFU_ratio ]).T ))

        spax_vertices = np.array([[ 0.19491447,  0.6375365 ],[-0.45466557,  0.48756913],[-0.64958004, -0.14996737],[-0.19491447, -0.6375365 ],[ 0.45466557, -0.48756913], [ 0.64958004,  0.14996737]])

        dat =  parameters['corr_factor']*flux.squeeze() + bkg*parameters['sc_bkg'] if 'sc_bkg' in parameters else parameters['corr_factor']*flux.squeeze() + bkg
        
        model_slice = pyifu.spectroscopy.get_slice( data = dat, xy =list(pixMap.values()) , spaxel_vertices=spax_vertices, variance=None, indexes=list(pixMap.keys()), lbda=np.array([lbda]))


        return model_slice




    def chi_square_slice(self, parameters, fix_parameters, sedm_data, sedm_var, intrinsec_data, lbda, bkg):

        model = self.evaluate_slice_model(parameters=parameters, fix_parameters=fix_parameters, intrinsec_data=intrinsec_data, lbda=lbda, bkg=bkg)

        return np.nansum( (sedm_data - model.data)**2 / sedm_var )


    def fit_slice(self, lbda_ranges=None, metaslices=None, sliceid=0, fix_parameters = default_fixed_params_slice, apply_bkg=True, default_bounds=True, ftol=1e-4):

        sedm_data, sedm_var, lbda = self.get_sedm_data(lbda_ranges, metaslices)

        intrinsec_data = self.get_intrinsec_data(lbda_ranges,metaslices).copy()

        sedm_slice_data = np.atleast_2d(sedm_data)[sliceid]
        sedm_slice_var = np.atleast_2d(sedm_var)[sliceid]
        intrinsec_slice_data = np.atleast_2d(intrinsec_data)[sliceid]
        lbda_slice = lbda[sliceid]
               
        parameters = self.init_params_values.copy()
        if default_bounds:
            self.set_default_parameters_bounds()

        if apply_bkg:
            estim_bkg = abs(np.nanmedian(sedm_slice_data[sedm_slice_data<np.percentile(sedm_slice_data, 10)]))            
        else:
            estim_bkg=0
            self.update_parameter(param_to_pop = 'sc_bkg')
            parameters = self.current_params.copy()


        if fix_parameters is not None:
            self.update_parameter(param_to_pop = fix_parameters)
            parameters = self.current_params.copy()

        fit_params_init = np.array(list(parameters.values()))
        
        fit_params_name = list(parameters.keys())

        fit_params_bounds = list(self.parameters_bounds.copy().values())


        self.fit_params_init=fit_params_init
        self.fit_params_name=fit_params_name
        self.fit_params_bounds=fit_params_bounds

        #if not hasattr(self,'fit_values'):
        #    self.fit_values=dict()
        #    self.fit_values_err=dict()
        
        def chi_squareflat(x, fix_parameters=fix_parameters, sedm_data=sedm_slice_data, intrinsec_data =intrinsec_slice_data, sedm_var= sedm_slice_var, lbda=lbda_slice, bkg=estim_bkg):

            
            map_parameters = {i: j for i, j in zip(fit_params_name, x)}
            print(map_parameters)
            
            return self.chi_square_slice(map_parameters, fix_parameters=fix_parameters, sedm_data=sedm_data, sedm_var= sedm_var, intrinsec_data=intrinsec_data, lbda=lbda, bkg=bkg )

        
        res = optimize.minimize(chi_squareflat, fit_params_init, bounds=fit_params_bounds, method="L-BFGS-B", options={'ftol': ftol, 'gtol': 1e-04, 'eps': 1e-03, 'maxls':10}  )
        #m = Minuit.from_array_func(chi_squareflat, fit_params_init, limit=fit_params_bounds, name=fit_params_name)
        #m.tol=100
        #res=m.migrad()

        #self.fit_values.update({fr'slice{sliceid}':dict({k:v for k,v in zip( fit_params_name, m.values.values())})})
        #fit_values = ({fr'slice{sliceid}':dict({k:v for k,v in zip( fit_params_name, m.values.values())})})
        
        #self.fit_values.update({fr'slice{sliceid}':dict({k:v for k,v in zip( fit_params_name, res.x)})})
        #self.fit_values[fr'slice{sliceid}'].update({'lbda':lbda_slice})

        err = np.sqrt(max(1, abs(res.fun)) * ftol * np.diag(res.hess_inv.todense()))
        
        fit_values = {**dict({k:v for k,v in zip( fit_params_name, res.x)}),
                      **dict({k+"_err":v for k,v in zip( fit_params_name, err)})}
        return {f'slice{sliceid}':fit_values }
                      
        
        #fit_values = ({fr'slice{sliceid}':dict({k:v for k,v in zip( fit_params_name, res.x)})})
        #fit_values[fr'slice{sliceid}'].update({'lbda':lbda_slice})
        

        #fit_values_err = ({fr'slice{sliceid}':dict({k:v for k,v in zip( [s + '_err' for s in fit_params_name], err)})})
        #fit_values_err[fr'slice{sliceid}'].update({'lbda':lbda_slice})
        #self.fit_values_err.update({fr'slice{sliceid}':dict({k:v for k,v in zip( [s + '_err' for s in fit_params_name], err)})})
        #self.fit_values_err[fr'slice{sliceid}'].update({'lbda':lbda_slice})

        #return (fit_values, fit_values_err,res)


    def fit_multislice(self, ncore=1, lbda_ranges=None, metaslices=None, sliceid=None, fix_parameters = default_fixed_params, default_bounds=True):

        if sliceid==None:
            sliceid=np.arange(metaslices)
            
        if ncore==1:

            for sl in sliceid:

                self.fit_slice(lbda_ranges=lbda_range, metaslices=metaslices, sliceid=sl, fix_parameters = fix_parameters, default_bounds=default_bounds)

        if ncore>1:
            print('TO DO')
            return

        return self.fit_values


    def get_fitted_adr(self, x0, y0, x0_err, y0_err, lbda, lbdaref=6500, show=False, savefile=None):
        import pyifu
        import pyifu.adr as adr
        from scipy import optimize
        
        adrset=pyifu.adr.ADR()
        datacube = self.sedm_cube.copy()
        datacube.load_adr()
               
        adrset.set(airmass=datacube.adr.airmass, lbdaref=datacube.adr.lbdaref, pressure=datacube.adr.pressure, 
            temperature=datacube.adr.temperature, parangle=datacube.adr.parangle, relathumidity=datacube.adr.relathumidity )
       
        if lbdaref is not None:        
            adrset.set( lbdaref=lbdaref)

        xref_init = x0[lbda==lbda[abs(adrset.lbdaref-lbda)==np.min(abs(adrset.lbdaref-lbda))]][0]
        yref_init = y0[lbda==lbda[abs(adrset.lbdaref-lbda)==np.min(abs(adrset.lbdaref-lbda))]][0]
        
        def fcnscale(X):
            
            adrset.set( parangle=X[0] )
            adrset.set( airmass=X[1] )
            xref=X[2]
            yref=X[3] 
            
            codat=np.array([x0,y0])
            codat_err=np.array([x0_err,y0_err])
            comod=adrset.refract(xref,yref,lbda, unit=IFU_scale)
                                  
            return (np.sum((codat-comod)**2/codat_err**2))

        scaletest=optimize.minimize(fcnscale,np.array([datacube.adr.parangle, datacube.adr.airmass, xref_init, yref_init]) )
        self.fit_airmass = scaletest.x[1]
        self.fit_parangle = scaletest.x[0]
        self.fit_xref = scaletest.x[2]
        self.fit_yref = scaletest.x[3]
        self.fit_adrobj = adrset
        #self.fit_err_airmass = np.diag(scaletest.hess_inv)[1]**0.5
        #self.fit_err_parangle = np.diag(scaletest.hess_inv)[0]**0.5

        if show:
            
            self.show_adr( adrset, x0, y0, x0_err, y0_err, self.fit_xref, self.fit_yref, lbda, savefile = savefile)

        
            
       
        return scaletest, self.fit_adrobj


    def get_fitted_psf(self, fit_params, fit_err=None, lbdaref=6500, psfmodel='Gauss_Mof_kernel'):

        if psfmodel=='Gauss_Mof_kernel':

            val = []
            err = []
            for sl in fit_params.keys():
                val.append(fit_params[sl])

            if fit_err is not None:
                for sl in fit_err.keys():
                    err.append(fit_err[sl])
                
            dval = {}
            derr = {}
            for sl in range(len(val)):   
                for key in set(list(val[sl].keys())):
                    try:
                        dval.setdefault(key,[]).append(val[sl][key])        
                    except KeyError:
                        pass

                if fit_err is not None:
                    
                    for key in set(list(err[sl].keys())):
                        try:
                            derr.setdefault(key,[]).append(err[sl][key])        
                        except KeyError:
                            pass
                
            if fit_err is not None:
                alpha = np.sum( np.dot(np.array(dval['alpha']), np.array(derr['alpha_err'])**-2))/np.sum( np.array(derr['alpha_err'])**-2 )  
                eta = np.sum( np.dot( np.array(dval['eta']), np.array(derr['eta_err'])**-2))/np.sum( np.array(derr['eta_err'])**-2 )  
                A = np.sum( np.dot(np.array(dval['A']), np.array(derr['A_err'])**-2))/np.sum( np.array(derr['A_err'])**-2 )  
                B = np.sum( np.dot(np.array(dval['B']), np.array(derr['B_err'])**-2))/np.sum( np.array(derr['B_err'])**-2 )
                sigma_err = np.array(derr['sigmaref_err'])

            else :
                alpha = np.mean(dval['alpha'])
                eta = np.mean(dval['eta'])
                A = np.mean(dval['A'])
                B = np.mean(dval['B'])
                sigma_err = np.ones(shape=len(dval['sigmaref']))

            lbda = np.array(dval['lbda'])
            
            
            def get_sigmaref(X):
                sigref=X[0]

                return( np.sum(( (psfker.chrom_sigma( sigref, lbda, lbdaref, rho=-1/5) - np.array(dval['sigmaref']))**2/ np.array(derr['sigmaref_err'])**2)))

            import scipy
            sci=scipy.optimize.minimize(get_sigmaref, np.array([1]) )
            sigref_fit = sci.x[0]

            self.fit_chrom_psf = dict({'A':A, 'B':B, 'alpha':alpha,'sigmaref':sigref_fit, 'eta':eta })
            
            return self.fit_chrom_psf
        

        if psfmodel!= 'Gauss_Mof_kernel':
            print(psfmodel,"Not Implemented")





    def fit_coeff_bkg(self, flatmodel ):

        spaxproj = flatmodel.copy()
        testspax = np.array([spaxproj[i]['flux'].values for i in range(len(spaxproj))]).copy()
        
        #idx = np.arange(0, testspax.shape[-1])
        bkg_estimate = np.nanmedian(self.sedm_cube.get_index_data(self.sedm_cube.get_faintest_spaxels(50)), axis=0)
        
        def fun(coeff, bkg, slices):
            testspax = np.array([spaxproj[i]['flux'].values for i in range(len(spaxproj))]).copy()
            testspax[slices,:] *= coeff
            testspax[slices,:] += bkg * bkg_estimate[slices]
            
            return np.nansum( (testspax[slices,:] - self.sedm_cube.data[slices,:])**2/self.sedm_cube.variance[slices,:] )

        coeff_arr = np.ones(shape=(220))
        bkg_arr = np.ones(shape=(220))

        print('coeff and background fitting...')
        
        for sl in range(220):
        
        #def coeff_process(sl):
            
            def fun_flat(coeff, bkg):
                
                return fun(coeff,bkg, slices=sl)
        
            m = Minuit(fun_flat, coeff=1, bkg=1, errordef=1);
            m.migrad();
            coeff_arr[sl]*=m.values[0]
            bkg_arr[sl]*=m.values[1] * bkg_estimate[sl]
            
            if sl%20==0:
                print(sl,'/220')
                       
        return(coeff_arr, bkg_arr)

    


    def evaluate_model_cube(self, parameters, fix_parameters=[], lbda_ranges=[], metaslices=None, use_bin_data=False,  nb_process='auto', fit_coeff=True, getresidu=False):

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

        if fit_coeff:

            coeff_arr, bkg_arr = self.fit_coeff_bkg(flat_model)
            self.fit_coeff_cube = coeff_arr
            self.fit_bkg_cube = bkg_arr

            Model_cube =  self.scene.Build_model_cube( target_ifu=IFU_coord ,target_image = self.target_imagecoord, ifu_ratio=IFU_ratio, corr_factor = coeff_arr, bkg=bkg_arr,)

        else :
            
            Model_cube =  self.scene.Build_model_cube( target_ifu=IFU_coord ,target_image = self.target_imagecoord, ifu_ratio=IFU_ratio,  corr_factor = parameters['corr_factor'])

        self.model_cube = Model_cube

        if getresidu:
            import pyifu
            cuberesidu=pyifu.spectroscopy.get_cube(data =( self.sedm_cube.data - Model_cube.data), lbda = self.sedm_cube.lbda, spaxel_mapping = self.scene.pixmapping)
            spax_vertices = np.array([[ 0.19491447,  0.6375365 ], [-0.45466557,  0.48756913], [-0.64958004, -0.14996737], [-0.19491447, -0.6375365 ], [ 0.45466557, -0.48756913], [ 0.64958004,  0.14996737]])
        
            cuberesidu.set_spaxel_vertices( spax_vertices )

            self.residual_cube = cuberesidu

            return Model_cube, cuberesidu
       
        return Model_cube


    


                


    def set_parameters_values_init(self, fix_parameter = default_fixed_params, corr_factor=1, sc_bkg=1):

        
        IFU_target = self.IFU_target_initcoor
        
        self._init_IFU_target = dict({'x0_IFU': IFU_target[0], 'y0_IFU':IFU_target[1]})
        self._init_adr_parameter = self.scene.adr.data.copy()
        self._init_PSF_parameter = self.scene.psfmodel.params.copy()
        
        self._corr_factor = dict({"corr_factor":corr_factor})
        self._scale_bkg = dict({"sc_bkg":sc_bkg})

        self.init_params_values = dict()
        self.init_params_values.update( self._init_adr_parameter)
        self.init_params_values.update( self._init_PSF_parameter)
        self.init_params_values.update( self._init_IFU_target)
        self.init_params_values.update( self._corr_factor)
        self.init_params_values.update( self._scale_bkg)

        


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

        
        IFU_target = self.IFU_target_initcoor
            
        self.parameters_bounds['airmass'] = default_airmass_bounds

        for (k,bound) in zip(self.scene.psfmodel.params.keys(), self.scene.psfmodel.psfmodel._bounds) :
            self.parameters_bounds[k] = bound
            
        self.parameters_bounds['corr_factor'] = (0.0001,2)
        self.parameters_bounds['sc_bkg']=(0,None)
        self.parameters_bounds['x0_IFU'] = ( IFU_target[0]-6, IFU_target[0]+6)
        self.parameters_bounds['y0_IFU'] = ( IFU_target[1]-6, IFU_target[1]+6)


    def set_IFU_target_init(self,IFU_target):

        if IFU_target is not None:
            self.IFU_target_initcoor = IFU_target
        else:
            self.IFU_target_initcoor = np.array([0,0])
        
            

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
                if k in self.parameters_bounds.keys():
                    self.parameters_bounds.pop(k)

        

    def lbda_step_bin(self, lbda_ranges, metaslice):

        STEP_LBDA_RANGE = np.linspace(lbda_ranges[0],lbda_ranges[1], metaslice+1)
        return np.asarray([STEP_LBDA_RANGE[:-1], STEP_LBDA_RANGE[1:]]).T

        
    def set_lbda_ranges(self, lbda_ranges):

        self.lbda_ranges = lbda_ranges

    
    def set_metaslices(self, metaslices):

        self.metaslices = metaslices




    def show_adr(self, adrobj, x0, y0, x0_err, y0_err,xref,yref, lbda, savefile = None):
        
        fig,ax=plt.subplots( figsize=(8,8))


        import pyifu
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        
        colormap = cm.jet
        normalize = mcolors.Normalize(vmin=np.min(lbda), vmax=np.max(lbda))
        s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)

        colors = plt.cm.jet((lbda-np.min(lbda))/(np.max(lbda)-np.min(lbda)))
        
        ax.scatter(x0,y0,cmap=colormap, c=lbda, label='fit centroid')
        ax.errorbar(x0,y0,x0_err, y0_err, fmt='none', color=colors)
          
        adrfit=ax.scatter(adrobj.refract(xref,yref, lbda , unit = 0.558)[0], adrobj.refract(xref,yref, lbda , unit = 0.558)[1], marker='o',cmap=colormap, c=lbda, fc='none',edgecolors='k', label='fitted adr')
   
        from matplotlib.lines import Line2D
        Line2D([0], [0], marker='o',linestyle='',  markersize=8, fillstyle=Line2D.fillStyles[-1],label=r'Theoretical ADR ')
        Line2D([0], [0],marker='o',linestyle='',markeredgecolor='k', markerfacecolor='k',  markersize=8, fillstyle=Line2D.fillStyles[-1],label=r'Fitted position ')
   
        ax.legend()        
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel(r'x(spx* $\sqrt{3}/2$ )')
        ax.set_ylabel(r'y(spx* $\sqrt{3}/2$ )')
        fig.colorbar(s_map, label=r'$\lambda$', ax=ax, use_gridspec=True)
        fig.suptitle(fr'ADR fit for ' + self.sedm_cube.filename.rsplit('.')[0].rsplit('/')[-1].rsplit('_')[-1]+ '\n' +
                     fr'$x_{{ref}}= {np.round(self.fit_xref,2)},y_{{ref}}= {np.round(self.fit_yref,2)}, \lambda_{{ref}}= {adrobj.lbdaref}\AA  $' + '\n' +
                     fr'$Airmass= {np.round( self.fit_airmass,2)},Parangle= {np.round( self.fit_parangle,2)}  $')
       
        ax.set_aspect('equal',adjustable='datalim')

        
        if savefile != None:
            
            fig.savefig( savefile )
          
            



    

# End of fitter.py ========================================================
