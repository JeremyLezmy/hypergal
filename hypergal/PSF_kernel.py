#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          PSF_kernel.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: rlezmy $
# Created on:        $Date: 2021/01/26 18:38:59 $
# Modified on:       2021/05/05 14:49:46
# Copyright:         2019, Jeremy Lezmy
# $Id: PSF_kernel.py, 2021/01/26 18:38:59  JL $
################################################################################

"""
.. _PSF_kernel.py:

PSF_kernel.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/26 18:38:59'
__adv__ = 'PSF_kernel.py'

import os
import sys
import datetime


import numpy as np
import math
from scipy.stats import norm
from collections import OrderedDict
import matplotlib.pyplot as plt


def Gauss_Kernel_2D(x_stddev=1, y_stddev=1,theta=0):
        
        
    """
    Parameters
    --------------
        
    x_stddev = float (default=1)
    y_stddev = float (default=1)

    theta = float in rad (default=0)
        
    References
        ----------
    [1] https://en.wikipedia.org/wiki/Gaussian_function
    """
    import math
    import numpy as np
    
    x_mean=0
    y_mean=0
    
    x_size = np.ceil(8 * np.max([x_stddev, y_stddev]))
    if x_size % 2 == 0:
        x_size +=  1
    y_size=x_size
    
    
    x_range = (-(int(x_size) - 1) // 2, (int(x_size) - 1) // 2 + 1)
    y_range = (-(int(y_size) - 1) // 2, (int(y_size) - 1) // 2 + 1)
    
    x = np.arange(*x_range)
    y = np.arange(*y_range)
    
    
    x, y = np.meshgrid(x, y)
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = x_stddev ** 2
    ystd2 = y_stddev ** 2
    xdiff = x - x_mean
    ydiff = y - y_mean
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    amplitude = 1. / (2 * np.pi * x_stddev * y_stddev)
    
    
    kernel= amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)))
    
    return kernel


def Moffat_Kernel_2D(alpha_x=3, alpha_y=3, beta=7, theta=0):
        
        
    """
    Parameters
    --------------
    
    gamma = float (default=3)
    
    alpha_x = float (default=7)
    alpha_y = float (default=7)
        
    theta = float in rad (default=0)
        
        
    """
        
    import math
        
    fwhmx = 2.0 * np.abs(alpha_x) * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    fwhmy = 2.0 * np.abs(alpha_y) * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    
    x_size = math.ceil(4 * np.max([fwhmx,fwhmy]))
    if x_size % 2 == 0:
        x_size +=  1
    y_size=x_size
    
    
    x_range = (-(int(x_size) - 1) // 2, (int(x_size) - 1) // 2 + 1)
    y_range = (-(int(y_size) - 1) // 2, (int(y_size) - 1) // 2 + 1)
    
    x = np.arange(*x_range)
    y = np.arange(*y_range)
        
    
    
    A=(np.cos(theta)/alpha_x)**2 + (np.sin(theta)/alpha_y)**2
    B=(np.sin(theta)/alpha_x)**2 + (np.cos(theta)/alpha_y)**2
    C=2*np.sin(theta)*np.cos(theta)*(1/(alpha_x)**2 - 1/(alpha_x)**2)
    
    x, y = np.meshgrid(x, y)
    
    x_0=0
    y_0=0
    
    rr_gg = (A*(x - x_0) ** 2 + B*(y - y_0) ** 2 + C*(x - x_0)*(y - y_0)) 
    
    amplitude = (beta - 1.0) / (np.pi * alpha_x * alpha_y)
    kernel= amplitude * (1 + rr_gg) ** (-beta)
    
    
    return kernel/np.sum(kernel)




def Mof(x,beta,alpha):

    """
    """
    return (1+(x/alpha)**2)**(-beta)



def Gauss_Mof(sigma, alpha, eta, A, B):

    """
    """

    import math
    import numpy as np
    
    x_mean=0
    y_mean=0
    
    beta=alph_to_beta(alpha)
    
    fwhm = 2.0 * np.abs(alpha) * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
    
    x_size = math.ceil(20 * np.max([sigma, fwhm/2]))
    if x_size % 2 == 0:
        x_size +=  1
    y_size=x_size        

    x_range = (-(int(x_size) - 1) // 2, (int(x_size) - 1) // 2 + 1)
    y_range = (-(int(y_size) - 1) // 2, (int(y_size) - 1) // 2 + 1)

    x = np.arange(*x_range)
    y = np.arange(*y_range)

    x, y = np.meshgrid(x, y)
    xcentroid=0
    ycentroid=0,
    dx, dy = (x-xcentroid), (y-ycentroid)

    normalisation = (np.pi / np.sqrt(A - B**2) * (2 * eta * sigma**2 + alpha**2
                                                      / (beta - 1)))

    


    r= np.sqrt( dx**2 + A*dy**2 + 2*B * (dx*dy) ) 
   
    gaussian = np.exp(-0.5 * r**2 / sigma**2)

    Moffat = (1+(r/alpha)**2)**(-beta)
        
    return  (eta*gaussian + Moffat)/normalisation 
   



def read_psf_model(psfmodel):

    if 'Gauss_Mof_kernel' in psfmodel:

        return Gauss_Mof_kernel()

    if 'Kolmo_extend' in psfmodel:

        return Kolmo_extend()
    

def chrom_sigma( sigmaref, lbda, lbdaref, rho=-1/5):
        """ Evolution of the standard deviation as a function of lbda. """
        return sigmaref * (lbda / lbdaref)**(rho)


class Gauss_Mof_kernel():

    def __init__(self):
        
        #self._init_parameters = dict( { 'A':1.5, 'B':0, 'eta':2, 'sigma':1, 'alpha':2.5} )
        #self._bounds = [ (0,5), (-5,5), (0,None), (0.1,15), (0.6,15)  ]
        self._init_parameters = dict( { 'A':1.5, 'B':0, 'alpha':2, 'sigmaref':1, 'eta':2} )
        self._bounds = [ (-10,10), (-2,2), (0.01,15), (0.05,5), (0,None)  ]
        self._name = 'Gaussian+Moffat'


    def alph_to_beta(self, alpha, b0=1.35, b1=0.22):
        """
        """
        return (b0+alpha*b1)



    def chrom_sigma(self, sigmaref, lbda, lbdaref, rho=-1/5.):
        """ Evolution of the standard deviation as a function of lbda. """
        return sigmaref * (lbda / lbdaref)**(rho)



    
    def evaluate(self, A, B, alpha, sigmaref, eta, lbda, lbdaref,  normed=True):

            import math
            import numpy as np
            
            x_mean=0
            y_mean=0
            
            
            beta = self.alph_to_beta(alpha)
            sigma = self.chrom_sigma(sigmaref=sigmaref, lbda=lbda, lbdaref=lbdaref)
            
            
            fwhm = 2.0 * np.abs(alpha) * np.sqrt(2.0 ** (1.0 / beta) - 1.0)
            
            x_size = math.ceil(20 * np.max([sigma, fwhm/2]))
            if x_size % 2 == 0:
                x_size +=  1
            y_size=x_size        

            x_range = (-(int(x_size) - 1) // 2, (int(x_size) - 1) // 2 + 1)
            y_range = (-(int(y_size) - 1) // 2, (int(y_size) - 1) // 2 + 1)
            
            x = np.arange(*x_range)
            y = np.arange(*y_range)
            
            x, y = np.meshgrid(x, y)
            xcentroid=0
            ycentroid=0,
            dx, dy = (x-xcentroid), (y-ycentroid)
            
            normalisation = (np.pi / np.sqrt(A - B**2) * (2 * eta * sigma**2 + alpha**2
                                                      / (beta - 1)))

    


            r= np.sqrt( dx**2 + A*dy**2 + 2*B * (dx*dy) ) 
            
            gaussian = np.exp(-0.5 * r**2 / sigma**2)
            
            Moffat = (1+(r/alpha)**2)**(-beta)
            
            return  (eta*gaussian + Moffat)/normalisation 

    
    


class PSF_kernel():

    def __init__(self, psfmodel):

        self.psfmodel = read_psf_model(psfmodel)

        self.params = self.psfmodel._init_parameters.copy()
        self._bounds = self.psfmodel._bounds.copy()
               

    def update_parameter(self, **kwargs):
    
        for (k,v) in kwargs.items():
            if k in self.params.keys():
                self.params[k] = v

    def get_kernel_data(self, lbda, lbdaref):
            params = self.params.copy()
            params.update({'lbda':lbda,'lbdaref':lbdaref})
            return(self.psfmodel.evaluate(**params))
        
        
    def show_kernel(self, ax=None, **kwargs):

        if ax ==None:
            fig,ax = plt.subplots()

        else:
            fig = ax.figure

        ax.imshow(self.psfmodel.evaluate(**self.params), **kwargs)
        
        
    def convolution(self, data, lbda, lbdaref, **kwargs):
        
        from astropy.convolution import convolve
        params = self.params.copy()
        params.update({'lbda':lbda,'lbdaref':lbdaref})
        convolve_data = convolve(np.nan_to_num(data), self.psfmodel.evaluate(**params), normalize_kernel=False, **kwargs )
        
        return convolve_data





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



# End of PSF_kernel.py ========================================================
