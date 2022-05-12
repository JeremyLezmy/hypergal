#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import datetime
from astropy import constants
import numpy as np
import multiprocessing as mp
import shutil
import warnings
from astropy.convolution import convolve, Gaussian1DKernel
try:
    from pcigale import init, genconf, check, run
    from pcigale.session.configuration import Configuration
except:
    warnings.warn("Cigale isn't installed on this machine")

import pyifu
import collections.abc


def command_cigale(command, file_path=None):
    '''
    Call pcigale commands through python function rather than shell commands.\n
    Note that the run command requires to work in current terminal directory,
    data, config and results files are moved back after the operation.

    Parameters
    ----------
    command : string
        Available pcigale command are 'init', 'genconf', 'check', 'run'

    file_path : string
        Path to data, config and result files, if different from the current directory.\n
        Default is None.
    '''

    configfile = ''
    if file_path != None:
        configfile += file_path
    configfile += 'pcigale.ini'  # The configfile MUST have this name.

    if sys.version_info[:2] < (3, 6):
        raise Exception(f"Python {sys.version_info[0]}.{sys.version_info[1]} is"
                        f" unsupported. Please upgrade to Python 3.6 or later.")

    # We set the sub processes start method to spawn because it solves
    # deadlocks when a library cannot handle being used on two sides of a
    # forked process. This happens on modern Macs with the Accelerate library
    # for instance. On Linux we should be pretty safe with a fork, which allows
    # to start processes much more rapidly.

    if sys.platform.startswith('linux'):
        mp.set_start_method('fork', force=True)
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
            # pcigale run command requires the data and config files to be in the
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
                print(
                    f"The out/ directory already exists, the old one was renamed to {name}")
            files = ['pcigale.ini', 'pcigale.ini.spec', 'test.mag', 'out/']
            move_files(actual_path, file_path, files)

    else:
        print(f'Command \'{command}\' was not recognized. Available commands are'
              f' \'init\', \'genconf\', \'check\' and \'run\'')


def move_files(old_path, new_path, files, verbose=False):
    '''
    Move files (and/or directories) from one location to another.

    Parameters
    ----------
    old_path : string
        Name of the current path to the files.

    new_path : string
        Name of the new path to the files.

    files : [string]
        Names of every files(and/or directories) to move.

    verbose : bool
        If True, print old and new locations.\n
        Default to False.
    '''
    old_location = [old_path+l for l in files]
    new_location = [new_path+l for l in files]

    try:
        _ = [os.rename(l, nl) for l, nl in zip(old_location, new_location)]
    except OSError:
        _ = [shutil.move(l, nl) for l, nl in zip(old_location, new_location)]
    if verbose:
        print(old_location)
        print("moved to")
        print(new_location)


def update_config(d, u):
    """
    Tools to update dict of dict without losing not changed values

    Parameters
    ----------
    d: dict
        Dict to update

    u: dict
        Dict used to update

    Returns
    -------
    Dictionary
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# ============== #
#  Convertsion   #
# ============== #


def flux_aa_to_hz(flux, wavelength):
    """
    Convert flux in __/Angstrom to __/Hertz

    Parameters
    ----------
    flux : float, array
        Flux to convert

    wavelength : float
        (effective) wavelength at which the flux correspond to.

    Returns
    -------
    Float, Array
    """
    return flux * (wavelength**2 / constants.c.to("AA/s").value)


def flux_hz_to_aa(flux, wavelength):
    """ Convert flux in __/Hertz to __/Angstrom

    Parameters
    ----------
    flux : float, array
        Flux to convert

    wavelength : float
        (effective) wavelength at which the flux correspond to.

    Returns
    -------
    Float, Array
    """

    return flux / (wavelength**2 / constants.c.to("AA/s").value)


def flux_aa_to_mjy(flux, wavelength):
    """ Convert flux in __/Angstrom to mJy

    Parameters
    ----------
    flux : float, array
        Flux to convert

    wavelength : float
        (effective) wavelength at which the flux correspond to.

    Returns
    -------
    Float, Array
    """
    return flux_hz_to_mjy(flux_aa_to_hz(flux, wavelength))


def flux_mjy_to_aa(flux, wavelength):
    """ Convert flux in  mJy to __/Angstrom

    Parameters
    ----------
    flux : float, array
        Flux to convert

    wavelength : float
    (effective) wavelength at which the flux correspond to.

    Returns
    -------
    Float, Array
    """
    return flux_mjy_to_hz(flux_hz_to_aa(flux, wavelength))


def flux_hz_to_mjy(flux):
    """
    Convert flux in erg/s/cm^2/Hertz to mJy

    Parameters
    ----------
    flux : float, array
         flux to convert

    Return
    ---------
    Float, Array
    """

    return flux*10**26


def flux_mjy_to_hz(flux):
    """
    Convert flux in mJy to erg/s/cm^2/Hertz

    Parameters
    ----------
    flux : float, array
        Flux to convert

    Returns
    -------
    Float, Array
    """

    return flux*10**-26


def spectra_to_3dcube(spectra, lbda, spx_map, spx_vert=None, **kwargs):
    """ Build 3d cube with pyifu method.

    Parameters
    ----------
    spec: array
        Spectre of shape NxM where N is the number of spectra (spaxels), and M the number of wavelength (slices)

    lbda: array
        Array of wavelength of shape M. Must match the 2nd dimension of spec.

    spx_map: dict
        Spaxel mapping of length N. Must match the 1st dimension of spec

    spx_vert: array
        Spaxel vertices, give the information of the spaxel shape (square, hexagonal etc)

    Returns
    -------
    pyifu.Cube
    """

    if len(spec) != len(spx_map):
        raise ValueError("Shapes of spec and spaxel_mapping do not match.")

    return pyifu.spectroscopy.get_cube(data=spectra.T, lbda=lbda,
                                       spaxel_mapping=spx_map,
                                       spaxel_vertices=spx_vert, **kwargs)


def gauss_convolve_variable_width(a, sig, prec=10.):
    '''
    approximate convolution with a kernel that varies along the spectral
        direction, by stretching the data by the inverse of the kernel's
        width at a given position
    N.B.: this is an approximation to the proper operation, which
        involves iterating over each pixel of each template and
        performing ~10^6 convolution operations
    Parameters:
     - a: N-D array; convolution will occur along the final axis
     - sig: 1-D array (must have same length as the final axis of a);
        describes the varying width of the kernel
     - prec: precision argument. When higher, oversampling is more thorough
    '''

    assert (len(sig) == a.shape[-1]), '\tLast dimension of `a` must equal \
        length of `sig` (each element of a must have a convolution width)'

    sig0 = sig.max()  # the "base" width that results in minimal blurring
    # if prec = 1, just use sig0 as base.

    n = np.rint(prec * sig0/sig).astype(int)
    # print(n.min())
    #print('\tWarped array length: {}'.format(n.sum()))
    # define "warped" array a_w with n[i] instances of a[:,:,i]
    a_w = np.repeat(a, n, axis=-1)
    # now a "warped" array sig_w
    sig_w = np.repeat(sig, n)
    # define start and endpoints for each value
    nl = np.cumsum(np.insert(n, 0, 0))[:-1]
    nr = np.cumsum(n)
    # now the middle of the interval
    nm = np.rint(np.median(np.column_stack((nl, nr)), axis=1)).astype(int)

    # print nm

    # print a_w.shape, sig_w.shape # check against n.sum()

    # now convolve the whole thing with a Gaussian of width sig0
    print('\tCONVOLVE...')
    # account for the increased precision required
    a_w_f = np.empty_like(a_w)
    # have to iterate over the rows and columns, to avoid MemoryError
    # c = 0  # counter (don't judge me, it was early in the morning)
    # return(a_w)
    for i in range(a_w_f.shape[0]):
        # for j in range(a_w_f.shape[1]):
        '''c += 1
        print '\t\tComputing convolution {} of {}...'.format(
            c, a_w_f.shape[0] * a_w_f.shape[1])'''
        # a_w_f[i,] = ndimage.gaussian_filter1d(a_w[i,], prec*sig0)
        a_w_f[i] = convolve(a_w[i], Gaussian1DKernel(prec*sig0),
                            boundary='extend')
    # print a_w_f.shape # should be the same as the original shape

    # and downselect the pixels (take approximate middle of each slice)
    # f is a mask that will be built and applied to a_w_f

    # un-warp the newly-convolved array by selecting only the slices
    # in dimension -1 that are in nm
    a_f = a_w_f[:,  nm]

    return a_f


# def sedm_lsf(lbda, a=23.585, b=-5.394, slice_unit=True):
#    """ """
#    if slice_unit:
#        from pysedm.sedm import SEDM_LBDA
#        return (a*lbda/6000+b)/(SEDM_LBDA[1]-SEDM_LBDA[0])
#    else:
#        return (a*lbda/6000+b)

def sedm_lsf(lbda, a0=19.100, a=24.629, b=8.202, slice_unit=True):
    """ """
    from pysedm.sedm import SEDM_LBDA
    lbda = np.atleast_1d(lbda)
    coeffs = np.array([a0, a, b])
    nx = 2*lbda/SEDM_LBDA[-1] - 1
    if slice_unit:
        return np.polynomial.legendre.legval(nx, coeffs)/(SEDM_LBDA[1]-SEDM_LBDA[0])
    else:
        return np.polynomial.legendre.legval(nx, coeffs)
