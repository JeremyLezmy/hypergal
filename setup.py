#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          setup.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/02/04 14:20:22 $
# Modified on:       2021/05/21 17:57:54
# Copyright:         2019, Jeremy Lezmy
# $Id: setup.py, 2021/02/04 14:20:22  JL $
################################################################################

"""
.. _setup.py:

setup.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>'
__date__ = '2021/02/04 14:20:22'
__adv__ = 'setup.py'

import os
import sys
import datetime



DESCRIPTION = "hypergal: Python module for Host modeling"
LONG_DESCRIPTION = """ hypergal: Python module for Hyperspectral Galaxy Modeling through SEDfitting and photometric source """

DISTNAME = 'hypergal'
AUTHOR = 'Jeremy Lezmy'
MAINTAINER = 'Jeremy Lezmy' 
MAINTAINER_EMAIL = 'lezmy@ipnl.in2p3.fr'
URL = 'https://gitlab.in2p3.fr/JLezmy/hypergal'
LICENSE = ''
DOWNLOAD_URL = 'https://gitlab.in2p3.fr/JLezmy/hypergal'
VERSION = '1.0.2'

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    try:
        import pathos
    except ImportError:
        install_requires.append('pathos')

    try:
        import ztfquery
    except ImportError:
        install_requires.append('ztfquery')

    try:
        import pyifu
    except ImportError:
        install_requires.append('pyifu')
    
    try:
        import pysedm
    except ImportError:
        install_requires.append('pysedm')
    
    try:
        import astropy
    except ImportError:
        install_requires.append('astropy')

    try:
        import pylephare
    except ImportError:
        install_requires.append('pylephare')
       
     
    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ['hypergal', 'hypergal/scripts','hypergal/photometry','hypergal/spectroscopy','hypergal/utils','hypergal/script' ]

    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=packages,
          scripts=["hypergal/scripts/FullProcess.py" ] ,
          package_data={'hypergal':['config/cigale.json', 'data/filters/*']},
          include_package_data=True,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',              
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
      )




# End of setup.py ========================================================
