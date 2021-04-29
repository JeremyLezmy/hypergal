#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Filename:          testscript.py
# Description:       script description
# Author:            Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>
# Author:            $Author: jlezmy $
# Created on:        $Date: 2021/01/31 15:03:03 $
# Modified on:       2021/04/29 10:35:12
# Copyright:         2019, Jeremy Lezmy
# $Id: testscript.py, 2021/01/31 15:03:03  JL $
################################################################################

"""
.. _testscript.py:

testscript.py
==============


"""
__license__ = "2019, Jeremy Lezmy"
__docformat__ = 'reStructuredText'
__author__ = 'Jeremy Lezmy <jeremy.lezmy@ipnl.in2p3.fr>'
__date__ = '2021/01/31 15:03:03'
__adv__ = 'testscript.py'

import os
import sys
import datetime


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from hypergal import Host_removing as Hostrem
from hypergal import sedm_target 
from hypergal import Panstarrs_target as ps1targ
from hypergal import SED_Fitting as sedfit
from hypergal import intrinsec_cube
from hypergal import geometry_tool as geotool
import pandas as pd
import matplotlib.pyplot as plt



import numpy as np

if __name__ == '__main__' :

    
    import argparse


    parser = argparse.ArgumentParser()

    class StoreDictKeyPair(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            my_dict = {}
            for kv in values.split(","):
                k,v = kv.split("=")
                my_dict[k] = v
            setattr(namespace, self.dest, my_dict)


    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
         
    parser.add_argument('-t',"--target",type=str, default = "ZTF20abhrmxh", help="target name, for instance ZTF20abhrmxh")
    parser.add_argument('-n',"--night",type=str, default = "20200703", help="night of the observation, for instance 20200703")
    parser.add_argument('-obh',"--obs_hour",type=str, default = "09_58_56", help="hour of the observation, format : HH_MM_SS" )
    parser.add_argument('-cp',"--cube_path",type=str, default = 'default', help="Cube path to load in case you want to use a specific one (for instance a cube where you first removed some spaxels)" )

    parser.add_argument("--cal_air", default = 'header', help="airmass to use for flux calibration. Default is the header one" )
    parser.add_argument("--byecr", type=str2bool, nargs='?', const=True, default=True, help="Apply Byecr first?")
    parser.add_argument("--sbyecr", type=str2bool, nargs='?', const=True, default=True, help="Save Byecr cube?")
    
    parser.add_argument('-z',"--redshift",type=float, default = None, help="redshift of the target, default is query from fritz")
    parser.add_argument('-sedtar', "--IFU_target_coord", nargs=2, type=float, default = None, help="target's coord in spaxel unit in the sedm ifu. Default is given by the astrometry")
    parser.add_argument("--IFU_ratio", nargs=1, type=float, default = 2.235)
    
    parser.add_argument('-ph',"--photosource", type=str, default = 'Panstarrs', help="Photometric source for the host modeling. Default is Panstarrs")
    parser.add_argument('-ps1size', "--PS1_size", type=int, default = 140, help="Size in pixels of the cutout loaded from PS1. Default is 150 pix ~ 37.5 arcs ")
    parser.add_argument('-ps1sub',"--PS1_subsample", type=int, default = 2, help="Bin of the ps1 grid, default is 2")

    parser.add_argument('-sedf',"--SED_Fitter", type=str, default = 'cigale', help="Which SEDfitter to use, default is cigale")
    parser.add_argument('-specdir',"--spec_dirout", type=str, default = 'default', help="Name of the directory to store the .spec files in lepharework/pylephare/. Default is in lepharework/pylephare/*actual_time*.")
    parser.add_argument('-sedfdir',"--SED_Fitter_dirout", type=str, default = 'none', help="In case spec already has been computed, path of the saved spec")
    parser.add_argument("--sedfdatdirout", type=str, default = None, help="Where to store the sampled spectrum?")
    parser.add_argument("--mod_cig", type=str, default = 'default', help="Modules to use (stellar, nebular, attenuation etc... for cigale")
    parser.add_argument("--path_cig", type=str, default = None, help="Where to store the cigale files (txt file with input, pcigal.ini, pcigal.ini.spec, and output diretory). ")
    parser.add_argument("--out_dir_cig", type=str, default ='out/', help="Name of the output directory with cigale fit ")
    
    parser.add_argument("--snr", type=float, default = 3, help="Sig/noise ratio selection for the computation of the spectra with Sedfitter. Default is 3")

    parser.add_argument('-psffit',"--psfmodel_fit", type=str, default = 'Gauss_Mof_kernel', help="Which psf model to apply on the host modeling, default is Gaussian + Moffat " )

    parser.add_argument('-pih', "--plot_init_hexagrid",type=str2bool, nargs='?', const=True, default=False, help="show initial hexagrid over the photo image?")

    parser.add_argument('-nc', "--nb_process", default = 'auto' , help="How many core for the multiprocessing computation? Default is quantity of availabe core - 2" )
    parser.add_argument("-f","--fit", type=str2bool, nargs='?', const=True, default=True, help="Run the fit?")

    parser.add_argument('-lrf',"--lbda_range_fit", nargs=2, type=float, default=[4500,8500], help="Lambda range in AA considered for the fit. Default is [4500,8500]")
    parser.add_argument('-mf',"--metaslices_fit", type=int, default=5, help="Quantity of metaslices considered for the fit. Default is 5")

    parser.add_argument("-fv","--set_fit_values", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2..." , default=None, help="Set of values to give if you already know the best parameters. If so, you can skip the fit by setting -f False. ")

    parser.add_argument("-mdir","--model_dirout", nargs=1, type=str, default='default', help="where to save the model? Default is in the sedm cube directory")
    parser.add_argument("-rdir","--residu_dirout", nargs=1, type=str, default='default', help="where to save the residu? Default is in the sedm cube directory")
    
    parser.add_argument('-extr', "--extractstar",type=str2bool, nargs='?', const=True, default=True, help="run the extractstar?")

    parser.add_argument('-extr_l', "--extractstar_lbda",type=float, nargs=2, default=[4500,7500], help="lbda range for extractstar")
    parser.add_argument('-extr_b', "--extractstar_bins",type=int, nargs=1, default=6, help="nb metaslices for extractstar")

    parser.add_argument('-extr_c', "--extractstar_centroid",type=str, nargs=1, default='fitted', help="init centroid extractstar")


    parser.add_argument('-shfo', "--show_full",type=str2bool, nargs='?', const=True, default=True, help="show full output?")
    parser.add_argument("--slid",type=int,  default=2, help="sliceid to show in the extracted spectra")
    parser.add_argument('-sfod', "--save_full_output_dir",type=str, default='default', help="where to save the final output png (with name of the file) ")
    
    
    
    



    

    args = parser.parse_args()


    sedm_base = sedm_target.SEDM_tools( args.target, args.night, args.obs_hour)

    if args.byecr ==True:
        cube_origin = sedm_base.get_cube(path=args.cube_path)
        cube_byecr = sedm_base.get_byecr_cube( cube_origin, save=args.sbyecr)
        sedm_base.get_calib_cube( cube_byecr, which_airmass=args.cal_air)

    else :
        sedm_base.get_calib_cube(path=args.cube_path, which_airmass=args.cal_air)

    if args.redshift==None:
        redshift = sedm_base.get_Host_redshift_fritz()
    else:
        redshift = args.redshift

    if args.IFU_target_coord==None:
        IFU_target = sedm_base.get_estimate_target_coord()
    else:
        IFU_target = args.IFU_target_coord

    init_adr = sedm_base.cube_cal.adr.copy()

    ra, dec = ps1targ.HR_to_deg(sedm_base.cube.header['OBJRA'],sedm_base.cube.header['OBJDEC'] )
    

    if args.photosource=='Panstarrs':

        photo_s = ps1targ.Panstarrs_target(ra,dec)
        photo_s.load_cutout( args.PS1_size )
        geodf = photo_s.build_geo_dataframe(subsample = args.PS1_subsample)
        pix_coord_targ = photo_s.rfilter.coords_to_pixel(ra,dec)
        pixelsize = photo_s.get_pix_size()

    if args.SED_Fitter=='Lephare':

        if args.SED_Fitter_dirout is not 'none':

            spec, lbda = np.load(args.SED_Fitter_dirout)['spec'], np.load(args.SED_Fitter_dirout)['lbda']

        else:

            Leph = sedfit.Lephare_SEDfitting(pd.DataFrame(geodf))           
            
            Leph.Setup_Lephare( spec_dirout=args.spec_dirout, Sig_noise_ratio=args.snr, redshift=redshift)
            Leph.run_Lephare()

            spec,lbda = Leph.get_Sample_spectra(save_dirout_data = args.sedfdatdirout)

            #####PLOT POSSIBILITIES???


    if args.SED_Fitter=='cigale':

        if args.SED_Fitter_dirout is not 'none':

            spec, lbda = np.load(args.SED_Fitter_dirout)['spec'], np.load(args.SED_Fitter_dirout)['lbda']

        else:

            cig_geodf = photo_s.make_cigale_compatible()
            cg = sedfit.Cigale_sed(cig_geodf)
            cg.setup_cigale_df( SNR=args.snr, redshift=redshift)
            cg.initiate_cigale(sed_modules = args.mod_cig, cores = args.nb_process)
            cg.run_cigale( path_result = args.path_cig, result_dir_name=args.out_dir_cig)
            
            spec, lbda = cg.get_Sample_spectra(save_dirout_data = args.sedfdatdirout)
            

    init_hexagrid = geotool.get_cube_grid( sedm_base.cube_cal, scale = args.IFU_ratio, targShift=IFU_target, x0=pix_coord_targ[0], y0=pix_coord_targ[1]  )

    #####PLOT POSSIBILITIES???


    overlaycube = intrinsec_cube.Intrinsec_cube( photo_s.full_grid, pixelsize, init_hexagrid, spec, lbda, args.psfmodel_fit)
    overlaycube.load_adr(init_adr)


    
    hostfitter = Hostrem.Host_removing( sedm_base, overlaycube, pix_coord_targ, IFU_target )

    if args.plot_init_hexagrid:
        fig,ax = plt.subplots()
        geotool.show_Mutipolygon(init_hexagrid,ax=ax)

        photo_s.show(ax=ax ,origin='lower')
        plt.show()
    
    

    if args.fit==True:
        hostfitter.fit( fix_parameters= Hostrem.default_fixed_params, lbda_ranges = args.lbda_range_fit, metaslices=args.metaslices_fit )
        print('fitted_values:', hostfitter.fit_values)

   
    if args.set_fit_values is not None:
        param = hostfitter.init_params_values
        param.update(args.set_fit_values)
        hostfitter.fit_values = param
        print(hostfitter.fit_values)
        for k in hostfitter.fit_values.keys():
            hostfitter.fit_values[k]=float(hostfitter.fit_values[k])
    
    hostfitter.get_fitted_cubes(model_save_dirout=args.model_dirout, residu_save_dirout=args.residu_dirout )


    if args.extractstar:

        hostfitter.extract_star_spectra(  step1range= args.extractstar_lbda, 
                              step1bins=args.extractstar_bins,  centroid=args.extractstar_centroid )

        if args.show_full:

            hostfitter.show_full_output(sliceid=args.slid, savefile_dirout=args.save_full_output_dir )
            plt.show()
    

 

#args = parser.parse_args(sys.argv[1:])




# End of testscript.py ========================================================
