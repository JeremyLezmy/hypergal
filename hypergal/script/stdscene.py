import os
import numpy as np
from ..psf import GaussMoffat2D as GM2D
import pyifu
from .daskbasics import DaskHyperGal
from .scenemodel import DaskScene 
from ..fit import SceneFitter, Priors
from ..scene.std import MultiSliceParametersSTD
from .. import io
from .. import psf
import pandas
import os
from dask.distributed import wait
from astropy.io import fits
from pysedm.sedm import SEDM_LBDA
import pysedm
from ..spectroscopy import adr as spectroadr
from dask import delayed
from pysedm import fluxcalibration

def get_and_dw_cubefile_from_whatdata(overwrite=False, download=True, client=None, **kwargs):
        
        from ztfquery import sedm
        s = sedm.SEDMQuery()
        df = s.get_whatdata( std_only=True, **kwargs )
        
        if download:
            dow = s.download_target_cubes( targetname=list(np.unique(df['target'].values)), client=client, ioprop=kwargs, overwrite=overwrite)
            for i in range(len(dow)):
                wait(dow[i])
        cff=[]
        for i in range(len(df.index)):
            path = os.path.join(sedm.SEDMLOCALSOURCE, df.index[i][0], 'e3d_crr_b_'+ df['filename'][i].replace('.fits','')+ '_'+ df['target'][i]+ '.fits')
            cff.append(path)
            if not os.path.isfile(path):
                cff.remove(path)

        return cff

class DaskSTD():
    
    
    @classmethod
    def compute_single(cls, cubefile, lbda_range=[4500,9000], nslices=9, 
                       psfmodel = 'GaussMoffat2D', curved_bkgd=True, 
                       fix_params=None, use_subslice=False, onlyvalid=True,
                       save_metaplot=True, save_result=True, return_result=True, returnspec=True, compute_calib=True):

        info        = io.parse_filename(cubefile)
        cubeid      = info["sedmid"]
        name        = info["name"]
        filedir     = os.path.dirname(cubefile)       
        
        header = fits.getheader(cubefile)

        # PLOTS
        plotbase    = os.path.join(filedir, "hypergal", info["name"], info["sedmid"])
        dirplotbase = os.path.dirname(plotbase)
        if not os.path.isdir(dirplotbase):
            os.makedirs(dirplotbase, exist_ok=True)
        
        #stdcube = delayed(DaskScene.remove_out_spaxels)(DaskHyperGal.get_calibrated_cube(cubefile, as_wcscube=False) )
        
        cube = delayed( pysedm.get_sedmcube)(cubefile, apply_byecr=True)
        stdcube = delayed(DaskScene.remove_out_spaxels)( cube )

        thickness = np.diff(lbda_range)/nslices

        std_meta = delayed(stdcube.to_metacube)(lbda_range,nslices)
        
        header_df = delayed(pandas.DataFrame)( dict(header), index=[cubeid+'_'+name])
        
        stored = []
        
        savefile = None if save_metaplot is False else os.path.join(dirplotbase, f"fit_metaslices_")
    
        best_meta_fit = cls.fit_std_cube( std_meta, nslices, 
                                          psfmodel, None, thickness, use_subslice,
                                          curved_bkgd, onlyvalid, fix_params, savefile)

        
        if not returnspec:
            if save_result:
                stored.append( best_meta_fit.to_hdf( *(os.path.join(dirplotbase, 'hgout_STD_' + cubeid + '_' + name +'.h5'), 'meta_slicefit') ))
                stored.append( header_df.to_hdf( *(os.path.join(dirplotbase, 'hgout_STD_' + cubeid + '_' + name +'.h5'), 'header') ))
                return stored

            if return_result: 
                stored.append( best_meta_fit)
                stored.append(header_df)

            return stored
        
        else:
            
            if save_result:
                stored.append( best_meta_fit.to_hdf(*io.get_slicefit_datafile(cubefile, "meta")) )
            if return_result: 
                stored.append( best_meta_fit)
            
            saveplot_psf = plotbase + '_' + name + '_psf3d_fit.png'
            meta_ms_param = delayed(MultiSliceParametersSTD)(best_meta_fit, cubefile=cubefile, 
                                                         pointsourcemodel='GaussMoffat3D',
                                                             load_adr=True, load_pointsource=True, saveplot_adr = plotbase + "_adr_fit.png", saveplot_psf = saveplot_psf)
            
            bestfit_completfit = cls.fit_std_cube(stdcube, nslices=len(SEDM_LBDA),
                                        mslice_param=meta_ms_param,  psfmodel=psfmodel,  curved_bkgd=curved_bkgd,
                                        fix_params=['a_ps', 'b_ps', 'alpha_ps', 'eta_ps'])
            
            if save_result:
                stored.append( bestfit_completfit.to_hdf(*io.get_slicefit_datafile(cubefile, "full")) )
            if return_result: 
                stored.append(bestfit_completfit)
                
            stored.append( cls.get_target_spec(bestfit_completfit, delayed(fits.getheader)(cubefile), savefile=io.e3dfilename_to_hgspec(cubefile, 'target') ) )
            
            
            if compute_calib:
                
                
                cube = delayed( pysedm.get_sedmcube)(cubefile)
                values = delayed(best_meta_fit.unstack)()['values']
                errors=delayed(best_meta_fit.unstack)()['errors']
                lbda = delayed(best_meta_fit.unstack)()['values']['lbda']
                adr,x = delayed(spectroadr.ADRFitter.fit_adr_from_values, nout=2)(values=values, lbda=lbda, errors = errors,filename_or_header = delayed(fits.getheader)(cubefile))
                airmass = adr.airmass
                savefluxcal = os.path.join(os.path.dirname(cubefile), 'fluxcal_hypergal_' + info["sedmid"] + '_' + name + '.fits')
                
                stored.append(cls.get_fluxcalib( spobj=cls.get_target_spec(bestfit_completfit, delayed(fits.getheader)(cubefile)), airmass=airmass, cubefile=cubefile
                                            , savefluxcal=savefluxcal, saveplot=None))
            
            return stored
    
    
    @classmethod
    def compute_multiple(cls, cubefiles, lbda_range=[4500,9000], nslices=9, 
                       psfmodel = 'GaussMoffat2D', curved_bkgd=True, 
                       fix_params=None, use_subslice=False, onlyvalid=True,
                       save_metaplot=False, save_result=False, return_result=True, concat=True, returnspec=True, compute_calib=False):
        
        keys = []
        fit=[]
        objinfo=[]
        stored=[]
        for cubefile in cubefiles:
            
            info        = io.parse_filename(cubefile)
            cubeid      = info["sedmid"]
            name        = info["name"]
            filedir     = os.path.dirname(cubefile)       

            keys.append(cubeid+'_'+name)
            single = cls.compute_single( cubefile, lbda_range=lbda_range, nslices=nslices, 
                       psfmodel = psfmodel, curved_bkgd=curved_bkgd, 
                       fix_params=fix_params, use_subslice=use_subslice, onlyvalid=onlyvalid,
                       save_metaplot=save_metaplot, save_result=save_result, return_result=return_result, returnspec=returnspec)
            
                              
            if returnspec:
                stored.append(single)
                              
            elif not returnspec:    
                fit.append(single[0])
                objinfo.append(single[1])
                              
        
        if returnspec:
            return stored
                              
        elif not returnspec and concat:
            stored.append(delayed(pandas.concat)(fit, axis=0, keys=keys))
            stored.append(delayed(pandas.concat)(objinfo, axis=0))

            return stored
        
        else:
            stored.append(fit)
            stored.append(objinfo)
            return stored

    @classmethod
    def fit_std_cube(cls, stdcube, nslices, psfmodel, mslice_param=None, thickness=None,
                     use_subslice=False, curved_bkgd=True, onlyvalid=True,
                     fix_params=None, save_metaplot=None):
    
        best_fits = {}
        
        for i_ in range(nslices):
            # the slices
            std_slice   = stdcube.get_slice(index=i_, slice_object=True)
            
            savefile  = None if save_metaplot is None else save_metaplot +fr'{i_}.png'
            if mslice_param is not None:
                guess = mslice_param.get_guess(std_slice.lbda, squeeze=True)
            else:
                guess = {}
           
            if type(psfmodel)==str:
                psf_ = getattr(psf,psfmodel)()
            else:
                psf_ = psfmodel
        
            best_fits[np.round(std_slice.lbda,2)] = delayed(SceneFitter.fit_std)( std_slice=std_slice, 
                                                                                   psf=psf_, 
                                                                                   lbda_thickness=thickness, header=stdcube.header,
                                                                                   use_subslice=use_subslice, 
                                                                                    guess=guess,
                                                                                   curved_bkgd=curved_bkgd, onlyvalid=onlyvalid ,
                                                                                   savefile=savefile, fix_params=fix_params)
            
        return delayed(pandas.concat)(best_fits)
            
        
        
    
    
    @staticmethod
    def get_target_spec(fullfit, header, savefile=None):
        specval = delayed(fullfit.xs)('ampl_ps', level=1)['values'].values
        specerr = delayed(fullfit.xs)('ampl_ps', level=1)['errors'].values
        speclbda = delayed(fullfit.xs)('lbda', level=1)['values'].values
        speccoef = delayed(fullfit.xs)('norm_comp', level=1)['values'].values
        specfi = delayed(pyifu.spectroscopy.get_spectrum)(speclbda, specval*speccoef/header['EXPTIME'], (specerr*speccoef/header['EXPTIME'])**2, header)
        
        if savefile != None:        
            if savefile.rsplit('.')[-1]=='txt':
                asci = True
            elif savefile.rsplit('.')[-1]=='fits':
                asci = False
            specfi = delayed(specfi.writeto)(savefile,  ascii=asci)
        
        return(specfi)
    
    
    
    @staticmethod
    def get_fluxcalib( spobj, airmass, cubefile, savefluxcal, saveplot=None):
        from pysedm import fluxcalibration
        newsp, fl = delayed(fluxcalibration.get_fluxcalibrator, nout=2)( spobj)
        spobj = delayed(DaskSTD.apply_fluxcal)( spobj, airmass, newsp)
        
        if savefluxcal is not None:
            return delayed(fl.fluxcalspectrum.writeto)(savefluxcal)
        
        if saveplot is not None and savefluxcal is not None:
            return delayed(show_fluxcalibrated_standard)(spobj, saveplot), delayed(fl.fluxcalspectrum.writeto)(savefluxcal)
        
        if saveplot is not None:
            return delayed(show_fluxcalibrated_standard)(spobj, saveplot)
        
        
    @staticmethod
    def apply_fluxcal(specobj, airmass, fl):
        
        sp = specobj.copy()
        sp.scale_by( fl.get_inversed_sensitivity(airmass))
        
        return sp
        
        
        
        
        
def show_fluxcalibrated_standard(stdspectrum, savefile=None):
    """ """
    AVOIDANCE_AREA = {"telluric":[[7450,7750],[6850,7050]],
                 "absorption":[[6400,6650],[4810,4910],
                                [4300,4350],[4030,4130],[3900,3980]]}

    POLYDEGREE = 40
    import numpy as np
    import warnings
    from propobject import BaseObject
    from pyifu.spectroscopy import Spectrum, get_spectrum
    from pysedm.sedm import get_sedm_version
    import matplotlib.pyplot as mpl
    import pycalspec
    from astropy import units
    from pysedm import io
    ### Data
    
    
    objectname = stdspectrum.header['OBJECT'].replace("STD-","").split()[0]
    #
    try:
        specref = pycalspec.std_spectrum(objectname).filter(5).reshape(stdspectrum.lbda,"linear")
    except IOError:
        print("WARNING: ref flux not found so no flux scale plot possible")
        return None
    specres = get_spectrum(stdspectrum.lbda, specref.data / stdspectrum.data )
    scale_ratio = specres.data.mean()
    specres.scale_by(scale_ratio)
    #
    try:
        dtime = io.filename_to_time( stdspectrum.filename) - io.filename_to_time( stdspectrum.header["CALSRC"] )
    except:
        dtime = None
    ###

    fig = mpl.figure(figsize=[12,12], dpi=90)

    ax   = fig.add_axes([0.13,0.43,0.8,0.5])
    axr  = fig.add_axes([0.13,0.13,0.8,0.28])

    stdspectrum.show(ax=ax, label="observation", show=False)#, yscalefill=True)
    specref.show(ax=ax, color="C1", label="calspec", show=False)#, yscalefill=True)

    axr.axhline(1, ls="--", color="0.5")
    axr.axhspan(0.9,1.1, color="0.5", alpha=0.1)
    axr.axhspan(0.8,1.2, color="0.5", alpha=0.1)
    # Residual
    specres.show(ax=axr, color="k", lw=1.5, show=False)

    for l in AVOIDANCE_AREA["telluric"]: 
        ax.axvspan(l[0],l[1], color=mpl.cm.Blues(0.6), alpha=0.2, zorder=0)
        axr.axvspan(l[0],l[1], color=mpl.cm.Blues(0.6), alpha=0.2, zorder=0)
        
    axr.text(0.02,0.92, "scale ratio: %.1f"%scale_ratio, transform=axr.transAxes,
                va="top", ha="left", backgroundcolor= mpl.cm.binary(0.0001,0.8))

    ax.set_xlabel(["" for l in ax.get_xlabel()])
    ax.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ A$^{-1}$]")
    if dtime is not None:
        ax.set_title( "%s"%objectname+ r" | t$_{obs}$ - t$_{fluxcal}$ : %.2f hours"%((dtime.sec * units.second).to("hour")).value)
    #else:
        #ax.set_title( 'STD-'+"%s"%objectname + ' | EXPTIME: ' + str(stdspectrum.header['EXPTIME'])  + 's | (2020-08-31, ID: 12-22-57)')

    axr.set_ylabel(r"Flux ratio")
    axr.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_yscale("log")
    ax.set_ylim(0.8*np.min(specref.data))
    ax.legend(loc="lower left")
    fig.suptitle( 'Fluxcalibrated spectra with new PSF model for STD-'+"%s"%objectname +'\n'+ ' | EXPTIME: ' + str(stdspectrum.header['EXPTIME'])  + f's | ( {stdspectrum.header["OBSDATE"]} , ID:{stdspectrum.header["OBSTIME"].rsplit(".")[0].replace(":","-")})',fontsize=16, fontweight="bold")

    if savefile:
        fig.savefig(savefile)
    else:
        return fig
