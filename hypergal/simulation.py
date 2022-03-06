import pysedm
import pandas as pd
import numpy as np
from .spectroscopy import adr as spectroadr
from .spectroscopy import WCSCube
from .scene import basics as basescene
from .psf import GaussMoffat3D
from . import photometry
from pysedm import astrometry, fluxcalibration
from hypergal import io as ioh
import warnings
from pysedm.io import parse_filename
from shapely.geometry import Polygon, Point
from hypergal.script import scenemodel
from pysedm.dask import base
from .photometry import basics as photobasics
from matplotlib import pyplot as plt


def load_simulation_parameters(**kwargs):
    """ """
    import os
    from . import _PACKAGE_ROOT

    file_ = os.path.join(
        _PACKAGE_ROOT, f"data/simulation/simulation_inputs.csv")
    if not os.path.isfile(file_):
        raise IOError(f"not such file {file_}")
    return pd.read_csv(file_, index_col=0, **kwargs)


def get_spectra(basename, spec_type, **kwargs):
    """ """
    import os
    from . import _PACKAGE_ROOT

    fold = spec_type + '_spectra'
    file_ = os.path.join(
        _PACKAGE_ROOT, f"data/simulation/{spec_type}/{basename}")
    if not os.path.isfile(file_):
        raise IOError(f"not such file {file_}")
    return np.loadtxt(file_)


def get_host_list(**kwargs):
    """ """
    import os
    from . import _PACKAGE_ROOT

    file_ = os.path.join(
        _PACKAGE_ROOT, f"data/simulation/Pure_Host_simulation.txt")
    if not os.path.isfile(file_):
        raise IOError(f"not such file {file_}")
    return np.loadtxt(file_, dtype='str')


class SimulationHypergal():

    IFU_SCALE = 0.558

    def __init__(self, modelspectra_filename, purehost_filename, psfparams=None, load_source_df=True):
        """ """

        pure_host = scenemodel.DaskScene.remove_out_spaxels(
            pysedm.get_sedmcube(purehost_filename))
        model_spectra = np.loadtxt(modelspectra_filename)
        if np.shape(model_spectra)[0] == 214:
            model_spectra = self.fix_spectra(model_spectra)
        self.set_metafithost(self.get_metafit_purehost(purehost_filename))
        self.set_modelspectra(model_spectra)
        self.set_purehost(pure_host)
        self.set_psfparams(psfparams)
        if load_source_df:
            self.load_source_df(purehost_filename)
        self.pure_host.header.update({'SIMU_SN': modelspectra_filename})

    @classmethod
    def simulate(cls, modelspectra_filename, purehost_filename, psfparams, host_distance, contrast, savecube=None):

        this = cls(modelspectra_filename, purehost_filename, psfparams)
        this.get_random_sn_position(host_distance)
        this.get_coeff_spectra(contrast=contrast)
        _, _ = this.get_poisson_sn_simu()
        simulated_cube = this.get_simulated_cube(savecube=savecube)

        return this

    @staticmethod
    def fix_spectra(model_spectra):
        """ """

        miss = [l for l in np.round(
            pysedm.sedm.SEDM_LBDA, 1) if l not in model_spectra.T[0]]

        a0, b0 = np.polyfit(model_spectra.T[0][:2], model_spectra.T[1][:2], 1)
        a1, b1 = np.polyfit(model_spectra.T[0][-2:], model_spectra.T[1][-2:], 1)

        newlbda = np.insert(model_spectra.T[0], 0, miss[:3])
        newlbda = np.concatenate((newlbda, np.array(miss[-3:])))
        newval = np.insert(model_spectra.T[1], 0, np.array(miss[:3])*a0+b0)
        newval = np.concatenate((newval,  np.array(miss[-3:])*a1+b1))
        newvar = np.insert(model_spectra.T[2], 0, np.array(newval[:3])**2)
        newvar = np.concatenate((newvar, np.array(newval[-3:])**2))

        fix_spec = np.array([newlbda, newval, newvar]).T
        return fix_spec

    def load_source_df(self, cubefile):
        """ """
        path = cubefile
        info = parse_filename(path)
        contain = info["sedmid"]
        targ = info["name"]

        from ztfquery import sedm, fritz
        try:
            fsource = fritz.FritzSource.from_name(targ)
        except OSError:
            warnings.warn(
                f"The target {targ} doesn't exist in Fritz!")
            return [], None, None
        radec = fsource.get_coordinates()
        cutouts = photometry.PS1CutOuts.from_radec(
            *radec, filters=["ps1.r", "ps1.i"], size=100)
        cutcube = cutouts.to_cube(binfactor=2)
        sources = cutouts.extract_sources(filter_="ps1.i", thres=40,
                                          savefile=None)
        source = sources[np.sqrt((sources.x-cutouts.radec_to_xy(*radec)[0])**2+(sources.y-cutouts.radec_to_xy(*radec)[1])**2) == np.min(
            np.sqrt((sources.x-cutouts.radec_to_xy(*radec)[0])**2+(sources.y-cutouts.radec_to_xy(*radec)[1])**2))]

        self._sourcedf = source

        adr = self.get_adr()

        wcscube = WCSCube.from_sedmcube(
            self.pure_host, radec, [adr._fit_xref, adr._fit_yref])
        host_pos = wcscube.wcs.all_world2pix(cutcube.wcs.all_pix2world(np.array(
            [source.x.iloc[0], source.y.iloc[0]])[:, None].T, 0), 0)[0]

        self._hostcoord = host_pos
        self._wcscube = wcscube

    def get_random_sn_position(self, sn_distance_from_host, set_header=True):
        """ """

        self.set_sn_distance_from_host(sn_distance_from_host)
        r = sn_distance_from_host
        theta = np.random.random() * 2 * np.pi
        x = self.hostcoord[0] + r * np.cos(theta)
        y = self.hostcoord[1] + r * np.sin(theta)
        host_fromcenter = np.sqrt(np.sum(self.hostcoord**2))
        circle_lim = Point(0, 0).buffer(host_fromcenter+1)

        n = 0
        while not circle_lim.contains(Point(x, y)):
            r = sn_distance_from_host
            theta = np.random.random() * 2 * np.pi
            x = self.hostcoord[0] + r * np.cos(theta)
            y = self.hostcoord[1] + r * np.sin(theta)
            host_fromcenter = np.sqrt(np.sum(self.hostcoord**2))
            circle_lim = Point(0, 0).buffer(host_fromcenter+1)
            n += 1
            if n > 50:
                warnings.warn(
                    f" After {n} try, did not find sn position with the required condition in the IFU. Change the distance_from_host parameter")
                return None
        self._sim_sn_pos = np.array([x, y])
        if set_header:
            self.pure_host.header.update({'SIMU_X': x})
            self.pure_host.header.update({'SIMU_Y': y})
            if hasattr(self, 'wcscube'):
                radec = self.wcscube.xy_to_radec(*np.array([x, y]))
                self.pure_host.header.update(
                    {'SIMU_RA': list(radec.squeeze())[0]})
                self.pure_host.header.update(
                    {'SIMU_DEC': list(radec.squeeze())[1]})
        return np.array([x, y])

    def get_adr(self, lbdaref=6000):
        """ """
        x0 = self.metafithost.xs('xoff', level=1)['values'].values
        x0err = self.metafithost.xs('xoff', level=1)['errors'].values
        y0 = self.metafithost.xs('yoff', level=1)['values'].values
        y0err = self.metafithost.xs('yoff', level=1)['errors'].values
        lbda = self.metafithost.xs('lbda', level=1)['values'].values

        ADRFitter = spectroadr.ADRFitter(xpos=x0, ypos=y0, xpos_err=x0err, ypos_err=y0err,
                                         lbda=lbda, init_adr=spectroadr.ADR.from_header(self.pure_host.header), lbdaref=lbdaref)

        ADRFitter.fit_adr()
        self._adr = ADRFitter
        return ADRFitter

    def get_refracted_pos(self, xref, yref, adr=None, lbda=None):
        """ """
        if adr == None:
            adr = self.get_adr()

        if lbda == None:
            lbda = self.pure_host.lbda

        refracted = adr.refract(
            xref, yref, lbda, unit=self.IFU_SCALE)

        return refracted

    def load_psf3d(self, xref, yref, adr=None, psfparams=None):
        """ """
        dic = self.metafithost.unstack().loc[2].loc['values'].to_dict()
        if adr == None:
            adr = self.get_adr()

        psfmodel = GaussMoffat3D()
        params = dict({k.replace('_ps', ''): v for k, v in dic.items()
                      if k.replace('_ps', '') in psfmodel.PARAMETER_NAMES})

        psf3d = basescene.PointSource3D.from_adr(psfmodel, self.pure_host.get_spaxel_polygon(
            format='multipolygon'), lbda=self.pure_host.lbda, adr=adr, xref=xref, yref=yref, spaxel_comp_unit=self.IFU_SCALE, **params)
        #keys = [ k for k in psf3d.BASE_PS_PARAMETERS if 'ampl' in k]

        if psfparams is not None:
            psf3d.psf.update_parameters(**psfparams)
        self._psf3d = psf3d
        _ = self.get_fwhm()

    def get_fwhm(self, lbda=None):
        """ """
        xgrid, ygrid = np.meshgrid(np.linspace(
            0., 10, 100), np.linspace(0., 10, 100))
        xc, yc = 0, 0
        dx = xgrid-xc
        dy = ygrid-yc
        rgrid = np.sqrt(dx**2 + self.psf3d.psf.a_ell*dy **
                        2 + 2*self.psf3d.psf.b_ell * (dx*dy))

        radiusrav = rgrid.ravel()
        radiusrav.sort()
        radiusrav = radiusrav[::-1]
        if lbda is None:
            lbda = self.adr.lbdaref
        prof = self.psf3d.psf.get_radial_profile(radiusrav, lbda)
        fwhm = radiusrav[np.where(abs(prof/np.max(prof) - 0.5) ==
                                  np.min(abs(prof/np.max(prof) - 0.5)))[0]][0]*2*self.IFU_SCALE

        self._fwhm = np.round(fwhm, 2)
        return fwhm

    def get_coeff_spectra(self, contrast=None, psfparams=None, set_header=True, set_sn_adu=True, apply_savgol=True, windows=5, degree=3):
        """ """
        if contrast != None:
            self.set_contrast(contrast)
        if psfparams != None:
            self.set_psfparams(psfparams)

        refracted = self.get_refracted_pos(*self.sim_sn_pos)
        self.load_psf3d(*self.sim_sn_pos, psfparams=self._psfparams)
        fluxcalfile = self.fluxcalfile
        fluxcal = fluxcalibration.load_fluxcal_spectrum(fluxcalfile)

        specampl = self.model_spectra.T[1]*self.pure_host.header['EXPTIME']

        if apply_savgol:
            from scipy.signal import savgol_filter
            specampl = savgol_filter(specampl, windows, degree)
            specampl[specampl < 0] = np.array(
                self.model_spectra.T[1]*self.pure_host.header['EXPTIME'])[specampl < 0]
            self._savgol_applied = True

        sn_mod = self.psf3d.get_model(
            xoff=refracted[0], yoff=refracted[1], ampl=specampl, **self._psfparams)

        calcube = self.pure_host.copy()
        calcube.scale_by(fluxcal.get_inversed_sensitivity(
            self.pure_host.header.get("AIRMASS", 1)))

        newcu = calcube.get_new(newdata=calcube.data*(sn_mod/np.tile(
            np.max(sn_mod, axis=1)[np.newaxis:], (len(calcube.data.T), 1)).T))

        host_photo = np.nanmean(newcu.get_slice(lbda_trans=photobasics.get_filter('ztf.r', as_dataframe=False),
                                                slice_object=True).data)

        sncube = calcube.get_new(newdata=sn_mod)
        sn_photo = np.nanmean(sncube.get_slice(lbda_trans=photobasics.get_filter('ztf.r', as_dataframe=False),
                                               slice_object=True).data)

        integral_sn = (host_photo*self.contrast)/(1-self.contrast)
        coeff_sn = integral_sn/sn_photo
        if set_header:
            self.pure_host.header.update({'SN_COEFF': coeff_sn})
            self.pure_host.header.update({'CONTRAST': self.contrast})
            self.pure_host.header.update(**self.psf3d.psf.parameters)
            self.pure_host.header.update(
                **{'HG'+k[:6]: v for (k, v) in self.adr.data.items()})
            self.pure_host.header.update({'SAVGOL': apply_savgol})
            if apply_savgol:
                self.pure_host.header.update({'SAVGOL_W': windows})
                self.pure_host.header.update({'SAVGOL_D': degree})
        if set_sn_adu:
            snsimu_adu = sn_mod.T * \
                fluxcal.get_inversed_sensitivity(
                    self.pure_host.header.get("AIRMASS", 1))
            self._sn_adu = snsimu_adu.T*coeff_sn

    def get_poisson_sn_simu(self, sn_adu=None, return_var=True):
        """ """
        if sn_adu == None:
            sn_adu = self.sn_adu

        poisson_simu = np.random.poisson(sn_adu)
        newvariance = sn_adu
        self._sn_poisson = poisson_simu
        self._sn_poisson_var = newvariance

        sncube = self.pure_host.get_new(
            newdata=poisson_simu, newvariance=newvariance)

        varcube = self.pure_host.get_new(
            newdata=(self.pure_host.variance + sncube.data))

        lbda_filter = np.average(photobasics.get_filter('ztf.r', as_dataframe=False)[
                                 0], weights=photobasics.get_filter('ztf.r', as_dataframe=False)[1])

        circle_lim = Point(*self.adr.refract(*self.sim_sn_pos,
                           lbda_filter)).buffer(2*self.fwhm/(2*self.IFU_SCALE))
        varcube_sub = varcube.get_partial_cube(
            varcube.get_spaxels_within_polygon(circle_lim), np.arange(len(varcube.lbda)))
        sncube_sub = sncube.get_partial_cube(
            sncube.get_spaxels_within_polygon(circle_lim), np.arange(len(sncube.lbda)))

        sn_slice_photo = sncube_sub.get_slice(lbda_trans=photobasics.get_filter('ztf.r', as_dataframe=False),
                                              slice_object=True, )

        varcube_slice_photo = varcube_sub.get_slice(lbda_trans=photobasics.get_filter('ztf.r', as_dataframe=False),
                                                    slice_object=True)

        SNR = np.nanmean(sn_slice_photo.data)/(np.nanmean(
            varcube_slice_photo.data**0.5)/len(varcube_slice_photo.data)**0.5)
        self.pure_host.header.update({'SNR': np.round(SNR, 5)})

        if return_var:
            return poisson_simu, newvariance

        return poisson_simu

    def get_simulated_cube(self, snpoisson=None, snpoissonvar=None, savecube=None):
        """ """
        if snpoisson == None:
            snpoisson = self.sn_poisson
        if snpoissonvar == None:
            snpoissonvar = self.sn_poisson_var

        simulated_cube = self.pure_host.get_new(
            newdata=self.pure_host.data + snpoisson, newvariance=self.pure_host.variance + snpoissonvar)

        from astropy.io import fits
        header = fits.Header()
        header.update(**self.pure_host.header)
        simulated_cube.set_header(header)
        simulated_cube.spec_prop.update(**{
            'lspix': 220,
            'wstep': 25.57077625570673,
            'lstep': 25.57077625570673,
            'wstart': 3700.0,
            'lstart': 3700.0})
        self._simulated_cube = simulated_cube
        if savecube != None:
            simulated_cube.set_filename(savecube)
            simulated_cube.writeto(simulated_cube.filename, headerbased=True)

        return simulated_cube

    def get_metafit_purehost(self, purehost_filename):
        """ """
        metapath = purehost_filename.replace(
            '.fits', '.h5').replace('e3d', 'hgout')
        return pd.read_hdf(metapath, 'meta_slicefit')

    def set_modelspectra(self, model_spectra):
        """ """
        self._modelspectra = model_spectra

    def set_purehost(self, pure_host):
        """ """
        self._purehost = pure_host

    def set_metafithost(self, metadf):
        """ """
        self._metafithost = metadf

    def set_psfparams(self, psfparams):
        """ """
        if psfparams is not None:
            self._psfparams = psfparams
        else:
            psfparams = dict({'alpha': 2.5, 'eta': 0.8, 'rho': -0.2})
            self._psfparams = psfparams

    def set_sn_distance_from_host(self, dist):
        self._sn_distance_from_host = dist

    def set_contrast(self, contrast):
        """ """
        self._contrast = contrast

    def show_spec_simu(self, ax=None):
        """ """

        if ax == None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        spec = self.model_spectra
        specampl = spec.T[1]*self.pure_host.header['EXPTIME']
        lbda = spec.T[0]
        specampl *= self.pure_host.header['SN_COEFF']

        specerr = spec.T[2]**0.5
        specerr *= self.pure_host.header['EXPTIME']
        specerr *= self.pure_host.header['SN_COEFF']

        ax.plot(lbda, specampl, label='Simulated Spectra', color='k')
        ax.fill_between(lbda, specampl+specerr, specampl -
                        specerr, color='k', alpha=0.2)
        if self._savgol_applied:
            from scipy.signal import savgol_filter
            specampl = savgol_filter(
                specampl, self.pure_host.header['SAVGOL_W'], self.pure_host.header['SAVGOL_D'])
            ax.plot(lbda, specampl,
                    label='Simulated Spectra with savgol filter', color='r')
        ax.legend()
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=13)
        ax.set_ylabel(r'Flux (erg units)', fontsize=13)
        ax.tick_params(labelsize=13)

    def show_simulated_cube(self, lbdarange=[4000, 8000], vmin='0', vmax='99.5'):
        """ """

        lbdamin, lbdamax = lbdarange
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        flag = (self.pure_host.lbda > lbdamin) & (self.pure_host.lbda < lbdamax)
        vmin_sim = np.percentile(
            np.mean(self.simulated_cube.data[flag], axis=0), float(vmin))
        vmax_sim = np.percentile(
            np.mean(self.simulated_cube.data[flag], axis=0), float(vmax))

        vmin = np.percentile(
            np.mean(self.pure_host.data[flag], axis=0), float(vmin))
        vmax = np.percentile(
            np.mean(self.pure_host.data[flag], axis=0), float(vmax))

        self.simulated_cube._display_im_(
            axim=ax, lbdalim=[lbdamin, lbdamax], vmin=vmin_sim, vmax=vmax_sim)
        ax.scatter(*self.sim_sn_pos, zorder=10, c='r',
                   marker='x', alpha=0.3, label='Simu SN')
        ax.scatter(*self.hostcoord, zorder=10, c='k',
                   marker='x', alpha=0.3, label='Host')
        ax.set_aspect('equal')
        self.pure_host._display_im_(
            axim=ax2, lbdalim=[lbdamin, lbdamax], vmin=vmin, vmax=vmax)
        ax2.set_aspect('equal')
        ax2.scatter(*self.hostcoord, zorder=10, c='k', marker='x', alpha=0.3)

        ax.set_title('Simulated cube')
        ax2.set_title('Pure Host cube')
        ax.legend(loc='lower left')

    def show_profil(self, lbda=None):
        """ """

        xgrid, ygrid = np.meshgrid(np.linspace(
            0., 10, 100), np.linspace(0., 10, 100))
        xc, yc = 0, 0
        dx = xgrid-xc
        dy = ygrid-yc
        rgrid = np.sqrt(dx**2 + self.psf3d.psf.a_ell*dy **
                        2 + 2*self.psf3d.psf.b_ell * (dx*dy))

        radiusrav = rgrid.ravel()
        radiusrav.sort()
        radiusrav = radiusrav[::-1]

        if lbda is None:
            lbda = self.adr.lbdaref
        prof = self.psf3d.psf.get_radial_profile(radiusrav, lbda)

        alpha, beta, eta, sigma = self.psf3d.psf.get_alpha(lbda), self.psf3d.psf.get_beta(
            lbda), self.psf3d.psf.get_eta(), self.psf3d.psf.get_sigma(lbda)
        fig, ax = plt.subplots(figsize=(10, 5))
        normalisation = (np.pi / np.sqrt(self.psf3d.psf.a_ell - self.psf3d.psf.b_ell**2) *
                         (2 * eta * sigma**2 + alpha**2 / (beta - 1)))
        gaussian = np.exp(-0.5 * radiusrav**2 / sigma**2)
        moffat = (1+(radiusrav/alpha)**2)**(-beta)

        ax.plot(radiusrav, (eta*gaussian) /
                normalisation, c='g', label='Gaussian')
        ax.plot(radiusrav, moffat/normalisation, c='b', label='Moffat')

        ax.plot(radiusrav, prof, label="Gaussian + Moffat model", c='r')
        ax.set_xlim(0, 8)
        ax.set_ylim(0, np.max(prof))
        fwhm = radiusrav[np.where(abs(prof/np.max(prof) - 0.5) ==
                                  np.min(abs(prof/np.max(prof) - 0.5)))[0]][0]*2*self.IFU_SCALE

        ax.vline(fwhm/(2 * self.IFU_SCALE), ymin=0, ymax=np.max(prof),
                 c='k', ls='--', label=f'FWHM={np.round(fwhm,2)}"')

        ax.legend()
        ax.set_xlabel('Elliptical radius (spx)')
        ax.set_ylabel('Profil (normalized)')
        fig.suptitle(fr'PSF Profile at $\lambda = {lbda}\AA $')

    @property
    def model_spectra(self):
        """ """
        return self._modelspectra

    @property
    def pure_host(self):
        """ """
        return self._purehost

    @property
    def psf_params(self):
        """ """
        return self._psfparams

    @property
    def metafithost(self):
        """ """
        return self._metafithost

    @property
    def psf3d(self):
        """ """
        if not hasattr(self, '_psf3d'):
            warnings.warn(
                'You first should load the PSF through self.load_psf3d()')
        return self._psf3d

    @property
    def sourcedf(self):
        """ """
        return self._sourcedf

    @property
    def wcscube(self):
        """ """
        return self._wcscube

    @property
    def hostcoord(self):
        """ """
        return self._hostcoord

    @property
    def sn_distance_from_host(self):
        """ """
        return self._sn_distance_from_host

    @property
    def sim_sn_pos(self):
        """ """
        return self._sim_sn_pos

    @property
    def contrast(self):
        """ """
        return self._contrast

    @property
    def adr(self):
        """ """
        return self._adr

    @property
    def fluxcalfile(self):
        """ """
        return base.get_fluxcal_file(self.pure_host, hgfirst=True)

    @property
    def sn_adu(self):
        """ """
        return self._sn_adu

    @property
    def sn_poisson(self):
        """ """
        return self._sn_poisson

    @property
    def sn_poisson_var(self):
        """ """
        return self._sn_poisson_var

    @property
    def simulated_cube(self):
        """ """
        return self._simulated_cube

    @property
    def savgol_applied(self):
        """ """
        return self._savgol_applied

    @property
    def fwhm(self):
        """ """
        if not hasattr(self, '_fwhm'):
            warnings.warn(
                'You first should load the PSF through self.load_psf3d()')
            return None
        return self._fwhm
