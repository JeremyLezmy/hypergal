""" """

from .basics import BackgroundCurved, PointSource, SliceScene
import numpy as np
import warnings
import matplotlib.pyplot as plt
#import shapely
from shapely.geometry import Point
from ..spectroscopy import adr as spectroadr
import pandas


class STD_SliceScene(PointSource, SliceScene):

    def __init__(self, std_slice, psf, use_subslice=False, header=None, lbda_thickness=None, adapt_flux=True, curved_bkgd=True, **kwargs):
        """ """
        self.set_psf(psf)

        if use_subslice:
            std_slice = self.get_subslice(std_slice)

        if header != None:
            std_slice.set_header(header)
        self.set_slice(std_slice, "comp", adapt_flux=adapt_flux)
        self._norm_in = np.nan
        self._bkgd_in = np.nan
        mpoly = std_slice.get_spaxel_polygon(format='multipolygon')
        self.set_mpoly(mpoly)
        self._centroids = self.get_centroids(mpoly)
        self.update_psfparams(**kwargs)
        self._has_curved_bkgd = curved_bkgd
        self._lbda_thickness = lbda_thickness

    @classmethod
    def from_slice(cls, std_slice, psf, use_subslice=False, header=None, lbda_thickness=None, adapt_flux=True, curved_bkgd=True, **kwargs):
        """ """
        return cls(std_slice, psf, use_subslice=use_subslice, header=header, lbda_thickness=lbda_thickness, adapt_flux=adapt_flux, curved_bkgd=curved_bkgd, **kwargs)

    def guess_parameters(self):

        xoff, yoff = np.median(self.slice_comp.index_to_xy(
            self.slice_comp.get_brightest_spaxels(5)), axis=0)

        geom_guess = dict({'xoff': xoff, 'yoff': yoff})

        psf_guess = dict(
            {k+'_ps': v for k, v in self.psf.guess_parameters().items()})

        from shapely.geometry import Point
        x, y = geom_guess['xoff'], geom_guess['yoff']
        p = Point(x, y)
        circle = p.buffer(5)
        idx = self.slice_comp.get_spaxels_within_polygon(circle)

        ampl = np.nansum(self.flux_comp[[self.slice_comp.indexes[i] in np.array(
            idx) for i in range(len(self.slice_comp.indexes))]])
        bkgd = 0
        base_guess = {
            **{"ampl_ps": ampl, "background_ps": bkgd}
        }
        if self.has_curved_bkgd:
            base_guess.update(
                **{k: 0 for k in BackgroundCurved.BACKGROUND_PARAMETERS if k not in ['background']})

        guess_step1 = {**base_guess, **geom_guess, **psf_guess}
        self.update(**guess_step1)

        return guess_step1

    def update(self, ignore_extra=False, **kwargs):
        """ 
        Can update any parameter through kwarg option.\n
        Might be self.BASE_PARAMETER, self.PSF_PARAMETERS or self.GEOMETRY_PARAMETERS
        """
        baseparams = {}
        psfparams = {}
        geomparams = {}
        for k, v in kwargs.items():
            # Change the baseline scene
            if k in self.BASE_PARAMETERS:
                baseparams[k] = v

            # Change the scene PSF
            elif k in self.PSF_PARAMETERS:
                psfparams[k] = v

            # Change the scene geometry
            elif k in self.GEOMETRY_PARAMETERS:
                geomparams[k] = v

            # or crash
            elif not ignore_extra:
                raise ValueError(f"Unknow input parameter {k}={v}")

        self.update_baseparams(**baseparams)
        self.update_overlayparam(**geomparams)

        for k in psfparams.keys():
            if k in self.PSF_PARAMETERS:
                self.psf.update_parameters(
                    **{k.replace('_ps', ''): psfparams[k]})

    def get_model(self, ampl_ps=None, background_ps=None, overlayparam=None, psfparam=None, bkgdx=None, bkgdy=None, bkgdxy=None, bkgdxx=None, bkgdyy=None, fill_comp=False):

        if ampl_ps is None:
            ampl_ps = self.baseparams["ampl_ps"]

        if background_ps is None and not self.has_curved_bkgd:
            background_ps = self.baseparams["background_ps"]

        elif background_ps is None and self.has_curved_bkgd:
            x, y = self.slice_comp_xy
            coeffs = dict({k.replace('_ps', ''): v for k, v in self.baseparams.items(
            ) if k in BackgroundCurved.BACKGROUND_PARAMETERS or k == 'background_ps'})
            background_ps = BackgroundCurved.get_background(x, y, coeffs)

        elif background_ps is not None and self.has_curved_bkgd:

            x, y = self.slice_comp_xy
            coeffs = dict({'background': background_ps, 'bkgdx': bkgdx,
                          'bkgdy': bkgdy, 'bkgdxy': bkgdxy, 'bkgdxx': bkgdxx, 'bkgdyy': bkgdyy})
            background_ps = BackgroundCurved.get_background(x, y, coeffs)

        if psfparam is not None:
            psf_psparam = {k.replace(
                '_ps', ''): v for k, v in psfparam.items() if k in self.PSF_PARAMETERS}

        else:
            psf_psparam = self.psf.parameters

        if overlayparam is not None:
            geom_params = {k: v for k, v in overlayparam.items()
                           if k in self.GEOMETRY_PARAMETERS}
        else:
            geom_params = self.overlayparam

        ps_profile = super(STD_SliceScene, self).get_model(
            xoff=geom_params['xoff'], yoff=geom_params['yoff'], ampl=ampl_ps, bkg=background_ps, **psf_psparam)

        return ps_profile

    def update_overlayparam(self, **kwargs):
        """ 
        Set parameters from self.GEOMETRY_PARAMETERS (xoff and yoff)
        """
        for k, v in kwargs.items():
            if k in self.GEOMETRY_PARAMETERS:
                self.overlayparam[k] = v
            else:
                warnings.warn(f"{k} is not a geom parameters, ignored")
                continue

    def get_subslice(self, std_slice):

        x, y = np.median(std_slice.index_to_xy(
            std_slice.get_brightest_spaxels(5)), axis=0)
        p = Point(x, y)
        circle = p.buffer(20)
        idx = std_slice.get_spaxels_within_polygon(circle)
        sub_slice = std_slice.get_subslice(
            [i for i in std_slice.indexes if i in idx])

        return sub_slice

    def show(self, savefile=None, titles=True, axes=None, vmin='10', vmax='99', cmap='cividis', cmapproj=None, fill_comp=True, index=None, logscale=True, add_colorbar=True):

        flux_model = self.get_model(
            fill_comp=fill_comp) + self.bkgd_comp/self.norm_comp
        flux_comp = self.flux_comp + self.bkgd_comp/self.norm_comp
        flux_comp /= np.max(flux_model)
        flux_model /= np.max(flux_model)

        from hypergal.utils import tools
        if cmapproj is None:
            cmapproj = cmap

        if axes is not None:
            ax, axm, axr = axes
            fig = ax.figure
        else:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        vmin, vmax = tools.parse_vmin_vmax(flux_model, vmin, vmax)

        prop = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "lw": 0.1, "edgecolor": "0.1",
                "adjust": True, "index": index}

        res = flux_comp-flux_model
        res /= flux_model
        RMS = np.sqrt(
            (1/len(flux_comp) * np.sum(((flux_comp-flux_model)/flux_model)**2)))*100
        prop = {**prop, **{"vmin": -0.5, "vmax": 0.5, "cmap": "coolwarm"}}
        title_res = "Residual (data-model)/model [±50%]" + \
            "\n" + fr" RMSEpe = {np.round(RMS,2)}%"
        self.show_psf(ax=axes[2], adjust=True, flux=res,
                      edgecolor=None, vmin=-0.5, vmax=0.5, cmap="coolwarm")

        if not logscale or vmin < 0 or vmax < 0:

            title_dat = "Data"
            if self.has_curved_bkgd:
                title_mod = "Model (2nd order background)"
            else:
                title_mod = "Model (uniform background)"
            self.show_psf(ax=axes[0], adjust=True, flux=flux_comp,
                          cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=None)
            self.show_psf(ax=axes[1], adjust=True, flux=flux_model,
                          cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=None)

        elif logscale:

            title_dat = "Log Data"
            if self.has_curved_bkgd:
                title_mod = "Log Model (2nd order background)"
            else:
                title_mod = "Log Model (uniform background)"
            self.show_psf(ax=axes[0], adjust=True, flux=np.log(
                flux_comp), cmap=cmap, vmin=np.log(vmin), vmax=np.log(vmax), edgecolor=None)
            self.show_psf(ax=axes[1], adjust=True, flux=np.log(
                flux_model), cmap=cmap, vmin=np.log(vmin), vmax=np.log(vmax), edgecolor=None)

        clearwhich = ["left", "right", "top", "bottom"]
        for ax_ in axes:
            ax_.set_yticks([])
            ax_.set_xticks([])
            [ax_.spines[which].set_visible(False) for which in clearwhich]
            ax_.set_aspect('equal')
            ax_.set_rasterized(False)

        if titles:
            prop = dict(loc="left", color="0.5", fontsize="small")
            axes[0].set_title(title_dat, **prop)
            axes[1].set_title(title_mod, **prop)
            axes[2].set_title(title_res, **prop)

        if add_colorbar:
            import matplotlib as mpl
            if logscale and vmin > 0 and vmax > 0:
                norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
                cmapdat = axes[0].figure.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=[axes[0], axes[1]], pad=.05, extend='both', fraction=.05)
                cmapdat.set_label(r'Flux (normalized)',
                                  color="0.5", fontsize="medium", labelpad=10)

            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                cmapdat = axes[0].figure.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=[axes[0], axes[1]], pad=.05, extend='both', fraction=.05)
                cmapdat.set_label(r'Flux (normalized)',
                                  color="0.5", fontsize="medium", labelpad=10)

            normres = mpl.colors.Normalize(vmin=-0.5, vmax=0.5)
            cmapres = axes[2].figure.colorbar(
                mpl.cm.ScalarMappable(norm=normres, cmap='coolwarm'),
                ax=axes[2], pad=.05, extend='both', fraction=.05)
            cmapres.set_label(r'Residual', color="0.5",
                              fontsize="medium", labelpad=10)

        if savefile is not None:
            fig.savefig(savefile)

        return fig

    def show_profile(self, ax=None, logscale=True, savefile=None, radius_spx=15):

        xoff, yoff = self.overlayparam.values()
        p = Point(xoff, yoff)
        circle = p.buffer(radius_spx)

        mslice = self.slice_comp.copy()
        norm_comp = self.norm_comp
        bkgd_comp = self.bkgd_comp

        sndat = mslice.data
        snerr = mslice.variance**0.5

        if self.has_curved_bkgd:
            ampl_ps = self.baseparams['ampl_ps']
            bkgd = self.baseparams['background_ps']
            x, y = self.slice_comp_xy
            coeffs = dict({k.replace('_ps', ''): v for k, v in self.baseparams.items(
            ) if k in BackgroundCurved.BACKGROUND_PARAMETERS and k != 'background'})
            coeffs.update({'background': 0})
            struc_bkgd = BackgroundCurved.get_background(x, y, coeffs)
            sndat -= struc_bkgd*norm_comp
        else:
            ampl_ps, bkgd = self.baseparams.values()

        eta = self.psf.parameters['eta']
        alpha = self.psf.parameters['alpha']
        sigma = self.psf.get_sigma()

        xslice, yslice = np.array(mslice.index_to_xy(mslice.indexes)).T
        xcsn, ycsn = xoff, yoff
        dxsn = xslice-xcsn
        dysn = yslice-ycsn
        rsn = np.sqrt(dxsn**2 + self.psf.a_ell*dysn **
                      2 + 2*self.psf.b_ell * (dxsn*dysn))

        beta = self.psf.get_beta()
        x, y = np.meshgrid(np.linspace(0., radius_spx, 100),
                           np.linspace(0., radius_spx, 100))
        xc, yc = 0, 0
        dx = x-xc
        dy = y-yc
        r = np.sqrt(dx**2 + self.psf.a_ell*dy**2 + 2*self.psf.b_ell * (dx*dy))

        if ax is not None:

            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=(8, 5))

        radius = r.ravel()
        radius.sort()
        radius = radius[::-1]
        profil = self.psf.get_radial_profile(radius)
        normalisation = (np.pi / np.sqrt(self.psf.a_ell - self.psf.b_ell**2) *
                         (2 * eta * sigma**2 + alpha**2 / (beta - 1)))
        gaussian = np.exp(-0.5 * radius**2 / sigma**2)
        moffat = (1+(radius/alpha)**2)**(-beta)

        fwhm = self.fwhm

        ax.plot(radius, (profil*ampl_ps*norm_comp + bkgd*norm_comp +
                bkgd_comp)/np.max(sndat), label="Gaussian + Moffat model", c='r')
        ax.scatter(rsn, sndat/np.max(sndat), c='k', label="Datas", s=16)
        ax.errorbar(rsn, sndat/np.max(sndat), snerr /
                    np.max(sndat), fmt='none', c='k')
        ax.plot(radius, (eta*gaussian*ampl_ps*norm_comp + (bkgd*norm_comp + bkgd_comp)
                * normalisation)/(np.max(sndat)*normalisation), c='g', label='Gaussian')
        ax.plot(radius, (moffat*ampl_ps*norm_comp + (bkgd*norm_comp + bkgd_comp)
                * normalisation)/(np.max(sndat)*normalisation), c='b', label='Moffat')

        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(0.5*(bkgd*norm_comp + bkgd_comp)/np.max(sndat), 1.2)

        else:
            ax.set_ylim(-0.2, 1.1)
        ax.set_xlim(np.min(rsn)-0.1, radius_spx)
        ax.hlines((bkgd*norm_comp + bkgd_comp)/np.max(sndat), 0, radius_spx,
                  ls='--', color='grey', alpha=0.5, label='Background')
        ax.set_xlabel('Elliptical Radius (spx)')
        ax.set_ylabel(' Flux (normalized) ')
        prop = dict(loc="center", color="0.5", fontsize="medium")
        ax.set_title(fr' SN profile with Gaussian + Moffat model' + '\n' +
                     fr'$\lambda = {np.round(self.slice_comp.lbda,2)}$ $\AA$ | Fitted FWHM = {np.round(fwhm,2)}" (0.558"/spx)', **prop)
        ax.legend()
        ax.set_rasterized(False)

        if savefile is not None:
            fig.savefig(savefile)

        return fig

    def show_contour(self, ax=None, logscale=True, savefile=None, cmap='viridis', vmin='10', vmax='99.9'):

        mslice = self.slice_comp.copy()

        xoff, yoff = self.overlayparam.values()
        ampl_ps = self.baseparams['ampl_ps']

        xslice, yslice = np.array(mslice.index_to_xy(mslice.indexes)).T
        xcsn, ycsn = xoff, yoff
        dxsn = xslice-xcsn
        dysn = yslice-ycsn
        rsn = np.sqrt(dxsn**2 + self.psf.a_ell*dysn **
                      2 + 2*self.psf.b_ell * (dxsn*dysn))

        sndat = mslice.data

        xt, yt = np.meshgrid(np.arange(-20, 20, 0.1), np.arange(-20, 20, 0.1))
        xc, yc = xoff, yoff
        dx = xt-xc
        dy = yt-yc
        r = np.sqrt(dx**2 + self.psf.a_ell*dy**2 + 2*self.psf.b_ell * (dx*dy))

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=(7, 7))

        profil = self.psf.get_radial_profile(r)
        lbda = self.slice_comp.lbda

        from matplotlib.lines import Line2D
        flux_model = profil
        flux_comp = self.flux_comp
        from hypergal.utils import tools

        if logscale:
            vmin, vmax = tools.parse_vmin_vmax(np.log(flux_comp), vmin, vmax)
            self.show_psf(ax=ax, adjust=True, flux=np.log(flux_comp),
                          cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=None)

        else:
            vmin, vmax = tools.parse_vmin_vmax(flux_comp, vmin, vmax)
            self.show_psf(ax=ax, adjust=True, flux=flux_comp,
                          cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=None)

        ax.contour(xt, yt, profil*ampl_ps, levels=np.logspace(np.log(np.percentile(flux_comp, 96)),
                   np.log(np.max(flux_comp)), 10), linestyles='dashed', cmap="YlGn", alpha=0.6)
        sedm = ax.tricontour(np.ravel(xslice.tolist()), np.ravel(yslice.tolist()), np.ravel((flux_comp).tolist(
        )), levels=np.logspace(np.log(np.percentile(flux_comp, 96)), np.log(np.max(flux_comp)), 10),  cmap="YlGn", alpha=0.6)

        ax.set_ylim(yoff-9, yoff+9)
        ax.set_xlim(xoff-9, xoff+9)
        import matplotlib as mpl
        cmap = mpl.cm.coolwarm
        custom_lines = [Line2D([0], [0], color=cmap(0.), ls='--'),
                        Line2D([0], [0], color=cmap(0.), ls='-')]
        ax.legend(custom_lines, [fr'GM Model', 'Datas'], loc="upper right")
        ax.set_aspect('equal')
        ax.set_rasterized(False)
        if savefile is not None:
            fig.savefig(savefile)

        return fig

    def show_profile_contour(self, logscale_profile=True, logscale_contour=True, radius_spx=15, cmap='viridis', vmin='10', vmax='99.9', savefile=None):

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        axp = fig.add_subplot(gs[:2])
        axc = fig.add_subplot(gs[2])

        self.show_profile(ax=axp, logscale=logscale_profile,
                          radius_spx=radius_spx)
        self.show_contour(ax=axc, logscale=logscale_contour,
                          cmap=cmap, vmin=vmin, vmax=vmax)
        if savefile is not None:
            fig.savefig(savefile)

        return fig

    def report_std(self, logscale_profile=True, logscale_contour=True, radius_spx=15, vmin='10', vmax='99.9',
                   titles=True, cmap='cividis', cmapproj=None, fill_comp=True, index=None,
                   logscale_imshow=True, add_colorbar=True, savefile=None):

        fig = plt.figure(figsize=(18, 12))

        gs = fig.add_gridspec(2, 3, hspace=0.3)

        ax = fig.add_subplot(gs[0, 0])
        axm = fig.add_subplot(gs[0, 1])
        axr = fig.add_subplot(gs[0, 2])

        axp = fig.add_subplot(gs[1, :2])
        axc = fig.add_subplot(gs[1, 2])
        try:
            self.show(axes=[ax, axm, axr], logscale=logscale_imshow,  cmap=cmap, vmin=vmin, vmax=vmax,
                      fill_comp=fill_comp, index=index, cmapproj=cmapproj, add_colorbar=add_colorbar, titles=titles)
            self.show_profile(ax=axp, logscale=logscale_profile,
                              radius_spx=radius_spx)
            self.show_contour(ax=axc, logscale=logscale_contour,
                              cmap=cmap, vmin=vmin, vmax=vmax)
        except ValueError:
            logscale_profile = False
            logscale_contour = False
            logscale_imshow = False
            self.show(axes=[ax, axm, axr], logscale=logscale_imshow,  cmap=cmap, vmin=vmin, vmax=vmax,
                      fill_comp=fill_comp, index=index, cmapproj=cmapproj, add_colorbar=add_colorbar, titles=titles)
            self.show_profile(ax=axp, logscale=logscale_profile,
                              radius_spx=radius_spx)
            self.show_contour(ax=axc, logscale=logscale_contour,
                              cmap=cmap, vmin=vmin, vmax=vmax)

        import datetime
        import hypergal
        fig.text(
            0.5, 0.01, f"hypergal version {hypergal.__version__} | made the {datetime.datetime.now().date().isoformat()} | J.Lezmy (lezmy@ipnl.in2p3.fr)", ha='center', color='grey', fontsize=10)
        fig.suptitle(self.std_name + fr' ({self.std_date} , ID: {self.std_id})' + '\n' +
                     fr'Airmass: {self.std_airmass} | parangle: {self.std_parangle}° | $\bf{{\lambda}}$ : {np.round(self.lbda)} $\bf{{\AA}}$ ({self.lbda_thickness} $\bf{{\AA}}$ thick)| Exptime = {self.std_exptime}s',
                     fontsize=13, fontweight="bold", y=0.97)

        if savefile is not None:
            fig.savefig(savefile)

        return fig

    @property
    def fwhm(self):

        x, y = np.meshgrid(np.linspace(0., 20, 100), np.linspace(0., 20, 100))
        xc, yc = 0, 0
        dx = x-xc
        dy = y-yc
        r = np.sqrt(dx**2 + self.psf.a_ell*dy**2 + 2*self.psf.b_ell * (dx*dy))
        radius = r.ravel()
        radius.sort()
        radius = radius[::-1]
        profil = self.psf.get_radial_profile(radius)
        sndat = self.slice_comp.data

        fwhm = radius[np.where(abs(profil/np.max(profil) - 0.5) ==
                               np.min(abs(profil/np.max(profil) - 0.5)))[0]][0]*2*0.558

        return fwhm

    @property
    def GEOMETRY_PARAMETERS(self):
        return ['xoff', 'yoff']

    @property
    def std_name(self):
        return self.slice_comp.header['NAME']

    @property
    def lbda_thickness(self):
        return int(self._lbda_thickness)  # AA

    @property
    def std_airmass(self):
        return self.slice_comp.header['AIRMASS']

    @property
    def std_parangle(self):
        return self.slice_comp.header['TEL_PA']

    @property
    def std_date(self):
        return self.slice_comp.header['OBSDATE'].replace('-', '/')

    @property
    def std_id(self):
        return self.slice_comp.header["OBSTIME"].rsplit(".")[0].replace(":", "-")

    @property
    def std_exptime(self):
        return self.slice_comp.header['EXPTIME']

    @property
    def lbda(self):
        return self.slice_comp.lbda  # AA

    @property
    def overlayparam(self):
        if not hasattr(self, "_overlayparam"):
            self._overlayparam = {
                k: 0. if "xoff" in k else 0. for k in self.GEOMETRY_PARAMETERS}
        return self._overlayparam

    @property
    def BASE_PARAMETERS(self):

        basepar = ['ampl_ps', 'background_ps']
        if self.has_curved_bkgd:
            basepar += BackgroundCurved.BACKGROUND_PARAMETERS
            basepar.remove('background')
        return list(np.unique(basepar))

    @property
    def BASE_PS_PARAMETERS(self):

        return self.BASE_PARAMETERS

    @property
    def has_curved_bkgd(self):
        """ All parameters names """
        return self._has_curved_bkgd

    @property
    def slice_comp_xy(self):
        """ All parameters names """

        return self.centroids.T

    @property
    def _slice_in(self):
        """ All parameters names """

        return self.slice_comp


class MultiSliceParametersSTD():

    """ """

    def __init__(self, dataframe, cubefile=None,
                 pointsourcemodel='GaussMoffat3D',
                 load_adr=False, load_pointsource=False, saveplot_adr=None, saveplot_psf=None):
        """ """

        self.set_data(dataframe)
        if load_adr:
            if cubefile is None:
                raise ValueError("cubefile must be given to load_adr")
            self.load_adr(cubefile, saveplot=saveplot_adr)

        if load_pointsource:
            self.load_pointsource(
                pointsourcemodel=pointsourcemodel, saveplot=saveplot_psf)

    @classmethod
    def read_hdf(cls, filename, key,
                 cubefile=None, pointsourcemodel='GaussMoffat3D',
                 load_adr=False):
        """ """
        import pandas
        dataframe = pandas.read_hdf(filename, key)
        return cls(dataframe, cubefile=cubefile, pointsourcemodel=pointsourcemodel,
                   load_adr=load_adr)

    @classmethod
    def from_dataframe(cls, dataframe, **kwargs):
        """ """
        return cls(dataframe,  **kwargs)

    # ============= #
    #   Method      #
    # ============= #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, dataframe):
        """ """
        values = dataframe.unstack(level=1)["values"]
        errors = dataframe.unstack(level=1)["errors"]

        values_ps = values[[values.columns[i] for i in range(
            len(values.columns)) if values.columns[i].endswith('ps')]].copy()
        values_ps.columns = [values_ps.columns[i].rsplit(
            '_ps')[0] for i in range(len(values_ps.columns))]

        errors_ps = errors[[errors.columns[i] for i in range(
            len(errors.columns)) if errors.columns[i].endswith('ps')]].copy()
        errors_ps.columns = [errors_ps.columns[i].rsplit(
            '_ps')[0] for i in range(len(errors_ps.columns))]

        self._values_ps = values_ps
        self._errors_ps = errors_ps

        self._data = dataframe
        self._values = values
        self._errors = errors

    # -------- #
    #  LOADER  #
    # -------- #
    def load_adr(self, cubefile, saveplot=None):
        """ """
        self._adr, self._adr_ref = spectroadr.ADRFitter.fit_adr_from_values(self.values, self.lbda, cubefile,
                                                                            errors=self.errors, saveplot=saveplot)

    def load_pointsource(self, pointsourcemodel="GaussMoffat3D", saveplot=None):
        """ """
        from hypergal import psf

        self._pointsource3d = getattr(psf.gaussmoffat, pointsourcemodel).fit_from_values(
            self.values_ps, self.lbda, errors=self.errors_ps, saveplot=saveplot)
        self._pointsourcemodel = pointsourcemodel

    # -------- #
    #  GETTER  #
    # -------- #
    def get_guess(self, lbda, psfmodel="Gauss2D", pointsourcemodel="GaussMoffat2D", as_dataframe=False, squeeze=True):
        """ """

        if pointsourcemodel == "GaussMoffat2D" and self.pointsourcemodel == "GaussMoffat3D":
            guesses = self.get_gaussmoffat_guess(lbda)

        else:
            raise NotImplementedError(
                "Only GaussMoffat2D Pointsource model implemented")

        if squeeze and len(guesses) == 1:
            return guesses[0]

        if squeeze and len(guesses) > 1:
            allguess = {}
            for d in guesses:
                allguess.update(d)
            return allguess

        if not squeeze and len(guesses) > 1:
            allguess = {}
            for d in guesses:
                allguess.update(d)
            return allguess if not as_dataframe else pandas.DataFrame.from_records(allguess).T

        return guesses if not as_dataframe else pandas.DataFrame.from_records(guesses).T

    def get_gaussmoffat_guess(self, lbda):
        """ """
        lbda = np.atleast_1d(lbda)
        guesses = []
        for i, lbda_ in enumerate(lbda):
            guess = {}
            # -- ADR
            # Position
            xoff, yoff = self.adr.refract(self.adr_ref[0], self.adr_ref[1],
                                          lbda_,
                                          unit=spectroadr.IFU_SCALE)
            guess["xoff"] = xoff
            guess["yoff"] = yoff
            # -- PSF
            # Ellipse
            guess["a_ps"] = self.pointsource3d.a_ell
            guess["b_ps"] = self.pointsource3d.b_ell
            # Profile

            guess["alpha_ps"] = self.pointsource3d.get_alpha(lbda_)[0]
            guess["eta_ps"] = np.average(
                self.values["eta_ps"], weights=1/self.errors["eta_ps"]**2)
            # -- Base Parameters
            for k in ["ampl_ps", "background_ps"]:
                err = self.errors[k][(self.errors[k] != 0)
                                     & (self.errors[k] != np.NaN)]
                val = self.values[k][(self.errors[k] != 0)
                                     & (self.errors[k] != np.NaN)]
                guess[k] = np.average(val, weights=1/err**2)

            guesses.append(guess)

        return guesses

    # ============= #
    #  Properties   #
    # ============= #
    #
    # - Input
    @property
    def data(self):
        """ """
        return self._data

    @property
    def values(self):
        """ """
        return self._values

    @property
    def errors(self):
        """ """
        return self._errors

    @property
    def values_ps(self):
        """ """
        return self._values_ps

    @property
    def errors_ps(self):
        """ """
        return self._errors_ps

    @property
    def lbda(self):
        """ """
        return self.values["lbda"]

    #
    # - Derived
    @property
    def adr(self):
        """ """
        return self._adr

    @property
    def adr_ref(self):
        """ """
        return self._adr_ref

    @property
    def pointsource3d(self):
        """ """
        return self._pointsource3d

    @property
    def pointsourcemodel(self):
        """ """
        if hasattr(self, '_pointsourcemodel'):
            return self._pointsourcemodel
        else:
            return None
