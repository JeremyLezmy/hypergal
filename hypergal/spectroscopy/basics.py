""" Module Containing the basic objects """
import warnings
import numpy as np

from ..photometry.astrometry import WCSHolder, get_source_ellipses
from pyifu.spectroscopy import Cube


def sedmcube_to_wcscube(cube, radec=None, spxy=None, store_data=False, get_filename=False):
    """ """
    wcscube = WCSCube.from_sedmcube(cube, radec, spxy)

    if store_data:
        wcscube.writeto(wcscube.filename)

    if get_filename:
        return wcscube.filename

    return wcscube


class WCSCube(Cube, WCSHolder):
    """ 
    Inherits from pyifu.spectroscopy.Cube() object and ztfimg.astrometry.WCSHolder() object.\n
    Load existing 3D cube or create one from cutout images.\n
    Allow to manipulate spaxels (remove, select etc) and bring WCS solution associate to the cube.\n
    """

    @classmethod
    def read_sedmfile(cls, cubefile, radec=None, spxy=None):
        """ 
        Instantiate WCSCube object from filename

        Parameters
        ----------
        cubefile: string
            Filename of the cube to load.

        Returns
        -------
        WCSCube object

        """
        from pysedm import get_sedmcube
        return cls.from_sedmcube(get_sedmcube(cubefile), radec, spxy)

    @classmethod
    def from_sedmcube(cls, cube, radec=None, spxy=None):
        """ 
        Instantiate WCSCube object from pyifu.Cube object.

        Parameters
        ----------
        cube: pyifu.Cube
            Cube to load.

        Returns
        -------
        WCSCube object

        """
        from pysedm import astrometry
        from astropy.io import fits

        try:
            wcsdict = astrometry.get_wcs_dict(cube.filename, radec, spxy)
            #astrom = astrometry.Astrometry(cube.filename)
            # if np.logical_or(*abs(astrom.get_target_coordinate()) > (20, 20)):
            #    warnings.warn(
            #        f'Astrometry out of the field of view in {cube.filename} at (x,y) = {astrom.get_target_coordinate()}, build of the astrometry assuming radec={radec} at (x,y)=(0,0)')
            #    wcsdict = astrometry.get_wcs_dict(cube.filename, radec, (0, 0))
        except OSError:
            warnings.warn(
                f'No Astrometry file for {cube.filename} , build of the astrometry assuming radec={radec} at (x,y)=(0,0)')
            wcsdict = astrometry.get_wcs_dict(cube.filename, radec, (0, 0))

        keys = ["EXPTIME", "ADCSPEED", "TEMP", "GAIN_SET", "ADC", "MODEL", "SNSR_NM", "SER_NO", "TELESCOP",
                "GAIN", "CAM_NAME", "INSTRUME", "UTC", "END_SHUT", "OBSDATE", "OBSTIME", "LST", "MJD_OBS",
                "JD", "APPEQX", "EQUINOX", "RA", "TEL_RA", "DEC", "TEL_DEC", "TEL_AZ", "TEL_EL", "AIRMASS",
                "TEL_PA", "RA_OFF", "DEC_OFF", "TELHASP", "TELDECSP", "FOCPOS", "IFUFOCUS", "IFUFOC2",
                "DOMEST", "DOMEMO", "DOME_GAP", "DOMEAZ", "OBJECT", "OBJTYPE", "IMGTYPE", "OBJNAME",
                "OBJEQX", "OBJRA", "OBJDEC", "ORA_RAT", "ODEC_RAT", "SUNRISE", "SUNSET", "TEL_MO",
                "SOL_RA", "SOL_DEC", "WIND_DIR", "WSP_CUR", "WSP_AVG", "OUT_AIR", "OUT_HUM", "OUT_DEW",
                "IN_AIR", "IN_HUM", "IN_DEW", "MIR_TEMP", "TOP_AIR", "WETNESS", "FILTER", "NAME",
                "P60PRID", "P60PRNM", "P60PRPI", "REQ_ID", "OBJ_ID",
                "ENDAIR", "ENDDOME", "END_RA", "END_DEC", "END_PA", "BIASSUB", "BIASSUB2",
                "CCDBKGD", "ORIGIN", "FLAT3D", "FLATSRC", "ATMCORR", "ATMSRC", "ATMSCALE", "IFLXCORR",
                "IFLXREF", "CCDIFLX", "HYPERGAL", "FLUXCAL"]
        nheader = {k: cube.header[k] for k in keys if k in cube.header}
        cube.set_header(fits.Header({**nheader, **wcsdict}))
        from .. import __version__ as hgvs
        header = {**dict(cube.header), **dict({'HYPERGAL': f'{hgvs}'})}
        cube.set_header(header)

        this = cls.from_data(data=cube.data,
                             variance=cube.variance,
                             lbda=cube.lbda, header=cube.header,
                             spaxel_vertices=cube.spaxel_vertices,
                             spaxel_mapping=cube.spaxel_mapping)

        # hdf5 is a better format to store header with WCS for Cubes.
        this.set_filename(cube.filename.replace(".fits", ".h5"))
        return this

    @classmethod
    def from_cutouts(cls, hgcutout, header_id=0, influx=True, binfactor=None, xy_center=None,
                     cleanheader=True):
        """ 
        Instantiate WCSCube object from hypergal.photometry.CutOuts() object.

        Parameters
        ----------
        hgcutout: CutOuts
            Cutouts to use.

        header_id: int
            Index of the list of avalables cutouts to use to global header.\n
            Default is 0.

        influx: bool
            Load cutouts data in flux unit (erg/s/cm2/AA) or in counts.\n
            Default is True

        binfactor: int
            Binning factor to use on the cutouts to restride the datas.\n
            Default is None (==1)

        xy_center: array
            If not None must be 2 elements. Translate the cube get the center at xy_center\n
            Default is None.

        cleanheader: bool
            If True, clean the Header informations which only consern the individuals images.\n
            Default is True

        Returns
        -------
        WCSCube object

        """

        lbda = np.asarray(hgcutout.lbda)
        sort_lbda = np.argsort(lbda)

        # Data
        filters = np.asarray(hgcutout.filters)[sort_lbda]
        lbda = lbda[sort_lbda]
        data = hgcutout.get_data(influx=influx)[sort_lbda]
        variance = hgcutout.get_variance(influx=influx)[sort_lbda]
        spaxel_vertices = np.asarray(
            [[0, 0], [1, 0], [1, 1], [0, 1]])-0.5  # centered
        if binfactor is not None:
            binfactor = int(binfactor)
            if binfactor == 1:
                warnings.warn("binfactor=1, this means nothing to do.")
            else:
                from ..utils.array import restride
                data = np.sum(
                    restride(data, (1, binfactor, binfactor)), axis=(-2, -1))
                variance = np.sum(
                    restride(variance, (1, binfactor, binfactor)), axis=(-2, -1))
                spaxel_vertices *= binfactor
        else:
            binfactor = 1

        # Header
        header = hgcutout.instdata[header_id].header
        header[f"FROMPHOT"] = True
        header[f"RESTRIDE"] = binfactor

        if cleanheader:
            from astropy.io import fits
            header = fits.Header({k: v for k, v in dict(header).items()
                                  if "." not in k and k not in ["HISTORY", "COMMENT"]})
        for i, filter_ in enumerate(filters):
            header[f"FILTER{i}"] = filter_

        # Mapping
        xsize, ysize = np.asarray(data[header_id].shape)
        pixels_ = np.mgrid[0:xsize*binfactor:binfactor,
                           0:ysize*binfactor:binfactor]

        init_shape = np.shape(pixels_)
        spaxels_xy = pixels_.reshape(2, init_shape[1]*init_shape[2]).T
        if xy_center is not None:
            spaxels_xy = np.asarray(spaxels_xy, dtype="float")-xy_center
        spaxel_mapping = {i: v for i, v in enumerate(spaxels_xy)}

        #
        # Init
        return cls.from_data(data=np.concatenate(data.T, axis=0).T,
                             variance=np.concatenate(variance.T, axis=0).T,
                             lbda=lbda, header=header,
                             spaxel_vertices=spaxel_vertices,
                             spaxel_mapping=spaxel_mapping)

    # ================ #
    #   Methods        #
    # ================ #
    def set_header(self, header, *args, **kwargs):
        """ 
        Set header to the WCSCube.

        Parameters
        ----------
        header: dict
            Header to set

        Returns
        -------

        """
        _ = super().set_header(header)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.load_wcs(header)

    def get_target_removed(self, target_pos=None, radius=3, store=False, **kwargs):
        """ 
        Return a partial cube by removing spaxels around a given region.

        Parameters
        ----------
        target_pos: array
            2 elements array for x/y coordinates (in spx unit). Correpond to the center of the region you want to remove.

        radius: float:
            Radius (in spx unit) of the area you want to remove.\n
            Default is 3 spx.

        store: bool
            If True, store the cube with target removed.\n
            Default is False

        Returns
        -------
        Cube with target removed.
        """
        from . import sedmtools
        if target_pos is None:
            target_pos = sedmtools.get_target_position(self)

        return sedmtools.remove_target_spx(self, target_pos, radius=radius, store=store, **kwargs)

    def get_extsource_cube(self, sourcedf, wcsin, radec=None, wcsout=None, sourcescale=5, radius=6,
                           boundingrect=False, sn_only=False, slice_id=None):
        """ 
        Return a partial cube by removing spaxels outisde a given source delimitation.

        Parameters
        ----------
        sourcedf: DataFrame
            Dataframe of spaxels which delimits the sources. (see hypergal.photomotry.basics.CutOuts() ) 

        wcsin,wcsout: astropy WCS
            astropy WCS solution instance to convert xy<->radec \n    
            If wcsout is None (Default), then wcsout == self.wcs

        sourcescale: float -optional-
            This multiply a and b. 1 means second moment (1 sigma)

        boundingrect: bool -optional-
            If True, will reshape the sources geometry into rectangular slices\n
            Otherwise, slices will have shape of the sources delimitation.\n
            Default is False.

        slice_id: int/list
            If None (Default), will return a cube all the slices of WCSCube instance.\n
            Else, will only consider the given slice index.

        Returns
        -------
        New WCSCube

        """
        if len(sourcedf) == 0:
            return self
        if sn_only:
            radius = 15
        from shapely.geometry import Polygon, Point
        if wcsout is None:
            wcsout = self.wcs

        e_out = get_source_ellipses(sourcedf, wcs=wcsin,  wcsout=wcsout, system="out",
                                    sourcescale=sourcescale)

        if radec is not None:

            target_pos = self.radec_to_xy(*radec).flatten()
            p = Point(*target_pos)
            circle = p.buffer(radius)

        if boundingrect:
            [xmin, ymin], [xmax, ymax] = np.percentile(
                np.concatenate([e_.xy for e_ in e_out]), [0, 100], axis=0)

            if radec is not None:
                xmin_target, ymin_target = target_pos - radius
                xmax_target, ymax_target = target_pos + radius

                xmin, ymin = np.min([xmin, xmin_target]), np.min(
                    [ymin, ymin_target])
                xmax, ymax = np.max([xmax, xmax_target]), np.max(
                    [ymax, ymax_target])

            polys = [
                Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])]
            spaxels = np.unique(np.concatenate(
                [self.get_spaxels_within_polygon(poly_) for poly_ in polys]))

        else:
            polys = [Polygon(e_.xy) for e_ in e_out]

            spaxels = np.unique(np.concatenate(
                [self.get_spaxels_within_polygon(poly_) for poly_ in polys]))
            if radec is not None:
                spaxels = np.unique(np.concatenate(
                    [spaxels, self.get_spaxels_within_polygon(circle)]))
                if sn_only:
                    spaxels = np.unique(self.get_spaxels_within_polygon(circle))

        if len(spaxels) < 5:
            return self
        if slice_id is None:
            slice_id = np.arange(len(self.lbda))

        newcube = self.get_partial_cube(spaxels,  slice_id)

        if boundingrect:
            newcube.header["NAXIS1"] = xmax-xmin
            newcube.header["NAXIS2"] = ymax-ymin
        # No else because it might mess-up with the WCS solution

        return newcube
