""" geometry tools """

import numpy as np
import geopandas
from shapely import geometry, vectorized, affinity


    
import geopandas
from shapely import geometry, vectorized, affinity

def transform_geometry(geom, rotation=None, scale=None, xy_offset=None):
    """ use shapely.affinity to translate, rotate or scale the input geometry.

    Parameters
    ----------
    rotation: [float, None] -optional-
        rotation angle (in deg)
        - None means ignored

    scale: [float, None] -optional-
        scale the geometry (ref = centroid)
        - None means ignored

    xy_offset: [float, None] -optional-
        shifts the centroid by -xoff and -yoff 
        with xoff, yoff= xy_offset
        - None means ignored        

    Returns
    -------
    Geometry
    """
    if xy_offset is not None:
        xoff, yoff = xy_offset
        geom = affinity.translate(geom, -xoff, -yoff)
        
    if rotation is not None:
        geom = affinity.rotate(geom, rotation)

    if scale is not None:
        geom = affinity.scale(geom, xfact=scale, yfact=scale)

    return geom


def get_mpoly(spaxelhandler, rotation=None, scale=None, xy_offset=None):
    """ build the spaxels mutlipolygone out of a slice/clube 
    
    Returns
    -------
    shapely.geometry.Multipolyon
    """
    mpoly_in = spaxelhandler.get_spaxel_polygon(remove_nan=True, format="multipolygon")
    return transform_geometry(mpoly_in, xy_offset=xy_offset, rotation=rotation,  scale=scale)
    


class Overlay( object ):
    #
    # Notation
    #  {}_in is projected into {}_comp
    #
    
    def __init__(self, mpoly_in=None, mpoly_comp=None):
        """  multipolygone _in to be projected into the multipolygone _comp """
        self.set_multipolygon(mpoly_in, "in")
        self.set_multipolygon(mpoly_comp, "comp")        
        
    @classmethod
    def from_slices(cls, slice_in, slice_comp, 
                       xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=None, scale_comp=None):
        """ instantiate the object given slices (could also be cubes, should be a pyifu.SpaxelHandler)
        
        Parameters
        ----------
        slice_in, slice_comp: [SpaxelHandlers]
            slice (or cube) _in to be projected into _comp's geometry
            
         xy_in, xy_comp: [2d-array (float) or None]
             reference coordinates (target position) for the _in and _comp geometries
             e.g. xy_comp = [3.1,-1.3]

         rotation_in, rotation_comp: [float or None]
             rotation (in degree) or the _in and _comp geomtries

         scale_in, scale_comp:  [float or None]
             scale of the _in and _comp geometries
        
        Returns
        -------
        cls()
        """
        mpoly_in = get_mpoly(slice_in, rotation=rotation_in, scale=scale_in, xy_offset=xy_in)
        mpoly_comp = get_mpoly(slice_comp, rotation=rotation_comp, scale=scale_comp, xy_offset=xy_comp)
        return cls(mpoly_in=mpoly_in, mpoly_comp=mpoly_comp)

    # ============= #
    #  Methods      #
    # ============= #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_multipolygon(self, multipoly, which):
        """ set the 'in' or 'comp' geometry
        
        Parameters
        ----------
        multipoly: [geometry]
            Multipolygon (spaxels) defining the geometries or _in or _comp

        which: [string]
            Which geometry are you providing ?
            - in or which.
            a ValueError is raise if which is not 'in' or 'comp'

        Returns
        -------
        None
        """
        if which == "in":
            self._mpoly_in = multipoly
        elif which == "comp":
            self._mpoly_comp = multipoly
        else:
            raise ValueError(f"which can be 'in' or 'comp', {which} given")
        
    def set_overlaydf(self, overlaydf):
        """ sets the self.overlaydf containing the geopandas overlay dataframe"""
        self._overlaydf = overlaydf
        
    def reset_overlaydf(self):
        """ sets self.overlaydf back to None. It will be re-evaluated the next time you need it. """
        self.set_overlaydf(None)
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_projected_flux(self, flux, **kwargs):
        """ project a flux (from {}_in) into the {}_comp geometry using self.overlaydf 
        (callling the classmethod self.project_flux() 
        """
        return self.project_flux(flux, self.overlaydf,  **kwargs)
        
    # -------- #
    #  LOADER  #
    # -------- #
    def load_overlaydf(self, **kwargs):
        """  measures the overlay df and set it.
        uses self.get_overlaydf()
        """
        self.set_overlaydf(self.get_overlaydf(self.mpoly_in, self.mpoly_comp, **kwargs))
        
    # -------- #
    #  CHANGE  #
    # -------- #
    def change_in(self, rotation=None, scale=None, xy_offset=None, reset_overlay=True):
        """ Changes the _in geometry using transform_geometry
        
        Parameters
        ----------
        rotation: [float, None] -optional-
            rotation angle (in deg)
            - None means ignored

        scale: [float, None] -optional-
            scale the geometry (ref = centroid)
            - None means ignored

        xy_offset: [float, None] -optional-
            shifts the centroid by -xoff and -yoff 
            with xoff, yoff= xy_offset
            - None means ignored        
            
        //
        reset_overlay: [bool] -optional-
            shall this reset the overlay (you should)

        Returns
        -------
        None
        """
        new_mpoly = transform_geometry(self.mpoly_in, xy_offset=xy_offset, rotation=rotation,  scale=scale)
        self.set_multipolygon(new_mpoly, "in")
        if reset_overlay:
            self.reset_overlaydf()
        
    def change_comp(self, rotation=None, scale=None, xy_offset=None, reset_overlay=True):
        """ Changes the _comp geometry using transform_geometry
        
        Parameters
        ----------
        rotation: [float, None] -optional-
            rotation angle (in deg)
            - None means ignored

        scale: [float, None] -optional-
            scale the geometry (ref = centroid)
            - None means ignored

        xy_offset: [float, None] -optional-
            shifts the centroid by -xoff and -yoff 
            with xoff, yoff= xy_offset
            - None means ignored        
            
        //
        reset_overlay: [bool] -optional-
            shall this reset the overlay (you should)

        Returns
        -------
        None
        """
        new_mpoly = transform_geometry(self.mpoly_comp, xy_offset=xy_offset, rotation=rotation, scale=scale)
        self.set_multipolygon(new_mpoly, "comp")
        if reset_overlay:
            self.reset_overlaydf()
    
    
    # ============= #
    #  INTERNAL     #
    # ============= #
    @classmethod
    def project_slices(cls, slice_in, slice_comp, 
                       xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=None, scale_comp=None, use_overlapping=True):
        """ """
        # Build the overlay
        overlaydf = cls.get_slices_overlaydf(slice_in, slice_comp, 
                           xy_in=xy_in, xy_comp=xy_comp, 
                           rotation_in=rotation_in, rotation_comp=rotation_comp,
                           scale_in=scale_in, scale_comp=scale_comp, use_overlapping=use_overlapping)
        
        projected_flux = cls.project_flux(slice_in.data, overlaydf)
        if slice_in.has_variance():
            projected_variance = cls.project_flux(np.sqrt(slice_in.variance), overlaydf)**2
            
        
    @classmethod
    def get_slices_overlaydf(cls, slice_in, slice_comp, 
                       xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=None, scale_comp=None, 
                        use_overlapping=True):
        """ """
        # This take ~300ms
        mpoly_in = get_mpoly(slice_in, rotation=rotation_in, scale=scale_in, xy_offset=xy_in)
        mpoly_comp = get_mpoly(slice_comp, rotation=rotation_comp, scale=scale_comp, xy_offset=xy_comp)
        return cls.get_overlaydf(mpoly_in, mpoly_comp, use_overlapping=use_overlapping)
    
    # ============= #
    #  Main Methods #
    # ============= #
    @staticmethod
    def project_flux(flux, overlaydf, index=None, **kwargs):
        """ """
        norm = flux.mean()
        sin = pandas.DataFrame(flux/norm, index=index, columns=["flux"], **kwargs)
        dfout_ = pandas.merge(overlaydf, sin, left_index=True, right_index=True)
        dfout_["outflux"] = dfout_["area"]*dfout_["flux"]
        return dfout_.groupby("id_comp")["outflux"].sum()*norm
    
    @classmethod
    def get_overlaydf(cls, mpoly_in, mpoly_comp, use_overlapping=True):
        """ """
        id_in = np.arange(len(mpoly_in))
        if use_overlapping:
            mpoly_in, flag = cls.get_overlapping(mpoly_in, mpoly_comp.convex_hull)
            id_in = id_in[flag]
            
        geoin = geopandas.GeoDataFrame(geometry=list(mpoly_in))
        geoin["id_in"] = id_in

        geocomp = geopandas.GeoDataFrame(geometry=list(mpoly_comp))
        geocomp["id_comp"] = np.arange(len(geocomp))
        
        interect =  geopandas.overlay(geoin, geocomp, 
                                     how='intersection')
        if len(geoin.area.unique())==1:
            area_ = geoin.area[0]
        else:
            warnings.warn("Normalisation not implemented ; 1 used")
            area_ = 1
            
        interect["area"] = interect.area / area_
        return interect.set_index("id_in")
        
    @staticmethod
    def get_overlapping(mpolyin, contour):
        """ """
        # individual vertices on the multipolygin
        verts = np.asarray([m.exterior.xy  for m in list(mpolyin)])
        all_verts = np.moveaxis(verts, 0,1)
        # its shape (how many)
        polyshape = np.shape(all_verts)[1:]
        # the corners 
        xs,ys = np.concatenate(all_verts.T,axis=0).T
        # is a corner within the input contour ?
        flags = vectorized.contains(contour, xs,ys)
        # as a vertice any corner within the contour ?
        flags_v = np.any(flags.reshape(polyshape[::-1]), axis=0)
        # Get a new Multipolygon made of the overlapping polygons.
        return geometry.MultiPolygon([geometry.Polygon(v.T) for v in verts[flags_v]]), flags_v
    
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def mpoly_in(self):
        """ """
        return self._mpoly_in
    
    @property    
    def mpoly_comp(self):
        """ """
        return self._mpoly_comp
    
    @property    
    def overlaydf(self):
        """ """
        if not hasattr(self,"_overlaydf") or self._overlaydf is None:
            self.load_overlaydf()
        return self._overlaydf
        
