""" geometry tools """

import warnings
import numpy as np
import pandas
import geopandas
from shapely import geometry, vectorized, affinity

from pyifu.adr import ADR

def transform_geometry(geom, rotation=None, scale=None, xoff=None, yoff=None, origin=(0,0)):
    """ Use shapely.affinity to translate, rotate or scale the input geometry.

    Parameters
    ----------
    rotation: float, None -optional-
        Rotation angle (in deg) \n
        - None means ignored

    scale: float, None -optional-
        Scale the geometry (ref = centroid) \n
        - None means ignored

    xoff,yoff: float, None -optional-
        Shifts the centroid by -xoff and -yoff \n
        - None means ignored        

    Returns
    -------
    Geometry
    """
    if (xoff is not None and xoff !=0) or (yoff is not None and yoff !=0):
        if xoff is None: xoff=0
        if yoff is None: yoff=0
        geom = affinity.translate(geom, -xoff, -yoff)
        
    if rotation is not None:
        geom = affinity.rotate(geom, rotation, origin=origin)

    if scale is not None:
        geom = affinity.scale(geom, xfact=scale, yfact=scale, origin=origin)

    return geom

def transform3d_geometry(geom, rotation=None, scale=None, xoff=None, yoff=None, origin=(0,0)):
    
    """ Use shapely.affinity to translate, rotate or scale the input geometry.
    
    Notes
    -----

    Only xoff and yoff as list have been implemented 

    Parameters
    ----------
    rotation: float, None -optional-
        Rotation angle (in deg) \n
        - None means ignored

    scale: float, None -optional-
        Scale the geometry (ref = centroid) \n
        - None means ignored

    xoff, yoff: list of float, None -optional-
        Shifts the centroid by -xoff and -yoff \n
        - None means ignored        

    Returns
    -------
    Geometry

    """
    if xoff is not None or yoff is not None:
        if xoff is None: xoff = np.zeros( len(xoff) )
        if yoff is None: yoff = np.zeros( len(yoff) )

        geom_ = [list(affinity.translate(geom, -xoff_, -yoff_).geoms) for xoff_,yoff_ in zip(xoff,yoff)]    
        geom = geometry.MultiPolygon( list(np.concatenate(geom_)))
        
        
    if rotation is not None:
        geom = affinity.rotate(geom, rotation, origin=origin)

    if scale is not None:
        geom = affinity.scale(geom, xfact=scale, yfact=scale, origin=origin)

    return geom
        


def get_mpoly(spaxelhandler, rotation=None, scale=None, xoff=None, yoff=None):
    """ Build the spaxels mutlipolygone out of a slice/clube 
    
    Returns
    -------
    shapely.geometry.Multipolyon

    """
    mpoly_in = spaxelhandler.get_spaxel_polygon(remove_nan=True, format="multipolygon")
    return transform_geometry(mpoly_in, xoff=xoff, yoff=yoff, rotation=rotation,  scale=scale)

    
def show_polygon(poly, facecolor="C0", edgecolor="k", ax=None, adjust=False, **kwargs):
    """ 
    Show shapely.Polygon object.

    Parameters
    ----------
    facecolor,edgecolor: string
        Go to Matplotlib.patches.Polygon \n
        Default is "C0" ad "k"
    
    ax: Matplotlib.Axes -optional-
        You can provide your own Axes \n
        Default is None

    adjust: bool
        If True, adjust the xlim/ylim of ax in to the extrem vertices of the Polygon \n
        Defult is False

    kwargs:
        Goes to Matplotlib.patches.Polygon

    Returns
    -------
    Matplotlib.Axes
    
    """
    import matplotlib.pyplot as mpl
    from matplotlib import patches
    if ax is None:
        fig = mpl.figure(figsize=[4,4])
        ax = fig.add_subplot(111)
        adjust = True
    else:
        fig = ax.figure
        
    verts = np.asarray(poly.exterior.xy).T
    ax.add_patch(patches.Polygon(verts,facecolor=facecolor, edgecolor=edgecolor, **kwargs))
    if adjust:
        ax.set_xlim(*np.percentile(verts[:,0], [0,100]))
        ax.set_ylim(*np.percentile(verts[:,1], [0,100]))
    return ax


class Overlay( object ):
    #
    # Notation
    #  {}_in is projected into {}_comp
    #
    PARAMETER_NAMES = ["xoff", "yoff", "scale", "rotation"]
        
    def __init__(self, mpoly_in=None, mpoly_comp=None,
                     geoparam_in={}, geoparam_comp={},
                     reload_poly=True):
        """ Multipolygon _in to be projected into the multipolygon _comp 

        Attributes
        ----------
        mpoly_in: shapely.Multipolygon
            Multipolygon to project
        
        mpoly_comp: shapely.Multipolygon
            Multipolygon where you want to project mpoly_in
        
        """
        self.set_multipolygon(mpoly_in, "in")
        self.set_multipolygon(mpoly_comp, "comp")        
        self._geoparam_in   = {**{k:None for k in self.PARAMETER_NAMES},   **geoparam_in}
        self._geoparam_comp = {**{k:None for k in self.PARAMETER_NAMES}, **geoparam_comp}
        if reload_poly:
            self.change_in(reload=True)            
            self.change_comp(reload=True)
            
    @classmethod
    def from_slices(cls, slice_in, slice_comp, 
                       xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=None, scale_comp=None, reload_poly=True):
        """ Instantiate the object given slices (could also be cubes, should be a pyifu.SpaxelHandler)
        
        Parameters
        ----------
        slice_in,slice_comp: SpaxelHandlers
            slice (or cube) _in to be projected into _comp's geometry
            
         xy_in,xy_comp: 2d-array (float) or None
             Reference coordinates (target position) for the _in and _comp geometries
             e.g. xy_comp = [3.1,-1.3]

         rotation_in,rotation_comp: float or None
             Rotation (in degree) or the _in and _comp geomtries

         scale_in,scale_comp:  float or None
             Scale of the _in and _comp geometries
        
        Returns
        -------
        cls()
        """

        x_in, y_in = (None, None) if xy_in is None else xy_in
        x_comp, y_comp = (None, None) if xy_comp is None else xy_comp
            
        geoparam_in = {"rotation":rotation_in, "scale":scale_in, "xoff":x_in, "yoff":y_in}
        geoparam_comp = {"rotation":rotation_comp, "scale":scale_comp, "xoff":x_comp, "yoff":y_comp}
        
        mpoly_in   = slice_in.get_spaxel_polygon(  remove_nan=True, format="multipolygon")
        mpoly_comp = slice_comp.get_spaxel_polygon(remove_nan=True, format="multipolygon")
        
        return cls(mpoly_in=mpoly_in, mpoly_comp=mpoly_comp,
                    geoparam_in=geoparam_in, geoparam_comp=geoparam_comp,
                       reload_poly=reload_poly)

    # ============= #
    #  Methods      #
    # ============= #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_multipolygon(self, multipoly, which, is_orig=True):
        """ Set the 'in' or 'comp' geometry
        
        Parameters
        ----------
        multipoly: geometry
            Multipolygon (spaxels) defining the geometries or _in or _comp

        which: string
            Which geometry are you providing (in or comp)? \n
            A ValueError is raise if which is not 'in' or 'comp'
            
        is_orig: [bool] -optional-
            Should this multipolygon be considered as the origin one.

        Returns
        -------
        None
        """
        if which == "in":
            self._mpoly_in = multipoly
            if is_orig:
                self._mpoly_in_orig = multipoly
        elif which == "comp":
            self._mpoly_comp = multipoly
            if is_orig:
                self._mpoly_comp_orig = multipoly
        else:
            raise ValueError(f"which can be 'in' or 'comp', {which} given")
        
    def set_overlaydf(self, overlaydf):
        """ Sets the self.overlaydf containing the geopandas overlay dataframe"""
        self._overlaydf = overlaydf
        
    def reset_overlaydf(self):
        """ Sets self.overlaydf back to None. It will be re-evaluated the next time you need it. """
        self.set_overlaydf(None)
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_projected_flux(self, flux, **kwargs):
        """ Project a flux (from {}_in) into the {}_comp geometry using self.overlaydf \n
        (callling the classmethod self.project_flux() 
        """
        return self.project_flux(flux, self.overlaydf,  **kwargs)
        
    # -------- #
    #  LOADER  #
    # -------- #
    def load_overlaydf(self, **kwargs):
        """ Measures the overlay df and set it. \n
        Uses self.get_overlaydf()
        """
        self.set_overlaydf( self.get_overlaydf(self.mpoly_in, self.mpoly_comp, **kwargs))
        
    # -------- #
    #  CHANGE  #
    # -------- #
    def change_in(self, rotation=None, scale=None, xoff=None, yoff=None, reset_overlay=True,
                        reload=False):
        """ Changes the _in geometry using transform_geometry
        
        Parameters
        ----------
        rotation: float, None -optional-
            Rotation angle (in deg)\n
            - None means ignored

        scale: float, None -optional-
            Scale the geometry (ref = centroid)\n
            - None means ignored

        xoff,yoff: float, None -optional-
            Shifts the centroid by -xoff and -yoff \n
            - None means ignored        
            
        reset_overlay: bool -optional-
            Shall this reset the overlay (you should)

        Returns
        -------
        None
        """
        new_param = {k:v for k,v in locals().items()
                         if k in self.PARAMETER_NAMES and\
                         v is not None and \
                         v != self.geoparam_in[k]}
                         
        if len(new_param) == 0:
            if not reload:
                return None
            else:
                new_param = {}

        new_geoparam = {**self._geoparam_in, **new_param}
        new_mpoly = transform_geometry(self.mpoly_in_orig, **new_geoparam)
        
        self.set_multipolygon(new_mpoly, "in", is_orig=False)
        self._geoparam_in = new_geoparam
        
        if reset_overlay:
            self.reset_overlaydf()
        
    def change_comp(self, rotation=None, scale=None, xoff=None, yoff=None, reset_overlay=True,
                          reload=False):
        """ Changes the _comp geometry using transform_geometry
        
        Parameters
        ----------
        rotation: float, None -optional-
            Rotation angle (in deg)\n
            - None means ignored

        scale: float, None -optional-
            Scale the geometry (ref = centroid) \n
            - None means ignored

        xoff,yoff: float, None -optional-
            Shifts the centroid by -xoff and -yoff \n
            - None means ignored        
            
        reset_overlay: bool -optional-
            Shall this reset the overlay (you should)

        Returns
        -------
        None
        """
        new_param = {k:v for k,v in locals().items()
                         if k in self.PARAMETER_NAMES and\
                         v is not None and \
                         v != self.geoparam_comp[k]}
        if len(new_param) == 0:
            if not reload:
                return None
            else:
                new_param = {}
        
        new_geoparam = {**self._geoparam_comp, **new_param}
        new_mpoly = transform_geometry(self.mpoly_comp_orig, **new_geoparam)

        self.set_multipolygon(new_mpoly, "comp", is_orig=False)
        self._geoparam_comp = new_geoparam
        
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
        """ Instantiate the object given slices (could also be cubes, should be a pyifu.SpaxelHandler) 
        and directly process to the projection.
       
        Parameters
        ----------
        slice_in,slice_comp: SpaxelHandlers
            Slice (or cube) _in to be projected into _comp's geometry
            
        xy_in,xy_comp: 2d-array (float) or None
             Reference coordinates (target position) for the _in and _comp geometries\n
             e.g. xy_comp = [3.1,-1.3]

        rotation_in,rotation_comp: float or None
             Rotation (in degree) or the _in and _comp geomtries

        scale_in,scale_comp:  float or None
             Scale of the _in and _comp geometries
        
        Returns
        -------
        cls()
        """
        # Build the overlay
        overlaydf = cls.get_slices_overlaydf(slice_in, slice_comp, 
                           xy_in=xy_in, xy_comp=xy_comp, 
                           rotation_in=rotation_in, rotation_comp=rotation_comp,
                           scale_in=scale_in, scale_comp=scale_comp, use_overlapping=use_overlapping)
        
        projected_flux = cls.project_flux(slice_in.data, overlaydf)
        if slice_in.has_variance():
            projected_variance = cls.project_flux(np.sqrt(slice_in.variance), overlaydf)**2
            
    
    # ============= #
    #  Main Methods #
    # ============= #
    @staticmethod
    def project_flux(flux, overlaydf, index=None, **kwargs):
        """ 
        Project the flux in mpoly_in to mpoly_comp according to the measured overlay between them.

        Paramaters
        ----------
        flux: array
            Flux in mpoly_in

        overlaydf: DataFrame
            Measured overlay

        Returns
        -------
        Pandas.Dataframe
        """
        norm = flux.mean()
        sin = pandas.DataFrame(flux/norm, index=index, columns=["flux"], **kwargs)
        dfout_ = pandas.merge(overlaydf, sin, left_index=True, right_index=True)
        dfout_["outflux"] = dfout_["area"]*dfout_["flux"]
        return dfout_.groupby("id_comp")["outflux"].sum()*norm
    
    @classmethod
    def get_overlaydf(cls, mpoly_in, mpoly_comp, use_overlapping=True,
                          area_ok=1e-3, warn_ifok=False):
        """ 
        Get overlay dataframe with overlapping areas for each polygon, given multipolygon in and out.
       
        Parameters
        ----------
        mpoly_in: shapely.Multipolygon
            Multipolygon to project
        
        mpoly_comp: shapely.Multipolygon
            Multipolygon where you want to project mpoly_in
            
        use_overlapping: bool -optional-
            If True (Default), pre-process with a selection of polygons which will overlap each others

        area_ok: float -optional-
            Threshold on which we consider that overlapping areas are almost the same. 
            We therefore consider the mean of these areas.\n
            Remind that area = 1 means full overlapping (same polygon).\n
            Default is 1e-3

        warn_ifok: bool -optional-
            Allow warning message if all uniques areas are below area_ok.\n
            Default is False
        
        Returns
        -------
        Pandas.DataFrame()

        """
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
        unique_area = geoin.area.unique()
        if len(unique_area)==1:
            area_ = geoin.area[0]
        elif np.all(np.diff(unique_area)<area_ok):
            if warn_ifok:
                warnings.warn(f"areas are almost unique (vary by less than {area_ok}) | mean used.")
            area_ = np.mean(unique_area)
        else:
            warnings.warn(f"all area are not unique {unique_area} | Normalisation not implemented ; 1 used")
            area_ = 1
            
        interect["area"] = interect.area / area_
        return interect.set_index("id_in")
        
    @staticmethod
    def get_overlapping(mpolyin, contour):
        """ 
        Get Multipolygon of overlapping region according to a multipolygon_in and a contour of destination.

        Parameters
        ----------
        mpolyin: geometry
            Multipolygon that we want to project 

        contour: geometry
            Contour of the out geometry where you want to project mpolyin

        Returns
        -------
        shapely.geometry.Multipolygon()

        """
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


    # -------- #
    #  PLOTTER #
    # -------- #
    def show(self, ax=None, flux_in=None, lw_in=0.5, lw_comp=0.5, adjust=False, **kwargs):
        """ 
        Show mpoly_in and mpoly_comp on the same Axe.

        Parameters
        ----------
        ax: Matplotlib.Axes
            You can provide your own Axe (one)
        
        flux_in: array -optional-
            Flux corresponding to self.mpoly_in \n
            Default is None.
        
        lw_in,lw_comp: float -optional-
            Linewidth param

        adjust: bool -optional-
            If True, adjust the xlim/ylim of ax with extrema values of the verticies.

        kwargs:
            Goes to self.show_mpoly()
        
        Returns
        -------
        Matplotlib.Axes

        """
        import matplotlib.pyplot as mpl
        if ax is None:
            fig = mpl.figure(figsize=[6,6])
            ax = fig.add_subplot(111)
            adjust = True
        else:
            fig = ax.figure

        
        ec_in = "0.5" if flux_in is not None else "k"
        _ = self.show_mpoly("in",   ax=ax, facecolor="C0", edgecolor=ec_in,  flux=flux_in, zorder=3, lw=lw_in, **kwargs)
        _ = self.show_mpoly("comp", ax=ax, facecolor="None", edgecolor="0.7", lw=lw_comp, zorder=5)

        if adjust:
            vert_outin = np.asarray(self.mpoly_in.convex_hull.exterior.xy).T
            ax.set_xlim(*np.percentile(vert_outin[:,0], [0,100]))
            ax.set_ylim(*np.percentile(vert_outin[:,1], [0,100]))
        
        return ax

    def show_projection(self, flux_in, savefile=None, axes=None, vmin=None, vmax=None):
        """ 
        Show the projected flux in the mpoly_comp geometry.

        Parameters
        ----------
        flux_in: array
            flux corresponding to self.mpoly_in (flux which has been projected before the process)

        savefile: string -optional-
            If not None (Default), fig.savefile(savefile)

        axes: Axes -optional-
            You can provide 2 axes (one for before/ after projection of flux_in) \n
            Default is None.

        vmin,vmax: None or string or float/int -optional-
            If string, the corresponding percentile is computed and used for colormap scaling. \n
            Otherwise, use the given value.

        Returns
        -------
        Figure

        """
        import matplotlib.pyplot as mpl
        from hypergal.utils.tools import parse_vmin_vmax

        if axes is None:
            fig = mpl.figure(figsize=[8,4])
            axl = fig.add_axes([0.05,0.1,0.4,0.8])
            axr = fig.add_axes([0.55,0.1,0.4,0.8])
        else:
            fig = ax.figure
            axl, axr = fig.axes 

        vmin, vmax = parse_vmin_vmax(flux_in, vmin, vmax)

        prop = dict(vmin=vmin, vmax=vmax)
        _ = self.show(flux_in=flux_in, ax=axl, lw_in=0, adjust=True, **prop)
        flux_proj = self.get_projected_flux(flux_in)
        _ = self.show_mpoly("comp", ax=axr, flux=flux_proj, edgecolor="0.7",lw=0.5, zorder=5, adjust=True,
                           **prop)

        axl.set_title("Projection", fontsize="small", color="0.5", loc="left")
        axr.set_title("Projected", fontsize="small", color="0.5", loc="left")

        if savefile is not None:
            fig.savefig(savefile)
            
        return fig
    
    def show_mpoly(self, which, ax=None, facecolor=None, edgecolor="k", adjust=False,
                  flux=None, cmap="cividis", vmin=None, vmax=None, **kwargs):
        """ 
        Show multipolygon with its corresponding flux if provided.

        Parameters
        ----------
        which: string
            "in" or "comp" multipolygon
        
        ax: Axes -optional-
            You can provide your own ax (one)
        
        facecolor,edgecolor,cmap: string
            Go to Matplotlib parameters

        flux: array -optional-
            Flux corresponding to self.mpoly_*which* \n
            Default is None.

        vmin,vmax: float, None, string -optional-
            Colorbar limits. 
            - Ignored if flux is None \n
            - String: used as flux percentile\n
            - Float: used value\n
            - None: converted to '1' and '99'
        
        kwargs
            Goes to geometry.show_polygon()

        Returns
        -------
        Matplotlib.Axes    

        """
        import matplotlib.pyplot as mpl
        from .tools import parse_vmin_vmax
        if which not in ["in", "comp"]:
            raise ValueError("which much be 'in' or 'comp'")

        if ax is None:
            fig = mpl.figure(figsize=[6,6])
            ax = fig.add_subplot(111)
            adjust = True
        else:
            fig = ax.figure

        mpoly = getattr(self, f"mpoly_{which}")
        if flux is not None:
            cmap = mpl.cm.get_cmap(cmap)
            vmin, vmax = parse_vmin_vmax(flux, vmin, vmax)
            colors = cmap( (flux-vmin)/(vmax-vmin) )
        else:
            colors = [facecolor]*len(mpoly)

        
        for i,poly in enumerate(mpoly):
            _ = show_polygon(poly, facecolor=colors[i], edgecolor=edgecolor, ax=ax, **kwargs)

        if adjust:
            verts = np.asarray(mpoly.convex_hull.exterior.xy).T
            ax.set_xlim(*np.percentile(verts[:,0], [0,100]))
            ax.set_ylim(*np.percentile(verts[:,1], [0,100]))

        return ax
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def mpoly_in(self):
        """ 
        Multipolygon to project.
        """
        return self._mpoly_in
    
    @property    
    def mpoly_comp(self):
        """ 
        Multipolygon where we want to project mpoly_in
        """
        return self._mpoly_comp

    @property
    def mpoly_in_orig(self):
        """ 
        Original definition of the Multipolygon to project.
        """
        return self._mpoly_in_orig
    
    @property    
    def mpoly_comp_orig(self):
        """ 
        Original definition of the Multipolygon where we want to project mpoly_comp
        """
        return self._mpoly_comp_orig

    
    @property
    def geoparam_in(self):
        """ Dictionary containing the geometric parameters for the _in slice"""
        return self._geoparam_in

    @property
    def geoparam_comp(self):
        """ Dictionary containing the geometric parameters for the _comp slice"""
        return self._geoparam_comp

    @property    
    def overlaydf(self):
        """ 
        Overlay dataframe with overlapping informations between mpoly_in and mpoly_out
        """
        if not hasattr(self,"_overlaydf") or self._overlaydf is None:
            self.load_overlaydf()
        return self._overlaydf


class Overlay3D( Overlay ):
    
    
    @classmethod
    def from_cubes(cls, cube_in, cube_comp,
                       xy_in=None, xy_comp=None, 
                       rotation_in=None, rotation_comp=None,
                       scale_in=None, scale_comp=None, reload_poly=True):
        """ Instantiate the object given slices (could also be cubes, should be a pyifu.SpaxelHandler)
        
        = Only _comp is treated as a cube = 
        
        Parameters
        ----------
        slice_in,slice_comp: SpaxelHandlers
            slice (or cube) _in to be projected into _comp's geometry
            
         xy_in,xy_comp: 2d-array (float) or None
             Reference coordinates (target position) for the _in and _comp geometries
             e.g. xy_comp = [3.1,-1.3]

         rotation_in,rotation_comp: float or None
             Rotation (in degree) or the _in and _comp geomtries

         scale_in,scale_comp:  float or None
             Scale of the _in and _comp geometries
        
        Returns
        -------
        cls()
        """
        nslices = len(cube_comp.lbda)
            
        this = cls.from_slices( cube_in, cube_comp, 
                                xy_in=xy_in, xy_comp=xy_comp,
                                rotation_in=rotation_in, rotation_comp=rotation_comp,
                                scale_in=scale_in, scale_comp=scale_comp,
                                reload_poly=False)
    
        this.set_nslices( len(cube_comp.lbda) )
        
        if this.geoparam_comp["xoff"] is not None:
            xoff = np.atleast_1d(this.geoparam_comp["xoff"]) 
            if len(xoff)==1:
                this.geoparam_comp["xoff"] = np.ones(this.nslices)*xoff[0]
            elif len(xoff) == this.nslices:
                this.geoparam_comp["xoff"] = np.asarray(xoff)
            else:
                raise ValueError("input xoff size don't match the number of slices cube_comp")
                
        if this.geoparam_comp["yoff"] is not None:
            yoff = np.atleast_1d(this.geoparam_comp["yoff"]) 
            if len(xoff)==1:
                this.geoparam_comp["yoff"] = np.ones(this.nslices)*yoff[0]
            elif len(xoff) == this.nslices:
                this.geoparam_comp["yoff"] = np.asarray(yoff)
            else:
                raise ValueError("input yoff size don't match the number of slices cube_comp")
        if reload_poly:
            this.change_comp(reload=True)
            this.change_in(reload=True)
            
        return this
        
    def change_comp(self, rotation=None, scale=None, xoff=None, yoff=None, reset_overlay=True, 
                   atol=1e-4, reload=False):
        """ Changes the _comp geometry using transform_geometry
        
        Parameters
        ----------
        rotation: float, None -optional-
            Rotation angle (in deg)\n
            - None means ignored

        scale: float, None -optional-
            Scale the geometry (ref = centroid) \n
            - None means ignored

        xoff,yoff: float, None -optional-
            Shifts the centroid by -xoff and -yoff \n
            - None means ignored        
            
        reset_overlay: bool -optional-
            Shall this reset the overlay (you should)

        atol: [float] -optional-
            if a new value is given, how far from the former one 
            this is supposed to be to be considered as changed.
            (using np.allclose(new,former, atol=atol))

        Returns
        -------
        None
        """
        new_param = {k:v for k,v in locals().items()
                         if k in self.PARAMETER_NAMES and\
                         v is not None and \
                         not np.allclose(v, self.geoparam_comp[k], atol=atol)}
        
        if len(new_param) == 0:
            if not reload:
                return None
            else:
                new_param = {}
                
        new_geoparam = {**self._geoparam_comp, **new_param}
        new_mpoly = transform3d_geometry(self.mpoly_comp_orig, **new_geoparam)

        self.set_multipolygon(new_mpoly, "comp", is_orig=False)
        self._geoparam_comp = new_geoparam
        
        if reset_overlay:
            self.reset_overlaydf()
        
    def get_slices_overlaydf(self, correct_id_comp=True):
        """ get the list of individual slice overlaydf """
        nspaxels = self.nspaxels_comp
        nslices  = self.nslices
        
        overlaydf_flatten = self.overlaydf
        # including boundary issues
        shaped_idx = np.unique(overlaydf_flatten["id_comp"]).reshape(self.nslices, self.nspaxels_comp)
        overlays = [overlaydf_flatten[overlaydf_flatten["id_comp"].isin(id_)]
                             for id_ in shaped_idx]

        if correct_id_comp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i,overlay_ in enumerate(overlays):
                    overlay_["id_comp"] -= i*nspaxels
                    
        return overlays
    
    def get_projected_flux(self, flux3d, **kwargs):
        """ """
        overlays = self.get_slices_overlaydf(correct_id_comp=True)
        if len(np.shape(flux3d)) == 2 and np.shape(flux3d)[0]==1:
            flux3d = np.squeeze(flux3d)
    
        if len(np.shape(flux3d)) == 1:
            return [self.project_flux(flux3d, overlaydf_,  **kwargs)
                                       for overlaydf_ in overlays]
        if len(overlays) != len(flux3d):
            raise ValueError(f"flux3d must have the same size as overlays : {len(flux3d)} vs. {len(overlays)}")
            
        return [self.project_flux(flux, overlaydf_,  **kwargs)
               for flux, overlaydf_ in zip(flux3d, overlays)]
        
    def set_nslices(self, nslices):
        """ Set the number of slices ('comp')
        
        Parameters
        ----------
        nslice: int
            Number of slice.

        Returns
        -------
        None
        """
        self._nslices = nslices
        
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def nslices(self):
        """ """ 
        return self._nslices

    @property
    def nspaxels_comp(self):
        """ """
        return len( self.mpoly_comp_orig )


class OverlayADR(Overlay3D):
    
    ADR_PARAMETERS = ["parangle", "airmass"] # unit = scale
    
    @classmethod
    def from_cubes(cls, cube_in, cube_comp, spaxel_comp_unit,
                        xy_in=None, xy_comp=None,
                        rotation_in=None, rotation_comp=None,
                        scale_in=None, scale_comp=None,
                       reload_poly=True):
        """ Instantiate the object given slices (could also be cubes, should be a pyifu.SpaxelHandler)
        
        = Only _comp is treated as a cube = 
        
        Parameters
        ----------
        slice_in,slice_comp: SpaxelHandlers
            slice (or cube) _in to be projected into _comp's geometry
            
         xy_in,xy_comp: 2d-array (float) or None
             Reference coordinates (target position) for the _in and _comp geometries
             e.g. xy_comp = [3.1,-1.3]

         rotation_in,rotation_comp: float or None
             Rotation (in degree) or the _in and _comp geomtries

         scale_in,scale_comp:  float or None
             Scale of the _in and _comp geometries
        
        Returns
        -------
        cls()
        """
        nslices = len(cube_comp.lbda)
            
        this = cls.from_slices( cube_in, cube_comp, 
                                xy_in=xy_in, xy_comp=xy_comp,
                                rotation_in=rotation_in, rotation_comp=rotation_comp,
                                scale_in=scale_in, scale_comp=scale_comp,
                                reload_poly=False)
    
        this.set_nslices( len(cube_comp.lbda) )
        adr = ADR.from_header( cube_comp.header)
        this.set_adr(adr, spaxel_comp_unit)
        this.set_lbda(cube_comp.lbda)
        if reload_poly:            
            this.change_comp(reload=True)
            this.change_in(reload=True)            
        return this
    
    def change_comp(self, rotation=None, scale=None, xoff=None, yoff=None,
                        parangle=None, airmass=None,
                        reset_overlay=True, atol=1e-4, reload=False):
        """ Changes the _comp geometry using transform_geometry
        
        Parameters
        ----------
        rotation: float, None -optional-
            Rotation angle (in deg)\n
            - None means ignored

        scale: float, None -optional-
            Scale the geometry (ref = centroid) \n
            - None means ignored

        xref,yref: float, None -optional-
            Reference position for the refraction computation with ADR() object \n
            - None means not refraction
            
        airmass, parangle: float, None -optional- 
            ADR parameters. parangle is the parallactic angle (in Deg)
            
        reset_overlay: bool -optional-
            Shall this reset the overlay (you should)

        atol: [float] -optional-
            if a new value is given, how far from the former one 
            this is supposed to be to be considered as changed.
            (using np.allclose(new,former, atol=atol))

        Returns
        -------
        None
        """
        new_param = {k:v for k,v in locals().items()
                         if (k in self.PARAMETER_NAMES or k in self.ADR_PARAMETERS) and\
                         v is not None and \
                         not np.allclose(v, self.geoparam_comp[k], atol=atol)}
        if len(new_param) == 0:
            if not reload:
                return None
            else:
                new_param = {}
        #
        # Change the ADR parameters 
        adrparam = {k:v for k in new_param.items() if k in self.ADR_PARAMETERS}
        if len(adrparam)>0:
            self.adr.set(**adrparam)

        #
        # This is the new geoparam_comp
        new_geoparam = {**self._geoparam_comp, **new_param}
        # And here are the transformation dict | but with xoff, xoff only been the refence
        transform_params = {k:v for k in new_param.items() if k not in self.ADR_PARAMETERS}
        # The 3D xoff are there:
        xoff3d, yoff3d = self.adr.refract(new_geoparam["xoff"], new_geoparam["yoff"],
                                              self.lbda, unit=self._spaxel_unit)
        # and wd want to use then for the 3D transformation
        transform_params["xoff"] = xoff3d
        transform_params["yoff"] = yoff3d

        #
        # Let's run the 3D transformation
        new_mpoly = transform3d_geometry(self.mpoly_comp_orig, **transform_params)
        self.set_multipolygon(new_mpoly, "comp", is_orig=False)
        
        #
        # and store de new geoparam (recall there xoff, yoff are the references
        self._geoparam_comp = new_geoparam
        
        if reset_overlay:
            self.reset_overlaydf()

    
    def set_adr(self, adr, spaxel_unit):
        """ Set ADR object (see pyifu.adr.ADR) """
        self._adr = adr
        self._spaxel_unit = spaxel_unit
        
    def set_lbda(self, lbda):
        """ Set slice wavelengths """
        self._lbda = lbda

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def adr(self):
        """ ADR object (see pyifu.adr.ADR) """
        if not hasattr(self, '_adr'):
            return None
        return self._adr
    
    @property
    def lbda(self):
        """ slice wavelengths in [AA] """
        if not hasattr(self, '_lbda'):
            return None
        return self._lbda
