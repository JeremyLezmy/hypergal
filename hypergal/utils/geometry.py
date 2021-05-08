""" geometry tools """

import geopandas
from shapely import geometry, vectorized, affinity


    
def transform_geomtry(geometry, rotation=None, scale=None, xy_center=None):
    """ """
    if xy_center is not None:
        xoff, yoff = xy_center
        geometry = affinity.translate(geometry, -xoff, -yoff)
        
    if rotation is not None:
        geometry = affinity.rotate(geometry, rotation)

    if scale is not None:
        geometry = affinity.scale(geometry, xfact=scale, yfact=scale)

    return geometry


class Overlay( object ):
    
    @classmethod
    def get_overlay(cls, mpoly_in, mpoly_comp, use_overlapping=True):
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
            norm_ = geoin.area[0]
        else:
            warnings.warn("Normalisation not implemented for multi area multipolygone (mpoly_in) ; norm=1 used")
            norm_ = 1
            
        interect["area"] = interect.area / norm_
        return interect
        
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
