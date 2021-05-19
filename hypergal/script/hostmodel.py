from .daskbasics import DaskHyperGal
from pysedm import io
from pysedm.sedm import SEDM_LBDA
class DaskHost( DaskHyperGal ):

    def compute(self, cubefile, radec, redshift,
                    binfactor=2,
                    filters=["ps1.g","ps1.r", "ps1.i","ps1.z","ps1.y"],
                    source_filter="ps1.r", source_thres=2,
                    scale_cout=15, scale_sedm=10, rmtarget=2,
                    filters_fit=["ps1.r", "ps1.i","ps1.z"],
                    psfmodel="Gauss2D", ncores=1):
        """ """
        cubeid      = io.parse_filename(cubefile)["sedmid"]
        working_dir = f"tmp_{cubeid}"

        prop_sourcecube = dict(binfactor=binfactor,
                               filters=filters,
                               source_filter=source_filter,
                               source_thres=source_thres, scale_cout=scale_cout,
                               scale_sedm=scale_sedm, rmtarget=rmtarget)
        
        source_coutcube__source_sedmcube = self.get_sourcecubes(cubefile, radec,
                                                                **prop_sourcecube)
                                                        
        
        source_coutcube  = source_coutcube__source_sedmcube[0]
        source_sedmcube  = source_coutcube__source_sedmcube[1]
        adrpsf_cout_params =  self.fit_cout_slices(source_coutcube, source_sedmcube, radec,
                                                   filterin=filters, filters_to_use=filters_fit,
                                                   psfmodel=psfmodel)
        
        int_cube = self.run_sedfitter(source_coutcube,
                                          redshift=redshift, working_dir=working_dir,
                                          sedfitter="cigale", ncores=ncores, lbda=SEDM_LBDA)
        
        return {"params":adrpsf_cout_params, "cubeint":int_cube}
