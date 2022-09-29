# HyperGal

Pipeline for supernovae extraction and classification within the SEDm/ZTF project. Please have a look at [the _HyperGal_ paper](https://arxiv.org/abs/2209.10882).

HyperGal is a fully chromatic scene modeler, which uses pre-transient photometric images to generate a hyperspectral model of the host galaxy; it is based on the CIGALE SED fitter used as a physically-motivated spectral interpolator. The galaxy model, complemented by a point source and a diffuse background component, is projected onto the SEDm spectro-spatial observation space and adjusted to observations

## Acknowledgement

This project has received funding from the Project IDEXLYON at the University of Lyon under the Investments for the Future Program (ANR-16-IDEX-0005), and from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement n°759194 - USNAC).

## References
If you are using HyperGal or a supernova spectrum obtained from it, please cite [the _HyperGal_ paper](https://arxiv.org/abs/2209.10882).
***
# Installation

```bash
git clone https://github.com/JeremyLezmy/HyperGal.git
cd HyperGal
python setup.py install
```
``` pip ``` installation will be available soon.

*** 
# Dependencies

The following dependencies are automatically installed:

- _numpy_, _scipy_, _pandas_, _matpotlib_, _astropy_ (basic anaconda)
- _pysedm_ and its own dependencies (```pip install pysedm``` , see https://github.com/MickaelRigault/pysedm) 
- _ztfquery_ (```pip install ztfquery``` AND see https://github.com/MickaelRigault/ztfquery for path configuration) 
- _dask_ (```python -m pip install "dask[complete]"    # Install everything``` see https://docs.dask.org/en/stable/install.html) 
- _geopandas_ (```pip install geopandas``` ) 
- _iminuit_ (version<2.0 ```pip install iminuit<=2.0``` ) 
- _ztfimg_ (```pip install ztfimg``` see https://github.com/MickaelRigault/ztfimg)
- _Cigale_ (version 2020. The 2022 version should work too, but has not been tested. See https://cigale.lam.fr/download/ for installation)
