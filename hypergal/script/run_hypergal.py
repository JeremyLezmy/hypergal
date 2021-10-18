#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pysedm.io import parse_filename
from hypergal.script import stdscene, scenemodel
from dask.distributed import Client
from dask_jobqueue import SGECluster
import dask
import numpy as np
import sys
import os


def limit_numpy(nthreads=4):

    import os
    threads = str(nthreads)
    print(f"threads {threads}")
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads


limit_numpy(4)


module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)


if __name__ == '__main__':

    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', "--workers", type=int, default=10,
                        help="Scale the cluster to N workers/target. Default is 10.")
    parser.add_argument('-f', "--filename", default=None, type=str,
                        help="File to use with list of cube format (ex e3d_crr_b_ifu20210222_09_18_45_ZTF21aamokak.fits)")

    parser.add_argument('-t', "--target", nargs='*', type=str,
                        help="Targets to process. Must be target name format (ex ZTF21aamokak)")

    parser.add_argument("--ignore_astrom", type=str2bool, nargs='?', const=True, default=True,
                        help="If True, still process if astrometry isn't available. In that case, --radec and --xy are mandatory.")

    parser.add_argument("--contains", nargs='*', type=none_or_str,
                        help=" If one target name is given, you can give another information as date YYYYMMDD or ID hh_mm_ss.")

    parser.add_argument("--radec", type=float, nargs=2, default=None,
                        help="If no Astrometry, you have to give the radec information. Default is None.")

    parser.add_argument("--redshift", type=float, default=None,
                        help="If known, you can manually set the redshift. Otherwise the one from Fritz will be used. Default is None.")

    parser.add_argument("--xy", type=float, nargs=2, default=None,
                        help=" You can manually set the xy position of the target in the SEDM IFU. Mandatory if no astrometry available. Default is None.")

    parser.add_argument("--lbdarange", type=float, nargs=2, default=[
                        5000, 8500], help="Wavelength range to consider for the fit process. Default is [5000, 8500] AA")

    parser.add_argument("--nslices", type=int, default=6,
                        help="Number of metaslices to consider for the fit process. Default is 6.")

    parser.add_argument("--build_astro", type=str2bool, nargs='?', const=True, default=False,
                        help="If you want (need) to build astrometry, must be True. Default is False.")

    parser.add_argument("--curved_bkgd", type=str2bool, nargs='?', const=True, default=True,
                        help="Use curved background model if True, flat if False. Default is True.")

    args = parser.parse_args()

    cluster = SGECluster(name="dask-worker",  walltime="05:00:00",
                         memory="5GB", death_timeout=120,
                         project="P_ztf", resource_spec="sps=1",
                         cores=1, processes=1)

    if args.filename is not None:
        cubes = np.loadtxt(args.filename, dtype=str)
        dates, sedmid, _, names = pd.DataFrame(
            [parse_filename(files) for files in cubes]).values.transpose()

        cluster.scale(args.workers)
        client = Client(cluster)

        for (targ, date) in zip(names, dates):

            currentpwd = os.getcwd()
            stored = []
            stored.append(scenemodel.DaskScene.compute_targetcubes(name=targ, client=client, contains=date, manual_z=args.redshift,
                          manual_radec=args.radec, rmtarget=None, testmode=False, split=True, lbda_range=args.lbdarange, curved_bkgd=args.curved_bkgd))
            future = client.compute(stored)
            dask.distributed.wait(future)
            if os.getcwd() != currentpwd:
                os.chdir(currentpwd)
    elif args.filename is None and len(args.target) > 0:

        cluster.scale(args.workers)
        client = Client(cluster)
        if args.contains == None or len(args.contains) != len(args.target):
            contains = np.tile(None, len(args.target))
        else:
            contains = args.contains
        for (targ, contain) in zip(args.target, contains):

            currentpwd = os.getcwd()
            stored = []
            stored.append(scenemodel.DaskScene.compute_targetcubes(name=targ, client=client, contains=contain, manual_z=args.redshift,
                          manual_radec=args.radec, rmtarget=None, testmode=False, split=True, lbda_range=args.lbdarange, curved_bkgd=args.curved_bkgd))
            future = client.compute(stored)
            dask.distributed.wait(future)
            if os.getcwd() != currentpwd:
                os.chdir(currentpwd)
    else:

        raise ValueError(
            'Target input has to be ZTF format as "ZTF21aamokak", or a .txt file with filename or filepath for each target')
