#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import numpy as np
import dask
from dask_jobqueue import SGECluster, SLURMCluster
from dask.distributed import Client
from hypergal.script import scenemodel
from pysedm.io import parse_filename
from hypergal import io as ioh
from hypergal import photometry

SEDM_SCALE = 0.558
PS_SCALE = 0.25
DEFAULT_SCALE_RATIO = SEDM_SCALE/PS_SCALE
FWHM = 2.5  # arcsec
target_radius = (3*2.5/SEDM_SCALE)/2


def limit_numpy(nthreads=1):

    import os
    threads = str(nthreads)
    print(f"threads {threads}")
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads


limit_numpy(1)


def get_num_workers(client):
    """
    :param client: active dask client
    :return: the number of workers registered to the scheduler
    """
    scheduler_info = client.scheduler_info()

    return len(scheduler_info['workers'].keys())


print('sys.executable:', sys.executable)


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

    parser.add_argument("--env", type=str, default='SLURM',
                        help="Which cluster environment. Might be SGE or SLURM")

    parser.add_argument("--logsdir", type=str, default='/sps/ztf/users/jlezmy/dask/logs',
                        help="logs directory for dask")

    parser.add_argument('-w', "--workers", type=int, default=10,
                        help="Scale the cluster to N workers/target. Default is 10.")

    parser.add_argument("--wncores", type=int, default=1,
                        help="Number of core to give to each worker.")

    parser.add_argument("--min_workers", type=int, default=8,
                        help="Scale the cluster to N workers/target. Default is 10.")

    parser.add_argument('-f', "--filename", default=None, type=str,
                        help="File to use with list of cube format (ex e3d_crr_b_ifu20210222_09_18_45_ZTF21aamokak.fits)")

    parser.add_argument('-t', "--target", nargs='*', type=str,
                        help="Targets to process. Must be target name format (ex ZTF21aamokak)")

    parser.add_argument("--ignore_astrom", type=str2bool, nargs='?', const=True, default=True,
                        help="If True, still process if astrometry isn't available. In that case, --radec and --xy are mandatory.")

    parser.add_argument("--host_only", type=str2bool, nargs='?', const=True, default=False,
                        help="If True, set the SN component to zero.")

    parser.add_argument("--sn_only", type=str2bool, nargs='?', const=True, default=False,
                        help="If True, set the Host component to zero.")

    parser.add_argument("--ovwr_wd", type=str2bool, nargs='?', const=True, default=True,
                        help="If True, overwrite temporary folder where SEDfitter outputs are stored.")

    parser.add_argument("--contains", nargs='*', type=none_or_str,
                        help=" If one target name is given, you can give another information as date YYYYMMDD or ID hh_mm_ss.")

    parser.add_argument("--date_range", type=none_or_str, nargs=2, default=None,
                        help="Date range for whatdata request.")

    parser.add_argument("--radec", type=float, nargs=2, default=None,
                        help="If no Astrometry, you have to give the radec information. Default is None.")

    parser.add_argument("--redshift", type=float, default=None,
                        help="If known, you can manually set the redshift. Otherwise the one from Fritz will be used. Default is None.")

    parser.add_argument("--xy", type=float, nargs=2, default=None,
                        help=" You can manually set the xy position of the target in the SEDM IFU. Mandatory if no astrometry available. Default is None.")

    parser.add_argument("--suffix_plot", default=None, type=str,
                        help="Add suffix for plot filename")

    parser.add_argument("--suffix_savedata", default='', type=str,
                        help="Add suffix for saved data filename (spectra, host, cubes etc)")

    parser.add_argument("--lbdarange", type=float, nargs=2, default=[
                        5000, 8500], help="Wavelength range to consider for the fit process. Default is [5000, 8500] AA")
    parser.add_argument("--size", type=int, default=180,
                        help="PS1 images size in pixel (0.25 arcsec/pix).")

    parser.add_argument("--target_radius", type=float, default=10,
                        help="Target radius in spaxel unit which will be selected for extraction. Default is 10 (5 arcsec for SEDm)")

    parser.add_argument("--limit_pos", type=float, default=5,
                        help="limit freedom for position fit in spaxel unit. Default is 5 spaxels around guess/init position")

    parser.add_argument("--max_ratio", type=float, default=0.9,
                        help="Max ratio between host size and median PSF size for considering host component.")

    parser.add_argument("--nslices", type=int, default=6,
                        help="Number of metaslices to consider for the fit process. Default is 6.")

    parser.add_argument("--build_astro", type=str2bool, nargs='?', const=True, default=False,
                        help="If you want (need) to build astrometry, must be True. Default is False.")

    parser.add_argument("--prefit_photo", type=str2bool, nargs='?', const=True, default=True,
                        help=" Fit from photometric images while SEDfitting is computing? Default is True.")

    parser.add_argument("--curved_bkgd", type=str2bool, nargs='?', const=True, default=True,
                        help="Use curved background model if True, flat if False. Default is True.")

    parser.add_argument("--use_exist_intcube", type=str2bool, nargs='?', const=True, default=True,
                        help="Use existing intrinsic cube if it exists.")

    parser.add_argument("--push_to_slack", type=str2bool, nargs='?', const=True, default=True,
                        help="Push to slack?")

    parser.add_argument("--force_fullscene", type=str2bool, nargs='?', const=True, default=False,
                        help="Force full scene fit (Host + SN + Background) whatever the contrast")

    parser.add_argument("--is_simu", type=str2bool, nargs='?', const=True, default=False,
                        help="Is simulation?")

    parser.add_argument("--apply_byecr", type=str2bool, nargs='?', const=True, default=False,
                        help="Apply byecr? Default is True.")

    parser.add_argument('--channel', type=str,
                        default='C02N2U9L88L', help='Slack channel to push')

    args = parser.parse_args()

    if args.env == 'SGE':

        cluster = SGECluster(name="dask-worker",  walltime="12:00:00",
                             memory="10GB", death_timeout=240,
                             project="P_ztf", resource_spec="sps=1", local_directory='$TMPDIR',
                             cores=args.wncores, processes=1)
    elif args.env == 'SLURM':

        cluster = SLURMCluster(name="dask-worker",  walltime="12:00:00",
                               memory="10GB", death_timeout=240,
                               project="ztf", log_directory=args.logsdir, local_directory='$TMPDIR',
                               cores=args.wncores, processes=1,
                               job_extra=['-L sps'])

    if args.filename is not None:
        cubes = np.loadtxt(args.filename, dtype=str)
        dates, sedmid, _, names = pd.DataFrame(
            [parse_filename(files) for files in cubes]).values.transpose()

        cluster.scale(args.workers)
        client = Client(cluster)

        for (targ, date) in zip(names, dates):

            stored = []
            stored.append(scenemodel.DaskScene.compute_targetcubes(name=targ, client=client, contains=date, manual_z=args.redshift,
                          manual_radec=args.radec, rmtarget=None, testmode=False, split=True, lbda_range=args.lbdarange, curved_bkgd=args.curved_bkgd))
            future = client.compute(stored)
            dask.distributed.wait(future)

    elif args.filename is None and len(args.target) > 0:

        cluster.scale(args.workers)
        client = Client(cluster)
        curr_num_workers = 0
        import time
        start_time = time.time()

        while curr_num_workers < np.min([args.workers, args.min_workers]):
            curr_num_workers = get_num_workers(client)
            time.sleep(1)

        print(
            f'{time.time() - start_time} seconds to register {curr_num_workers} workers')

        import pprint
        print('Check os.getcwd on all workers: \n')
        pprint.pprint(client.run(os.getcwd))
        print('Check which python is used on all workers: \n')
        import shutil
        pprint.pprint(client.run(lambda: shutil.which("python")))
        print('Check main packages versions on all workers: \n')
        pprint.pprint(client.get_versions(packages=['shapely', 'pysedm',
                                                    'hypergal', 'ztfquery',
                                                    'pyifu', 'iminuit']))
        if args.contains == None or len(args.contains) != len(args.target):
            contains = np.tile(None, len(args.target))
        else:
            contains = args.contains
        for (targ, contain) in zip(args.target, contains):

            path = None
            if targ.endswith('.fits'):
                path = targ
                info = parse_filename(targ)
                contain = info["sedmid"]
                targ = info["name"]

            sn_only = args.sn_only
            _, radec, _ = ioh.get_target_info(targ, contains=contain)
            if args.radec != None:
                radec = args.radec
            cutouts = photometry.PS1CutOuts.from_radec(*radec)
            cutcube = cutouts.to_cube(binfactor=2)
            sources = cutouts.extract_sources(filter_="ps1.i", thres=3,
                                              savefile=None)

            if not args.host_only and not args.sn_only and not args.force_fullscene:
                if len(sources) == 0:
                    sn_only = True
                else:
                    source = sources[np.sqrt((sources.x-cutouts.radec_to_xy(*radec)[0])**2+(sources.y-cutouts.radec_to_xy(*radec)[1])**2) == np.min(
                        np.sqrt((sources.x-cutouts.radec_to_xy(*radec)[0])**2+(sources.y-cutouts.radec_to_xy(*radec)[1])**2))]
                    wcsin = cutcube.wcs
                    extcubesource = cutcube.get_extsource_cube(
                        source, wcsin=wcsin, radec=radec, sourcescale=3, radius=0, boundingrect=False, sn_only=False)
                    extcubesn = cutcube.get_extsource_cube(
                        source, wcsin=wcsin, radec=radec, sourcescale=3, radius=target_radius*DEFAULT_SCALE_RATIO, boundingrect=False, sn_only=True)
                    ratio = 1 - abs(len(np.intersect1d(extcubesource.indexes, extcubesn.indexes)) - len(
                        extcubesource.indexes))/len(extcubesource.indexes)
                    if ratio > args.max_ratio:
                        sn_only = True
                    else:
                        sn_only = False
            stored = []
            to_stored, cubefiles = scenemodel.DaskScene.compute_targetcubes(name=targ, client=client, cubefiles_=path, contains=contain, manual_z=args.redshift, manual_radec=args.radec, return_cubefile=True, date_range=args.date_range,
                                                                            rmtarget=None, testmode=False, split=True, lbda_range=args.lbdarange, xy_ifu_guess=args.xy, build_astro=args.build_astro, curved_bkgd=args.curved_bkgd, sn_only=sn_only, host_only=args.host_only, use_exist_intcube=args.use_exist_intcube, overwrite_workdir=args.ovwr_wd, suffix_plot=args.suffix_plot, size=args.size, apply_byecr=args.apply_byecr, prefit_photo=args.prefit_photo, suffix_savedata=args.suffix_savedata, target_radius=args.target_radius, limit_pos=args.limit_pos, ncores=args.wncores)
            stored.append(to_stored)

            if len(cubefiles) == 0 and args.push_to_slack:
                m = f"'HyperGal report: No e3d cubefile for {targ}!'"
                if path is not None and not os.path.exists(path):
                    m = f"'HyperGal report: {os.path.basename(path)} does not exist!'"
                ch = args.channel
                command = f"python /pbs/home/j/jlezmy/test_slack_push.py  -m {m} --channel {ch}"
                os.system(command)
            for (n_, cubefile) in enumerate(cubefiles):
                future = client.compute(stored[n_])
                dask.distributed.wait(future)

                if not args.host_only:
                    try:
                        import pysnid
                        targetspec = cubefile.replace(
                            ".fits", ".txt").replace("e3d", "hgspec_target"+args.suffix_savedata)
                        if os.path.exists(targetspec):
                            snidfile = targetspec.replace(
                                'spec', 'snid_bestspec'+args.suffix_savedata).replace('.txt', '.png')
                            snidres = pysnid.run_snid(targetspec)
                            if snidres is not None:
                                snidres.show(savefile=snidfile)
                    except ImportError:
                        import warnings
                        warnings.warn(
                            'pysnid is not installed. You can clone it from https://github.com/MickaelRigault/pysnid.git')
                        snidfile = None
                    except FileNotFoundError:
                        import warnings
                        warnings.warn(
                            "SNID didn't find any template to match the datas.")
                        snidfile = None
                info = parse_filename(cubefile)
                cubeid = info["sedmid"]
                name = info["name"]
                filedir = os.path.dirname(cubefile)
                plotbase = os.path.join(filedir, "hypergal",
                                        info["name"], info["sedmid"])
                if args.suffix_plot is not None:
                    plotbase = os.path.join(filedir, "hypergal",
                                            info["name"], args.suffix_plot + info["sedmid"])
                dirplotbase = os.path.dirname(plotbase)
                logfile = os.path.join(
                    dirplotbase, args.suffix_savedata+'logfile.yml')
                if args.is_simu:
                    logfile = os.path.join(
                        dirplotbase, args.suffix_savedata + 'logfile_'+os.path.basename(cubefile.rsplit('.')[0])+'.yml')
                import yaml
                if os.path.exists(logfile):
                    os.remove(logfile)
                with open(logfile, 'w') as outfile:
                    yaml.dump(client.get_worker_logs(), outfile,
                              indent=3, default_flow_style=False)
                if args.push_to_slack:
                    filepath = plotbase + '_' + name + '_global_report.png'
                    mf = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']})'"
                    if sn_only:
                        mf = f"'HyperGal report (As SN only): {info['name']} {info['sedmid'][-8::]} | ({info['date']})'"
                    elif args.host_only:
                        mf = f"'HyperGal report (As Host only): {info['name']} {info['sedmid'][-8::]} | ({info['date']})'"
                    if args.is_simu and args.suffix_plot is not None:
                        mf = mf.replace(
                            'HyperGal report', f'HyperGal report for {args.suffix_plot}')
                    ch = args.channel
                    if os.path.exists(filepath):

                        targetspec = cubefile.replace(
                            ".fits", ".txt").replace("e3d", "hgspec_target" + args.suffix_savedata)
                        compspec = plotbase + '_' + name + '_all_comp_fit.png'
                        hostspec = cubefile.replace(
                            ".fits", ".txt").replace("e3d", "hgspec_host" + args.suffix_savedata)

                        if args.host_only or snidfile is None:
                            command = f"python /pbs/home/j/jlezmy/test_slack_push.py  -f {filepath} -mf {mf} --targetspec {targetspec} --hostspec {hostspec} --ver_plot {compspec} --channel {ch}"
                        else:
                            command = f"python /pbs/home/j/jlezmy/test_slack_push.py  -f {filepath} -mf {mf} --targetspec {targetspec} --hostspec {hostspec} --ver_plot {compspec} --snid {snidfile} --channel {ch}"
                    else:

                        try:
                            if os.path.exists(logfile):
                                with open(logfile, 'r') as fp:
                                    read_data = yaml.load(
                                        fp, Loader=yaml.FullLoader)
                                if read_data is None:
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process.'"

                                elif 'ConnectionResetError' in str(list(read_data.values())) or 'max_connections' in str(list(read_data.values())) or 'Worker stream died' in str(list(read_data.values())) or 'failed during get data' in str(list(read_data.values())) or 'Stream is closed' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: Dask workers died, try to run again.'"
                                elif 'hypergal.spectroscopy.sedfitting.Cigale' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: SEDFitting failed.'"
                                elif 'All objects passed were None' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: Hypergal did not converge.'"
                                elif 'apply_byecr' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: byecr failed.'"
                                elif 'calibrate_cube' in str(list(read_data.values())) or 'get_fluxcal_file' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: Calibration failed.'"
                                elif 'CutOut.from_radec' in str(list(read_data.values())) or 'CutOut.from_sedmfile' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: PS1 cutouts request failed.'"
                                elif 'Memory use is high but worker has no data to store to disk' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: Memory use is high but worker has no data to store to disk, try to run again.'"
                                elif 'most likely due to a circular import' in str(list(read_data.values())):
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process: Circular importation error with shapely, try to run again.'"

                                else:
                                    m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process.'"
                            else:
                                m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process.'"
                        except:
                            m = f"'HyperGal report: {info['name']} {info['sedmid'][-8::]} | ({info['date']}) failed to process.'"
                        if args.is_simu and args.suffix_plot is not None:
                            m = m.replace(
                                'HyperGal report', f'HyperGal report for {args.suffix_plot}')
                        command = f"python /pbs/home/j/jlezmy/test_slack_push.py  -m {m} --channel {ch}"
                    os.system(command)
                # if n_ < len(cubefiles)-1:
                #    client.restart()
                #    curr_num_workers = 0
                #    start_time = time.time()
                #    while curr_num_workers < np.min([args.workers, args.min_workers]):
                #        curr_num_workers = get_num_workers(client)
                #        time.sleep(1)
                #    print(
                #        f'{time.time() - start_time} seconds to register {curr_num_workers} workers')

    else:

        raise ValueError(
            'Target input has to be ZTF format as "ZTF21aamokak", or a .txt file with filename or filepath for each target')
