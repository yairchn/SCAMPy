import subprocess
import generate_namelist
import json
import numpy as np
import netCDF4 as nc
import scm_iterationP
import geoMC
import argparse


def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('ncore')
    parser.add_argument('theta', type=float)
    parser.add_argument('case_name')
    parser.add_argument('true_path')
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('D', nargs='?', type=int, default=1)
    parser.add_argument('s', nargs='?', type=float, default=2.0)
    parser.add_argument('N', nargs='?', type=int, default=100)
    parser.add_argument('num_samp')
    parser.add_argument('num_burnin')
    parser.add_argument('step_sizes', nargs='?', type=float,
                        default=[.05, .1, 1, 1, .7])  # this first value is for mcmc
    parser.add_argument('step_nums', nargs='?', type=int, default=[1, 1, 4, 1, 2])
    parser.add_argument('algs', nargs='?', type=str, default=('RWM', 'MALA', 'HMC', 'mMALA', 'mHMC'))
    args = parser.parse_args()
    ncore = args.ncore
    theta0 = args.theta
    case_name = args.case_name
    true_path = args.true_path
    tuning_log = open("/cluster/scratch/yairc/scampy/tuning_log.txt", "w")
    tuning_log.write("parameters recived")

    # load true data
    true_data = nc.Dataset(true_path + 'stats/Stats.' + case_name + '.nc', 'r')
    tuning_log.write("load true data")

    # consider opening a matrix for costfun and storing all the iterations
    txt = 'ABCDEFGHIJK'
    fname = '/cluster/scratch/yairc/scampy/'+ 'tuning_record_'+case_name+txt[int(ncore)]+'.nc'
    tuning_record = nc.Dataset(fname,'w')
    initiate_record(fname)

    # define the lambda function to compute the cost function theta for each iteration
    costFun = lambda theta, geom_opt: scm_iterationP.scm_iterP(ncore,true_data, theta, case_name, geom_opt)
    tuning_log.write("define Lambda as scm_iter")
    # set boudaries for the mcmc
    uppbd = 2.0 * np.ones(args.D)
    lowbd = np.zeros(args.D)

    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO], args.step_sizes[args.algNO], args.step_nums[args.algNO]))

    # call Parallel_mcmc.py
    mc_fun = geoMC.geoMC(theta0, costFun, args.algs[args.algNO],
                         args.step_sizes[args.algNO], args.step_nums[args.algNO],lowbd, uppbd,
                         'bounce').sample


    tuning_log.write("call geoMC")
    mc_args = (args.num_samp, args.num_burnin)
    mc_fun(*mc_args)
    tuning_log.close()
    tuning_record.close()

    return

def initiate_record(fname):


    tuning_record = nc.Dataset(fname, "w", format="NETCDF4")
    grp_stats = tuning_record.createGroup('data')
    grp_stats.createDimension('z', nz) # get this from namelistfile
    grp_stats.createDimension('t', nt) # get this from namelistfile
    grp_stats.createDimension('dim', None)
    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
    thetal_mean = grp_stats.createVariable('thetal', 'f4', ('t', 'z', 'dim'))
    qt_mean = grp_stats.createVariable(' qt_mean', 'f4', ('t', 'z', 'dim'))
    ql_mean = grp_stats.createVariable(' ql_mean', 'f4', ('t', 'z', 'dim'))
    temperature = grp_stats.createVariable('temperature', 'f4', ('t', 'z', 'dim'))
    tune_param = grp_stats.createVariable('tune_param', 'f4', 'dim')
    costFun = grp_stats.createVariable('costFun', 'f4', 'dim')  # this might be a problem if dim=1 implies 2 value

    return

if __name__ == "__main__":
    main()
