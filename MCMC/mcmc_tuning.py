import numpy as np
import subprocess
import argparse
import netCDF4 as nc
import geoMC
import scm_iteration
import sys
import time
import os
from create_records import initiate_record

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2019)
#                         case.            true data address                                                                                       true data type
# python mcmc_tuning.py 'Bomex' '/Users/yaircohen/Documents/codes/scampy/Output.Bomex.july7/stats/' SCM
# python mcmc_tuning.py 'Bomex' '/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/' LES
# python mcmc_tuning.py 'TRMM_LBA' '/Users/yaircohen/Documents/SCAMPy_out/mcmc_tuning/sweep/TRMM_LBA/Output.TRMM_LBA.original/' SCM
# python mcmc_tuning.py 'TRMM_LBA' '/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.TRMM_LBA.newtracers_NO_ICE3/' LES
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    parser.add_argument('true_path')
    parser.add_argument('model_type')
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('D', nargs='?', type=int, default=1)
    parser.add_argument('s', nargs='?', type=float, default=2.0)
    parser.add_argument('N', nargs='?', type=int, default=100)
    parser.add_argument('num_samp', nargs='?', type=int, default=50) # this is the total number of samples 6000
    parser.add_argument('num_burnin', nargs='?', type=int, default=10) # this is the number of burning samples 1000
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.05, .1, 1, 1, .7]) # this first value is for mcmc
    parser.add_argument('step_nums', nargs='?', type=int, default=[1, 1, 4, 1, 2])
    parser.add_argument('algs', nargs='?', type=str, default=('RWM', 'MALA', 'HMC', 'mMALA', 'mHMC'))
    args = parser.parse_args()
    case_name = args.case_name
    true_path = args.true_path
    model_type = args.model_type
    # compile the SCM
    localpath = os.getcwd()
    myscampyfolder = localpath[0:-5]
    subprocess.call("CC=mpicc python setup.py build_ext --inplace",  shell=True, cwd=myscampyfolder)
    # generate namelist for the tuning
    subprocess.call("python generate_namelist.py " + case_name,  shell=True, cwd=myscampyfolder)
    # load true data
    true_data = nc.Dataset(true_path + 'stats/Stats.'+ case_name + '.nc', 'r')

    # consider opening a matrix for costfun and storing all the iterations
    ncore = 1
    # theta0 = initial guess of tuning parameter/s
    theta0 = [0.5]
    output_filename = 'tuning_record.nc'
    #tuning_record = nc.Dataset(output_filename, 'w')
    initiate_record(output_filename, theta0)
    uppbd = 10.0 * np.ones(len(theta0))
    lowbd = 0.0 * np.ones(len(theta0))#(args.D)
    if lowbd>=uppbd:
        sys.exit('lowbd must be smaller than uppbd')

    # define the lambda function to compute the cost function "theta" for each iteration
    costFun = lambda theta, geom_opt: scm_iteration.scm_iter(true_data, myscampyfolder, theta, case_name, output_filename, model_type, geom_opt)

    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO], args.step_sizes[args.algNO], args.step_nums[args.algNO]))

    mc_fun = geoMC.geoMC(theta0, costFun, args.algs[args.algNO],
                         args.step_sizes[args.algNO], args.step_nums[args.algNO],lowbd, uppbd,'reject').sample

    mc_args = (args.num_samp, args.num_burnin)
    mc_fun(*mc_args)
    return

if __name__ == '__main__':
    main()