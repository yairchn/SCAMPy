import numpy as np
import subprocess
import argparse
import netCDF4 as nc
import geoMC
import scm_iteration
import pylab as plt

np.set_printoptions(precision=3, suppress=True)
np.random.seed(2017)

# python mcmc_tuning.py 'Bomex' '/Users/yaircohen/PycharmProjects/scampy/Output.Bomex.original/'
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    parser.add_argument('true_path')
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('D', nargs='?', type=int, default=1)
    parser.add_argument('s', nargs='?', type=float, default=2.0)
    parser.add_argument('N', nargs='?', type=int, default=100)
    parser.add_argument('num_samp', nargs='?', type=int, default=60) # this is the tot number of samples 6000
    parser.add_argument('num_burnin', nargs='?', type=int, default=10) # this is the number of burning samples 1000
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.05, .1, 1, 1, .7]) # this first value is for mcmc
    parser.add_argument('step_nums', nargs='?', type=int, default=[1, 1, 4, 1, 2])
    parser.add_argument('algs', nargs='?', type=str, default=('RWM', 'MALA', 'HMC', 'mMALA', 'mHMC'))
    args = parser.parse_args()
    case_name = args.case_name
    true_path = args.true_path

    plt.figure('thetal profile')
    # generate namelist for the tuning
    subprocess.call("python generate_namelist.py " + case_name,  shell=True)
    # load true data
    true_data = nc.Dataset(true_path + 'stats/Stats.'+ case_name + '.nc', 'r')

    # consider opening a matrix for costfun and storing all the iterations
    ncore = 1
    txt = 'ABCDEFGHIJK'
    fname = 'tuning_record_' + case_name + txt[int(ncore)] + '.nc'
    #tuning_record = nc.Dataset(fname, 'w')
    initiate_record(fname)

    # define the lambda function to compute the cost function theta for each iteration
    costFun = lambda theta, geom_opt: scm_iteration.scm_iter(true_data, theta, case_name, fname, geom_opt)

    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO], args.step_sizes[args.algNO], args.step_nums[args.algNO]))

    theta0 = 0.9
    # call Parallel_mcmc.py
    mc_fun = geoMC.geoMC(theta0, costFun, args.algs[args.algNO],
                         args.step_sizes[args.algNO], args.step_nums[args.algNO], -.5 * np.ones(args.D), [],
                         'bounce').sample

    mc_args = (args.num_samp, args.num_burnin)
    mc_fun(*mc_args)
    return

def initiate_record(fname):


    tuning_record = nc.Dataset(fname, "w", format="NETCDF4")
    print 'yair'
    grp_stats = tuning_record.createGroup('data')
    grp_stats.createDimension('z', 75) # get this from namelistfile
    grp_stats.createDimension('t', 361) # get this from namelistfile
    grp_stats.createDimension('dim', None)
    grp_stats.createDimension('sim', 1)
    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
    thetal_mean = grp_stats.createVariable('thetal', 'f4', ('t', 'z', 'dim'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('t', 'z', 'dim'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('t', 'z', 'dim'))
    temperature = grp_stats.createVariable('temperature', 'f4', ('t', 'z', 'dim'))
    tune_param = grp_stats.createVariable('tune_param', 'f4', 'dim')
    costFun = grp_stats.createVariable('costFun', 'f4', 'dim')  # this might be a problem if dim=1 implies 2 value
    nsim = grp_stats.createVariable('nsim', 'f4', 'sim')
    nsim = 0
    appendvar = tuning_record.variables['nsim']
    appendvar = nsim
    tuning_record.close()
    return


if __name__ == '__main__':
    main()
