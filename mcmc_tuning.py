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
    parser.add_argument('num_samp', nargs='?', type=int, default=6000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.1, .1, 1, 1, .7])
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

    # define the lambda function to compute the cost function theta for each iteration
    costFun = lambda theta, geom_opt: scm_iteration.scm_iter(true_data, theta, case_name, geom_opt)

    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO], args.step_sizes[args.algNO], args.step_nums[args.algNO]))

    theta0 = 0.9
    mc_fun = geoMC.geoMC(theta0, costFun, args.algs[args.algNO],
                         args.step_sizes[args.algNO], args.step_nums[args.algNO], -.5 * np.ones(args.D), [],
                         'bounce').sample

    mc_args = (args.num_samp, args.num_burnin)
    mc_fun(*mc_args)

if __name__ == '__main__':
    main()
