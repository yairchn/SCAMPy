
import numpy as np
import subprocess
import argparse
import netCDF4 as nc
import geoMC
import scm_iteration


np.set_printoptions(precision=3, suppress=True)
np.random.seed(2017)

# python run_SCM.py 'Bomex' '/Users/yaircohen/PycharmProjects/scampy/Output.Bomex.original/'
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

    # generate namelist for the tuning
    subprocess.call("python generate_namelist.py " + case_name,  shell=True) # cwd='/Users/yaircohen/PycharmProjects/scampy/',
    # load true data
    true_data = nc.Dataset(true_path + 'stats/Stats.'+ case_name + '.nc', 'r')

    # define the lambda function to compute the cost function theta for each iteration guess
    costFun = lambda theta, geom_opt: scm_iteration.scm_iter(true_data, theta, case_name, geom_opt)  # calculate costfun ,

    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO], args.step_sizes[args.algNO], args.step_nums[args.algNO]))

    theta0 = np.random.random_sample()
    #theta0 = np.random.randn(args.D)
    mc_fun = geoMC.geoMC(theta0, costFun, args.algs[args.algNO],
                         args.step_sizes[args.algNO], args.step_nums[args.algNO], -.5 * np.ones(args.D), [],
                         'bounce').sample

    mc_args = (args.num_samp, args.num_burnin)
    mc_fun(*mc_args)

#
# def MCMC_paramlist(entr_detr_factor, case_name):
#
#     paramlist = {}
#     paramlist['meta'] = {}
#     paramlist['meta']['casename'] = case_name
#
#     paramlist['turbulence'] = {}
#     paramlist['turbulence']['prandtl_number'] = 1.0
#     paramlist['turbulence']['Ri_bulk_crit'] = 0.0
#
#     paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
#     paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] =  0.1
#     paramlist['turbulence']['EDMF_PrognosticTKE']['surface_scalar_coeff'] = 0.1
#     paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.05
#     paramlist['turbulence']['EDMF_PrognosticTKE']['w_entr_coeff'] = 0.5 # "b1"
#     paramlist['turbulence']['EDMF_PrognosticTKE']['w_buoy_coeff'] =  0.5 # "b2"
#     paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.1
#     paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 5.0
#     paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = entr_detr_factor
#     paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = entr_detr_factor
#     paramlist['turbulence']['EDMF_PrognosticTKE']['vel_pressure_coeff'] = 0.0
#     paramlist['turbulence']['EDMF_PrognosticTKE']['vel_buoy_coeff'] = 1.0
#
#     paramlist['turbulence']['EDMF_BulkSteady'] = {}
#     paramlist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.05
#     paramlist['turbulence']['EDMF_BulkSteady']['w_entr_coeff'] = 2.0  #"w_b"
#     paramlist['turbulence']['EDMF_BulkSteady']['w_buoy_coeff'] = 1.0
#     paramlist['turbulence']['EDMF_BulkSteady']['max_area_factor'] = 5.0
#     paramlist['turbulence']['EDMF_BulkSteady']['entrainment_factor'] = 0.5
#     paramlist['turbulence']['EDMF_BulkSteady']['detrainment_factor'] = 0.5
#
#     paramlist['turbulence']['updraft_microphysics'] = {}
#     paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.01
#
#     return  paramlist


# def write_file(paramlist):
#
#     print('=====>',paramlist)
#     fh = open('paramlist_'+paramlist['meta']['casename']+ '.in', 'w')
#     json.dump(paramlist, fh, sort_keys=True, indent=4)
#     fh.close()
#
#     return


if __name__ == '__main__':
    main()
