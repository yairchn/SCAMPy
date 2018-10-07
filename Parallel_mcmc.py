import subprocess
import generate_namelist
import json
import numpy as np
import argparse
import math
import pprint
import os

lons = np.linspace(0,180,36)
lons = lons[::-1]
times_retained = list(np.arange(100)* 86400)
# pefect model
# python Parallel_mcmc.py ncore casename        truepath                               num_samp num_burn modeltype   theta
# python Parallel_mcmc.py 5       TRMM_LBA     '/cluster/home/yairc/SCAMPy/LES_stats/' 1000     500       SCM        50.0
# python Parallel_mcmc.py 5       Bomex        '/cluster/home/yairc/SCAMPy/LES_stats/' 1000     500       LES        50.0
# python Parallel_mcmc.py 5       Bomex        '/cluster/home/yairc/SCAMPy/LES_stats/' 1000     500       SCM        50.0
# python Parallel_mcmc.py 5       TRMM_LBA     '/cluster/home/yairc/SCAMPy/LES_stats/' 1000     500       LES        50.0
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('ncores', type=int, default=5)
    parser.add_argument('case_name')
    parser.add_argument('true_path')
    parser.add_argument('num_samp',  type=int, default=6000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=500)
    parser.add_argument('model_type')
    parser.add_argument('theta')
    args = parser.parse_args()
    ncores = args.ncores
    case_name = args.case_name
    true_path = args.true_path
    num_samp = int(args.num_samp)
    num_burnin = args.num_burnin
    model_type = args.model_type
    theta = args.theta

    # generate namelist and edit output to scratch folder
    subprocess.call("python generate_namelist.py " + case_name, shell=True)
    namelistfile = open('/cluster/home/yairc/SCAMPy/' + case_name + '.in', 'r+')
    namelist = json.load(namelistfile)
    namelist['output']['output_root'] = '/scratch/yairc/SCAMPy/'
    if case_name == 'TRMM_LBA':
       namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'inverse_w'
    else:
       namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'b_w2'
    print 'entrainment closure '=  namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']
    newnamelistfile = open('/cluster/home/yairc/SCAMPy/' + case_name + '.in','w')
    json.dump(namelist, newnamelistfile, sort_keys=True, indent=4)
    newnamelistfile.close()

    num_samp_tot = num_samp+num_burnin

    for i in range(0,ncores):
        ncore = i
        #for len(theta)>1
        #    run_str = 'bsub -n 1 -W 120:00 mpirun python mcmc_tuningP.py ' + str(ncore) + ' ' + case_name + ' ' + true_path + ' ' + str(num_samp) + ' ' + str(num_burnin)+ ' ' + model_type

        run_str = 'bsub -n 1 -W 120:00 mpirun python mcmc_tuningP.py ' + str(ncore) + ' ' + str(
            theta) + ' ' + case_name + ' ' + true_path + ' ' + str(num_samp_tot) + ' ' + str(num_burnin) + ' ' + model_type
        print(run_str)
        subprocess.call([run_str], shell=True)


    return


if __name__ == "__main__":
    main()
