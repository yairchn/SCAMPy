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

# python Parallel_mcmc.py 0.7 5 Bomex '/cluster/scratch/yairc/scampy/Output.Bomex.original/' 6000 1000
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('theta')
    parser.add_argument('ncores', type=int, default=5)
    parser.add_argument('case_name')
    parser.add_argument('true_path')
    parser.add_argument('num_samp',  type=int, default=6000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=1000)
    args = parser.parse_args()
    theta = args.theta
    ncores = args.ncores
    case_name = args.case_name
    true_path = args.true_path
    num_samp_tot = int(args.num_samp)
    num_burnin = args.num_burnin


    # generate namelist and edit output to scratch folder
    subprocess.call("python generate_namelist.py " + case_name, shell=True)
    namelistfile = open('/cluster/home/yairc/scampy/' + case_name + '.in', 'r+')
    namelist = json.load(namelistfile)
    namelist['output']['output_root'] = '/cluster/scratch/yairc/scampy/'
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'buoyancy_sorting'
    newnamelistfile = open('/cluster/home/yairc/scampy/' + case_name + '.in','w')
    json.dump(namelist, newnamelistfile, sort_keys=True, indent=4)
    newnamelistfile.close()

    num_samp = math.trunc((num_samp_tot-num_burnin)/ncores) + num_burnin

    for i in range(0,ncores):
        ncore = i
        run_str = 'bsub -n 1 -W 24:00 mpirun python mcmc_tuningP.py ' + str(ncore) + ' ' + str(theta) + ' ' + case_name + ' ' + true_path + ' ' + str(num_samp) + ' ' + str(num_burnin)
        print(run_str)
        subprocess.call([run_str], shell=True)


    return


if __name__ == "__main__":
    main()
