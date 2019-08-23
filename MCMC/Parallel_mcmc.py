import subprocess
# import generate_namelist
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
    parser.add_argument('step_sizes', nargs='?', type=int, default=0.05)
    parser.add_argument('step_nums', nargs='?', type=int, default=1)
    parser.add_argument('model_type')
    parser.add_argument('theta')
    args = parser.parse_args()
    ncores = args.ncores
    case_name = args.case_name
    true_path = args.true_path
    num_samp = int(args.num_samp)
    num_burnin = args.num_burnin
    step_sizes = args.step_sizes
    step_nums =  args.step_nums
    model_type = args.model_type
    theta = args.theta

    localpath = os.getcwd()
    # generate namelist and edit output to scratch folder
    subprocess.call("python generate_namelist.py " + case_name, shell=True, cwd=localpath[0:-4])
    namelistfile = open(localpath[0:-4] + case_name + '.in', 'r+')
    namelist = json.load(namelistfile)
    namelist['output']['output_root'] = localpath[0:-4]
    newnamelistfile = open(localpath[0:-4] + case_name + '.in','w')
    json.dump(namelist, newnamelistfile, sort_keys=True, indent=4)
    newnamelistfile.close()

    num_samp_tot = num_samp+num_burnin

    for ncore in range(0,ncores):
        sh_generator(ncore, case_name, theta, true_path, num_samp_tot, num_burnin, model_type)
        #for len(theta)>1
        #    run_str = 'bsub -n 1 -W 120:00 mpirun python mcmc_tuningP.py ' + str(ncore) + ' ' + case_name + ' ' + true_path + ' ' + str(num_samp) + ' ' + str(num_burnin)+ ' ' + model_type
        run_str = "sbatch run_" + str(ncore) + ".sh"
        print(run_str)
        subprocess.call([run_str], shell=True)

    return


if __name__ == "__main__":
    main()


def sh_generator(ncore, case_name, theta, true_path, num_samp_tot, num_burnin, model_type):

    sh_file=open("run_"+ncore+".sh",mode="w",encoding="utf-8")
    sh_file.write("#!/bin/bash\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH --time=24:00:00\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)\n")
    sh_file.write("\n")
    sh_file.write("##SBATCH --qos=debug\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH –-mem-per-cpu=1G   # memory per CPU core\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH -J “bomex_test”\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH –-mail-type=BEGIN\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH –-mail-type=END\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH –-mail-type=FAIL\n")
    sh_file.write("\n")
    sh_file.write("#LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE\n")
    sh_file.write("\n")
    sh_file.write("srun python mcmc_tuningP.py " + str(ncore) + " " + str(theta) + " " + case_name + " " + true_path + " " + str(num_samp_tot) + " " + str(num_burnin) + " " + model_type)
    sh_file.close()
    return