import argparse

# python sh_generator.py 5 case true_path num_samp num_burnin model_type theta
# python sh_generator.py 5 Bomex '/cluster/home/yairc/SCAMPy/LES_stats/' 1000 500 LES 50.0
def main():

    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('ncore', type=int, default=5)
    parser.add_argument('case_name')
    parser.add_argument('true_path')
    parser.add_argument('num_samp',  type=int, default=6000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=500)
    parser.add_argument('step_sizes', nargs='?', type=int, default=0.05)
    parser.add_argument('step_nums', nargs='?', type=int, default=1)
    parser.add_argument('model_type')
    parser.add_argument('theta')
    args = parser.parse_args()
    ncore = args.ncore
    case_name = args.case_name
    true_path = args.true_path
    num_samp = int(args.num_samp)
    num_burnin = args.num_burnin
    step_sizes = args.step_sizes
    step_nums =  args.step_nums
    model_type = args.model_type
    theta = args.theta
    jobname = case_name + "_mcmc"
    sh_file=open("run_"+str(ncore)+".sh",mode="w")
    sh_file.write("#!/bin/bash")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH --time=00:10:00 # walltime")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH --ntasks=64   # number of processor cores (i.e. tasks)")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("##SBATCH --qos=debug")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH --mem-per-cpu=1G   # memory per CPU core")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("#SBATCH -J "+jobname+"   # job name")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("#LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE")
    sh_file.write("\n")
    sh_file.write("\n")
    sh_file.write("srun python mcmc_tuningP.py " + str(ncore) + " " + str(theta) + " " + case_name + " " + true_path + " " + str(num_samp+num_burnin) + " " + str(num_burnin) + " " + model_type)
    sh_file.close()

    return

if __name__ == "__main__":
    main()
