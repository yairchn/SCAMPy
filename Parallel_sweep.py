import subprocess
import numpy as np
import argparse


lons = np.linspace(0,180,36)
lons = lons[::-1]
times_retained = list(np.arange(100)* 86400)

# python Parallel_sweep.py Bomex

def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()
    case_name = args.case_name

    nvar = 10
    sweep_var = np.linspace(0.0, 1.0, num=nvar)
    txt = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for i in range(0,nvar):
        sweep_var_i = sweep_var[i]
        txt_i = txt[i]
        run_str = 'bsub -n 1 -W 120:00 mpirun python parameter_sweep_P.py ' + case_name + ' ' +  str(sweep_var_i) + ' ' + txt_i
        print(run_str)
        subprocess.call([run_str], shell=True)


    return


if __name__ == "__main__":
    main()
