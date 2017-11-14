import argparse
import json


def main():
    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='SCAMPy')
    parser.add_argument("namelist")
    parser.add_argument("paramlist")
    args = parser.parse_args()
    #namelist = args.namelist[0:6]+args.namelist[-3:]
    file_namelist1 = open("Bomex0.in").read()
    file_namelist = open(args.namelist).read()
    file_paramlist = open(args.paramlist).read()
    print(file_namelist)
    print(type(file_namelist))
    namelist1 = json.loads(file_namelist1)
    print('Bomex0.in works')
    namelist = json.loads(file_namelist)
    del file_namelist
    paramlist = json.loads(file_paramlist)
    del file_paramlist

    main1d(namelist, paramlist)

    return

def main1d(namelist, paramlist):
    import Simulation1d
    Simulation = Simulation1d.Simulation1d(namelist, paramlist)
    Simulation.initialize(namelist)
    Simulation.run()
    print('The simulation has completed.')

    return

if __name__ == "__main__":
    main()





