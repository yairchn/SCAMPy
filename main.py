import argparse
import json


def main():
    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='SCAMPy')
    parser.add_argument("namelist")
    parser.add_argument("paramlist")
    args = parser.parse_args()
    namelist = args.namelist[0:4]+args.namelist[6:8]
    print('new namelist in main.py is',namelist)
    #file_namelist = open(args.namelist).read()
    file_namelist = open(namelist).read()
    file_paramlist = open(args.paramlist).read()
    print(type(file_namelist)) # yair
    print(args.namelist)  # yair
    print(type(args.paramlist))  # yair
    print(args.paramlist)  # yair

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





