import argparse
import json
import pprint
from sys import exit
import uuid
import ast
import copy

#Adapated from PyCLES: https://github.com/pressel/pycles

def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')

    args = parser.parse_args()

    case_name = args.case_name

    namelist_defaults = {}
    namelist_defaults['grid'] = {}
    namelist_defaults['grid']['dims'] = 1
    namelist_defaults['grid']['gw'] = 2

    namelist_defaults['thermodynamics'] = {}
    namelist_defaults['thermodynamics']['thermal_variable'] = 'thetal'
    namelist_defaults['thermodynamics']['sgs'] = 'mean'

    namelist_defaults['time_stepping'] = {}

    namelist_defaults['microphysics'] = {}
    namelist_defaults['microphysics']['rain_model'] = 'None'

    namelist_defaults['turbulence'] = {}
    namelist_defaults['turbulence']['scheme'] = 'EDMF_PrognosticTKE'

    namelist_defaults['turbulence']['EDMF_PrognosticTKE'] = {}
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['updraft_number'] = 1
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'moisture_deficit'
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['extrapolate_buoyancy'] = True
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts'] = False
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['use_local_micro'] = True
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['use_constant_plume_spacing'] = False
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity'] = False
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['constant_area'] = False
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['calculate_tke'] = True
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['mixing_length'] = 'sbtd_eq'

    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_closure_buoy'] = 'normalmode'
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_closure_drag'] = 'normalmode'
    namelist_defaults['turbulence']['EDMF_PrognosticTKE']['pressure_closure_asp_label'] = 'const'

    namelist_defaults['output'] = {}
    namelist_defaults['output']['output_root'] = './'

    namelist_defaults['stats_io'] = {}
    namelist_defaults['stats_io']['stats_dir'] = 'stats'
    namelist_defaults['stats_io']['frequency'] = 60.0

    namelist_defaults['meta'] = {}

    if case_name == 'Bomex':
        namelist = Bomex(namelist_defaults)
    elif case_name == 'life_cycle_Tan2018':
        namelist = life_cycle_Tan2018(namelist_defaults)
    elif case_name == 'Soares':
        namelist = Soares(namelist_defaults)
    elif case_name == 'Rico':
        namelist = Rico(namelist_defaults)
    elif case_name == 'TRMM_LBA':
        namelist = TRMM_LBA(namelist_defaults)
    elif case_name == 'ARM_SGP':
        namelist = ARM_SGP(namelist_defaults)
    elif case_name == 'GATE_III':
        namelist = GATE_III(namelist_defaults)
    elif case_name == 'DYCOMS_RF01':
        namelist = DYCOMS_RF01(namelist_defaults)
    elif case_name == 'GABLS':
        namelist = GABLS(namelist_defaults)
    elif case_name == 'SP':
        namelist = SP(namelist_defaults)
    else:
        print('Not a valid case name')
        exit()

    write_file(namelist)


def Soares(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 125
    namelist['grid']['dz'] = 30.0

    namelist['time_stepping']['dt'] = 30.0
    namelist['time_stepping']['t_max'] = 8 * 3600.0

    namelist['meta']['simname'] = 'Soares'
    namelist['meta']['casename'] = 'Soares'

    return namelist

def Bomex(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 75
    namelist['grid']['dz'] = 100 / 2.5

    namelist['time_stepping']['dt'] = 20.0
    namelist['time_stepping']['t_max'] = 21600.0

    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'

    return namelist

def life_cycle_Tan2018(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 75
    namelist['grid']['dz'] = 100 / 2.5

    namelist['time_stepping']['dt'] = 30.0
    namelist['time_stepping']['t_max'] = 6*3600.0
    namelist['meta']['simname'] = 'life_cycle_Tan2018'
    namelist['meta']['casename'] = 'life_cycle_Tan2018'

    return namelist

def Rico(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 100
    namelist['grid']['dz'] = 40.0

    namelist['time_stepping']['dt'] = 20.0
    namelist['time_stepping']['t_max'] = 86400.0

    namelist['microphysics']['rain_model'] = 'cutoff'
    # namelist['microphysics']['rain_model'] = 'clima_1m'

    namelist['meta']['simname'] = 'Rico'
    namelist['meta']['casename'] = 'Rico'

    return namelist

def TRMM_LBA(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 400
    namelist['grid']['dz'] = 40

    namelist['time_stepping']['dt'] = 30.0
    namelist['time_stepping']['t_max'] = 21600.0

    namelist['microphysics']['rain_model'] = 'cutoff'
    # namelist['microphysics']['rain_model'] = 'clima_1m'

    namelist['meta']['simname'] = 'TRMM_LBA'
    namelist['meta']['casename'] = 'TRMM_LBA'

    return namelist

def ARM_SGP(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 220
    namelist['grid']['dz'] = 20

    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 14.5
    namelist['meta']['simname'] = 'ARM_SGP'
    namelist['meta']['casename'] = 'ARM_SGP'

    return namelist

def GATE_III(namelist_defaults):

    # adopted from: "Large eddy simulation of Maritime Deep Tropical Convection",
    # By Khairoutdinov et al (2009)  JAMES, vol. 1, article #15
    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 1700
    namelist['grid']['dz'] = 10

    namelist['time_stepping']['dt'] = 5.0
    namelist['time_stepping']['t_max'] = 3600.0 * 24.0
    namelist['meta']['simname'] = 'GATE_III'
    namelist['meta']['casename'] = 'GATE_III'

    return namelist

def DYCOMS_RF01(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 300
    namelist['grid']['dz'] = 5

    namelist['time_stepping']['dt'] = 10.0
    namelist['time_stepping']['t_max'] = 60 * 60 * 4.
    namelist['meta']['simname'] = 'DYCOMS_RF01'
    namelist['meta']['casename'] = 'DYCOMS_RF01'

    return namelist

def GABLS(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 128
    namelist['grid']['dz'] = 3.125

    namelist['time_stepping']['dt'] = 1.0
    namelist['time_stepping']['t_max'] = 9 * 3600.0
    namelist['meta']['simname'] = 'GABLS'
    namelist['meta']['casename'] = 'GABLS'

    return namelist

# Sullivan Patton not fully implemented - Ignacio
def SP(namelist_defaults):

    namelist = copy.deepcopy(namelist_defaults)

    namelist['grid']['nz'] = 256
    namelist['grid']['dz'] = 8

    namelist['time_stepping']['dt'] = 5.0
    namelist['time_stepping']['t_max'] = 7200.0
    namelist['meta']['simname'] = 'SP'
    namelist['meta']['casename'] = 'SP'

    return namelist

def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    #pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
