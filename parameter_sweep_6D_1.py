import subprocess
import argparse
import json
import numpy as np
import netCDF4 as nc
import os
from shutil import copyfile

import pylab as plt
# python parameter_sweep_6D_1.py case_name
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()
    case_name = args.case_name

    file_case = open(case_name + '.in').read()
    namelist = json.loads(file_case)
    uuid = namelist['meta']['uuid']
    path = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[-5:] + '/stats/Stats.' + case_name + '.nc'
    path1 = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[-5:] + '/paramlist_sweep.in'
    tmax = namelist['time_stepping']['t_max']
    dt = namelist['time_stepping']['dt']

    freq = namelist['stats_io']['frequency']
    nz   = namelist['grid']['nz']

    src = '/Users/yaircohen/PycharmProjects/scampy/' + case_name + '_sweep.in'
    dst = '/Users/yaircohen/PycharmProjects/scampy/' + case_name + '.in'
    copyfile(src, dst)

    nt = int((tmax+dt)/freq)+1   # nt = int(tmax/freq)+1
    nvar = 40
    sweep_var = np.linspace(0.1, 4.0, num=nvar)

    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5

    destination = '/Users/yaircohen/Documents/SCAMPy_out/parameter_sweep/'
    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + '.nc', 'w', format='NETCDF4')
    grp_stats = out_stats.createGroup('profiles')
    grp_stats.createDimension('zz', nz)

    grp_stats.createDimension('tt', nt)
    grp_stats.createDimension('var1', 6)
    grp_stats.createDimension('var2', nvar)

    time = grp_stats.createVariable('t', 'f4', 'tt')
    height = grp_stats.createVariable('z', 'f4', 'zz')
    var = grp_stats.createVariable('var', 'f4', 'var2')
    lwp = grp_stats.createVariable('lwp', 'f4', ('tt', 'var1', 'var2'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('tt', 'var1', 'var2'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('tt', 'var1', 'var2'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('tt', 'var1', 'var2'))
    updraft_area = grp_stats.createVariable('updraft_area', 'f4',('tt', 'zz', 'var1', 'var2'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('tt', 'zz', 'var1', 'var2'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('tt', 'zz', 'var1', 'var2'))
    temperature_mean = grp_stats.createVariable('temperature_mean', 'f4',('tt', 'zz', 'var1', 'var2'))
    RH_mean = grp_stats.createVariable('RH_mean', 'f4', ('tt', 'zz', 'var1', 'var2'))
    updraft_w = grp_stats.createVariable('updraft_w', 'f4',('tt', 'zz', 'var1', 'var2'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4',('tt', 'zz', 'var1', 'var2'))
    buoyancy_mean = grp_stats.createVariable('buoyancy_mean', 'f4',('tt', 'zz', 'var1', 'var2'))
    massflux = grp_stats.createVariable('massflux', 'f4', ('tt', 'zz', 'var1', 'var2'))
    env_tke = grp_stats.createVariable('env_tke', 'f4', ('tt', 'zz', 'var1', 'var2'))
    updraft_thetal_precip = grp_stats.createVariable('updraft_thetal_precip', 'f4',('tt', 'zz', 'var1', 'var2'))
    out_stats.close()

    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + '.nc', 'a', format='NETCDF4')
    var[:] = sweep_var

    for i1 in range(0,6):
        for i2 in range(0, nvar):
            sweep_var_ij = [0.0,0.0,0.0,0.0,0.0,0.0]
            if i1==5:
                print sweep_var_ij
            sweep_var_ij[i1] = sweep_var[i2]
            paramlist = sweep(sweep_var_ij)
            write_file(paramlist)
            file_case = open('paramlist_sweep.in').read()
            current = json.loads(file_case)
            #subprocess.call("python main.py " + case_name + "_sweep.in paramlist_sweep.in", shell=True)
            #try:
            subprocess.check_output("python main.py " + case_name + "_sweep.in paramlist_sweep.in", shell=True)
            print('========================')
            print('running ' + case_name + ' var = ' + str(sweep_var_ij))
            print('========================')
            data = nc.Dataset(path, 'r')
            height[:] = data.groups['profiles'].variables['z']
            time[:] = data.groups['profiles'].variables['t']
            lwp[:, i1, i2] = np.multiply(data.groups['timeseries'].variables['lwp'],1.0)
            cloud_cover[:, i1, i2] = np.multiply(data.groups['timeseries'].variables['cloud_cover'],1.0)
            cloud_top[:, i1, i2] = np.multiply(data.groups['timeseries'].variables['cloud_top'],1.0)
            cloud_base[:, i1, i2] = np.multiply(data.groups['timeseries'].variables['cloud_base'],1.0)

            updraft_area[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['updraft_area'],1.0)
            ql_mean[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['ql_mean'],1.0)
            qt_mean[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['qt_mean'],1.0)
            thetal_mean[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['thetal_mean'],1.0)
            temperature_mean[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['temperature_mean'],1.0)
            massflux[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['massflux'],1.0)
            buoyancy_mean[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['buoyancy_mean'],1.0)
            env_tke[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['env_tke'],1.0)
            updraft_thetal_precip[:, :, i1, i2] = np.multiply(data.groups['profiles'].variables['updraft_thetal_precip'],1.0)


            p0 = np.multiply(data.groups['reference'].variables['p0'], 1.0)
            P0, sP0 = np.meshgrid(p0, p0)
            temperature_mean_ = np.multiply(data.groups['profiles'].variables['temperature_mean'],
                                            1.0)
            FT = np.multiply(17.625, (np.divide(np.subtract(temperature_mean_, 273.15),
                                                (np.subtract(temperature_mean_, 273.15 + 243.04)))))
            qt_mean_ = np.multiply(data.groups['profiles'].variables['qt_mean'], 1.0)
            ql_mean_ = np.multiply(data.groups['profiles'].variables['ql_mean'], 1.0)
            RH_mean_ = np.multiply(epsi * np.exp(FT), np.divide(
                np.add(np.subtract(1, qt_mean_), epsi_inv * (qt_mean_ - ql_mean_)),
                np.multiply(epsi_inv, np.multiply(p0, (qt_mean_ - ql_mean_)))))
            RH_mean[:, :, i1, i2] = RH_mean_

            os.remove(path)
            # except:
            #     print('failed ' + case_name + ' var = ' + str(sweep_var_ij))
            #     a = np.empty(nt)
            #     a[:] = np.nan
            #     b = np.empty((nt, nz,))
            #     b[:] = np.nan
            #
            #     lwp[:, i1, i2] = a
            #     cloud_cover[:, i1, i2] = a
            #     cloud_top[:, i1, i2] = a
            #     cloud_base[:, i1, i2] = a
            #     updraft_area[:, :, i1, i2] = b
            #     ql_mean[:, :, i1, i2] = b
            #     RH_mean[:, :, i1, i2] = b
            #     qt_mean[:, :, i1, i2] = b
            #     updraft_w[:, :, i1, i2] = b
            #     thetal_mean[:, :, i1, i2] = b
            #     temperature_mean[:, :, i1, i2] = b
            #     massflux[:, :, i1, i2] = b
            #     buoyancy_mean[:, :, i1, i2] = b
            #     env_tke[:, :, i1, i2] = b
            #     updraft_thetal_precip[:, :, i1, i2] = b

    out_stats.close()
    print('========================')
    print('======= SWEEP END ======')
    print('========================')

def sweep(sweep_var_ij):
    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = 'sweep'

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.0

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['domain_length'] = 5000.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.2
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.3
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 5.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha1'] = float(sweep_var_ij[0])
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha2'] = float(sweep_var_ij[1])
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha3'] = float(sweep_var_ij[2])
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha1'] = float(sweep_var_ij[3])
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha2'] = float(sweep_var_ij[4])
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha3'] = float(sweep_var_ij[5])

    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0 / 3.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff'] = 0.375
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing'] = 1500.0
    paramlist['turbulence']['updraft_microphysics'] = {}
    paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.01

    return paramlist

def write_file(paramlist):

    fh = open('paramlist_'+paramlist['meta']['casename']+ '.in', 'w')
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return

if __name__ == '__main__':
    main()
