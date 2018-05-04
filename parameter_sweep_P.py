import subprocess
import argparse
import json
import numpy as np
import netCDF4 as nc
import os
from shutil import copyfile
# python parameter_sweep_P.py case_name sweep_var_i txt
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    parser.add_argument('sweep_var')
    parser.add_argument('txt')
    args = parser.parse_args()
    case_name = args.case_name
    sweep_var_i = float(args.sweep_var)
    txt = args.txt

    file_case = open(case_name + '.in').read()
    namelist = json.loads(file_case)
    uuid = namelist['meta']['uuid']
    uuid[-5:] = 'swip'+txt
    namelist['meta']['uuid'] = uuid
    namelist['stats_io']['frequency'] = 600.0
    path = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[-5:] + '/stats/Stats.' + case_name + txt + '.nc'
    path1 = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[-5:] + '/paramlist_sweep'+txt+'.in'
    tmax = namelist['time_stepping']['t_max']
    dt = namelist['time_stepping']['dt']

    freq = namelist['stats_io']['frequency']
    nz   = namelist['grid']['nz']

    src = '/Users/yaircohen/PycharmProjects/scampy/' + case_name + '_sweep.in'
    dst = '/Users/yaircohen/PycharmProjects/scampy/' + case_name + txt + '.in'
    copyfile(src, dst)

    nvar = 10
    sweep_var = np.linspace(0.0, 1.0, num=nvar)

    nt = int((tmax+dt)/freq)+1   # nt = int(tmax/freq)+1
    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5

    destination = '/Users/yaircohen/Documents/SCAMPy_out/parameter_sweep/'
    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + txt + '.nc', 'w', format='NETCDF4')
    grp_stats = out_stats.createGroup('profiles')
    grp_stats.createDimension('zz', nz)
    grp_stats.createDimension('tt', nt)
    grp_stats.createDimension('var2', None)
    grp_stats.createDimension('var3', None)
    grp_stats.createDimension('var4', None)
    grp_stats.createDimension('var5', None)
    grp_stats.createDimension('var6', None)

    time = grp_stats.createVariable('t', 'f4', 'tt')
    height = grp_stats.createVariable('z', 'f4', 'zz')
    var = grp_stats.createVariable('var', 'f4', 'var2')
    lwp = grp_stats.createVariable('lwp', 'f4', ('tt', 'var2', 'var3', 'var4', 'var5', 'var6'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('tt', 'var2', 'var3', 'var4', 'var5', 'var6'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('tt', 'var2', 'var3', 'var4', 'var5', 'var6'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('tt', 'var2', 'var3', 'var4', 'var5', 'var6'))
    updraft_area = grp_stats.createVariable('updraft_area', 'f4',('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    temperature_mean = grp_stats.createVariable('temperature_mean', 'f4',('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    RH_mean = grp_stats.createVariable('RH_mean', 'f4', ('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    updraft_w = grp_stats.createVariable('updraft_w', 'f4',('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4',('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    buoyancy_mean = grp_stats.createVariable('buoyancy_mean', 'f4',('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    massflux = grp_stats.createVariable('massflux', 'f4', ('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    env_tke = grp_stats.createVariable('env_tke', 'f4', ('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    updraft_thetal_precip = grp_stats.createVariable('updraft_thetal_precip', 'f4',('tt', 'zz', 'var2', 'var3', 'var4', 'var5', 'var6'))
    out_stats.close()

    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + txt + '.nc', 'a', format='NETCDF4')

    var[:] = sweep_var

    sweep_var_i1 = sweep_var_i
    for i2 in range(0, nvar):
        sweep_var_i2 = sweep_var[i2]
        for i3 in range(0, nvar):
            sweep_var_i3 = sweep_var[i3]
            for i4 in range(0, nvar):
                sweep_var_i4 = sweep_var[i4]
                for i5 in range(0, nvar):
                    sweep_var_i5 = sweep_var[i5]
                    for i6 in range(0, nvar):
                        sweep_var_i6 = sweep_var[i6]

                        sweep_var_ij = [sweep_var_i1,sweep_var_i2,sweep_var_i3,sweep_var_i4,sweep_var_i5,sweep_var_i6]
                        paramlist = sweep(sweep_var_ij,txt)
                        write_file(paramlist)
                        file_case = open('paramlist_sweep.in').read()
                        current = json.loads(file_case)
                        print('========================')
                        print('running '+case_name+' var = '+ str(sweep_var_ij))
                        print('========================')
                        #subprocess.call("python main.py " + case_name + "_sweep.in paramlist_sweep.in", shell=True)
                        try:
                            subprocess.check_output("python main.py " + case_name + "_sweep" + txt + ".in paramlist_sweep" + txt + ".in", shell=True)
                            data = nc.Dataset(path, 'r')
                            height = data.groups['profiles'].variables['z']
                            time = data.groups['profiles'].variables['t']
                            lwp[:, i2, i3, i4, i5, i6] = data.groups['timeseries'].variables['lwp']
                            cloud_cover[:, i2, i3, i4, i5, i6] = data.groups['timeseries'].variables['cloud_cover']
                            cloud_top[:, i2, i3, i4, i5, i6] = data.groups['timeseries'].variables['cloud_top']
                            cloud_base[:, i2, i3, i4, i5, i6] = data.groups['timeseries'].variables['cloud_base']

                            updraft_area[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['updraft_area']
                            ql_mean[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['ql_mean']
                            qt_mean[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['qt_mean']
                            thetal_mean[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['thetal_mean']
                            temperature_mean[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['temperature_mean']
                            massflux[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['massflux']
                            buoyancy_mean[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['buoyancy_mean']
                            env_tke[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['env_tke']
                            updraft_thetal_precip[:, :, i2, i3, i4, i5, i6] = data.groups['profiles'].variables['updraft_thetal_precip']


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
                            RH_mean[:, :, i2, i3, i4, i5, i6] = RH_mean_

                            os.remove(path)
                        except:
                            a = np.empty(nt)
                            a[:] = np.nan
                            b = np.empty((nt, nz,))
                            b[:] = np.nan

                            lwp[:, i2, i3, i4, i5, i6] = a
                            cloud_cover[:, i2, i3, i4, i5, i6] = a
                            cloud_top[:, i2, i3, i4, i5, i6] = a
                            cloud_base[:, i2, i3, i4, i5, i6] = a
                            updraft_area[:, :, i2, i3, i4, i5, i6] = b
                            ql_mean[:, :, i2, i3, i4, i5, i6] = b
                            RH_mean[:, :, i2, i3, i4, i5, i6] = b
                            qt_mean[:, :, i2, i3, i4, i5, i6] = b
                            updraft_w[:, :, i2, i3, i4, i5, i6] = b
                            thetal_mean[:, :, i2, i3, i4, i5, i6] = b
                            temperature_mean[:, :, i2, i3, i4, i5, i6] = b
                            massflux[:, :, i2, i3, i4, i5, i6] = b
                            buoyancy_mean[:, :, i2, i3, i4, i5, i6] = b
                            env_tke[:, :, i2, i3, i4, i5, i6] = b
                            updraft_thetal_precip[:, :, i2, i3, i4, i5, i6] = b

    out_stats.close()
    print('========================')
    print('======= SWEEP END ======')
    print('========================')

def sweep(sweep_var_ij,txt):
    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = 'sweep'+txt

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.0

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['domain_length'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.2
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.3
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 5.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha1'] = float(sweep_var_ij[0])
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha2'] = float(sweep_var_ij[1])
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha3'] = float(sweep_var_ij[2])
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha1'] = float(sweep_var_ij[3])
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha2'] = float(sweep_var_ij[4])
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha3'] = float(sweep_var_ij[5])

    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0 / 3.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff'] = 0.375
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing'] = 500.0
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
