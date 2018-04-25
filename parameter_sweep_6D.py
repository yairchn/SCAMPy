import subprocess
import argparse
import json
import numpy as np
import netCDF4 as nc
import os
from shutil import copyfile
import pylab as plt
# python parameter_sweep_6D.py case_name
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()
    case_name = args.case_name

    file_case = open(case_name + '.in').read()
    namelist = json.loads(file_case)
    uuid = namelist['meta']['uuid']
    print(uuid)
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

    nvar = 2
    III = np.zeros(nvar)
    sweep_var = np.linspace(0.0, 1.0, num=nvar)

    #sweep_var,sweep_var = np.meshgrid(sweep_varn,sweep_varm )
    #sweep_var = [0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18]
    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5

    # destination = '/Users/yaircohen/Documents/SCAMPy_out/parameter_sweep/'
    # out_stats = nc.Dataset(destination + '/Stats.sweep_'+case_name+'.nc', 'w', format='NETCDF4')
    # grp_stats = out_stats.createGroup('profiles')
    # grp_stats.createDimension('z', nz)
    # grp_stats.createDimension('t', nt)
    # grp_stats.createDimension('var', nvar)
    # grp_stats.createDimension('r', nr)

    _z = np.zeros((nz))
    _t = np.zeros((nt))
    _lwp = np.zeros((nt,nvar,nvar,nvar,nvar,nvar,nvar))
    _cloud_cover = np.zeros((nt,nvar,nvar,nvar,nvar,nvar,nvar))
    _cloud_top = np.zeros((nt,nvar,nvar,nvar,nvar,nvar,nvar))
    _cloud_base = np.zeros((nt,nvar,nvar,nvar,nvar,nvar,nvar))
    _updraft_area = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _ql_mean = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _qt_mean = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _temperature_mean = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _RH_mean = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _updraft_w = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _thetal_mean = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _massflux = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _buoyancy_mean = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _env_tke = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _updraft_thetal_precip = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _massflux = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))
    _sweep_var = np.zeros((nt,nz,nvar,nvar,nvar,nvar,nvar,nvar))


    for i1 in range(0,nvar):
        sweep_var_i1 = sweep_var[i1]
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
                            paramlist = sweep(sweep_var_ij)
                            write_file(paramlist)
                            file_case = open('paramlist_sweep.in').read()
                            current = json.loads(file_case)
                            print('========================')
                            print('running '+case_name+' var = '+ str(sweep_var_ij))
                            print('========================')
                            #subprocess.call("python main.py " + case_name + "_sweep.in paramlist_sweep.in", shell=True)
                            try:
                                subprocess.check_output("python main.py " + case_name + "_sweep.in paramlist_sweep.in", shell=True)

                                data = nc.Dataset(path, 'r')
                                zz = data.groups['profiles'].variables['z']
                                tt = data.groups['profiles'].variables['t']

                                lwp_ = np.multiply(data.groups['timeseries'].variables['lwp'], 1.0)
                                cloud_cover_ = np.multiply(data.groups['timeseries'].variables['cloud_cover'], 1.0)
                                cloud_top_ = np.multiply(data.groups['timeseries'].variables['cloud_top'], 1.0)
                                cloud_base_ = np.multiply(data.groups['timeseries'].variables['cloud_base'], 1.0)

                                updraft_area_ = np.multiply(data.groups['profiles'].variables['updraft_area'], 1.0)
                                ql_mean_ = np.multiply(data.groups['profiles'].variables['ql_mean'], 1.0)
                                qt_mean_ = np.multiply(data.groups['profiles'].variables['qt_mean'], 1.0)
                                temperature_mean_ = np.multiply(data.groups['profiles'].variables['temperature_mean'],
                                                                1.0)
                                updraft_w_ = np.multiply(data.groups['profiles'].variables['updraft_w'], 1.0)
                                thetal_mean_ = np.multiply(data.groups['profiles'].variables['thetal_mean'], 1.0)
                                massflux_ = np.multiply(data.groups['profiles'].variables['massflux'], 1.0)
                                buoyancy_mean_ = np.multiply(data.groups['profiles'].variables['buoyancy_mean'], 1.0)
                                env_tke_ = np.multiply(data.groups['profiles'].variables['env_tke'], 1.0)
                                updraft_thetal_precip_ = np.multiply(
                                    data.groups['profiles'].variables['updraft_thetal_precip'], 1.0)
                                p0 = np.multiply(data.groups['reference'].variables['p0'], 1.0)
                                P0, sP0 = np.meshgrid(p0, p0)
                                FT = np.multiply(17.625, (np.divide(np.subtract(temperature_mean_, 273.15),
                                                                    (np.subtract(temperature_mean_, 273.15 + 243.04)))))
                                RH_mean_ = np.multiply(epsi * np.exp(FT), np.divide(
                                    np.add(np.subtract(1, qt_mean_), epsi_inv * (qt_mean_ - ql_mean_)),
                                    np.multiply(epsi_inv, np.multiply(p0, (qt_mean_ - ql_mean_)))))
                                os.remove(path)

                                _lwp[:, i1, i2, i3, i4, i5, i6] = lwp_[0:nt]
                                _cloud_cover[:, i1, i2, i3, i4, i5, i6] = cloud_cover_[0:nt]
                                _cloud_top[:, i1, i2, i3, i4, i5, i6] = cloud_top_[0:nt]
                                _cloud_base[:, i1, i2, i3, i4, i5, i6] = cloud_base_[0:nt]
                                _t = tt[0:nt]
                                _z = zz
                                _updraft_area[:, :, i1, i2, i3, i4, i5, i6] = updraft_area_[0:nt, 0:nz]
                                _ql_mean[:, :, i1, i2, i3, i4, i5, i6] = ql_mean_[0:nt, 0:nz]
                                _RH_mean[:, :, i1, i2, i3, i4, i5, i6] = RH_mean_[0:nt, 0:nz]
                                _qt_mean[:, :, i1, i2, i3, i4, i5, i6] = qt_mean_[0:nt, 0:nz]
                                _updraft_w[:, :, i1, i2, i3, i4, i5, i6] = updraft_w_[0:nt, 0:nz]
                                _thetal_mean[:, :, i1, i2, i3, i4, i5, i6] = thetal_mean_[0:nt, 0:nz]
                                _temperature_mean[:, :, i1, i2, i3, i4, i5, i6] = temperature_mean_[0:nt, 0:nz]
                                _massflux[:, :, i1, i2, i3, i4, i5, i6] = massflux_[0:nt, 0:nz]
                                _buoyancy_mean[:, :, i1, i2, i3, i4, i5, i6] = buoyancy_mean_[0:nt, 0:nz]
                                _env_tke[:, :, i1, i2, i3, i4, i5, i6] = env_tke_[0:nt, 0:nz]
                                _updraft_thetal_precip[:, :, i1, i2, i3, i4, i5, i6] = updraft_thetal_precip_[0:nt,
                                                                                       0:nz]

                            except:
                                a = np.empty(nt)
                                a[:] = np.nan
                                b = np.empty((nt, nz,))
                                b[:] = np.nan

                                _lwp[:, i1, i2, i3, i4, i5, i6] = a
                                _cloud_cover[:, i1, i2, i3, i4, i5, i6] = a
                                _cloud_top[:, i1, i2, i3, i4, i5, i6] = a
                                _cloud_base[:, i1, i2, i3, i4, i5, i6] = a
                                _updraft_area[:, :, i1, i2, i3, i4, i5, i6] = b
                                _ql_mean[:, :, i1, i2, i3, i4, i5, i6] = b
                                _RH_mean[:, :, i1, i2, i3, i4, i5, i6] = b
                                _qt_mean[:, :, i1, i2, i3, i4, i5, i6] = b
                                _updraft_w[:, :, i1, i2, i3, i4, i5, i6] = b
                                _thetal_mean[:, :, i1, i2, i3, i4, i5, i6] = b
                                _temperature_mean[:, :, i1, i2, i3, i4, i5, i6] = b
                                _massflux[:, :, i1, i2, i3, i4, i5, i6] = b
                                _buoyancy_mean[:, :, i1, i2, i3, i4, i5, i6] = b
                                _env_tke[:, :, i1, i2, i3, i4, i5, i6] = b
                                _updraft_thetal_precip[:, :, i1, i2, i3, i4, i5, i6] = b



    os.remove(path1)
    _sweep_var = sweep_var
    destination = '/Users/yaircohen/Documents/SCAMPy_out/parameter_sweep/'
    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + '.nc', 'w', format='NETCDF4')
    grp_stats = out_stats.createGroup('profiles')
    grp_stats.createDimension('z', nz)
    grp_stats.createDimension('t', nt)
    grp_stats.createDimension('var1', nvar)
    grp_stats.createDimension('var2', nvar)
    grp_stats.createDimension('var3', nvar)
    grp_stats.createDimension('var4', nvar)
    grp_stats.createDimension('var5', nvar)
    grp_stats.createDimension('var6', nvar)

    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    var = grp_stats.createVariable('var', 'f4', 'var1')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'var1','var2','var3','var4','var5','var6'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'var1','var2','var3','var4','var5','var6'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'var1','var2','var3','var4','var5','var6'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'var1','var2','var3','var4','var5','var6'))
    updraft_area = grp_stats.createVariable('updraft_area', 'f4', ('t', 'z','var1','var2','var3','var4','var5','var6'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    temperature_mean = grp_stats.createVariable('temperature_mean', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    RH_mean = grp_stats.createVariable('RH_mean', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    updraft_w = grp_stats.createVariable('updraft_w', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    buoyancy_mean = grp_stats.createVariable('buoyancy_mean', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    massflux = grp_stats.createVariable('massflux', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    env_tke = grp_stats.createVariable('env_tke', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    updraft_thetal_precip = grp_stats.createVariable('updraft_thetal_precip', 'f4', ('t', 'z', 'var1','var2','var3','var4','var5','var6'))
    #costFun = grp_stats.createVariable('costFun', 'f4', ('r','var_n','var_m'))
    print np.shape(_sweep_var)
    print np.shape(var)

    var[:] = _sweep_var
    t[:] = _t
    z[:] = _z
    lwp[:,:,:,:,:,:,:] = _lwp
    cloud_cover[:,:,:,:,:,:,:] = _cloud_cover
    cloud_top[:,:,:,:,:,:,:] = _cloud_top
    cloud_base[:,:,:,:,:,:,:] = _cloud_base
    updraft_area[:,:,:,:,:,:,:,:] = _updraft_area
    ql_mean[:,:,:,:,:,:,:,:] = _ql_mean
    qt_mean[:, :,:,:,:,:,:,:] = _qt_mean
    temperature_mean[:, :,:,:,:,:,:,:] = _temperature_mean
    RH_mean[:, :,:,:,:,:,:,:] = _RH_mean
    updraft_w[:,:,:,:,:,:,:,:] = _updraft_w
    thetal_mean[:,:,:,:,:,:,:,:] = _thetal_mean
    buoyancy_mean[:,:,:,:,:,:,:,:] = _buoyancy_mean
    massflux[:,:,:,:,:,:,:,:] = _massflux
    env_tke[:,:,:,:,:,:,:,:] = _env_tke
    updraft_thetal_precip[:,:,:,:,:,:,:,:] = _updraft_thetal_precip

    print '---------------------------------'
    print np.shape(var)
    print np.shape(_sweep_var)
    print III

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
