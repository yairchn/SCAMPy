import subprocess
import argparse
import json
import numpy as np
import netCDF4 as nc
import os
from shutil import copyfile

# python parameter_sweep.py case_name
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

    nt = int((tmax+dt)/freq)   # nt = int(tmax/freq)+1
    II=0
    nr = 5
    nvar = 50
    sweep_var = np.linspace(2.0, 5.0, num=nvar)
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
    _lwp = np.zeros((nt,nvar))
    _cloud_cover = np.zeros((nt,nvar))
    _cloud_top = np.zeros((nt,nvar))
    _cloud_base = np.zeros((nt,nvar))
    _updraft_area = np.zeros((nt,nz,nvar))
    _ql_mean = np.zeros((nt,nz,nvar))
    _qt_mean = np.zeros((nt, nz, nvar))
    _temperature_mean = np.zeros((nt, nz, nvar))
    _RH_mean = np.zeros((nt, nz, nvar))
    _updraft_w = np.zeros((nt,nz,nvar))
    _thetal_mean = np.zeros((nt,nz,nvar))
    _massflux = np.zeros((nt, nz, nvar))
    _buoyancy_mean = np.zeros((nt,nz,nvar))
    _env_tke = np.zeros((nt,nz,nvar))
    _updraft_thetal_precip = np.zeros((nt,nz,nvar))
    _massflux = np.zeros((nt,nz,nvar))
    _sweep_var = np.zeros(nvar)


    for i in range(0,nvar):
        sweep_var_i = sweep_var[i]
        paramlist = sweep(sweep_var_i)
        write_file(paramlist)
        file_case = open('paramlist_sweep.in').read()
        current = json.loads(file_case)
        print('========================')
        print('running '+case_name+' var = '+ str(sweep_var_i))
        print('========================')
        subprocess.call("python main.py " + case_name + "_sweep.in paramlist_sweep.in", shell=True)

        data = nc.Dataset(path, 'r')
        zz = data.groups['profiles'].variables['z']
        tt = data.groups['profiles'].variables['t']

        lwp_ = np.multiply(data.groups['timeseries'].variables['lwp'], 1.0)
        cloud_cover_ = np.multiply(data.groups['timeseries'].variables['cloud_cover'],1.0)
        cloud_top_ = np.multiply(data.groups['timeseries'].variables['cloud_top'],1.0)
        cloud_base_ = np.multiply(data.groups['timeseries'].variables['cloud_base'],1.0)

        updraft_area_ = np.multiply(data.groups['profiles'].variables['updraft_area'],1.0)
        ql_mean_ = np.multiply(data.groups['profiles'].variables['ql_mean'],1.0)
        qt_mean_ = np.multiply(data.groups['profiles'].variables['qt_mean'], 1.0)
        temperature_mean_ = np.multiply(data.groups['profiles'].variables['temperature_mean'], 1.0)
        updraft_w_ = np.multiply(data.groups['profiles'].variables['updraft_w'],1.0)
        thetal_mean_ = np.multiply(data.groups['profiles'].variables['thetal_mean'],1.0)
        massflux_ = np.multiply(data.groups['profiles'].variables['massflux'], 1.0)
        buoyancy_mean_ = np.multiply(data.groups['profiles'].variables['buoyancy_mean'],1.0)
        env_tke_ = np.multiply(data.groups['profiles'].variables['env_tke'],1.0)
        updraft_thetal_precip_ = np.multiply(data.groups['profiles'].variables['updraft_thetal_precip'], 1.0)
        p0 = np.multiply(data.groups['reference'].variables['p0'],1.0)
        P0, sP0 = np.meshgrid(p0, p0)
        FT = np.multiply(17.625, (np.divide(np.subtract(temperature_mean_, 273.15), (np.subtract(temperature_mean_, 273.15 + 243.04)))))
        RH_mean_ = np.multiply(epsi * np.exp(FT), np.divide(np.add(np.subtract(1, qt_mean_), epsi_inv * (qt_mean_-ql_mean_)),
                                                         np.multiply(epsi_inv, np.multiply(p0,(qt_mean_-ql_mean_)))))
        os.remove(path)
        print np.shape(lwp_)
        try:

            _lwp[:, II] = lwp_[0:nt]
            _cloud_cover[:,II] = cloud_cover_[0:nt]
            _cloud_top[:,II] = cloud_top_[0:nt]
            _cloud_base[:,II] = cloud_base_[0:nt]
            _t = tt[0:nt]
            _z = zz
            _updraft_area[:,:,II] = updraft_area_[0:nt,0:nz]
            _ql_mean[:,:,II] = ql_mean_[0:nt,0:nz]
            _qt_mean[:, :, II] = qt_mean_[0:nt, 0:nz]
            _updraft_w[:,:,II] = updraft_w_[0:nt,0:nz]
            _thetal_mean[:,:,II] = thetal_mean_[0:nt,0:nz]
            _temperature_mean[:, :, II] = temperature_mean_[0:nt, 0:nz]
            _massflux[:, :, II] = massflux_[0:nt, 0:nz]
            _buoyancy_mean[:,:,II] = buoyancy_mean_[0:nt,0:nz]
            _env_tke[:,:,II] = env_tke_[0:nt,0:nz]
            _updraft_thetal_precip[:,:,II] = updraft_thetal_precip_[0:nt,0:nz]
            _sweep_var[II] = sweep_var_i
            II += 1
        except:
            print 'passing at ', II
            pass



        os.remove(path1)

    destination = '/Users/yaircohen/Documents/SCAMPy_out/parameter_sweep/'
    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + '.nc', 'w', format='NETCDF4')
    grp_stats = out_stats.createGroup('profiles')
    grp_stats.createDimension('z', nz)
    grp_stats.createDimension('t', nt)
    grp_stats.createDimension('var', II)

    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    var = grp_stats.createVariable('var', 'f4', 'var')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'var'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'var'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'var'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'var'))
    updraft_area = grp_stats.createVariable('updraft_area', 'f4', ('t', 'z','var'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('t', 'z', 'var'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('t', 'z', 'var'))
    temperature_mean = grp_stats.createVariable('temperature_mean', 'f4', ('t', 'z', 'var'))
    RH_mean = grp_stats.createVariable('RH_mean', 'f4', ('t', 'z', 'var'))
    updraft_w = grp_stats.createVariable('updraft_w', 'f4', ('t', 'z', 'var'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4', ('t', 'z', 'var'))
    buoyancy_mean = grp_stats.createVariable('buoyancy_mean', 'f4', ('t', 'z', 'var'))
    massflux = grp_stats.createVariable('massflux', 'f4', ('t', 'z', 'var'))
    env_tke = grp_stats.createVariable('env_tke', 'f4', ('t', 'z', 'var'))
    updraft_thetal_precip = grp_stats.createVariable('updraft_thetal_precip', 'f4', ('t', 'z', 'var'))
    #costFun = grp_stats.createVariable('costFun', 'f4', ('r','var'))
    print np.shape(_sweep_var)
    print np.shape(II)

    var[:] = _sweep_var
    t[:] = _t
    z[:] = _z
    lwp[:,:] = _lwp
    cloud_cover[:,:] = _cloud_cover
    cloud_top[:,:] = _cloud_top
    cloud_base[:,:] = _cloud_base
    updraft_area[:,:,:] = _updraft_area
    ql_mean[:,:,:] = _ql_mean
    qt_mean[:, :, :] = _qt_mean
    temperature_mean[:, :, :] = _temperature_mean
    RH_mean[:, :, :] = _RH_mean
    updraft_w[:,:,:] = _updraft_w
    thetal_mean[:,:,:] = _thetal_mean
    buoyancy_mean[:,:,:] = _buoyancy_mean
    massflux[:, :, :] = _massflux
    env_tke[:,:,:] = _env_tke
    updraft_thetal_precip[:, :, :] = _updraft_thetal_precip

    print '---------------------------------'
    print np.shape(var)
    print np.shape(_sweep_var)
    print II
    print '---------------------------------'
    var[:] = _sweep_var[0:II]
    print np.shape(_t)
    print np.shape(t)
    #t[:] = _t
    #z[:] = _z
    print '---------------------------------'
    print np.shape(lwp)
    print np.shape(_lwp)
    print II
    print '---------------------------------'

    lwp[:,:] = _lwp[:,0:II]
    cloud_cover[:,:] = _cloud_cover[:,0:II]
    cloud_top[:,:] = _cloud_top[:,0:II]
    cloud_base[:,:] = _cloud_base[:,0:II]
    updraft_area[:,:,:] = _updraft_area[:,:,0:II]
    ql_mean[:,:,:] = _ql_mean[:,:,0:II]
    qt_mean[:, :, :] = _qt_mean[:, :, 0:II]
    updraft_w[:,:,:] = _updraft_w[:,:,0:II]
    massflux[:,:,:] = _massflux[:,:,0:II]
    buoyancy_mean[:,:,:] = _buoyancy_mean[:,:,0:II]
    env_tke[:,:,:] = _env_tke[:,:,0:II]
    updraft_thetal_precip[:, :, :] = _updraft_thetal_precip[:,:,0:II]

    out_stats.close()
    print('========================')
    print('======= SWEEP END ======')
    print('========================')





def sweep(sweep_var_i):
    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = 'sweep'

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.0

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.2
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.3
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 5.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = float(sweep_var_i)
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = float(sweep_var_i)
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
