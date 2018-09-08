import subprocess
import argparse
import json
import numpy as np
import netCDF4 as nc
import os
from shutil import copyfile
import pylab as plt
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

    nr = 5
    nvar = 10
    mvar = 10
    III = np.zeros(mvar)
    sweep_varn = np.linspace(0.0, 1.0, num=nvar)
    sweep_varm = np.linspace(0.0, 1.0, num=mvar)
    sweep_var,sweep_var = np.meshgrid(sweep_varn,sweep_varm )
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
    _lwp = np.zeros((nt,nvar,mvar))
    _cloud_cover = np.zeros((nt,nvar,mvar))
    _cloud_top = np.zeros((nt,nvar,mvar))
    _cloud_base = np.zeros((nt,nvar,mvar))
    _updraft_area = np.zeros((nt,nz,nvar,mvar))
    _ql_mean = np.zeros((nt,nz,nvar,mvar))
    _qt_mean = np.zeros((nt, nz, nvar,mvar))
    _temperature_mean = np.zeros((nt, nz, nvar,mvar))
    _RH_mean = np.zeros((nt, nz, nvar,mvar))
    _updraft_w = np.zeros((nt,nz,nvar,mvar))
    _thetal_mean = np.zeros((nt,nz,nvar,mvar))
    _massflux = np.zeros((nt, nz, nvar,mvar))
    _buoyancy_mean = np.zeros((nt,nz,nvar,mvar))
    _env_tke = np.zeros((nt,nz,nvar,mvar))
    _updraft_thetal_precip = np.zeros((nt,nz,nvar,mvar))
    _massflux = np.zeros((nt,nz,nvar,mvar))
    _sweep_var = np.zeros((nvar,mvar))


    for i in range(0,nvar):
        sweep_var_i = sweep_varn[i]
        for j in range(0, mvar):
            sweep_var_j = sweep_varm[j]
            sweep_var_ij = [sweep_var_i,sweep_var_j]
            paramlist = sweep(sweep_var_ij)
            write_file(paramlist)
            file_case = open('paramlist_sweep.in').read()
            current = json.loads(file_case)
            print('========================')
            print('running '+case_name+' var = '+ str(sweep_var_ij))
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


            #II = III[j]
            _lwp[:, i,j] = lwp_[0:nt]
            _cloud_cover[:, i,j] = cloud_cover_[0:nt]
            _cloud_top[:, i,j] = cloud_top_[0:nt]
            _cloud_base[:, i,j] = cloud_base_[0:nt]
            _t = tt[0:nt]
            _z = zz
            _updraft_area[:,:,i,j]  = updraft_area_[0:nt,0:nz]
            _ql_mean[:,:,i,j]  = ql_mean_[0:nt,0:nz]
            _qt_mean[:, :, i,j]  = qt_mean_[0:nt, 0:nz]
            _updraft_w[:,:,i,j]  = updraft_w_[0:nt,0:nz]
            _thetal_mean[:,:,i,j]  = thetal_mean_[0:nt,0:nz]
            _temperature_mean[:, :, i,j]  = temperature_mean_[0:nt, 0:nz]
            _massflux[:, :, i,j]  = massflux_[0:nt, 0:nz]
            _buoyancy_mean[:,:,i,j]  = buoyancy_mean_[0:nt,0:nz]
            _env_tke[:,:,i,j]  = env_tke_[0:nt,0:nz]
            _updraft_thetal_precip[:,:,i,j]  = updraft_thetal_precip_[0:nt,0:nz]


            #II += 1
            #III[j] = II

                # print 'passing at ', III[j]
                # pass
    # there is a oproblem with the fact that the outcioe matrix [navr,mvar] might not be square as some values of i j got nans


            os.remove(path1)
    _sweep_var = sweep_var
    destination = '/Users/yaircohen/Documents/SCAMPy_out/parameter_sweep/'
    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + '.nc', 'w', format='NETCDF4')
    grp_stats = out_stats.createGroup('profiles')
    grp_stats.createDimension('zdim', nz)
    grp_stats.createDimension('tdim', nt)
    grp_stats.createDimension('var_n', III[0])
    grp_stats.createDimension('var_m', III[1])

    t = grp_stats.createVariable('t', 'f4', 'tdim')
    z = grp_stats.createVariable('z', 'f4', 'zdim')
    var = grp_stats.createVariable('var', 'f4', ('var_n','var_m'))
    lwp = grp_stats.createVariable('lwp', 'f4', ('tdim', 'var_n', 'var_m'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('tdim', 'var_n','var_m'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('tdim', 'var_n','var_m'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('tdim', 'var_n','var_m'))
    updraft_area = grp_stats.createVariable('updraft_area', 'f4', ('t', 'z','var_n','var_m'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    temperature_mean = grp_stats.createVariable('temperature_mean', 'f4', ('t', 'z', 'var_n','var_m'))
    RH_mean = grp_stats.createVariable('RH_mean', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    updraft_w = grp_stats.createVariable('updraft_w', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    buoyancy_mean = grp_stats.createVariable('buoyancy_mean', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    massflux = grp_stats.createVariable('massflux', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    env_tke = grp_stats.createVariable('env_tke', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    updraft_thetal_precip = grp_stats.createVariable('updraft_thetal_precip', 'f4', ('tdim', 'zdim', 'var_n','var_m'))
    #costFun = grp_stats.createVariable('costFun', 'f4', ('r','var_n','var_m'))
    print np.shape(_sweep_var)
    print np.shape(III)

    var[:,:] = _sweep_var
    t[:] = _t
    z[:] = _z
    lwp[:,:] = _lwp
    cloud_cover[:,:,:] = _cloud_cover
    cloud_top[:,:,:] = _cloud_top
    cloud_base[:,:,:] = _cloud_base
    updraft_area[:,:,:,:] = _updraft_area
    ql_mean[:,:,:,:] = _ql_mean
    qt_mean[:, :, :,:] = _qt_mean
    temperature_mean[:, :, :,:] = _temperature_mean
    RH_mean[:, :, :,:] = _RH_mean
    updraft_w[:,:,:,:] = _updraft_w
    thetal_mean[:,:,:,:] = _thetal_mean
    buoyancy_mean[:,:,:,:] = _buoyancy_mean
    massflux[:, :, :,:] = _massflux
    env_tke[:,:,:,:] = _env_tke
    updraft_thetal_precip[:, :, :,:] = _updraft_thetal_precip

    print '---------------------------------'
    print np.shape(var)
    print np.shape(_sweep_var)
    print III

    # lwp[:,:] = _lwp[:,0:III[0],0:III[1]]
    # cloud_cover[:,:] = _cloud_cover[:,0:III[0],0:III[1]]
    # cloud_top[:,:] = _cloud_top[:,0:III[0],0:III[1]]
    # cloud_base[:,:] = _cloud_base[:,0:III[0],0:III[1]]
    # updraft_area[:,:,:] = _updraft_area[:,:,0:III[0],0:III[1]]
    # ql_mean[:,:,:] = _ql_mean[:,:,0:III[0],0:III[1]]
    # qt_mean[:, :, :] = _qt_mean[:, :, 0:III[0],0:III[1]]
    # updraft_w[:,:,:] = _updraft_w[:,:,0:III[0],0:III[1]]
    # massflux[:,:,:] = _massflux[:,:,0:III[0],0:III[1]]
    # buoyancy_mean[:,:,:] = _buoyancy_mean[:,:,0:III[0],0:III[1]]
    # env_tke[:,:,:] = _env_tke[:,:,0:III[0],0:III[1]]
    # updraft_thetal_precip[:, :, :] = _updraft_thetal_precip[:,:,0:III[0],0:III[1]]

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
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.2
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.3
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 5.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha1'] = 0.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha2'] = 0.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha3'] = float(sweep_var_ij[0])
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha4'] = 0.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha5'] = 0.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha6'] = float(sweep_var_ij[1])

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
