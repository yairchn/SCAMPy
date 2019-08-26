import subprocess
import argparse
from shutil import copyfile
import json
import pprint as pp
from sys import exit
import uuid
import ast
import numpy as np
import netCDF4 as nc
import os

# python parameter_sweep.py case_name
def main():
    parser = argparse.ArgumentParser(prog='Paramlist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()
    case_name = args.case_name

    localpath = os.getcwd()
    myscampyfolder = localpath[0:-5]
    # generate namelist and paramlist files
    subprocess.call("python generate_paramlist.py " + case_name,  shell=True, cwd=myscampyfolder)
    subprocess.call("python generate_namelist.py " + case_name,  shell=True, cwd=myscampyfolder)

    # produce sweep specific files
    copyfile(myscampyfolder+'/paramlist_'+case_name+'.in',myscampyfolder+'/paramlist_sweep'+case_name+'.in')
    case_file = open(myscampyfolder+"/"+case_name+'.in','r')
    namelist = json.load(case_file)
    uuid0 = namelist['meta']['uuid']
    uuid = uuid0[0:-5]+'sweep'
    # changes to namelist file, uuid, dt ,  tmax etc.
    namelist['meta']['uuid'] = uuid
    case_file.close()

    newnamelistfile = open(myscampyfolder+"/"+case_name+ '_sweep.in', 'w')
    json.dump(namelist, newnamelistfile, sort_keys=True, indent=4)
    newnamelistfile.close()

    stats_path = myscampyfolder + '/Output.' + case_name + '.' + uuid[-5:] + '/stats/Stats.' + case_name + '.nc'
    paramlist_path1 = myscampyfolder + '/Output.' + case_name + '.' + uuid[-5:] + '/paramlist_' + case_name + '.in'
    tmax = namelist['time_stepping']['t_max']
    #dt   = namelist['time_stepping']['dt']
    freq = namelist['stats_io']['frequency']
    nz   = namelist['grid']['nz']
    nt = int(tmax/freq)+1
    II=0
    nvar = 5
    sweep_var = np.linspace(0.1, 2.0, num=nvar)

    _z = np.zeros((nz))
    _t = np.zeros((nt))
    _lwp = np.zeros((nt,nvar))
    _updraft_cloud_cover = np.zeros((nt,nvar))
    _updraft_cloud_top = np.zeros((nt,nvar))
    _updraft_cloud_base = np.zeros((nt,nvar))
    _updraft_area = np.zeros((nt,nz,nvar))
    _ql_mean = np.zeros((nt,nz,nvar))
    _updraft_w = np.zeros((nt,nz,nvar))
    _thetal_mean = np.zeros((nt,nz,nvar))
    _massflux = np.zeros((nt, nz, nvar))
    _updraft_buoyancy = np.zeros((nt,nz,nvar))
    _tke_mean = np.zeros((nt,nz,nvar))
    _updraft_thetal_precip = np.zeros((nt,nz,nvar))
    _sweep_var = np.zeros(nvar)

    for i in range(0,nvar):
        sweep_var_i = sweep_var[i]
        paramlist = update_paramlist(sweep_var_i)
        write_file(paramlist)
        namelistfile = open(myscampyfolder+'/paramlist_sweep'+case_name+'.in').read()
        current = json.loads(namelistfile)

        print('========================')
        print('running '+case_name+' with tuning variable = '+ str(sweep_var_i))
        print("python main.py " + case_name + "_sweep.in paramlist_sweep"+case_name+".in")
        print('========================')
        subprocess.call("python main.py " + case_name + "_sweep.in paramlist_sweep"+case_name+".in", shell=True, cwd=myscampyfolder)

        data = nc.Dataset(stats_path, 'r')
        zz = data.groups['profiles'].variables['z']
        tt = data.groups['profiles'].variables['t']

        lwp_ = np.multiply(data.groups['timeseries'].variables['lwp'], 1.0)
        updraft_cloud_cover_ = np.multiply(data.groups['timeseries'].variables['updraft_cloud_cover'],1.0)
        updraft_cloud_top_ = np.multiply(data.groups['timeseries'].variables['updraft_cloud_top'],1.0)
        updraft_cloud_base_ = np.multiply(data.groups['timeseries'].variables['updraft_cloud_base'],1.0)

        updraft_area_ = np.multiply(data.groups['profiles'].variables['updraft_area'],1.0)
        ql_mean_ = np.multiply(data.groups['profiles'].variables['ql_mean'],1.0)
        updraft_w_ = np.multiply(data.groups['profiles'].variables['updraft_w'],1.0)
        thetal_mean_ = np.multiply(data.groups['profiles'].variables['thetal_mean'],1.0)
        massflux_ = np.multiply(data.groups['profiles'].variables['massflux'], 1.0)
        updraft_buoyancy_ = np.multiply(data.groups['profiles'].variables['updraft_buoyancy'],1.0)
        tke_mean_ = np.multiply(data.groups['profiles'].variables['tke_mean'],1.0)
        updraft_thetal_precip_ = np.multiply(data.groups['profiles'].variables['updraft_thetal_precip'], 1.0)
        print(np.shape(updraft_buoyancy_))
        try:
            _lwp[:, II] = lwp_[0:nt]
            _updraft_cloud_cover[:,II] = updraft_cloud_cover_[0:nt]
            _updraft_cloud_top[:,II] = updraft_cloud_top_[0:nt]
            _updraft_cloud_base[:,II] = updraft_cloud_base_[0:nt]
            _t = tt[0:nt]
            _z = zz
            _updraft_area[:,:,II] = updraft_area_[0:nt,0:nz]
            _ql_mean[:,:,II] = ql_mean_[0:nt,0:nz]
            _updraft_w[:,:,II] = updraft_w_[0:nt,0:nz]
            _thetal_mean[:,:,II] = thetal_mean_[0:nt,0:nz]
            _massflux[:, :, II] = massflux_[0:nt, 0:nz]
            _updraft_buoyancy[:,:,II] = updraft_buoyancy_[0:nt,0:nz]
            _tke_mean[:,:,II] = tke_mean_[0:nt,0:nz]
            _updraft_thetal_precip[:,:,II] = updraft_thetal_precip_[0:nt,0:nz]
            _sweep_var[II] = sweep_var_i
            II += 1
        except:
            print('passing')
            pass



        os.remove(stats_path)
        os.remove(paramlist_path1)

    destination = './Output.Parameter_Sweep.' + case_name+"/"
    try:
        os.mkdir(destination)
    except:
        os.rmdir(destination)
        os.mkdir(destination)
        print('directory exists, might be overwriting!')
    out_stats = nc.Dataset(destination + '/Stats.sweep_' + case_name + '.nc', 'w', format='NETCDF4')
    grp_stats = out_stats.createGroup('profiles')
    grp_stats.createDimension('z', nz)
    grp_stats.createDimension('t', nt)
    grp_stats.createDimension('var', II)

    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    var = grp_stats.createVariable('var', 'f4', 'var')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'var'))
    updraft_cloud_cover = grp_stats.createVariable('updraft_cloud_cover', 'f4', ('t', 'var'))
    updraft_cloud_top = grp_stats.createVariable('updraft_cloud_top', 'f4', ('t', 'var'))
    updraft_cloud_base = grp_stats.createVariable('updraft_cloud_base', 'f4', ('t', 'var'))
    updraft_area = grp_stats.createVariable('updraft_area', 'f4', ('t', 'z','var'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('t', 'z', 'var'))
    updraft_w = grp_stats.createVariable('updraft_w', 'f4', ('t', 'z', 'var'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4', ('t', 'z', 'var'))
    massflux = grp_stats.createVariable('massflux', 'f4', ('t', 'z', 'var'))
    updraft_buoyancy = grp_stats.createVariable('updraft_buoyancy', 'f4', ('t', 'z', 'var'))
    tke_mean = grp_stats.createVariable('tke_mean', 'f4', ('t', 'z', 'var'))
    updraft_thetal_precip = grp_stats.createVariable('updraft_thetal_precip', 'f4', ('t', 'z', 'var'))
    print('---------------------------------')
    print(np.shape(var))
    print(np.shape(_sweep_var))
    print(II)
    print('---------------------------------')
    var[:] = _sweep_var[0:II]
    print(np.shape(_t))
    print(np.shape(t))
    #t[:] = _t
    #z[:] = _z
    print('---------------------------------')
    print(np.shape(lwp))
    print(np.shape(_lwp))
    print(II)
    print('---------------------------------')

    lwp[:,:] = _lwp[:,0:II]
    updraft_cloud_cover[:,:] = _updraft_cloud_cover[:,0:II]
    updraft_cloud_top[:,:] = _updraft_cloud_top[:,0:II]
    updraft_cloud_base[:,:] = _updraft_cloud_base[:,0:II]
    updraft_area[:,:,:] = _updraft_area[:,:,0:II]
    ql_mean[:,:,:] = _ql_mean[:,:,0:II]
    updraft_w[:,:,:] = _updraft_w[:,:,0:II]
    massflux[:,:,:] = _massflux[:,:,0:II]
    updraft_buoyancy[:,:,:] = _updraft_buoyancy[:,:,0:II]
    tke_mean[:,:,:] = _tke_mean[:,:,0:II]
    updraft_thetal_precip[:, :, :] = _updraft_thetal_precip[:,:,0:II]

    out_stats.close()
    print('========================')
    print('======= SWEEP END ======')
    print('========================')





def update_paramlist(sweep_var_i): # vel_pressure_coeff_i

    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = 'Bomex'

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.2

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.16
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.35
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 9.9
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.03
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 3.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor'] = 0.05*sweep_var_i
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_erf_const'] = 0.5
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0/3.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing'] = 500.0

    paramlist['turbulence']['updraft_microphysics'] = {}
    paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.1

    return  paramlist

def write_file(paramlist):

    fh = open('paramlist_'+paramlist['meta']['casename']+ '.in', 'w')
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return

if __name__ == '__main__':
    main()
