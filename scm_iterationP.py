import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
import pylab as plt
from shutil import copyfile

def scm_iterP(ncore, true_data, theta,  case_name, geom_opt=0):

    src = '/cluster/home/yairc/scampy/' + case_name + '.in'
    dst = '/cluster/home/yairc/scampy/' + case_name + str(ncore) + '.in'
    #copyfile(src, dst)
    namelistfile = open(src,'r')
    namelist = json.load(namelistfile)
    #path0 = namelist['meta']['uuid']
    uuid0 = namelist['meta']['uuid']
    uuid = uuid0[0:-5]+'tune'+ ncore
    namelist['meta']['uuid'] = uuid
    new_dir = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[-5:] + '/stats/'
    new_path = new_dir + 'Stats.' + case_name + '.nc'
    case_name0 = namelist['meta']['casename']
    namelist['meta']['casename'] = case_name0+ncore
    newnamelistfile = open(dst, 'w')
    json.dump(namelist, newnamelistfile, sort_keys=True, indent=4)
    namelistfile.close()

    # receive parameter value and generate paramlist file for new data
    paramlist = MCMC_paramlist(theta, case_name+ncore)
    write_file(paramlist)

    # call scampy and generate new
    # here i need to call paramlist with aserial number that changes for each cluster
    print('============ start iteration with paramater = ',theta) # + str(ncore)
    runstring = 'python main.py ' + case_name + ncore + '.in ' + 'paramlist_' + case_name + '.in'
    print(runstring)
    subprocess.call(runstring, shell=True)  # cwd = '/Users/yaircohen/PycharmProjects/scampy/',
    print('============ iteration end')

    # load NC of the now data
    print(new_path)
    new_data = nc.Dataset(new_path, 'r')
    # generate or estimate
    costFun = generate_costFun(theta, true_data, new_data, new_dir) # + prior knowledge -log(PDF) of value for the theta

    os.remove(new_path)

    return costFun

def generate_costFun(theta, true_data,new_data, new_dir):

    s_lwp = new_data.groups['timeseries'].variables['lwp']
    p_lwp = true_data.groups['timeseries'].variables['lwp']
    z_s = new_data.groups['profiles'].variables['z']
    t_s = new_data.groups['profiles'].variables['t']
    z_p = true_data.groups['profiles'].variables['z']
    t_p = true_data.groups['profiles'].variables['t']
    tp1 = np.where(t_p[:] > 5.0 * 3600.0)[0][0]
    ts1 = np.where(t_s[:] > 5.0 * 3600.0)[0][0]
    s_thetal = new_data.groups['profiles'].variables['thetal_mean']
    p_thetali = true_data.groups['profiles'].variables['thetal_mean']
    s_CF = new_data.groups['timeseries'].variables['cloud_cover']
    p_CF = true_data.groups['timeseries'].variables['cloud_cover']

    Theta_p = np.mean(p_thetali[tp1:,:],0)
    Theta_s = np.mean(s_thetal[ts1:, :],0)
    CAPE = np.multiply(Theta_s,0.0)
    for k in range(0,len(z_p)):
        CAPE[k] = np.abs(Theta_p[k] - Theta_s[k])
    d_CAPE = np.sum(CAPE)
    #p_CAPE = np.trapz(Theta_p,z_p)
    #s_CAPE = np.trapz(Theta_s,z_s)

    #dCAPE = s_CAPE - p_CAPE
    dlwp = np.mean(s_lwp[ts1:], 0)- np.mean(p_lwp[tp1:], 0)
    dCF = np.mean(s_CF[ts1:], 0) - np.mean(p_CF[tp1:], 0)

    # yair - I stopped here
    u = np.sqrt(d_CAPE**2 + dlwp**2 + dCF**2)

    # call record
    create_record(theta, u, new_data, new_dir)
    # store data
    print('============> CostFun = ', u, '  <============')
    plt.ion()
    plt.plot(Theta_p, z_p, 'b', linewidth = 3)
    plt.plot(Theta_s, z_s, 'r')
    plt.pause(0.05)
    return u

def MCMC_paramlist(theta, case_name): # vel_pressure_coeff_i,

    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = case_name

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.0

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] =  0.18
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_scalar_coeff'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.3
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 20.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = float(theta)
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = float(theta)
    paramlist['turbulence']['EDMF_PrognosticTKE']['vel_pressure_coeff'] = 5e-05
    paramlist['turbulence']['EDMF_PrognosticTKE']['vel_buoy_coeff'] = 0.6666666666666666

    paramlist['turbulence']['EDMF_BulkSteady'] = {}
    paramlist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.05
    paramlist['turbulence']['EDMF_BulkSteady']['w_entr_coeff'] = 2.0  #"w_b"
    paramlist['turbulence']['EDMF_BulkSteady']['w_buoy_coeff'] = 1.0
    paramlist['turbulence']['EDMF_BulkSteady']['max_area_factor'] = 5.0
    paramlist['turbulence']['EDMF_BulkSteady']['entrainment_factor'] = 0.5
    paramlist['turbulence']['EDMF_BulkSteady']['detrainment_factor'] = 0.5

    paramlist['turbulence']['updraft_microphysics'] = {}
    paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.1
    return paramlist

def write_file(paramlist):
    fh = open("paramlist_"+paramlist['meta']['casename']+ ".in", 'w')
    print(type(paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']))
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return


def create_record(theta_, costFun_, new_data, new_dir):

    z_s = np.multiply(new_data.groups['profiles'].variables['z'],1.0)
    t_s = np.multiply(new_data.groups['profiles'].variables['t'],1.0)
    s_thetal = np.multiply(new_data.groups['profiles'].variables['thetal_mean'],1.0)

    nt = np.shape(t_s)[0]
    nz = np.shape(z_s)[0]


    fname = new_dir + 'tuning_record.nc'

    if os.path.isfile(fname):

        # load existing record
        old_record = nc.Dataset(fname, 'r')
        lwp1_ = np.multiply(old_record.groups['data'].variables['lwp'], 1.0)
        cloud_cover1_ = np.multiply(old_record.groups['data'].variables['cloud_cover'], 1.0)
        cloud_top1_ = np.multiply(old_record.groups['data'].variables['cloud_top'], 1.0)
        cloud_base1_ = np.multiply(old_record.groups['data'].variables['cloud_base'], 1.0)
        thetal1_ = np.multiply(old_record.groups['data'].variables['thetal_mean'], 1.0)
        theta1_ = np.multiply(old_record.groups['data'].variables['thetal'], 1.0)
        costFun1_ = np.multiply(old_record.groups['data'].variables['costFun'], 1.0)

        # find the length of the third dim of thetal for the number of the tuned simulation
        if  thetal1_.ndim < 3:
            dim = 1
        else:
            dim =len( thetal1_[0,0,:])

        # build a new record that will overwrite the old one
        tuning_recored = nc.Dataset(fname, 'w', format='NETCDF4')
        grp_stats = tuning_recored.createGroup('data')
        grp_stats.createDimension('z', nz)
        grp_stats.createDimension('t', nt)
        grp_stats.createDimension('dim', dim + 1)

        # create variables
        _lwp = np.zeros((nt, dim + 1))
        _cloud_cover = np.zeros((nt, dim + 1))
        _cloud_top = np.zeros((nt, dim + 1))
        _cloud_base = np.zeros((nt, dim + 1))
        _thetal = np.zeros((nt, nz, dim + 1))
        _theta = np.zeros((dim + 1))
        _costFun = np.zeros((dim + 1))

        # load old data
        lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
        cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
        cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['cloud_top'], 1.0)
        cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['cloud_base'], 1.0)
        thetal_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)

        # store old data in first part of new variables
        _t = np.multiply(t_s, 1.0)
        _z = np.multiply(z_s, 1.0)
        _lwp[:, 0:dim] = lwp1_
        _cloud_cover[:, 0:dim] = cloud_cover1_
        _cloud_top[:, 0:dim] = cloud_top1_
        _cloud_base[:, 0:dim] = cloud_base1_
        _thetal[:, 0:dim] = thetal1_
        _theta[0:dim] = theta1_
        _costFun[0:dim] = costFun1_

        # add new data to variables
        _lwp[:, dim + 1] = lwp_
        _cloud_cover[:, dim + 1] = cloud_cover_
        _cloud_top[:, dim + 1] = cloud_top_
        _cloud_base[:, dim + 1] = cloud_base_
        _thetal[:, dim + 1] = thetal_
        _theta[dim + 1] = theta_
        _costFun[dim + 1] = costFun_

        t = grp_stats.createVariable('t', 'f4', 't')
        z = grp_stats.createVariable('z', 'f4', 'z')
        lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
        cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
        cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
        cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
        thetal = grp_stats.createVariable('updraft_area', 'f4', ('t', ('t', 'z', 'dim')))
        theta = grp_stats.createVariable('updraft_area', 'f4', ('t', 'dim'))
        costFun = grp_stats.createVariable('updraft_area', 'f4', ('t', 'dim'))

        t[:] = _t
        z[:] = _z
        lwp[:, :] = _lwp
        cloud_cover[:, :] = _cloud_cover
        cloud_top[:, :] = _cloud_top
        cloud_base[:, :] = _cloud_base
        thetal[:, :, :] = _thetal
        theta[:] = _theta
        costFun[:] = _costFun

        tuning_recored.close()

    else:

        tuning_recored = nc.Dataset(fname, 'w', format='NETCDF4')
        grp_stats = tuning_recored.createGroup('profiles')
        grp_stats.createDimension('z', nz)
        grp_stats.createDimension('t', nt)
        grp_stats.createDimension('dim', 1)

        lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
        cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
        cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['cloud_top'], 1.0)
        cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['cloud_base'], 1.0)
        thetal_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)

        t = grp_stats.createVariable('t', 'f4', 't')
        z = grp_stats.createVariable('z', 'f4', 'z')
        lwp = grp_stats.createVariable('lwp', 'f4', 't')
        cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', 't')
        cloud_top = grp_stats.createVariable('cloud_top', 'f4', 't')
        cloud_base = grp_stats.createVariable('cloud_base', 'f4', 't')
        thetal = grp_stats.createVariable('updraft_area', 'f4', ('t', 'z'))
        theta = grp_stats.createVariable('updraft_area', 'f4', 'dim')
        costFun = grp_stats.createVariable('updraft_area', 'f4', 'dim') # this might be a problem if dim=1 implies 2 value

        _t = np.multiply(t_s, 1.0)
        _z = np.multiply(z_s, 1.0)
        _lwp = lwp_
        _cloud_cover = cloud_cover_
        _cloud_top = cloud_top_
        _cloud_base = cloud_base_
        _thetal = thetal_
        _theta = theta_
        _costFun = costFun_

        t[:] = _t
        z[:] = _z
        lwp[:] = _lwp
        cloud_cover[:] = _cloud_cover
        cloud_top[:] = _cloud_top
        cloud_base[:] = _cloud_base
        thetal[:, :] = _thetal
        theta[:] = _theta
        costFun[:] = _costFun

        tuning_recored.close()


    return
