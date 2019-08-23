import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
import time

def create_record(theta_, costFun_, new_data, fname):
    t0 = time.time()
    # load existing data to variables
    lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
    cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_cover'], 1.0)
    cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_top'], 1.0)
    cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_base'], 1.0)
    thetal_mean_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)
    temperature_mean_ = np.multiply(new_data.groups['profiles'].variables['temperature_mean'], 1.0)
    qt_mean_ = np.multiply(new_data.groups['profiles'].variables['qt_mean'], 1.0)
    ql_mean_ = np.multiply(new_data.groups['profiles'].variables['ql_mean'], 1.0)

    # load old data and close netCDF
    tuning_record = nc.Dataset(fname, 'a')

    nnsim = np.multiply(tuning_record.groups['data'].variables['nsim'],1.0)[0]

    if nnsim==0.0:
        tune_param = tuning_record.groups['data'].variables['tune_param']
        print(np.shape(tune_param))
        tune_param = theta_
        costFun = tuning_record.groups['data'].variables['costFun']
        costFun = costFun_
        nsim = tuning_record.groups['data'].variables['nsim']
        nsim[0] = np.add(nnsim,1.0)
        tuning_record.close()

    else:
        nsim_ = np.multiply(tuning_record.groups['data'].variables['nsim'],1.0)[0]
        nsim = tuning_record.groups['data'].variables['nsim']
        m=len(theta_)
        tune_param = tuning_record.groups['data'].variables['tune_param']
        tune_param[nsim_,:] = theta_
        costFun = tuning_record.groups['data'].variables['costFun']
        costFun[nsim_] = costFun_
        #nsim_ = tuning_record.groups['data'].variables['nsim']
        nsim_ = np.add(nsim_, 1.0)
        nsim[0] = nsim_
        tuning_record.close()
    t1 = time.time()
    print('time for create record = ', t1 - t0)


    return


def create_record_full(theta_, costFun_, new_data, fname):
    t0 = time.time()
    # load existing data to variables
    lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
    cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_cover'], 1.0)
    cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_top'], 1.0)
    cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_base'], 1.0)
    thetal_mean_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)
    temperature_mean_ = np.multiply(new_data.groups['profiles'].variables['temperature_mean'], 1.0)
    qt_mean_ = np.multiply(new_data.groups['profiles'].variables['qt_mean'], 1.0)
    ql_mean_ = np.multiply(new_data.groups['profiles'].variables['ql_mean'], 1.0)

    # load old data and close netCDF
    tuning_record = nc.Dataset(fname, 'a')

    nnsim = np.multiply(tuning_record.groups['data'].variables['nsim'],1.0)[0]

    if nnsim==0.0:

        lwp = tuning_record.groups['data'].variables['lwp']
        lwp = lwp_
        cloud_cover = tuning_record.groups['data'].variables['updraft_cloud_cover']
        cloud_cover = cloud_cover_
        cloud_top = tuning_record.groups['data'].variables['updraft_cloud_top']
        cloud_top= cloud_top_
        cloud_base = tuning_record.groups['data'].variables['updraft_cloud_base']
        cloud_base = cloud_base_
        thetal_mean = tuning_record.groups['data'].variables['thetal_mean']
        thetal_mean = thetal_mean_
        temperature_mean = tuning_record.groups['data'].variables['temperature_mean']
        temperature_mean = temperature_mean_
        qt_mean = tuning_record.groups['data'].variables['qt_mean']
        qt_mean = qt_mean_
        ql_mean = tuning_record.groups['data'].variables['ql_mean']
        ql_mean = ql_mean_
        tune_param = tuning_record.groups['data'].variables['tune_param']
        print(np.shape(tune_param))
        tune_param = theta_
        costFun = tuning_record.groups['data'].variables['costFun']
        costFun = costFun_
        nsim = tuning_record.groups['data'].variables['nsim']
        nsim[0] = np.add(nnsim,1.0)
        tuning_record.close()

    else:
        nsim_ = np.multiply(tuning_record.groups['data'].variables['nsim'],1.0)[0]
        nsim = tuning_record.groups['data'].variables['nsim']
        lwp = tuning_record.groups['data'].variables['lwp']
        lwp[:, nsim_] = lwp_
        cloud_cover = tuning_record.groups['data'].variables['updraft_cloud_cover']
        cloud_cover[:, nsim_] = cloud_cover_
        cloud_top = tuning_record.groups['data'].variables['updraft_cloud_top']
        cloud_top[:, nsim_] = cloud_top_
        cloud_base = tuning_record.groups['data'].variables['updraft_cloud_base']
        cloud_base[:, nsim_] = cloud_base_
        thetal_mean = tuning_record.groups['data'].variables['thetal_mean']
        print(np.shape(thetal_mean_))
        print(np.shape(thetal_mean))
        thetal_mean[:, :,nsim_] = thetal_mean_
        temperature_mean = tuning_record.groups['data'].variables['temperature_mean']
        temperature_mean[:, :,nsim_] = temperature_mean_
        qt_mean = tuning_record.groups['data'].variables['qt_mean']
        qt_mean[:, :,nsim_] = qt_mean_
        ql_mean = tuning_record.groups['data'].variables['ql_mean']
        ql_mean[:, :,nsim_] = ql_mean_
        m=len(theta_)
        tune_param = tuning_record.groups['data'].variables['tune_param']
        tune_param[nsim_,:] = theta_
        costFun = tuning_record.groups['data'].variables['costFun']
        costFun[nsim_] = costFun_
        #nsim_ = tuning_record.groups['data'].variables['nsim']
        nsim_ = np.add(nsim_, 1.0)
        nsim[0] = nsim_
        tuning_record.close()
    t1 = time.time()
    print('time for create record = ', t1 - t0)


    return

def record_data(theta_, u, new_data, fname):

    # get nbew data
    print(new_data)
    lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
    cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_cover'], 1.0)
    cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_top'], 1.0)
    cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_base'], 1.0)
    thetal_mean_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)
    temperature_mean_ = np.multiply(new_data.groups['profiles'].variables['temperature_mean'], 1.0)
    qt_mean_ = np.multiply(new_data.groups['profiles'].variables['qt_mean'], 1.0)
    ql_mean_ = np.multiply(new_data.groups['profiles'].variables['ql_mean'], 1.0)

    # append to tuning_record
    data  = nc.Dataset(fname,'a')
    nsim1 = np.multiply(data.groups['data'].variables['nsim'],1)+1
    appendvar = data.variables['lwp']
    appendvar[:,nsim1] = lwp_
    appendvar = data.variables['updraft_cloud_cover']
    appendvar[:,nsim1] = cloud_cover_
    appendvar = data.variables['updraft_cloud_top']
    appendvar[:,nsim1] = cloud_top_
    appendvar = data.variables['updraft_cloud_base']
    appendvar[:,nsim1] = cloud_base_
    appendvar = data.variables['thetal_mean']
    appendvar[:,:,nsim1] = thetal_mean_
    appendvar = data.variables['temperature_mean']
    appendvar[:,:,nsim1] = temperature_mean_
    appendvar = data.variables['qt_mean']
    appendvar[:,:,nsim1] = qt_mean_
    appendvar = data.variables['ql_mean']
    appendvar[:,:,nsim1] = ql_mean_

    appendvar = data.variables['tune_param']
    appendvar[nsim1] = theta_
    appendvar = data.variables['costFun']
    appendvar[nsim1-1] = u

    data.close()

    return

def initiate_record(output_filename, theta):
    print('ir 177')
    m = len(theta)
    print('ir 179')
    tuning_record = nc.Dataset(output_filename, "w", format="NETCDF4")
    grp_stats = tuning_record.createGroup('data')
    print('ir 182')
    grp_stats.createDimension('z', 75)  # get this from namelistfile
    grp_stats.createDimension('t', 182)  # get this from namelistfile
    grp_stats.createDimension('dim', None)
    grp_stats.createDimension('ntheta', m)
    grp_stats.createDimension('sim', 1)
    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
    thetal_mean = grp_stats.createVariable('thetal_mean', 'f4', ('t', 'z', 'dim'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('t', 'z', 'dim'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('t', 'z', 'dim'))
    temperature_mean = grp_stats.createVariable('temperature_mean', 'f4', ('t', 'z', 'dim'))
    tune_param = grp_stats.createVariable('tune_param', 'f4', ('dim', 'ntheta'))
    costFun = grp_stats.createVariable('costFun', 'f4', 'dim')  # this might be a problem if dim=1 implies 2 value
    nsim = grp_stats.createVariable('nsim', 'f4', 'dim')
    nsim = tuning_record.groups['data'].variables['nsim']
    nsim[:] = 0
    print('ir 203')

    tuning_record.close()
    return
