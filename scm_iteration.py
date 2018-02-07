import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
import pylab as plt
import time

def scm_iter(true_data, theta,  case_name, fname, model_type, geom_opt=0):

    file_case = open('/Users/yaircohen/PycharmProjects/scampy/' + case_name + '.in').read()
    namelist = json.loads(file_case)
    uuid = namelist['meta']['uuid']
    new_path = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[
                                                                                 -5:] + '/stats/Stats.' + case_name + '.nc'
    # receive parameter value and generate paramlist file for new data
    paramlist = MCMC_paramlist(theta, case_name)
    write_file(paramlist)

    # call scampy and generate new data

    timeout = 60
    t0 = time.time()
    #print 'time max is now' + str(t0 + timeout)
    I=0
    while time.time() < t0 + timeout and I==0:
        print('============ start iteration with paramater = ',theta)
        subprocess.call("python main.py " + case_name + ".in " + "paramlist_" + case_name + ".in", shell=True) # cwd = '/Users/yaircohen/PycharmProjects/scampy/',
        I=1
        print('============ iteration end')
    t1 = time.time()
    print 'time for a scampy run = ',t1-t0
    # load NC of the now data
    new_data = nc.Dataset('/Users/yaircohen/PycharmProjects/scampy'+new_path[1:], 'r')
    # generate or estimate
    u = generate_costFun(theta, true_data, new_data, fname, model_type) # + prior knowlage -log(PDF) of value for the theta
    #record_data(theta, u, new_data, fname)
    print('/Users/yaircohen/PycharmProjects/scampy' + new_path[1:])
    os.remove('/Users/yaircohen/PycharmProjects/scampy' + new_path[1:])

    return u

def generate_costFun(theta, true_data,new_data, fname, model_type):
    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5
    t0 = 0.0

    s_lwp = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
    z_s = np.multiply(new_data.groups['profiles'].variables['z'], 1.0)
    t_s = np.multiply(new_data.groups['profiles'].variables['t'], 1.0)
    ts1 = np.where(t_s[:] > t0 * 3600.0)[0][0]
    s_thetal = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)
    s_temperature = np.multiply(new_data.groups['profiles'].variables['temperature_mean'], 1.0)
    s_buoyancy = np.multiply(new_data.groups['profiles'].variables['buoyancy_mean'], 1.0)
    s_p0 = np.multiply(new_data.groups['reference'].variables['p0'], 1.0)
    s_ql = np.multiply(new_data.groups['profiles'].variables['ql_mean'], 1.0)
    s_qt = np.multiply(new_data.groups['profiles'].variables['qt_mean'], 1.0)
    s_qv = s_qt - s_ql
    s_CF = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
    FT = np.multiply(17.625,
                     (np.divide(np.subtract(s_temperature, 273.15), (np.subtract(s_temperature, 273.15 + 243.04)))))
    s_RH = np.multiply(epsi * np.exp(FT),
                       np.divide(np.add(np.subtract(1, s_qt), epsi_inv * (s_qt - s_ql)),
                                 np.multiply(epsi_inv, np.multiply(s_p0, (s_qt - s_ql)))))
    ztop = len(z_s) - 2  # LES and SCM models dont stretch mto the same  height in deep convection
    # define true data
    if model_type == 'LES':



        p_lwp = np.multiply(true_data.groups['timeseries'].variables['lwp'], 1.0)
        z_p = np.multiply(true_data.groups['profiles'].variables['z'], 1.0)
        t_p = np.multiply(true_data.groups['profiles'].variables['t'], 1.0)
        tp1 = np.where(t_p[:] > t0 * 3600.0)[0][0]
        p_thetali = np.multiply(true_data.groups['profiles'].variables['thetali_mean'], 1.0)
        p_temperature = np.multiply(true_data.groups['profiles'].variables['temperature_mean'], 1.0)
        p_buoyancy = np.multiply(true_data.groups['profiles'].variables['buoyancy_mean'], 1.0)
        p_p0 = np.multiply(true_data.groups['reference'].variables['p0'], 1.0)
        p_ql = np.multiply(true_data.groups['profiles'].variables['ql_mean'], 1.0)
        p_qt = np.multiply(true_data.groups['profiles'].variables['qt_mean'], 1.0)
        p_qv = p_qt - p_ql
        p_CF = np.multiply(true_data.groups['timeseries'].variables['cloud_fraction'], 1.0)
        FT = np.multiply(17.625,
                         (np.divide(np.subtract(p_temperature, 273.15), (np.subtract(p_temperature, 273.15 + 243.04)))))
        p_RH = np.multiply(epsi * np.exp(FT),
                           np.divide(np.add(np.subtract(1, p_qt), epsi_inv * (p_qt - p_ql)),
                                     np.multiply(epsi_inv, np.multiply(p_p0, (p_qt - p_ql)))))

    elif model_type == 'SCM':
        p_lwp = np.multiply(true_data.groups['timeseries'].variables['lwp'], 1.0)
        z_p = np.multiply(true_data.groups['profiles'].variables['z'], 1.0)
        t_p = np.multiply(true_data.groups['profiles'].variables['t'], 1.0)
        tp1 = np.where(t_p[:] > t0 * 3600.0)[0][0]
        p_thetali = np.multiply(true_data.groups['profiles'].variables['thetal_mean'], 1.0)
        p_temperature = np.multiply(true_data.groups['profiles'].variables['temperature_mean'], 1.0)
        p_buoyancy = np.multiply(true_data.groups['profiles'].variables['buoyancy_mean'], 1.0)
        p_p0 = np.multiply(true_data.groups['reference'].variables['p0'], 1.0)
        p_ql = np.multiply(true_data.groups['profiles'].variables['ql_mean'], 1.0)
        p_qt = np.multiply(true_data.groups['profiles'].variables['qt_mean'], 1.0)
        p_qv = p_qt - p_ql
        p_CF = np.multiply(true_data.groups['timeseries'].variables['cloud_cover'], 1.0)
        FT = np.multiply(17.625,
                         (np.divide(np.subtract(p_temperature, 273.15), (np.subtract(p_temperature, 273.15 + 243.04)))))
        p_RH = np.multiply(epsi * np.exp(FT),
                           np.divide(np.add(np.subtract(1, p_qt), epsi_inv * (p_qt - p_ql)),
                                     np.multiply(epsi_inv, np.multiply(p_p0, (p_qt - p_ql)))))
    else:
        print 'model type not recognized'
        exit()

    Theta_p0 = np.mean(p_thetali[tp1:, :], 0)
    Theta_p = np.interp(z_s, z_p, Theta_p0)
    T_p0 = np.mean(p_temperature[tp1:, :], 0)
    T_p = np.interp(z_s, z_p, T_p0)
    RH_p0 = np.mean(p_RH[tp1:, :], 0)
    RH_p = np.interp(z_s, z_p, RH_p0)
    qt_p0 = np.mean(p_qt[tp1:, :], 0)
    qt_p = np.interp(z_s, z_p, qt_p0)
    ql_p0 = np.mean(p_ql[tp1:, :], 0)
    ql_p = np.interp(z_s, z_p, ql_p0)
    b_p0 = np.mean(p_buoyancy[tp1:, :], 0)
    b_p = np.interp(z_s, z_p, b_p0)


    Theta_s = np.mean(s_thetal[ts1:, :], 0)
    T_s = np.mean(s_temperature[ts1:, :], 0)
    RH_s = np.mean(s_RH[ts1:, :], 0)
    qt_s = np.mean(s_qt[ts1:, :], 0)
    ql_s = np.mean(s_ql[ts1:, :], 0)
    b_s = np.mean(s_buoyancy[ts1:, :], 0)

    CAPE_theta = np.zeros(ztop)
    CAPE_T = np.zeros(ztop)
    CAPE_RH = np.zeros(ztop)
    CAPE_b = np.zeros(ztop)
    CAPE_qt = np.zeros(ztop)
    CAPE_ql = np.zeros(ztop)
    print np.shape(T_p)
    print np.shape(z_s)
    for k in range(0, ztop):
        CAPE_theta[k] = np.abs(Theta_p[k] - Theta_s[k])
        CAPE_T[k] = np.abs(T_p[k] - T_s[k])
        CAPE_RH[k] = np.abs(RH_p[k] - RH_s[k])
        CAPE_b[k] = np.abs(b_p[k] - b_s[k])
        CAPE_qt[k] = np.abs(qt_p[k] - qt_s[k])
        CAPE_ql[k] = np.abs(ql_p[k] - ql_s[k])

    var_theta = np.sqrt(np.var(CAPE_theta))
    var_T = np.sqrt(np.var(CAPE_T))
    var_RH = np.sqrt(np.var(CAPE_RH))
    var_b = np.sqrt(np.var(CAPE_b))
    var_qt = np.sqrt(np.var(CAPE_qt))
    var_ql = np.sqrt(np.var(CAPE_ql))
    var_CF = np.sqrt(np.var(p_CF[tp1:], 0))
    var_lwp = np.sqrt(np.var(p_lwp[tp1:], 0))

    d_CAPE_theta = np.sum(CAPE_theta)
    d_CAPE_T = np.sum(CAPE_T)
    d_CAPE_RH = np.sum(CAPE_RH)
    d_CAPE_b = np.sum(CAPE_b)
    d_CAPE_qt = np.sum(CAPE_qt)
    d_CAPE_ql = np.sum(CAPE_ql)
    dCF = np.mean(s_CF[ts1:], 0) - np.mean(p_CF[ts1:], 0)
    dlwp = np.mean(s_lwp[ts1:], 0) - np.mean(p_lwp[ts1:], 0)

    rnoise = 1.0
    f = np.diag([dlwp, dCF, d_CAPE_qt, d_CAPE_ql])
    sigma = np.multiply(rnoise, np.diag(
        [1 / np.max([var_lwp, 0.001]), 1 / np.max([var_CF, 0.001]), 1 / np.max([var_qt, 0.001]),
         1 / np.max([var_ql, 0.001])]))
    J0 = np.divide(np.linalg.norm(np.dot(sigma, f), ord=None), 2.0)  # ord=None for matrix gives the 2-norm
    logp = 0.0
    u = np.multiply(J0 - logp, 1.0)
    create_record(theta, u, new_data, fname)
    print('============> CostFun = ', u, '  <============')
    #plt.ion()
    #plt.plot(Theta_p, z_p, 'b', linewidth = 3)
    #plt.plot(Theta_s, z_s, 'r')
    #plt.pause(0.05)
    return u

def MCMC_paramlist(theta, case_name): # vel_pressure_coeff_i,

    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = case_name

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.0

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 2.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['vel_buoy_coeff'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['vel_pressure_coeff'] = float(theta)

    paramlist['turbulence']['EDMF_BulkSteady'] = {}
    paramlist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.18
    paramlist['turbulence']['EDMF_BulkSteady']['w_entr_coeff'] = 2.0
    paramlist['turbulence']['EDMF_BulkSteady']['w_buoy_coeff'] = 1.0
    paramlist['turbulence']['EDMF_BulkSteady']['max_area_factor'] = 1.0
    paramlist['turbulence']['EDMF_BulkSteady']['entrainment_factor'] = 1.0
    paramlist['turbulence']['EDMF_BulkSteady']['detrainment_factor'] = 1.0

    paramlist['turbulence']['updraft_microphysics'] = {}
    paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.01

    return paramlist


def write_file(paramlist):
    fh = open("paramlist_"+paramlist['meta']['casename']+ ".in", 'w')
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return


def create_record(theta_, costFun_, new_data, fname):
    t0 = time.time()
    # load existing data to variables
    lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
    cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
    cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['cloud_top'], 1.0)
    cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['cloud_base'], 1.0)
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
        # cloud_cover = tuning_record.groups['data'].variables['cloud_cover']
        # cloud_cover = cloud_cover_
        # cloud_top = tuning_record.groups['data'].variables['cloud_top']
        # cloud_top= cloud_top_
        # cloud_base = tuning_record.groups['data'].variables['cloud_base']
        # cloud_base = cloud_base_
        # thetal_mean = tuning_record.groups['data'].variables['thetal_mean']
        # thetal_mean = thetal_mean_
        # temperature_mean = tuning_record.groups['data'].variables['temperature_mean']
        # temperature_mean = temperature_mean_
        # qt_mean = tuning_record.groups['data'].variables['qt_mean']
        # qt_mean = qt_mean_
        # ql_mean = tuning_record.groups['data'].variables['ql_mean']
        # ql_mean = ql_mean_
        tune_param = tuning_record.groups['data'].variables['tune_param']
        tune_param = theta_
        costFun = tuning_record.groups['data'].variables['costFun']
        costFun = costFun_
        nsim = tuning_record.groups['data'].variables['nsim']
        nsim[0] = np.add(nnsim,1.0)
        tuning_record.close()

    else:
        nsim_ = np.multiply(tuning_record.groups['data'].variables['nsim'],1.0)[0]
        nsim = tuning_record.groups['data'].variables['nsim']
        # lwp = tuning_record.groups['data'].variables['lwp']
        # lwp[:, nsim_] = lwp_
        # cloud_cover = tuning_record.groups['data'].variables['cloud_cover']
        # cloud_cover[:, nsim_] = cloud_cover_
        # cloud_top = tuning_record.groups['data'].variables['cloud_top']
        # cloud_top[:, nsim_] = cloud_top_
        # cloud_base = tuning_record.groups['data'].variables['cloud_base']
        # cloud_base[:, nsim_] = cloud_base_
        # thetal_mean = tuning_record.groups['data'].variables['thetal_mean']
        # print np.shape(thetal_mean_)
        # print np.shape(thetal_mean)
        # thetal_mean[:, :,nsim_] = thetal_mean_
        # temperature_mean = tuning_record.groups['data'].variables['temperature_mean']
        # temperature_mean[:, :,nsim_] = temperature_mean_
        # qt_mean = tuning_record.groups['data'].variables['qt_mean']
        # qt_mean[:, :,nsim_] = qt_mean_
        # ql_mean = tuning_record.groups['data'].variables['ql_mean']
        # ql_mean[:, :,nsim_] = ql_mean_

        tune_param = tuning_record.groups['data'].variables['tune_param']
        tune_param[nsim_] = theta_
        costFun = tuning_record.groups['data'].variables['costFun']
        costFun[nsim_] = costFun_
        #nsim_ = tuning_record.groups['data'].variables['nsim']
        nsim_ = np.add(nsim_, 1.0)
        nsim[0] = nsim_
        tuning_record.close()
    t1 = time.time()
    print 'time for create record = ', t1 - t0


    return

# def record_data(theta_, u, new_data, fname):
#
#     # get nbew data
#     print new_data
#     lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
#     cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
#     cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['cloud_top'], 1.0)
#     cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['cloud_base'], 1.0)
#     thetal_mean_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)
#     temperature_mean_ = np.multiply(new_data.groups['profiles'].variables['temperature_mean'], 1.0)
#     qt_mean_ = np.multiply(new_data.groups['profiles'].variables['qt_mean'], 1.0)
#     ql_mean_ = np.multiply(new_data.groups['profiles'].variables['ql_mean'], 1.0)
#
#     # append to tuning_record
#     data  = nc.Dataset(fname,'a')
#     nsim1 = np.multiply(data.groups['data'].variables['nsim'],1)+1
#     appendvar = data.variables['lwp']
#     appendvar[:,nsim1] = lwp_
#     appendvar = data.variables['cloud_cover']
#     appendvar[:,nsim1] = cloud_cover_
#     appendvar = data.variables['cloud_top']
#     appendvar[:,nsim1] = cloud_top_
#     appendvar = data.variables['cloud_base']
#     appendvar[:,nsim1] = cloud_base_
#     appendvar = data.variables['thetal_mean']
#     appendvar[:,:,nsim1] = thetal_mean_
#     appendvar = data.variables['temperature_mean']
#     appendvar[:,:,nsim1] = temperature_mean_
#     appendvar = data.variables['qt_mean']
#     appendvar[:,:,nsim1] = qt_mean_
#     appendvar = data.variables['ql_mean']
#     appendvar[:,:,nsim1] = ql_mean_
#
#     appendvar = data.variables['tune_param']
#     appendvar[nsim1] = theta_
#     appendvar = data.variables['costFun']
#     appendvar[nsim1-1] = u
#
#     data.close()
#
#     return