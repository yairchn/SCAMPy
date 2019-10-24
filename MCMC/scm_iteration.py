import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
import time
from create_records import create_record, create_record_full

def scm_iter(true_data, myscampyfolder, theta,  case_name, output_filename, model_type, geom_opt=0):

    file_case = open(myscampyfolder +"/"+ case_name + '.in').read()
    namelist = json.loads(file_case)
    uuid = namelist['meta']['uuid']
    new_path = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[
                                                                                 -5:] + '/stats/Stats.' + case_name + '.nc'
    # receive parameter value and generate paramlist file for new data
    namelist['stats_io']['frequency'] = 600.0  # namelist['time_stepping']['t_max']
    paramlist = MCMC_paramlist(theta, case_name)
    write_file(paramlist)

    # call scampy and generate new data
    timeout = 60
    t0 = time.time()
    #print 'time max is now' + str(t0 + timeout)
    I=0
    while time.time() < t0 + timeout and I==0:
        print('============ start iteration with paramater = ',theta)
        subprocess.call("python main.py " + case_name + ".in " + "paramlist_" + case_name + ".in", shell=True, cwd=myscampyfolder)
        I=1
        print('============ iteration end')
    t1 = time.time()
    print('time for a scampy run = ',t1-t0)
    # load NC of the now data
    new_data = nc.Dataset(myscampyfolder+new_path[1:], 'r')
    # generate or estimate
    u = generate_costFun(theta, true_data, new_data, output_filename, model_type) # + prior knowlage -log(PDF) of value for the theta
    #record_data(theta, u, new_data, output_filename)
    os.remove(myscampyfolder + new_path[1:])

    return u

def generate_costFun(theta, true_data,new_data, output_filename, model_type):
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
    s_CF = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_cover'], 1.0)
    s_CT = np.multiply(new_data.groups['timeseries'].variables['updraft_cloud_top'], 1.0)
    s_CT[np.where(s_CT < 0.0)] = 0.0
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
        p_CT = np.multiply(true_data.groups['timeseries'].variables['cloud_top'], 1.0)
        p_CT[np.where(p_CT < 0.0)] = 0.0
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
        p_CF = np.multiply(true_data.groups['timeseries'].variables['updraft_cloud_cover'], 1.0)
        p_CT = np.multiply(true_data.groups['timeseries'].variables['updraft_cloud_top'], 1.0)
        p_CT[np.where(p_CT < 0.0)] = 0.0
        FT = np.multiply(17.625,
                         (np.divide(np.subtract(p_temperature, 273.15), (np.subtract(p_temperature, 273.15 + 243.04)))))
        p_RH = np.multiply(epsi * np.exp(FT),
                           np.divide(np.add(np.subtract(1, p_qt), epsi_inv * (p_qt - p_ql)),
                                     np.multiply(epsi_inv, np.multiply(p_p0, (p_qt - p_ql)))))
    else:
        print('model type not recognized')
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
    s_CT_temp = np.multiply(s_CT, 0.0)
    for tt in range(0, len(t_s)):
        s_CT_temp[tt] = np.interp(s_CT[tt], z_s, T_s)
    p_CT_temp = np.multiply(p_CT, 0.0)
    for tt in range(0, len(t_p)):
        p_CT_temp[tt] = np.interp(p_CT[tt], z_s, T_p)


    CAPE_theta = np.zeros(ztop)
    CAPE_T = np.zeros(ztop)
    CAPE_RH = np.zeros(ztop)
    CAPE_b = np.zeros(ztop)
    CAPE_qt = np.zeros(ztop)
    CAPE_ql = np.zeros(ztop)


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
    var_CF = np.sqrt(np.var(s_CF[-30:] - p_CF[-30:], 0))
    var_CT = np.sqrt(np.var(s_CT[-30:] - p_CT[-30:], 0))
    var_CT_temp = np.sqrt(np.var(s_CT_temp[-30:] - p_CT_temp[-30:], 0))  # (np.var(s_CT[ts1:], 0))
    var_lwp = np.sqrt(np.var(s_lwp[-30:] - p_lwp[-30:], 0))  # var_lwp = (np.var(s_lwp[ts1:], 0))

    d_CAPE_theta = np.sum(CAPE_theta)
    d_CAPE_T = np.sum(CAPE_T)
    d_CAPE_RH = np.sum(CAPE_RH)
    d_CAPE_b = np.sum(CAPE_b)
    d_CAPE_qt = np.sum(CAPE_qt)
    d_CAPE_ql = np.sum(CAPE_ql)
    dCF = np.mean(s_CF[ts1:], 0) - np.mean(p_CF[ts1:], 0)
    dCT = np.mean(s_CT[ts1:], 0) - np.mean(p_CT[ts1:], 0)
    dCT_temp = np.mean(s_CT_temp[ts1:], 0) - np.mean(p_CT_temp[ts1:], 0)
    dlwp = np.mean(s_lwp[ts1:], 0) - np.mean(p_lwp[ts1:], 0)

    rnoise = 1.0
    f = np.diag([dlwp, dCF, d_CAPE_theta])
    sigma = np.multiply(rnoise, np.diag([1 / var_lwp, 1 / var_CF, 1 / var_CT]))
    J0 = np.divide(np.linalg.norm(np.dot(sigma, f), ord=None), 2.0)  # ord=None for matrix gives the 2-norm
    p = np.zeros(len([theta]))
    mean_ = 100.0
    std_ = 40
    for i in range(0, len([theta])):
        p[i] = np.multiply(np.divide(1.0, theta[i] * np.sqrt(2 * np.pi) * std_),
                           np.exp(-(theta[i] - mean_) ** 2 / (2 * std_ ** 2)))
    u = np.multiply(J0 - np.sum(np.log(p)), 1.0)

    create_record(theta, u, new_data, output_filename)
    print('============> CostFun = ', u, '  <============')
    return u

def MCMC_paramlist(theta, case_name):

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
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 3.0*float(theta[0])
    paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor'] = 0.05
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_erf_const'] = 0.5
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0/3.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff'] = 1.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing'] = 500.0

    paramlist['turbulence']['updraft_microphysics'] = {}
    paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.1

    return paramlist


def write_file(paramlist):
    fh = open("paramlist_"+paramlist['meta']['casename']+ ".in", 'w')
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return