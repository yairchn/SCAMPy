import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
import pylab as plt

def scm_iter(true_data, theta,  case_name, geom_opt=0):

    file_case = open('/Users/yaircohen/PycharmProjects/scampy/' + case_name + '.in').read()
    namelist = json.loads(file_case)
    uuid = namelist['meta']['uuid']
    new_path = namelist['output']['output_root'] + 'Output.' + case_name + '.' + uuid[
                                                                                 -5:] + '/stats/Stats.' + case_name + '.nc'
    # receive parameter value and generate paramlist file for new data
    paramlist = MCMC_paramlist(theta, case_name)
    write_file(paramlist)

    # call scampy and generate new data
    print('============ start iteration with paramater = ',theta)
    subprocess.call("python main.py " + case_name + ".in " + "paramlist_" + case_name + ".in", shell=True) # cwd = '/Users/yaircohen/PycharmProjects/scampy/',
    print('============ iteration end')

    # load NC of the now data
    new_data = nc.Dataset('/Users/yaircohen/PycharmProjects/scampy'+new_path[1:], 'r')
    # generate or estimate
    costFun = generate_costFun(true_data, new_data) # + prior knowlage -log(PDF) of value for the theta
    os.remove('/Users/yaircohen/PycharmProjects/scampy' + new_path[1:])

    return costFun

def generate_costFun(true_data,new_data):
    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5
    t0 = 0.0
    # define true data
    p_lwp = np.multiply(true_data.groups['timeseries'].variables['lwp'],1.0)
    z_p = np.multiply(true_data.groups['profiles'].variables['z'],1.0)
    t_p = np.multiply(true_data.groups['profiles'].variables['t'],1.0)
    tp1 = np.where(t_p[:] > t0 * 3600.0)[0][0]
    p_thetali = np.multiply(true_data.groups['profiles'].variables['thetal_mean'],1.0)
    p_temperature = np.multiply(true_data.groups['profiles'].variables['temperature_mean'],1.0)
    p_buoyancy = np.multiply(true_data.groups['profiles'].variables['buoyancy_mean'],1.0)
    p_p0 = np.multiply(true_data.groups['reference'].variables['p0'],1.0)
    p_ql = np.multiply(true_data.groups['profiles'].variables['ql_mean'],1.0)
    p_qt = np.multiply(true_data.groups['profiles'].variables['qt_mean'],1.0)
    p_qv = p_qt - p_ql
    p_P0, s_P0 = np.meshgrid(p_p0, p_p0)
    p_CF = np.multiply(true_data.groups['timeseries'].variables['cloud_cover'],1.0)
    p_RH = np.multiply(epsi * np.exp(17.625 * (p_temperature - 273.15) / (p_temperature - 273.15 + 243.04)),
                       np.divide(1 - p_qt + epsi_inv * p_qv, np.multiply(epsi_inv, p_qv * np.rot90(p_P0, k=1))))

    s_lwp = np.multiply(new_data.groups['timeseries'].variables['lwp'],1.0)
    z_s = np.multiply(new_data.groups['profiles'].variables['z'],1.0)
    t_s = np.multiply(new_data.groups['profiles'].variables['t'],1.0)
    ts1 = np.where(t_s[:] > t0 * 3600.0)[0][0]
    s_thetal = np.multiply(new_data.groups['profiles'].variables['thetal_mean'],1.0)
    s_temperature = np.multiply(new_data.groups['profiles'].variables['temperature_mean'],1.0)
    s_buoyancy = np.multiply(new_data.groups['profiles'].variables['buoyancy_mean'],1.0)
    s_p0 = np.multiply(new_data.groups['reference'].variables['p0'],1.0)
    s_ql = np.multiply(new_data.groups['profiles'].variables['ql_mean'],1.0)
    s_qt = np.multiply(new_data.groups['profiles'].variables['qt_mean'],1.0)
    s_qv = s_qt - s_ql
    s_P0, s_P0 = np.meshgrid(s_p0, s_p0)
    s_CF = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'],1.0)
    s_RH = np.multiply(epsi * np.exp(17.625 * (s_temperature - 273.15) / (s_temperature - 273.15 + 243.04)),
                       np.divide(1 - s_qt + epsi_inv * s_qv, np.multiply(epsi_inv, s_qv * np.rot90(s_P0, k=1))))

    Theta_p = np.mean(p_thetali[tp1:, :], 0)
    T_p = np.mean(p_temperature[tp1:, :], 0)
    RH_p = np.mean(p_RH[tp1:, :], 0)
    qt_p = np.mean(p_qt[tp1:, :], 0)
    ql_p = np.mean(p_ql[tp1:, :], 0)
    b_p = np.mean(p_buoyancy[tp1:, :], 0)

    Theta_s = np.mean(p_thetali[ts1:, :], 0)
    T_s = np.mean(p_temperature[ts1:, :], 0)
    RH_s = np.mean(p_RH[ts1:, :], 0)
    qt_s = np.mean(p_qt[ts1:, :], 0)
    ql_s = np.mean(p_ql[ts1:, :], 0)
    b_s = np.mean(p_buoyancy[ts1:, :], 0)

    CAPE_theta = np.zeros(len(z_s))
    CAPE_T = np.zeros(len(z_s))
    CAPE_RH = np.zeros(len(z_s))
    CAPE_b = np.zeros(len(z_s))
    CAPE_qt = np.zeros(len(z_s))
    CAPE_ql = np.zeros(len(z_s))

    for k in range(0, len(z_s)):
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

    rnoise = 0.5
    f = [[d_CAPE_qt, 0, 0], [0, d_CAPE_theta, 0], [0, 0, dCF]]
    sigma = np.multiply(rnoise, [[1 / np.max([var_qt, 0.001]), 0, 0], [0, 1 / np.max([var_theta, 0.001]), 0],
                                 [0, 0, 1 / np.max([var_CF, 0.001])]])
    J0 = np.divide(np.linalg.norm(np.dot(sigma, f), ord=None), 0.5)  # ord=None for matrix gives the 2-norm
    logp = 0.0
    u = np.multiply(J0 - logp, 1.0)

    print('============> CostFun = ', u, '  <============')
    plt.ion()
    plt.plot(Theta_p, z_p, 'b', linewidth = 3)
    plt.plot(Theta_s, z_s, 'r')
    plt.pause(0.05)
    return u

def MCMC_paramlist(theta, case_name): # vel_pressure_coeff_i,

    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = 'sweep'

    paramlist['turbulence'] = {}
    paramlist['turbulence']['prandtl_number'] = 1.0
    paramlist['turbulence']['Ri_bulk_crit'] = 0.0

    paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.15
    paramlist['turbulence']['EDMF_PrognosticTKE']['surface_scalar_coeff'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.03
    # paramlist['turbulence']['EDMF_PrognosticTKE']['w_entr_coeff'] = 0.5 # "b1"
    # paramlist['turbulence']['EDMF_PrognosticTKE']['w_buoy_coeff'] =  0.5  # "b2"

    paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 0.5
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = float(theta)
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = float(theta)
    paramlist['turbulence']['EDMF_PrognosticTKE']['vel_pressure_coeff'] = 5e-05
    paramlist['turbulence']['EDMF_PrognosticTKE']['vel_buoy_coeff'] = 1.0

    paramlist['turbulence']['EDMF_BulkSteady'] = {}
    paramlist['turbulence']['EDMF_BulkSteady']['surface_area'] = 0.05
    paramlist['turbulence']['EDMF_BulkSteady']['w_entr_coeff'] = 2.0  # "w_b"
    paramlist['turbulence']['EDMF_BulkSteady']['w_buoy_coeff'] = 1.0
    paramlist['turbulence']['EDMF_BulkSteady']['max_area_factor'] = 5.0
    paramlist['turbulence']['EDMF_BulkSteady']['entrainment_factor'] = 0.5
    paramlist['turbulence']['EDMF_BulkSteady']['detrainment_factor'] = 0.5

    paramlist['turbulence']['updraft_microphysics'] = {}
    paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.01

    return paramlist

def write_file(paramlist):
    fh = open("paramlist_"+paramlist['meta']['casename']+ ".in", 'w')
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return