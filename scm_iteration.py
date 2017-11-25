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
    s_T = new_data.groups['profiles'].variables['temperature_mean']
    p_T = true_data.groups['profiles'].variables['temperature_mean']
    s_p0 = new_data.groups['reference'].variables['p0']
    p_p0 = true_data.groups['reference'].variables['p0']
    s_ql = new_data.groups['profiles'].variables['ql_mean']
    p_ql = true_data.groups['profiles'].variables['ql_mean']
    s_qt = new_data.groups['profiles'].variables['qt_mean']
    p_qt = true_data.groups['profiles'].variables['qt_mean']
    s_qv = s_qt - s_ql
    p_qv = p_qt - p_ql
    s_P0, s_P0 = np.meshgrid(s_p0, s_p0)
    p_P0, s_P0 = np.meshgrid(p_p0, p_p0)
    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5
    s_RH = np.multiply(epsi * np.exp(17.625 * (s_T - 273.15) / (s_T - 273.15 + 243.04)),
                       np.divide(1 - s_qt + epsi_inv * s_qv, np.multiply(epsi_inv, s_qv * s_P0)))
    p_RH = np.multiply(epsi * np.exp(17.625 * (p_T - 273.15) / (p_T - 273.15 + 243.04)),
                       np.divide(1 - s_qt + epsi_inv * s_qv, np.multiply(epsi_inv, s_qv * p_P0)))

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
    json.dump(paramlist, fh, sort_keys=True, indent=4)
    fh.close()

    return