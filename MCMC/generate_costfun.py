import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
import time
from create_records import create_record, create_record_full

def costfun(G, true_data, model_type):
    epsi = 287.1 / 461.5
    epsi_inv = 287.1 / 461.5
    t0 = 0.0

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
    f = np.diag([dlwp, dCF, dCT])
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