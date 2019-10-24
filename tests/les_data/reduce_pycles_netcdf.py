import numpy as np
import netCDF4 as nc
import pylab as plt
import argparse
import subprocess
import os
# command line:
# python reduce_pycles_netcdf.py input   output
def main():
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("fullfilename")
    parser.add_argument("fname")
    args = parser.parse_args()
    fullfilename = args.fullfilename
    fname = args.fname

    data = nc.Dataset(fullfilename, 'r')
    buoyancy_mean_  = np.array(data.groups['profiles'].variables['buoyancy_mean'])
    env_w_ = data.groups['profiles'].variables['env_w']
    temperature_mean_ = data.groups['profiles'].variables['temperature_mean']
    u_mean_ = data.groups['profiles'].variables['u_mean']
    v_mean_ = data.groups['profiles'].variables['v_mean']
    tke_mean_ = data.groups['profiles'].variables['tke_mean']
    v_translational_mean_ = data.groups['profiles'].variables['v_translational_mean']
    u_translational_mean_ = data.groups['profiles'].variables['u_translational_mean']
    updraft_buoyancy_ = data.groups['profiles'].variables['updraft_b']
    updraft_fraction_ = data.groups['profiles'].variables['updraft_fraction']

    resolved_z_flux_thetali_ = data.groups['profiles'].variables['resolved_z_flux_thetali']
    resolved_z_flux_qt_ = data.groups['profiles'].variables['resolved_z_flux_qt']

    updraft_ddz_p_alpha_ = data.groups['profiles'].variables['updraft_ddz_p_alpha']
    rho_ = data.groups['reference'].variables['rho0_half']
    p0_ = np.multiply(data.groups['reference'].variables['rho0_half'],100.0)

    # try the TKE diagnostics outputs
    tke_prod_A_ = data.groups['profiles'].variables['tke_prod_A']
    tke_prod_B_ = data.groups['profiles'].variables['tke_prod_B']
    tke_prod_D_ = data.groups['profiles'].variables['tke_prod_D']
    tke_prod_P_ = data.groups['profiles'].variables['tke_prod_P']
    tke_prod_T_ = data.groups['profiles'].variables['tke_prod_T']
    tke_prod_S_ = data.groups['profiles'].variables['tke_prod_S']
    tke_nd_mean_ = data.groups['profiles'].variables['tke_nd_mean']
    try:
        qr_mean_ = np.multiply(data.groups['profiles'].variables['qr_mean'],1000.0)
        env_qr_ = np.multiply(data.groups['profiles'].variables['env_qr'],1000.0)
        updraft_qr_ = np.multiply(data.groups['profiles'].variables['updraft_qr'],1000.0)
    except:
        qr_mean_ = np.zeros_like(env_w_)
        env_qr_ = np.zeros_like(env_w_)
        updraft_qr_ = np.zeros_like(env_w_)
    try:
        qi_mean_ = np.multiply(data.groups['profiles'].variables['qi_mean'],1000.0)
        env_qi_ = np.multiply(data.groups['profiles'].variables['env_qi'],1000.0)
        updraft_qi_ = np.multiply(data.groups['profiles'].variables['updraft_qi'],1000.0)
    except:
        qi_mean_ = np.multiply(np.zeros_like(env_w_),1000.0)
        env_qi_ = np.multiply(np.zeros_like(env_w_),1000.0)
        updraft_qi_ = np.multiply(np.zeros_like(env_w_),1000.0)
    try:
        qt_mean_ = np.multiply(data.groups['profiles'].variables['qt_mean'],1000.0)
        qt_mean2_ = np.multiply(data.groups['profiles'].variables['qt_mean2'],1e6)
        env_qt_ = np.multiply(data.groups['profiles'].variables['env_qt'],1000.0)
        env_qt2_ = np.multiply(data.groups['profiles'].variables['env_qt2'],1e6)
        updraft_qt_ = np.multiply(data.groups['profiles'].variables['updraft_qt'],1000.0)
    except:
        qt_mean_ = np.zeros_like(env_w_)
        qt_mean2_ = np.zeros_like(env_w_)
        env_qt_ = np.zeros_like(env_w_)
        env_qt2_ = np.zeros_like(env_w_)
        updraft_qt_ = np.zeros_like(env_w_)
    try:
        ql_mean_ = np.multiply(data.groups['profiles'].variables['ql_mean'],1000.0)
        env_ql_ = np.multiply(data.groups['profiles'].variables['env_ql'],1000.0)
        updraft_ql_ = np.multiply(data.groups['profiles'].variables['updraft_ql'],1000.0)
    except:
        ql_mean_ = np.zeros_like(env_w_)
        env_ql_ = np.zeros_like(env_w_)
        updraft_ql_ = np.zeros_like(env_w_)
    try:
        env_qt_thetali_ = np.multiply(data.groups['profiles'].variables['env_qt_thetali'],1000.0)
    except:
        env_qt_thetali_ = np.zeros_like(env_w_)

    try:
        thetali_mean_ = data.groups['profiles'].variables['thetali_mean']
        thetali_mean2_ = data.groups['profiles'].variables['thetali_mean2']
    except:
        thetali_mean_ = data.groups['profiles'].variables['theta_mean']
        thetali_mean2_ = data.groups['profiles'].variables['theta_mean2']

    env_thetali_ = data.groups['profiles'].variables['env_thetali']
    env_thetali2_ = data.groups['profiles'].variables['env_thetali2']
    env_buoyancy_ = data.groups['profiles'].variables['env_b']
    updraft_thetali_ = data.groups['profiles'].variables['updraft_thetali']
    updraft_w_ = data.groups['profiles'].variables['updraft_w']

    friction_velocity_mean_ = data.groups['timeseries'].variables['friction_velocity_mean']
    shf_surface_mean_ = data.groups['timeseries'].variables['shf_surface_mean']
    lhf_surface_mean_ = data.groups['timeseries'].variables['lhf_surface_mean']
    try:
        cloud_fraction_ = data.groups['timeseries'].variables['cloud_fraction']
        cloud_base_ = data.groups['timeseries'].variables['cloud_base']
        cloud_top_ = data.groups['timeseries'].variables['cloud_top']
    except:
        cloud_fraction_ = np.zeros_like(lhf_surface_mean_)
        cloud_base_ = np.zeros_like(lhf_surface_mean_)
        cloud_top_ = np.zeros_like(lhf_surface_mean_)
    try:
        lwp_ = data.groups['timeseries'].variables['lwp']
    except:
    	lwp_ = np.zeros_like(lhf_surface_mean_)
    # thetali_srf_int_ = data.groups['timeseries'].variables['thetali_srf_int'] # this is here since

    z_half_ = np.divide(data.groups['profiles'].variables['z_half'],1000.0)
    t_ = np.divide(data.groups['profiles'].variables['t'],3600.0)


    # flux diagnosis
    a_1_a = np.multiply(updraft_fraction_, np.subtract(1.0,updraft_fraction_))
    rho_temp = np.tile(rho_,(np.shape(updraft_fraction_)[0],1))
    updraft_buoyancy_ -=buoyancy_mean_
    env_buoyancy_ -=buoyancy_mean_
    Hvar_mean_ = calc_covar(thetali_mean2_, thetali_mean_, thetali_mean_)
    QTvar_mean_ = calc_covar(qt_mean2_,       qt_mean_,      qt_mean_)
    env_Hvar_ = calc_covar(env_thetali2_,   env_thetali_,  env_thetali_)
    env_QTvar_ = calc_covar(env_qt2_,        env_qt_,       env_qt_)
    env_HQTcov_ = calc_covar(env_qt_thetali_, env_qt_,       env_thetali_)
    massflux_h_        = np.multiply(a_1_a,np.multiply(np.subtract(updraft_w_, env_w_), np.subtract(updraft_thetali_, env_thetali_)))
    massflux_qt_       = np.multiply(a_1_a,np.multiply(np.subtract(updraft_w_, env_w_), np.subtract(updraft_qt_, env_qt_)))
    total_flux_h_      = np.array(resolved_z_flux_thetali_[:, :])
    total_flux_qt_     = np.array(resolved_z_flux_qt_[:, :])
    diffusive_flux_h_  = np.subtract(total_flux_h_,massflux_h_)
    diffusive_flux_qt_ = np.subtract(total_flux_qt_,massflux_qt_)
    massflux_h_        = np.multiply(rho_temp, massflux_h_)
    massflux_qt_       = np.multiply(rho_temp, massflux_qt_)
    total_flux_h_      = np.multiply(rho_temp, total_flux_h_)
    total_flux_qt_     = np.multiply(rho_temp, total_flux_qt_)
    diffusive_flux_h_  = np.multiply(rho_temp, diffusive_flux_h_)
    diffusive_flux_qt_ = np.multiply(rho_temp, diffusive_flux_qt_)

    output = nc.Dataset(fname, "w", format="NETCDF4")
    output.createDimension('z', len(z_half_))
    output.createDimension('t', len(t_))
    output.createDimension('dim', None)
    output.createGroup("profiles")
    output.createGroup("timeseries")

    t = output.createVariable('t', 'f4', 't')
    z_half = output.createVariable('z_half', 'f4', 'z')

    profiles_grp = output.groups["profiles"]

    rho = profiles_grp.createVariable('rho','f4',('z'))
    p0 = profiles_grp.createVariable('p0','f4',('z'))
    Hvar_mean = profiles_grp.createVariable('Hvar_mean','f4',('t','z'))
    QTvar_mean = profiles_grp.createVariable('QTvar_mean','f4',('t','z'))
    env_Hvar = profiles_grp.createVariable('env_Hvar','f4',('t','z'))
    env_QTvar = profiles_grp.createVariable('env_QTvar','f4',('t','z'))
    env_HQTcov = profiles_grp.createVariable('env_HQTcov','f4',('t','z'))
    massflux_h = profiles_grp.createVariable('massflux_h','f4',('t','z'))
    massflux_qt = profiles_grp.createVariable('massflux_qt','f4',('t','z'))
    total_flux_h = profiles_grp.createVariable('total_flux_h','f4',('t','z'))
    total_flux_qt = profiles_grp.createVariable('total_flux_qt','f4',('t','z'))
    diffusive_flux_h = profiles_grp.createVariable('diffusive_flux_h','f4',('t','z'))
    diffusive_flux_qt = profiles_grp.createVariable('diffusive_flux_qt','f4',('t','z'))
    buoyancy_mean = profiles_grp.createVariable('buoyancy_mean','f4',('t','z'))
    resolved_z_flux_thetali = profiles_grp.createVariable('resolved_z_flux_thetali','f4',('t','z'))
    resolved_z_flux_qt = profiles_grp.createVariable('resolved_z_flux_qt','f4',('t','z'))
    temperature_mean = profiles_grp.createVariable('temperature_mean','f4',('t','z'))
    updraft_ddz_p_alpha = profiles_grp.createVariable('updraft_ddz_p_alpha','f4',('t','z'))
    thetali_mean = profiles_grp.createVariable('thetali_mean','f4',('t','z'))
    qt_mean = profiles_grp.createVariable('qt_mean','f4',('t','z'))
    ql_mean = profiles_grp.createVariable('ql_mean','f4',('t','z'))
    u_mean = profiles_grp.createVariable('u_mean','f4',('t','z'))
    v_mean = profiles_grp.createVariable('v_mean','f4',('t','z'))
    tke_mean = profiles_grp.createVariable('tke_mean','f4',('t','z'))
    v_translational_mean = profiles_grp.createVariable('v_translational_mean','f4',('t','z'))
    u_translational_mean = profiles_grp.createVariable('u_translational_mean','f4',('t','z'))
    updraft_buoyancy = profiles_grp.createVariable('updraft_buoyancy','f4',('t','z'))
    updraft_fraction = profiles_grp.createVariable('updraft_fraction','f4',('t','z'))
    env_thetali = profiles_grp.createVariable('env_thetali','f4',('t','z'))
    updraft_thetali = profiles_grp.createVariable('updraft_thetali','f4',('t','z'))
    env_qt = profiles_grp.createVariable('env_qt','f4',('t','z'))
    updraft_qt = profiles_grp.createVariable('updraft_qt','f4',('t','z'))
    env_ql = profiles_grp.createVariable('env_ql','f4',('t','z'))
    env_buoyancy = profiles_grp.createVariable('env_buoyancy','f4',('t','z'))
    updraft_ql = profiles_grp.createVariable('updraft_ql','f4',('t','z'))
    qr_mean = profiles_grp.createVariable('qr_mean','f4',('t','z'))
    env_qr = profiles_grp.createVariable('env_qr','f4',('t','z'))
    updraft_qr = profiles_grp.createVariable('updraft_qr','f4',('t','z'))
    updraft_w = profiles_grp.createVariable('updraft_w','f4',('t','z'))
    env_w = profiles_grp.createVariable('env_w','f4',('t','z'))
    thetali_mean2 = profiles_grp.createVariable('thetali_mean2','f4',('t','z'))
    qt_mean2 = profiles_grp.createVariable('qt_mean2','f4',('t','z'))
    env_thetali2 = profiles_grp.createVariable('env_thetali2','f4',('t','z'))
    env_qt2 = profiles_grp.createVariable('env_qt2','f4',('t','z'))
    env_qt_thetali = profiles_grp.createVariable('env_qt_thetali','f4',('t','z'))
    tke_prod_A = profiles_grp.createVariable('tke_prod_A','f4',('t','z'))
    tke_prod_B = profiles_grp.createVariable('tke_prod_B','f4',('t','z'))
    tke_prod_D = profiles_grp.createVariable('tke_prod_D','f4',('t','z'))
    tke_prod_P = profiles_grp.createVariable('tke_prod_P','f4',('t','z'))
    tke_prod_T = profiles_grp.createVariable('tke_prod_T','f4',('t','z'))
    tke_prod_S = profiles_grp.createVariable('tke_prod_S','f4',('t','z'))
    tke_nd_mean = profiles_grp.createVariable('tke_nd_mean','f4',('t','z'))

    timeseries_grp = output.groups['timeseries']
    cloud_fraction = timeseries_grp.createVariable('cloud_fraction','f4','t')
    cloud_base = timeseries_grp.createVariable('cloud_base','f4','t')
    cloud_top = timeseries_grp.createVariable('cloud_top','f4','t')
    friction_velocity_mean = timeseries_grp.createVariable('friction_velocity_mean','f4','t')
    shf_surface_mean = timeseries_grp.createVariable('shf_surface_mean','f4','t')
    lhf_surface_mean = timeseries_grp.createVariable('lhf_surface_mean','f4','t')
    lwp = timeseries_grp.createVariable('lwp','f4','t')
    # thetali_srf_int = timeseries_grp.createVariable('thetali_srf_int','f4','t')

    rho[:] = rho_[:]
    p0[:] = p0_[:]
    Hvar_mean[:,:] = Hvar_mean_[:,:]
    QTvar_mean[:,:] = QTvar_mean_[:,:]
    env_Hvar[:,:] = env_Hvar_[:,:]
    env_QTvar[:,:] = env_QTvar_[:,:]
    env_HQTcov[:,:] = env_HQTcov_[:,:]
    massflux_h[:,:] = massflux_h_[:,:]
    massflux_qt[:,:] = massflux_qt_[:,:]
    total_flux_h[:,:] = total_flux_h_[:,:]
    total_flux_qt[:,:] = total_flux_qt_[:,:]
    diffusive_flux_h[:,:] = diffusive_flux_h_[:,:]
    diffusive_flux_qt[:,:] = diffusive_flux_qt_[:,:]
    buoyancy_mean[:,:] = buoyancy_mean_[:,:]
    resolved_z_flux_thetali[:,:] = resolved_z_flux_thetali_[:,:]
    resolved_z_flux_qt[:,:] = resolved_z_flux_qt_[:,:]
    temperature_mean[:,:] = temperature_mean_[:,:]
    updraft_ddz_p_alpha[:,:] = updraft_ddz_p_alpha_[:,:]
    thetali_mean[:,:] = thetali_mean_[:,:]
    qt_mean[:,:] = qt_mean_[:,:]
    ql_mean[:,:] = ql_mean_[:,:]
    u_mean[:,:] = u_mean_[:,:]
    v_mean[:,:] = v_mean_[:,:]
    tke_mean[:,:] = tke_mean_[:,:]
    v_translational_mean[:,:] = v_translational_mean_[:,:]
    u_translational_mean[:,:] = u_translational_mean_[:,:]
    updraft_buoyancy[:,:] = updraft_buoyancy_[:,:]
    updraft_fraction[:,:] = updraft_fraction_[:,:]
    env_thetali[:,:] = env_thetali_[:,:]
    updraft_thetali[:,:] = updraft_thetali_[:,:]
    env_qt[:,:] = env_qt_[:,:]
    updraft_qt[:,:] = updraft_qt_[:,:]
    env_ql[:,:] = env_ql_[:,:]
    updraft_ql[:,:] = updraft_ql_[:,:]
    qr_mean[:,:] = qr_mean_[:,:]
    env_qr[:,:] = env_qr_[:,:]
    updraft_qr[:,:] = updraft_qr_[:,:]
    updraft_w[:,:] = updraft_w_[:,:]
    env_w[:,:] = env_w_[:,:]
    thetali_mean2[:,:] = thetali_mean2_[:,:]
    qt_mean2[:,:] = qt_mean2_[:,:]
    env_thetali2[:,:] = env_thetali2_[:,:]
    env_qt2[:,:] = env_qt2_[:,:]
    env_qt_thetali[:,:] = env_qt_thetali_[:,:]
    env_buoyancy[:,:] = env_buoyancy_[:,:]
    tke_prod_A[:,:] = tke_prod_A_[:,:]
    tke_prod_B[:,:] = tke_prod_B_[:,:]
    tke_prod_D[:,:] = tke_prod_D_[:,:]
    tke_prod_P[:,:] = tke_prod_P_[:,:]
    tke_prod_T[:,:] = tke_prod_T_[:,:]
    tke_prod_S[:,:] = tke_prod_S_[:,:]
    tke_nd_mean[:,:] = tke_nd_mean_[:,:]

    cloud_fraction[:] = cloud_fraction_[:]
    cloud_base[:] = cloud_base_[:]
    cloud_top[:] = cloud_top_[:]
    friction_velocity_mean[:] = friction_velocity_mean_[:]
    shf_surface_mean[:] = shf_surface_mean_[:]
    lhf_surface_mean[:] = lhf_surface_mean_[:]
    lwp[:] = lwp_[:]

    z_half[:] = z_half_[:]
    t[:] = t_[:]
    output.close()

def calc_covar(var_sq, var1, var2):

    covar = np.subtract(var_sq,np.multiply(var1,var2))
    return covar

if __name__ == '__main__':
    main()

