import numpy as np
import netCDF4 as nc
import subprocess
import json
import os
from shutil import copyfile

def scm_iterP(ncore, true_data, theta,  case_name, geom_opt=0):

    txt = 'ABCDEFGHIJK'
    src = '/cluster/home/yairc/scampy/' + case_name + '.in'
    dst = '/cluster/home/yairc/scampy/' + case_name + txt[int(ncore)] + '.in'
    #src = '/Users/yaircohen/PycharmProjects/scampy/' + case_name + '.in'
    #dst = '/Users/yaircohen/PycharmProjects/scampy/' + case_name + txt[int(ncore)] + '.in'

    copyfile(src, dst)

    namelistfile = open(dst,'r')
    namelist = json.load(namelistfile)
    uuid0 = namelist['meta']['uuid']
    uuid = uuid0[0:-5]+'tune'+ txt[int(ncore)]
    namelist['meta']['uuid'] = uuid
    case0 = namelist['meta']['casename']
    case = case0 + txt[int(ncore)]
    namelist['meta']['casename'] = case
    namelist['meta']['simname'] = case
    namelistfile.close()

    new_dir = namelist['output']['output_root'] + 'Output.' + case_name + txt[int(ncore)] + '.' + uuid[-5:] + '/stats/'
    new_path = new_dir + 'Stats.' + case_name + txt[int(ncore)]+ '.nc'
    newnamelistfile = open(dst, 'w')
    json.dump(namelist, newnamelistfile, sort_keys=True, indent=4)
    newnamelistfile.close()

    # receive parameter value and generate paramlist file for new data
    paramlist = MCMC_paramlist(theta, case_name+txt[int(ncore)])
    write_file(paramlist)

    print('============ start iteration with paramater = ', theta)  # + str(ncore)
    runstring = 'python main.py ' + case_name  + txt[int(ncore)] + '.in paramlist_Bomex' + txt[int(ncore)] + '.in'  #
    subprocess.call(runstring, shell=True)  # cwd = '/Users/yaircohen/PycharmProjects/scampy/',
    print('============ iteration end')

    # load NC of the now data
    new_data = nc.Dataset(new_path, 'r')
    # generate or estimate
    costFun = generate_costFun(theta, true_data, new_data, new_dir) # + prior knowledge -log(PDF) of value for the theta

    os.remove(new_path)

    return costFun

def generate_costFun(theta, true_data,new_data, new_dir):


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
    p_CF = np.multiply(true_data.groups['timeseries'].variables['cloud_cover'],1.0)
    FT = np.multiply(17.625, (np.divide(np.subtract(p_temperature, 273.15), (np.subtract(p_temperature, 273.15 + 243.04)))))
    p_RH = np.multiply(epsi * np.exp(FT),
                           np.divide(np.add(np.subtract(1, p_qt), epsi_inv * (p_qt - p_ql)),
                                     np.multiply(epsi_inv, np.multiply(p_p0, (p_qt - p_ql)))))


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
    s_CF = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'],1.0)
    FT = np.multiply(17.625,
                     (np.divide(np.subtract(s_temperature, 273.15), (np.subtract(s_temperature, 273.15 + 243.04)))))
    s_RH = np.multiply(epsi * np.exp(FT),
                       np.divide(np.add(np.subtract(1, s_qt), epsi_inv * (s_qt - s_ql)),
                                 np.multiply(epsi_inv, np.multiply(s_p0, (s_qt - s_ql)))))

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
    ft = np.transpose(f)
    sigma = np.multiply(rnoise, [[1 / np.max([var_qt, 0.001]), 0, 0], [0, 1 / np.max([var_theta, 0.001]), 0],
                                 [0, 0, 1 / np.max([var_CF, 0.001])]])
    #J0 = np.divide(np.linalg.norm(np.dot(sigma, f), ord=None), 0.5)  # ord=None for matrix gives the 2-norm
    J0 = np.divide(np.linalg.norm(np.dot(ft,(np.dot(sigma, f)))), 2.0)  # check the torder of dot products
    logp = 0.0
    u = np.multiply(J0 - logp, 1.0)

    # call record
    create_record(theta, u, new_data, new_dir)
    # store data
    print('============> CostFun = ', u, '  <============')
    return u

def MCMC_paramlist(theta, case_name): # vel_pressure_coeff_i,

    paramlist = {}
    paramlist['meta'] = {}
    paramlist['meta']['casename'] = case_name

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


def create_record(theta_, costFun_, new_data, new_dir):

    z_s = np.multiply(new_data.groups['profiles'].variables['z'],1.0)
    t_s = np.multiply(new_data.groups['profiles'].variables['t'],1.0)
    s_thetal = np.multiply(new_data.groups['profiles'].variables['thetal_mean'],1.0)

    nt = np.shape(t_s)[0]
    nz = np.shape(z_s)[0]


    fname = new_dir + 'tuning_record.nc'

    #tuning_recored = nc.Dataset(fname, 'r+', format='NETCDF4')

    if os.path.isfile(fname):
        print('scm_iter line 147')

        # load old data and close netCDF
        old_record = nc.Dataset(fname, 'r')
        lwp1_ = np.multiply(old_record.groups['data'].variables['lwp'], 1.0)
        cloud_cover1_ = np.multiply(old_record.groups['data'].variables['cloud_cover'], 1.0)
        cloud_top1_ = np.multiply(old_record.groups['data'].variables['cloud_top'], 1.0)
        cloud_base1_ = np.multiply(old_record.groups['data'].variables['cloud_base'], 1.0)
        thetal1_ = np.multiply(old_record.groups['data'].variables['thetal'], 1.0)
        tune_param1_ = np.multiply(old_record.groups['data'].variables['tune_param'], 1.0)
        costFun1_ = np.multiply(old_record.groups['data'].variables['costFun'], 1.0)
        old_record.close()

        # load existing data to variables
        lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
        cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
        cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['cloud_top'], 1.0)
        cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['cloud_base'], 1.0)
        thetal_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)

        # stack both
        _lwp = np.hstack((np.atleast_2d(lwp_.reshape((-1, 1))), lwp1_))
        _cloud_cover = np.hstack((np.atleast_2d(cloud_cover_.reshape((-1, 1))), cloud_cover1_))
        _cloud_top = np.hstack((np.atleast_2d(cloud_top_.reshape((-1, 1))), cloud_top1_))
        _cloud_base = np.hstack((np.atleast_2d(cloud_base_.reshape((-1, 1))), cloud_base1_))
        _thetal = np.dstack((thetal1_, thetal_))
        _tune_param = np.hstack((tune_param1_, theta_))
        _costFun = np.hstack((costFun1_, costFun_))
        # overwrite the netCDF
        tuning_record = nc.Dataset(fname, 'r+')

        tuning_record.groups['data'].variables['lwp'][:,:] = _lwp
        tuning_record.groups['data'].variables['cloud_cover'][:,:] = _cloud_cover
        tuning_record.groups['data'].variables['cloud_top'][:,:] = _cloud_top
        tuning_record.groups['data'].variables['cloud_base'][:,:] = _cloud_base
        tuning_record.groups['data'].variables['thetal'][:,:,:] = _thetal
        tuning_record.groups['data'].variables['tune_param'][:] = _tune_param
        tuning_record.groups['data'].variables['costFun'][:] = _costFun

        tuning_record.close()

        old_record = nc.Dataset(fname, 'r')
        _lwp1_ = np.multiply(old_record.groups['data'].variables['lwp'], 1.0)
        _cloud_cover1_ = np.multiply(old_record.groups['data'].variables['cloud_cover'], 1.0)
        _cloud_top1_ = np.multiply(old_record.groups['data'].variables['cloud_top'], 1.0)
        _cloud_base1_ = np.multiply(old_record.groups['data'].variables['cloud_base'], 1.0)
        _thetal1_ = np.multiply(old_record.groups['data'].variables['thetal'], 1.0)
        _tune_param1_ = np.multiply(old_record.groups['data'].variables['tune_param'], 1.0)
        _costFun1_ = np.multiply(old_record.groups['data'].variables['costFun'], 1.0)
        old_record.close()



        # # find the length of the third dim of thetal for the number of the tuned simulation
        # if  thetal1_.ndim < 3:
        #     dim = 0
        # else:
        #     dim =len( thetal1_[0,0,:])
        # old_record.close()
        #
        # # build a new record that will overwrite the old one
        #print(fname)
        #tuning_recored = nc.Dataset(fname, 'r+', format='NETCDF4') - yair
        #grp_stats = tuning_recored.createGroup('data') - yair
        #grp_stats.createDimension('z', nz)
        #grp_stats.createDimension('t', nt)
        #grp_stats.createDimension('dim', dim + 1)
        #
        # # create variables
        # _lwp = np.zeros((dim + 1,nt))
        # _cloud_cover = np.zeros((dim + 1,nt))
        # _cloud_top = np.zeros((dim + 1,nt))
        # _cloud_base = np.zeros((dim + 1,nt))
        # _thetal = np.zeros((dim + 1, nt, nz))
        # _theta = np.zeros((dim + 1,nt))
        # _costFun = np.zeros((dim + 1,nt))
        # print('concatenate works')
        # print('np.shape(_lwp) =', np.shape(_lwp))
        # print('np.shape(_cloud_cover) =', np.shape(_cloud_cover))
        # print('np.shape(_thetal) =', np.shape((_thetal)))
        # print('np.shape(_theta) =', np.shape(_theta))
        # print('np.shape(_costFun) =', np.shape(_costFun))
        # print('nt = ', nt)
        #
        #
        #
        #
        # # store old data in first part of new variables
        #_t = np.multiply(t_s, 1.0)
        #_z = np.multiply(z_s, 1.0)
        # _lwp[0:dim,:] = lwp1_
        # _cloud_cover[0:dim,:] = cloud_cover1_
        # _cloud_top[0:dim,:] = cloud_top1_
        # _cloud_base[0:dim,:] = cloud_base1_
        # _thetal[0:dim,:,:] = thetal1_
        # _theta[0:dim,:] = tune_param1_
        # _costFun[0:dim,:] = costFun1_
        #
        #
        #
        # # add new data to variables
        # print('np.shape(lwp_) =', np.shape(lwp_))
        # _lwp[dim + 1,:] = lwp_
        # _cloud_cover[dim + 1,:] = cloud_cover_
        # _cloud_top[dim + 1,:] = cloud_top_
        # _cloud_base[dim + 1,:] = cloud_base_
        # _thetal[dim + 1,:,:] = thetal_
        # _theta[dim + 1,:] = theta_
        # _costFun[dim + 1,:] = costFun_

        # t = grp_stats.createVariable('t', 'f4', 't')
        # z = grp_stats.createVariable('z', 'f4', 'z')
        # lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
        # cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
        # cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
        # cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
        # thetal = grp_stats.createVariable('thetal', 'f4', ('t', ('t', 'z', 'dim')))
        # tune_param = grp_stats.createVariable('tune_param', 'f4', ('t', 'dim'))
        # costFun = grp_stats.createVariable('costFun', 'f4', ('t', 'dim'))

        #t[:] = _t
        #z[:] = _z

    else:
        print('scm_iter line 257')
        tuning_record = nc.Dataset(fname, "w", format="NETCDF4")
        grp_stats = tuning_record.createGroup('data')
        grp_stats.createDimension('z', nz)
        grp_stats.createDimension('t', nt)
        grp_stats.createDimension('dim', None)



        lwp_ = np.multiply(new_data.groups['timeseries'].variables['lwp'], 1.0)
        cloud_cover_ = np.multiply(new_data.groups['timeseries'].variables['cloud_cover'], 1.0)
        cloud_top_ = np.multiply(new_data.groups['timeseries'].variables['cloud_top'], 1.0)
        cloud_base_ = np.multiply(new_data.groups['timeseries'].variables['cloud_base'], 1.0)
        thetal_ = np.multiply(new_data.groups['profiles'].variables['thetal_mean'], 1.0)

        t = grp_stats.createVariable('t', 'f4', 't')
        z = grp_stats.createVariable('z', 'f4', 'z')
        lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
        cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
        cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
        cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
        thetal = grp_stats.createVariable('thetal', 'f4', ('t', 'z', 'dim'))
        tune_param = grp_stats.createVariable('tune_param', 'f4', 'dim')
        costFun = grp_stats.createVariable('costFun', 'f4', 'dim') # this might be a problem if dim=1 implies 2 value

        _t = np.multiply(t_s, 1.0)
        _z = np.multiply(z_s, 1.0)
        _lwp = lwp_
        _cloud_cover = cloud_cover_
        _cloud_top = cloud_top_
        _cloud_base = cloud_base_
        _thetal = thetal_
        _tune_param = theta_
        _costFun = costFun_

        t[:] = _t
        z[:] = _z
        lwp[:,:] = np.atleast_1d(_lwp.reshape((-1, 1)))
        cloud_cover[:,:] = np.atleast_1d(_cloud_cover.reshape((-1, 1)))
        cloud_top[:,:] = np.atleast_1d(_cloud_top.reshape((-1, 1)))
        cloud_base[:,:] = np.atleast_1d(_cloud_base.reshape((-1, 1)))
        thetal[:,:,:] = np.atleast_3d(_thetal)
        tune_param[:] = _tune_param
        costFun[:] = _costFun

        tuning_record.close()




    return
