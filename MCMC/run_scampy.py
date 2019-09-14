import numpy as np
import subprocess
import argparse
import netCDF4 as nc
import sys
import time
import os
from create_records import initiate_record
from create_records import create_record, create_record_full
import json

# from python command line:
# import run_scampy
# instance = run_scampy.run_scampy('Bomex')
# instance.run(0.1)


np.set_printoptions(precision=3, suppress=True)
np.random.seed(2019)

class run_scampy(object):

    def __init__(self, case_name):
        self.case_name  = case_name
        self.type  = 'map'
        self.model_name  = 'scampy'

    def __call__(self, theta):
        return self.run(theta)

    def run(self, theta):
        localpath = os.getcwd()
        output_filename = localpath+'/tuning_log.nc'
        myscampyfolder = localpath[0:-5]
        initiate_record(output_filename, theta)
        # compile the SCM
        subprocess.call("CC=mpicc python setup.py build_ext --inplace",  shell=True, cwd=myscampyfolder)
        # generate namelist and paramlist
        subprocess.call("python generate_namelist.py " + self.case_name,  shell=True, cwd=myscampyfolder)
        # subprocess.call("python generate_paramlist.py " + case_name,  shell=True, cwd=myscampyfolder)
        # edit namelist and paramlist (add theta)
        file_case = open(myscampyfolder +"/"+ self.case_name + '.in').read()
        namelist = json.loads(file_case)
        uuid = namelist['meta']['uuid']
        new_path = namelist['output']['output_root'] + 'Output.' + self.case_name + '.' +  uuid[-5:] + '/stats/Stats.' + self.case_name + '.nc'
        paramlist = self.MCMC_paramlist(theta, self.case_name)
        self.write_file(paramlist, myscampyfolder)
        # run scampy with theta in paramlist
        subprocess.call("python main.py " + self.case_name + ".in " + "paramlist_" + self.case_name + ".in", shell=True, cwd=myscampyfolder)
        # load NC of the new scampy data
        new_data = nc.Dataset(myscampyfolder+new_path[1:], 'r')
        # calculate the cost fun u, print values
        u = self.generate_G(theta, new_data)
        # store theta and u
        create_record(theta, u, new_data, output_filename)
        # remove the simualtion data
        os.remove(myscampyfolder + new_path[1:])
        print("=========================== done")
        return u

# def run(case_name, true_path, model_type, theta):

#     localpath = os.getcwd()
#     output_filename = localpath+'/tuning_log.nc'
#     myscampyfolder = localpath[0:-5]
#     initiate_record(output_filename, theta)
#     # load true data
#     true_data = nc.Dataset(true_path, 'r')
#     # compile the SCM
#     subprocess.call("CC=mpicc python setup.py build_ext --inplace",  shell=True, cwd=myscampyfolder)
#     # generate namelist and paramlist
#     subprocess.call("python generate_namelist.py " + case_name,  shell=True, cwd=myscampyfolder)
#     # subprocess.call("python generate_paramlist.py " + case_name,  shell=True, cwd=myscampyfolder)
#     # edit namelist and paramlist (add theta)
#     file_case = open(myscampyfolder +"/"+ case_name + '.in').read()
#     namelist = json.loads(file_case)
#     uuid = namelist['meta']['uuid']
#     new_path = namelist['output']['output_root'] + 'Output.' + case_name + '.' +  uuid[-5:] + '/stats/Stats.' + case_name + '.nc'
#     paramlist = MCMC_paramlist(theta, case_name)
#     write_file(paramlist, myscampyfolder)

#     # run scampy with theta in paramlist\
#     subprocess.call("python main.py " + case_name + ".in " + "paramlist_" + case_name + ".in", shell=True, cwd=myscampyfolder)
#     # load NC of the new scampy data
#     new_data = nc.Dataset(myscampyfolder+new_path[1:], 'r')
#     # calculate the cost fun u, print values
#     u = generate_costFun(theta, true_data, new_data, model_type)
#     # store theta and u
#     create_record(theta, u, new_data, output_filename)
#     # remove the simualtion data
#     os.remove(myscampyfolder + new_path[1:])
#     return theta, u

    def generate_G(self, theta, new_data):
        epsi = 287.1 / 461.5
        epsi_inv = 287.1 / 461.5
        t0 = 0.0
        z =           np.array(new_data.groups['profiles'].variables['z'])
        t =           np.array(new_data.groups['profiles'].variables['t'])
        t1 =          260 #np.where(t[:] > t0 * 3600.0)[0][0]
        p0 =          np.array(new_data.groups['reference'].variables['p0'])

        lwp =         np.array(new_data.groups['timeseries'].variables['lwp_mean'])
        thetal =      np.array(new_data.groups['profiles'].variables['thetal_mean'])
        temperature = np.array(new_data.groups['profiles'].variables['temperature_mean'])
        buoyancy =    np.array(new_data.groups['profiles'].variables['buoyancy_mean'])
        ql =          np.array(new_data.groups['profiles'].variables['ql_mean'])
        qt =          np.array(new_data.groups['profiles'].variables['qt_mean'])
        CF =          np.array(new_data.groups['timeseries'].variables['cloud_cover_mean'])
        CT =          np.array(new_data.groups['timeseries'].variables['cloud_top_mean'])
        FT =          np.multiply(17.625,(np.divide(np.subtract(temperature, 273.15), (np.subtract(temperature, 273.15 + 243.04)))))
        RH =          np.multiply(epsi * np.exp(FT),np.divide(np.add(np.subtract(1, qt), epsi_inv * (qt - ql)),np.multiply(epsi_inv, np.multiply(p0, (qt - ql)))))
        ztop = len(z) - 2  # LES and SCM models dont stretch mto the same  height in deep convection
        qv = qt - ql
        CT[np.where(CT < 0.0)] = 0.0
        CF[np.where(CF == np.max(z)-10.0)] = 0.0

        Theta_s = np.mean(thetal[t1:, :], 0)
        T_s     = np.mean(temperature[t1:, :], 0)
        RH_s    = np.mean(RH[t1:, :], 0)
        qt_s    = np.mean(qt[t1:, :], 0)
        ql_s    = np.mean(ql[t1:, :], 0)
        b_s     = np.mean(buoyancy[t1:, :], 0)
        s_CT_temp = np.multiply(CT, 0.0)
        for tt in range(0, len(t)):
            s_CT_temp[tt] = np.interp(CT[tt], z, T_s)

        G = np.linalg.norm(np.diag([lwp, CF, CT]))
        return G

    def MCMC_paramlist(self, theta, case_name):

        paramlist = {}
        paramlist['meta'] = {}
        paramlist['meta']['casename'] = case_name
        paramlist['meta']['simname'] = case_name

        paramlist['turbulence'] = {}
        paramlist['turbulence']['prandtl_number'] = 1.0
        paramlist['turbulence']['Ri_bulk_crit'] = 0.2

        paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
        paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
        paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.16
        paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.35
        paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 9.9
        paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.03* theta
        paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 3.0
        paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor'] = 0.05
        paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_erf_const'] = 0.5
        paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0/3.0
        paramlist['turbulence']['EDMF_PrognosticTKE']['aspect_ratio'] = 0.25
        paramlist['turbulence']['updraft_microphysics'] = {}
        paramlist['turbulence']['updraft_microphysics']['max_supersaturation'] = 0.1
        return paramlist


    def write_file(self, paramlist, folder):
        fh = open(folder + "/paramlist_"+paramlist['meta']['casename']+ ".in", 'w')
        json.dump(paramlist, fh, sort_keys=True, indent=4)
        fh.close()

        return
