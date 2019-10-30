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
import shutil
import string
import random

# from python command line:
# import run_scampy
# instance = run_scampy.run_scampy('Bomex')
# instance.run(0.1)


np.set_printoptions(precision=3, suppress=True)
np.random.seed(2019)

class forward_scampy(object):

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
        #initiate_record(output_filename, theta)
        # compile the SCM
        subprocess.call("CC=mpicc python setup.py build_ext --inplace",  shell=True, cwd=myscampyfolder)
        # generate namelist and paramlist
        subprocess.call("python generate_namelist.py " + self.case_name,  shell=True, cwd=myscampyfolder)
        # subprocess.call("python generate_paramlist.py " + case_name,  shell=True, cwd=myscampyfolder)
        # edit namelist and paramlist (add theta)
        letters = string.ascii_lowercase
        a=random.choice(letters)
        b=random.choice(letters)

        file_case = open(myscampyfolder +"/"+ self.case_name + '.in').read()
        namelist = json.loads(file_case)
        uuid0 = namelist['meta']['uuid']
        uuid =uuid0[0:-2] + a+b
        namelist['meta']['uuid'] = uuid
        namelist['meta']['simname'] = self.case_name +'_'+ a+b
        new_path = myscampyfolder  + '/Output.' + self.case_name+'_'+a+b+ '.' +  uuid[-5:] + '/stats/Stats.' + self.case_name+'_'+a+b+'.nc'
        print(new_path)
        paramlist = self.MCMC_paramlist(theta, self.case_name)
        self.write_file(paramlist, myscampyfolder)
        self.write_namefile(namelist, myscampyfolder, a, b)
        # run scampy with theta in paramlist
        print("=========================== run scampy with uuid", uuid[-5:])
        print("python main.py " + self.case_name+"_"+a+b+".in " + "paramlist_" + self.case_name + ".in")
        subprocess.call("python main.py " + self.case_name+"_"+a+b+".in " + "paramlist_" + self.case_name + ".in", shell=True, cwd=myscampyfolder)
        # load NC of the new scampy data
        os.remove(myscampyfolder  + '/'+ self.case_name+'_'+a+b+ '.in')
        print(new_path)
        new_data = nc.Dataset(new_path, 'r')
        # calculate the cost fun u, print values
        G = self.compute_G(theta, new_data)
        # store theta and u
        # create_record(theta, G, new_data, output_filename)
        # remove the simualtion data
        shutil.rmtree(myscampyfolder  + '/Output.' + self.case_name+'_'+a+b + '.' +  uuid[-5:] + '/')
        print("=========================== done", G)
        return G

    def compute_G(self, theta, new_data):
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
        # s_CT_temp = np.multiply(CT, 0.0)
        # for tt in range(0, len(t)):
        #     s_CT_temp[tt] = np.interp(CT[tt], z, T_s)
        A = np.mean(lwp)
        B = np.mean(CF)
        C = np.mean(CT)

        # G = np.diag([A, B, C]).flatten() if A B and C are vectors or matrixes use flatten
        G = np.array([A, B, C]) # is A B and C are scalars
        print(A, B, C)
        return G

    def MCMC_paramlist(self, theta, case_name):

        theta_used = theta[0]

        paramlist = {}
        paramlist['meta'] = {}
        paramlist['meta']['casename'] = case_name
        paramlist['meta']['simname'] = case_name

        paramlist['turbulence'] = {}
        paramlist['turbulence']['prandtl_number_0'] = 1.0
        paramlist['turbulence']['Ri_bulk_crit'] = 0.2

        paramlist['turbulence']['EDMF_PrognosticTKE'] = {}
        paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.1
        paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff'] = 0.18
        paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff'] = 0.26
        paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor'] = 9.9
        paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.03# * theta_used
        # print(type(0.03 * theta_used))
        paramlist['turbulence']['EDMF_PrognosticTKE']['sorting_factor'] = 4.0
        paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor'] = 0.05
        paramlist['turbulence']['EDMF_PrognosticTKE']['sorting_power'] = 2.0
        paramlist['turbulence']['EDMF_PrognosticTKE']['aspect_ratio'] = 0.25
        # This constant_plume_spacing corresponds to plume_spacing/alpha_d in the Tan et al paper,
        #with values plume_spacing=500.0, alpha_d = 0.375
        paramlist['turbulence']['EDMF_PrognosticTKE']['constant_plume_spacing'] = 1333.0

        # TODO: merge the tan18 buoyancy forluma into normalmode formula -> simply set buoy_coeff1 as 1./3. and buoy_coeff2 as 0.
        paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff'] = 1.0/3.0

        paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_buoy_coeff1'] = 1.0/3.0
        paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_buoy_coeff2'] = 0.0
        paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_adv_coeff'] = 0.75
        paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_drag_coeff'] = 1.0
        return paramlist


    def write_file(self, paramlist, folder):
        fh = open(folder + "/paramlist_"+paramlist['meta']['casename']+ ".in", 'w')
        json.dump(paramlist, fh, sort_keys=True, indent=4)
        fh.close()

        return

    def write_namefile(self, namelist, folder, a, b):
        fh = open(folder + "/"+namelist['meta']['casename']+"_"+a+b+".in", 'w')
        json.dump(namelist, fh, sort_keys=True, indent=4)
        fh.close()

        return
