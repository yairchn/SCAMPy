import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import numpy as np

import main as scampy
import common as cmn
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    cmn.removing_files
    setup = cmn.simulation_setup('ARM_SGP')

    # change the defaults if needed
    # setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_ARM_SGP(sim_data):
    """
    plot ARM_SGP timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/ARM_SGP/")
    except:
        print('ARM_SGP folder exists')
    try:
        os.mkdir(localpath + "/plots/output/ARM_SGP/all_variables/")
    except:
        print('ARM_SGP/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/ARM_SGP.nc")):
        les_data = Dataset(localpath + "/les_data/ARM_SGP.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/j7s0jmmkwn7av62/ARM_SGP.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/ARM_SGP.nc "+url_)
        les_data = Dataset(localpath + "/les_data/ARM_SGP.nc", 'r')

    f1 = "plots/output/ARM_SGP/"
    f2 = f1 + "all_variables/"
    cn = "ARM_SGP_"
    t0 = 8
    t1 = 11
    cb_min = [0., 0.]
    cb_max = [0.05, 7]
    fixed_cbar = True
    cb_min_t = [295, 295, 300, 0, 0, 0, -1, -1, -1, 2.5, 2.5, 9,\
                -0.32, 0, 9, -0.1,\
                 0, -0.1, 0,\
                -0.25, -0.12, -0.08,\
                -0.0, -0.44, -0.045]
    cb_max_t = [335, 335, 312, 0.05, 0.016, 4, 1, 1, 1, 20, 20, 20,\
                0, 7, 11, 0.1,\
                0.28, 0.03, 1.05,\
                0.12, 0.25, 0.18,\
                0.44, 0.0, 0.25]

    scm_dict = cmn.read_scm_data(sim_data)
    les_dict = cmn.read_les_data(les_data)

    scm_dict_t = cmn.read_scm_data_timeseries(sim_data)
    les_dict_t = cmn.read_les_data_timeseries(les_data)

    pls.plot_closures(scm_dict, les_dict, t0, t1, cn+"closures.pdf", folder=f1)
    pls.plot_spec_hum(scm_dict, les_dict, t0, t1, cn+"humidities.pdf", folder=f1)
    pls.plot_upd_prop(scm_dict, les_dict, t0, t1, cn+"updraft_properties.pdf", folder=f1)
    pls.plot_tke_comp(scm_dict, les_dict, t0, t1, cn+"tke_components.pdf", folder=f1)

    pls.plot_cvar_mean(scm_dict, les_dict, t0, t1, cn+"var_covar_mean.pdf", folder=f2)
    pls.plot_cvar_comp(scm_dict, t0, t1, cn+"var_covar_components.pdf", folder=f2)
    pls.plot_tke_break(scm_dict, les_dict, t0, t1, cn+"tke_breakdown.pdf",folder=f2)

    pls.plot_contour_t(scm_dict, les_dict, fixed_cbar, cb_min_t, cb_max_t, folder=f2)
    pls.plot_mean_prof(scm_dict, les_dict, t0, t1, folder=f2)

    pls.plot_main(scm_dict_t, les_dict_t, scm_dict, les_dict,
                  cn+"main_timeseries.pdf", cb_min, cb_max, folder=f1)

    pls.plot_1D(scm_dict_t, les_dict_t, cn, folder=f2)
