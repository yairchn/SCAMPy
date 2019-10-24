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
    setup = cmn.simulation_setup('Bomex')
    # change the defaults
    #setup['namelist']['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_Bomex(sim_data):
    """
    plot Bomex timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Bomex/")
    except:
        print('Bomex folder exists')
    try:
        os.mkdir(localpath + "/plots/output/Bomex/all_variables/")
    except:
        print('Bomex/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/Bomex.nc")):
        les_data = Dataset(localpath + "/les_data/Bomex.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/zrhxou8i80bfdk2/Bomex.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/Bomex.nc "+url_)
        les_data = Dataset(localpath + "/les_data/Bomex.nc", 'r')

    f1 = "plots/output/Bomex/"
    f2 = f1 + "all_variables/"
    cn = "Bomex_"
    t0 = 5
    t1 = 6
    cb_min = [0., 0.]
    cb_max = [0.021, 4.8]
    fixed_cbar = True
    cb_min_t = [298, 298, 298.5, 0, 0, 0, -1, -1, -1, 2, 2, 12.5,\
                -0.15, 0, -9, -2,\
                 0, -0.075, 0,\
                -0.09, -0.015, -0.08,\
                 0., -0.16, -1e-5]
    cb_max_t = [312, 312, 301, 0.02, 0.008, 2, 1, 1, 1, 18, 18, 18,\
                0, 4, -4, 1,\
                0.24, 0.035, .5,\
                0.015, 0.075, 0.04,\
                0.16, 0.02, 0.0002]

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
    pls.plot_tke_break(scm_dict, les_dict, t0, t1,  cn+"tke_breakdown.pdf", folder=f2)

    pls.plot_contour_t(scm_dict, les_dict, fixed_cbar, cb_min_t, cb_max_t, folder=f2)
    pls.plot_mean_prof(scm_dict, les_dict, t0, t1, folder=f2)

    pls.plot_main(scm_dict_t, les_dict_t, scm_dict, les_dict,
                  cn+"main_timeseries.pdf", cb_min, cb_max, folder=f1)

    pls.plot_1D(scm_dict_t, les_dict_t, cn, folder=f2)
