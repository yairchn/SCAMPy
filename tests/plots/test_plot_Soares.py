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
import subprocess
import main as scampy
import common as cmn
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    cmn.removing_files
    setup = cmn.simulation_setup('Soares')

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data


def test_plot_Soares(sim_data):
    """
    plot Soares timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Soares/")
    except:
        print('Soares folder exists')
    try:
        os.mkdir(localpath + "/plots/output/Soares/all_variables/")
    except:
        print('Soares/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/Soares.nc")):
        les_data = Dataset(localpath + "/les_data/Soares.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/wkfy1mcbbo9iyx7/Soares.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/Soares.nc "+url_)
        les_data = Dataset(localpath + "/les_data/Soares.nc", 'r')

    f1 = "plots/output/Soares/"
    f2 = f1 + "all_variables/"
    cn = "Soares_"
    t0 = 7
    t1 = 8
    cb_min = [0., 0.]
    cb_max = [0.01, 2.5]
    fixed_cbar = True
    cb_min_t = [300, 300, 300, 0, 0, 0, 0, 0, 0, 2, 2, 4.5,\
                -0.4, 0, -0.7, -0.45,\
                0, -0.03, 0,\
                -0.045, -0.06, 0,\
                0, 0, -0.04]
    cb_max_t = [305, 305, 302, 1, 1, 1, 1, 1, 1, 5, 5, 5,\
                0, 2.4, 0.7, 0.45,\
                0.28, 0.01, 1.6,\
                0.06, 0.045, 0.1,\
                0.01, 0.1, 0.08]

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
