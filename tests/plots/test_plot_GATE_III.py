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
    setup = cmn.simulation_setup('GATE_III')

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

@pytest.mark.skip(reason="GATE not working yet")
def test_plot_GATE_III(sim_data):
    """
    plot GATE_III timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/GATE_III/")
    except:
        print('GATE_III folder exists')
    try:
        os.mkdir(localpath + "/plots/output/GATE_III/all_variables/")
    except:
        print('GATE_III/all_variables folder exists')

    if (os.path.exists(localpath + "/les_data/GATE_III.nc")):
        les_data = Dataset(localpath + "/les_data/GATE_III.nc", 'r')
    else:
        url_ = "https://www.dropbox.com/s/snhxbzxt4btgiis/TRMM_LBA.nc?dl=0"
        os.system("wget -O "+localpath+"/les_data/TRMM_LBA.nc "+url_)
        les_data = Dataset(localpath + "/les_data/GATE_III.nc", 'r')

    f1 = "plots/output/GATE_III/"
    f2 = f1 + "all_variables/"
    cn = "GATE_III_"
    t0 = 22
    t1 = 24
    cb_min = [0, 0] #TODO
    cb_max = [1, 1] #TODO
    fixed_cbar = True
    cb_min_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                0, 0, 0, 0,\
                0, 0, 0,\
                0, 0, 0,\
                0, 0, 0]#TODO
    cb_max_t = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\
                1, 1, 1, 1,\
                1, 1, 1,\
                1, 1, 1,\
                1, 1, 1]#TODO

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
