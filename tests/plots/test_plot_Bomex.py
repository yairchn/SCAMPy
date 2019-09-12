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

def test_plot_timeseries_Bomex(sim_data):
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

    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_closures(data_to_plot, les_data_to_plot,5,6,           "Bomex_closures.pdf",           folder="plots/output/Bomex/")
    pls.plot_humidities(data_to_plot, les_data_to_plot,5,6,         "Bomex_humidities.pdf",         folder="plots/output/Bomex/")
    pls.plot_updraft_properties(data_to_plot, les_data_to_plot,5,6, "Bomex_updraft_properties.pdf", folder="plots/output/Bomex/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 5,6,    "Bomex_tke_components.pdf",     folder="plots/output/Bomex/")

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/Bomex/all_variables/")
    pls.plot_mean(data_to_plot, les_data_to_plot,5,6,            folder="plots/output/Bomex/all_variables/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 5,6, "Bomex_var_covar_mean.pdf", folder="plots/output/Bomex/all_variables/")
    pls.plot_var_covar_components(data_to_plot,5,6,              "Bomex_var_covar_components.pdf", folder="plots/output/Bomex/all_variables/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 5,6,  "Bomex_tke_breakdown.pdf", folder="plots/output/Bomex/all_variables/")

def test_plot_timeseries_1D_Bomex(sim_data):
    """
    plot Bomex 1D timeseries
    """
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Bomex/")
        print()
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

    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)
    data_to_plot_ = cmn.read_data_srs(sim_data)
    les_data_to_plot_ = cmn.read_les_data_srs(les_data)

    pls.plot_main_timeseries(data_to_plot, les_data_to_plot, data_to_plot_, les_data_to_plot_, "Bomex_main_timeseries.pdf",folder="plots/output/Bomex/")
    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot,  folder="plots/output/Bomex/all_variables/")
