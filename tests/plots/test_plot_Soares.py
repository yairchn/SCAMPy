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


def test_plot_timeseries_Soares(sim_data):
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
    les_data = Dataset(localpath + '/les_data/Soares.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_closures(data_to_plot, les_data_to_plot,7,8,           "Soares_closures.pdf",           folder="plots/output/Soares/")
    pls.plot_humidities(data_to_plot, les_data_to_plot,7,8,         "Soares_humidities.pdf",         folder="plots/output/Soares/")
    pls.plot_updraft_properties(data_to_plot, les_data_to_plot,7,8, "Soares_updraft_properties.pdf", folder="plots/output/Soares/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 7,8,    "Soares_tke_components.pdf",     folder="plots/output/Soares/")

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/Soares/all_variables/")
    pls.plot_mean(data_to_plot, les_data_to_plot,7,8,            folder="plots/output/Soares/all_variables/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 7,8, "Soares_var_covar_mean.pdf", folder="plots/output/Soares/all_variables/")
    pls.plot_var_covar_components(data_to_plot,7,8,              "Soares_var_covar_components.pdf", folder="plots/output/Soares/all_variables/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 7,8,  "Soares_tke_breakdown.pdf", folder="plots/output/Soares/all_variables/")

def test_plot_timeseries_1D_Soares(sim_data):
    """
    plot Soares 1D timeseries
    """
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Soares/")
        print()
    except:
        print('Soares folder exists')
    try:
        os.mkdir(localpath + "/plots/output/Soares/all_variables/")
    except:
        print('Soares/all_variables folder exists')
    les_data = Dataset(localpath + '/les_data/Soares.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)
    data_to_plot_ = cmn.read_data_srs(sim_data)
    les_data_to_plot_ = cmn.read_les_data_srs(les_data)

    pls.plot_main_timeseries(data_to_plot, les_data_to_plot, data_to_plot_, les_data_to_plot_,"Soares_main_timeseries.pdf", folder="plots/output/Soares/")
    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot,  folder="plots/output/Soares/all_variables/")
