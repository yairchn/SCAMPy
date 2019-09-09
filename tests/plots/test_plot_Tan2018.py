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
    setup = cmn.simulation_setup('life_cycle_Tan2018')

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data


@pytest.mark.skip(reason="need to run new LES with tracers")
def test_plot_timeseries_Tan2018(sim_data):
    """
    plot Tan2018 timeseries
    """
    # make directory
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Tan2018/")
    except:
        print('Tan2018 folder exists')
    try:
        os.mkdir(localpath + "/plots/output/Tan2018/all_variables/")
    except:
        print('Tan2018/all_variables folder exists')
    les_data = Dataset(localpath + '/les_data/Tan2018.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_closures(data_to_plot, les_data_to_plot,5,6,           "Tan2018_closures.pdf",           folder="plots/output/Tan2018/")
    pls.plot_humidities(data_to_plot, les_data_to_plot,5,6,         "Tan2018_humidities.pdf",         folder="plots/output/Tan2018/")
    pls.plot_updraft_properties(data_to_plot, les_data_to_plot,5,6, "Tan2018_updraft_properties.pdf", folder="plots/output/Tan2018/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 5,6,    "Tan2018_tke_components.pdf",     folder="plots/output/Tan2018/")

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/Tan2018/all_variables/")
    pls.plot_mean(data_to_plot, les_data_to_plot,5,6,            folder="plots/output/Tan2018/all_variables/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 5,6, "Tan2018_var_covar_mean.pdf", folder="plots/output/Tan2018/all_variables/")
    pls.plot_var_covar_components(data_to_plot,5,6,              "Tan2018_var_covar_components.pdf", folder="plots/output/Tan2018/all_variables/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 5,6,  "Tan2018_tke_breakdown.pdf", folder="plots/output/Tan2018/all_variables/")

@pytest.mark.skip(reason="need to run new LES with tracers")
def test_plot_timeseries_1D_Tan2018(sim_data):
    """
    plot Tan2018 1D timeseries
    """
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/Tan2018/")
        print()
    except:
        print('Tan2018 folder exists')
    try:
        os.mkdir(localpath + "/plots/output/Tan2018/all_variables/")
    except:
        print('Tan2018/all_variables folder exists')
    les_data = Dataset(localpath + '/les_data/Tan2018.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)
    data_to_plot_ = cmn.read_data_srs(sim_data)
    les_data_to_plot_ = cmn.read_les_data_srs(les_data)

    pls.plot_main_timeseries(data_to_plot, les_data_to_plot, data_to_plot_, les_data_to_plot_, "Tan2018_main_timeseries.pdf",folder="plots/output/Tan2018/")
    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot,  folder="plots/output/Tan2018/all_variables/")
