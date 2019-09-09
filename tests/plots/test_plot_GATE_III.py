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
def test_plot_timeseries_GATE_III(sim_data):
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
    les_data = Dataset(localpath + '/les_data/GATE_III.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_closures(data_to_plot, les_data_to_plot,22,24,           "GATE_III_closures.pdf",           folder="plots/output/GATE_III/")
    pls.plot_humidities(data_to_plot, les_data_to_plot,22,24,         "GATE_III_humidities.pdf",         folder="plots/output/GATE_III/")
    pls.plot_updraft_properties(data_to_plot, les_data_to_plot,22,24, "GATE_III_updraft_properties.pdf", folder="plots/output/GATE_III/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 22,24,    "GATE_III_tke_components.pdf",     folder="plots/output/GATE_III/")

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/GATE_III/all_variables/")
    pls.plot_mean(data_to_plot, les_data_to_plot,22,24,            folder="plots/output/GATE_III/all_variables/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 22,24, "GATE_III_var_covar_mean.pdf", folder="plots/output/GATE_III/all_variables/")
    pls.plot_var_covar_components(data_to_plot,22,24,              "GATE_III_var_covar_components.pdf", folder="plots/output/GATE_III/all_variables/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 22,24,  "GATE_III_tke_breakdown.pdf", folder="plots/output/GATE_III/all_variables/")

@pytest.mark.skip(reason="GATE not working yet")
def test_plot_timeseries_1D_GATE_III(sim_data):
    """
    plot GATE_III 1D timeseries
    """
    localpath = os.getcwd()
    try:
        os.mkdir(localpath + "/plots/output/GATE_III/")
        print()
    except:
        print('GATE_III folder exists')
    try:
        os.mkdir(localpath + "/plots/output/GATE_III/all_variables/")
    except:
        print('GATE_III/all_variables folder exists')
    les_data = Dataset(localpath + '/les_data/GATE_III.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)
    data_to_plot_ = cmn.read_data_srs(sim_data)
    les_data_to_plot_ = cmn.read_les_data_srs(les_data)

    pls.plot_main_timeseries(data_to_plot, les_data_to_plot, data_to_plot_, les_data_to_plot_, "GATE_III_main_timeseries.pdf", folder="plots/output/GATE_III/")
    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot,  folder="plots/output/GATE_III/all_variables/")
