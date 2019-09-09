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
    setup = cmn.simulation_setup('life_cycle_Tan2018')

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

def test_plot_Tan2018(sim_data):
    """
    plot Tan2018 profiles
    """
    data_to_plot = cmn.read_data_avg(sim_data, n_steps=100)

    pls.plot_mean(data_to_plot,   "Tan2018_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "Tan2018_quicklook_drafts.pdf")

def test_plot_timeseries_Tan2018(sim_data):
    """
    plot Tan2018 timeseries
    """
    data_to_plot = cmn.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "Tan2018")

def test_plot_timeseries_1D_Tan2018(sim_data):
    """
    plot Tan2018 1D timeseries
    """
    data_to_plot = cmn.read_data_timeseries(sim_data)

    pls.plot_timeseries_1D(data_to_plot, "Tan2018_timeseries_1D.pdf")

def test_plot_var_covar_Tan2018(sim_data):
    """
    plot Tan2018 var covar
    """
    data_to_plot = cmn.read_data_avg(sim_data, n_steps=100, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot,       "Tan2018_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot, "Tan2018_var_covar_components.pdf")
