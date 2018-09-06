import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./tests")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import pprint as pp
import numpy as np

import main as scampy
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    setup = pls.simulation_setup('DYCOMS_RF01')
    # chenge the defaults
    setup["namelist"]['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var'] = True

    # run scampy
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf files after tests
    request.addfinalizer(pls.removing_files)

    return sim_data

def test_plot_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 quicklook profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100)

    pls.plot_mean(data_to_plot,   "DYCOMS_RF01_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "DYCOMS_RF01_quicklook_drafts.pdf")

def test_plot_var_covar_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 quicklook profiles
    """
    data_to_plot = pls.read_data_avg(sim_data, n_steps=100, var_covar=True)

    pls.plot_var_covar_mean(data_to_plot,       "DYCOMS_RF01_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot, "DYCOMS_RF01_var_covar_components.pdf")

def test_plot_timeseries_DYCOMS(sim_data):
    """
    plot timeseries
    """
    data_to_plot = pls.read_data_srs(sim_data)

    pls.plot_timeseries(data_to_plot, "DYCOMS")

def test_plot_timeseries_1D_DYCOMS_RF01(sim_data):
    """
    plot DYCOMS_RF01 1D timeseries
    """
    data_to_plot = pls.read_data_timeseries(sim_data)

    pls.plot_timeseries_1D(data_to_plot, "DYCOMS_RF01_timeseries_1D.pdf")

