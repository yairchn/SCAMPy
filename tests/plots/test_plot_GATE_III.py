import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset

import pytest
import pprint as pp
import numpy as np

import main as scampy
import common as cmn
import plot_scripts as pls

@pytest.fixture(scope="module")
def sim_data(request):

    # generate namelists and paramlists
    setup = cmn.simulation_setup('GATE_III')

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

@pytest.mark.skip(reason="GATE simulation hangs up")
def test_plot_GATE_III(sim_data):
    """
    plot GATE_III profiles
    """
    data_to_plot = cmn.read_data_avg(sim_data, n_steps=100)

    pls.plot_mean(data_to_plot,   "GATE_III_quicklook.pdf")
    pls.plot_drafts(data_to_plot, "GATE_III_quicklook_drafts.pdf")

@pytest.mark.skip(reason="GATE simulation hangs up")
def test_plot_timeseries_GATE_III(sim_data):
    """
    plot timeseries
    """
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data = Dataset('/Users/yaircohen/Documents/codes/scampy/tests/les_data/GATE_III.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_timeseries(data_to_plot, "GATE_III")
    pls.plot_mean(data_to_plot, les_data_to_plot,22,24,            "GATE_III_quicklook.pdf")
    pls.plot_closures(data_to_plot, les_data_to_plot,22,24,        "GATE_III_closures.pdf")
    pls.plot_drafts(data_to_plot, les_data_to_plot,22,24,          "GATE_III_quicklook_drafts.pdf")
    pls.plot_velocities(data_to_plot, les_data_to_plot,22,24,      "GATE_III_velocities.pdf")
    pls.plot_main(data_to_plot, les_data_to_plot,22,24,           "GATE_III_main.pdf")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 22,24, "GATE_III_var_covar_mean.pdf")
    pls.plot_var_covar_components(data_to_plot,22,24,              "GATE_III_var_covar_components.pdf")


@pytest.mark.skip(reason="GATE simulation hangs up")
def test_plot_timeseries_1D_GATE_III(sim_data):
    """
    plot GATE_III 1D timeseries
    """
    les_data = Dataset('/Users/yaircohen/Documents/codes/scampy/tests/les_data/GATE_III.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)

    pls.plot_timeseries_1D(data_to_plot, "GATE_III_timeseries_1D.pdf")