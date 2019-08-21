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
    setup = cmn.simulation_setup('TRMM_LBA')

    # run scampy
    subprocess.call("python setup.py build_ext --inplace", shell=True, cwd='../')
    scampy.main1d(setup["namelist"], setup["paramlist"])

    # simulation results
    sim_data = Dataset(setup["outfile"], 'r')

    # remove netcdf file after tests
    request.addfinalizer(cmn.removing_files)

    return sim_data

# @pytest.mark.skip(reason="deep convection not working with current defaults")
def test_plot_timeseries_TRMM_LBA(sim_data):
    """
    plot TRMM_LBA timeseries
    """
    # make directory
    try:
        os.mkdir("/Users/yaircohen/Documents/codes/scampy/tests/plots/output/TRMM_LBA/")
    except:
        print('TRMM_LBA folder exists')
    les_data = Dataset('/Users/yaircohen/Documents/codes/scampy/tests/les_data/TRMM_LBA.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/TRMM_LBA/")
    pls.plot_mean(data_to_plot, les_data_to_plot,5,6,            folder="plots/output/TRMM_LBA/")
    pls.plot_closures(data_to_plot, les_data_to_plot,5,6,        "TRMM_LBA_closures.pdf", folder="plots/output/TRMM_LBA/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 5,6, "TRMM_LBA_var_covar_mean.pdf", folder="plots/output/TRMM_LBA/")
    pls.plot_var_covar_components(data_to_plot,5,6,              "TRMM_LBA_var_covar_components.pdf", folder="plots/output/TRMM_LBA/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 5,6, "TRMM_LBA_tke_components.pdf", folder="plots/output/TRMM_LBA/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 5,6,  "TRMM_LBA_tke_breakdown.pdf", folder="plots/output/TRMM_LBA/")

def test_plot_timeseries_1D_TRMM_LBA(sim_data):
    """
    plot TRMM_LBA 1D timeseries
    """
    try:
        os.mkdir("/Users/yaircohen/Documents/codes/scampy/tests/plots/output/TRMM_LBA/")
    except:
        print('TRMM_LBA folder exists')
    les_data = Dataset('/Users/yaircohen/Documents/codes/scampy/tests/les_data/TRMM_LBA.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)

    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot, folder="plots/output/TRMM_LBA/")