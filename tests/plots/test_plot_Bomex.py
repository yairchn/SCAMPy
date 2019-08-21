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
    try:
        os.mkdir("/Users/yaircohen/Documents/codes/scampy/tests/plots/output/Bomex/")
    except:
        print('Bomex folder exists')
    les_data = Dataset('/Users/yaircohen/Documents/codes/scampy/tests/les_data/Bomex.nc', 'r')
    data_to_plot = cmn.read_data_srs(sim_data)
    les_data_to_plot = cmn.read_les_data_srs(les_data)

    pls.plot_timeseries(data_to_plot, les_data_to_plot,          folder="plots/output/Bomex/")
    pls.plot_mean(data_to_plot, les_data_to_plot,5,6,            folder="plots/output/Bomex/")
    pls.plot_closures(data_to_plot, les_data_to_plot,5,6,        "Bomex_closures.pdf", folder="plots/output/Bomex/")
    pls.plot_var_covar_mean(data_to_plot, les_data_to_plot, 5,6, "Bomex_var_covar_mean.pdf", folder="plots/output/Bomex/")
    pls.plot_var_covar_components(data_to_plot,5,6,              "Bomex_var_covar_components.pdf", folder="plots/output/Bomex/")
    pls.plot_tke_components(data_to_plot, les_data_to_plot, 5,6, "Bomex_tke_components.pdf", folder="plots/output/Bomex/")
    pls.plot_tke_breakdown(data_to_plot, les_data_to_plot, 5,6,  "Bomex_tke_breakdown.pdf", folder="plots/output/Bomex/")

def test_plot_timeseries_1D_Bomex(sim_data):
    """
    plot Bomex 1D timeseries
    """
    try:
        os.mkdir("/Users/yaircohen/Documents/codes/scampy/tests/plots/output/Bomex/")
    except:
        print('Bomex folder exists')
    les_data = Dataset('/Users/yaircohen/Documents/codes/scampy/tests/les_data/Bomex.nc', 'r')
    data_to_plot = cmn.read_data_timeseries(sim_data)
    les_data_to_plot = cmn.read_les_data_timeseries(les_data)

    pls.plot_timeseries_1D(data_to_plot,  les_data_to_plot, folder="plots/output/Bomex/")
