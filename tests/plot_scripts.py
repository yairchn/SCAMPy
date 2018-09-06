import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./tests")

import os
import subprocess
import json
import warnings

from netCDF4 import Dataset
import numpy as np
import pprint as pp

import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import ticker

def simulation_setup(case):
    """
    generate namelist and paramlist files for scampy
    choose the name of the output folder
    """
    # Filter annoying Cython warnings that serve no good purpose.
    # see https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    # simulation related parameters
    os.system("python generate_namelist.py " + case)
    file_case = open(case + '.in').read()
    # turbulence related parameters
    os.system("python generate_paramlist.py " +  case)
    file_params = open('paramlist_' + case + '.in').read()

    namelist  = json.loads(file_case)
    paramlist = json.loads(file_params)

    namelist['output']['output_root'] = "./Tests."
    namelist['meta']['uuid'] = case
    # TODO - copied from NetCDFIO
    # ugly way to know the name of the folder where the data is saved
    uuid = str(namelist['meta']['uuid'])
    outpath = str(
        os.path.join(
            namelist['output']['output_root'] +
            'Output.' +
            namelist['meta']['simname'] +
            '.' +
            uuid[len(uuid )-5:len(uuid)]
        )
    )
    outfile = outpath + "/stats/Stats." + case + ".nc"

    res = {"namelist"  : namelist,
           "paramlist" : paramlist,
           "outfile"   : outfile}
    return res


def removing_files():
    """
    remove the folder with netcdf files from tests
    """
    #TODO - think of something better
    cmd = "rm -r Tests.Output.*"
    subprocess.call(cmd , shell=True)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Taken from https://gist.github.com/jakevdp/91077b0cae40f8f8244a

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def read_data_avg(sim_data, n_steps=0, var_covar=False):
    """
    Read in the data from netcdf file into a dictionary that can be used for plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    n_steps   - number of timesteps to average over
    var_covar - flag for when we also want to read in the variance and covariance fields
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "env_qr", "updraft_qr", "updraft_w", "env_w"]
    variables_var = [\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain"
                ]
    if var_covar:
        variables.extend(variables_var)

    # read the data from t=init and t=end
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:])}

    time = [0, -1]
    for var in variables:
        data_to_plot[var] = []
        for it in range(2):
            if ("buoyancy" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 10000) #cm2/s3
            elif ("qt" in var or "ql" in var or "qr" in var):
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 1000)  #g/kg
            elif ("p0" in var):
                data_to_plot[var].append(np.array(sim_data["reference/" + var][time[it], :]) * 100)  #hPa
            else:
                data_to_plot[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # add averaging over last n_steps timesteps
    if(n_steps > 0):
        for var in variables:
            for time_it in range(-2, -1*n_steps-1, -1):
                if ("buoyancy" in var):
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 10000  #cm2/s3
                elif ("qt" in var or "ql" in var or "qr" in var):
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :]) * 1000   #g/kg
                elif ("p0" in var):
                    data_to_plot[var][1] += np.array(sim_data["reference/" + var][time_it, :]) * 100   #hPa
                else:
                    data_to_plot[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

            data_to_plot[var][1] /= n_steps

    return data_to_plot


def read_rad_data_avg(sim_data, n_steps=0):
    """
    Read in the radiation forcing data from netcdf files into a dictionary that can be used for plots
    (so far its only applicable to the DYCOMS radiative forcing)

    Input:
    sim_data - netcdf Dataset with simulation results
    n_steps  - number of timesteps to average over
    """
    variables = ["rad_flux", "rad_dTdt"]

    time = [0, -1]
    rad_data = {"z" : np.array(sim_data["profiles/z"][:])}
    for var in variables:
        rad_data[var] = []
        for it in range(2):
            if ("rad_dTdt" in var):
                rad_data[var].append(np.array(sim_data["profiles/" + var][time[it], :]) * 60 * 60 * 24) # K/day
            else:
                rad_data[var].append(np.array(sim_data["profiles/" + var][time[it], :]))

    # add averaging over last n_steps timesteps
    if(n_steps > 0):
        for var in variables:
            for time_it in range(-2, -1*n_steps-1, -1):
                if ("rad_dTdt" in var):
                    rad_data[var][1] += np.array(sim_data["profiles/" + var][time_it, :] * 60 * 60 * 24) # K/day
                else:
                    rad_data[var][1] += np.array(sim_data["profiles/" + var][time_it, :])

            rad_data[var][1] /= n_steps

    return rad_data


def read_data_srs(sim_data, var_covar=False):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries of profiles plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    var_covar - flag for when we also want to read in the variance and covariance fields
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql", "updraft_thetal",\
                 "env_qr", "updraft_qr", "updraft_w", "env_w", "env_thetal",\
                 "massflux_h", "diffusive_flux_h", "total_flux_h",\
                 "massflux_qt","diffusive_flux_qt","total_flux_qt",\
                 "eddy_viscosity", "eddy_diffusivity", "mixing_length",\
                 "entrainment_sc", "detrainment_sc", "massflux"\
                ]
    variables_var = [\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain"\
                ]
    if var_covar:
        variables.extend(variables_var)

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data_to_plot[var] = []
        if ("buoyancy" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 10000 #cm2/s3
        elif ("qt" in var or "ql" in var or "qr" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 1000  #g/kg
        elif ("p0" in var):
            data_to_plot[var] = np.transpose(np.array(sim_data["reference/" + var][:, :])) * 100   #hPa
        else:
            data_to_plot[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :]))

    return data_to_plot


def read_data_timeseries(sim_data):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries plots

    Input:
    sim_data - netcdf Dataset with simulation results
    """
    variables = ["updraft_cloud_cover", "updraft_cloud_base", "updraft_cloud_top", "lwp",\
                 "ustar", "shf", "lhf", "Tsurface"]

    # read the data
    data_to_plot = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data_to_plot[var] = []
        data_to_plot[var] = np.array(sim_data["timeseries/" + var][:])

    return data_to_plot


def plot_mean(data, title, folder="tests/output/"):
    """
    Plots mean profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # read data
    qv_mean = np.array(data["qt_mean"]) - np.array(data["ql_mean"])

    # data to plot
    x_lab  = ['QV [g/kg]', 'QL [g/kg]',      'QR [g/kg]',      'THL [K]',           'buoyancy [cm2/s3]',   'TKE [m2/s2]']
    plot_x = [qv_mean,      data["ql_mean"],  data["qr_mean"], data["thetal_mean"],  data["buoyancy_mean"], data["tke_mean"]]
    color  = ["palegreen", "forestgreen"]
    label  = ["ini", "end"]

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        for it in range(2): #init, end
            plots[plot_it].plot(plot_x[plot_it][it], data["z_half"], '.-', color=color[it], label=label[it])

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_drafts(data, title, folder="tests/output/"):
    """
    Plots updraft and environment profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 18})
    mpl.rc('lines', linewidth=4, markersize=10)

    # read data
    qv_mean    = np.array(data["qt_mean"])    - np.array(data["ql_mean"])
    env_qv     = np.array(data["env_qt"])     - np.array(data["env_ql"])
    updraft_qv = np.array(data["updraft_qt"]) - np.array(data["updraft_ql"])

    # data to plot
    x_lab    = ["QV [g/kg]", "QL [g/kg]",        "QR [g/kg]",        "w [m/s]",         "updraft buoyancy [cm2/s3]",  "updraft area [%]"]
    plot_upd = [qv_mean,     data["updraft_ql"], data["updraft_qr"], data["updraft_w"], data["updraft_buoyancy"],     data["updraft_area"]]
    plot_env = [env_qv,      data["env_ql"],     data["env_qr"],     data["env_w"]]
    plot_mean= [updraft_qv,  data["ql_mean"],    data["qr_mean"]]

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        #plot updrafts
        if (plot_it != 5):
            plots[plot_it].plot(plot_upd[plot_it][1], data["z_half"], ".-", color="blue", label="upd")
        if (plot_it == 5):
            plots[plot_it].plot(plot_upd[plot_it][1] * 100, data["z_half"], ".-", color="blue", label="upd")
        # plot environment
        if (plot_it < 4):
            plots[plot_it].plot(plot_env[plot_it][1], data["z_half"], ".-", color="red", label="env")
        # plot mean
        if (plot_it < 3):
            plots[plot_it].plot(plot_mean[plot_it][1], data["z_half"], ".-", color="purple", label="mean")

    plots[0].legend()
    plt.savefig(folder + title)
    plt.clf()


def plot_var_covar_mean(data, title, folder="tests/output/"):
    """
    Plots variance and covariance profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 16})
    mpl.rc('lines', linewidth=4, markersize=10)

    # data to plot
    x_lab         = ["Hvar",       "QTvar",       "HQTcov"]
    plot_var_mean = ["Hvar_mean",  "QTvar_mean",  "HQTcov_mean"]
    plot_var_env  = ["env_Hvar",   "env_QTvar",   "env_HQTcov"]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        plots[plot_it].plot(data[plot_var_mean[plot_it]][1], data["z_half"], ".-", label=plot_var_mean[plot_it], c="black")
        plots[plot_it].plot(data[plot_var_env[plot_it]][1],  data["z_half"], ".-", label=plot_var_env[plot_it],  c="red")

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_var_covar_components(data, title, folder="tests/output/"):
    """
    Plots variance and covariance components profiles from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 16})
    mpl.rc('lines', linewidth=4, markersize=10)

    # data to plot
    plot_Hvar_c   = ["Hvar_dissipation",   "Hvar_entr_gain",   "Hvar_detr_loss",   "Hvar_shear",   "Hvar_rain"]
    plot_QTvar_c  = ["QTvar_dissipation",  "QTvar_entr_gain",  "QTvar_detr_loss",  "QTvar_shear",  "QTvar_rain"]
    plot_HQTcov_c = ["HQTcov_dissipation", "HQTcov_entr_gain", "HQTcov_detr_loss", "HQTcov_shear", "HQTcov_rain"]
    color_c       = ['green',              'pink',             'purple',           'orange',       'blue']

    x_lab         = ["Hvar",      "QTvar",      "HQTcov"]
    plot_var_data = [plot_Hvar_c, plot_QTvar_c, plot_HQTcov_c]

    # iteration over plots
    plots = []
    for plot_it in range(3):
        plots.append(plt.subplot(1,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel(x_lab[plot_it])
        plots[plot_it].set_ylabel('z [m]')
        plots[plot_it].set_ylim([0, data["z_half"][-1] + (data["z_half"][1] - data["z_half"][0]) * 0.5])
        plots[plot_it].grid(True)
        plots[plot_it].xaxis.set_major_locator(ticker.MaxNLocator(2))

        for var in range(5):
            plots[plot_it].plot(data[plot_var_data[plot_it][var]][1],   data["z_half"], ".-", label=plot_Hvar_c[var],  c=color_c[var])

    plots[0].legend()
    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries_1D(data, title, folder="tests/output/"):
    """
    Plots timeseries from Scampy

    Input:
    data   - dictionary with previousely read it data
    title  - name for the created plot
    folder - folder where to save the created plot
    """
    # customize defaults
    fig = plt.figure(1)
    fig.set_figheight(8)
    fig.set_figwidth(14)
    mpl.rcParams.update({'font.size': 12})
    mpl.rc('lines', linewidth=2, markersize=6)

    # data to plot
    plot_y = [data["updraft_cloud_cover"], data["updraft_cloud_top"], data["lwp"], data["ustar"], data["lhf"], data["Tsurface"]]
    y_lab  = ['updr cl. cover',            'updr CB, CT',             'LWP',       'u star',      'shf',       'T surf']

    # iteration over plots
    plots = []
    for plot_it in range(6):
        plots.append(plt.subplot(2,3,plot_it+1))
                               #(rows, columns, number)
        plots[plot_it].set_xlabel('t [s]')
        plots[plot_it].set_ylabel(y_lab[plot_it])
        plots[plot_it].set_xlim([0, data["t"][-1]])
        plots[plot_it].grid(True)
        plots[plot_it].plot(data["t"], plot_y[plot_it], '.-', color="b")
        if plot_it == 1:
            plots[plot_it].plot(data["t"], data["updraft_cloud_base"], '.-', color="r", label="CB")
            plots[plot_it].plot(data["t"], data["updraft_cloud_top"],  '.-', color="b", label="CT")
            plots[plot_it].legend()
        if plot_it == 4:
            plots[plot_it].plot(data["t"], data["lhf"], '.-', color="r", label='lhf')
            plots[plot_it].plot(data["t"], data["shf"], '.-', color="b", label='shf')
            plots[plot_it].legend()

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries(data, case, folder="tests/output/"):
    """
    Plots the time series of Scampy simulations

    Input:
    data   - dictionary with previousely read it data
    case   - name for the tested simulation (to be used in the plot name)
    folder - folder where to save the created plot
    """
    # customize figure parameters
    fig = plt.figure(1)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    mpl.rcParams.update({'font.size': 20})

    # read data
    z_half   = data["z_half"]
    time     = data["t"] / 60. / 60.
    mean_qv  = data["qt_mean"]    - data["ql_mean"]
    updr_qv  = data["updraft_qt"] - data["updraft_ql"]
    env_qv   = data["env_qt"]     - data["env_ql"]
    env_area = 1. - data["updraft_area"]

    # data to plot
    mean_data  = [data["thetal_mean"],    data["buoyancy_mean"],    data["tke_mean"],      mean_qv,                data["ql_mean"],           data["qr_mean"]]
    mean_label = ["mean thl [K]",         "mean buo [cm2/s3]",      "mean TKE [m2/s2]",    "mean qv [g/kg]",       "mean ql [g/kg]",          "mean qr [g/kg]"]
    mean_cb    = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds,           mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    env_data   = [data["env_thetal"],     env_area,                 data["env_w"],         env_qv,                 data["env_ql"],            data["env_qr"]]
    env_label  = ["env thl [K]",          "env area [%]",           "env w [m/s]",         "env qv [g/kg]",        "env ql [g/kg]",           "env qr [g/kg]"]
    env_cb     = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds_r,         mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    updr_data  = [data["updraft_thetal"], data["updraft_area"],     data["updraft_w"],     updr_qv,                data["updraft_ql"],        data["updraft_qr"]]
    updr_label = ["updr thl [K]",   "     updr area [%]",           "updr w [m/s]",        "updr qv [g/kg]",       "updr ql [g/kg]",          "updr qr [g/kg"]
    updr_cb    = [mpl.cm.Reds,            mpl.cm.Reds,              mpl.cm.Reds,           mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    flux_data  = [data["massflux_h"],     data["diffusive_flux_h"], data["total_flux_h"],  data["massflux_qt"],    data["diffusive_flux_qt"], data["total_flux_qt"]]
    flux_label = ["M_FL thl",             "D_FL thl ",              "tot FL thl",          "M_FL qt",              "D_FL qt",                 "tot FL qt"]
    flux_cb    = [mpl.cm.Spectral,        mpl.cm.Spectral,          mpl.cm.Spectral,       mpl.cm.Spectral_r,      mpl.cm.Spectral_r,         mpl.cm.Spectral_r]

    misc_data  = [data["eddy_viscosity"], data["eddy_diffusivity"], data["mixing_length"], data["entrainment_sc"], data["detrainment_sc"],    data["massflux"]]
    misc_label = ["eddy visc",            "eddy diff",              "mix. length",         "entr sc",              "detr sc",                 "mass flux"]
    misc_cb    = [mpl.cm.Blues,           mpl.cm.Blues,             mpl.cm.Blues,          mpl.cm.Blues,           mpl.cm.Blues,              mpl.cm.Blues]

    data_to_plot = [mean_data,  env_data,  updr_data,  flux_data,  misc_data]
    labels       = [mean_label, env_label, updr_label, flux_label, misc_label]
    titles       = ["01mean",   "02env",   "03updr",   "04flx",    "05misc"]
    cbs          = [mean_cb,    env_cb,    updr_cb,    flux_cb,    misc_cb]

    # iteration over plots
    for var in range(5):
        ax   = []
        plot = []
        for plot_it in range(6):
            ax.append(fig.add_subplot(2,3,plot_it+1))
                                    #(rows, columns, number)
            ax[plot_it].set_xlabel('t [hrs]')
            ax[plot_it].set_ylabel('z [m]')
            plot.append(ax[plot_it].pcolormesh(time, z_half, data_to_plot[var][plot_it], cmap=discrete_cmap(32, cbs[var][plot_it]), rasterized=True))
            fig.colorbar(plot[plot_it], ax=ax[plot_it], label=labels[var][plot_it])

        #plt.tight_layout()
        plt.savefig(folder + case + "_timeseries_" + titles[var] + ".pdf")
        plt.clf()
