import sys
sys.path.insert(0, "./")
sys.path.insert(0, "../")

import numpy as np

import matplotlib as mpl
mpl.use('Agg')  # To allow plotting when display is off
import matplotlib.pyplot as plt
from matplotlib import ticker

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


def plot_mean(data, title, folder="plots/output/"):
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


def plot_drafts(data, title, folder="plots/output/"):
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


def plot_var_covar_mean(data, title, folder="plots/output/"):
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


def plot_var_covar_components(data, title, folder="plots/output/"):
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


def plot_timeseries_1D(data, title, folder="plots/output/"):
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
    plot_y = [data["updraft_cloud_cover"], data["updraft_cloud_top"], data["ustar"], data["lhf"], data["shf"], data["Tsurface"]]
    y_lab  = ['updr cl. cover',            'updr CB, CT',             'ustar',       'lhf',      'shf',       'T surf']

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

    plt.tight_layout()
    plt.savefig(folder + title)
    plt.clf()


def plot_timeseries(data, case, folder="plots/output/"):
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
