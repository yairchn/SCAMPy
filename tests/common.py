import os
import subprocess
import json
import warnings
import pprint as pp
from netCDF4 import Dataset
import numpy as np

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
    os.system("python ../generate_namelist.py " + case)
    file_case = open(case + '.in').read()
    namelist  = json.loads(file_case)
    # fh = open(namelist['meta']['casename']+ ".in", 'w')
    # add here changes to namelist file:
    namelist['output']['output_root'] = "./Tests."
    namelist['meta']['uuid'] = case
    namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'] = 'buoyancy_sorting'
    write_file(case+".in",namelist)
    pp.pprint(namelist)

    os.system("python ../generate_paramlist.py " +  case)
    file_params = open('paramlist_' + case + '.in').read()
    paramlist = json.loads(file_params)
    # add here changes to paramlist file such as:
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor'] = 0.1
    paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor'] = 2.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_erf_const'] = 2.0
    paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor'] = 0.05
    # paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area'] = 0.5
    write_file("paramlist_"+case+".in",paramlist)
    pp.pprint(paramlist)

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
    Remove the folder with netcdf files from tests.
    Remove the in files generated by scampy.
    """
    cmd = "rm -r Tests.Output.*"
    subprocess.call(cmd , shell=True)
    cmd = "rm *.in"
    subprocess.call(cmd , shell=True)


def read_data_srs(sim_data):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries of profiles plots

    Input:
    sim_data  - netcdf Dataset with simulation results
    """
    variables = ["temperature_mean", "thetal_mean", "qt_mean", "ql_mean", "qr_mean",\
                 "buoyancy_mean", "b_mix","u_mean", "v_mean", "tke_mean",\
                 "updraft_buoyancy", "updraft_area", "env_qt", "updraft_qt", "env_ql", "updraft_ql", "updraft_thetal",\
                 "env_qr", "updraft_qr", "updraft_w", "env_w", "env_thetal","updraft_RH", "env_RH",\
                 "massflux_h", "diffusive_flux_h", "total_flux_h",\
                 "massflux_qt","diffusive_flux_qt","total_flux_qt","turbulent_entrainment",\
                 "eddy_viscosity", "eddy_diffusivity", "mixing_length", "mixing_length_ratio",\
                 "entrainment_sc", "detrainment_sc", "massflux", "nh_pressure", "eddy_diffusivity",\
                 "Hvar_mean", "QTvar_mean", "HQTcov_mean", "env_Hvar", "env_QTvar", "env_HQTcov",\
                 "Hvar_dissipation", "QTvar_dissipation", "HQTcov_dissipation",\
                 "Hvar_entr_gain", "QTvar_entr_gain", "HQTcov_entr_gain",\
                 "Hvar_detr_loss", "QTvar_detr_loss", "HQTcov_detr_loss",\
                 "Hvar_shear", "QTvar_shear", "HQTcov_shear",\
                 "Hvar_rain", "QTvar_rain", "HQTcov_rain","tke_entr_gain","tke_detr_loss",\
                 "tke_advection","tke_buoy","tke_dissipation","tke_pressure","tke_transport","tke_shear"\
                ]
    # read the data
    data = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}

    for var in variables:
        data[var] = []
        if ("qt" in var or "ql" in var or "qr" in var):
            try:
                data[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :])) * 1000  #g/kg
            except:
                data[var] = np.transpose(np.array(sim_data["profiles/w_mean" ][:, :])) * 0  #g/kg
        elif ("p0" in var):
            data[var] = np.transpose(np.array(sim_data["reference/" + var][:, :])) * 100   #hPa
        else:
            data[var] = np.transpose(np.array(sim_data["profiles/"  + var][:, :]))

    return data


def read_les_data_srs(les_data):
    """
    Read in the data from netcdf file into a dictionary that can be used for timeseries of profiles plots

    Input:
    les_data - netcdf Dataset with specific fileds taken from LES stats file
    """
    variables = ["temperature_mean", "thetali_mean", "qt_mean", "ql_mean", "buoyancy_mean",\
                "u_mean", "v_mean", "tke_mean","v_translational_mean", "u_translational_mean",\
                 "updraft_buoyancy", "updraft_fraction", "env_thetali", "updraft_thetali", "env_qt", "updraft_qt", "env_ql", "updraft_ql",\
                 "qr_mean", "env_qr", "updraft_qr", "updraft_w", "env_w",  "env_buoyancy", "updraft_ddz_p_alpha",\
                 "thetali_mean2", "qt_mean2", "env_thetali2", "env_qt2", "env_qt_thetali",\
                 "tke_prod_A" ,"tke_prod_B" ,"tke_prod_D" ,"tke_prod_P" ,"tke_prod_T" ,"tke_prod_S", "Hvar_mean" ,"QTvar_mean" ,"env_Hvar" ,"env_QTvar" ,"env_HQTcov",\
                 "massflux_h" ,"massflux_qt" ,"total_flux_h" ,"total_flux_qt" ,"diffusive_flux_h" ,"diffusive_flux_qt"]

    les = {"z_half" : np.array(les_data["z_half"][:]), "t" : np.array(les_data["t"][:])}
    les["rho"] = np.array(les_data["profiles/rho"][:])
    les["p0"] = np.array(les_data["profiles/p0"][:])
    for var in variables:
        les[var] = []
        les[var] = np.transpose(np.array(les_data["profiles/"+var][:, :]))
    return les


def read_data_timeseries(sim_data):
    """
    Read in the 1D data from netcdf file into a dictionary that can be used for timeseries plots

    Input:
    sim_data - netcdf Dataset with simulation results
    """
    variables = ["cloud_cover_mean", "cloud_base_mean", "cloud_top_mean",\
                 "ustar", "lwp_mean", "shf", "lhf", "Tsurface", "rd"]

    # read the data
    data = {"z_half" : np.array(sim_data["profiles/z_half"][:]), "t" : np.array(sim_data["profiles/t"][:])}
    maxz = np.max(data['z_half'])
    # maxz = 1400.0
    for var in variables:
        data[var] = []
        data[var] = np.array(sim_data["timeseries/" + var][:])

    CT = np.array(sim_data["timeseries/cloud_top_mean"][:])
    CT[np.where(CT<=0.0)] = np.nan
    data["cloud_top_mean"] = CT

    CB = np.array(sim_data["timeseries/cloud_base_mean"][:])
    CB[np.where(CB>=maxz)] = np.nan
    data["cloud_base_mean"] = CB

    return data

def read_les_data_timeseries(les_data):
    """
    Read in the 1D data from netcdf file into a dictionary that can be used for timeseries plots

    Input:
    les_data - netcdf Dataset with specific fileds taken from LES stats file
    """
    # read the data
    les = {"z_half_les" : np.array(les_data["z_half"][:]), "t" : np.array(les_data["t"][:])}
    maxz = np.max(les['z_half_les'])
    CF = np.array(les_data["timeseries/cloud_fraction_mean"][:])
    CF[np.where(CF<=0.0)] = np.nan
    les["cloud_cover_mean"] = CF

    CT = np.array(les_data["timeseries/cloud_top_mean"][:])
    CT[np.where(CT<=0.0)] = np.nan
    les["cloud_top_mean"] = CT
    CB = np.array(les_data["timeseries/cloud_base_mean"][:])
    CB[np.where(CB>maxz)] = np.nan
    les["cloud_base_mean"] = CB

    les["ustar"] = np.array(les_data["timeseries/friction_velocity_mean"][:])
    les["shf"] = np.array(les_data["timeseries/shf_surface_mean"][:])
    les["lhf"] = np.array(les_data["timeseries/lhf_surface_mean"][:])
    les["lwp_mean"] = np.array(les_data["timeseries/lwp_mean"][:])
    return les


def write_file(name, list):
    fh = open(name, 'w')
    json.dump(list, fh, sort_keys=True, indent=4)
    fh.close()

    return