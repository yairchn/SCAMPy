import netCDF4 as nc
import numpy as np


def main(fname):

    return
def initiate_record(fname):


    tuning_record = nc.Dataset(fname, "w", format="NETCDF4")
    grp_stats = tuning_record.createGroup('data')
    grp_stats.createDimension('z', 75) # get this from namelistfile
    grp_stats.createDimension('t', 361) # get this from namelistfile
    grp_stats.createDimension('dim', None)
    t = grp_stats.createVariable('t', 'f4', 't')
    z = grp_stats.createVariable('z', 'f4', 'z')
    lwp = grp_stats.createVariable('lwp', 'f4', ('t', 'dim'))
    cloud_cover = grp_stats.createVariable('cloud_cover', 'f4', ('t', 'dim'))
    cloud_top = grp_stats.createVariable('cloud_top', 'f4', ('t', 'dim'))
    cloud_base = grp_stats.createVariable('cloud_base', 'f4', ('t', 'dim'))
    thetal_mean = grp_stats.createVariable('thetal', 'f4', ('t', 'z', 'dim'))
    qt_mean = grp_stats.createVariable('qt_mean', 'f4', ('t', 'z', 'dim'))
    ql_mean = grp_stats.createVariable('ql_mean', 'f4', ('t', 'z', 'dim'))
    temperature = grp_stats.createVariable('temperature', 'f4', ('t', 'z', 'dim'))
    tune_param = grp_stats.createVariable('tune_param', 'f4', 'dim')
    costFun = grp_stats.createVariable('costFun', 'f4', 'dim')  # this might be a problem if dim=1 implies 2 value

    return tuning_record

def record_data(theta_, u, new_data):

    print costFun.shape + 1

    nsim =  costFun.shape + 1
    # add new data to netCDF file
    lwp_ = np.multiply(new_data.groups['data'].variables['lwp'], 1.0)
    cloud_cover_ = np.multiply(new_data.groups['data'].variables['cloud_cover'], 1.0)
    cloud_top_ = np.multiply(new_data.groups['data'].variables['cloud_top'], 1.0)
    cloud_base_ = np.multiply(new_data.groups['data'].variables['cloud_base'], 1.0)
    thetal_mean_ = np.multiply(new_data.groups['data'].variables['thetal_mean'], 1.0)
    temperature_mean_ = np.multiply(new_data.groups['data'].variables['temperature_mean'], 1.0)
    qt_mean_ = np.multiply(new_data.groups['data'].variables['qt_mean'], 1.0)
    ql_mean_ = np.multiply(new_data.groups['data'].variables['ql_mean'], 1.0)

    lwp[:, nsim] = lwp_
    cloud_cover[:, nsim] = cloud_cover_
    cloud_top[:, nsim] = cloud_top_
    cloud_base[:, nsim] = cloud_base_
    thetal_mean[:, :, nsim] = thetal_mean_
    temperature_mean[:, :, nsim] = temperature_mean_
    qt_mean[:, :, nsim] = qt_mean_
    ql_mean[:, :, nsim] = ql_mean_
    tune_param[nsim] = theta_
    costFun[nsim] = u

    return


if __name__ == '__main__':
    main()
