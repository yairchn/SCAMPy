#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
import sys
cimport  EDMF_Updrafts
from Grid cimport Grid
cimport EDMF_Environment
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport SurfaceBase
from Cases cimport  CasesBase
from ReferenceState cimport  ReferenceState
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport  *
from turbulence_functions cimport *
from utility_functions cimport *
from libc.math cimport fmax, sqrt, exp, pow, cbrt, fmin, fabs
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import netCDF4 as nc

cdef class EDMF_PrognosticTKE(ParameterizationBase):
    # Initialize the class
    def __init__(self, namelist, paramlist, Grid Gr, ReferenceState Ref):
        # Initialize the base parameterization class
        ParameterizationBase.__init__(self, paramlist,  Gr, Ref)

        # Set the number of updrafts (1)
        try:
            self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
        except:
            self.n_updrafts = 1
            print('Turbulence--EDMF_PrognosticTKE: defaulting to single updraft')
        try:
            self.use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
        except:
            self.use_steady_updrafts = False
        try:
            self.use_local_micro = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
        except:
            self.use_local_micro = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to local (level-by-level) microphysics')

        try:
            self.calc_tke = namelist['turbulence']['EDMF_PrognosticTKE']['calculate_tke']
        except:
            self.calc_tke = True

        try:
            self.calc_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']
        except:
            self.calc_scalar_var = False
        if (self.calc_scalar_var==True and self.calc_tke==False):
            sys.exit('Turbulence--EDMF_PrognosticTKE: >>calculate_tke<< must be set to True when >>calc_scalar_var<< is True (to calculate the mixing length for the variance and covariance calculations')

        try:
            if str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_z':
                self.entr_detr_fp = entr_detr_inverse_z
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'dry':
                self.entr_detr_fp = entr_detr_dry
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_w':
                self.entr_detr_fp = entr_detr_inverse_w
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'b_w2':
                self.entr_detr_fp = entr_detr_b_w2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke':
                self.entr_detr_fp = entr_detr_tke
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'suselj':
                self.entr_detr_fp = entr_detr_suselj
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'buoyancy_sorting':
                self.entr_detr_fp = entr_detr_buoyancy_sorting
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'none':
                self.entr_detr_fp = entr_detr_none
            else:
                print('Turbulence--EDMF_PrognosticTKE: Entrainment rate namelist option is not recognized')
        except:
            self.entr_detr_fp = entr_detr_b_w2
            print('Turbulence--EDMF_PrognosticTKE: defaulting to cloudy entrainment formulation')
        if(self.calc_tke == False and 'tke' in str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'])):
             sys.exit('Turbulence--EDMF_PrognosticTKE: >>calc_tke<< must be set to True when entrainment is using tke')

        try:
            self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        except:
            self.similarity_diffusivity = False
            print('Turbulence--EDMF_PrognosticTKE: defaulting to TKE-based eddy diffusivity')
        if(self.similarity_diffusivity == False and self.calc_tke ==False):
            sys.exit('Turbulence--EDMF_PrognosticTKE: either >>use_similarity_diffusivity<< or >>calc_tke<< flag is needed to get the eddy diffusivities')

        if(self.similarity_diffusivity == True and self.calc_tke == True):
           print("TKE will be calculated but not used for eddy diffusivity calculation")

        try:
            self.extrapolate_buoyancy = namelist['turbulence']['EDMF_PrognosticTKE']['extrapolate_buoyancy']
        except:
            self.extrapolate_buoyancy = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to extrapolation of updraft buoyancy along a pseudoadiabat')

        try:
            self.mixing_scheme = str(namelist['turbulence']['EDMF_PrognosticTKE']['mixing_length'])
        except:
            self.mixing_scheme = 'default'
            print 'Using (Tan et al, 2018) default'

        # Get values from paramlist
        # set defaults at some point?
        self.surface_area = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.max_area_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        self.entrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
        self.detrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
        self.entrainment_erf_const = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_erf_const']
        self.turbulent_entrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor']
        self.pressure_buoy_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.pressure_drag_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
        self.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
        # "Legacy" coefficients used by the steady updraft routine
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        if self.calc_tke == True:
            self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
            self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']

        # Need to code up as paramlist option?
        self.minimum_area = 1e-5

        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = EDMF_Updrafts.UpdraftVariables(self.n_updrafts, namelist,paramlist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = EDMF_Updrafts.UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar)
        # Create the class for updraft microphysics
        self.UpdMicro = EDMF_Updrafts.UpdraftMicrophysics(paramlist, self.n_updrafts, Gr, Ref)

        # Create the environment variable class (major diagnostic and prognostic variables)
        self.EnvVar = EDMF_Environment.EnvironmentVariables(namelist,Gr)
        # Create the class for environment thermodynamics
        self.EnvThermo = EDMF_Environment.EnvironmentThermodynamics(namelist, paramlist, Gr, Ref, self.EnvVar)

        # Entrainment rates
        self.entr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        #self.press = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Detrainment rates
        self.detr_sc = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        self.buoyant_frac = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.b_mix = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # turbulent entrainment
        self.frac_turb_entr = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.frac_turb_entr_full = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.turb_entr_W = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.turb_entr_H = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.turb_entr_QT = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Pressure term in updraft vertical momentum equation
        self.nh_pressure = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

        # mixing length
        self.mixing_length = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.horizontal_KM = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.horizontal_KH = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # diagnosed tke budget terms
        self.tke_transport = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_advection = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # Near-surface BC of updraft area fraction
        self.area_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.h_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.qt_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')

        # Mass flux tendencies of mean scalars (for output)
        self.massflux_tendency_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_tendency_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')


        # (Eddy) diffusive tendencies of mean scalars (for output)
        self.diffusive_tendency_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_tendency_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # Vertical fluxes for output
        self.massflux_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.massflux_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_flux_h = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        self.diffusive_flux_qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')
        if self.calc_tke:
            self.massflux_tke = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # Added by Ignacio : Length scheme in use (mls), and smooth min effect (ml_ratio)
        self.prandtl_nvec = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.mls = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.ml_ratio = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.l_entdet = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.b = np.zeros((Gr.nzg,),dtype=np.double, order='c')


        data = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/closure_diagnostics/Bomex/SF100/stats/Stats.Bomex.nc','r')
        # data = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/closure_diagnostics/DYCOMS/stats/Stats.DYCOMS_RF01.nc','r')
        # data = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/closure_diagnostics/TRMM_LBA/stats/Stats.TRMM_LBA.nc','r')
        z = np.multiply(data.groups['reference'].variables['zp_half'],1.0)
        dapdz_upd_ = np.multiply(np.nanmean(data.groups['profiles'].variables['updraft_ddz_p_alpha'][180:-1],axis=0),1.0)
        self.dapdz_upd = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], z, dapdz_upd_)
        return

    cpdef initialize(self, GridMeanVariables GMV):
        self.UpdVar.initialize(GMV)
        return

    # Initialize the IO pertaining to this class
    cpdef initialize_io(self, NetCDFIO_Stats Stats):

        self.UpdVar.initialize_io(Stats)
        self.EnvVar.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('entrainment_sc')
        Stats.add_profile('detrainment_sc')
        Stats.add_profile('nh_pressure')
        Stats.add_profile('horizontal_KM')
        Stats.add_profile('horizontal_KH')
        Stats.add_profile('buoyant_frac')
        Stats.add_profile('b_mix')
        Stats.add_ts('rd')
        Stats.add_profile('turbulent_entrainment')
        Stats.add_profile('turbulent_entrainment_full')
        Stats.add_profile('turbulent_entrainment_W')
        Stats.add_profile('turbulent_entrainment_H')
        Stats.add_profile('turbulent_entrainment_QT')
        Stats.add_profile('massflux')
        Stats.add_profile('massflux_h')
        Stats.add_profile('massflux_qt')
        Stats.add_profile('massflux_tendency_h')
        Stats.add_profile('massflux_tendency_qt')
        Stats.add_profile('diffusive_flux_h')
        Stats.add_profile('diffusive_flux_qt')
        Stats.add_profile('diffusive_tendency_h')
        Stats.add_profile('diffusive_tendency_qt')
        Stats.add_profile('total_flux_h')
        Stats.add_profile('total_flux_qt')
        Stats.add_profile('mixing_length')
        Stats.add_profile('updraft_qt_precip')
        Stats.add_profile('updraft_thetal_precip')
        # Diff mixing lengths: Ignacio
        Stats.add_profile('ed_length_scheme')
        Stats.add_profile('mixing_length_ratio')
        Stats.add_profile('entdet_balance_length')
        Stats.add_profile('interdomain_tke_t')
        if self.calc_tke:
            Stats.add_profile('tke_buoy')
            Stats.add_profile('tke_dissipation')
            Stats.add_profile('tke_entr_gain')
            Stats.add_profile('tke_detr_loss')
            Stats.add_profile('tke_shear')
            Stats.add_profile('tke_pressure')
            Stats.add_profile('tke_interdomain')
            Stats.add_profile('tke_transport')
            Stats.add_profile('tke_advection')

        if self.calc_scalar_var:
            Stats.add_profile('Hvar_dissipation')
            Stats.add_profile('QTvar_dissipation')
            Stats.add_profile('HQTcov_dissipation')
            Stats.add_profile('Hvar_entr_gain')
            Stats.add_profile('QTvar_entr_gain')
            Stats.add_profile('Hvar_detr_loss')
            Stats.add_profile('QTvar_detr_loss')
            Stats.add_profile('HQTcov_detr_loss')
            Stats.add_profile('HQTcov_entr_gain')
            Stats.add_profile('Hvar_shear')
            Stats.add_profile('QTvar_shear')
            Stats.add_profile('HQTcov_shear')
            Stats.add_profile('Hvar_rain')
            Stats.add_profile('QTvar_rain')
            Stats.add_profile('HQTcov_rain')
            Stats.add_profile('Hvar_interdomain')
            Stats.add_profile('QTvar_interdomain')
            Stats.add_profile('HQTcov_interdomain')


        return

    cpdef io(self, NetCDFIO_Stats Stats):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg-self.Gr.gw
            double [:] mean_entr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_nh_pressure = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_detr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] massflux = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mf_h = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mf_qt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_frac_turb_entr = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_frac_turb_entr_full = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_turb_entr_W = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_turb_entr_H = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_turb_entr_QT = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_horizontal_KM = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_horizontal_KH = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_buoyant_frac = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_b_mix = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        self.UpdVar.io(Stats)
        self.EnvVar.io(Stats)

        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_ts('rd', self.pressure_plume_spacing)
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                mf_h[k] = interp2pt(self.massflux_h[k], self.massflux_h[k-1])
                mf_qt[k] = interp2pt(self.massflux_qt[k], self.massflux_qt[k-1])
                massflux[k] = interp2pt(self.m[0,k], self.m[0,k-1])
                if self.UpdVar.Area.bulkvalues[k] > 0.0:
                    for i in xrange(self.n_updrafts):
                        mean_entr_sc[k] += self.UpdVar.Area.values[i,k] * self.entr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_detr_sc[k] += self.UpdVar.Area.values[i,k] * self.detr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_nh_pressure[k] += self.UpdVar.Area.values[i,k] * self.nh_pressure[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_frac_turb_entr_full[k] += self.UpdVar.Area.values[i,k] * self.frac_turb_entr_full[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_frac_turb_entr[k] += self.UpdVar.Area.values[i,k] * self.frac_turb_entr[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_W[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_W[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_H[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_H[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_QT[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_QT[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_horizontal_KM[k] += self.UpdVar.Area.values[i,k] * self.horizontal_KM[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_horizontal_KH[k] += self.UpdVar.Area.values[i,k] * self.horizontal_KH[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_buoyant_frac[k] += self.UpdVar.Area.values[i,k] * self.buoyant_frac[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_b_mix[k] += self.UpdVar.Area.values[i,k] * self.b_mix[i,k]/self.UpdVar.Area.bulkvalues[k]

        Stats.write_profile('turbulent_entrainment', mean_frac_turb_entr[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('turbulent_entrainment_full', mean_frac_turb_entr_full[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('turbulent_entrainment_W', mean_turb_entr_W[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('turbulent_entrainment_H', mean_turb_entr_H[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('turbulent_entrainment_QT', mean_turb_entr_QT[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('horizontal_KM', mean_horizontal_KM[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('horizontal_KH', mean_horizontal_KH[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('entrainment_sc', mean_entr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('detrainment_sc', mean_detr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('buoyant_frac', mean_buoyant_frac[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('b_mix', mean_b_mix[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('nh_pressure', mean_nh_pressure[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux', massflux[self.Gr.gw:self.Gr.nzg-self.Gr.gw ])
        Stats.write_profile('massflux_h', mf_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_qt', mf_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_tendency_h', self.massflux_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('massflux_tendency_qt', self.massflux_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_flux_h', self.diffusive_flux_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_flux_qt', self.diffusive_flux_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_tendency_h', self.diffusive_tendency_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('diffusive_tendency_qt', self.diffusive_tendency_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('total_flux_h', np.add(mf_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw],
                                                   self.diffusive_flux_h[self.Gr.gw:self.Gr.nzg-self.Gr.gw]))
        Stats.write_profile('total_flux_qt', np.add(mf_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw],
                                                    self.diffusive_flux_qt[self.Gr.gw:self.Gr.nzg-self.Gr.gw]))
        Stats.write_profile('mixing_length', self.mixing_length[kmin:kmax])
        Stats.write_profile('updraft_qt_precip', self.UpdMicro.prec_source_qt_tot[kmin:kmax])
        Stats.write_profile('updraft_thetal_precip', self.UpdMicro.prec_source_h_tot[kmin:kmax])

        #Different mixing lengths : Ignacio
        Stats.write_profile('ed_length_scheme', self.mls[kmin:kmax])
        Stats.write_profile('mixing_length_ratio', self.ml_ratio[kmin:kmax])
        Stats.write_profile('entdet_balance_length', self.l_entdet[kmin:kmax])
        Stats.write_profile('interdomain_tke_t', self.b[kmin:kmax])
        if self.calc_tke:
            self.compute_covariance_dissipation(self.EnvVar.TKE)
            Stats.write_profile('tke_dissipation', self.EnvVar.TKE.dissipation[kmin:kmax])
            Stats.write_profile('tke_entr_gain', self.EnvVar.TKE.entr_gain[kmin:kmax])
            self.compute_covariance_detr(self.EnvVar.TKE)
            Stats.write_profile('tke_detr_loss', self.EnvVar.TKE.detr_loss[kmin:kmax])
            Stats.write_profile('tke_shear', self.EnvVar.TKE.shear[kmin:kmax])
            Stats.write_profile('tke_buoy', self.EnvVar.TKE.buoy[kmin:kmax])
            Stats.write_profile('tke_pressure', self.EnvVar.TKE.press[kmin:kmax])
            Stats.write_profile('tke_interdomain', self.EnvVar.TKE.interdomain[kmin:kmax])
            self.compute_tke_transport()
            Stats.write_profile('tke_transport', self.tke_transport[kmin:kmax])
            self.compute_tke_advection()
            Stats.write_profile('tke_advection', self.tke_advection[kmin:kmax])

        if self.calc_scalar_var:
            self.compute_covariance_dissipation(self.EnvVar.Hvar)
            Stats.write_profile('Hvar_dissipation', self.EnvVar.Hvar.dissipation[kmin:kmax])
            self.compute_covariance_dissipation(self.EnvVar.QTvar)
            Stats.write_profile('QTvar_dissipation', self.EnvVar.QTvar.dissipation[kmin:kmax])
            self.compute_covariance_dissipation(self.EnvVar.HQTcov)
            Stats.write_profile('HQTcov_dissipation', self.EnvVar.HQTcov.dissipation[kmin:kmax])
            Stats.write_profile('Hvar_entr_gain', self.EnvVar.Hvar.entr_gain[kmin:kmax])
            Stats.write_profile('QTvar_entr_gain', self.EnvVar.QTvar.entr_gain[kmin:kmax])
            Stats.write_profile('HQTcov_entr_gain', self.EnvVar.HQTcov.entr_gain[kmin:kmax])
            self.compute_covariance_detr(self.EnvVar.Hvar)
            self.compute_covariance_detr(self.EnvVar.QTvar)
            self.compute_covariance_detr(self.EnvVar.HQTcov)
            Stats.write_profile('Hvar_detr_loss', self.EnvVar.Hvar.detr_loss[kmin:kmax])
            Stats.write_profile('QTvar_detr_loss', self.EnvVar.QTvar.detr_loss[kmin:kmax])
            Stats.write_profile('HQTcov_detr_loss', self.EnvVar.HQTcov.detr_loss[kmin:kmax])
            Stats.write_profile('Hvar_shear', self.EnvVar.Hvar.shear[kmin:kmax])
            Stats.write_profile('QTvar_shear', self.EnvVar.QTvar.shear[kmin:kmax])
            Stats.write_profile('HQTcov_shear', self.EnvVar.HQTcov.shear[kmin:kmax])
            Stats.write_profile('Hvar_rain', self.EnvVar.Hvar.rain_src[kmin:kmax])
            Stats.write_profile('QTvar_rain', self.EnvVar.QTvar.rain_src[kmin:kmax])
            Stats.write_profile('HQTcov_rain', self.EnvVar.HQTcov.rain_src[kmin:kmax])
            Stats.write_profile('Hvar_interdomain', self.EnvVar.Hvar.interdomain[kmin:kmax])
            Stats.write_profile('QTvar_interdomain', self.EnvVar.QTvar.interdomain[kmin:kmax])
            Stats.write_profile('HQTcov_interdomain', self.EnvVar.HQTcov.interdomain[kmin:kmax])


        return



    # Perform the update of the scheme

    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg - self.Gr.gw

        self.update_inversion(GMV, Case.inversion_option)
        self.compute_pressure_plume_spacing(GMV, Case)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        if TS.nstep == 0:
            self.decompose_environment(GMV, 'values')
            self.EnvThermo.satadjust(self.EnvVar, True)
            self.initialize_covariance(GMV, Case)
            with nogil:
                for k in xrange(self.Gr.nzg):
                    if self.calc_tke:
                        self.EnvVar.TKE.values[k] = GMV.TKE.values[k]
                    if self.calc_scalar_var:
                        self.EnvVar.Hvar.values[k] = GMV.Hvar.values[k]
                        self.EnvVar.QTvar.values[k] = GMV.QTvar.values[k]
                        self.EnvVar.HQTcov.values[k] = GMV.HQTcov.values[k]

        self.decompose_environment(GMV, 'values')

        if self.use_steady_updrafts:
            self.compute_diagnostic_updrafts(GMV, Case)
        else:
            self.compute_prognostic_updrafts(GMV, Case, TS)

        # TODO -maybe not needed? - both diagnostic and prognostic updrafts end with decompose_environment
        # But in general ok here without thermodynamics because MF doesnt depend directly on buoyancy
        self.decompose_environment(GMV, 'values')

        self.update_GMV_MF(GMV, TS)
        # (###)
        # decompose_environment +  EnvThermo.satadjust + UpdThermo.buoyancy should always be used together
        # This ensures that:
        #   - the buoyancy of updrafts and environment is up to date with the most recent decomposition,
        #   - the buoyancy of updrafts and environment is updated such that
        #     the mean buoyancy with repect to reference state alpha_0 is zero.
        self.decompose_environment(GMV, 'mf_update')
        self.EnvThermo.satadjust(self.EnvVar, True)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.compute_eddy_diffusivities_tke(GMV, Case)

        self.update_GMV_ED(GMV, Case, TS)
        self.compute_covariance(GMV, Case, TS)

        # Back out the tendencies of the grid mean variables for the whole timestep by differencing GMV.new and
        # GMV.values
        ParameterizationBase.update(self, GMV, Case, TS)

        return

    cpdef compute_prognostic_updrafts(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):

        cdef:
            Py_ssize_t iter_
            double time_elapsed = 0.0

        self.set_subdomain_bcs()
        self.UpdVar.set_new_with_values()
        self.UpdVar.set_old_with_values()
        self.set_updraft_surface_bc(GMV, Case)
        self.dt_upd = np.minimum(TS.dt, 0.5 * self.Gr.dz/fmax(np.max(self.UpdVar.W.values),1e-10))
        while time_elapsed < TS.dt:
            self.compute_horizontal_eddy_diffusivities(GMV)
            self.compute_turbulent_entrainment(GMV,Case)
            self.compute_nh_pressure(TS)
            self.compute_entrainment_detrainment(GMV, Case, TS)
            self.solve_updraft_velocity_area(GMV,TS)
            self.solve_updraft_scalars(GMV, Case, TS)
            self.UpdVar.set_values_with_new()
            self.zero_area_fraction_cleanup(GMV)
            time_elapsed += self.dt_upd
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * self.Gr.dz/fmax(np.max(self.UpdVar.W.values),1e-10))
            # (####)
            # TODO - see comment (###)
            # It would be better to have a simple linear rule for updating environment here
            # instead of calling EnvThermo saturation adjustment scheme for every updraft.
            # If we are using quadratures this is expensive and probably unnecessary.
            self.decompose_environment(GMV, 'values')
            self.EnvThermo.satadjust(self.EnvVar, False)
            self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)
            self.set_subdomain_bcs()
        return

    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz
            double dzi = self.Gr.dzi
            eos_struct sa
            entr_struct ret
            entr_in_struct input
            double a,b,c, w, w_km,  w_mid, w_low, denom, arg
            double entr_w, detr_w, B_k, area_k, w2

        self.set_updraft_surface_bc(GMV, Case)
        # self.compute_entrainment_detrainment(GMV, Case)


        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.H.values[i,gw] = self.h_surface_bc[i]
                self.UpdVar.QT.values[i,gw] = self.qt_surface_bc[i]
                # Find the cloud liquid content
                sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[gw],
                         self.UpdVar.QT.values[i,gw], self.UpdVar.H.values[i,gw])
                self.UpdVar.QL.values[i,gw] = sa.ql
                self.UpdVar.T.values[i,gw] = sa.T
                self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[gw], self.UpdVar.T.values[i,gw],
                                                                   &self.UpdVar.QT.values[i,gw], &self.UpdVar.QL.values[i,gw],
                                                                   &self.UpdVar.QR.values[i,gw], &self.UpdVar.H.values[i,gw],
                                                                   i, gw)
                for k in xrange(gw+1, self.Gr.nzg-gw):
                    denom = 1.0 + self.entr_sc[i,k] * dz
                    self.UpdVar.H.values[i,k] = (self.UpdVar.H.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.H.values[k])/denom
                    self.UpdVar.QT.values[i,k] = (self.UpdVar.QT.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.QT.values[k])/denom


                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                             self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                    self.UpdVar.QL.values[i,k] = sa.ql
                    self.UpdVar.T.values[i,k] = sa.T
                    self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[k], self.UpdVar.T.values[i,k],
                                                                       &self.UpdVar.QT.values[i,k], &self.UpdVar.QL.values[i,k],
                                                                       &self.UpdVar.QR.values[i,k], &self.UpdVar.H.values[i,k],
                                                                       i, k)
        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.QR.set_bcs(self.Gr)
        self.UpdVar.H.set_bcs(self.Gr)
        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.satadjust(self.EnvVar, False)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        # Solve updraft velocity equation
        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.W.values[i, self.Gr.gw-1] = self.w_surface_bc[i]
                self.entr_sc[i,gw] = 2.0 /dz # 0.0 ?
                self.detr_sc[i,gw] = 0.0
                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    area_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                    if area_k >= self.minimum_area:
                        w_km = self.UpdVar.W.values[i,k-1]
                        entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                        detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                        B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        w2 = ((self.vel_buoy_coeff * B_k + 0.5 * w_km * w_km * dzi)
                              /(0.5 * dzi +entr_w + (self.pressure_drag_coeff/self.pressure_plume_spacing)/sqrt(fmax(area_k,self.minimum_area))))
                        if w2 > 0.0:
                            self.UpdVar.W.values[i,k] = sqrt(w2)
                        else:
                            self.UpdVar.W.values[i,k:] = 0
                            break
                    else:
                        self.UpdVar.W.values[i,k:] = 0




        self.UpdVar.W.set_bcs(self.Gr)

        cdef double au_lim
        with nogil:
            for i in xrange(self.n_updrafts):
                au_lim = self.max_area_factor * self.area_surface_bc[i]
                self.UpdVar.Area.values[i,gw] = self.area_surface_bc[i]
                w_mid = 0.5* (self.UpdVar.W.values[i,gw])
                for k in xrange(gw+1, self.Gr.nzg):
                    w_low = w_mid
                    w_mid = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                    if w_mid > 0.0:
                        if self.entr_sc[i,k]>(0.9/dz):
                            self.entr_sc[i,k] = 0.9/dz

                        self.UpdVar.Area.values[i,k] = (self.Ref.rho0_half[k-1]*self.UpdVar.Area.values[i,k-1]*w_low/
                                                        (1.0-(self.entr_sc[i,k]-self.detr_sc[i,k])*dz)/w_mid/self.Ref.rho0_half[k])
                        # # Limit the increase in updraft area when the updraft decelerates
                        if self.UpdVar.Area.values[i,k] >  au_lim:
                            self.UpdVar.Area.values[i,k] = au_lim
                            self.detr_sc[i,k] =(self.Ref.rho0_half[k-1] * self.UpdVar.Area.values[i,k-1]
                                                * w_low / au_lim / w_mid / self.Ref.rho0_half[k] + self.entr_sc[i,k] * dz -1.0)/dz
                    else:
                        # the updraft has terminated so set its area fraction to zero at this height and all heights above
                        self.UpdVar.Area.values[i,k] = 0.0
                        self.UpdVar.H.values[i,k] = GMV.H.values[k]
                        self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                        self.UpdVar.QR.values[i,k] = GMV.QR.values[k]
                        #TODO wouldnt it be more consistent to have here?
                        #self.UpdVar.QL.values[i,k] = GMV.QL.values[k]
                        #self.UpdVar.T.values[i,k] = GMV.T.values[k]
                        sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                                 self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                        self.UpdVar.QL.values[i,k] = sa.ql
                        self.UpdVar.T.values[i,k] = sa.T

        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.satadjust(self.EnvVar, False)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.UpdVar.Area.set_bcs(self.Gr)

        self.UpdMicro.prec_source_h_tot = np.sum(np.multiply(self.UpdMicro.prec_source_h, self.UpdVar.Area.values), axis=0)
        self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt, self.UpdVar.Area.values), axis=0)

        return

    cpdef update_inversion(self,GridMeanVariables GMV, option):
        ParameterizationBase.update_inversion(self, GMV,option)
        return

    cpdef compute_mixing_length(self, double obukhov_length, GridMeanVariables GMV):

        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double tau =  get_mixing_tau(self.zi, self.wstar)
            double l1, l2, l3, z_, N
            double l[3]
            double ri_grad, shear2
            double qt_dry, th_dry, t_cloudy, qv_cloudy, qt_cloudy, th_cloudy
            double lh, cpm, prefactor, d_buoy_thetal_dry, d_buoy_qt_dry
            double d_buoy_thetal_cloudy, d_buoy_qt_cloudy, d_buoy_thetal_total, d_buoy_qt_total
            double grad_thl_plus=0.0, grad_qt_plus=0.0, grad_thv_plus=0.0
            double thv, grad_qt, grad_qt_low, grad_thv_low, grad_thv
            double grad_b_thl, grad_b_qt
            double m_eps = 1.0e-9 # Epsilon to avoid zero
            double a, c_neg, wc_upd_nn, wc_env

        if (self.mixing_scheme == 'sbl'):
            for k in xrange(gw, self.Gr.nzg-gw):
                z_ = self.Gr.z_half[k]
                # kz scale (surface layer)
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_ /(3.75*self.tke_ed_coeff) * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                elif obukhov_length > 0.0: #stable
                    l2 = vkb * z_ /(3.75*self.tke_ed_coeff)#/  (1. + 2.7 *z_/obukhov_length)
                else:
                    l2 = vkb * z_/(3.75*self.tke_ed_coeff)

                # Shear-dissipation TKE equilibrium scale (Stable)
                shear2 = pow((GMV.U.values[k+1] - GMV.U.values[k-1]) * 0.5 * self.Gr.dzi, 2) + \
                    pow((GMV.V.values[k+1] - GMV.V.values[k-1]) * 0.5 * self.Gr.dzi, 2) + \
                    pow((self.EnvVar.W.values[k] - self.EnvVar.W.values[k-1]) * self.Gr.dzi, 2)

                qt_dry = self.EnvThermo.qt_dry[k]
                th_dry = self.EnvThermo.th_dry[k]
                t_cloudy = self.EnvThermo.t_cloudy[k]
                qv_cloudy = self.EnvThermo.qv_cloudy[k]
                qt_cloudy = self.EnvThermo.qt_cloudy[k]
                th_cloudy = self.EnvThermo.th_cloudy[k]
                lh = latent_heat(t_cloudy)
                cpm = cpm_c(qt_cloudy)
                grad_thl_low = grad_thl_plus
                grad_qt_low = grad_qt_plus
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.Gr.dzi
                grad_thl = interp2pt(grad_thl_low, grad_thl_plus)
                grad_qt = interp2pt(grad_qt_low, grad_qt_plus)
                # g/theta_ref
                prefactor = g * ( Rd / self.Ref.alpha0_half[k] /self.Ref.p0_half[k]) * exner_c(self.Ref.p0_half[k])

                d_buoy_thetal_dry = prefactor * (1.0 + (eps_vi-1.0) * qt_dry)
                d_buoy_qt_dry = prefactor * th_dry * (eps_vi-1.0)

                if self.EnvVar.CF.values[k] > 0.0:
                    d_buoy_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_buoy_qt_cloudy = (lh / cpm / t_cloudy * d_buoy_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_buoy_thetal_cloudy = 0.0
                    d_buoy_qt_cloudy = 0.0

                d_buoy_thetal_total = (self.EnvVar.CF.values[k] * d_buoy_thetal_cloudy
                                        + (1.0-self.EnvVar.CF.values[k]) * d_buoy_thetal_dry)
                d_buoy_qt_total = (self.EnvVar.CF.values[k] * d_buoy_qt_cloudy
                                    + (1.0-self.EnvVar.CF.values[k]) * d_buoy_qt_dry)

                # Partial buoyancy gradients
                grad_b_thl  = grad_thl * d_buoy_thetal_total
                grad_b_qt = grad_qt  * d_buoy_qt_total
                ri_thl = grad_b_thl / fmax(shear2, m_eps)
                ri_qt  = grad_b_qt / fmax(shear2, m_eps)
                ri_grad = fmin(ri_thl+ri_qt, 0.25) # Ri_grad used in Prandtl number calculation.

                # Turbulent Prandtl number:
                if obukhov_length <= 0.0: # globally convective
                    self.prandtl_nvec[k] = 0.74
                elif obukhov_length > 0.0: #stable
                    # CSB (Dan Li, 2019), with Pr_neutral=0.74 and w1=40.0/13.0
                    self.prandtl_nvec[k] = 0.74*( 2.0*ri_grad/
                        (1.0+(53.0/13.0)*ri_grad -sqrt( (1.0+(53.0/13.0)*ri_grad)**2.0 - 4.0*ri_grad ) ) )

                l3 = sqrt(self.tke_diss_coeff/fmax(self.tke_ed_coeff, m_eps)) * sqrt(self.EnvVar.TKE.values[k])
                l3 /= sqrt(fmax(shear2 - grad_b_thl/self.prandtl_nvec[k] - grad_b_qt/self.prandtl_nvec[k], m_eps))
                if ( shear2 - grad_b_thl/self.prandtl_nvec[k] - grad_b_qt/self.prandtl_nvec[k] < m_eps):
                    l3 = 1.0e6

                # Limiting stratification scale (Deardorff, 1976)
                thv = theta_virt_c(self.Ref.p0_half[k], self.EnvVar.T.values[k], self.EnvVar.QT.values[k],
                    self.EnvVar.QL.values[k])
                grad_thv_low = grad_thv_plus
                grad_thv_plus = ( theta_virt_c(self.Ref.p0_half[k+1], self.EnvVar.T.values[k+1], self.EnvVar.QT.values[k+1],
                    self.EnvVar.QL.values[k+1])  -  thv) * self.Gr.dzi
                grad_thv = interp2pt(grad_thv_low, grad_thv_plus)

                N = sqrt(fmax(g/thv*grad_thv, 0.0))
                if N > 0.0:
                    l1 = fmin(sqrt(fmax(0.4*self.EnvVar.TKE.values[k],0.0))/N, 1.0e6)
                else:
                    l1 = 1.0e6

                l[0]=l2; l[1]=l1; l[2]=l3;

                j = 0
                while(j<len(l)):
                    if l[j]<m_eps or l[j]>1.0e6:
                        l[j] = 1.0e6
                    j += 1
                self.mls[k] = np.argmin(l)
                self.mixing_length[k] = auto_smooth_minimum(l, 0.1)
                self.ml_ratio[k] = self.mixing_length[k]/l[int(self.mls[k])]

        elif (self.mixing_scheme == 'sbtd_eq'):
            for k in xrange(gw, self.Gr.nzg-gw):
                z_ = self.Gr.z_half[k]
                # kz scale (surface layer)
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_ /(3.75*self.tke_ed_coeff) * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                elif obukhov_length > 0.0: #stable
                    l2 = vkb * z_ /(3.75*self.tke_ed_coeff) # /  (1. + 2.7 *z_/obukhov_length)
                else:
                    l2 = vkb * z_ /(3.75*self.tke_ed_coeff)

                # Buoyancy-shear-subdomain exchange-dissipation TKE equilibrium scale
                shear2 = pow((GMV.U.values[k+1] - GMV.U.values[k-1]) * 0.5 * self.Gr.dzi, 2) + \
                    pow((GMV.V.values[k+1] - GMV.V.values[k-1]) * 0.5 * self.Gr.dzi, 2) + \
                    pow((self.EnvVar.W.values[k] - self.EnvVar.W.values[k-1]) * self.Gr.dzi, 2)

                qt_dry = self.EnvThermo.qt_dry[k]
                th_dry = self.EnvThermo.th_dry[k]
                t_cloudy = self.EnvThermo.t_cloudy[k]
                qv_cloudy = self.EnvThermo.qv_cloudy[k]
                qt_cloudy = self.EnvThermo.qt_cloudy[k]
                th_cloudy = self.EnvThermo.th_cloudy[k]
                lh = latent_heat(t_cloudy)
                cpm = cpm_c(qt_cloudy)
                grad_thl_low = grad_thl_plus
                grad_qt_low = grad_qt_plus
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.Gr.dzi
                grad_thl = interp2pt(grad_thl_low, grad_thl_plus)
                grad_qt = interp2pt(grad_qt_low, grad_qt_plus)
                # g/theta_ref
                prefactor = g * ( Rd / self.Ref.alpha0_half[k] /self.Ref.p0_half[k]) * exner_c(self.Ref.p0_half[k])

                d_buoy_thetal_dry = prefactor * (1.0 + (eps_vi-1.0) * qt_dry)
                d_buoy_qt_dry = prefactor * th_dry * (eps_vi-1.0)

                if self.EnvVar.CF.values[k] > 0.0:
                    d_buoy_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_buoy_qt_cloudy = (lh / cpm / t_cloudy * d_buoy_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_buoy_thetal_cloudy = 0.0
                    d_buoy_qt_cloudy = 0.0

                d_buoy_thetal_total = (self.EnvVar.CF.values[k] * d_buoy_thetal_cloudy
                                        + (1.0-self.EnvVar.CF.values[k]) * d_buoy_thetal_dry)
                d_buoy_qt_total = (self.EnvVar.CF.values[k] * d_buoy_qt_cloudy
                                    + (1.0-self.EnvVar.CF.values[k]) * d_buoy_qt_dry)

                # Partial buoyancy gradients
                grad_b_thl = grad_thl * d_buoy_thetal_total
                grad_b_qt  = grad_qt  * d_buoy_qt_total
                ri_thl = grad_thl * d_buoy_thetal_total / fmax(shear2, m_eps)
                ri_qt  = grad_qt  * d_buoy_qt_total / fmax(shear2, m_eps)
                ri_grad = fmin(ri_thl+ri_qt, 0.25)

                # Turbulent Prandtl number:
                if obukhov_length <= 0.0: # globally convective
                    self.prandtl_nvec[k] = 0.74
                elif obukhov_length > 0.0: #stable
                    # CSB (Dan Li, 2019), with Pr_neutral=0.74 and w1=40.0/13.0
                    self.prandtl_nvec[k] = 0.74*( 2.0*ri_grad/
                        (1.0+(53.0/13.0)*ri_grad -sqrt( (1.0+(53.0/13.0)*ri_grad)**2.0 - 4.0*ri_grad ) ) )

                # Production/destruction terms
                a = self.tke_ed_coeff*(shear2 - grad_b_thl/self.prandtl_nvec[k] - grad_b_qt/self.prandtl_nvec[k])* sqrt(self.EnvVar.TKE.values[k])
                # Dissipation term
                c_neg = self.tke_diss_coeff*self.EnvVar.TKE.values[k]*sqrt(self.EnvVar.TKE.values[k])
                # Subdomain exchange term
                self.b[k] = 0.0
                for nn in xrange(self.n_updrafts):
                    wc_upd_nn = (self.UpdVar.W.values[nn,k]+self.UpdVar.W.values[nn,k-1])/2.0
                    wc_env = (self.EnvVar.W.values[k] - self.EnvVar.W.values[k-1])/2.0
                    self.b[k] += self.UpdVar.Area.values[nn,k]*wc_upd_nn*self.detr_sc[nn,k]/(1.0-self.UpdVar.Area.bulkvalues[k])*(
                        (wc_upd_nn-wc_env)*(wc_upd_nn-wc_env)/2.0-self.EnvVar.TKE.values[k])

                if abs(a) > m_eps and 4.0*a*c_neg > - self.b[k]*self.b[k]:
                    self.l_entdet[k] = fmax( -self.b[k]/2.0/a + sqrt( self.b[k]*self.b[k] + 4.0*a*c_neg )/2.0/a, 0.0)
                elif abs(a) < m_eps and abs(self.b[k]) > m_eps:
                    self.l_entdet[k] = c_neg/self.b[k]

                l3 = self.l_entdet[k]

                # Limiting stratification scale (Deardorff, 1976)
                thv = theta_virt_c(self.Ref.p0_half[k], self.EnvVar.T.values[k], self.EnvVar.QT.values[k],
                    self.EnvVar.QL.values[k])
                grad_thv_low = grad_thv_plus
                grad_thv_plus = ( theta_virt_c(self.Ref.p0_half[k+1], self.EnvVar.T.values[k+1], self.EnvVar.QT.values[k+1],
                    self.EnvVar.QL.values[k+1]) - thv) * self.Gr.dzi
                grad_thv = interp2pt(grad_thv_low, grad_thv_plus)
                if thv*grad_thv==0.0:
                    print('thv,grad_thv in mixing lentgh', thv,grad_thv)
                N = sqrt(fmax(g/thv*grad_thv, 0.0))
                if N > 0.0:
                    l1 = fmin(sqrt(fmax(0.4*self.EnvVar.TKE.values[k],0.0))/N, 1.0e6)
                else:
                    l1 = 1.0e6

                l[0]=l2; l[1]=l1; l[2]=l3;

                j = 0
                while(j<len(l)):
                    if l[j]<m_eps or l[j]>1.0e6:
                        l[j] = 1.0e6
                    j += 1

                self.mls[k] = np.argmin(l)
                self.mixing_length[k] = auto_smooth_minimum(l, 0.1)
                self.ml_ratio[k] = self.mixing_length[k]/l[int(self.mls[k])]

        else:
            # default mixingscheme , see Tan et al. (2018)
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    l1 = tau * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))
                    z_ = self.Gr.z_half[k]
                    if obukhov_length < 0.0: #unstable
                        l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                    elif obukhov_length > 0.0: #stable
                        l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
                        l1 = 1.0/m_eps
                    else:
                        l2 = vkb * z_
                    self.mixing_length[k] = fmax( 1.0/(1.0/fmax(l1,m_eps) + 1.0/l2), 1e-3)
                    self.prandtl_nvec[k] = 1.0
        return


    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double lm
            double we_half
            double pr
            double ri_thl, shear2

        if self.similarity_diffusivity:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV, Case)
        else:
            self.compute_mixing_length(Case.Sur.obukhov_length, GMV)
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    lm = self.mixing_length[k]
                    pr = self.prandtl_nvec[k]
                    self.KM.values[k] = self.tke_ed_coeff * lm * sqrt(fmax(self.EnvVar.TKE.values[k],0.0) )
                    self.KH.values[k] = self.KM.values[k] / pr

        return

    cpdef compute_horizontal_eddy_diffusivities(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i, k
            double l, R_up

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                for i in xrange(self.n_updrafts):
                    if self.UpdVar.Area.values[i,k]>0.0:
                        R_up = self.pressure_plume_spacing*sqrt(self.UpdVar.Area.values[i,k])
                        l = fmin(self.mixing_length[k],R_up)
                        self.horizontal_KM[i,k] = self.turbulent_entrainment_factor*sqrt(fmax(GMV.TKE.values[k],0.0))*l
                        self.horizontal_KH[i,k] = self.horizontal_KM[i,k] / self.prandtl_nvec[k]
                    else:
                        self.horizontal_KM[i,k] = 0.0
                        self.horizontal_KH[i,k] = 0.0

        return


    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case):

        self.update_inversion(GMV, Case.inversion_option)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        cdef:
            Py_ssize_t i, gw = self.Gr.gw
            double zLL = self.Gr.z_half[gw]
            double ustar = Case.Sur.ustar, oblength = Case.Sur.obukhov_length
            double alpha0LL  = self.Ref.alpha0_half[gw]
            double qt_var = get_surface_variance(Case.Sur.rho_qtflux*alpha0LL,
                                                 Case.Sur.rho_qtflux*alpha0LL, ustar, zLL, oblength)
            double h_var = get_surface_variance(Case.Sur.rho_hflux*alpha0LL,
                                                 Case.Sur.rho_hflux*alpha0LL, ustar, zLL, oblength)

            double a_ = self.surface_area/self.n_updrafts
            double surface_scalar_coeff

        # with nogil:
        for i in xrange(self.n_updrafts):
            surface_scalar_coeff= percentile_bounds_mean_norm(1.0-self.surface_area+i*a_,
                                                                   1.0-self.surface_area + (i+1)*a_ , 1000)

            self.area_surface_bc[i] = self.surface_area/self.n_updrafts
            self.w_surface_bc[i] = 0.0
            self.h_surface_bc[i] = (GMV.H.values[gw] + surface_scalar_coeff * sqrt(h_var))
            self.qt_surface_bc[i] = (GMV.QT.values[gw] + surface_scalar_coeff * sqrt(qt_var))
        return

    cpdef reset_surface_covariance(self, GridMeanVariables GMV, CasesBase Case):
        flux1 = Case.Sur.rho_hflux
        flux2 = Case.Sur.rho_qtflux
        cdef:
            double zLL = self.Gr.z_half[self.Gr.gw]
            double ustar = Case.Sur.ustar, oblength = Case.Sur.obukhov_length
            double alpha0LL  = self.Ref.alpha0_half[self.Gr.gw]
            #double get_surface_variance = get_surface_variance(flux1, flux2 ,ustar, zLL, oblength)
        if self.calc_tke:
            GMV.TKE.values[self.Gr.gw] = get_surface_tke(Case.Sur.ustar,
                                                     self.wstar,
                                                     self.Gr.z_half[self.Gr.gw],
                                                     Case.Sur.obukhov_length)
        if self.calc_scalar_var:
            GMV.Hvar.values[self.Gr.gw] = get_surface_variance(flux1*alpha0LL,flux1*alpha0LL, ustar, zLL, oblength)
            GMV.QTvar.values[self.Gr.gw] = get_surface_variance(flux2*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
            GMV.HQTcov.values[self.Gr.gw] = get_surface_variance(flux1*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
        return


    # Find values of environmental variables by subtracting updraft values from grid mean values
    # whichvals used to check which substep we are on--correspondingly use 'GMV.SomeVar.value' (last timestep value)
    # or GMV.SomeVar.mf_update (GMV value following massflux substep)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals):

        # first make sure the 'bulkvalues' of the updraft variables are updated
        self.UpdVar.set_means(GMV)

        cdef:
            Py_ssize_t k, gw = self.Gr.gw
            double val1, val2, au_full
        if whichvals == 'values':

            with nogil:
                for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1
                    self.EnvVar.QT.values[k] = val1 * GMV.QT.values[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                    self.EnvVar.H.values[k] = val1 * GMV.H.values[k] - val2 * self.UpdVar.H.bulkvalues[k]
                    # Have to account for staggering of W--interpolate area fraction to the "full" grid points
                    # Assuming GMV.W = 0!
                    au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                    self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]

            if self.calc_tke:
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W, self.EnvVar.TKE, &GMV.W.values[0],&GMV.W.values[0], &GMV.TKE.values[0])
            if self.calc_scalar_var:
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar, &GMV.H.values[0],&GMV.H.values[0], &GMV.Hvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.QT,self.UpdVar.QT,self.EnvVar.QT,self.EnvVar.QT,self.EnvVar.QTvar, &GMV.QT.values[0],&GMV.QT.values[0], &GMV.QTvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.QT,self.EnvVar.H, self.EnvVar.QT,self.EnvVar.HQTcov, &GMV.H.values[0],&GMV.QT.values[0], &GMV.HQTcov.values[0])



        elif whichvals == 'mf_update':
            # same as above but replace GMV.SomeVar.values with GMV.SomeVar.mf_update

            with nogil:
                for k in xrange(self.Gr.nzg-1):
                    val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                    val2 = self.UpdVar.Area.bulkvalues[k] * val1

                    self.EnvVar.QT.values[k] = val1 * GMV.QT.mf_update[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                    self.EnvVar.H.values[k] = val1 * GMV.H.mf_update[k] - val2 * self.UpdVar.H.bulkvalues[k]
                    # Have to account for staggering of W
                    # Assuming GMV.W = 0!
                    au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                    self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]

            if self.calc_tke:
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W, self.EnvVar.TKE,
                                 &GMV.W.values[0],&GMV.W.values[0], &GMV.TKE.values[0])
            if self.calc_scalar_var:
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar,
                                 &GMV.H.values[0],&GMV.H.values[0], &GMV.Hvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar,
                                 &GMV.QT.values[0],&GMV.QT.values[0], &GMV.QTvar.values[0])
                self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov,
                                 &GMV.H.values[0], &GMV.QT.values[0], &GMV.HQTcov.values[0])


        return

    # Note: this assumes all variables are defined on half levels not full levels (i.e. phi, psi are not w)
    # if covar_e.name is not 'tke'.
    cdef get_GMV_CoVar(self, EDMF_Updrafts.UpdraftVariable au,
                        EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                        EDMF_Environment.EnvironmentVariable phi_e,  EDMF_Environment.EnvironmentVariable psi_e,
                        EDMF_Environment.EnvironmentVariable_2m covar_e,
                       double *gmv_phi, double *gmv_psi, double *gmv_covar):
        cdef:
            Py_ssize_t i,k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),au.bulkvalues)
            double phi_diff, psi_diff
            double tke_factor = 1.0


        #with nogil:
        for k in xrange(self.Gr.nzg):
            if covar_e.name == 'tke':
                tke_factor = 0.5
                phi_diff = interp2pt(phi_e.values[k-1]-gmv_phi[k-1], phi_e.values[k]-gmv_phi[k])
                psi_diff = interp2pt(psi_e.values[k-1]-gmv_psi[k-1], psi_e.values[k]-gmv_psi[k])
            else:
                tke_factor = 1.0
                phi_diff = phi_e.values[k]-gmv_phi[k]
                psi_diff = psi_e.values[k]-gmv_psi[k]


            gmv_covar[k] = tke_factor * ae[k] * phi_diff * psi_diff + ae[k] * covar_e.values[k]
            for i in xrange(self.n_updrafts):
                if covar_e.name == 'tke':
                    phi_diff = interp2pt(phi_u.values[i,k-1]-gmv_phi[k-1], phi_u.values[i,k]-gmv_phi[k])
                    psi_diff = interp2pt(psi_u.values[i,k-1]-gmv_psi[k-1], psi_u.values[i,k]-gmv_psi[k])
                else:
                    phi_diff = phi_u.values[i,k]-gmv_phi[k]
                    psi_diff = psi_u.values[i,k]-gmv_psi[k]

                gmv_covar[k] += tke_factor * au.values[i,k] * phi_diff * psi_diff
        return


    cdef get_env_covar_from_GMV(self, EDMF_Updrafts.UpdraftVariable au,
                                EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                                EDMF_Environment.EnvironmentVariable phi_e, EDMF_Environment.EnvironmentVariable psi_e,
                                EDMF_Environment.EnvironmentVariable_2m covar_e,
                                double *gmv_phi, double *gmv_psi, double *gmv_covar):
        cdef:
            Py_ssize_t i,k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),au.bulkvalues)
            double phi_diff, psi_diff
            double tke_factor = 1.0
        if covar_e.name == 'tke':
            tke_factor = 0.5

        #with nogil:
        for k in xrange(self.Gr.nzg):
            if ae[k] > 0.0:
                if covar_e.name == 'tke':
                    phi_diff = interp2pt(phi_e.values[k-1] - gmv_phi[k-1],phi_e.values[k] - gmv_phi[k])
                    psi_diff = interp2pt(psi_e.values[k-1] - gmv_psi[k-1],psi_e.values[k] - gmv_psi[k])
                else:
                    phi_diff = phi_e.values[k] - gmv_phi[k]
                    psi_diff = psi_e.values[k] - gmv_psi[k]

                covar_e.values[k] = gmv_covar[k] - tke_factor * ae[k] * phi_diff * psi_diff
                for i in xrange(self.n_updrafts):
                    if covar_e.name == 'tke':
                        phi_diff = interp2pt(phi_u.values[i,k-1] - gmv_phi[k-1],phi_u.values[i,k] - gmv_phi[k])
                        psi_diff = interp2pt(psi_u.values[i,k-1] - gmv_psi[k-1],psi_u.values[i,k] - gmv_psi[k])
                    else:
                        phi_diff = phi_u.values[i,k] - gmv_phi[k]
                        psi_diff = psi_u.values[i,k] - gmv_psi[k]

                    covar_e.values[k] -= tke_factor * au.values[i,k] * phi_diff * psi_diff
                covar_e.values[k] = covar_e.values[k]/ae[k]
            else:
                covar_e.values[k] = 0.0
        return

    cpdef compute_turbulent_entrainment(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            double tau =  get_mixing_tau(self.zi, self.wstar)
            double a, a_full, K, K_full, R_up, R_up_full, wu_half

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    a = self.UpdVar.Area.values[i,k]
                    a_full = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                    R_up = self.pressure_plume_spacing*sqrt(a)
                    R_up_full = self.pressure_plume_spacing*sqrt(a_full)
                    wu_half = interp2pt(self.UpdVar.W.values[i,k], self.UpdVar.W.values[i,k-1])
                    if a*wu_half  > 0.0:
                        self.turb_entr_H[i,k]  = (2.0/R_up**2.0)*self.Ref.rho0_half[k] * a * self.horizontal_KH[i,k]  * \
                                                    (self.EnvVar.H.values[k] - self.UpdVar.H.values[i,k])
                        self.turb_entr_QT[i,k] = (2.0/R_up**2.0)*self.Ref.rho0_half[k]* a * self.horizontal_KH[i,k]  * \
                                                     (self.EnvVar.QT.values[k] - self.UpdVar.QT.values[i,k])
                        self.frac_turb_entr[i,k]    = (2.0/R_up**2.0) * self.horizontal_KH[i,k] / wu_half

                    else:
                        self.turb_entr_H[i,k] = 0.0
                        self.turb_entr_QT[i,k] = 0.0

                    if a_full*self.UpdVar.W.values[i,k] > 0.0:
                        K_full = interp2pt(self.horizontal_KM[i,k],self.horizontal_KM[i,k-1])
                        self.turb_entr_W[i,k]  = (2.0/R_up_full**2.0)*self.Ref.rho0[k] * a_full * K_full  * \
                                                    (self.EnvVar.W.values[k]-self.UpdVar.W.values[i,k])
                        self.frac_turb_entr_full[i,k] = (2.0/R_up_full**2.0) * K_full / self.UpdVar.W.values[i,k]
                    else:
                        self.turb_entr_W[i,k] = 0.0

        return

    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            entr_struct ret
            entr_in_struct input
            eos_struct sa
            double t0 = 6*3600.0
            double transport_plus, transport_minus
            long quadrature_order = 3


        self.UpdVar.get_cloud_base_top_cover()
        input.wstar = self.wstar

        input.dz = self.Gr.dz
        input.zbl = self.compute_zbl_qt_grad(GMV)

        #Bomex
        les_z = [  40.0,   80.0,  120.0,  160.0,  200.0,  240.0,  280.0,  320.0,  360.0,  400.0,  440.0,  480.0,
                  520.0,  560.0,  600.0,  640.0,  680.0,  720.0,  760.0,  800.0,  840.0,  880.0,  920.0,  960.0,
                 1000.0, 1040.0, 1080.0, 1120.0, 1160.0, 1200.0, 1240.0, 1280.0, 1320.0, 1360.0, 1400.0, 1440.0,
                 1480.0, 1520.0, 1560.0, 1600.0, 1640.0, 1680.0, 1720.0, 1760.0, 1800.0, 1840.0, 1880.0, 1920.0,
                 1960.0, 2000.0, 2040.0, 2080.0, 2120.0, 2160.0, 2200.0, 2240.0, 2280.0, 2320.0, 2360.0, 2400.0,
                 2440.0, 2480.0, 2520.0, 2560.0, 2600.0, 2640.0, 2680.0, 2720.0, 2760.0, 2800.0, 2840.0, 2880.0,
                 2920.0, 2960.0, 3000.0]

        les_eps = [ 0.00000000e+00,  0.00000000e+00,  4.49728936e-03, 2.87948092e-03,
                    2.08951803e-03,  1.60350311e-03,  1.28516105e-03,  1.10242655e-03,
                    9.72152100e-04,  8.83858438e-04,  8.77751207e-04,  9.45005387e-04,
                    1.15658431e-03,  1.55078419e-03,  1.99003492e-03,  2.29274043e-03,
                    2.45978320e-03,  2.64266866e-03,  2.82431420e-03,  2.07840483e-03,
                    6.71879585e-04,  4.93498870e-04,  9.74082257e-04,  1.34989853e-03,
                    1.58703677e-03,  1.53923080e-03,  1.51237089e-03,  1.49133765e-03,
                    1.44355580e-03,  1.54095781e-03,  1.61386967e-03,  1.50875317e-03,
                    1.38239877e-03,  1.37918242e-03,  1.23491995e-03,  9.74901874e-04,
                    7.40931430e-04,  3.66733976e-03,  4.41620150e-03,  3.82222464e-03,
                    2.49322790e-03,  4.07008917e-03,  7.12453458e-04,  2.02255238e-03,
                    2.24107395e-03,  1.05655463e-03,  2.42617134e-04,  0.00000000e+00,
                    0.00000000e+00,  9.30836313e-05,  0.00000000e+00,  2.11893183e-04,
                    1.63551766e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00]

        les_del = [0.00000000e+00, 0.00000000e+00, 1.96420168e-03, 1.82430375e-03,
                   1.73227225e-03, 1.61080336e-03, 1.56263484e-03, 1.62488040e-03,
                   1.70033242e-03, 1.81653711e-03, 2.09165747e-03, 2.45382722e-03,
                   2.99142903e-03, 3.54313303e-03, 3.90718204e-03, 3.84591615e-03,
                   3.71223904e-03, 3.58100097e-03, 3.59238620e-03, 2.71202217e-03,
                   1.86011865e-03, 1.84003698e-03, 2.04925016e-03, 2.21483696e-03,
                   2.31481471e-03, 2.36186927e-03, 2.26893964e-03, 2.39386298e-03,
                   2.39456223e-03, 2.37286437e-03, 2.37326241e-03, 2.48636569e-03,
                   2.54804748e-03, 2.84976771e-03, 2.67548268e-03, 2.71652123e-03,
                   3.28338780e-03, 6.76064968e-03, 1.15032850e-02, 7.01016503e-03,
                   5.24781665e-03, 6.88957344e-03, 2.15333383e-03, 3.50015487e-03,
                   3.55636768e-03, 1.76389066e-03, 4.36657860e-04, 8.35420286e-05,
                   5.29498013e-04, 1.18626063e-03, 1.18237577e-03, 1.23739675e-03,
                   2.61280695e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

        #upd 7
        les_z0 = [ 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0,
                   420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0]

        les_eps0 = [0.0, 0.007151032251177144, 0.012752113368040115, 0.005988539541988313,  0.004282150293599231,  0.003545566491834995,  0.004342613726457053,  0.0035389572072166272,
                 0.002790356164528793, 0.0017916592579873799,0.000984358257630152,  0.0003984043112863245, 0.0018778302270292908, 0.002285165100878707,  0.0015550835321015378,
                 0.0020640063326635375,0.0019162753514598793,0.00245809465544908,   0.0021627643759586155, 0.002810920450765202,  0.0027280839325874127, 0.004361366216703746,
                 0.0026612739965294614,0.002106410654028828, 0.00025972884501378575,0.00012651809014006604,0.0003690116578876232, 0.0004004293611326121, 0.0003567165893022612,
                 0.0004820640571380742,0.00027298917941642,  0.0004247780935361723, 0.001654404644455766,  0.001813400957275164,  0.003958294719055029]

        les_del0 =    [0.0, 0.00000000e+00       , 0.00000000e+00         ,0.0012819486314423036 ,0.00000000e+00          ,0.00000000e+00         ,0.00000000e+00       ,0.00000000e+00,
                    0.00000000e+00       , 0.00000000e+00         ,0.00000000e+00        ,0.00017204429822013088  ,0.004442152368803824   ,0.000775150296885889   ,0.0002726485702498611,
                    0.00000000e+00       , 0.0005835581788702592  ,0.002997471910900405  ,0.00575250947116905     ,0.006128707582037827   ,0.003373912437113889   ,0.0030896100332640517,
                    0.004429761734661555 ,  0.0004838674672345965 ,0.00063522177863887   ,0.00000000e+00          ,0.00000000e+00         ,0.00000000e+00        ,0.00000000e+00,
                    0.00000000e+00       ,0.00000000e+00          ,0.0012552443261196996 ,0.004447598150555128    ,0.0075441882007852646  ,0.008630663163982682]

        # TRMM_LBA
        # les_z = [  200.0,   400.0,   600.0,   800.0,  1000.0,  1200.0,  1400.0,  1600.0,  1800.0,  2000.0,
        #           2200.0,  2400.0,  2600.0,  2800.0,  3000.0,  3200.0,  3400.0,  3600.0,  3800.0,  4000.0,
        #           4200.0,  4400.0,  4600.0,  4800.0,  5000.0,  5200.0,  5400.0,  5600.0,  5800.0,  6000.0,
        #           6200.0,  6400.0,  6600.0,  6800.0,  7000.0,  7200.0,  7400.0,  7600.0,  7800.0,  8000.0,
        #           8200.0,  8400.0,  8600.0,  8800.0,  9000.0,  9200.0,  9400.0,  9600.0,  9800.0, 10000.0,
        #          10200.0, 10400.0, 10600.0, 10800.0, 11000.0, 11200.0, 11400.0, 11600.0, 11800.0, 12000.0,
        #          12200.0, 12400.0, 12600.0, 12800.0, 13000.0, 13200.0, 13400.0, 13600.0, 13800.0, 14000.0,
        #          14200.0, 14400.0, 14600.0, 14800.0, 15000.0, 15200.0, 15400.0, 15600.0, 15800.0, 16000.0,
        #          16200.0, 16400.0, 16600.0, 16800.0, 17000.0, 17200.0, 17400.0, 17600.0, 17800.0, 18000.0,
        #          18200.0, 18400.0, 18600.0, 18800.0, 19000.0, 19200.0, 19400.0, 19600.0, 19800.0, 20000.0,
        #          20200.0, 20400.0, 20600.0, 20800.0, 21000.0, 21200.0, 21400.0, 21600.0, 21800.0, 22000.0]

        # les_eps = [ 0.00000000e+00,  0.00000000e+00,  4.25462201e-04,  3.05401176e-04,
        #             1.09330932e-03,  3.11100956e-03,  5.48286676e-03,  9.37448231e-04,
        #             5.47812624e-04,  1.63638081e-03,  1.64910132e-03,  1.65076694e-03,
        #             1.66392723e-03,  1.76761613e-03,  1.86423476e-03,  1.84072003e-03,
        #             1.54123858e-03,  1.21725900e-03,  1.37057567e-03,  1.16631570e-03,
        #             1.53736761e-03,  1.12560949e-03,  9.46402550e-04,  1.02561660e-03,
        #             1.01162756e-03,  1.01172908e-03,  1.03971466e-03,  1.05987944e-03,
        #             1.05521840e-03,  1.05786203e-03,  1.07022434e-03,  1.07102023e-03,
        #             1.04790760e-03,  1.06825400e-03,  1.10649103e-03,  1.13316569e-03,
        #             1.15878365e-03,  1.14032169e-03,  1.08603823e-03,  1.05005627e-03,
        #             1.05654118e-03,  1.09338620e-03,  1.09917812e-03,  1.09273994e-03,
        #             1.18180723e-03,  1.49190519e-03,  8.36323446e-04,  1.34221259e-03,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             6.51115043e-06,  3.05857314e-03,  1.28395793e-04,  1.42199670e-02,
        #             2.12813678e-01,  0.00000000e+00,  4.62666985e-03,  0.00000000e+00,
        #             1.59361158e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #             0.00000000e+00,  0.00000000e+00]

        # les_del = [0.00000000e+00,  0.00000000e+00,  1.55546716e-04,  5.12061442e-04,
        #            1.42634006e-03,  3.53559768e-03,  6.07937382e-03,  2.29400747e-03,
        #            2.62431088e-03,  2.71230130e-03,  2.28918657e-03,  2.05602676e-03,
        #            1.90860031e-03,  1.89593236e-03,  1.84706134e-03,  1.80062803e-03,
        #            1.39688630e-03,  1.75950263e-03,  1.73483809e-03,  1.43426014e-03,
        #            1.93524744e-03,  1.81969557e-03,  1.29839499e-03,  1.35452156e-03,
        #            1.34716787e-03,  1.36604420e-03,  1.39758117e-03,  1.45083164e-03,
        #            1.48884928e-03,  1.42167363e-03,  1.45435817e-03,  1.65444395e-03,
        #            1.57236310e-03,  1.63875221e-03,  1.65397513e-03,  1.65880584e-03,
        #            1.74975974e-03,  1.79260796e-03,  1.69118234e-03,  1.93944361e-03,
        #            1.59703042e-03,  1.82471190e-03,  2.86230843e-03,  1.02399144e-02,
        #            6.37446051e-03,  2.33416245e-02,  1.06264315e-02,  7.16668749e-03,
        #            5.38331026e-03,  1.09527050e-02,  1.80801675e-02,  2.73526754e-03,
        #            6.55442658e-02,  6.94306303e-03,  7.63695539e-03,  3.57930606e-02,
        #            2.24940598e-01,  0.00000000e+00,  1.01789652e-02,  3.84682514e-04,
        #            1.94369297e-03,  2.07028689e-03,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00]

        # # DYCOMS
        # les_z = [   5.0,   10.0,   15.0,   20.0,   25.0,   30.0,   35.0,   40.0,   45.0,   50.0,   55.0,   60.0,
        #            65.0,   70.0,   75.0,   80.0,   85.0,   90.0,   95.0,  100.0,  105.0,  110.0,  115.0,  120.0,
        #           125.0,  130.0,  135.0,  140.0,  145.0,  150.0,  155.0,  160.0,  165.0,  170.0,  175.0,  180.0,
        #           185.0,  190.0,  195.0,  200.0,  205.0,  210.0,  215.0,  220.0,  225.0,  230.0,  235.0,  240.0,
        #           245.0,  250.0,  255.0,  260.0,  265.0,  270.0,  275.0,  280.0,  285.0,  290.0,  295.0,  300.0,
        #           305.0,  310.0,  315.0,  320.0,  325.0,  330.0,  335.0,  340.0,  345.0,  350.0,  355.0,  360.0,
        #           365.0,  370.0,  375.0,  380.0,  385.0,  390.0,  395.0,  400.0,  405.0,  410.0,  415.0,  420.0,
        #           425.0,  430.0,  435.0,  440.0,  445.0,  450.0,  455.0,  460.0,  465.0,  470.0,  475.0,  480.0,
        #           485.0,  490.0,  495.0,  500.0,  505.0,  510.0,  515.0,  520.0,  525.0,  530.0,  535.0,  540.0,
        #           545.0,  550.0,  555.0,  560.0,  565.0,  570.0,  575.0,  580.0,  585.0,  590.0,  595.0,  600.0,
        #           605.0,  610.0,  615.0,  620.0,  625.0,  630.0,  635.0,  640.0,  645.0,  650.0,  655.0,  660.0,
        #           665.0,  670.0,  675.0,  680.0,  685.0,  690.0,  695.0,  700.0,  705.0,  710.0,  715.0,  720.0,
        #           725.0,  730.0,  735.0,  740.0,  745.0,  750.0,  755.0,  760.0,  765.0,  770.0,  775.0,  780.0,
        #           785.0,  790.0,  795.0,  800.0,  805.0,  810.0,  815.0,  820.0,  825.0,  830.0,  835.0,  840.0,
        #           845.0,  850.0,  855.0,  860.0,  865.0,  870.0,  875.0,  880.0,  885.0,  890.0,  895.0,  900.0,
        #           905.0,  910.0,  915.0,  920.0,  925.0,  930.0,  935.0,  940.0,  945.0,  950.0,  955.0,  960.0,
        #           965.0,  970.0,  975.0,  980.0,  985.0,  990.0,  995.0, 1000.0, 1005.0, 1010.0, 1015.0, 1020.0,
        #          1025.0, 1030.0, 1035.0, 1040.0, 1045.0, 1050.0, 1055.0, 1060.0, 1065.0, 1070.0, 1075.0, 1080.0,
        #          1085.0, 1090.0, 1095.0, 1100.0, 1105.0, 1110.0, 1115.0, 1120.0, 1125.0, 1130.0, 1135.0, 1140.0,
        #          1145.0, 1150.0, 1155.0, 1160.0, 1165.0, 1170.0, 1175.0, 1180.0, 1185.0, 1190.0, 1195.0, 1200.0,
        #          1205.0, 1210.0, 1215.0, 1220.0, 1225.0, 1230.0, 1235.0, 1240.0, 1245.0, 1250.0, 1255.0, 1260.0,
        #          1265.0, 1270.0, 1275.0, 1280.0, 1285.0, 1290.0, 1295.0, 1300.0, 1305.0, 1310.0, 1315.0, 1320.0,
        #          1325.0, 1330.0, 1335.0, 1340.0, 1345.0, 1350.0, 1355.0, 1360.0, 1365.0, 1370.0, 1375.0, 1380.0,
        #          1385.0, 1390.0, 1395.0, 1400.0, 1405.0, 1410.0, 1415.0, 1420.0, 1425.0, 1430.0, 1435.0, 1440.0,
        #          1445.0, 1450.0, 1455.0, 1460.0, 1465.0, 1470.0, 1475.0, 1480.0, 1485.0, 1490.0, 1495.0, 1500.0]

        # les_eps = [  0.00000000e+00,  0.00000000e+00,  5.23582190e-02,  2.83414017e-02,
        #              1.99831287e-02,  1.94865337e-02,  1.62510007e-02,  1.38405252e-02,
        #              1.22494728e-02,  1.11596708e-02,  1.01009139e-02,  9.21599987e-03,
        #              8.57887636e-03,  7.89580332e-03,  7.45407259e-03,  7.06078905e-03,
        #              6.66845959e-03,  6.43126735e-03,  6.19534844e-03,  5.74028006e-03,
        #              5.19715272e-03,  4.91293793e-03,  4.67197959e-03,  4.56501598e-03,
        #              4.50461207e-03,  4.36550144e-03,  4.11530959e-03,  4.06875069e-03,
        #              4.00776747e-03,  3.76775280e-03,  3.45295470e-03,  3.02211153e-03,
        #              3.03490144e-03,  3.19386192e-03,  3.11166018e-03,  3.09300708e-03,
        #              2.90813607e-03,  2.91498120e-03,  2.97512128e-03,  2.75586568e-03,
        #              2.53162651e-03,  2.42663181e-03,  2.46675033e-03,  2.34712818e-03,
        #              2.28095432e-03,  2.22044184e-03,  1.99775344e-03,  2.06216875e-03,
        #              2.12342213e-03,  1.95790516e-03,  1.97573439e-03,  2.22115773e-03,
        #              2.08127140e-03,  1.85090993e-03,  1.76715869e-03,  1.68249478e-03,
        #              1.70181365e-03,  1.62100413e-03,  1.53302050e-03,  1.56725758e-03,
        #              1.61716834e-03,  1.58790060e-03,  1.70970347e-03,  1.79450348e-03,
        #              1.58180692e-03,  1.23729791e-03,  1.25428806e-03,  1.42588798e-03,
        #              1.34662256e-03,  1.31560354e-03,  1.28864227e-03,  1.25480134e-03,
        #              1.32648699e-03,  1.33884482e-03,  1.22258949e-03,  1.15857385e-03,
        #              1.24289301e-03,  1.22654681e-03,  1.05093799e-03,  8.74586253e-04,
        #              7.57847433e-04,  8.42629653e-04,  1.03560696e-03,  1.17265098e-03,
        #              1.21057497e-03,  1.03093911e-03,  9.68490578e-04,  1.15056536e-03,
        #              1.30134885e-03,  1.02551346e-03,  5.57059355e-04,  6.67833760e-04,
        #              9.32265801e-04,  9.09539350e-04,  8.37958726e-04,  8.55693785e-04,
        #              8.45439351e-04,  9.24588971e-04,  9.23683134e-04,  8.26502986e-04,
        #              8.21681438e-04,  7.72532578e-04,  7.12057256e-04,  7.28829341e-04,
        #              1.00260709e-03,  1.09603616e-03,  1.05433458e-03,  1.06394400e-03,
        #              8.78429291e-04,  6.99753780e-04,  8.14699426e-04,  9.20365526e-04,
        #              9.35643679e-04,  9.33594017e-04,  8.09251927e-04,  8.77814145e-04,
        #              9.37823489e-04,  1.91340215e-04,  3.58565841e-04,  9.35623775e-04,
        #              9.42744411e-04,  2.50983270e-04,  2.40760706e-04,  7.50764732e-04,
        #              0.00000000e+00,  3.36315578e-04,  1.96563153e-03,  2.59571322e-03,
        #              2.60479624e-03,  2.48871138e-03,  2.80731679e-03,  2.89528212e-03,
        #              2.93557861e-03,  3.12004619e-03,  3.08641306e-03,  3.04463185e-03,
        #              3.51992430e-03,  3.85695761e-03,  3.99290554e-03,  4.40476909e-03,
        #              4.84023741e-03,  4.91178054e-03,  4.75899733e-03,  4.66014028e-03,
        #              4.85408841e-03,  4.50938782e-03,  4.03135861e-03,  4.01057236e-03,
        #              4.07015961e-03,  4.11105111e-03,  3.60278079e-03,  3.11001970e-03,
        #              3.12443172e-03,  2.85551856e-03,  2.29539731e-03,  1.91935118e-03,
        #              1.36637670e-03,  9.22633235e-04,  4.96946102e-04,  2.84782620e-04,
        #              4.46895969e-04,  2.86123352e-04,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              1.30962114e-03,  2.14688317e-03,  2.64877729e-03,  7.52088606e-02,
        #              1.40114647e-01,  2.57969823e-02,  1.12591782e-03,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]


        # les_del = [0.00000000e+00,  0.00000000e+00,  1.65499641e-02,  8.32085671e-03,
        #            8.71201477e-03,  1.15303492e-02,  1.03461504e-02,  9.10283342e-03,
        #            8.45088133e-03,  7.75887020e-03,  7.11255848e-03,  6.80479318e-03,
        #            6.06470504e-03,  5.60856944e-03,  5.41114393e-03,  4.85756678e-03,
        #            4.78796679e-03,  4.52163786e-03,  4.49248179e-03,  3.94863702e-03,
        #            3.76032752e-03,  3.50065956e-03,  3.38529941e-03,  3.12706252e-03,
        #            3.27364856e-03,  3.12886426e-03,  2.93481423e-03,  3.01129664e-03,
        #            2.81992562e-03,  2.60476938e-03,  2.41688265e-03,  2.23145116e-03,
        #            2.33605173e-03,  2.24303088e-03,  2.17860582e-03,  2.14586055e-03,
        #            2.02860502e-03,  2.22705471e-03,  2.01600126e-03,  1.95071877e-03,
        #            1.70485636e-03,  1.91068818e-03,  1.87167465e-03,  1.67714924e-03,
        #            1.72795805e-03,  1.57237010e-03,  1.53579069e-03,  1.44078922e-03,
        #            1.43999016e-03,  1.34504783e-03,  1.49640078e-03,  1.76076252e-03,
        #            1.52377840e-03,  1.50894317e-03,  1.32551563e-03,  1.36094697e-03,
        #            1.25997827e-03,  1.27452120e-03,  1.29759187e-03,  1.36843980e-03,
        #            1.13871168e-03,  1.39813274e-03,  1.44854133e-03,  1.40142924e-03,
        #            1.18949139e-03,  1.00345064e-03,  1.14743022e-03,  1.06284177e-03,
        #            1.09406434e-03,  1.14782350e-03,  1.20342296e-03,  1.03986159e-03,
        #            1.10658830e-03,  1.12837444e-03,  1.07864727e-03,  9.91259302e-04,
        #            1.21638053e-03,  1.09735960e-03,  9.66479879e-04,  8.32372952e-04,
        #            8.78721569e-04,  9.22283381e-04,  9.59333821e-04,  8.83042752e-04,
        #            9.86738328e-04,  9.61919415e-04,  1.01321812e-03,  1.24853461e-03,
        #            1.20681442e-03,  8.64363240e-04,  7.88633044e-04,  1.04166046e-03,
        #            9.56530136e-04,  8.80876286e-04,  9.69766695e-04,  1.08542399e-03,
        #            1.02234479e-03,  1.19674041e-03,  1.00085557e-03,  1.18628543e-03,
        #            1.08546289e-03,  9.35106273e-04,  9.42886673e-04,  9.62309590e-04,
        #            1.16177603e-03,  1.20901065e-03,  1.28760474e-03,  1.09929084e-03,
        #            9.98180117e-04,  1.04989648e-03,  1.25385310e-03,  1.09795742e-03,
        #            1.19881288e-03,  1.27158949e-03,  1.06625512e-03,  1.19265461e-03,
        #            1.10788573e-03,  3.93562055e-04,  5.22149600e-03,  3.39174810e-03,
        #            1.42249731e-03,  9.88170153e-04,  1.85269535e-03,  1.30101729e-03,
        #            0.00000000e+00,  1.24515675e-03,  1.68056202e-03,  2.25383875e-03,
        #            2.30351553e-03,  2.33030228e-03,  2.82345733e-03,  3.07035686e-03,
        #            3.26867024e-03,  3.67110417e-03,  3.45328643e-03,  3.42948006e-03,
        #            3.96175727e-03,  4.19770627e-03,  4.29857197e-03,  4.75408693e-03,
        #            5.22027026e-03,  5.25559663e-03,  5.10713336e-03,  5.12948429e-03,
        #            5.57959215e-03,  5.04536718e-03,  4.98999034e-03,  4.87411716e-03,
        #            4.81578452e-03,  4.83869876e-03,  4.90766655e-03,  4.33890564e-03,
        #            4.54655332e-03,  4.14532763e-03,  4.04341633e-03,  3.93354330e-03,
        #            3.70868444e-03,  3.54586576e-03,  3.52358442e-03,  3.80801058e-03,
        #            4.39534338e-03,  4.40545215e-03,  5.05287494e-03,  6.15281218e-03,
        #            6.53255825e-03,  7.19391830e-03,  8.45654290e-03,  1.04286027e-02,
        #            1.39780103e-02,  1.78802537e-02,  2.35128540e-02,  3.16961770e-02,
        #            4.02975326e-02,  5.78907854e-02,  1.48613253e-01,  5.87571200e-01,
        #            2.33012135e+00,  1.92750427e-01,  2.13261738e-02,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]


        # eps_ = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], les_z, les_eps)
        # del_ = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], les_z, les_del)
        # k_ztop = np.where(self.Gr.z_half>np.max(les_z0))
        z = np.multiply(les_z0,1.0)
        # print(np.shape(z))
        # print(np.shape(les_eps0))
        # print(np.shape(les_del0))
        # k_ztop = int(np.max(np.where(z < 700.0)[0]))
        # eps_ = np.interp(self.Gr.z_half[self.Gr.gw:k_ztop], les_z0, les_eps0)
        # del_ = np.interp(self.Gr.z_half[self.Gr.gw:k_ztop], les_z0, les_del0)

        # # k_ztop = int(np.max(np.where(z < 700.0)[0]))
        eps_ = np.multiply(self.Gr.z_half,0.0)
        del_ = np.multiply(self.Gr.z_half,0.0)
        eps_ = np.add(np.multiply(self.Gr.z_half,0.0),les_eps0[-1])
        del_ = np.add(np.multiply(self.Gr.z_half,0.0),les_del0[-1])
        eps_[self.Gr.gw:self.Gr.nzg-self.Gr.gw] = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], z, les_eps0)
        del_[self.Gr.gw:self.Gr.nzg-self.Gr.gw] = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], z, les_del0)

        for i in xrange(self.n_updrafts):
            input.zi = self.UpdVar.cloud_base[i]
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if TS.t>0*3600.0: # and self.Gr.z_half[k]>0.0
                    if (self.UpdVar.Area.values[i,k]>0.0) and (self.Gr.z_half[k]>20.0) and (self.Gr.z_half[k]<np.max(les_z0)):
                        self.entr_sc[i,k] = eps_[k-self.Gr.gw]
                        self.detr_sc[i,k] = del_[k-self.Gr.gw]
                    else:
                        self.entr_sc[i,k] = 0.0
                        self.detr_sc[i,k] = 0.0

                    self.buoyant_frac[i,k] = 0.0
                    self.buoyant_frac[i,k] = 0.0
                    self.b_mix[i,k] = self.EnvVar.B.values[k]
                else:
                    if self.UpdVar.Area.values[i,k]>0.0:
                        input.quadrature_order = quadrature_order
                        input.b = self.UpdVar.B.values[i,k]
                        input.w = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                        input.z = self.Gr.z_half[k]
                        input.af = self.UpdVar.Area.values[i,k]
                        input.tke = self.EnvVar.TKE.values[k]
                        input.ml = self.mixing_length[k]
                        input.qt_env = self.EnvVar.QT.values[k]
                        input.ql_env = self.EnvVar.QL.values[k]
                        input.H_env = self.EnvVar.H.values[k]
                        input.T_env = self.EnvVar.T.values[k]
                        input.b_env = self.EnvVar.B.values[k]
                        input.b_mean = GMV.B.values[k]
                        input.w_env = self.EnvVar.W.values[k]
                        input.H_up = self.UpdVar.H.values[i,k]
                        input.T_up = self.UpdVar.T.values[i,k]
                        input.qt_up = self.UpdVar.QT.values[i,k]
                        input.ql_up = self.UpdVar.QL.values[i,k]
                        input.p0 = self.Ref.p0_half[k]
                        input.alpha0 = self.Ref.alpha0_half[k]
                        input.env_Hvar = self.EnvVar.Hvar.values[k]
                        input.env_QTvar = self.EnvVar.QTvar.values[k]
                        input.env_HQTcov = self.EnvVar.HQTcov.values[k]
                        input.c_eps = self.entrainment_factor
                        input.erf_const = self.entrainment_erf_const
                        input.c_del = self.detrainment_factor
                        if TS.t>6*3600.0:
                            input.nh_pressure = self.nh_pressure[i,k]
                        else:
                            input.nh_pressure = 0.0
                        if self.calc_tke:
                                input.tke = self.EnvVar.TKE.values[k]

                        input.T_mean = (self.EnvVar.T.values[k]+self.UpdVar.T.values[i,k])/2
                        input.L = 20000.0 # need to define the scale of the GCM grid resolution
                        ## Ignacio
                        if input.zbl-self.UpdVar.cloud_base[i] > 0.0:
                            input.poisson = np.random.poisson(self.Gr.dz/((input.zbl-self.UpdVar.cloud_base[i])/10.0))
                        else:
                            input.poisson = 0.0
                        ## End: Ignacio
                        ret = self.entr_detr_fp(input)
                        self.entr_sc[i,k] = ret.entr_sc
                        self.detr_sc[i,k] = ret.detr_sc
                        self.buoyant_frac[i,k] = ret.buoyant_frac
                        self.b_mix[i,k] = ret.b_mix

                    else:
                        self.entr_sc[i,k] = 0.0
                        self.detr_sc[i,k] = 0.0
                        self.buoyant_frac[i,k] = 0.0
                        self.buoyant_frac[i,k] = 0.0
                        self.b_mix[i,k] = self.EnvVar.B.values[k]



        return

    cpdef double compute_zbl_qt_grad(self, GridMeanVariables GMV):
    # computes inversion height as z with max gradient of qt
        cdef:
            double qt_up, qt_, z_
            double zbl_qt = 0.0
            double qt_grad = 0.0

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            z_ = self.Gr.z_half[k]
            qt_up = GMV.QT.values[k+1]
            qt_ = GMV.QT.values[k]

            if fabs(qt_up-qt_)*self.Gr.dzi > qt_grad:
                qt_grad = fabs(qt_up-qt_)*self.Gr.dzi
                zbl_qt = z_

        return zbl_qt

    cpdef compute_pressure_plume_spacing(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            double cpm, lv

        cpm = cpm_c(Case.Sur.qsurface)
        self.pressure_plume_spacing = fmax(cpm*Case.Sur.Tsurface*Case.Sur.bflux /(g*Case.Sur.ustar**2.0),self.Gr.dz)
        self.pressure_plume_spacing = 500.0
        return

    # cpdef compute_nh_pressure(self, TimeStepping TS):
    #     cdef:
    #         Py_ssize_t i, k
    #         double a_k, B_k, press_buoy, press_drag
    #     for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
    #         for i in xrange(self.n_updrafts):
    #             a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
    #             B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
    #             if a_k>0.0:
    #                 press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
    #                 press_drag = -1.0 * self.Ref.rho0[k] * sqrt(a_k) * (self.pressure_drag_coeff/self.pressure_plume_spacing
    #                                 * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])*fabs(self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k]))
    #                 self.nh_pressure[i,k] = press_buoy + press_drag
    #             else:
    #                 self.nh_pressure[i,k] = 0.0

    #     return


    cpdef compute_nh_pressure(self, TimeStepping TS):
        cdef:
            Py_ssize_t i, k
            double a_k, B_k, press_buoy, press_drag
            double t0 = 0*3600.0
        # from Jia's file
        # data = nc.Dataset('/Users/yaircohen/Downloads/dapdz_upd_1hourave_new.nc','r')
        # z = np.multiply(data.variables['z'],1.0)
        # dapdz_upd_ = np.multiply(data.variables['dapdz_upd'],1.0)
        # from pycles netCDF
        # data = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/closure_diagnostics/Bomex/SF100/stats/Stats.Bomex.nc','r')
        # # data = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/closure_diagnostics/DYCOMS/stats/Stats.DYCOMS_RF01.nc','r')
        # # data = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/clima_master/closure_diagnostics/TRMM_LBA/stats/Stats.TRMM_LBA.nc','r')
        # z = np.multiply(data.groups['reference'].variables['zp_half'],1.0)
        # dapdz_upd_ = np.multiply(np.nanmean(data.groups['profiles'].variables['updraft_ddz_p_alpha'][180:-1],axis=0),1.0)
        # dapdz_upd = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], z, dapdz_upd_)

        # specific updraft
        les_z = np.array([ 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0,
                   420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0])

        dpdz_upd = np.array([0.0, 0.0013605344748957257 ,0.0022549231758839726 ,0.0028587799249451275 ,0.0020621353002184353 ,0.0012685893330976123 ,0.00117602095555019 ,0.0011137422285152545 ,
                     0.0007153629267539907 ,0.0010090359032559858 ,0.0019254242564072847 ,0.0019709685903357946 ,0.001446410269250941 ,0.0010699210873611196 ,0.001593588877382966 ,
                     0.002213213706256346 ,0.00225419760215865 ,0.0015816597911974585 ,0.0010572863055320358 ,0.000936743353419042 ,0.0005690065472866644 ,0.0001711258724905599 ,
                     6.986517444010342e-05 ,-8.056958676153904e-05 ,-0.0007114959389552761 ,-0.0013023329947997636 ,-0.0014785663284440794 ,-0.0013169075421715676 ,-0.001441501108782491 ,
                     -0.0012020833329614959 ,0.0005057485300091287 ,0.0024063731807559786 ,0.003126271169412471 ,0.0029476837426758748 ,0.0016472478305815008])

        z = np.multiply(les_z,1.0)
        # k_ztop = int(np.max(np.where(z < 700.0)[0]))
        # dapdz_upd = np.add(np.multiply(self.Gr.z_half,0.0),0.0016472478305815008)
        dapdz_upd = np.multiply(self.Gr.z_half,0.0)
        dapdz_upd[self.Gr.gw:self.Gr.nzg-self.Gr.gw] = np.interp(self.Gr.z_half[self.Gr.gw:self.Gr.nzg-self.Gr.gw], z, dpdz_upd)

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            for i in xrange(self.n_updrafts):
                if TS.t>6*3600.0:
                    # print('1645')
                    # if dapdz_upd[k-self.Gr.gw]>1.0:
                    #     dapdz_upd[k-self.Gr.gw]=0.0

                    if (self.UpdVar.Area.values[i,k]>0.0) and (self.Gr.z_half[k]>20.0) and (self.Gr.z_half[k]<np.max(les_z)):
                        self.nh_pressure[i,k] = -self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]*dapdz_upd[k-self.Gr.gw]
                    else:
                        self.nh_pressure[i,k] = 0.0
                        # a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                        # B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        # if a_k>0.0:
                        #     press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
                        #     press_drag = -1.0 * self.Ref.rho0[k] * sqrt(a_k) * (self.pressure_drag_coeff/self.pressure_plume_spacing
                        #                     * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])*fabs(self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k]))
                        #     self.nh_pressure[i,k] = press_buoy + press_drag
                        # else:
                        #     self.nh_pressure[i,k] = 0.0

                    # if self.dapdz_upd[k-self.Gr.gw]>0.0:
                    #     self.nh_pressure[i,k] = -self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]*self.dapdz_upd[k-self.Gr.gw]
                    # else:
                    #     a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                    #     B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                    #     if a_k>0.0:
                    #         press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
                    #         press_drag = -1.0 * self.Ref.rho0[k] * sqrt(a_k) * (self.pressure_drag_coeff/self.pressure_plume_spacing
                    #                         * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])*fabs(self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k]))
                    #         self.nh_pressure[i,k] = press_buoy + press_drag
                    #     else:
                    #         self.nh_pressure[i,k] = 0.0
                else:
                    a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                    B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                    if a_k>0.0:
                        press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
                        press_drag = -1.0 * self.Ref.rho0[k] * sqrt(a_k) * (self.pressure_drag_coeff/self.pressure_plume_spacing
                                        * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])*fabs(self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k]))
                        self.nh_pressure[i,k] = press_buoy + press_drag
                    else:
                        self.nh_pressure[i,k] = 0.0
                    # print(self.Gr.z_half[k], self.nh_pressure[i,k], dapdz_upd[k-self.Gr.gw])

            # data_time = nc.Dataset(path,'r') + t0
            # ind2 = int(mt.ceil(TS.t/600.0))
            # ind1 = int(mt.trunc(TS.t/600.0))
            # if TS.t<(600.0+t0): # first 10 min use the radiative forcing of t=10min (as in the paper)
            #     for k in xrange(self.Fo.Gr.nzg):
            #         for i in xrange(self.n_updrafts):
            #             self.nh_pressure[i,k] = data[0,k]
            # elif TS.t>(18900.0+t0):
            #     for k in xrange(self.Fo.Gr.nzg):
            #         for i in xrange(self.n_updrafts):
            #             self.nh_pressure[i,k] = (data[31,k]-data[30,k])/(data_time[31]-data_time[30])\
            #                                   *(np.max(data_time)/60.0-data_time[30])+data[30,k]

            # else:
            #     if TS.t%(600.0+t0) == 0:
            #         for k in xrange(self.Fo.Gr.nzg):
            #             for i in xrange(self.n_updrafts):
            #                 self.nh_pressure[i,k] = data[ind1,k]
            #     else: # in all other cases - interpolate
            #         for k in xrange(self.Fo.Gr.nzg):
            #             for i in xrange(self.n_updrafts):
            #                 self.nh_pressure[i,k]    = (data[ind2,k]-data[ind1,k])\
            #                         /(data_time[ind2]-data_time[ind1])\
            #                         *(TS.t/60.0-_data_time[ind1])+data[ind1,k]

        return

    cpdef zero_area_fraction_cleanup(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i, k

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            for i in xrange(self.n_updrafts):
                if self.UpdVar.Area.values[i,k]<self.minimum_area:
                    self.UpdVar.Area.values[i,k] = 0.0
                    self.UpdVar.W.values[i,k] = GMV.W.values[k]
                    self.UpdVar.B.values[i,k] = GMV.B.values[k]
                    self.UpdVar.H.values[i,k] = GMV.H.values[k]
                    self.UpdVar.QT.values[i,k] = GMV.QT.values[k]
                    self.UpdVar.T.values[i,k] = GMV.T.values[k]
                    self.UpdVar.QL.values[i,k] = GMV.QL.values[k]
                    self.UpdVar.QR.values[i,k] = GMV.QR.values[k]
                    self.UpdVar.THL.values[i,k] = GMV.THL.values[k]

            if np.sum(self.UpdVar.Area.values[:,k])==0.0:
                self.EnvVar.W.values[k] = GMV.W.values[k]
                self.EnvVar.B.values[k] = GMV.B.values[k]
                self.EnvVar.H.values[k] = GMV.H.values[k]
                self.EnvVar.QT.values[k] = GMV.QT.values[k]
                self.EnvVar.T.values[k] = GMV.T.values[k]
                self.EnvVar.QL.values[k] = GMV.QL.values[k]
                self.EnvVar.QR.values[k] = GMV.QR.values[k]
                self.EnvVar.THL.values[k] = GMV.THL.values[k]

        return


    cpdef set_subdomain_bcs(self):

        self.UpdVar.W.set_bcs(self.Gr)
        self.UpdVar.Area.set_bcs(self.Gr)
        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.QR.set_bcs(self.Gr)
        self.UpdVar.T.set_bcs(self.Gr)
        self.UpdVar.B.set_bcs(self.Gr)

        self.EnvVar.W.set_bcs(self.Gr)
        self.EnvVar.H.set_bcs(self.Gr)
        self.EnvVar.T.set_bcs(self.Gr)
        self.EnvVar.QL.set_bcs(self.Gr)
        self.EnvVar.QT.set_bcs(self.Gr)

        return


    cpdef solve_updraft_velocity_area(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dzi = self.Gr.dzi
            double dti_ = 1.0/self.dt_upd
            double dt_ = 1.0/dti_
            double whalf_kp, whalf_k
            double au_lim
            double anew_k, a_k, a_km, entr_w, detr_w, B_k, entr_term, detr_term, rho_ratio
            double adv, buoy, exch # groupings of terms in velocity discrete equation

        with nogil:
            for i in xrange(self.n_updrafts):
                self.entr_sc[i,gw] = 2.0 * dzi # 0.0 ?
                self.detr_sc[i,gw] = 0.0
                self.UpdVar.W.new[i,gw-1] = self.w_surface_bc[i]
                self.UpdVar.Area.new[i,gw] = self.area_surface_bc[i]
                au_lim = self.area_surface_bc[i] * self.max_area_factor

                for k in range(gw, self.Gr.nzg-gw):

                    # First solve for updated area fraction at k+1
                    whalf_kp = interp2pt(self.UpdVar.W.values[i,k], self.UpdVar.W.values[i,k+1])
                    whalf_k = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    adv = -self.Ref.alpha0_half[k+1] * dzi *( self.Ref.rho0_half[k+1] * self.UpdVar.Area.values[i,k+1] * whalf_kp
                                                              -self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * whalf_k)
                    entr_term = self.UpdVar.Area.values[i,k+1] * whalf_kp * (self.entr_sc[i,k+1] )
                    detr_term = self.UpdVar.Area.values[i,k+1] * whalf_kp * (- self.detr_sc[i,k+1])


                    self.UpdVar.Area.new[i,k+1]  = fmax(dt_ * (adv + entr_term + detr_term) + self.UpdVar.Area.values[i,k+1], 0.0)
                    if self.UpdVar.Area.new[i,k+1] > au_lim:
                        self.UpdVar.Area.new[i,k+1] = au_lim
                        if self.UpdVar.Area.values[i,k+1] > 0.0:
                            self.detr_sc[i,k+1] = (((au_lim-self.UpdVar.Area.values[i,k+1])* dti_ - adv -entr_term)/(-self.UpdVar.Area.values[i,k+1]  * whalf_kp))
                        else:
                            # this detrainment rate won't affect scalars but would affect velocity
                            self.detr_sc[i,k+1] = (((au_lim-self.UpdVar.Area.values[i,k+1])* dti_ - adv -entr_term)/(-au_lim  * whalf_kp))

                    # Now solve for updraft velocity at k
                    rho_ratio = self.Ref.rho0[k-1]/self.Ref.rho0[k]
                    anew_k = interp2pt(self.UpdVar.Area.new[i,k], self.UpdVar.Area.new[i,k+1])
                    if anew_k >= self.minimum_area:
                        a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                        a_km = interp2pt(self.UpdVar.Area.values[i,k-1], self.UpdVar.Area.values[i,k])
                        entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                        detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                        B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        adv = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * self.UpdVar.W.values[i,k] * dzi
                               - self.Ref.rho0[k-1] * a_km * self.UpdVar.W.values[i,k-1] * self.UpdVar.W.values[i,k-1] * dzi)
                        exch = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k]
                                * (entr_w * self.EnvVar.W.values[k] - detr_w * self.UpdVar.W.values[i,k] ) + self.turb_entr_W[i,k])
                        buoy= self.Ref.rho0[k] * a_k * B_k
                        self.UpdVar.W.new[i,k] = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * dti_
                                                  -adv + exch + buoy + self.nh_pressure[i,k])/(self.Ref.rho0[k] * anew_k * dti_)

                        if self.UpdVar.W.new[i,k] <= 0.0:
                            self.UpdVar.W.new[i,k:] = 0.0
                            self.UpdVar.Area.new[i,k+1:] = 0.0
                            break
                    else:
                        self.UpdVar.W.new[i,k:] = 0.0
                        self.UpdVar.Area.new[i,k+1:] = 0.0
                        # keep this in mind if we modify updraft top treatment!
                        break

        return

    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            double dzi = self.Gr.dzi
            double dti_ = 1.0/self.dt_upd
            double m_k, m_km
            Py_ssize_t gw = self.Gr.gw
            double H_entr, QT_entr
            double c1, c2, c3, c4
            eos_struct sa
            double qt_var, h_var

        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.H.new[i,gw] = self.h_surface_bc[i]
                self.UpdVar.QT.new[i,gw] = self.qt_surface_bc[i]
                self.UpdVar.QR.new[i,gw] = 0.0 #TODO

                if self.use_local_micro:
                    # do saturation adjustment
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp,
                             self.Ref.p0_half[gw], self.UpdVar.QT.new[i,gw], self.UpdVar.H.new[i,gw])
                    self.UpdVar.QL.new[i,gw] = sa.ql
                    self.UpdVar.T.new[i,gw] = sa.T
                    # remove precipitation (update QT, QL and H)
                    self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[gw], self.UpdVar.T.new[i,gw],
                                                                       &self.UpdVar.QT.new[i,gw], &self.UpdVar.QL.new[i,gw],
                                                                       &self.UpdVar.QR.new[i,gw], &self.UpdVar.H.new[i,gw],
                                                                       i, gw)

                # starting from the bottom do entrainment at each level
                for k in xrange(gw+1, self.Gr.nzg-gw):
                    H_entr = self.EnvVar.H.values[k]
                    QT_entr = self.EnvVar.QT.values[k]

                    # write the discrete equations in form:
                    # c1 * phi_new[k] = c2 * phi[k] + c3 * phi[k-1] + c4 * phi_entr
                    if self.UpdVar.Area.new[i,k] >= self.minimum_area:
                        m_k = (self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k]
                               * interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k]))
                        m_km = (self.Ref.rho0_half[k-1] * self.UpdVar.Area.values[i,k-1]
                               * interp2pt(self.UpdVar.W.values[i,k-2], self.UpdVar.W.values[i,k-1]))
                        c1 = self.Ref.rho0_half[k] * self.UpdVar.Area.new[i,k] * dti_
                        c2 = (self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * dti_
                              - m_k * (dzi + self.detr_sc[i,k]))
                        c3 = m_km * dzi
                        c4 = m_k * self.entr_sc[i,k]

                        self.UpdVar.H.new[i,k] =  (c2 * self.UpdVar.H.values[i,k]  + c3 * self.UpdVar.H.values[i,k-1]
                                                   + c4 * H_entr + self.turb_entr_H[i,k])/c1
                        self.UpdVar.QT.new[i,k] = (c2 * self.UpdVar.QT.values[i,k] + c3 * self.UpdVar.QT.values[i,k-1]
                                                   + c4* QT_entr + self.turb_entr_QT[i,k])/c1
                    else:
                        self.UpdVar.H.new[i,k] = GMV.H.values[k]
                        self.UpdVar.QT.new[i,k] = GMV.QT.values[k]

                    # find new temperature
                    sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                             self.UpdVar.QT.new[i,k], self.UpdVar.H.new[i,k])
                    self.UpdVar.QL.new[i,k] = sa.ql
                    self.UpdVar.T.new[i,k] = sa.T

                    if self.use_local_micro:
                        # remove precipitation (pdate QT, QL and H)
                        self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half[k], self.UpdVar.T.new[i,k],
                                                                       &self.UpdVar.QT.new[i,k], &self.UpdVar.QL.new[i,k],
                                                                       &self.UpdVar.QR.new[i,k], &self.UpdVar.H.new[i,k],
                                                                       i, k)

        if self.use_local_micro:
            # save the total source terms for H and QT due to precipitation
            # TODO - add QR source
            self.UpdMicro.prec_source_h_tot = np.sum(np.multiply(self.UpdMicro.prec_source_h,
                                                                 self.UpdVar.Area.values), axis=0)
            self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt,
                                                                  self.UpdVar.Area.values), axis=0)
        else:
            # Compute the updraft microphysical sources (precipitation)
            #after the entrainment loop is finished
            self.UpdMicro.compute_sources(self.UpdVar)
            # Update updraft variables with microphysical source tendencies
            self.UpdMicro.update_updraftvars(self.UpdVar)

        return

    # After updating the updraft variables themselves:
    # 1. compute the mass fluxes (currently not stored as class members, probably will want to do this
    # for output purposes)
    # 2. Apply mass flux tendencies and updraft microphysical tendencies to GMV.SomeVar.Values (old time step values)
    # thereby updating to GMV.SomeVar.mf_update
    # mass flux tendency is computed as 1st order upwind

    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            Py_ssize_t k, i
            Py_ssize_t gw = self.Gr.gw
            double mf_tend_h=0.0, mf_tend_qt=0.0
            double env_h_interp, env_qt_interp
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        # Compute the mass flux and associated scalar fluxes
        with nogil:
            for i in xrange(self.n_updrafts):
                self.m[i,gw-1] = 0.0
                for k in xrange(self.Gr.gw, self.Gr.nzg-1):
                    a = interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1])
                    self.m[i,k] =  self.Ref.rho0[k]*a*interp2pt(ae[k],ae[k+1])*(self.UpdVar.W.values[i,k] - self.EnvVar.W.values[k])


        self.massflux_h[gw-1] = 0.0
        self.massflux_qt[gw-1] = 0.0
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                self.massflux_h[k] = 0.0
                self.massflux_qt[k] = 0.0
                env_h_interp = interp2pt(self.EnvVar.H.values[k], self.EnvVar.H.values[k+1])
                env_qt_interp = interp2pt(self.EnvVar.QT.values[k], self.EnvVar.QT.values[k+1])
                for i in xrange(self.n_updrafts):
                    self.massflux_h[k] += self.m[i,k] * (interp2pt(self.UpdVar.H.values[i,k],
                                                                   self.UpdVar.H.values[i,k+1]) - env_h_interp )
                    self.massflux_qt[k] += self.m[i,k] * (interp2pt(self.UpdVar.QT.values[i,k],
                                                                    self.UpdVar.QT.values[i,k+1]) - env_qt_interp )

        # Compute the  mass flux tendencies
        # Adjust the values of the grid mean variables
        with nogil:

            for k in xrange(self.Gr.gw, self.Gr.nzg):
                mf_tend_h = -(self.massflux_h[k] - self.massflux_h[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)
                mf_tend_qt = -(self.massflux_qt[k] - self.massflux_qt[k-1]) * (self.Ref.alpha0_half[k] * self.Gr.dzi)

                GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h + self.UpdMicro.prec_source_h_tot[k]
                GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt + self.UpdMicro.prec_source_qt_tot[k]

                #No mass flux tendency for U, V
                GMV.U.mf_update[k] = GMV.U.values[k]
                GMV.V.mf_update[k] = GMV.V.values[k]
                # Prepare the output
                self.massflux_tendency_h[k] = mf_tend_h
                self.massflux_tendency_qt[k] = mf_tend_qt

        return

    # Update the grid mean variables with the tendency due to eddy diffusion
    # Km and Kh have already been updated
    # 2nd order finite differences plus implicit time step allows solution with tridiagonal matrix solver
    # Update from GMV.SomeVar.mf_update to GMV.SomeVar.new
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double dzi = self.Gr.dzi
            double [:] a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] x = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double [:] rho_ae_K = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, &rho_ae_K[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])

        # Solve QT
        with nogil:
            for k in xrange(nz):
                x[k] =  self.EnvVar.QT.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_qtflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.QT.new[k+gw] = GMV.QT.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.QT.values[k+gw])
                self.diffusive_tendency_qt[k+gw] = (GMV.QT.new[k+gw] - GMV.QT.mf_update[k+gw]) * TS.dti
            # get the diffusive flux
            self.diffusive_flux_qt[gw] = interp2pt(Case.Sur.rho_qtflux, -rho_ae_K[gw] * dzi *(self.EnvVar.QT.values[gw+1]-self.EnvVar.QT.values[gw]) )
            for k in xrange(self.Gr.gw+1, self.Gr.nzg-self.Gr.gw):
                self.diffusive_flux_qt[k] = -0.5 * self.Ref.rho0_half[k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.QT.values[k+1]-self.EnvVar.QT.values[k-1])

        # Solve H
        with nogil:
            for k in xrange(nz):
                x[k] = self.EnvVar.H.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_hflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.H.new[k+gw] = GMV.H.mf_update[k+gw] + ae[k+gw] *(x[k] - self.EnvVar.H.values[k+gw])
                self.diffusive_tendency_h[k+gw] = (GMV.H.new[k+gw] - GMV.H.mf_update[k+gw]) * TS.dti
            # get the diffusive flux
            self.diffusive_flux_h[gw] = interp2pt(Case.Sur.rho_hflux, -rho_ae_K[gw] * dzi *(self.EnvVar.H.values[gw+1]-self.EnvVar.H.values[gw]) )
            for k in xrange(self.Gr.gw+1, self.Gr.nzg-self.Gr.gw):
                self.diffusive_flux_h[k] = -0.5 * self.Ref.rho0_half[k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.H.values[k+1]-self.EnvVar.H.values[k-1])

        # Solve U
        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K[k] = 0.5 * (ae[k]*self.KM.values[k]+ ae[k+1]*self.KM.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, &rho_ae_K[0], &self.Ref.rho0_half[0],
                                    &ae[0], &a[0], &b[0], &c[0])
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.U.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_uflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.U.new[k+gw] = x[k]

        # Solve V
        with nogil:
            for k in xrange(nz):
                x[k] = GMV.V.values[k+gw]
            x[0] = x[0] + TS.dt * Case.Sur.rho_vflux * dzi * self.Ref.alpha0_half[gw]/ae[gw]
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        with nogil:
            for k in xrange(nz):
                GMV.V.new[k+gw] = x[k]

        GMV.QT.set_bcs(self.Gr)
        GMV.QR.set_bcs(self.Gr)
        GMV.H.set_bcs(self.Gr)
        GMV.U.set_bcs(self.Gr)
        GMV.V.set_bcs(self.Gr)

        return

    cpdef compute_tke_buoy(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double d_alpha_thetal_dry, d_alpha_qt_dry
            double d_alpha_thetal_cloudy, d_alpha_qt_cloudy
            double d_alpha_thetal_total, d_alpha_qt_total
            double lh, prefactor, cpm
            double qt_dry, th_dry, t_cloudy, qv_cloudy, qt_cloudy, th_cloudy
            double grad_thl_minus=0.0, grad_qt_minus=0.0, grad_thl_plus=0, grad_qt_plus=0
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        # Note that source terms at the gw grid point are not really used because that is where tke boundary condition is
        # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                qt_dry = self.EnvThermo.qt_dry[k]
                th_dry = self.EnvThermo.th_dry[k]
                t_cloudy = self.EnvThermo.t_cloudy[k]
                qv_cloudy = self.EnvThermo.qv_cloudy[k]
                qt_cloudy = self.EnvThermo.qt_cloudy[k]
                th_cloudy = self.EnvThermo.th_cloudy[k]

                lh = latent_heat(t_cloudy)
                cpm = cpm_c(qt_cloudy)
                grad_thl_minus = grad_thl_plus
                grad_qt_minus = grad_qt_plus
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.Gr.dzi

                prefactor = Rd * exner_c(self.Ref.p0_half[k])/self.Ref.p0_half[k]

                d_alpha_thetal_dry = prefactor * (1.0 + (eps_vi-1.0) * qt_dry)
                d_alpha_qt_dry = prefactor * th_dry * (eps_vi-1.0)

                if self.EnvVar.CF.values[k] > 0.0:
                    d_alpha_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_alpha_qt_cloudy = (lh / cpm / t_cloudy * d_alpha_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_alpha_thetal_cloudy = 0.0
                    d_alpha_qt_cloudy = 0.0

                d_alpha_thetal_total = (self.EnvVar.CF.values[k] * d_alpha_thetal_cloudy
                                        + (1.0-self.EnvVar.CF.values[k]) * d_alpha_thetal_dry)
                d_alpha_qt_total = (self.EnvVar.CF.values[k] * d_alpha_qt_cloudy
                                    + (1.0-self.EnvVar.CF.values[k]) * d_alpha_qt_dry)

                # TODO - check
                self.EnvVar.TKE.buoy[k] = g / self.Ref.alpha0_half[k] * ae[k] * self.Ref.rho0_half[k] \
                                   * ( \
                                       - self.KH.values[k] * interp2pt(grad_thl_plus, grad_thl_minus) * d_alpha_thetal_total \
                                       - self.KH.values[k] * interp2pt(grad_qt_plus,  grad_qt_minus)  * d_alpha_qt_total\
                                     )
        return

    cpdef compute_tke_pressure(self):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double wu_half, we_half, press_half

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.EnvVar.TKE.press[k] = 0.0
                for i in xrange(self.n_updrafts):
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    we_half = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                    press_half = interp2pt(self.nh_pressure[i,k-1], self.nh_pressure[i,k])
                    self.EnvVar.TKE.press[k] += (we_half - wu_half) * press_half
        return



    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            double qv, alpha


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.QL.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QL.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QL.values[k])

                # TODO - change to prognostic?
                GMV.QR.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QR.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QR.values[k])

                GMV.T.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.T.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.T.values[k])
                qv = GMV.QT.values[k] - GMV.QL.values[k]

                GMV.THL.values[k] = t_to_thetali_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k],
                                                   GMV.QL.values[k], 0.0)
                GMV.B.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.B.bulkvalues[k]
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.B.values[k])


        return


    cpdef compute_covariance(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):

        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed
            self.compute_mixing_length(Case.Sur.obukhov_length, GMV)
        if self.calc_tke:
            self.compute_tke_buoy(GMV)
            self.compute_covariance_entr(self.EnvVar.TKE, self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W)
            self.compute_covariance_shear(GMV, self.EnvVar.TKE, &self.UpdVar.W.values[0,0], &self.UpdVar.W.values[0,0], &self.EnvVar.W.values[0], &self.EnvVar.W.values[0])
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.W,self.UpdVar.W,self.EnvVar.W, self.EnvVar.W, self.EnvVar.TKE)
            self.compute_tke_pressure()
        if self.calc_scalar_var:
            self.compute_covariance_entr(self.EnvVar.Hvar, self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H)
            self.compute_covariance_entr(self.EnvVar.QTvar, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT)
            self.compute_covariance_entr(self.EnvVar.HQTcov, self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT)
            self.compute_covariance_shear(GMV, self.EnvVar.Hvar, &self.UpdVar.H.values[0,0], &self.UpdVar.H.values[0,0], &self.EnvVar.H.values[0], &self.EnvVar.H.values[0])
            self.compute_covariance_shear(GMV, self.EnvVar.QTvar, &self.UpdVar.QT.values[0,0], &self.UpdVar.QT.values[0,0], &self.EnvVar.QT.values[0], &self.EnvVar.QT.values[0])
            self.compute_covariance_shear(GMV, self.EnvVar.HQTcov, &self.UpdVar.H.values[0,0], &self.UpdVar.QT.values[0,0], &self.EnvVar.H.values[0], &self.EnvVar.QT.values[0])
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.H,self.UpdVar.H,self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.QT,self.UpdVar.QT,self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.H,self.UpdVar.QT,self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov)
            self.compute_covariance_rain(TS, GMV) # need to update this one

        self.reset_surface_covariance(GMV, Case)
        if self.calc_tke:
            self.update_covariance_ED(GMV, Case,TS, GMV.W, GMV.W, GMV.TKE, self.EnvVar.TKE, self.EnvVar.W, self.EnvVar.W, self.UpdVar.W, self.UpdVar.W)
        if self.calc_scalar_var:
            self.update_covariance_ED(GMV, Case,TS, GMV.H, GMV.H, GMV.Hvar, self.EnvVar.Hvar, self.EnvVar.H, self.EnvVar.H, self.UpdVar.H, self.UpdVar.H)
            self.update_covariance_ED(GMV, Case,TS, GMV.QT,GMV.QT, GMV.QTvar, self.EnvVar.QTvar, self.EnvVar.QT, self.EnvVar.QT, self.UpdVar.QT, self.UpdVar.QT)
            self.update_covariance_ED(GMV, Case,TS, GMV.H, GMV.QT, GMV.HQTcov, self.EnvVar.HQTcov, self.EnvVar.H, self.EnvVar.QT, self.UpdVar.H, self.UpdVar.QT)
            self.cleanup_covariance(GMV)
        return



    cpdef initialize_covariance(self, GridMeanVariables GMV, CasesBase Case):

        cdef:
            Py_ssize_t k
            double ws= self.wstar, us = Case.Sur.ustar, zs = self.zi, z

        self.reset_surface_covariance(GMV, Case)

        if self.calc_tke:
            if ws > 0.0:
                with nogil:
                    for k in xrange(self.Gr.nzg):
                        z = self.Gr.z_half[k]
                        GMV.TKE.values[k] = ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
            # TKE initialization from Beare et al, 2006
            if Case.casename =='GABLS':
                with nogil:
                    for k in xrange(self.Gr.nzg):
                        z = self.Gr.z_half[k]
                        if (z<=250.0):
                            GMV.TKE.values[k] = 0.4*(1.0-z/250.0)*(1.0-z/250.0)*(1.0-z/250.0)
        if self.calc_scalar_var:
            if ws > 0.0:
                with nogil:
                    for k in xrange(self.Gr.nzg):
                        z = self.Gr.z_half[k]
                        # need to rethink of how to initilize the covarinace profiles - for nowmI took the TKE profile
                        GMV.Hvar.values[k]   = GMV.Hvar.values[self.Gr.gw] * ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
                        GMV.QTvar.values[k]  = GMV.QTvar.values[self.Gr.gw] * ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
                        GMV.HQTcov.values[k] = GMV.HQTcov.values[self.Gr.gw] * ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
            # TKE initialization from Beare et al, 2006
            if Case.casename =='GABLS':
                with nogil:
                    for k in xrange(self.Gr.nzg):
                        z = self.Gr.z_half[k]
                        if (z<=250.0):
                            GMV.Hvar.values[k] = 0.4*(1.0-z/250.0)*(1.0-z/250.0)*(1.0-z/250.0)
                        GMV.QTvar.values[k]  = 0.0
                        GMV.HQTcov.values[k] = 0.0
        self.compute_mixing_length(Case.Sur.obukhov_length, GMV)
        return


    cpdef cleanup_covariance(self, GridMeanVariables GMV):
        cdef:
            double tmp_eps = 1e-18

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if GMV.TKE.values[k] < tmp_eps:
                    GMV.TKE.values[k] = 0.0
                if GMV.Hvar.values[k] < tmp_eps:
                    GMV.Hvar.values[k] = 0.0
                if GMV.QTvar.values[k] < tmp_eps:
                    GMV.QTvar.values[k] = 0.0
                if fabs(GMV.HQTcov.values[k]) < tmp_eps:
                    GMV.HQTcov.values[k] = 0.0
                if self.EnvVar.Hvar.values[k] < tmp_eps:
                    self.EnvVar.Hvar.values[k] = 0.0
                if self.EnvVar.TKE.values[k] < tmp_eps:
                    self.EnvVar.TKE.values[k] = 0.0
                if self.EnvVar.QTvar.values[k] < tmp_eps:
                    self.EnvVar.QTvar.values[k] = 0.0
                if fabs(self.EnvVar.HQTcov.values[k]) < tmp_eps:
                    self.EnvVar.HQTcov.values[k] = 0.0


    cdef void compute_covariance_shear(self,GridMeanVariables GMV, EDMF_Environment.EnvironmentVariable_2m Covar, double *UpdVar1, double *UpdVar2, double *EnvVar1, double *EnvVar2):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)
            double diff_var1 = 0.0
            double diff_var2 = 0.0
            double du = 0.0
            double dv = 0.0
            double tke_factor = 1.0
            double du_low, dv_low
            double du_high = 0.0
            double dv_high = 0.0
            double k_eddy

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            if Covar.name == 'tke':
                du_low = du_high
                dv_low = dv_high
                du_high = (GMV.U.values[k+1] - GMV.U.values[k]) * self.Gr.dzi
                dv_high = (GMV.V.values[k+1] - GMV.V.values[k]) * self.Gr.dzi
                diff_var2 = (EnvVar2[k] - EnvVar2[k-1]) * self.Gr.dzi
                diff_var1 = (EnvVar1[k] - EnvVar1[k-1]) * self.Gr.dzi
                tke_factor = 0.5
                k_eddy = self.KM.values[k]
            else:
            # Defined correctly only for covariance between half-level variables.
                du_low = 0.0
                dv_low = 0.0
                du_high = 0.0
                dv_high = 0.0
                diff_var2 = interp2pt((EnvVar2[k+1] - EnvVar2[k]),(EnvVar2[k] - EnvVar2[k-1])) * self.Gr.dzi
                diff_var1 = interp2pt((EnvVar1[k+1] - EnvVar1[k]),(EnvVar1[k] - EnvVar1[k-1])) * self.Gr.dzi
                tke_factor = 1.0
                k_eddy = self.KH.values[k]
            with nogil:
                Covar.shear[k] = tke_factor*2.0*(self.Ref.rho0_half[k] * ae[k] * k_eddy *
                            (diff_var1*diff_var2 +  pow(interp2pt(du_low, du_high),2.0)  +  pow(interp2pt(dv_low, dv_high),2.0)))
        return

    cdef void compute_covariance_interdomain_src(self, EDMF_Updrafts.UpdraftVariable au,
                        EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                        EDMF_Environment.EnvironmentVariable phi_e,  EDMF_Environment.EnvironmentVariable psi_e,
                        EDMF_Environment.EnvironmentVariable_2m Covar):
        cdef:
            Py_ssize_t i,k
            double phi_diff, psi_diff, tke_factor

        #with nogil:
        for k in xrange(self.Gr.nzg):
            Covar.interdomain[k] = 0.0
            for i in xrange(self.n_updrafts):
                if Covar.name == 'tke':
                    tke_factor = 0.5
                    phi_diff = interp2pt(phi_u.values[i,k-1], phi_u.values[i,k])-interp2pt(phi_e.values[k-1], phi_e.values[k])
                    psi_diff = interp2pt(psi_u.values[i,k-1], psi_u.values[i,k])-interp2pt(psi_e.values[k-1], psi_e.values[k])
                else:
                    tke_factor = 1.0
                    phi_diff = phi_u.values[i,k]-phi_e.values[k]
                    psi_diff = psi_u.values[i,k]-psi_e.values[k]

                Covar.interdomain[k] += tke_factor*au.values[i,k] * (1.0-au.values[i,k]) * phi_diff * psi_diff
        return

    cdef void compute_covariance_entr(self, EDMF_Environment.EnvironmentVariable_2m Covar, EDMF_Updrafts.UpdraftVariable UpdVar1,
                EDMF_Updrafts.UpdraftVariable UpdVar2, EDMF_Environment.EnvironmentVariable EnvVar1, EDMF_Environment.EnvironmentVariable EnvVar2):
        cdef:
            Py_ssize_t i, k
            double tke_factor
            double updvar1, updvar2, envvar1, envvar2, combined_entr, combined_detr, K

        # here the diffusive componenet of trhe turbulent entrainment is added to the dynamic entr and detrainment
        #with nogil:
        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            Covar.entr_gain[k] = 0.0
            Covar.detr_loss[k] = 0.0
            for i in xrange(self.n_updrafts):
                if self.UpdVar.Area.values[i,k] > self.minimum_area:
                    R_up = self.pressure_plume_spacing*sqrt(self.UpdVar.Area.values[i,k])
                    if Covar.name =='tke':
                        updvar1 = interp2pt(UpdVar1.values[i,k], UpdVar1.values[i,k-1])
                        updvar2 = interp2pt(UpdVar2.values[i,k], UpdVar2.values[i,k-1])
                        envvar1 = interp2pt(EnvVar1.values[k], EnvVar1.values[k-1])
                        envvar2 = interp2pt(EnvVar2.values[k], EnvVar2.values[k-1])
                        tke_factor = 0.5
                        K = self.horizontal_KM[i,k]
                    else:
                        updvar1 = UpdVar1.values[i,k]
                        updvar2 = UpdVar2.values[i,k]
                        envvar1 = EnvVar1.values[k]
                        envvar2 = EnvVar2.values[k]
                        tke_factor = 1.0
                        K = self.horizontal_KH[i,k]
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    combined_entr = self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k] * fabs(w_u)*self.detr_sc[i,k]\
                                     + 2.0/(R_up**2.0)*self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]*K
                    combined_detr = self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k] * fabs(w_u)*self.entr_sc[i,k]\
                                     + 2.0/(R_up**2.0)*self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]*K

                    Covar.entr_gain[k]  -= tke_factor*combined_entr * (updvar1 - envvar1) * (updvar2 - envvar2)
                    Covar.detr_loss[k]  -= combined_detr * Covar.values[k]
        return


    cdef void compute_covariance_detr(self, EDMF_Environment.EnvironmentVariable_2m Covar):
        cdef:
            Py_ssize_t i, k
        #with nogil:
        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            Covar.detr_loss[k] = 0.0
            for i in xrange(self.n_updrafts):
                w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                Covar.detr_loss[k] += self.UpdVar.Area.values[i,k] * fabs(w_u) * self.entr_sc[i,k]
            Covar.detr_loss[k] *= self.Ref.rho0_half[k] * Covar.values[k]
        return

    cpdef compute_covariance_rain(self, TimeStepping TS, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i, k
            # TODO defined again in compute_covariance_shear and compute_covaraince
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.EnvVar.TKE.rain_src[k] = 0.0
                self.EnvVar.Hvar.rain_src[k]   = self.Ref.rho0_half[k] * ae[k] * 2. * self.EnvThermo.Hvar_rain_dt[k]   * TS.dti
                self.EnvVar.QTvar.rain_src[k]  = self.Ref.rho0_half[k] * ae[k] * 2. * self.EnvThermo.QTvar_rain_dt[k]  * TS.dti
                self.EnvVar.HQTcov.rain_src[k] = self.Ref.rho0_half[k] * ae[k] *      self.EnvThermo.HQTcov_rain_dt[k] * TS.dti

        return


    cdef void compute_covariance_dissipation(self, EDMF_Environment.EnvironmentVariable_2m Covar):
        cdef:
            Py_ssize_t i
            double m
            Py_ssize_t k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                Covar.dissipation[k] = (self.Ref.rho0_half[k] * ae[k] * Covar.values[k]
                                    *pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0e-3) * self.tke_diss_coeff)
        return


    cpdef compute_tke_advection(self):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double drho_ae_we_e_minus
            double drho_ae_we_e_plus = 0.0

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                drho_ae_we_e_minus = drho_ae_we_e_plus
                drho_ae_we_e_plus = (self.Ref.rho0_half[k+1] * ae[k+1] *self.EnvVar.TKE.values[k+1]
                    * (self.EnvVar.W.values[k+1] + self.EnvVar.W.values[k])/2.0
                    - self.Ref.rho0_half[k] * ae[k] * self.EnvVar.TKE.values[k]
                    * (self.EnvVar.W.values[k] + self.EnvVar.W.values[k-1])/2.0 ) * self.Gr.dzi
                self.tke_advection[k] = interp2pt(drho_ae_we_e_minus, drho_ae_we_e_plus)
        return

    cpdef compute_tke_transport(self):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
            double dtke_high = 0.0
            double dtke_low
            double rho_ae_K_m_plus
            double drho_ae_K_m_de_plus = 0.0
            double drho_ae_K_m_de_low

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw-1):
                drho_ae_K_m_de_low = drho_ae_K_m_de_plus
                drho_ae_K_m_de_plus = (self.Ref.rho0_half[k+1] * ae[k+1] * self.KM.values[k+1] *
                    (self.EnvVar.TKE.values[k+2]-self.EnvVar.TKE.values[k])* 0.5 * self.Gr.dzi
                    - self.Ref.rho0_half[k] * ae[k] * self.KM.values[k] *
                    (self.EnvVar.TKE.values[k+1]-self.EnvVar.TKE.values[k-1])* 0.5 * self.Gr.dzi
                    ) * self.Gr.dzi
                self.tke_transport[k] = interp2pt(drho_ae_K_m_de_low, drho_ae_K_m_de_plus)
        return

    cdef void update_covariance_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS, VariablePrognostic GmvVar1, VariablePrognostic GmvVar2,
            VariableDiagnostic GmvCovar, EDMF_Environment.EnvironmentVariable_2m Covar, EDMF_Environment.EnvironmentVariable  EnvVar1, EDMF_Environment.EnvironmentVariable  EnvVar2,
                                   EDMF_Updrafts.UpdraftVariable  UpdVar1, EDMF_Updrafts.UpdraftVariable  UpdVar2):
        cdef:
            Py_ssize_t k, kk, i
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg
            Py_ssize_t nz = self.Gr.nz
            double dzi = self.Gr.dzi
            double dti = TS.dti
            double alpha0LL  = self.Ref.alpha0_half[self.Gr.gw]
            double zLL = self.Gr.z_half[self.Gr.gw]
            double [:] a = np.zeros((nz,),dtype=np.double, order='c')
            double [:] b = np.zeros((nz,),dtype=np.double, order='c')
            double [:] c = np.zeros((nz,),dtype=np.double, order='c')
            double [:] x = np.zeros((nz,),dtype=np.double, order='c')
            double [:] ae = np.subtract(np.ones((nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)
            double [:] ae_old = np.subtract(np.ones((nzg,),dtype=np.double, order='c'), np.sum(self.UpdVar.Area.old,axis=0))
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')
            double [:] whalf = np.zeros((nzg,),dtype=np.double, order='c')
            double  D_env = 0.0
            double Covar_surf, wu_half, K, Kp

        for k in xrange(1,nzg-1):
            if  Covar.name == 'tke':
                K = self.KM.values[k]
                Kp = self.KM.values[k+1]
            else:
                K = self.KH.values[k]
                Kp = self.KH.values[k+1]
            rho_ae_K_m[k] = 0.5 * (ae[k]*K+ ae[k+1]*Kp)* self.Ref.rho0[k]
            whalf[k] = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
        wu_half = interp2pt(self.UpdVar.W.bulkvalues[gw-1], self.UpdVar.W.bulkvalues[gw])

        if GmvCovar.name=='tke':
            GmvCovar.values[gw] =get_surface_tke(Case.Sur.ustar, self.wstar, self.Gr.z_half[gw], Case.Sur.obukhov_length)

        elif GmvCovar.name=='thetal_var':
            GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_hflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        elif GmvCovar.name=='qt_var':
            GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_qtflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        elif GmvCovar.name=='thetal_qt_covar':
            GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)

        self.get_env_covar_from_GMV(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, &GmvVar1.values[0], &GmvVar2.values[0], &GmvCovar.values[0])

        Covar_surf = Covar.values[gw]

        with nogil:
            for kk in xrange(nz):
                k = kk+gw
                D_env = 0.0

                for i in xrange(self.n_updrafts):
                    if self.UpdVar.Area.values[i,k]>self.minimum_area:
                        with gil:
                            if Covar.name == 'tke':
                                K = self.horizontal_KM[i,k]
                            else:
                                K = self.horizontal_KH[i,k]

                            R_up = self.pressure_plume_spacing*sqrt(self.UpdVar.Area.values[i,k])
                            wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                            D_env += self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * wu_half * self.entr_sc[i,k]\
                                     + 2.0/(R_up**2.0)*self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]*K
                    else:
                        D_env = 0.0

                a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                b[kk] = (self.Ref.rho0_half[k] * ae[k] * dti - self.Ref.rho0_half[k] * ae[k] * whalf[k] * dzi
                         + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                         + D_env
                         + self.Ref.rho0_half[k] * ae[k] * self.tke_diss_coeff
                                    *sqrt(fmax(self.EnvVar.TKE.values[k],0))/fmax(self.mixing_length[k],1.0) )
                c[kk] = (self.Ref.rho0_half[k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                x[kk] = (self.Ref.rho0_half[k] * ae_old[k] * Covar.values[k] * dti
                         + Covar.press[k] + Covar.buoy[k] + Covar.shear[k] + Covar.entr_gain[k] +  Covar.rain_src[k]) #

                a[0] = 0.0
                b[0] = 1.0
                c[0] = 0.0
                x[0] = Covar_surf

                b[nz-1] += c[nz-1]
                c[nz-1] = 0.0
        tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

        for kk in xrange(nz):
            k = kk + gw
            if Covar.name == 'thetal_qt_covar':
                Covar.values[k] = fmax(x[kk], - sqrt(self.EnvVar.Hvar.values[k]*self.EnvVar.QTvar.values[k]))
                Covar.values[k] = fmin(x[kk],   sqrt(self.EnvVar.Hvar.values[k]*self.EnvVar.QTvar.values[k]))
            else:
                Covar.values[k] = fmax(x[kk],0.0)
        Covar.set_bcs(self.Gr)

        self.get_GMV_CoVar(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, &GmvVar1.values[0], &GmvVar2.values[0], &GmvCovar.values[0])

        return