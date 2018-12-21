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
import pylab as plt
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
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'functional_form':
                self.entr_detr_fp = entr_detr_functional_form
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'b_w2':
                self.entr_detr_fp = entr_detr_b_w2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'upd_specific':
                self.entr_detr_fp = entr_detr_upd_specific
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke':
                self.entr_detr_fp = entr_detr_tke
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke2':
                self.entr_detr_fp = entr_detr_tke2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'suselj':
                self.entr_detr_fp = entr_detr_suselj
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'Poisson_entr':
                self.entr_detr_fp = entr_detr_Poisson_entr
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'sum':
                self.entr_detr_fp = entr_detr_linear_sum
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
            self.mixing_scheme = 'tke'
            print 'Using tke mixing length formulation as default'

        # Get values from paramlist
        # set defaults at some point?
        self.surface_area = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.max_area_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        self.entrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
        self.detrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
        # adding tuning coefficients for power laws in non dimensional entrainment functions
        try:
            self.entrainment_alpha1 = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha1']
            self.entrainment_alpha2 = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha2']
            self.entrainment_alpha3 = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_alpha3']
            self.detrainment_alpha1 = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha1']
            self.detrainment_alpha2 = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha2']
            self.detrainment_alpha3 = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_alpha3']
        except:
            self.entrainment_alpha1 = 1.0
            self.entrainment_alpha2 = 1.0
            self.entrainment_alpha3 = 1.0
            self.detrainment_alpha1 = 1.0
            self.detrainment_alpha2 = 1.0
            self.detrainment_alpha3 = 1.0

        self.entrainment_alpha1 = 1.0
        self.entrainment_alpha2 = 1.0
        self.entrainment_alpha3 = 1.0
        self.detrainment_alpha1 = 1.0
        self.detrainment_alpha2 = 1.0
        self.detrainment_alpha3 = 1.0

        self.pressure_buoy_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.pressure_drag_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
        self.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
        # "Legacy" coefficients used by the steady updraft routine
        self.vel_pressure_coeff = self.pressure_drag_coeff/self.pressure_plume_spacing
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        if self.calc_tke == True:
            self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
            self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']

        # Need to code up as paramlist option?
        self.minimum_area = 1e-3

        # GCM domain size
        self.gcm_resolution = 20000.0

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

        # w pressure terms
        self.w_press_term = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Pressure term in updraft vertical momentum equation
        self.updraft_pressure_sink = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

        # mixing length
        self.mixing_length = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # turbulent entrainment
        self.turb_entr_W = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.turb_entr_H = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        self.turb_entr_QT = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

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
        self.mls = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.ml_ratio = np.zeros((Gr.nzg,),dtype=np.double, order='c')
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

        if self.calc_tke:
            Stats.add_profile('tke_buoy')
            Stats.add_profile('tke_dissipation')
            Stats.add_profile('tke_entr_gain')
            Stats.add_profile('turbulent_entrainment_W')
            Stats.add_profile('turbulent_entrainment_H')
            Stats.add_profile('turbulent_entrainment_QT')
            Stats.add_profile('tke_turb_entr')
            Stats.add_profile('tke_detr_loss')
            Stats.add_profile('tke_shear')
            Stats.add_profile('tke_pressure')
            Stats.add_profile('tke_interdomain')

        if self.calc_scalar_var:
            Stats.add_profile('Hvar_dissipation')
            Stats.add_profile('QTvar_dissipation')
            Stats.add_profile('HQTcov_dissipation')
            Stats.add_profile('Hvar_entr_gain')
            Stats.add_profile('QTvar_entr_gain')
            Stats.add_profile('Hvar_turb_entr')
            Stats.add_profile('QTvar_turb_entr')
            Stats.add_profile('Hvar_detr_loss')
            Stats.add_profile('QTvar_detr_loss')
            Stats.add_profile('HQTcov_detr_loss')
            Stats.add_profile('HQTcov_entr_gain')
            Stats.add_profile('HQTcov_turb_entr')
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
            double [:] mean_turb_entr_W = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_turb_entr_H = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_turb_entr_QT = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_entr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_detr_sc = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] massflux = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mf_h = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mf_qt = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        self.UpdVar.io(Stats)
        self.EnvVar.io(Stats)

        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                mf_h[k] = interp2pt(self.massflux_h[k], self.massflux_h[k-1])
                mf_qt[k] = interp2pt(self.massflux_qt[k], self.massflux_qt[k-1])
                massflux[k] = interp2pt(self.m[0,k], self.m[0,k-1])
                if self.UpdVar.Area.bulkvalues[k] > 0.0:
                    for i in xrange(self.n_updrafts):
                        mean_turb_entr_W[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_W[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_H[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_H[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_QT[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_QT[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_entr_sc[k] += self.UpdVar.Area.values[i,k] * self.entr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_detr_sc[k] += self.UpdVar.Area.values[i,k] * self.detr_sc[i,k]/self.UpdVar.Area.bulkvalues[k]

        Stats.write_profile('turbulent_entrainment_W', mean_turb_entr_W[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('turbulent_entrainment_H', mean_turb_entr_H[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('turbulent_entrainment_QT', mean_turb_entr_QT[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('entrainment_sc', mean_entr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('detrainment_sc', mean_detr_sc[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
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

        if self.calc_tke:
            self.compute_covariance_dissipation(self.EnvVar.TKE)
            Stats.write_profile('tke_dissipation', self.EnvVar.TKE.dissipation[kmin:kmax])
            Stats.write_profile('tke_entr_gain', self.EnvVar.TKE.entr_gain[kmin:kmax])
            Stats.write_profile('tke_turb_entr', self.EnvVar.TKE.entr_gain[kmin:kmax])
            self.compute_covariance_detr(self.EnvVar.TKE)
            Stats.write_profile('tke_detr_loss', self.EnvVar.TKE.detr_loss[kmin:kmax])
            Stats.write_profile('tke_shear', self.EnvVar.TKE.shear[kmin:kmax])
            Stats.write_profile('tke_buoy', self.EnvVar.TKE.buoy[kmin:kmax])
            Stats.write_profile('tke_pressure', self.EnvVar.TKE.press[kmin:kmax])
            Stats.write_profile('tke_interdomain', self.EnvVar.TKE.interdomain[kmin:kmax])

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
            Stats.write_profile('Hvar_turb_entr', self.EnvVar.Hvar.turb_entr[kmin:kmax])
            Stats.write_profile('QTvar_turb_entr', self.EnvVar.QTvar.turb_entr[kmin:kmax])
            Stats.write_profile('HQTcov_turb_entr', self.EnvVar.HQTcov.turb_entr[kmin:kmax])
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

        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        if TS.nstep == 0:
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

        self.UpdVar.set_new_with_values()
        self.UpdVar.set_old_with_values()
        self.set_updraft_surface_bc(GMV, Case)
        self.dt_upd = np.minimum(TS.dt, 0.5 * self.Gr.dz/fmax(np.max(self.UpdVar.W.values),1e-10))
        while time_elapsed < TS.dt:
            self.compute_w_pressure_term(Case)
            self.compute_entrainment_detrainment(GMV, Case, TS)
            self.compute_turbulent_entrainment(GMV,Case)
            self.solve_updraft_velocity_area(GMV,TS)
            self.solve_updraft_scalars(GMV, Case, TS)
            self.UpdVar.set_values_with_new()
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
        self.compute_w_pressure_term(Case)
        #self.compute_entrainment_detrainment(GMV, Case)
        self.compute_turbulent_entrainment(GMV,Case)


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
                self.entr_sc[i,gw] = 2.0 /dz
                self.detr_sc[i,gw] = 0.0
                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    area_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                    if area_k >= self.minimum_area:
                        w_km = self.UpdVar.W.values[i,k-1]
                        entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                        detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                        B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        w2 = ((self.vel_buoy_coeff * B_k + 0.5 * w_km * w_km * dzi)
                              /(0.5 * dzi +entr_w + self.vel_pressure_coeff/sqrt(fmax(area_k,self.minimum_area))))
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
                        #sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                        #         self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
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
            double l1, l2, l3, l4, l5, z_, N
            double l[5]
            double pr_vec[2]
            double ri_grad, shear2, ri_bulk
            double du_high = 0.0
            double dv_high = 0.0
            double dw_high = 2.0 * self.EnvVar.W.values[gw]  * self.Gr.dzi
            double du_low, dv_low, dw_low, H_lapse_rate ,QT_lapse_rate
            double w_u, u_u, v_u, w_e, u_e, v_e, a_
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)
            double qt_dry, th_dry
            double t_cloudy, qv_cloudy, qt_cloudy, th_cloudy
            double lh, cpm, prefactor, d_alpha_thetal_dry, d_alpha_qt_dry
            double d_alpha_thetal_cloudy, d_alpha_qt_cloudy, d_alpha_thetal_total, d_alpha_qt_total
            double grad_thl_minus=0.0, grad_qt_minus=0.0, grad_thl_plus=0, grad_qt_plus=0
            double thv_, thv_e, thv_u, prandtl

        # Grisogono, B. (2010), Generalizing ‘z‐less’ mixing length for stable boundary 
        # layers. Q.J.R. Meteorol. Soc., 136: 213-221. doi:10.1002/qj.529
        if (self.mixing_scheme == 'grisogono'):
            # print 'Using Grisogono mixing length'
            g = 9.81
            for k in xrange(gw, self.Gr.nzg-gw):

                shear2 = pow((GMV.U.values[k+1] - GMV.U.values[k]) * self.Gr.dzi, 2) + \
                pow((GMV.V.values[k+1] - GMV.V.values[k]) * self.Gr.dzi, 2)
                ri_bulk = g * (GMV.THL.values[k] - GMV.THL.values[gw]) * self.Gr.z_half[k]/ \
                GMV.THL.values[gw] / (GMV.U.values[k] * GMV.U.values[k] + GMV.V.values[k] * GMV.V.values[k])

                if (shear2>1e-6) and ri_bulk>0.05:
                	# print(str(self.Gr.z_half[k]))
                    ri_grad = (g/GMV.THL.values[k])*abs(GMV.THL.values[k+1] - GMV.THL.values[k])* self.Gr.dzi
                    ri_grad /=   shear2
                    l1 = sqrt(fmax(self.EnvVar.TKE.values[k],0.0))/sqrt(shear2)
                    # b=10.27 obtained from Mellor, G. L., and T. Yamada (1982), 
                    # Development of a turbulence closure model for geophysical fluid problems, Rev. Geophys., 20(4),
                    # 851–875, doi: 10.1029/RG020i004p00851. C_eps can be used instead.
                    # Prandtl number substituted by function of Ri, otherwise condition for gradient of 
                    # temperature must be added.
                    l2 = l1 * (1.0+ ri_grad/(1.6 + 10.0*ri_grad)) * sqrt(self.tke_diss_coeff/self.tke_ed_coeff)
                    self.mixing_length[k] = fmax(l2, 1e-3)
                else:       
                    l1 = tau * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))
                    z_ = self.Gr.z_half[k]
                    if obukhov_length < 0.0: #unstable
                        l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                    elif obukhov_length > 0.0: #stable
                        l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
                        l1 = 1e9
                    else:
                        l2 = vkb * z_
                    self.mixing_length[k] = fmax( 1.0/(1.0/fmax(l1,1e-10) + 1.0/l2), 1e-3)  

        # Suselj et al. (2013), A Unified Model for Moist Convective Boundary Layers Based 
        # on a Stochastic Eddy-Diffusivity/Mass-Flux Parameterization. J. Atmos. Sci., 70,
        # https://doi.org/10.1175/JAS-D-12-0106.1
        elif self.mixing_scheme=='suselj':
            for k in xrange(gw, self.Gr.nzg-gw):
                g = 9.81
                z_ = self.Gr.z_half[k]
                l[0] = fmax(vkb * z_, 1e-6);
                # tau = 400.0 # Value taken in the paper
                l[1] = tau * sqrt(fmax(self.GMV.TKE.values[k],1.0e-8))
                thv_ = theta_virt_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], GMV.QR.values[k])
                thv_u = theta_virt_c(self.Ref.p0_half[k+1], GMV.T.values[k+1], GMV.QT.values[k+1], GMV.QL.values[k+1], GMV.QR.values[k+1])
                if (thv_u-thv_>0.0):
                    N = sqrt(g/GMV.THL.values[k]*fmax(GMV.THL.values[k+1]-GMV.THL.values[k-1], 1.0e-10)*0.5*self.Gr.dzi)
                    l[2] = fmax(0.7*sqrt(fmax(self.GMV.TKE.values[k],1.0e-8))/N, self.Gr.dz)
                else:
                    l[2] = 1.0e10
                self.mixing_length[k] = 1.0/(1.0/l[1]+1.0/l[2]) + (l[0]-1.0/(1.0/l[1]+1.0/l[2]))*exp(-z_/(0.1*self.zi))

        elif (self.mixing_scheme == 'sbl'):
            #print 'Shear mixing length'
            g = 9.81
            for k in xrange(gw, self.Gr.nzg-gw):
                z_ = self.Gr.z_half[k]
                shear2 = pow((GMV.U.values[k+1] - GMV.U.values[k-1]) * 0.5 * self.Gr.dzi, 2) + \
                    pow((GMV.V.values[k+1] - GMV.V.values[k-1]) * 0.5 * self.Gr.dzi, 2)
                ri_bulk = g * (GMV.THL.values[k] - GMV.THL.values[gw]) * self.Gr.z_half[k]/ \
                GMV.THL.values[gw] / (GMV.U.values[k] * GMV.U.values[k] + GMV.V.values[k] * GMV.V.values[k])
                THL_lapse_rate = fmax(fabs((self.EnvVar.THL.values[k+1]-self.EnvVar.THL.values[k-1])*0.5*self.Gr.dzi),1e-10)
                QT_lapse_rate = fmax(fabs((self.EnvVar.QT.values[k+1]-self.EnvVar.QT.values[k-1])*0.5*self.Gr.dzi),1e-10)
                
                # kz scale (surface layer terms)
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_ # * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                elif obukhov_length > 0.0: #stable
                    l2 = vkb * z_ #/  (1. + 2.7 *z_/obukhov_length)
                else:
                    l2 = vkb * z_

                # Shear-dissipation TKE equilibrium scale (Stable)
                qt_dry = self.EnvThermo.qt_dry[k]
                th_dry = self.EnvThermo.th_dry[k]
                t_cloudy = self.EnvThermo.t_cloudy[k]
                qv_cloudy = self.EnvThermo.qv_cloudy[k]
                qt_cloudy = self.EnvThermo.qt_cloudy[k]
                th_cloudy = self.EnvThermo.th_cloudy[k]

                lh = latent_heat(t_cloudy)
                cpm = cpm_c(qt_cloudy)
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.Gr.dzi

                prefactor = g * ( Rd / self.Ref.alpha0_half[k] /self.Ref.p0_half[k]) * exner_c(self.Ref.p0_half[k])

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

                # Partial Richardson numbers
                ri_thl = grad_thl_plus * d_alpha_thetal_total / fmax(shear2, 1e-10)
                ri_qt  = grad_qt_plus  * d_alpha_qt_total / fmax(shear2, 1e-10)
                
                # Turbulent Prandtl number
                pr_vec[0] = 1.6; pr_vec[1] =  0.6 + 1.0 * (ri_thl+ri_qt)/0.066
                prandtl = smooth_minimum(pr_vec, 7.0)

                l3 = sqrt(self.tke_diss_coeff/self.tke_ed_coeff) * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))/fmax(sqrt(shear2), 1.0e-10)
                l3 /= sqrt(fmax(1.0 - ri_thl/prandtl - ri_qt/prandtl, 1e-7))

                if (sqrt(shear2)< 1.0e-10 or 1.0 - ri_thl/prandtl - ri_qt/prandtl < 1e-7):
                    l3 = 1.0e6
                l3 = fmin(l3, 1.0e7)

                # Limiting stratification scale
                N = fmax(1e-8, sqrt(fmax(g/GMV.THL.values[k]*grad_thl_plus, 0.0)))
                l1 = fmin(sqrt(fmax(0.35*self.EnvVar.TKE.values[k],0.0))/N, 1000.0)
                if (N<1e-7):
                    l1 = 1.0e5
                l2 = fmin(l2, 1000.0)
                l[0]=l2; l[1]=l1; l[2]=l3; l[3]=1.0e5; l[4]=1.0e5
                # self.mixing_length[k] = smooth_minimum2(l, 0.1*self.Gr.dz) #
                j = 0
                while(j<len(l)):
                    if l[j]<1e-4:
                        l[j] = 10000.0
                    j += 1
                self.mls[k] = np.argmin(l)
                # l = sorted(l)

                # For Dycoms and Gabls
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.7*self.Gr.dz))
                # Fixed for Gabls mesh convergence study
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.7*3.125))
                # Fixed for Gabls mesh convergence study
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.7*5.0))
                # For Bomex
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.1*self.Gr.dz))
                # For mesh convergence study Bomex
                self.mixing_length[k] = smooth_minimum(l, 1.0/(0.1*40.0))
                self.ml_ratio[k] = self.mixing_length[k]/l[int(self.mls[k])]

        elif (self.mixing_scheme == 'trials'):
            #print 'Shear mixing length'
            g = 9.81
            for k in xrange(gw, self.Gr.nzg-gw):
                z_ = self.Gr.z_half[k]
                shear2 = pow((GMV.U.values[k+1] - GMV.U.values[k-1]) * 0.5 * self.Gr.dzi, 2) + \
                    pow((GMV.V.values[k+1] - GMV.V.values[k-1]) * 0.5 * self.Gr.dzi, 2)
                ri_bulk = g * (GMV.THL.values[k] - GMV.THL.values[gw]) * self.Gr.z_half[k]/ \
                GMV.THL.values[gw] / (GMV.U.values[k] * GMV.U.values[k] + GMV.V.values[k] * GMV.V.values[k])
                THL_lapse_rate = fmax(fabs((self.EnvVar.THL.values[k+1]-self.EnvVar.THL.values[k-1])*0.5*self.Gr.dzi),1e-10)
                QT_lapse_rate = fmax(fabs((self.EnvVar.QT.values[k+1]-self.EnvVar.QT.values[k-1])*0.5*self.Gr.dzi),1e-10)
                
                # kz scale (surface layer terms)
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_  * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                elif obukhov_length > 0.0: #stable
                    l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
                else:
                    l2 = vkb * z_
                self.mixing_length[k] = fmax( 1.0/(1.0/fmax(l1,1e-10) + 1.0/l2), 1e-3)

                # Shear-dissipation TKE equilibrium scale (Stable)
                qt_dry = self.EnvThermo.qt_dry[k]
                th_dry = self.EnvThermo.th_dry[k]
                t_cloudy = self.EnvThermo.t_cloudy[k]
                qv_cloudy = self.EnvThermo.qv_cloudy[k]
                qt_cloudy = self.EnvThermo.qt_cloudy[k]
                th_cloudy = self.EnvThermo.th_cloudy[k]

                lh = latent_heat(t_cloudy)
                cpm = cpm_c(qt_cloudy)
                grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.Gr.dzi
                grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.Gr.dzi

                prefactor = g * ( Rd / self.Ref.alpha0_half[k] /self.Ref.p0_half[k]) * exner_c(self.Ref.p0_half[k])

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

                ri_thl = grad_thl_plus * d_alpha_thetal_total / fmax(shear2, 1e-10)
                ri_qt  = grad_qt_plus  * d_alpha_qt_total / fmax(shear2, 1e-10)
                
                # if (z_<50.0):
                #     prandtl = 0.8 + 0.8 * z_/50.0
                # else:
                #     prandtl = 1.6

                # if (ri_bulk<.05):
                #     prandtl = 0.8 + 0.8 * ri_bulk/0.05
                # else:
                #     prandtl = 1.6
                pr_vec[0] = 1.6; pr_vec[1] =  0.6 + 1.0 * (ri_thl+ri_qt)/0.066
                prandtl = smooth_minimum(pr_vec, 7.0)

                l3 = sqrt(self.tke_diss_coeff/self.tke_ed_coeff) * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))/fmax(sqrt(shear2), 1.0e-10)
                l3 /= sqrt(fmax(1.0 - ri_thl/prandtl - ri_qt/prandtl, 1e-7))
                if (sqrt(shear2)< 1.0e-10 or 1.0 - ri_thl/prandtl - ri_qt/prandtl < 1e-7):
                    l3 = 1.0e6
                l3 = fmin(l3, 1.0e7)
                # print(z_, (1.0 - ri_thl/1.5 - ri_qt/1.5))
                # Temp variance length scale
                l4 = sqrt(self.tke_diss_coeff/self.tke_ed_coeff) * sqrt(fmax(self.EnvVar.Hvar.values[k],0.0))/fmax(sqrt(grad_thl_plus*grad_thl_plus), 1.0e-10)
                l5 = sqrt(self.tke_diss_coeff/self.tke_ed_coeff) * sqrt(fmax(self.EnvVar.QTvar.values[k],0.0))/fmax(sqrt(grad_qt_plus*grad_qt_plus), 1.0e-10)

                # Limiting stratification scale
                N = fmax( 1e-8, sqrt(fmax(g/GMV.THL.values[k]*grad_thl_plus, 0.0)))
                l1 = fmin(sqrt(fmax(0.35*self.EnvVar.TKE.values[k],0.0))/N, 1000.0)
                if (N<1e-7):
                    l1 = 1.0e5
                l2 = fmin(l2, 1000.0)
                l[0]=l2; l[1]=l1; l[2]=l3; l[3]=1.0e5; l[4]=1.0e5
                # self.mixing_length[k] = smooth_minimum2(l, 0.1*self.Gr.dz) #
                j = 0
                while(j<len(l)):
                    if l[j]<1e-4:
                        l[j] = 10000.0
                    j += 1
                self.mls[k] = np.argmin(l)
                # l = sorted(l)

                # For Dycoms and Gabls
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.7*self.Gr.dz))
                # Fixed for Gabls mesh convergence study
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.7*3.125))
                # Fixed for Gabls mesh convergence study
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.7*5.0))
                # For Bomex
                # self.mixing_length[k] = smooth_minimum(l, 1.0/(0.1*self.Gr.dz))
                # For mesh convergence study Bomex
                self.mixing_length[k] = smooth_minimum(l, 1.0/(0.1*40.0))
                self.ml_ratio[k] = self.mixing_length[k]/l[int(self.mls[k])]

        # Tan et al. (2018)
        elif (self.mixing_scheme == 'tke'):
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    l1 = tau * sqrt(fmax(self.EnvVar.TKE.values[k],0.0))
                    z_ = self.Gr.z_half[k]
                    if obukhov_length < 0.0: #unstable
                        l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
                    elif obukhov_length > 0.0: #stable
                        l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
                        l1 = 1e9
                    else:
                        l2 = vkb * z_
                    self.mixing_length[k] = fmax( 1.0/(1.0/fmax(l1,1e-10) + 1.0/l2), 1e-3)
        return

    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double lm
            double we_half
            double pr_vec[2]
            double prandtl, ri_thl, shear2

        if self.similarity_diffusivity:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV, Case)
        else:
            self.compute_mixing_length(Case.Sur.obukhov_length, GMV)
            with nogil:
                for k in xrange(gw, self.Gr.nzg-gw):
                    lm = self.mixing_length[k]
                    self.KM.values[k] = self.tke_ed_coeff * lm * sqrt(fmax(self.EnvVar.TKE.values[k],0.0) )
                    # Prandtl number is fixed. It should be defined as a function of height - Ignacio
                    self.KH.values[k] = self.KM.values[k] / self.prandtl_number

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
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)
        self.compute_mixing_length(Case.Sur.obukhov_length, GMV)
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    if self.UpdVar.Area.values[i,k] >= self.minimum_area:
                        l_full = interp2pt(self.mixing_length[k], self.mixing_length[k+1])
                        a_full = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                        self.turb_entr_W[i,k] = -2.0*(self.Ref.rho0[k] * self.tke_ed_coeff * l_full * sqrt(interp2pt(GMV.TKE.values[k],GMV.TKE.values[k+1]))
                                                  *(self.UpdVar.W.values[i,k]-self.EnvVar.W.values[k]))/self.pressure_plume_spacing**2.0
                        self.turb_entr_H[i,k] = -2.0*(self.Ref.rho0_half[k]  * self.tke_ed_coeff / self.prandtl_number * self.mixing_length[k] *sqrt(GMV.TKE.values[k])
                                                  *(self.UpdVar.H.values[i,k]-self.EnvVar.H.values[k]))/self.pressure_plume_spacing**2.0
                        self.turb_entr_QT[i,k] = -2.0*(self.Ref.rho0_half[k] *  self.tke_ed_coeff / self.prandtl_number * self.mixing_length[k] *sqrt(GMV.TKE.values[k])
                                                  *(self.UpdVar.QT.values[i,k]-self.EnvVar.QT.values[k]))/self.pressure_plume_spacing**2.0

                    else:
                        self.turb_entr_W[i,k] = 0.0
                        self.turb_entr_H[i,k] = 0.0
                        self.turb_entr_QT[i,k] = 0.0


        return


    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            entr_struct ret
            entr_in_struct input
            eos_struct sa
            double transport_plus, transport_minus, L, a_full_k , a_full_kp, dyn_detrainemnt, w_half_k , adv_aw, adv_aw2,
            long quadrature_order = 3
            double L0 = 100.0
            double [:] b_w2_ = np.zeros((self.Gr.nzg,),dtype=np.double, order='c')


        self.UpdVar.get_cloud_base_top_cover()


        input.wstar = self.wstar
        input.n_updrafts = self.n_updrafts
        input.dz = self.Gr.dz
        input.zbl = self.compute_zbl_qt_grad(GMV)
        input.alpha1e = self.entrainment_alpha1
        input.alpha2e = self.entrainment_alpha2
        input.alpha3e = self.entrainment_alpha3
        input.alpha1d = self.detrainment_alpha1
        input.alpha2d = self.detrainment_alpha2
        input.alpha3d = self.detrainment_alpha3
        # experimenet
        # TRMM_LBA
        #edmf_closures = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.TRMM_LBA.newtracers_ICE3/stats/edmf_closures.nc','r')
        #stats = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.TRMM_LBA.newtracers_ICE3/stats/Stats.TRMM_LBA.nc','r')

        # Bomex
        #edmf_closures = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/edmf_closures.nc','r')
        #stats = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Bomex.newtracers/stats/Stats.Bomex.nc','r')
        # ARM SGP
        #edmf_closures = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.ARM_SGP.newtracers/stats/edmf_closures.nc','r')
        #stats = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.ARM_SGP.newtracers/stats/Stats.ARM_SGP.nc','r')
        # Rico
        #edmf_closures = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Rico.BSr20/stats/edmf_closures.nc','r')
        #stats = nc.Dataset('/Users/yaircohen/Documents/PyCLES_out/newTracers/Output.Rico.BSr20/stats/Stats.Rico.nc','r')


        # z_les = np.multiply(stats.groups['profiles'].variables['z'], 1.0)
        # t_les = np.multiply(stats.groups['profiles'].variables['t'], 1.0)
        #
        # prog_entr_th = np.multiply(edmf_closures.groups['profiles'].variables['prog_entrainment_th'], 1.0)
        # prog_detr_th = np.multiply(edmf_closures.groups['profiles'].variables['prog_detrainment_th'], 1.0)
        # prog_entr_th[np.where(prog_entr_th>0.01)] = 0.01
        # prog_entr_th[np.where(prog_entr_th<=0.0)] = 0.01
        # prog_detr_th[np.where(prog_detr_th>0.01)] = 0.0
        # prog_detr_th[np.where(prog_detr_th<=0.0)] = 0.0
        #
        # buoyancy_mean = np.multiply(stats.groups['profiles'].variables['buoyancy_mean'], 1.0)
        # upd_buoyancy = np.subtract(np.multiply(stats.groups['profiles'].variables['updraft_b'], 1.0),buoyancy_mean)
        # partition_func = np.abs(np.ceil(np.sign(upd_buoyancy) / 2))
        # partition_func_1 = np.subtract(1.0,partition_func)
        #
        # prog_entr_th = np.multiply(prog_entr_th,partition_func)
        # prog_detr_th = np.multiply(prog_detr_th,partition_func_1)
        #
        #
        #
        # #prog_entr_th[np.where(prog_entr_th == 0.0)] = np.nan
        # #prog_detr_th[np.where(prog_detr_th == 0.0)] = np.nan
        #
        # #time1 = np.ceil(TS.t/3600.0)*3600.0
        # #time0 = np.floor(TS.t/3600.0)*3600.0
        #
        #
        # #epsilon = np.squeeze(np.nanmean(prog_entr_th[np.where((TS.t>=time0)& (TS.t<time1))],0))
        # #delta = np.squeeze(np.nanmean(prog_detr_th[np.where((TS.t>=time0)& (TS.t<time1))],0))
        #
        # epsilon = np.zeros((np.shape(prog_entr_th)[0],np.shape(self.Gr.z_half)[0]))
        # delta   = np.zeros((np.shape(prog_detr_th)[0],np.shape(self.Gr.z_half)[0]))
        #
        # for i in range(np.shape(prog_entr_th)[0]):
        #     epsilon[i,:] = np.interp(self.Gr.z_half,z_les,prog_entr_th[i,:])
        #     delta[i,:] = np.interp(self.Gr.z_half,z_les,prog_detr_th[i,:])


        for i in xrange(self.n_updrafts):
            input.zi = self.UpdVar.cloud_base[i]
            input.upd_number = i
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                input.quadrature_order = quadrature_order
                input.b = self.UpdVar.B.values[i,k]
                input.dbdz = (self.UpdVar.B.values[i,k+1]-self.UpdVar.B.values[i,k-1])\
                             /(self.Gr.z_half[k+1]-self.Gr.z_half[k-1])
                input.w = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                input.z = self.Gr.z_half[k]
                input.af = self.UpdVar.Area.values[i,k]
                input.tke = GMV.TKE.values[k]
                input.ml = self.mixing_length[k]
                input.qt_env = self.EnvVar.QT.values[k]
                input.ql_env = self.EnvVar.QL.values[k]
                input.H_env = self.EnvVar.H.values[k]
                input.b_env = self.EnvVar.B.values[k]
                input.w_env = self.EnvVar.W.values[k]
                input.H_up = self.UpdVar.H.values[i,k]
                input.qt_up = self.UpdVar.QT.values[i,k]
                input.ql_up = self.UpdVar.QL.values[i,k]
                input.p0 = self.Ref.p0_half[k]
                input.alpha0 = self.Ref.alpha0_half[k]
                input.env_Hvar = self.EnvVar.Hvar.values[k]
                input.env_QTvar = self.EnvVar.QTvar.values[k]
                input.env_HQTcov = self.EnvVar.HQTcov.values[k]
                # f or Poisson process closure
                b_w2_[k] = fmax(input.b,0.0)/fmax(input.w**2,1e-2)
                input.press = interp2pt(self.w_press_term[i,k],self.w_press_term[i,k-1])
                #print 'k',k, self.Gr.nzg, self.Gr.nz, np.shape(self.Gr.z_half)
                #print 'np.shape(b_w2_[self.Gr.gw:k])',np.shape(b_w2_[self.Gr.gw:k])
                #print 'np.shape(self.Gr.z_half[self.Gr.gw:k])', np.shape(self.Gr.z_half[self.Gr.gw:k])
                L = np.trapz(b_w2_[self.Gr.gw:k],self.Gr.z_half[self.Gr.gw:k])
                input.entr_poisson = np.random.poisson(L)/self.Gr.z_half[k]
                #input.entr_poisson = np.random.poisson(self.Gr.z_half[k]/fmax(L,0.00001))
                #print 'poisson L', k, input.entr_poisson, L
                input.beta = np.random.beta(2,2)
                input.rd = self.pressure_plume_spacing


                if self.calc_tke:
                        input.tke = self.EnvVar.TKE.values[k]
                        input.tke_ed_coeff  = self.tke_ed_coeff

                input.T_mean = (self.EnvVar.T.values[k]+self.UpdVar.T.values[i,k])/2
                #input.L = 20000.0 # need to define the scale of the GCM grid resolution
                ## Ignacio
                input.n_up = self.n_updrafts
                input.thv_e = theta_virt_c(self.Ref.p0_half[k], self.EnvVar.T.values[k], self.EnvVar.QT.values[k],
                     self.EnvVar.QL.values[k], self.EnvVar.QR.values[k])
                input.thv_u = theta_virt_c(self.Ref.p0_half[k], self.UpdVar.T.bulkvalues[k], self.UpdVar.QT.bulkvalues[k],
                     self.UpdVar.QL.bulkvalues[k], self.UpdVar.QR.bulkvalues[k])
                input.dwdz = (self.UpdVar.Area.values[i,k+1]*
                    interp2pt(self.UpdVar.W.values[i,k+1],self.UpdVar.W.values[i,k]) + 
                    (1.0-self.UpdVar.Area.values[i,k+1])*self.EnvVar.W.values[k+1] -
                    (self.UpdVar.Area.values[i,k-1]*
                    interp2pt(self.UpdVar.W.values[i,k-1],self.UpdVar.W.values[i,k-2]) + 
                    (1.0-self.UpdVar.Area.values[i,k-1])*self.EnvVar.W.values[k-1]) )/(2.0*self.Gr.dz)

                transport_plus = ( self.UpdVar.Area.values[i,k+1]*(1.0-self.UpdVar.Area.values[i,k+1])*
                    (interp2pt(self.UpdVar.W.values[i,k+1],self.UpdVar.W.values[i,k]) - self.EnvVar.W.values[k+1])*
                    (1.0-2.0*self.UpdVar.Area.values[i,k+1])*
                    (interp2pt(self.UpdVar.W.values[i,k+1],self.UpdVar.W.values[i,k]) - self.EnvVar.W.values[k+1])*
                    (interp2pt(self.UpdVar.W.values[i,k+1],self.UpdVar.W.values[i,k]) - self.EnvVar.W.values[k+1]) )

                transport_minus = ( self.UpdVar.Area.values[i,k-1]*(1.0-self.UpdVar.Area.values[i,k-1])*
                    (interp2pt(self.UpdVar.W.values[i,k-1],self.UpdVar.W.values[i,k-2]) - self.EnvVar.W.values[k-1])*
                    (1.0-2.0*self.UpdVar.Area.values[i,k+1])*
                    (interp2pt(self.UpdVar.W.values[i,k-1],self.UpdVar.W.values[i,k-2]) - self.EnvVar.W.values[k-1])*
                    (interp2pt(self.UpdVar.W.values[i,k-1],self.UpdVar.W.values[i,k-2]) - self.EnvVar.W.values[k-1]) )

                input.transport_der = (transport_plus - transport_minus)/2.0/self.Gr.dz

                if input.zbl-self.UpdVar.cloud_base[i] > 0.0:
                    input.poisson = np.random.poisson(self.Gr.dz/((input.zbl-self.UpdVar.cloud_base[i])/10.0))
                else:
                    input.poisson = 0.0
                ## End: Ignacio

                ret = self.entr_detr_fp(input)

                # dynamic detrainment
                if input.b>0.0:
                    a_full_k = interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k-1])
                    a_full_kp = interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1])
                    w_half_k = interp2pt(self.UpdVar.W.values[i,k],self.UpdVar.W.values[i,k-1])
                    adv_aw = (a_full_kp*self.UpdVar.W.values[i,k+1]-a_full_k*self.UpdVar.W.values[i,k])/fmax(w_half_k,0.1)/fmax(self.UpdVar.Area.values[i,k],self.minimum_area)
                    adv_aw2 = (a_full_kp*self.UpdVar.W.values[i,k+1]**2-a_full_k*self.UpdVar.W.values[i,k]**2)/fmax(w_half_k**2,0.01)
                    #logf = logistic(input.b, 0.1, 0.0)
                    logf = 1.0
                    dyn_detrainemnt = logf*(adv_aw2 - adv_aw - (input.b+input.press)/fmax(w_half_k**2,0.01))
                else:
                    dyn_detrainemnt = 0.0

                self.entr_sc[i,k] = ret.entr_sc * self.entrainment_factor/sqrt(self.n_updrafts)
                #self.detr_sc[i,k] = (ret.detr_sc+dyn_detrainemnt) * self.detrainment_factor
                self.detr_sc[i,k] = ret.detr_sc * self.detrainment_factor/sqrt(self.n_updrafts)

                ### experiment with loading entr profiles

                #self.entr_sc[i,k] = np.interp(TS.t,t_les,epsilon[:,k])
                #self.detr_sc[i,k] = np.interp(TS.t,t_les,delta[:,k])

                #self.entr_sc[i,k] /= 2.0
                #self.detr_sc[i,k] /= 2.0


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


    cpdef compute_w_pressure_term(self, CasesBase Case):
        cdef:
            double a_k, B_k, press_buoy, press_drag

        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
                press_drag = -1.0 * self.Ref.rho0[k] * a_k * (self.pressure_drag_coeff/self.pressure_plume_spacing
                                                                     * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])**2.0/sqrt(fmax(a_k,self.minimum_area)))
                self.w_press_term[i,k] = press_buoy + press_drag


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
            double adv, buoy, exch, press, press_buoy, press_drag # groupings of terms in velocity discrete equation

        with nogil:
            for i in xrange(self.n_updrafts):
                self.entr_sc[i,gw] = 2.0 * dzi
                self.detr_sc[i,gw] = 0.0
                self.UpdVar.W.new[i,gw-1] = self.w_surface_bc[i]
                self.UpdVar.Area.new[i,gw] = self.area_surface_bc[i]
                au_lim = self.area_surface_bc[i] * self.max_area_factor

                # adv = self.Ref.rho0[gw] * self.UpdVar.Area.values[i,gw] * self.UpdVar.W.values[i,gw] * self.UpdVar.W.values[i,gw] * dzi/2.0
                # exch = (self.Ref.rho0[gw] * self.UpdVar.Area.values[i,gw] * self.UpdVar.W.values[i,gw]* (self.entr_sc[i,gw] * self.EnvVar.W.values[gw] ))
                # buoy= self.Ref.rho0[gw] * self.UpdVar.Area.values[i,gw] * self.UpdVar.B.values[i,gw]
                # press_buoy =  -1.0 * self.Ref.rho0[gw] * self.UpdVar.Area.values[i,gw] * B_k * self.pressure_buoy_coeff
                # press_drag = -1.0 * self.Ref.rho0[gw] * sqrt(self.UpdVar.Area.values[i,gw]) * (self.pressure_drag_coeff/self.pressure_plume_spacing
                #                                              * (self.UpdVar.W.values[i,gw] -self.EnvVar.W.values[gw])**2.0)
                # self.w_press_term[i,gw] = press_buoy + press_drag
                # self.UpdVar.W.values[i,gw] = (self.Ref.rho0[gw] * self.UpdVar.Area.values[i,gw] * self.UpdVar.W.values[i,gw] * dti_
                #                           -adv + exch + buoy + self.w_press_term[i,gw])/(self.Ref.rho0[gw] * self.UpdVar.Area.values[i,gw] * dti_)
                # with gil:
                #     print self.UpdVar.W.values[i,gw]

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
                    # if TS.t>3600.0*1.0:
                    #     if self.UpdVar.Area.new[i,k+1]==0.0:
                    #         with gil:
                    #             print TS.t, k, self.UpdVar.Area.new[i,k+1], adv, entr_term , detr_term, self.UpdVar.Area.values[i,k+1] , whalf_k, whalf_kp
                    #             plt.figure()
                    #             plt.show()
                    if anew_k >= self.minimum_area:
                        a_k = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                        a_km = interp2pt(self.UpdVar.Area.values[i,k-1], self.UpdVar.Area.values[i,k])
                        entr_w = interp2pt(self.entr_sc[i,k], self.entr_sc[i,k+1])
                        detr_w = interp2pt(self.detr_sc[i,k], self.detr_sc[i,k+1])
                        B_k = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                        adv = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * self.UpdVar.W.values[i,k] * dzi
                               - self.Ref.rho0[k-1] * a_km * self.UpdVar.W.values[i,k-1] * self.UpdVar.W.values[i,k-1] * dzi)
                        exch = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k]
                                * (entr_w * self.EnvVar.W.values[k] - detr_w * self.UpdVar.W.values[i,k] )) + self.turb_entr_W[i,k]
                        buoy= self.Ref.rho0[k] * a_k * B_k
                        press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * fabs(B_k) * self.pressure_buoy_coeff
                        press_drag = -1.0 * self.Ref.rho0[k] * sqrt(a_k) * (self.pressure_drag_coeff/self.pressure_plume_spacing
                                                                     * (self.UpdVar.W.values[i,k] -self.EnvVar.W.values[k])**2.0)
                        self.w_press_term[i,k] = press_buoy + press_drag
                        self.updraft_pressure_sink[i,k] = self.w_press_term[i,k]
                        self.UpdVar.W.new[i,k] = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i,k] * dti_
                                                  -adv + exch + buoy + self.w_press_term[i,k])/(self.Ref.rho0[k] * anew_k * dti_)
                        if self.UpdVar.W.new[i,k] <= 0.0:
                            self.UpdVar.W.new[i,k:] = 0.0
                            self.UpdVar.Area.new[i,k+1:] = 0.0
                            break
                    else:
                        self.UpdVar.W.new[i,k:] = 0.0
                        self.UpdVar.Area.new[i,k+1:] = 0.0
                        # keep this in mind if we modify updraft top treatment!
                        self.updraft_pressure_sink[i,k:] = 0.0
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
            double c1, c2, c3, c4, l
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
                    l = self.mixing_length[k]
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
                                                   + c4 * H_entr )/c1 + self.turb_entr_H[i,k]
                        self.UpdVar.QT.new[i,k] = (c2 * self.UpdVar.QT.values[i,k] + c3 * self.UpdVar.QT.values[i,k-1]
                                                   + c4* QT_entr )/c1  + self.turb_entr_QT[i,k]

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

        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.QR.set_bcs(self.Gr)
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
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        # Compute the mass flux and associated scalar fluxes
        with nogil:
            for i in xrange(self.n_updrafts):
                self.m[i,gw-1] = 0.0
                for k in xrange(self.Gr.gw, self.Gr.nzg-1):
                    self.m[i,k] = ((self.UpdVar.W.values[i,k] - self.EnvVar.W.values[k] )* self.Ref.rho0[k]
                                   * interp2pt(self.UpdVar.Area.values[i,k],self.UpdVar.Area.values[i,k+1]))

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


        GMV.H.set_bcs(self.Gr)
        GMV.QT.set_bcs(self.Gr)
        GMV.QR.set_bcs(self.Gr)
        GMV.U.set_bcs(self.Gr)
        GMV.V.set_bcs(self.Gr)

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
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')

        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
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
            # get the diffusive flux
            self.diffusive_tendency_qt[k+gw] = (GMV.QT.new[k+gw] - GMV.QT.mf_update[k+gw]) * TS.dti
            self.diffusive_flux_qt[gw] = interp2pt(Case.Sur.rho_qtflux, -rho_ae_K_m[gw] * dzi *(self.EnvVar.QT.values[gw+1]-self.EnvVar.QT.values[gw]) )
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
            self.diffusive_flux_h[gw] = interp2pt(Case.Sur.rho_hflux, -rho_ae_K_m[gw] * dzi *(self.EnvVar.H.values[gw+1]-self.EnvVar.H.values[gw]) )
            for k in xrange(self.Gr.gw+1, self.Gr.nzg-self.Gr.gw):
                self.diffusive_flux_h[k] = -0.5 * self.Ref.rho0_half[k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.H.values[k+1]-self.EnvVar.H.values[k-1])

         # Solve U
        with nogil:
            for k in xrange(nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KM.values[k]+ ae[k+1]*self.KM.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, &rho_ae_K_m[0], &self.Ref.rho0_half[0],
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
            double wu_half, we_half
            double press_buoy, press_drag

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                self.EnvVar.TKE.press[k] = 0.0
                for i in xrange(self.n_updrafts):
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    we_half = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                    press_buoy= (-1.0 * self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k]
                                 * self.UpdVar.B.values[i,k] * self.pressure_buoy_coeff)
                    press_drag = (-1.0 * self.Ref.rho0_half[k] * sqrt(self.UpdVar.Area.values[i,k])
                                  * (self.pressure_drag_coeff/self.pressure_plume_spacing* (wu_half -we_half)*fabs(wu_half -we_half)))
                    self.EnvVar.TKE.press[k] += (we_half - wu_half) * (press_buoy + press_drag)
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
        print 'compute_covariance'
        #if TS.nstep > 0:
        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed
            self.compute_mixing_length(Case.Sur.obukhov_length, GMV)
        if self.calc_tke:
            self.compute_tke_buoy(GMV)
            self.compute_upd_tke_buoy(GMV)
            self.compute_covariance_entr(self.EnvVar.TKE, self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W)
            self.compute_upd_covariance_entr(self.UpdVar.TKE, self.EnvVar.TKE, self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W)
            self.compute_covariance_turb_entr(GMV, self.EnvVar.TKE)
            self.compute_covariance_shear(GMV, self.EnvVar.TKE, &self.UpdVar.W.values[0,0], &self.UpdVar.W.values[0,0], &self.EnvVar.W.values[0], &self.EnvVar.W.values[0])
            print 'before'
            self.compute_upd_covariance_shear(GMV, self.UpdVar.TKE, &self.UpdVar.W.values[0,0], &self.UpdVar.W.values[0,0], &self.EnvVar.W.values[0], &self.EnvVar.W.values[0])
            print 'after'
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.W,self.UpdVar.W,self.EnvVar.W, self.EnvVar.W, self.EnvVar.TKE)
            self.compute_tke_pressure()
        if self.calc_scalar_var:
            self.compute_covariance_entr(self.EnvVar.Hvar, self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H)
            self.compute_covariance_entr(self.EnvVar.QTvar, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT)
            self.compute_covariance_entr(self.EnvVar.HQTcov, self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT)
            self.compute_upd_covariance_entr(self.UpdVar.Hvar,self.EnvVar.Hvar, self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H)
            self.compute_upd_covariance_entr(self.UpdVar.QTvar,self.EnvVar.QTvar, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT)
            self.compute_upd_covariance_entr(self.UpdVar.HQTcov, self.EnvVar.HQTcov, self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT)
            self.compute_covariance_turb_entr(GMV, self.EnvVar.Hvar)
            self.compute_covariance_turb_entr(GMV, self.EnvVar.QTvar)
            self.compute_covariance_turb_entr(GMV, self.EnvVar.HQTcov)
            self.compute_covariance_shear(GMV, self.EnvVar.Hvar, &self.UpdVar.H.values[0,0], &self.UpdVar.H.values[0,0], &self.EnvVar.H.values[0], &self.EnvVar.H.values[0])
            self.compute_covariance_shear(GMV, self.EnvVar.QTvar, &self.UpdVar.QT.values[0,0], &self.UpdVar.QT.values[0,0], &self.EnvVar.QT.values[0], &self.EnvVar.QT.values[0])
            self.compute_covariance_shear(GMV, self.EnvVar.HQTcov, &self.UpdVar.H.values[0,0], &self.UpdVar.QT.values[0,0], &self.EnvVar.H.values[0], &self.EnvVar.QT.values[0])
            self.compute_upd_covariance_shear(GMV, self.UpdVar.Hvar, &self.UpdVar.H.values[0,0], &self.UpdVar.H.values[0,0], &self.EnvVar.H.values[0], &self.EnvVar.H.values[0])
            self.compute_upd_covariance_shear(GMV, self.UpdVar.QTvar, &self.UpdVar.QT.values[0,0], &self.UpdVar.QT.values[0,0], &self.EnvVar.QT.values[0], &self.EnvVar.QT.values[0])
            self.compute_upd_covariance_shear(GMV, self.UpdVar.HQTcov, &self.UpdVar.H.values[0,0], &self.UpdVar.QT.values[0,0], &self.EnvVar.H.values[0], &self.EnvVar.QT.values[0])
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.H,self.UpdVar.H,self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.QT,self.UpdVar.QT,self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.H,self.UpdVar.QT,self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov)
            self.compute_covariance_rain(TS, GMV) # need to update this one

        self.reset_surface_covariance(GMV, Case)
        if self.calc_tke:
            self.update_covariance_ED(GMV, Case,TS, GMV.W, GMV.W, GMV.TKE, self.EnvVar.TKE, self.EnvVar.W, self.EnvVar.W, self.UpdVar.W, self.UpdVar.W)
            self.update_upd_covariance_ED(GMV, Case,TS, GMV.W, GMV.W, GMV.TKE, self.UpdVar.TKE, self.EnvVar.TKE, self.EnvVar.W, self.EnvVar.W, self.UpdVar.W, self.UpdVar.W)
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
        if self.calc_scalar_var:
            if ws > 0.0:
                with nogil:
                    for k in xrange(self.Gr.nzg):
                        z = self.Gr.z_half[k]
                        # need to rethink of how to initilize the covarinace profiles - for nowmI took the TKE profile
                        GMV.Hvar.values[k]   = GMV.Hvar.values[self.Gr.gw] * ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
                        GMV.QTvar.values[k]  = GMV.QTvar.values[self.Gr.gw] * ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
                        GMV.HQTcov.values[k] = GMV.HQTcov.values[self.Gr.gw] * ws * 1.3 * cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * sqrt(fmax(1.0-z/zs,0.0))
            self.reset_surface_covariance(GMV, Case)
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
                if self.UpdVar.Hvar.values[0,k] < tmp_eps:
                    self.UpdVar.Hvar.values[0,k] = 0.0
                if self.UpdVar.TKE.values[0,k] < tmp_eps:
                    self.UpdVar.TKE.values[0,k] = 0.0
                if self.UpdVar.QTvar.values[0,k] < tmp_eps:
                    self.UpdVar.QTvar.values[0,k] = 0.0
                if fabs(self.UpdVar.HQTcov.values[0,k]) < tmp_eps:
                    self.UpdVar.HQTcov.values[0,k] = 0.0
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

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            if Covar.name == 'tke':
                du_low = du_high
                dv_low = dv_high
                du_high = (GMV.U.values[k+1] - GMV.U.values[k]) * self.Gr.dzi
                dv_high = (GMV.V.values[k+1] - GMV.V.values[k]) * self.Gr.dzi
                diff_var2 = (EnvVar2[k] - EnvVar2[k-1]) * self.Gr.dzi
                diff_var1 = (EnvVar1[k] - EnvVar1[k-1]) * self.Gr.dzi
                tke_factor = 0.5
            else:
                du_low = 0.0
                dv_low = 0.0
                du_high = 0.0
                dv_high = 0.0
                diff_var2 = interp2pt((EnvVar2[k+1] - EnvVar2[k]),(EnvVar2[k] - EnvVar2[k-1])) * self.Gr.dzi
                diff_var1 = interp2pt((EnvVar1[k+1] - EnvVar1[k]),(EnvVar1[k] - EnvVar1[k-1])) * self.Gr.dzi
                tke_factor = 1.0
            with nogil:
                Covar.shear[k] = tke_factor*2.0*(self.Ref.rho0_half[k] * ae[k] * self.KH.values[k] *
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
            double updvar1, updvar2, envvar1, envvar2

        #with nogil:
        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            Covar.entr_gain[k] = 0.0
            for i in xrange(self.n_updrafts):
                if Covar.name =='tke':
                    updvar1 = interp2pt(UpdVar1.values[i,k], UpdVar1.values[i,k-1])
                    updvar2 = interp2pt(UpdVar2.values[i,k], UpdVar2.values[i,k-1])
                    envvar1 = interp2pt(EnvVar1.values[k], EnvVar1.values[k-1])
                    envvar2 = interp2pt(EnvVar2.values[k], EnvVar2.values[k-1])
                    tke_factor = 0.5
                else:
                    updvar1 = UpdVar1.values[i,k]
                    updvar2 = UpdVar2.values[i,k]
                    envvar1 = EnvVar1.values[k]
                    envvar2 = EnvVar2.values[k]
                    tke_factor = 1.0
                w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                Covar.entr_gain[k] +=  tke_factor*self.UpdVar.Area.values[i,k] * fabs(w_u) * self.detr_sc[i,k] * \
                                             (updvar1 - envvar1) * (updvar2 - envvar2)
            Covar.entr_gain[k] *= self.Ref.rho0_half[k]
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


    cdef void compute_covariance_turb_entr(self, GridMeanVariables GMV, EDMF_Environment.EnvironmentVariable_2m Covar):
        cdef:
            Py_ssize_t i, k

        #with nogil:
        if Covar.name == 'tke':
            ck = self.tke_ed_coeff
        else:
            ck = self.tke_ed_coeff / self.prandtl_number

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            Covar.turb_entr[k] = 0.0
            for i in xrange(self.n_updrafts):
                if self.UpdVar.Area.values[i,k] >= self.minimum_area:
                    Covar.turb_entr[k] += -(self.Ref.rho0_half[k] * ck * self.mixing_length[k] * sqrt(GMV.TKE.values[k])
                                                  /self.pressure_plume_spacing**2.0*Covar.values[k])
                else:
                    Covar.turb_entr[k] += 0.0
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
                                    *pow(fmax(self.EnvVar.TKE.values[k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
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
            double Covar_surf, wu_half

        with nogil:
            for k in xrange(1,nzg-1):
                rho_ae_K_m[k] = 0.5 * (ae[k]*self.KH.values[k]+ ae[k+1]*self.KH.values[k+1])* self.Ref.rho0[k]
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
                    wu_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    D_env += self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * wu_half * self.entr_sc[i,k]


                a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                b[kk] = (self.Ref.rho0_half[k] * ae[k] * dti - self.Ref.rho0_half[k] * ae[k] * whalf[k] * dzi
                         + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                         + D_env
                         + self.Ref.rho0_half[k] * ae[k] * self.tke_diss_coeff
                                    *sqrt(fmax(self.EnvVar.TKE.values[k],0))/fmax(self.mixing_length[k],1.0) )
                c[kk] = (self.Ref.rho0_half[k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                x[kk] = (self.Ref.rho0_half[k] * ae_old[k] * Covar.values[k] * dti
                         + Covar.press[k] + Covar.buoy[k] + Covar.shear[k] + Covar.entr_gain[k] + Covar.turb_entr[k]  +  Covar.rain_src[k]) #

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

        self.get_GMV_CoVar(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, &GmvVar1.values[0], &GmvVar2.values[0], &GmvCovar.values[0])

        return


## ---------- updraft coavriances



    cdef void compute_upd_covariance_shear(self,GridMeanVariables GMV, EDMF_Updrafts.UpdraftVariable_2m Covar, double *UpdVar1, double *UpdVar2, double *EnvVar1, double *EnvVar2):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double diff_var1 = 0.0
            double diff_var2 = 0.0
            double du = 0.0
            double dv = 0.0
            double tke_factor = 1.0
            double du_low, dv_low
            double du_high = 0.0
            double dv_high = 0.0
        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.UpdVar.Area.values[i,k] >self.minimum_area:
                    if Covar.name == 'tke':
                        du_low = du_high
                        dv_low = dv_high
                        du_high = (GMV.U.values[k+1] - GMV.U.values[k]) * self.Gr.dzi
                        dv_high = (GMV.V.values[k+1] - GMV.V.values[k]) * self.Gr.dzi
                        diff_var2 = (UpdVar2[k] - UpdVar2[k-1]) * self.Gr.dzi
                        diff_var1 = (UpdVar1[k] - UpdVar1[k-1]) * self.Gr.dzi
                        tke_factor = 0.5
                    else:
                        du_low = 0.0
                        dv_low = 0.0
                        du_high = 0.0
                        dv_high = 0.0
                        diff_var2 = interp2pt((UpdVar2[k+1] - UpdVar2[k]),(UpdVar2[k] - UpdVar2[k-1])) * self.Gr.dzi
                        diff_var1 = interp2pt((UpdVar1[k+1] - UpdVar1[k]),(UpdVar1[k] - UpdVar1[k-1])) * self.Gr.dzi
                        tke_factor = 1.0
                    with nogil:
                        Covar.shear[i,k] = tke_factor*2.0*(self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * self.KH.values[k] *
                                    (diff_var1*diff_var2 +  pow(interp2pt(du_low, du_high),2.0)  +  pow(interp2pt(dv_low, dv_high),2.0)))
                else:
                    Covar.shear[i,k] = 0.0
        return

    cdef void compute_upd_covariance_entr(self, EDMF_Updrafts.UpdraftVariable_2m UpdCovar,EDMF_Environment.EnvironmentVariable_2m EnvCovar, EDMF_Updrafts.UpdraftVariable UpdVar1,
                EDMF_Updrafts.UpdraftVariable UpdVar2, EDMF_Environment.EnvironmentVariable EnvVar1, EDMF_Environment.EnvironmentVariable EnvVar2):
        cdef:
            Py_ssize_t i, k
            double tke_factor
            double updvar1, updvar2, envvar1, envvar2, diff_env1, diff_env2

        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.UpdVar.Area.values[i,k] >self.minimum_area:
                    w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                    if UpdCovar.name =='tke':
                        updvar1 = interp2pt(UpdVar1.values[i,k], UpdVar1.values[i,k-1])
                        updvar2 = interp2pt(UpdVar2.values[i,k], UpdVar2.values[i,k-1])
                        envvar1 = interp2pt(EnvVar1.values[k], EnvVar1.values[k-1])
                        envvar2 = interp2pt(EnvVar2.values[k], EnvVar2.values[k-1])
                        tke_factor = 0.5
                    else:
                        updvar1 = UpdVar1.values[i,k]
                        updvar2 = UpdVar2.values[i,k]
                        envvar1 = EnvVar1.values[k]
                        envvar2 = EnvVar2.values[k]
                        tke_factor = 1.0
                    # calculate the env mf contribution
                    diff_env1 = (updvar1 - envvar1)
                    diff_env2 = (updvar2 - envvar2)
                    UpdCovar.entr_gain[i,k] = self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]* fabs(w_u)*self.entr_sc[i,k]*\
                                               (EnvCovar.values[k]+tke_factor*diff_env1*diff_env2)
                else:
                    UpdCovar.entr_gain[i,k] = 0.0
        return

    # DOTO  this function entrains from all updrafts  - it is not complete

    # cdef void compute_upd_covariance_entr(self, EDMF_Updrafts.UpdraftVariable_2m UpdCovar,EDMF_Updrafts.UpdraftVariable_2m EnvCovar, EDMF_Updrafts.UpdraftVariable UpdVar1,
    #             EDMF_Updrafts.UpdraftVariable UpdVar2, EDMF_Environment.EnvironmentVariable EnvVar1, EDMF_Environment.EnvironmentVariable EnvVar2):
    #     cdef:
    #         Py_ssize_t i, k
    #         double tke_factor
    #         double updvar1, updvar2, envvar1, envvar2, diff_env1, diff_env2
    #
    #     for i in xrange(self.n_updrafts):
    #         for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
    #             w_u = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
    #             if UpdCovar.name =='tke':
    #                 updvar1 = interp2pt(UpdVar1.values[i,k], UpdVar1.values[i,k-1])
    #                 updvar2 = interp2pt(UpdVar2.values[i,k], UpdVar2.values[i,k-1])
    #                 envvar1 = interp2pt(EnvVar1.values[k], EnvVar1.values[k-1])
    #                 envvar2 = interp2pt(EnvVar2.values[k], EnvVar2.values[k-1])
    #             else:
    #                 updvar1 = UpdVar1.values[i,k]
    #                 updvar2 = UpdVar2.values[i,k]
    #                 envvar1 = EnvVar1.values[k]
    #                 envvar2 = EnvVar2.values[k]
    #             # calculate the env mf contribution
    #             diff_env1 = (updvar1 - envvar1)
    #             diff_env2 = (updvar2 - envvar2)
    #             # calculate the mf contribution from all other updrafts
    #             UpdCovar.entr_gain[i,k] = 0.0
    #             for j in xrange(self.n_updrafts):
    #                 if UpdCovar.name =='tke':
    #                     upd1var1 = interp2pt(UpdVar1.values[i,k], UpdVar1.values[i,k-1])
    #                     upd1var2 = interp2pt(UpdVar2.values[i,k], UpdVar2.values[i,k-1])
    #                     upd2var1 = interp2pt(UpdVar1.values[j,k], UpdVar1.values[j,k-1])
    #                     upd2var2 = interp2pt(UpdVar2.values[j,k], UpdVar2.values[j,k-1])
    #
    #                     tke_factor = 0.5
    #                 else:
    #                     upd1var1 = UpdVar1.values[i,k]
    #                     upd1var2 = UpdVar2.values[i,k]
    #                     upd2var1 = UpdVar1.values[j,k]
    #                     upd2var2 = UpdVar2.values[j,k]
    #                     tke_factor = 1.0
    #                 diff_upd1 = (upd1var1 - upd2var1)
    #                 diff_upd2 = (upd1var2 - upd2var2)
    #                 UpdCovar.entr_gain[i,k] += tke_factor*diff_upd1*diff_upd2*self.entr_sc[i,k]
    #                 if j != i:
    #                     UpdCovar.entr_gain[i,k] += UpdCovar.values[j,k]*self.entr_sc[i,k]
    #
    #             UpdCovar.entr_gain[i,k] += EnvCovar.values[j,k]*self.entr_sc[i,k]
    #             UpdCovar.entr_gain[i,k] *= self.Ref.rho0_half[k]*self.UpdVar.Area.values[i,k]* fabs(w_u)
    #     return

    cdef void compute_upd_covariance_detr(self, EDMF_Updrafts.UpdraftVariable_2m UpdCover):
        cdef:
            Py_ssize_t i, k
            double w_u

        #with nogil:
        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.UpdVar.Area.values[i,k] >self.minimum_area:
                    w_u = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                    UpdCover.detr_loss[i,k] = self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * fabs(w_u) * self.detr_sc[i,k] * UpdCover.values[i,k]
                else:
                    UpdCover.detr_loss[i,k] = 0.0
        return


    cdef void compute_upd_covariance_turb_entr(self, GridMeanVariables GMV, EDMF_Updrafts.UpdraftVariable_2m UpdCover):
        cdef:
            Py_ssize_t i, k

        #with nogil:
        if UpdCover.name == 'tke':
            ck = self.tke_ed_coeff
        else:
            ck = self.tke_ed_coeff / self.prandtl_number
        for i in xrange(self.n_updrafts):
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.UpdVar.Area.values[i,k] >= self.minimum_area:
                    UpdCover.turb_entr[i,k] += -(self.Ref.rho0_half[k] * ck * self.mixing_length[k] * sqrt(GMV.TKE.values[k])
                                                  /self.pressure_plume_spacing**2.0*UpdCover.values[i,k])
                else:
                    UpdCover.turb_entr[i,k] = 0.0
        return

    # TODO  for anna to add
    # cpdef compute_upd_covariance_rain(self, TimeStepping TS, GridMeanVariables GMV):
    #     cdef:
    #         Py_ssize_t i, k
    #
    #         double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues) # area of environment
    #
    #     with nogil:
    #         for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
    #             self.UpdVar.TKE.rain_src[k] = 0.0
    #             self.UpdVar.Hvar.rain_src[k]   = self.Ref.rho0_half[k] * ae[k] * 2. * self.EnvThermo.Hvar_rain_dt[k]   * TS.dti
    #             self.UpdVar.QTvar.rain_src[k]  = self.Ref.rho0_half[k] * ae[k] * 2. * self.EnvThermo.QTvar_rain_dt[k]  * TS.dti
    #             self.UpdVar.HQTcov.rain_src[k] = self.Ref.rho0_half[k] * ae[k] *      self.EnvThermo.HQTcov_rain_dt[k] * TS.dti
    #
    #     return


    cdef void compute_upd_covariance_dissipation(self, EDMF_Updrafts.UpdraftVariable_2m UpdCover):
        cdef:
            Py_ssize_t i
            double m
            Py_ssize_t k
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    if self.UpdVar.Area.values[i,k] >self.minimum_area:
                        UpdCover.dissipation[i,k] = (self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * UpdCover.values[i,k]
                                            *pow(fmax(self.UpdVar.TKE.values[i,k],0), 0.5)/fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
                    else:
                        UpdCover.dissipation[i,k] = 0.0
        return



    cdef void update_upd_covariance_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS, VariablePrognostic GmvVar1, VariablePrognostic GmvVar2,
            VariableDiagnostic GmvCovar, EDMF_Updrafts.UpdraftVariable_2m UpdCovar, EDMF_Environment.EnvironmentVariable_2m EnvCovar, EDMF_Environment.EnvironmentVariable  EnvVar1,
                EDMF_Environment.EnvironmentVariable  EnvVar2, EDMF_Updrafts.UpdraftVariable  UpdVar1, EDMF_Updrafts.UpdraftVariable  UpdVar2):
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
            double [:] rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')
            double [:] whalf = np.zeros((nzg,),dtype=np.double, order='c')
            double  D_upd = 0.0
            double Covar_surf, we_half

        for i in xrange(self.n_updrafts):

            with nogil:
                for k in xrange(1,nzg-1):
                    rho_ae_K_m[k] = 0.5 * (self.UpdVar.Area.values[i,k]*self.KH.values[k]+ self.UpdVar.Area.values[i,k+1]*self.KH.values[k+1])* self.Ref.rho0[k]
                    whalf[k] = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
            we_half = interp2pt(self.EnvVar.W.values[gw-1], self.EnvVar.W.values[gw])

            if GmvCovar.name=='tke':
                GmvCovar.values[gw] =get_surface_tke(Case.Sur.ustar, self.wstar, self.Gr.z_half[gw], Case.Sur.obukhov_length)

            elif GmvCovar.name=='thetal_var':
                GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_hflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
            elif GmvCovar.name=='qt_var':
                GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_qtflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
            elif GmvCovar.name=='thetal_qt_covar':
                GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)

            self.get_upd_covar_from_GMV(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, UpdCovar, EnvCovar, &GmvVar1.values[0], &GmvVar2.values[0], &GmvCovar.values[0])

            Covar_surf = UpdCovar.values[i,gw]

            with nogil:
                for kk in xrange(nz):
                    if self.UpdVar.Area.values[i,k] > self.minimum_area:
                        k = kk+gw
                        D_upd = 0.0

                        #for i in xrange(self.n_updrafts):
                        #    we_half = interp2pt(self.UpdVar.W.values[i,k-1], self.UpdVar.W.values[i,k])
                        D_upd = self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * whalf[k] * self.detr_sc[i,k]


                        a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
                        b[kk] = (self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * dti - self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * whalf[k] * dzi
                                 + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                                 + D_upd
                                 + self.Ref.rho0_half[k] * self.UpdVar.Area.values[i,k] * self.tke_diss_coeff
                                            *sqrt(fmax(self.EnvVar.TKE.values[k],0))/fmax(self.mixing_length[k],1.0) )
                        c[kk] = (self.Ref.rho0_half[k+1] * self.UpdVar.Area.values[i,k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
                        x[kk] = (self.Ref.rho0_half[k] * self.UpdVar.Area.old[i,k] * UpdCovar.values[i,k] * dti
                                 + UpdCovar.press[i,k] + UpdCovar.buoy[i,k] + UpdCovar.shear[i,k] + UpdCovar.entr_gain[i,k] + UpdCovar.turb_entr[i,k]  +  UpdCovar.rain_src[i,k]) #


                    else:
                        a[kk] = 0.0
                        b[kk] = 1.0
                        c[kk] = 0.0
                        x[kk] = 0.0


                    a[0] = 0.0
                    b[0] = 1.0
                    c[0] = 0.0
                    x[0] = Covar_surf

                    b[nz-1] += c[nz-1]
                    c[nz-1] = 0.0
                    with gil:
                        print UpdCovar.press[i,k] + UpdCovar.buoy[i,k] , UpdCovar.shear[i,k] , UpdCovar.entr_gain[i,k] , UpdCovar.turb_entr[i,k] ,  UpdCovar.rain_src[i,k]
                        #plt.figure()
                        #plt.show()

            #tridiag_solve(self.Gr.nz, &x[0],&a[0], &b[0], &c[0])

            for kk in xrange(nz):
                k = kk + gw
                if UpdCovar.name == 'thetal_qt_covar':
                    UpdCovar.values[i,k] = fmax(x[kk], - sqrt(self.UpdVar.Hvar.values[i,k]*self.UpdVar.QTvar.values[i,k]))
                    UpdCovar.values[i,k] = fmin(x[kk],   sqrt(self.UpdVar.Hvar.values[i,k]*self.UpdVar.QTvar.values[i,k]))
                else:
                    UpdCovar.values[i,k] = fmax(x[kk],0.0)
            # TODO yair check the function below with the sent variables
            self.get_upd_covar_from_GMV(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, UpdCovar, EnvCovar, &GmvVar1.values[0], &GmvVar2.values[0], &GmvCovar.values[0])

        return
    # TODO assuming one updraft for now
    cdef get_upd_covar_from_GMV(self, EDMF_Updrafts.UpdraftVariable au,
                                EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                                EDMF_Environment.EnvironmentVariable phi_e, EDMF_Environment.EnvironmentVariable psi_e,
                                EDMF_Updrafts.UpdraftVariable_2m covar_u,EDMF_Environment.EnvironmentVariable_2m covar_e,
                                double *gmv_phi, double *gmv_psi, double *gmv_covar):
        cdef:
            Py_ssize_t i,k
            double phi_diff, psi_diff
            double tke_factor = 1.0
        if covar_u.name == 'tke':
            tke_factor = 0.5

        #with nogil:
        #for i in xrange(self.n_updrafts):
        i=0
        for k in xrange(self.Gr.nzg):
            if au.values[i,k] > self.minimum_area:
                if covar_u.name == 'tke':
                    phi_diff = interp2pt(phi_u.values[i,k-1] - phi_e.values[k-1],phi_u.values[i,k] - phi_e.values[k])
                    psi_diff = interp2pt(psi_u.values[i,k-1] - psi_e.values[k-1],psi_u.values[i,k] - psi_e.values[k])
                else:
                    phi_diff = phi_u.values[i,k] - phi_e.values[k]
                    psi_diff = psi_u.values[i,k] - psi_e.values[k]

                covar_u.values[i,k] = (gmv_covar[k] - (1.0-au.values[i,k])*covar_e.values[k]
                            - tke_factor*au.values[i,k]*(1.0-au.values[i,k])*phi_diff*psi_diff)/au.values[i,k]
            else:
                covar_u.values[i,k] = 0.0
        return

    # TODO - I am assuming that the updraft is fully cloudy for now
    cpdef compute_upd_tke_buoy(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            double d_alpha_thetal, d_alpha_qt, qv_cloudy, th_cloudy
            double lh, prefactor, cpm
            double grad_thl_minus=0.0, grad_qt_minus=0.0, grad_thl_plus=0, grad_qt_plus=0
            double [:] ae = np.subtract(np.ones((self.Gr.nzg,),dtype=np.double, order='c'),self.UpdVar.Area.bulkvalues)

        # Note that source terms at the gw grid point are not really used because that is where tke boundary condition is
        # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(gw, self.Gr.nzg-gw):
                    th_cloudy = self.UpdVar.THL.values[i,k]
                    qv_cloudy = self.UpdVar.QT.values[i,k]-self.UpdVar.QL.values[i,k] # if ice is added this should be changed
                    lh = latent_heat(self.UpdVar.T.values[i,k])
                    cpm = cpm_c(self.UpdVar.QT.values[i,k])
                    grad_thl_minus = grad_thl_plus
                    grad_qt_minus = grad_qt_plus
                    grad_thl_plus = (self.UpdVar.THL.values[i,k+1] - self.UpdVar.THL.values[i,k]) * self.Gr.dzi
                    grad_qt_plus  = (self.UpdVar.QT.values[i,k+1]  - self.UpdVar.QT.values[i,k])  * self.Gr.dzi

                    prefactor = Rd * exner_c(self.Ref.p0_half[k])/self.Ref.p0_half[k]



                    d_alpha_thetal = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / self.UpdVar.T.values[i,k]) * qv_cloudy - self.UpdVar.QT.values[i,k])
                                             / (1.0 + lh * lh / cpm / Rv / self.UpdVar.T.values[i,k] / self.UpdVar.T.values[i,k] * qv_cloudy))
                    d_alpha_qt = (lh / cpm / self.UpdVar.T.values[i,k] * d_alpha_thetal - prefactor) * self.UpdVar.H.values[i,k]



                    # TODO - check
                    self.UpdVar.TKE.buoy[i,k] = g / self.Ref.alpha0_half[k] * self.UpdVar.Area.values[i,k] * self.Ref.rho0_half[k] \
                                       * (- self.KH.values[k] * interp2pt(grad_thl_plus, grad_thl_minus) * d_alpha_thetal \
                                           - self.KH.values[k] * interp2pt(grad_qt_plus,  grad_qt_minus)  * d_alpha_qt)
        return