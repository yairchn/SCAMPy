#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
import cython
import sys
from Grid cimport Grid
cimport EDMF_Updrafts
cimport EDMF_Environment
cimport EDMF_Rain
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport SurfaceBase
from Cases cimport  CasesBase
from ReferenceState cimport  ReferenceState
from TimeStepping cimport TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport *
from microphysics_functions cimport *
from turbulence_functions cimport *
from utility_functions cimport *
from libc.math cimport fmax, sqrt, exp, pow, cbrt, fmin, fabs
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

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

        # TODO - steady updrafts have not been tested!
        try:
            self.use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
        except:
            self.use_steady_updrafts = False

        try:
            self.calc_tke = namelist['turbulence']['EDMF_PrognosticTKE']['calculate_tke']
        except:
            self.calc_tke = True

        try:
            self.use_const_plume_spacing = namelist['turbulence']['EDMF_PrognosticTKE']['use_constant_plume_spacing']
        except:
            self.use_const_plume_spacing = False

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
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'moisture_deficit':
                self.entr_detr_fp = entr_detr_env_moisture_deficit
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
            if str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_buoy']) == 'tan18':
                self.pressure_func_buoy = pressure_tan18_buoy
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_buoy']) == 'normalmode':
                self.pressure_func_buoy = pressure_normalmode_buoy
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_buoy']) == 'normalmode_buoysin':
                self.pressure_func_buoy = pressure_normalmode_buoysin
            else:
                print('Turbulence--EDMF_PrognosticTKE: pressure closure in namelist option is not recognized')
        except:
            self.pressure_func_buoy = pressure_tan18_buoy
            print('Turbulence--EDMF_PrognosticTKE: defaulting to pressure closure Tan2018')

        self.drag_sign = False
        try:
            if str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_drag']) == 'tan18':
                self.pressure_func_drag = pressure_tan18_drag
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_drag']) == 'normalmode':
                self.pressure_func_drag = pressure_normalmode_drag
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_drag']) == 'normalmode_signdf':
                self.pressure_func_drag = pressure_normalmode_drag
                self.drag_sign = True
            else:
                print('Turbulence--EDMF_PrognosticTKE: pressure closure in namelist option is not recognized')
        except:
            self.pressure_func_drag = pressure_tan18_drag
            print('Turbulence--EDMF_PrognosticTKE: defaulting to pressure closure Tan2018')

        try:
            self.asp_label = str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_asp_label'])
        except:
            self.asp_label = 'const'
            print('Turbulence--EDMF_PrognosticTKE: H/2R defaulting to constant')

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
        self.constant_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['constant_plume_spacing']
        self.sorting_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['sorting_factor']
        self.sorting_power = paramlist['turbulence']['EDMF_PrognosticTKE']['sorting_power']
        self.turbulent_entrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['turbulent_entrainment_factor']
        self.pressure_buoy_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.aspect_ratio = paramlist['turbulence']['EDMF_PrognosticTKE']['aspect_ratio']

        if str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_buoy']) == 'normalmode':
            try:
                self.pressure_normalmode_buoy_coeff1 = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_buoy_coeff1']
                self.pressure_normalmode_buoy_coeff2 = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_buoy_coeff2']
            except:
                self.pressure_normalmode_buoy_coeff1 = self.pressure_buoy_coeff
                self.pressure_normalmode_buoy_coeff2 = 0.0
                print 'Using (Tan et al, 2018) parameters as default for Normal Mode pressure formula buoyancy term'

        if str(namelist['turbulence']['EDMF_PrognosticTKE']['pressure_closure_drag']) == 'normalmode':
            try:
                self.pressure_normalmode_adv_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_adv_coeff']
                self.pressure_normalmode_drag_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_normalmode_drag_coeff']
            except:
                self.pressure_normalmode_adv_coeff = 0.0
                self.pressure_normalmode_drag_coeff = 1.0
                print 'Using (Tan et al, 2018) parameters as default for Normal Mode pressure formula drag term'

        # "Legacy" coefficients used by the steady updraft routine
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        if self.calc_tke == True:
            self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
            self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']

        # Need to code up as paramlist option?
        self.minimum_area = 1e-5

        # Create the class for rain
        self.Rain = EDMF_Rain.RainVariables(namelist, Gr)
        if self.use_steady_updrafts == True and self.Rain.rain_model != "None":
            sys.exit('PrognosticTKE: rain model is available for prognostic updrafts only')
        self.RainPhysics = EDMF_Rain.RainPhysics(Gr, Ref)

        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = EDMF_Updrafts.UpdraftVariables(self.n_updrafts, namelist,paramlist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = EDMF_Updrafts.UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar, self.Rain)

        # Create the environment variable class (major diagnostic and prognostic variables)
        self.EnvVar = EDMF_Environment.EnvironmentVariables(namelist,Gr)
        # Create the class for environment thermodynamics
        self.EnvThermo = EDMF_Environment.EnvironmentThermodynamics(namelist, Gr, Ref, self.EnvVar, self.Rain)

        # Entrainment rates
        self.entr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        #self.press = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Detrainment rates
        self.detr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        self.sorting_function = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.b_mix = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # turbulent entrainment
        self.frac_turb_entr = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.frac_turb_entr_full = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.turb_entr_W = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.turb_entr_H = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.turb_entr_QT = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Pressure term in updraft vertical momentum equation
        self.nh_pressure = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.nh_pressure_b = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.nh_pressure_adv = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.nh_pressure_drag = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.asp_ratio = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.b_coeff = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

        # mixing length
        self.mixing_length = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.horizontal_KM = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        self.horizontal_KH = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # diagnosed tke budget terms
        self.tke_transport = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.tke_advection = np.zeros((Gr.nzg,),dtype=np.double, order='c')

        # Near-surface BC of updraft area fraction
        self.area_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.h_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.qt_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.pressure_plume_spacing = np.zeros((self.n_updrafts,),dtype=np.double,order='c')

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
        # Variable Prandtl number initialized as neutral value.
        self.prandtl_nvec = np.multiply( self.prandtl_number, np.ones((Gr.nzg,),dtype=np.double, order='c'))
        self.mls = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.ml_ratio = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.l_entdet = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        self.b = np.zeros((Gr.nzg,),dtype=np.double, order='c')
        return

    cpdef initialize(self, GridMeanVariables GMV):
        self.UpdVar.initialize(GMV)
        return

    # Initialize the IO pertaining to this class
    cpdef initialize_io(self, NetCDFIO_Stats Stats):

        self.UpdVar.initialize_io(Stats)
        self.EnvVar.initialize_io(Stats)
        self.Rain.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('entrainment_sc')
        Stats.add_profile('detrainment_sc')
        Stats.add_profile('nh_pressure')
        Stats.add_profile('nh_pressure_adv')
        Stats.add_profile('nh_pressure_drag')
        Stats.add_profile('nh_pressure_b')
        Stats.add_profile('asp_ratio')
        Stats.add_profile('b_coeff')

        Stats.add_profile('horizontal_KM')
        Stats.add_profile('horizontal_KH')
        Stats.add_profile('sorting_function')
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
            double [:] mean_nh_pressure_adv = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_nh_pressure_drag = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_nh_pressure_b = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_asp_ratio = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_b_coeff = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

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
            double [:] mean_sorting_function = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')
            double [:] mean_b_mix = np.zeros((self.Gr.nzg,), dtype=np.double, order='c')

        self.UpdVar.io(Stats, self.Ref)
        self.EnvVar.io(Stats, self.Ref)
        self.Rain.io(Stats, self.Ref)

        Stats.write_profile('eddy_viscosity', self.KM.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('eddy_diffusivity', self.KH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_ts('rd', np.mean(self.pressure_plume_spacing))
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
                        mean_nh_pressure_b[k] += self.UpdVar.Area.values[i,k] * self.nh_pressure_b[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_nh_pressure_adv[k] += self.UpdVar.Area.values[i,k] * self.nh_pressure_adv[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_nh_pressure_drag[k] += self.UpdVar.Area.values[i,k] * self.nh_pressure_drag[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_asp_ratio[k] += self.UpdVar.Area.values[i,k] * self.asp_ratio[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_b_coeff[k] += self.UpdVar.Area.values[i,k] * self.b_coeff[i,k]/self.UpdVar.Area.bulkvalues[k]

                        mean_frac_turb_entr_full[k] += self.UpdVar.Area.values[i,k] * self.frac_turb_entr_full[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_frac_turb_entr[k] += self.UpdVar.Area.values[i,k] * self.frac_turb_entr[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_W[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_W[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_H[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_H[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_turb_entr_QT[k] += self.UpdVar.Area.values[i,k] * self.turb_entr_QT[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_horizontal_KM[k] += self.UpdVar.Area.values[i,k] * self.horizontal_KM[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_horizontal_KH[k] += self.UpdVar.Area.values[i,k] * self.horizontal_KH[i,k]/self.UpdVar.Area.bulkvalues[k]
                        mean_sorting_function[k] += self.UpdVar.Area.values[i,k] * self.sorting_function[i,k]/self.UpdVar.Area.bulkvalues[k]
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
        Stats.write_profile('sorting_function', mean_sorting_function[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('b_mix', mean_b_mix[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('nh_pressure', mean_nh_pressure[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('nh_pressure_adv', mean_nh_pressure_adv[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('nh_pressure_drag', mean_nh_pressure_drag[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('nh_pressure_b', mean_nh_pressure_b[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('asp_ratio', mean_asp_ratio[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('b_coeff', mean_b_coeff[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

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
        Stats.write_profile('updraft_qt_precip', self.UpdThermo.prec_source_qt_tot[kmin:kmax])
        Stats.write_profile('updraft_thetal_precip', self.UpdThermo.prec_source_h_tot[kmin:kmax])

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
    cpdef update(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):
        cdef:
            Py_ssize_t k
            Py_ssize_t kmin = self.Gr.gw
            Py_ssize_t kmax = self.Gr.nzg - self.Gr.gw

        self.update_inversion(GMV, Case.inversion_option)
        self.compute_pressure_plume_spacing(GMV, Case)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        if TS.nstep == 0:
            self.decompose_environment(GMV, 'values')
            self.EnvThermo.microphysics(self.EnvVar, self.Rain, TS.dt)
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
        # decompose_environment +  EnvThermo.saturation_adjustment + UpdThermo.buoyancy should always be used together
        # This ensures that:
        #   - the buoyancy of updrafts and environment is up to date with the most recent decomposition,
        #   - the buoyancy of updrafts and environment is updated such that
        #     the mean buoyancy with repect to reference state alpha_0 is zero.
        self.decompose_environment(GMV, 'mf_update')
        self.EnvThermo.microphysics(self.EnvVar, self.Rain, TS.dt) # saturation adjustment + rain creation
        # Sink of environmental QT and H due to rain creation is applied in tridiagonal solver
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.compute_eddy_diffusivities_tke(GMV, Case)
        self.update_GMV_ED(GMV, Case, TS)
        self.compute_covariance(GMV, Case, TS)

        if self.Rain.rain_model == "clima_1m":
            # sum updraft and environment rain into bulk rain
            self.Rain.sum_subdomains_rain(self.UpdThermo, self.EnvThermo)

            # rain fall (all three categories are assumed to be falling though "grid-mean" conditions
            self.RainPhysics.solve_rain_fall(GMV, TS, self.Rain.QR,     self.Rain.RainArea)
            self.RainPhysics.solve_rain_fall(GMV, TS, self.Rain.Upd_QR, self.Rain.Upd_RainArea)
            self.RainPhysics.solve_rain_fall(GMV, TS, self.Rain.Env_QR, self.Rain.Env_RainArea)

            # rain evaporation (all three categories are assumed to be evaporating in "grid-mean" conditions
            self.RainPhysics.solve_rain_evap(GMV, TS, self.Rain.QR,     self.Rain.RainArea)
            self.RainPhysics.solve_rain_evap(GMV, TS, self.Rain.Upd_QR, self.Rain.Upd_RainArea)
            self.RainPhysics.solve_rain_evap(GMV, TS, self.Rain.Env_QR, self.Rain.Env_RainArea)

        # update grid-mean cloud fraction and cloud cover
        for k in xrange(self.Gr.nzg):
            self.EnvVar.Area.values[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]
            GMV.cloud_fraction.values[k] = \
                self.EnvVar.Area.values[k] * self.EnvVar.cloud_fraction.values[k] +\
                self.UpdVar.Area.bulkvalues[k] * self.UpdVar.cloud_fraction[k]
        GMV.cloud_cover = min(self.EnvVar.cloud_cover + np.sum(self.UpdVar.cloud_cover), 1)

        # Back out the tendencies of the grid mean variables for the whole timestep
        # by differencing GMV.new and GMV.values
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

        self.UpdThermo.clear_precip_sources()

        while time_elapsed < TS.dt:
            self.compute_entrainment_detrainment(GMV, Case)
            if self.turbulent_entrainment_factor > 1.0e-6:
                self.compute_horizontal_eddy_diffusivities(GMV)
                self.compute_turbulent_entrainment(GMV,Case)
            self.compute_nh_pressure()
            self.solve_updraft_velocity_area()
            self.solve_updraft_scalars(GMV)
            self.UpdThermo.microphysics(self.UpdVar, self.Rain, TS.dt)
            self.UpdVar.set_values_with_new()
            self.zero_area_fraction_cleanup(GMV)
            time_elapsed += self.dt_upd
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * self.Gr.dz/fmax(np.max(self.UpdVar.W.values),1e-10))
            # (####)
            # TODO - see comment (###)
            # It would be better to have a simple linear rule for updating environment here
            # instead of calling EnvThermo saturation adjustment scheme for every updraft.
            self.decompose_environment(GMV, 'values')
            self.EnvThermo.saturation_adjustment(self.EnvVar)
            self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)
            self.set_subdomain_bcs()

        self.UpdThermo.update_total_precip_sources()
        return

    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t i, k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz
            double dzi = self.Gr.dzi
            eos_struct sa
            mph_struct mph
            entr_struct ret
            entr_in_struct input
            double a,b,c, w, w_km,  w_mid, w_low, denom, arg
            double entr_w, detr_w, B_k, area_k, w2

        self.set_updraft_surface_bc(GMV, Case)
        self.compute_entrainment_detrainment(GMV, Case)

        with nogil:
            for i in xrange(self.n_updrafts):
                self.UpdVar.H.values[i,gw]  = self.h_surface_bc[i]
                self.UpdVar.QT.values[i,gw] = self.qt_surface_bc[i]

                # do saturation adjustment
                sa = eos(
                    self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp,
                    self.Ref.p0_half[gw], self.UpdVar.QT.values[i,gw],
                    self.UpdVar.H.values[i,gw]
                )
                self.UpdVar.QL.values[i,gw] = sa.ql
                self.UpdVar.T.values[i, gw] = sa.T

                for k in xrange(gw+1, self.Gr.nzg-gw):
                    denom = 1.0 + self.entr_sc[i,k] * dz
                    self.UpdVar.H.values[i,k]  = (self.UpdVar.H.values[i, k-1] + self.entr_sc[i,k] * dz * GMV.H.values[k])/denom
                    self.UpdVar.QT.values[i,k] = (self.UpdVar.QT.values[i,k-1] + self.entr_sc[i,k] * dz * GMV.QT.values[k])/denom

                    # do saturation adjustment
                    sa = eos(
                        self.UpdThermo.t_to_prog_fp, self.UpdThermo.prog_to_t_fp,
                        self.Ref.p0_half[k], self.UpdVar.QT.values[i,k],
                        self.UpdVar.H.values[i,k]
                    )
                    self.UpdVar.QL.values[i,k] = sa.ql
                    self.UpdVar.T.values[i, k] = sa.T

        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.H.set_bcs(self.Gr)

        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.saturation_adjustment(self.EnvVar)
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
                              /(0.5 * dzi +entr_w + (1.0/self.pressure_plume_spacing[i])/sqrt(fmax(area_k,self.minimum_area))))
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
                        #TODO wouldnt it be more consistent to have here?
                        #self.UpdVar.QL.values[i,k] = GMV.QL.values[k]
                        #self.UpdVar.T.values[i,k] = GMV.T.values[k]
                        sa = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, self.Ref.p0_half[k],
                                 self.UpdVar.QT.values[i,k], self.UpdVar.H.values[i,k])
                        self.UpdVar.QL.values[i,k] = sa.ql
                        self.UpdVar.T.values[i,k] = sa.T

        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.saturation_adjustment(self.EnvVar)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.UpdVar.Area.set_bcs(self.Gr)

        return

    cpdef update_inversion(self,GridMeanVariables GMV, option):
        ParameterizationBase.update_inversion(self, GMV,option)
        return

    cpdef compute_mixing_length(self, double obukhov_length, double ustar, GridMeanVariables GMV):

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
            double grad_thl_plus=0.0, grad_qt_plus=0.0, grad_thv_plus=0.0, grad_th_plus=0.0
            double thv, grad_qt, grad_qt_low, grad_thv_low, grad_thv
            double th, grad_th_low, grad_th, heating_ratio
            double grad_b_thl, grad_b_qt
            double m_eps = 1.0e-9 # Epsilon to avoid zero
            double a, c_neg, wc_upd_nn, wc_env

        if (self.mixing_scheme == 'sbl'):
            for k in xrange(gw, self.Gr.nzg-gw):
                z_ = self.Gr.z_half[k]
                # kz scale (surface layer)
                if obukhov_length < 0.0: #unstable
                    l2 = vkb * z_ /(sqrt(self.EnvVar.TKE.values[self.Gr.gw]/ustar/ustar)*self.tke_ed_coeff) * fmin(
                     (1.0 - 100.0 * z_/obukhov_length)**0.2, 1.0/vkb )
                else: # neutral or stable
                    l2 = vkb * z_ /(sqrt(self.EnvVar.TKE.values[self.Gr.gw]/ustar/ustar)*self.tke_ed_coeff)

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

                if self.EnvVar.cloud_fraction.values[k] > 0.0:
                    d_buoy_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_buoy_qt_cloudy = (lh / cpm / t_cloudy * d_buoy_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_buoy_thetal_cloudy = 0.0
                    d_buoy_qt_cloudy = 0.0

                d_buoy_thetal_total = (self.EnvVar.cloud_fraction.values[k] * d_buoy_thetal_cloudy
                                        + (1.0-self.EnvVar.cloud_fraction.values[k]) * d_buoy_thetal_dry)
                d_buoy_qt_total = (self.EnvVar.cloud_fraction.values[k] * d_buoy_qt_cloudy
                                    + (1.0-self.EnvVar.cloud_fraction.values[k]) * d_buoy_qt_dry)

                # Partial buoyancy gradients
                grad_b_thl  = grad_thl * d_buoy_thetal_total
                grad_b_qt = grad_qt  * d_buoy_qt_total
                ri_grad = fmin( grad_b_thl/fmax(shear2, m_eps) + grad_b_qt/fmax(shear2, m_eps) , 0.25)

                # Turbulent Prandtl number:
                if obukhov_length > 0.0 and ri_grad>0.0: #stable
                    # CSB (Dan Li, 2019), with Pr_neutral=0.74 and w1=40.0/13.0
                    self.prandtl_nvec[k] = 0.74*( 2.0*ri_grad/
                        (1.0+(53.0/13.0)*ri_grad -sqrt( (1.0+(53.0/13.0)*ri_grad)**2.0 - 4.0*ri_grad ) ) )
                else:
                    self.prandtl_nvec[k] = 0.74

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
                    l2 = vkb * z_ /(sqrt(self.EnvVar.TKE.values[self.Gr.gw]/ustar/ustar)*self.tke_ed_coeff) * fmin(
                     (1.0 - 100.0 * z_/obukhov_length)**0.2, 1.0/vkb )
                else: # neutral or stable
                    l2 = vkb * z_ /(sqrt(self.EnvVar.TKE.values[self.Gr.gw]/ustar/ustar)*self.tke_ed_coeff)

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

                if self.EnvVar.cloud_fraction.values[k] > 0.0:
                    d_buoy_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_buoy_qt_cloudy = (lh / cpm / t_cloudy * d_buoy_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_buoy_thetal_cloudy = 0.0
                    d_buoy_qt_cloudy = 0.0

                d_buoy_thetal_total = (self.EnvVar.cloud_fraction.values[k] * d_buoy_thetal_cloudy
                                        + (1.0-self.EnvVar.cloud_fraction.values[k]) * d_buoy_thetal_dry)
                d_buoy_qt_total = (self.EnvVar.cloud_fraction.values[k] * d_buoy_qt_cloudy
                                    + (1.0-self.EnvVar.cloud_fraction.values[k]) * d_buoy_qt_dry)

                # Partial buoyancy gradients
                grad_b_thl = grad_thl * d_buoy_thetal_total
                grad_b_qt  = grad_qt  * d_buoy_qt_total
                ri_grad = fmin( grad_b_thl/fmax(shear2, m_eps) + grad_b_qt/fmax(shear2, m_eps) , 0.25)

                # Turbulent Prandtl number:
                if obukhov_length > 0.0 and ri_grad>0.0: #stable
                    # CSB (Dan Li, 2019), with Pr_neutral=0.74 and w1=40.0/13.0
                    self.prandtl_nvec[k] = 0.74*( 2.0*ri_grad/
                        (1.0+(53.0/13.0)*ri_grad -sqrt( (1.0+(53.0/13.0)*ri_grad)**2.0 - 4.0*ri_grad ) ) )
                else:
                    self.prandtl_nvec[k] = 0.74

                # Production/destruction terms
                a = self.tke_ed_coeff*(shear2 - grad_b_thl/self.prandtl_nvec[k] - grad_b_qt/self.prandtl_nvec[k])* sqrt(self.EnvVar.TKE.values[k])
                # Dissipation term
                c_neg = self.tke_diss_coeff*self.EnvVar.TKE.values[k]*sqrt(self.EnvVar.TKE.values[k])
                # Subdomain exchange term
                self.b[k] = 0.0
                for nn in xrange(self.n_updrafts):
                    wc_upd_nn = (self.UpdVar.W.values[nn,k] + self.UpdVar.W.values[nn,k-1])/2.0
                    wc_env = (self.EnvVar.W.values[k] + self.EnvVar.W.values[k-1])/2.0
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
                
                th = theta_c(self.Ref.p0_half[k], self.EnvVar.T.values[k])
                grad_th_low = grad_th_plus
                grad_th_plus = ( theta_c(self.Ref.p0_half[k+1], self.EnvVar.T.values[k+1]) - th) * self.Gr.dzi
                grad_th = interp2pt(grad_th_low, grad_th_plus)

                # Effective static stability. heating_ratio reflects latent heat effects on stability.
                heating_ratio = 0.5
                grad_th_eff = grad_thv - heating_ratio*(
                    1.0 + 0.61*self.EnvVar.QT.values[k] - (1.0 + 0.61)*self.EnvVar.QL.values[k]
                    )*(grad_th - grad_thl*th/self.EnvVar.THL.values[k])

                N = sqrt(fmax(g/thv*grad_th_eff, 0.0))
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
            # default mixing scheme , see Tan et al. (2018)
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
            self.compute_mixing_length(Case.Sur.obukhov_length, Case.Sur.ustar, GMV)
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
                        R_up = self.pressure_plume_spacing[i]*sqrt(self.UpdVar.Area.values[i,k])
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
            
        if self.calc_tke:
            self.EnvVar.TKE.values[self.Gr.gw] = get_surface_tke(Case.Sur.ustar,
                                                     self.wstar,
                                                     self.Gr.z_half[self.Gr.gw],
                                                     Case.Sur.obukhov_length)
            self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W, self.EnvVar.TKE,
                &GMV.W.values[0],&GMV.W.values[0], &GMV.TKE.values[0])

        if self.calc_scalar_var:
            self.EnvVar.Hvar.values[self.Gr.gw] = get_surface_variance(flux1*alpha0LL,flux1*alpha0LL, ustar, zLL, oblength)
            self.EnvVar.QTvar.values[self.Gr.gw] = get_surface_variance(flux2*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
            self.EnvVar.HQTcov.values[self.Gr.gw] = get_surface_variance(flux1*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
            self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar,
                             &GMV.H.values[0],&GMV.H.values[0], &GMV.Hvar.values[0])
            self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar,
                             &GMV.QT.values[0],&GMV.QT.values[0], &GMV.QTvar.values[0])
            self.get_GMV_CoVar(self.UpdVar.Area,self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov,
                             &GMV.H.values[0], &GMV.QT.values[0], &GMV.HQTcov.values[0])
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

                    self.EnvVar.Area.values[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]
                    self.EnvVar.QT.values[k] = fmax(val1 * GMV.QT.values[k] - val2 * self.UpdVar.QT.bulkvalues[k],0.0) #Yair - this is here to prevent negative QT
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

                    self.EnvVar.QT.values[k] = fmax(val1 * GMV.QT.mf_update[k] - val2 * self.UpdVar.QT.bulkvalues[k],0.0)#Yair - this is here to prevent negative QT
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
                    R_up = self.pressure_plume_spacing[i]*sqrt(a)
                    R_up_full = self.pressure_plume_spacing[i]*sqrt(a_full)
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

    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case):
        cdef:
            Py_ssize_t k
            entr_struct ret
            entr_in_struct input
            eos_struct sa
            double transport_plus, transport_minus
            long quadrature_order = 3

        self.UpdVar.upd_cloud_diagnostics(self.Ref)

        input.wstar = self.wstar

        input.dz = self.Gr.dz
        input.zbl = self.compute_zbl_qt_grad(GMV)
        for i in xrange(self.n_updrafts):
            input.zi = self.UpdVar.cloud_base[i]
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
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
                    input.sort_pow = self.sorting_power
                    input.sort_fact = self.sorting_factor
                    input.rd = self.pressure_plume_spacing[i]
                    input.nh_pressure = self.nh_pressure[i,k]
                    input.RH_upd = self.UpdVar.RH.values[i,k]
                    input.RH_env = self.EnvVar.RH.values[k]

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
                    self.sorting_function[i,k] = ret.sorting_function
                    self.b_mix[i,k] = ret.b_mix

                else:
                    self.entr_sc[i,k] = 0.0
                    self.detr_sc[i,k] = 0.0
                    self.sorting_function[i,k] = 0.0
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
            double cpm

        for i in xrange(self.n_updrafts):
            if self.use_const_plume_spacing:
                self.pressure_plume_spacing[i] = self.constant_plume_spacing
            else:
                self.pressure_plume_spacing[i] = fmax(self.aspect_ratio*self.UpdVar.updraft_top[i], 500.0)
        return

    cpdef compute_nh_pressure(self):
        cdef:
            Py_ssize_t i,k, alen
            pressure_buoy_struct ret_b
            pressure_drag_struct ret_w
            pressure_in_struct input

        for i in xrange(self.n_updrafts):
            input.updraft_top = self.UpdVar.updraft_top[i]
            alen = len(np.argwhere(self.UpdVar.Area.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw]))
            input.a_med = np.median(self.UpdVar.Area.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw][:alen])
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                input.a_kfull = interp2pt(self.UpdVar.Area.values[i,k], self.UpdVar.Area.values[i,k+1])
                input.dzi = self.Gr.dzi
                input.z_full = self.Gr.z[k]

                input.a_khalf = self.UpdVar.Area.values[i,k]
                input.a_kphalf = self.UpdVar.Area.values[i,k+1]
                input.b_kfull = interp2pt(self.UpdVar.B.values[i,k], self.UpdVar.B.values[i,k+1])
                input.rho0_kfull = self.Ref.rho0[k]
                input.bcoeff_tan18 = self.pressure_buoy_coeff
                input.alpha1 = self.pressure_normalmode_buoy_coeff1
                input.alpha2 = self.pressure_normalmode_buoy_coeff2
                input.beta1 = self.pressure_normalmode_adv_coeff
                input.beta2 = self.pressure_normalmode_drag_coeff
                input.rd = self.pressure_plume_spacing[i]
                input.w_kfull = self.UpdVar.W.values[i,k]
                input.w_khalf = interp2pt(self.UpdVar.W.values[i,k], self.UpdVar.W.values[i,k-1])
                input.w_kphalf = interp2pt(self.UpdVar.W.values[i,k], self.UpdVar.W.values[i,k+1])
                input.w_kenv = self.EnvVar.W.values[k]
                input.drag_sign = self.drag_sign

                if self.asp_label == 'z_dependent':
                    input.asp_ratio = input.updraft_top/2.0/sqrt(input.a_kfull)/input.rd
                elif self.asp_label == 'median':
                    input.asp_ratio = input.updraft_top/2.0/sqrt(input.a_med)/input.rd
                elif self.asp_label == 'const':
                    # _ret.asp_ratio = 1.72
                    input.asp_ratio = 1.0

                if input.a_kfull>0.0:
                    ret_b = self.pressure_func_buoy(input)
                    ret_w = self.pressure_func_drag(input)
                    self.nh_pressure_b[i,k] = ret_b.nh_pressure_b
                    self.nh_pressure_adv[i,k] = ret_w.nh_pressure_adv
                    self.nh_pressure_drag[i,k] = ret_w.nh_pressure_drag

                    self.b_coeff[i,k] = ret_b.b_coeff
                    self.asp_ratio[i,k] = input.asp_ratio

                else:
                    self.nh_pressure_b[i,k] = 0.0
                    self.nh_pressure_adv[i,k] = 0.0
                    self.nh_pressure_drag[i,k] = 0.0

                    self.b_coeff[i,k] = 0.0
                    self.asp_ratio[i,k] = 0.0

                self.nh_pressure[i,k] = self.nh_pressure_b[i,k] + self.nh_pressure_adv[i,k] + self.nh_pressure_drag[i,k]

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
                    self.UpdVar.THL.values[i,k] = GMV.THL.values[k]

            if np.sum(self.UpdVar.Area.values[:,k])==0.0:
                self.EnvVar.W.values[k] = GMV.W.values[k]
                self.EnvVar.B.values[k] = GMV.B.values[k]
                self.EnvVar.H.values[k] = GMV.H.values[k]
                self.EnvVar.QT.values[k] = GMV.QT.values[k]
                self.EnvVar.T.values[k] = GMV.T.values[k]
                self.EnvVar.QL.values[k] = GMV.QL.values[k]
                self.EnvVar.THL.values[k] = GMV.THL.values[k]

        return


    cpdef set_subdomain_bcs(self):

        self.UpdVar.W.set_bcs(self.Gr)
        self.UpdVar.Area.set_bcs(self.Gr)
        self.UpdVar.H.set_bcs(self.Gr)
        self.UpdVar.QT.set_bcs(self.Gr)
        self.UpdVar.T.set_bcs(self.Gr)
        self.UpdVar.B.set_bcs(self.Gr)

        self.EnvVar.W.set_bcs(self.Gr)
        self.EnvVar.H.set_bcs(self.Gr)
        self.EnvVar.T.set_bcs(self.Gr)
        self.EnvVar.QL.set_bcs(self.Gr)
        self.EnvVar.QT.set_bcs(self.Gr)

        return

    cpdef solve_updraft_velocity_area(self):
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
                            self.UpdVar.W.new[i,k] = 0.0
                            self.UpdVar.Area.new[i,k+1] = 0.0
                            #break
                    else:
                        self.UpdVar.W.new[i,k] = 0.0
                        self.UpdVar.Area.new[i,k+1] = 0.0
                        # keep this in mind if we modify updraft top treatment!
                        #break
        return

    cpdef solve_updraft_scalars(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t k, i
            double dzi = self.Gr.dzi
            double dti_ = 1.0/self.dt_upd
            double m_k, m_km
            Py_ssize_t gw = self.Gr.gw
            double H_entr, QT_entr
            double c1, c2, c3, c4
            eos_struct sa

        with nogil:
            for i in xrange(self.n_updrafts):

                # at the surface:
                self.UpdVar.H.new[i,gw] = self.h_surface_bc[i]
                self.UpdVar.QT.new[i,gw] = self.qt_surface_bc[i]

                # saturation adjustment
                sa = eos(
                    self.UpdThermo.t_to_prog_fp, self.UpdThermo.prog_to_t_fp,
                    self.Ref.p0_half[gw], self.UpdVar.QT.new[i,gw],
                    self.UpdVar.H.new[i,gw]
                )
                self.UpdVar.QL.new[i,gw] = sa.ql
                self.UpdVar.T.new[i,gw] = sa.T

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
                                                   + c4 * QT_entr + self.turb_entr_QT[i,k])/c1

                    else:
                        self.UpdVar.H.new[i,k]  = GMV.H.values[k]
                        self.UpdVar.QT.new[i,k] = GMV.QT.values[k]

                    # saturation adjustment
                    sa = eos(
                        self.UpdThermo.t_to_prog_fp,
                        self.UpdThermo.prog_to_t_fp,
                        self.Ref.p0_half[k],
                        self.UpdVar.QT.new[i,k],
                        self.UpdVar.H.new[i,k]
                    )
                    self.UpdVar.QL.new[i,k] = sa.ql
                    self.UpdVar.T.new[i,k] = sa.T

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

                GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h + self.UpdThermo.prec_source_h_tot[k]
                GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt + self.UpdThermo.prec_source_qt_tot[k]

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
                GMV.QT.new[k+gw] = fmax(\
                                   GMV.QT.mf_update[k+gw]\
                                   + ae[k+gw] *(x[k] - self.EnvVar.QT.values[k+gw])\
                                   + self.EnvThermo.prec_source_qt[k+gw]\
                                   + self.RainPhysics.rain_evap_source_qt[k+gw]
                                   ,0.0)
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
                GMV.H.new[k+gw] = GMV.H.mf_update[k+gw]\
                                  + ae[k+gw] *(x[k] - self.EnvVar.H.values[k+gw])\
                                  + self.EnvThermo.prec_source_h[k+gw]\
                                  + self.RainPhysics.rain_evap_source_h[k+gw]
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

                if self.EnvVar.cloud_fraction.values[k] > 0.0:
                    d_alpha_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                             / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                    d_alpha_qt_cloudy = (lh / cpm / t_cloudy * d_alpha_thetal_cloudy - prefactor) * th_cloudy
                else:
                    d_alpha_thetal_cloudy = 0.0
                    d_alpha_qt_cloudy = 0.0

                d_alpha_thetal_total = (self.EnvVar.cloud_fraction.values[k] * d_alpha_thetal_cloudy
                                        + (1.0-self.EnvVar.cloud_fraction.values[k]) * d_alpha_thetal_dry)
                d_alpha_qt_total = (self.EnvVar.cloud_fraction.values[k] * d_alpha_qt_cloudy
                                    + (1.0-self.EnvVar.cloud_fraction.values[k]) * d_alpha_qt_dry)

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
                GMV.QL.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QL.bulkvalues[k] \
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QL.values[k])

                GMV.QR.values[k]  = self.Rain.QR.values[k] * self.Rain.RainArea.values[k]

                GMV.T.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.T.bulkvalues[k] \
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.T.values[k])

                qv = GMV.QT.values[k] - GMV.QL.values[k]

                GMV.THL.values[k] = t_to_thetali_c(self.Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k],
                                                   GMV.QL.values[k], 0.0)

                GMV.B.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.B.bulkvalues[k] \
                                    + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.B.values[k])

        return

    cpdef compute_covariance(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS):

        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed
            self.compute_mixing_length(Case.Sur.obukhov_length, Case.Sur.ustar, GMV)
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
                    R_up = self.pressure_plume_spacing[i]*sqrt(self.UpdVar.Area.values[i,k])
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

                    Covar.entr_gain[k]  += tke_factor * combined_entr * (updvar1 - envvar1) * (updvar2 - envvar2)
                    Covar.detr_loss[k]  += combined_detr * Covar.values[k]
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

        # Not necessary if BCs for variances are applied to environment.
        # if GmvCovar.name=='tke':
        #     GmvCovar.values[gw] =get_surface_tke(Case.Sur.ustar, self.wstar, self.Gr.z_half[gw], Case.Sur.obukhov_length)
        # elif GmvCovar.name=='thetal_var':
        #     GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_hflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        # elif GmvCovar.name=='qt_var':
        #     GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_qtflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        # elif GmvCovar.name=='thetal_qt_covar':
        #     GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        # self.get_env_covar_from_GMV(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, &GmvVar1.values[0], &GmvVar2.values[0], &GmvCovar.values[0])

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

                            R_up = self.pressure_plume_spacing[i]*sqrt(self.UpdVar.Area.values[i,k])
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
