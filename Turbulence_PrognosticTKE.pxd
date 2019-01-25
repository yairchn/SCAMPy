cimport EDMF_Updrafts
cimport EDMF_Environment
from Grid cimport Grid
from Variables cimport VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface cimport  SurfaceBase
from ReferenceState cimport  ReferenceState
from Cases cimport CasesBase
from TimeStepping cimport  TimeStepping
from NetCDFIO cimport NetCDFIO_Stats
from turbulence_functions cimport entr_struct, entr_in_struct
from Turbulence cimport ParameterizationBase


cdef class EDMF_PrognosticTKE(ParameterizationBase):
    cdef:
        Py_ssize_t n_updrafts
        EDMF_Updrafts.UpdraftVariables UpdVar
        EDMF_Updrafts.UpdraftMicrophysics UpdMicro
        EDMF_Updrafts.UpdraftThermodynamics UpdThermo
        EDMF_Environment.EnvironmentVariables EnvVar
        EDMF_Environment.EnvironmentThermodynamics EnvThermo
        entr_struct (*entr_detr_fp) (entr_in_struct entr_in) nogil
        bint use_local_micro
        bint similarity_diffusivity
        bint use_steady_updrafts
        bint calc_scalar_var
        bint calc_tke
        double surface_area
        double minimum_area
        double entrainment_factor
        double detrainment_factor
        double entrainment_alpha1
        double entrainment_alpha2
        double entrainment_alpha3
        double detrainment_alpha1
        double detrainment_alpha2
        double detrainment_alpha3
        double vel_pressure_coeff # used by diagnostic plume option; now calc'ed from Tan et al 2018 coefficient set
        double vel_buoy_coeff # used by diagnostic plume option; now calc'ed from Tan et al 2018 coefficient set
        double pressure_buoy_coeff # Tan et al. 2018: coefficient alpha_b in Eq. 30
        double pressure_drag_coeff # Tan et al. 2018: coefficient alpha_d in Eq. 30
        double pressure_plume_spacing # Tan et al. 2018: coefficient r_d in Eq. 30
        double dt_upd
        double [:,:] entr_sc
        double [:,:] detr_sc
        double [:,:] updraft_pressure_sink
        double [:] area_surface_bc
        double [:] h_surface_bc
        double [:] qt_surface_bc
        double [:] w_surface_bc
        double [:,:] m # mass flux
        double [:] massflux_h
        double [:] massflux_qt
        double [:] massflux_tke
        double [:] massflux_tendency_h
        double [:] massflux_tendency_qt
        double [:] diffusive_flux_h
        double [:] diffusive_flux_qt
        double [:] diffusive_tendency_h
        double [:] diffusive_tendency_qt
        double [:] mixing_length
        double [:,:] upd_mixing_length
        double [:] tke_buoy
        double [:] tke_dissipation
        double [:] tke_entr_gain
        double [:] tke_detr_loss
        double [:] tke_shear
        double [:] tke_pressure
        double max_area_factor
        double tke_ed_coeff
        double tke_diss_coeff
        double gcm_resolution
        double [:,:] w_press_term
        double [:,:] turb_entr_W
        double [:,:] turb_entr_H
        double [:,:] turb_entr_QT


        double [:] Hvar_shear
        double [:] QTvar_shear
        double [:] Hvar_entr_gain
        double [:] QTvar_entr_gain
        double [:] Hvar_detr_loss
        double [:] QTvar_detr_loss
        double [:] Hvar_diss_coeff
        double [:] QTvar_diss_coeff
        double [:] HQTcov
        double [:] HQTcov_shear
        double [:] HQTcov_entr_gain
        double [:] HQTcov_detr_loss
        double [:] HQTcov_diss_coeff
        double [:] Hvar_dissipation
        double [:] QTvar_dissipation
        double [:] HQTcov_dissipation
        double [:] Hvar_rain
        double [:] QTvar_rain
        double [:] HQTcov_rain


        double [:] mls
        double [:] ml_ratio
        str mixing_scheme

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats)
    cpdef update(self,GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_prognostic_updrafts(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_diagnostic_updrafts(self, GridMeanVariables GMV, CasesBase Case)
    cpdef update_inversion(self, GridMeanVariables GMV, option)
    cpdef compute_mixing_length(self, double obukhov_length, GridMeanVariables GMV)
    cpdef compute_eddy_diffusivities_tke(self, GridMeanVariables GMV, CasesBase Case)
    cpdef reset_surface_covariance(self, GridMeanVariables GMV, CasesBase Case)
    cpdef set_updraft_surface_bc(self, GridMeanVariables GMV, CasesBase Case)
    cpdef decompose_environment(self, GridMeanVariables GMV, whichvals)
    cpdef compute_entrainment_detrainment(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef solve_updraft_velocity_area(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef solve_updraft_scalars(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef update_GMV_MF(self, GridMeanVariables GMV, TimeStepping TS)
    cpdef update_GMV_ED(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_covariance(self, GridMeanVariables GMV, CasesBase Case, TimeStepping TS)
    cpdef compute_turbulent_entrainment(self, GridMeanVariables GMV, CasesBase Case)
    cpdef initialize_covariance(self, GridMeanVariables GMV, CasesBase Case)
    cpdef cleanup_covariance(self, GridMeanVariables GMV)
    cpdef compute_tke_buoy(self, GridMeanVariables GMV)
    cpdef compute_upd_tke_buoy(self, GridMeanVariables GMV)
    cpdef compute_tke_pressure(self)
    cpdef compute_upd_tke_pressure(self)
    cpdef compute_w_pressure_term(self, CasesBase Case)
    cpdef compute_updraft_diffusion(self, CasesBase Case)
    cdef void compute_covariance_dissipation(self, EDMF_Environment.EnvironmentVariable_2m Covar)
    cdef void compute_covariance_entr(self, EDMF_Environment.EnvironmentVariable_2m Covar, EDMF_Updrafts.UpdraftVariable_2m UpdCovar, EDMF_Updrafts.UpdraftVariable UpdVar1,
                EDMF_Updrafts.UpdraftVariable UpdVar2, EDMF_Environment.EnvironmentVariable EnvVar1, EDMF_Environment.EnvironmentVariable EnvVar2)
    cdef void compute_upd_covariance_entr(self, EDMF_Updrafts.UpdraftVariable_2m UpdCovar,EDMF_Environment.EnvironmentVariable_2m EnvCovar, EDMF_Updrafts.UpdraftVariable UpdVar1,
                EDMF_Updrafts.UpdraftVariable UpdVar2, EDMF_Environment.EnvironmentVariable EnvVar1, EDMF_Environment.EnvironmentVariable EnvVar2)
    cdef void compute_covariance_detr(self, EDMF_Environment.EnvironmentVariable_2m Covar)
    cdef void compute_upd_covariance_detr(self, EDMF_Updrafts.UpdraftVariable_2m Covar)
    cdef void compute_covariance_turb_entr(self, GridMeanVariables GMV, EDMF_Environment.EnvironmentVariable_2m EnvCovar, EDMF_Updrafts.UpdraftVariable_2m UpdCovar)
    cdef void compute_covariance_entr_massflux(self, EDMF_Environment.EnvironmentVariable_2m EnvCovar, EDMF_Updrafts.UpdraftVariable_2m UpdCovar, EDMF_Updrafts.UpdraftVariable UpdVar1,
                EDMF_Updrafts.UpdraftVariable UpdVar2, EDMF_Environment.EnvironmentVariable EnvVar1, EDMF_Environment.EnvironmentVariable EnvVar2)
    #cdef void compute_upd_covariance_turb_entr(self, GridMeanVariables GMV, EDMF_Updrafts.UpdraftVariable_2m Covar)
    cdef void compute_covariance_shear(self,GridMeanVariables GMV, EDMF_Environment.EnvironmentVariable_2m Covar,
                                       double *UpdVar1, double *UpdVar2, double *EnvVar1, double *EnvVar2)
    cdef void compute_upd_covariance_shear(self,GridMeanVariables GMV, EDMF_Updrafts.UpdraftVariable_2m Covar,
                                EDMF_Updrafts.UpdraftVariable UpdVar1, EDMF_Updrafts.UpdraftVariable UpdVar2)
    cpdef compute_covariance_rain(self, TimeStepping TS, GridMeanVariables GMV)
    cpdef compute_upd_covariance_rain(self, TimeStepping TS, GridMeanVariables GMV)
    cdef void compute_upd_covariance_dissipation(self, EDMF_Updrafts.UpdraftVariable_2m UpdCovar)
    cdef void compute_covariance_interdomain_src(self, EDMF_Updrafts.UpdraftVariable au, EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                        EDMF_Environment.EnvironmentVariable phi_e,  EDMF_Environment.EnvironmentVariable psi_e, EDMF_Environment.EnvironmentVariable_2m covar_e)
    cdef void update_covariance_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS, VariablePrognostic GmvVar1, VariablePrognostic GmvVar2,
            VariableDiagnostic GmvCovar, EDMF_Environment.EnvironmentVariable_2m Covar, EDMF_Updrafts.UpdraftVariable_2m UpdCovar, EDMF_Environment.EnvironmentVariable  EnvVar1, EDMF_Environment.EnvironmentVariable  EnvVar2,
                                   EDMF_Updrafts.UpdraftVariable  UpdVar1, EDMF_Updrafts.UpdraftVariable  UpdVar2)
    cdef void update_upd_covariance_ED(self, GridMeanVariables GMV, CasesBase Case,TimeStepping TS, VariablePrognostic GmvVar1, VariablePrognostic GmvVar2,
            VariableDiagnostic GmvCovar, EDMF_Updrafts.UpdraftVariable_2m Covar, EDMF_Environment.EnvironmentVariable_2m Covar_e, EDMF_Environment.EnvironmentVariable  EnvVar1, EDMF_Environment.EnvironmentVariable  EnvVar2,
                                   EDMF_Updrafts.UpdraftVariable  UpdVar1, EDMF_Updrafts.UpdraftVariable  UpdVar2)
    cpdef update_GMV_diagnostics(self, GridMeanVariables GMV)
    cpdef double compute_zbl_qt_grad(self, GridMeanVariables GMV)
    cdef get_GMV_CoVar(self, EDMF_Updrafts.UpdraftVariable au,
                        EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                        EDMF_Environment.EnvironmentVariable phi_e,  EDMF_Environment.EnvironmentVariable psi_e,
                        EDMF_Environment.EnvironmentVariable_2m covar_e, EDMF_Updrafts.UpdraftVariable_2m covar_u,
                       double *gmv_phi, double *gmv_psi, double *gmv_covar)
    cdef get_env_covar_from_GMV(self, EDMF_Updrafts.UpdraftVariable au,
                                EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                                EDMF_Environment.EnvironmentVariable phi_e, EDMF_Environment.EnvironmentVariable psi_e,
                                EDMF_Environment.EnvironmentVariable_2m covar_e, EDMF_Updrafts.UpdraftVariable_2m covar_u,
                                double *gmv_phi, double *gmv_psi, double *gmv_covar)

    cdef get_upd_covar_from_GMV(self, EDMF_Updrafts.UpdraftVariable au,
                                EDMF_Updrafts.UpdraftVariable phi_u, EDMF_Updrafts.UpdraftVariable psi_u,
                                EDMF_Environment.EnvironmentVariable phi_e, EDMF_Environment.EnvironmentVariable psi_e,
                                EDMF_Updrafts.UpdraftVariable_2m covar_u,EDMF_Environment.EnvironmentVariable_2m covar_e,
                                double *gmv_phi, double *gmv_psi, double *gmv_covar)
