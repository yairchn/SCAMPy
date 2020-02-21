#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
import sys
import cython

include "parameters.pxi"

from EDMF_Rain cimport RainVariables
from Grid cimport  Grid
from TimeStepping cimport TimeStepping
from ReferenceState cimport ReferenceState
from Variables cimport VariableDiagnostic, GridMeanVariables
from libc.math cimport fmax, fmin, sqrt, exp, erf
from thermodynamic_functions cimport  *
from microphysics_functions cimport *

cdef class EnvironmentVariable:
    def __init__(self, nz, loc, kind, name, units):
        self.values = np.zeros((nz,),dtype=np.double, order='c')
        self.flux = np.zeros((nz,),dtype=np.double, order='c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    cpdef set_bcs(self,Grid Gr):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        if self.name == 'w':
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0
            for k in xrange(1,Gr.gw):
                self.values[start_high+ k] = -self.values[start_high - k ]
                self.values[start_low- k] = -self.values[start_low + k  ]
        else:
            for k in xrange(Gr.gw):
                self.values[start_high + k +1] = self.values[start_high  - k]
                self.values[start_low - k] = self.values[start_low + 1 + k]

cdef class EnvironmentVariable_2m:
    def __init__(self, nz, loc, kind, name, units):
        self.values = np.zeros((nz,),dtype=np.double, order='c')
        self.dissipation = np.zeros((nz,),dtype=np.double, order='c')
        self.entr_gain = np.zeros((nz,),dtype=np.double, order='c')
        self.detr_loss = np.zeros((nz,),dtype=np.double, order='c')
        self.buoy = np.zeros((nz,),dtype=np.double, order='c')
        self.press = np.zeros((nz,),dtype=np.double, order='c')
        self.shear = np.zeros((nz,),dtype=np.double, order='c')
        self.interdomain = np.zeros((nz,),dtype=np.double, order='c')
        self.rain_src = np.zeros((nz,),dtype=np.double, order='c')
        if loc != 'half':
            print('Invalid location setting for variable! Must be half')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    cpdef set_bcs(self,Grid Gr):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        for k in xrange(Gr.gw):
            self.values[start_high + k +1] = self.values[start_high  - k]
            self.values[start_low - k] = self.values[start_low + 1 + k]

cdef class EnvironmentVariables:
    def __init__(self,  namelist, Grid Gr  ):
        cdef Py_ssize_t nz = Gr.nzg
        self.Gr = Gr

        self.W = EnvironmentVariable(nz, 'full', 'velocity', 'w','m/s' )
        self.QT = EnvironmentVariable( nz, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = EnvironmentVariable( nz, 'half', 'scalar', 'ql','kg/kg' )
        self.RH = EnvironmentVariable( nz, 'half', 'scalar', 'RH','%' )

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = EnvironmentVariable( nz, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = EnvironmentVariable( nz, 'half', 'scalar', 'thetal','K' )

        self.THL = EnvironmentVariable(nz, 'half', 'scalar', 'thetal', 'K')
        self.T = EnvironmentVariable( nz, 'half', 'scalar', 'temperature','K' )
        self.B = EnvironmentVariable( nz, 'half', 'scalar', 'buoyancy','m^2/s^3' )
        self.Area = EnvironmentVariable(nz, 'half', 'scalar', 'env_area', '-')
        self.cloud_fraction = EnvironmentVariable(nz, 'half', 'scalar', 'env_cloud_fraction', '-')

        # TODO - the flag setting is repeated from Variables.pyx logic
        if  namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            self.calc_tke = True
        else:
            self.calc_tke = False
        try:
            self.calc_tke = namelist['turbulence']['EDMF_PrognosticTKE']['calculate_tke']
        except:
            pass

        try:
            self.calc_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']
        except:
            self.calc_scalar_var = False
            print('Defaulting to non-calculation of scalar variances')

        try:
            self.EnvThermo_scheme = str(namelist['thermodynamics']['sgs'])
        except:
            self.EnvThermo_scheme = 'mean'
            print('Defaulting to saturation adjustment and microphysics with respect to environmental means')

        if self.calc_tke:
            self.TKE = EnvironmentVariable_2m( nz, 'half', 'scalar', 'tke','m^2/s^2' )

        if self.calc_scalar_var:
            self.QTvar = EnvironmentVariable_2m( nz, 'half', 'scalar', 'qt_var','kg^2/kg^2' )
            if namelist['thermodynamics']['thermal_variable'] == 'entropy':
                self.Hvar = EnvironmentVariable_2m(nz, 'half', 'scalar', 's_var', '(J/kg/K)^2')
                self.HQTcov = EnvironmentVariable_2m(nz, 'half', 'scalar', 's_qt_covar', '(J/kg/K)(kg/kg)' )
            elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
                self.Hvar = EnvironmentVariable_2m(nz, 'half', 'scalar', 'thetal_var', 'K^2')
                self.HQTcov = EnvironmentVariable_2m(nz, 'half', 'scalar', 'thetal_qt_covar', 'K(kg/kg)' )

        if self.EnvThermo_scheme == 'quadrature':
            if (self.calc_scalar_var == False):
                sys.exit('EDMF_Environment.pyx: scalar variance has to be calculated for quadrature saturation and microphysics')

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('env_w')
        Stats.add_profile('env_qt')
        Stats.add_profile('env_ql')
        Stats.add_profile('env_area')
        Stats.add_profile('env_temperature')
        Stats.add_profile('env_RH')

        if self.H.name == 's':
            Stats.add_profile('env_s')
        else:
            Stats.add_profile('env_thetal')

        if self.calc_tke:
            Stats.add_profile('env_tke')
        if self.calc_scalar_var:
            Stats.add_profile('env_Hvar')
            Stats.add_profile('env_QTvar')
            Stats.add_profile('env_HQTcov')

        Stats.add_profile('env_cloud_fraction')

        Stats.add_ts('env_cloud_base')
        Stats.add_ts('env_cloud_top')
        Stats.add_ts('env_cloud_cover')
        Stats.add_ts('env_lwp')

        return

    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState Ref):
        Stats.write_profile('env_w', self.W.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_qt', self.QT.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_ql', self.QL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_area', self.Area.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_temperature', self.T.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_RH', self.RH.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        if self.H.name == 's':
            Stats.write_profile('env_s', self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('env_thetal', self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        if self.calc_tke:
            Stats.write_profile('env_tke', self.TKE.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.calc_scalar_var:
            Stats.write_profile('env_Hvar', self.Hvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('env_QTvar', self.QTvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('env_HQTcov', self.HQTcov.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('env_cloud_fraction', self.cloud_fraction.values[self.Gr.gw : self.Gr.nzg-self.Gr.gw])

        self.env_cloud_diagnostics(Ref)
        # Assuming amximum overlap in environmental clouds
        Stats.write_ts('env_cloud_cover', self.cloud_cover)
        Stats.write_ts('env_cloud_base',  self.cloud_base)
        Stats.write_ts('env_cloud_top',   self.cloud_top)
        Stats.write_ts('env_lwp',         self.lwp)
        return

    cpdef env_cloud_diagnostics(self, ReferenceState Ref):
        cdef Py_ssize_t k
        self.cloud_top   = 0.
        self.cloud_base  = self.Gr.z_half[self.Gr.nzg - self.Gr.gw - 1]
        self.cloud_cover = 0.
        self.lwp         = 0.

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            self.lwp += Ref.rho0_half[k] * self.QL.values[k] * self.Area.values[k] * self.Gr.dz

            if self.QL.values[k] > 1e-8 and self.Area.values[k] > 1e-3:
                self.cloud_base  = fmin(self.cloud_base,  self.Gr.z_half[k])
                self.cloud_top   = fmax(self.cloud_top,   self.Gr.z_half[k])
                self.cloud_cover = fmax(self.cloud_cover, self.Area.values[k] * self.cloud_fraction.values[k])
        return

cdef class EnvironmentThermodynamics:
    def __init__(self, namelist, Grid Gr, ReferenceState Ref, EnvironmentVariables EnvVar, RainVariables Rain):
        self.Gr = Gr
        self.Ref = Ref
        try:
            self.quadrature_order = namelist['thermodynamics']['quadrature_order']
        except:
            self.quadrature_order = 5
        if EnvVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif EnvVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        self.qt_dry = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.th_dry = np.zeros(self.Gr.nzg, dtype=np.double, order='c')

        self.t_cloudy  = np.zeros(self.Gr.nzg, dtype=np.double, order ='c')
        self.qv_cloudy = np.zeros(self.Gr.nzg, dtype=np.double, order ='c')
        self.qt_cloudy = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.th_cloudy = np.zeros(self.Gr.nzg, dtype=np.double, order='c')

        self.Hvar_rain_dt   = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.QTvar_rain_dt  = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.HQTcov_rain_dt = np.zeros(self.Gr.nzg, dtype=np.double, order='c')

        self.prec_source_qt = np.zeros(self.Gr.nzg, dtype=np.double, order='c')
        self.prec_source_h  = np.zeros(self.Gr.nzg, dtype=np.double, order='c')

        return

    cdef void update_EnvVar(self, Py_ssize_t k, EnvironmentVariables EnvVar,
                            double T, double H, double qt, double ql,
                            double alpha) nogil :
        EnvVar.T.values[k]   = T
        EnvVar.THL.values[k] = H
        EnvVar.H.values[k]   = H
        EnvVar.QT.values[k]  = qt
        EnvVar.QL.values[k]  = ql
        EnvVar.B.values[k]   = buoyancy_c(self.Ref.alpha0_half[k], alpha)
        EnvVar.RH.values[k] = relative_humidity_c(self.Ref.p0_half[k], qt , ql , 0.0, T)
        return

    cdef void update_EnvRain_sources(self, Py_ssize_t k, EnvironmentVariables EnvVar,
                                     double qr_src, double thl_rain_src) nogil:

        self.prec_source_qt[k] = -qr_src * EnvVar.Area.values[k]
        self.prec_source_h[k]  = thl_rain_src * EnvVar.Area.values[k]

        return

    cdef void update_cloud_dry(self, Py_ssize_t k, EnvironmentVariables EnvVar,
                               double T, double th, double qt, double ql,
                               double qv) nogil :
        if ql > 0.0:
            EnvVar.cloud_fraction.values[k] = 1.0
            self.th_cloudy[k]   = th
            self.t_cloudy[k]    = T
            self.qt_cloudy[k]   = qt
            self.qv_cloudy[k]   = qv
        else:
            EnvVar.cloud_fraction.values[k] = 0.
            self.th_dry[k]      = th
            self.qt_dry[k]      = qt
        return

    cdef void saturation_adjustment(self, EnvironmentVariables EnvVar):

        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            eos_struct sa
            mph_struct mph
            double alpha

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                sa  = eos(self.t_to_prog_fp, self.prog_to_t_fp,
                          self.Ref.p0_half[k], EnvVar.QT.values[k],
                          EnvVar.H.values[k]
                         )

                EnvVar.T.values[k]   = sa.T
                EnvVar.QL.values[k]  = sa.ql
                alpha = alpha_c(self.Ref.p0_half[k], EnvVar.T.values[k],
                                EnvVar.QT.values[k],
                                EnvVar.QT.values[k] - EnvVar.QL.values[k]
                               )
                EnvVar.B.values[k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)

                self.update_cloud_dry(k, EnvVar,
                                      EnvVar.T.values[k], EnvVar.THL.values[k],
                                      EnvVar.QT.values[k], EnvVar.QL.values[k],
                                      EnvVar.QT.values[k] - EnvVar.QL.values[k]
                                     )
        return


    cdef void sgs_mean(self, EnvironmentVariables EnvVar, RainVariables Rain, double dt):

        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw
            eos_struct sa
            mph_struct mph

        if EnvVar.H.name != 'thetal':
            sys.exit('EDMF_Environment: rain source terms are defined for thetal as model variable')

        with nogil:
            for k in xrange(gw,self.Gr.nzg-gw):
                # condensation
                sa  = eos(
                    self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k],
                    EnvVar.QT.values[k], EnvVar.H.values[k]
                )
                # autoconversion and accretion
                mph = microphysics_rain_src(
                    Rain.rain_model,
                    EnvVar.QT.values[k],
                    sa.ql,
                    Rain.Env_QR.values[k],
                    EnvVar.Area.values[k],
                    sa.T,
                    self.Ref.p0_half[k],
                    self.Ref.rho0_half[k],
                    dt
                )

                self.update_EnvVar(k, EnvVar, sa.T, mph.thl, mph.qt, mph.ql, mph.alpha)
                self.update_cloud_dry(k, EnvVar, sa.T, mph.th,  mph.qt, mph.ql, mph.qv)
                self.update_EnvRain_sources(k, EnvVar, mph.qr_src, mph.thl_rain_src)
        return

    cdef void sgs_quadrature(self, EnvironmentVariables EnvVar, RainVariables Rain, double dt):
        a, w = np.polynomial.hermite.hermgauss(self.quadrature_order)

        #TODO - remember you output source terms multipierd by dt (bec. of instanteneous autoconcv)
        #TODO - add tendencies for GMV H, QT and QR due to rain
        #TODO - if we start using eos_smpl for the updrafts calculations
        #       we can get rid of the two categories for outer and inner quad. points

        cdef:
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t k, m_q, m_h
            double [:] abscissas = a
            double [:] weights = w
            # arrays for storing quadarature points and ints for labeling items in the arrays
            # a python dict would be nicer, but its 30% slower than this (for python 2.7. It might not be the case for python 3)
            double[:] inner_env, outer_env, inner_src, outer_src
            int i_ql, i_T, i_thl, i_alpha, i_cf, i_qt_cld, i_qt_dry, i_T_cld, i_T_dry, i_rf
            int i_SH_qt, i_Sqt_H, i_SH_H, i_Sqt_qt, i_Sqt, i_SH
            int env_len = 10
            int src_len = 6

            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim
            eos_struct sa
            mph_struct mph

        if EnvVar.H.name != 'thetal':
            sys.exit('EDMF_Environment: rain source terms are only defined for thetal as model variable')

        # initialize the quadrature points and their labels
        inner_env = np.zeros(env_len, dtype=np.double, order='c')
        outer_env = np.zeros(env_len, dtype=np.double, order='c')
        inner_src = np.zeros(src_len, dtype=np.double, order='c')
        outer_src = np.zeros(src_len, dtype=np.double, order='c')
        i_ql, i_T, i_thl, i_alpha, i_cf, i_qt_cld, i_qt_dry, i_T_cld, i_T_dry, i_rf = range(env_len)
        i_SH_qt, i_Sqt_H, i_SH_H, i_Sqt_qt, i_Sqt, i_SH = range(src_len)

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                if EnvVar.QTvar.values[k] != 0.0 and EnvVar.Hvar.values[k] != 0.0 and EnvVar.HQTcov.values[k] != 0.0:
                    sd_q = sqrt(EnvVar.QTvar.values[k])
                    sd_h = sqrt(EnvVar.Hvar.values[k])
                    corr = fmax(fmin(EnvVar.HQTcov.values[k]/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

                    # limit sd_q to prevent negative qt_hat
                    sd_q_lim = (1e-10 - EnvVar.QT.values[k])/(sqrt2 * abscissas[0])
                    # walking backwards to assure your q_t will not be smaller than 1e-10
                    # TODO - check
                    # TODO - change 1e-13 and 1e-10 to some epislon
                    sd_q = fmin(sd_q, sd_q_lim)
                    qt_var = sd_q * sd_q
                    sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h

                    # zero outer quadrature points
                    for idx in range(env_len):
                        outer_env[idx] = 0.0
                    for idx in range(src_len):
                        outer_src[idx] = 0.0

                    for m_q in xrange(self.quadrature_order):
                        qt_hat    = EnvVar.QT.values[k] + sqrt2 * sd_q * abscissas[m_q]
                        mu_h_star = EnvVar.H.values[k]  + sqrt2 * corr * sd_h * abscissas[m_q]

                        # zero inner quadrature points
                        for idx in range(env_len):
                            inner_env[idx] = 0.0
                        for idx in range(src_len):
                            inner_src[idx] = 0.0

                        for m_h in xrange(self.quadrature_order):
                            h_hat = sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star

                            # condensation
                            sa  = eos(
                                self.t_to_prog_fp, self.prog_to_t_fp,
                                self.Ref.p0_half[k], qt_hat, h_hat
                            )
                            # autoconversion and accretiom
                            mph = microphysics_rain_src(
                                Rain.rain_model,
                                qt_hat,
                                sa.ql,
                                Rain.Env_QR.values[k],
                                EnvVar.Area.values[k],
                                sa.T,
                                self.Ref.p0_half[k],
                                self.Ref.rho0_half[k],
                                dt
                            )

                            # environmental variables
                            inner_env[i_ql]     += mph.ql     * weights[m_h] * sqpi_inv
                            inner_env[i_T]      += sa.T       * weights[m_h] * sqpi_inv
                            inner_env[i_thl]    += mph.thl    * weights[m_h] * sqpi_inv
                            inner_env[i_alpha]  += mph.alpha  * weights[m_h] * sqpi_inv
                            # rain area fraction
                            if mph.qr_src > 0.0:
                                inner_env[i_rf]     += weights[m_h] * sqpi_inv
                            # cloudy/dry categories for buoyancy in TKE
                            if mph.ql  > 0.0:
                                inner_env[i_cf]     +=          weights[m_h] * sqpi_inv
                                inner_env[i_qt_cld] += mph.qt * weights[m_h] * sqpi_inv
                                inner_env[i_T_cld]  += sa.T   * weights[m_h] * sqpi_inv
                            else:
                                inner_env[i_qt_dry] += mph.qt * weights[m_h] * sqpi_inv
                                inner_env[i_T_dry]  += sa.T   * weights[m_h] * sqpi_inv
                            # products for variance and covariance source terms
                            inner_src[i_Sqt]    += -mph.qr_src                 * weights[m_h] * sqpi_inv
                            inner_src[i_SH]     +=  mph.thl_rain_src           * weights[m_h] * sqpi_inv
                            inner_src[i_Sqt_H]  += -mph.qr_src       * mph.thl * weights[m_h] * sqpi_inv
                            inner_src[i_Sqt_qt] += -mph.qr_src       * mph.qt  * weights[m_h] * sqpi_inv
                            inner_src[i_SH_H]   +=  mph.thl_rain_src * mph.thl * weights[m_h] * sqpi_inv
                            inner_src[i_SH_qt]  +=  mph.thl_rain_src * mph.qt  * weights[m_h] * sqpi_inv

                        for idx in range(env_len):
                            outer_env[idx] += inner_env[idx] * weights[m_q] * sqpi_inv
                        for idx in range(src_len):
                            outer_src[idx] += inner_src[idx] * weights[m_q] * sqpi_inv

                    # update environmental variables
                    self.update_EnvVar(k, EnvVar, outer_env[i_T], outer_env[i_thl],\
                                       outer_env[i_qt_cld] + outer_env[i_qt_dry],\
                                       outer_env[i_ql], outer_env[i_alpha])
                    self.update_EnvRain_sources(k, EnvVar, -outer_src[i_Sqt], outer_src[i_SH])

                    # update cloudy/dry variables for buoyancy in TKE
                    EnvVar.cloud_fraction.values[k] = outer_env[i_cf]
                    self.qt_dry[k]    = outer_env[i_qt_dry]
                    self.th_dry[k]    = theta_c(self.Ref.p0_half[k], outer_env[i_T_dry])
                    self.t_cloudy[k]  = outer_env[i_T_cld]
                    self.qv_cloudy[k] = outer_env[i_qt_cld] - outer_env[i_ql]
                    self.qt_cloudy[k] = outer_env[i_qt_cld]
                    self.th_cloudy[k] = theta_c(self.Ref.p0_half[k], outer_env[i_T_cld])

                    # update var/covar rain sources
                    self.Hvar_rain_dt[k]   = outer_src[i_SH_H]   - outer_src[i_SH]  * EnvVar.H.values[k]
                    self.QTvar_rain_dt[k]  = outer_src[i_Sqt_qt] - outer_src[i_Sqt] * EnvVar.QT.values[k]
                    self.HQTcov_rain_dt[k] = outer_src[i_SH_qt]  - outer_src[i_SH]  * EnvVar.QT.values[k] + \
                                             outer_src[i_Sqt_H]  - outer_src[i_Sqt] * EnvVar.H.values[k]

                else:
                    # if variance and covaraiance are zero do the same as in SA_mean
                    sa  = eos(
                        self.t_to_prog_fp, self.prog_to_t_fp,
                        self.Ref.p0_half[k], EnvVar.QT.values[k],
                        EnvVar.H.values[k]
                    )
                    mph = microphysics_rain_src(
                        Rain.rain_model,
                        EnvVar.QT.values[k],
                        sa.ql,
                        Rain.Env_QR.values[k],
                        EnvVar.Area.values[k],
                        sa.T,
                        self.Ref.p0_half[k],
                        self.Ref.rho0_half[k],
                        dt
                    )

                    self.update_EnvVar(k, EnvVar, sa.T, mph.thl, mph.qt, mph.ql, mph.alpha)
                    self.update_EnvRain_sources(k, EnvVar, mph.qr_src, mph.thl_rain_src)
                    self.update_cloud_dry(k, EnvVar, sa.T, mph.th,  mph.qt, mph.ql, mph.qv)

                    self.Hvar_rain_dt[k]   = 0.
                    self.QTvar_rain_dt[k]  = 0.
                    self.HQTcov_rain_dt[k] = 0.

        return

    cpdef microphysics(self, EnvironmentVariables EnvVar, RainVariables Rain, double dt):

        if EnvVar.EnvThermo_scheme == 'mean':
            self.sgs_mean(EnvVar, Rain, dt)

        elif EnvVar.EnvThermo_scheme == 'quadrature':
            self.sgs_quadrature(EnvVar, Rain, dt)

        else:
            sys.exit('EDMF_Environment: Unrecognized EnvThermo_scheme. Possible options: mean, quadrature')

        return
