#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

import numpy as np
import sys
include "parameters.pxi"
import cython
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
        self.QR = EnvironmentVariable( nz, 'half', 'scalar', 'qr','kg/kg' )
        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = EnvironmentVariable( nz, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = EnvironmentVariable( nz, 'half', 'scalar', 'thetal','K' )
        self.THL = EnvironmentVariable(nz, 'half', 'scalar', 'thetal', 'K')
        self.T = EnvironmentVariable( nz, 'half', 'scalar', 'temperature','K' )
        self.B = EnvironmentVariable( nz, 'half', 'scalar', 'buoyancy','m^2/s^3' )
        self.CF = EnvironmentVariable(nz, 'half', 'scalar','cloud_fraction', '-')

        # TKE   TODO   repeated from Variables.pyx logic
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
            self.EnvThermo_scheme = str(namelist['thermodynamics']['saturation'])
        except:
            self.EnvThermo_scheme = 'sa_mean'
            print('Defaulting to saturation adjustment with respect to environmental means')

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

        #TODO  - most likely a temporary solution (unless it could be useful for testing)
        try:
            self.use_prescribed_scalar_var = namelist['turbulence']['sgs']['use_prescribed_scalar_var']
        except:
            self.use_prescribed_scalar_var = False
        if self.use_prescribed_scalar_var == True:
            self.prescribed_QTvar  = namelist['turbulence']['sgs']['prescribed_QTvar']
            self.prescribed_Hvar   = namelist['turbulence']['sgs']['prescribed_Hvar']
            self.prescribed_HQTcov = namelist['turbulence']['sgs']['prescribed_HQTcov']

        if self.EnvThermo_scheme == 'sa_quadrature':
            if (self.calc_scalar_var == False and self.use_prescribed_scalar_var == False ):
                sys.exit('EDMF_Environment.pyx 96: scalar variance has to be specified for Sommeria Deardorff or quadrature saturation')

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('env_w')
        Stats.add_profile('env_qt')
        Stats.add_profile('env_ql')
        Stats.add_profile('env_qr')
        if self.H.name == 's':
            Stats.add_profile('env_s')
        else:
            Stats.add_profile('env_thetal')
        Stats.add_profile('env_temperature')
        if self.calc_tke:
            Stats.add_profile('env_tke')
        if self.calc_scalar_var:
            Stats.add_profile('env_Hvar')
            Stats.add_profile('env_QTvar')
            Stats.add_profile('env_HQTcov')
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('env_w', self.W.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_qt', self.QT.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_ql', self.QL.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('env_qr', self.QR.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.H.name == 's':
            Stats.write_profile('env_s', self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('env_thetal', self.H.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('env_temperature', self.T.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.calc_tke:
            Stats.write_profile('env_tke', self.TKE.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.calc_scalar_var:
            Stats.write_profile('env_Hvar', self.Hvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('env_QTvar', self.QTvar.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('env_HQTcov', self.HQTcov.values[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        #ToDo [suggested by CK for AJ ;]
        # Add output of environmental cloud fraction, cloud base, cloud top (while the latter can be gleaned from ql profiles
        # it is more convenient to simply have them in the stats files!
        # Add the same with respect to the grid mean
        return

cdef class EnvironmentThermodynamics:
    def __init__(self, namelist, paramlist, Grid Gr, ReferenceState Ref, EnvironmentVariables EnvVar):
        self.Gr = Gr
        self.Ref = Ref
        try:
            self.quadrature_order = namelist['condensation']['quadrature_order']
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

        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']

        return

    cdef void update_EnvVar(self, long k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double qr, double alpha) nogil :

        EnvVar.T.values[k]   = T
        EnvVar.THL.values[k] = H
        EnvVar.H.values[k]   = H
        EnvVar.QT.values[k]  = qt
        EnvVar.QL.values[k]  = ql
        EnvVar.QR.values[k] += qr
        EnvVar.B.values[k]   = buoyancy_c(self.Ref.alpha0_half[k], alpha)
        return

    cdef void update_cloud_dry(self, long k, EnvironmentVariables EnvVar, double T, double th, double qt, double ql, double qv) nogil :

        if ql > 0.0:
            EnvVar.CF.values[k] = 1.
            self.th_cloudy[k]   = th
            self.t_cloudy[k]    = T
            self.qt_cloudy[k]   = qt
            self.qv_cloudy[k]   = qv
        else:
            EnvVar.CF.values[k] = 0.
            self.th_dry[k]      = th
            self.qt_dry[k]      = qt
        return

    cdef void eos_update_SA_mean(self, EnvironmentVariables EnvVar, bint in_Env):

        cdef:
            Py_ssize_t k
            Py_ssize_t gw = self.Gr.gw

            eos_struct sa
            mph_struct mph

        if EnvVar.H.name != 'thetal':
            sys.exit('EDMF_Environment: rain source terms are defined for thetal as model variable')

        with nogil:
            for k in xrange(gw,self.Gr.nzg-gw):
                # condensation + autoconversion
                sa  = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k], EnvVar.QT.values[k], EnvVar.H.values[k])
                mph = microphysics(sa.T, sa.ql, self.Ref.p0_half[k], EnvVar.QT.values[k], self.max_supersaturation, in_Env)

                self.update_EnvVar(   k, EnvVar, mph.T, mph.thl, mph.qt, mph.ql, mph.qr, mph.alpha)
                self.update_cloud_dry(k, EnvVar, mph.T, mph.th,  mph.qt, mph.ql, mph.qv)

        return

    cdef void eos_update_SA_sgs(self, EnvironmentVariables EnvVar, bint in_Env):
        a, w = np.polynomial.hermite.hermgauss(self.quadrature_order)

        #TODO - remember you output source terms multipierd by dt (bec. of instanteneous autoconcv)
        #TODO - read prescribed var/covar from file to compare with LES data
        #TODO - add tendencies for GMV H, QT and QR due to rain

        cdef:
            Py_ssize_t gw = self.Gr.gw
            Py_ssize_t k, m_q, m_h
            double [:] abscissas = a
            double [:] weights = w
            # arrays for storing quadarature points and ints for labeling items in the arrays
            # a python dict would be nicer, but its 30% slower than this (for python 2.7. It might not be the case for python 3)
            double[:] inner_env, outer_env, inner_src, outer_src
            int i_ql, i_T, i_thl, i_alpha, i_cf, i_qr, i_qt_cld, i_qt_dry, i_T_cld, i_T_dry
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

        # for testing (to be removed)
        if EnvVar.use_prescribed_scalar_var:
            for k in xrange(gw, self.Gr.nzg-gw):
                if k * self.Gr.dz <= 1500:
                    EnvVar.QTvar.values[k]  = EnvVar.prescribed_QTvar
                else:
                    EnvVar.QTvar.values[k]  = 0.
                if k * self.Gr.dz <= 1500 and k * self.Gr.dz > 500:
                    EnvVar.Hvar.values[k]   = EnvVar.prescribed_Hvar
                else:
                    EnvVar.Hvar.values[k]   = 0.
                if k * self.Gr.dz <= 1500 and k * self.Gr.dz > 200:
                    EnvVar.HQTcov.values[k] = EnvVar.prescribed_HQTcov
                else:
                    EnvVar.HQTcov.values[k] = 0.

        # initialize the quadrature points and their labels
        inner_env = np.zeros(env_len, dtype=np.double, order='c')
        outer_env = np.zeros(env_len, dtype=np.double, order='c')
        inner_src = np.zeros(src_len, dtype=np.double, order='c')
        outer_src = np.zeros(src_len, dtype=np.double, order='c')
        i_ql, i_T, i_thl, i_alpha, i_cf, i_qr, i_qt_cld, i_qt_dry, i_T_cld, i_T_dry = range(env_len)
        i_SH_qt, i_Sqt_H, i_SH_H, i_Sqt_qt, i_Sqt, i_SH = range(src_len)

        with nogil:
            for k in xrange(gw, self.Gr.nzg-gw):
                if EnvVar.QTvar.values[k] != 0.0 and EnvVar.Hvar.values[k] != 0.0 and EnvVar.HQTcov.values[k] != 0.0:
                    sd_q = sqrt(EnvVar.QTvar.values[k])
                    sd_h = sqrt(EnvVar.Hvar.values[k])
                    corr = fmax(fmin(EnvVar.HQTcov.values[k]/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

                    # limit sd_q to prevent negative qt_hat
                    sd_q_lim = (1e-10 - EnvVar.QT.values[k])/(sqrt2 * abscissas[0])
                    sd_q = fmin(sd_q, sd_q_lim)
                    qt_var = sd_q * sd_q
                    sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h

                    # zero outer quadrature points
                    for idx in range(env_len):
                        outer_env[idx] = 0.0
                    if in_Env:
                        for idx in range(src_len):
                            outer_src[idx] = 0.0

                    for m_q in xrange(self.quadrature_order):
                        qt_hat    = EnvVar.QT.values[k] + sqrt2 * sd_q * abscissas[m_q]
                        mu_h_star = EnvVar.H.values[k]  + sqrt2 * corr * sd_h * abscissas[m_q]

                        # zero inner quadrature points
                        for idx in range(env_len):
                            inner_env[idx] = 0.0
                        if in_Env:
                            for idx in range(src_len):
                                inner_src[idx] = 0.0

                        for m_h in xrange(self.quadrature_order):
                            h_hat = sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star

                            # condensation + autoconversion
                            sa  = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k], qt_hat, h_hat)
                            mph = microphysics(sa.T, sa.ql, self.Ref.p0_half[k], qt_hat, self.max_supersaturation, in_Env)

                            # environmental variables
                            inner_env[i_ql]    += mph.ql    * weights[m_h] * sqpi_inv
                            inner_env[i_qr]    += mph.qr    * weights[m_h] * sqpi_inv
                            inner_env[i_T]     += mph.T     * weights[m_h] * sqpi_inv
                            inner_env[i_thl]   += mph.thl   * weights[m_h] * sqpi_inv
                            inner_env[i_alpha] += mph.alpha * weights[m_h] * sqpi_inv
                            # cloudy/dry categories for buoyancy in TKE
                            if mph.ql  > 0.0:
                                inner_env[i_cf]     +=          weights[m_h] * sqpi_inv
                                inner_env[i_qt_cld] += mph.qt * weights[m_h] * sqpi_inv
                                inner_env[i_T_cld]  += mph.T  * weights[m_h] * sqpi_inv
                            else:
                                inner_env[i_qt_dry] += mph.qt * weights[m_h] * sqpi_inv
                                inner_env[i_T_dry]  += mph.T  * weights[m_h] * sqpi_inv
                            # products for variance and covariance source terms
                            if in_Env:
                                inner_src[i_Sqt]    += -mph.qr                     * weights[m_h] * sqpi_inv
                                inner_src[i_SH]     +=  mph.thl_rain_src           * weights[m_h] * sqpi_inv
                                inner_src[i_Sqt_H]  += -mph.qr           * mph.thl * weights[m_h] * sqpi_inv
                                inner_src[i_Sqt_qt] += -mph.qr           * mph.qt  * weights[m_h] * sqpi_inv
                                inner_src[i_SH_H]   +=  mph.thl_rain_src * mph.thl * weights[m_h] * sqpi_inv
                                inner_src[i_SH_qt]  +=  mph.thl_rain_src * mph.qt  * weights[m_h] * sqpi_inv

                        for idx in range(env_len):
                            outer_env[idx] += inner_env[idx] * weights[m_q] * sqpi_inv
                        if in_Env:
                            for idx in range(src_len):
                                outer_src[idx] += inner_src[idx] * weights[m_q] * sqpi_inv

                    # update environmental variables
                    self.update_EnvVar(k, EnvVar, outer_env[i_T], outer_env[i_thl],\
                                       outer_env[i_qt_cld]+outer_env[i_qt_dry], outer_env[i_ql],\
                                       outer_env[i_qr], outer_env[i_alpha])
                    # update cloudy/dry variables for buoyancy in TKE
                    EnvVar.CF.values[k]  = outer_env[i_cf]
                    self.qt_dry[k]    = outer_env[i_qt_dry]
                    self.th_dry[k]    = theta_c(self.Ref.p0_half[k], outer_env[i_T_dry])
                    self.t_cloudy[k]  = outer_env[i_T_cld]
                    self.qv_cloudy[k] = outer_env[i_qt_cld] - outer_env[i_ql]
                    self.qt_cloudy[k] = outer_env[i_qt_cld]
                    self.th_cloudy[k] = theta_c(self.Ref.p0_half[k], outer_env[i_T_cld])
                    # update var/covar rain sources
                    if in_Env:
                        self.Hvar_rain_dt[k]   = outer_src[i_SH_H]   - outer_src[i_SH]  * EnvVar.H.values[k]
                        self.QTvar_rain_dt[k]  = outer_src[i_Sqt_qt] - outer_src[i_Sqt] * EnvVar.QT.values[k]
                        self.HQTcov_rain_dt[k] = outer_src[i_SH_qt]  - outer_src[i_SH]  * EnvVar.QT.values[k] + \
                                                 outer_src[i_Sqt_H]  - outer_src[i_Sqt] * EnvVar.H.values[k]

                else:
                    # the same as in SA_mean
                    sa  = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k], EnvVar.QT.values[k], EnvVar.H.values[k])
                    mph = microphysics(sa.T, sa.ql, self.Ref.p0_half[k], EnvVar.QT.values[k], self.max_supersaturation, in_Env)

                    self.update_EnvVar(   k, EnvVar, mph.T, mph.thl, mph.qt, mph.ql, mph.qr, mph.alpha)
                    self.update_cloud_dry(k, EnvVar, mph.T, mph.th,  mph.qt, mph.ql, mph.qv)

                    if in_Env:
                        self.Hvar_rain_dt[k]   = 0.
                        self.QTvar_rain_dt[k]  = 0.
                        self.HQTcov_rain_dt[k] = 0.

        return

    cpdef satadjust(self, EnvironmentVariables EnvVar, bint in_Env):

        if EnvVar.EnvThermo_scheme == 'sa_mean':
            self.eos_update_SA_mean(EnvVar, in_Env)
        elif EnvVar.EnvThermo_scheme == 'sa_quadrature':
            self.eos_update_SA_sgs(EnvVar, in_Env)
        else:
            sys.exit('EDMF_Environment: Unrecognized EnvThermo_scheme. Possible options: sa_mean, sa_quadrature')

        return
