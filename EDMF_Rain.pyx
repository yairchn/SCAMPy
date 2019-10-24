#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import cython
import numpy as np
import sys

cimport Grid
cimport ReferenceState
from Variables cimport GridMeanVariables
from EDMF_Environment cimport EnvironmentThermodynamics
from EDMF_Updrafts cimport UpdraftThermodynamics
from microphysics_functions cimport  *

include "parameters.pxi"

cdef class RainVariable:
    def __init__(self, nz, name, units):

        self.loc   = 'half'
        self.kind  = 'scalar'
        self.name  = name
        self.units = units

        self.values      = np.zeros((nz,), dtype=np.double, order='c')
        self.new         = np.zeros((nz,), dtype=np.double, order='c')
        self.flux        = np.zeros((nz,), dtype=np.double, order='c')

    cpdef set_bcs(self, Grid.Grid Gr):
        cdef:
            Py_ssize_t k

        for k in xrange(Gr.gw):
            self.values[Gr.nzg - Gr.gw + k] = self.values[Gr.nzg-Gr.gw - 1 - k]
            self.values[Gr.gw - 1 - k]      = self.values[Gr.gw + k]

        return

cdef class RainVariables:
    def __init__(self, namelist, Grid.Grid Gr):
        self.Gr = Gr
        cdef:
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t k

        self.QR           = RainVariable(nzg, 'qr',            'kg/kg')
        # temporary variables for diagnostics to know where the rain is coming from
        self.Upd_QR       = RainVariable(nzg, 'upd_qr',        'kg/kg')
        self.Env_QR       = RainVariable(nzg, 'env_qr',        'kg/kg')
        # in the future we could test prognostic equations for stratiform and updraft rain
        self.RainArea     = RainVariable(nzg, 'rain_area',     'rain_area_fraction [-]' )
        self.Upd_RainArea = RainVariable(nzg, 'upd_rain_area', 'updraft_rain_area_fraction [-]' )
        self.Env_RainArea = RainVariable(nzg, 'env_rain_area', 'environment_rain_area_fraction [-]' )

        self.mean_rwp = 0.
        self.upd_rwp = 0.
        self.env_rwp = 0.

        try:
            self.rain_model = str(namelist['microphysics']['rain_model'])
        except:
            print "EDMF_Rain: defaulting to no rain"
            self.rain_model = "None"

        if self.rain_model not in ["None", "cutoff", "clima_1m"]:
            sys.exit('rain model not recognized')

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('qr')
        Stats.add_profile('updraft_qr')
        Stats.add_profile('env_qr')
        Stats.add_profile('rain_area')
        Stats.add_profile('updraft_rain_area')
        Stats.add_profile('env_rain_area')
        Stats.add_ts('rwp_mean')
        Stats.add_ts('updraft_rwp')
        Stats.add_ts('env_rwp')
        return

    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState.ReferenceState Ref):
        Stats.write_profile('qr',                self.QR.values[self.Gr.gw           : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('updraft_qr',        self.Upd_QR.values[self.Gr.gw       : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('env_qr',            self.Env_QR.values[self.Gr.gw       : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('rain_area',         self.RainArea.values[self.Gr.gw     : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('updraft_rain_area', self.Upd_RainArea.values[self.Gr.gw : self.Gr.nzg - self.Gr.gw])
        Stats.write_profile('env_rain_area',     self.Env_RainArea.values[self.Gr.gw : self.Gr.nzg - self.Gr.gw])

        self.rain_diagnostics(Ref)
        Stats.write_ts('rwp_mean', self.mean_rwp)
        Stats.write_ts('updraft_rwp', self.upd_rwp)
        Stats.write_ts('env_rwp', self.env_rwp)
        return

    cpdef rain_diagnostics(self, ReferenceState.ReferenceState Ref):
        cdef Py_ssize_t k
        self.upd_rwp  = 0.
        self.env_rwp  = 0.
        self.mean_rwp = 0.

        for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            self.upd_rwp  += Ref.rho0_half[k] * self.Upd_QR.values[k] * self.Upd_RainArea.values[k] * self.Gr.dz
            self.env_rwp  += Ref.rho0_half[k] * self.Env_QR.values[k] * self.Env_RainArea.values[k] * self.Gr.dz
            self.mean_rwp += Ref.rho0_half[k] * self.QR.values[k]     * self.RainArea.values[k]     * self.Gr.dz
        return

    cpdef sum_subdomains_rain(self, UpdraftThermodynamics UpdThermo, EnvironmentThermodynamics EnvThermo):
        with nogil:
            for k in xrange(self.Gr.nzg):

                self.QR.values[k]       -= (EnvThermo.prec_source_qt[k] + UpdThermo.prec_source_qt_tot[k])
                self.Upd_QR.values[k]   -= UpdThermo.prec_source_qt_tot[k]
                self.Env_QR.values[k]   -= EnvThermo.prec_source_qt[k]

                # TODO Assuming that updraft and environment rain area fractions are either 1 or 0.
                if self.QR.values[k] > 0.:
                    self.RainArea.values[k] = 1
                if self.Upd_QR.values[k] > 0.:
                    self.Upd_RainArea.values[k] = 1
                if self.Env_QR.values[k] > 0.:
                    self.Env_RainArea.values[k] = 1
        return

cdef class RainPhysics:
    def __init__(self, Grid.Grid Gr, ReferenceState.ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref

        self.rain_evap_source_h  = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.rain_evap_source_qt = np.zeros((Gr.nzg,), dtype=np.double, order='c')

        return

    cpdef solve_rain_fall(
        self,
        GridMeanVariables GMV,
        TimeStepping TS,
        RainVariable QR,
        RainVariable RainArea
    ):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw  = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg

            double dz = self.Gr.dz
            double dt_model = TS.dt

            double CFL_out, CFL_in
            double CFL_limit = 0.5
            double rho_frac, area_frac

            double [:] term_vel     = np.zeros((nzg,), dtype=np.double, order='c')
            double [:] term_vel_new = np.zeros((nzg,), dtype=np.double, order='c')

            double dt_rain
            double t_elapsed = 0.

        # helper to calculate the rain velocity
        # TODO: assuming GMV.W = 0
        for k in xrange(nzg - gw - 1, gw - 1, -1):
            term_vel[k] = terminal_velocity(
                              QR.values[k],
                              self.Ref.rho0_half[k]
                           )

        # calculate the allowed timestep (CFL_limit >= v dt / dz)
        if max(term_vel[:]) != 0.:
            dt_rain = np.minimum(dt_model, CFL_limit * self.Gr.dz / max(term_vel[:]))

        # rain falling through the domain
        while t_elapsed < dt_model:
            for k in xrange(nzg - gw - 1, gw - 1, -1):

                CFL_out = dt_rain / dz * term_vel[k]

                if k == (nzg - gw - 1):
                    CFL_in = 0.
                else:
                    CFL_in = dt_rain / dz * term_vel[k+1]

                rho_frac  = self.Ref.rho0_half[k+1] / self.Ref.rho0_half[k]
                area_frac = 1. # RainArea.values[k] / RainArea.new[k]

                QR.new[k] = (QR.values[k]   * (1 - CFL_out) +\
                             QR.values[k+1] * CFL_in * rho_frac) * area_frac
                if QR.new[k] != 0.:
                    RainArea.new[k] = 1.

                term_vel_new[k] = terminal_velocity(
                                      QR.new[k],
                                      self.Ref.rho0_half[k]
                                  )

            t_elapsed += dt_rain

            QR.values[:] = QR.new[:]
            RainArea.values[:] = RainArea.new[:]

            term_vel[:] = term_vel_new[:]

            if np.max(np.abs(term_vel[:])) > np.finfo(float).eps:
                dt_rain = np.minimum(dt_model - t_elapsed,
                                     CFL_limit * self.Gr.dz / max(term_vel[:])
                                    )
            else:
                dt_rain = dt_model - t_elapsed

        return

    cpdef solve_rain_evap(
        self,
        GridMeanVariables GMV,
        TimeStepping TS,
        RainVariable QR,
        RainVariable RainArea
    ):
        cdef:
            Py_ssize_t k
            Py_ssize_t gw  = self.Gr.gw
            Py_ssize_t nzg = self.Gr.nzg

            double dz = self.Gr.dz
            double dt_model = TS.dt

            double tmp_evap

            bint flag_evaporate_all = False

        for k in xrange(gw, nzg - gw):

            flag_evaporate_all = False

            tmp_evap = max(0, conv_q_rai_to_q_vap(QR.values[k],
                                                  GMV.QT.values[k],
                                                  GMV.QL.values[k],
                                                  GMV.T.values[k],
                                                  self.Ref.p0_half[k],
                                                  self.Ref.rho0[k]
                                                ) * dt_model)

            if tmp_evap > QR.values[k]:
                flag_evaporate_all = True
                tmp_evap = QR.values[k]

            self.rain_evap_source_qt[k] = tmp_evap * RainArea.values[k]

            self.rain_evap_source_h[k]  = rain_source_to_thetal(
                self.Ref.p0[k],
                GMV.T.values[k],
                - tmp_evap
            ) * RainArea.values[k]

            if flag_evaporate_all:
                QR.values[k] = 0.
                RainArea.values[k] = 0.
            else:
                # TODO: assuming that rain evaporation doesn't change
                # rain area fraction
                # (should be changed for prognostic rain area fractions)
                QR.values[k] -= tmp_evap
        return
