#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True
#cython: cdivision=False

import numpy as np
include "parameters.pxi"
from thermodynamic_functions cimport  *
from microphysics_functions cimport  *
import cython
cimport Grid
cimport ReferenceState
cimport EDMF_Rain
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from libc.math cimport fmax, fmin

cdef class UpdraftVariable:
    def __init__(self, nu, nz, loc, kind, name, units):
        self.values = np.zeros((nu,nz),dtype=np.double, order='c')
        self.old = np.zeros((nu,nz),dtype=np.double, order='c')  # needed for prognostic updrafts
        self.new = np.zeros((nu,nz),dtype=np.double, order='c') # needed for prognostic updrafts
        self.tendencies = np.zeros((nu,nz),dtype=np.double, order='c')
        self.flux = np.zeros((nu,nz),dtype=np.double, order='c')
        self.bulkvalues = np.zeros((nz,), dtype=np.double, order = 'c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    cpdef set_bcs(self,Grid.Grid Gr):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t start_low = Gr.gw - 1
            Py_ssize_t start_high = Gr.nzg - Gr.gw - 1

        n_updrafts = np.shape(self.values)[0]

        if self.name == 'w' or self.name == 'rho_a_w' :
            for i in xrange(n_updrafts):
                self.values[i,start_high] = 0.0
                self.values[i,start_low] = 0.0
                for k in xrange(1,Gr.gw):
                    self.values[i,start_high+ k] = -self.values[i,start_high - k ]
                    self.values[i,start_low- k] = -self.values[i,start_low + k  ]
        else:
            for k in xrange(Gr.gw):
                for i in xrange(n_updrafts):
                    self.values[i,start_high + k +1] = self.values[i,start_high  - k]
                    self.values[i,start_low - k] = self.values[i,start_low + 1 + k]

        return

cdef class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Grid.Grid Gr):
        self.Gr = Gr
        self.n_updrafts = nu
        cdef:
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t i, k

        self.W    = UpdraftVariable(nu, nzg, 'full', 'velocity', 'w','m/s' )
        self.rhoaW   = UpdraftVariable(nu, nzg, 'full', 'velocity', 'rho_a_w','kg/m^2s' )

        self.Area = UpdraftVariable(nu, nzg, 'half', 'scalar', 'area_fraction','[-]' )
        self.QT = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'ql','kg/kg' )
        self.RH = UpdraftVariable(nu, nzg, 'half', 'scalar', 'RH','%' )

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal','K' )

        self.THL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal', 'K')
        self.T   = UpdraftVariable(nu, nzg, 'half', 'scalar', 'temperature','K' )
        self.B   = UpdraftVariable(nu, nzg, 'half', 'scalar', 'buoyancy','m^2/s^3' )
        self.rhoaH  = UpdraftVariable(nu, nzg, 'half', 'scalar', 'a_thetal','kg K/m^3' )
        self.rhoaQT = UpdraftVariable(nu, nzg, 'half', 'scalar', 'a_qt','kg/m^3' )


        if namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            try:
                use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
            except:
                use_steady_updrafts = False
            if use_steady_updrafts:
                self.prognostic = False
            else:
                self.prognostic = True
            self.updraft_fraction = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        else:
            self.prognostic = False
            self.updraft_fraction = paramlist['turbulence']['EDMF_BulkSteady']['surface_area']

        # cloud and rain diagnostics for output
        self.cloud_fraction = np.zeros((nzg,), dtype=np.double, order='c')

        self.cloud_base     = np.zeros((nu,),  dtype=np.double, order='c')
        self.cloud_top      = np.zeros((nu,),  dtype=np.double, order='c')
        self.cloud_cover    = np.zeros((nu,),  dtype=np.double, order='c')
        self.updraft_top    = np.zeros((nu,),  dtype=np.double, order='c')

        self.lwp = 0.
        return

    cpdef initialize(self, GridMeanVariables GMV):
        cdef:
            Py_ssize_t i,k
            Py_ssize_t gw = self.Gr.gw
            double dz = self.Gr.dz

        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):

                    self.W.values[i,k] = 0.0
                    self.rhoaW.values[i,k] = 0.0
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    if self.prognostic:
                        self.Area.values[i,k] = 0.0 #self.updraft_fraction/self.n_updrafts
                    else:
                        self.Area.values[i,k] = self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = GMV.QT.values[k]
                    self.QL.values[i,k] = GMV.QL.values[k]
                    self.H.values[i,k]  = GMV.H.values[k]
                    self.T.values[i,k]  = GMV.T.values[k]
                    self.B.values[i,k]  = 0.0
                    self.rhoaQT.values[i,k] = self.Area.values[i,k]*GMV.QT.values[k]
                    self.rhoaH.values[i,k]  = self.Area.values[i,k]*GMV.H.values[k]

                self.Area.values[i,gw] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_rho_a_w')
        Stats.add_profile('updraft_rho_a_qt')
        Stats.add_profile('updraft_RH')

        if self.H.name == 'thetal':
            Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_rho_a_thetal')
        else:
            # Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_s')
            Stats.add_profile('updraft_rho_a_s')

        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')

        Stats.add_profile('updraft_cloud_fraction')

        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')
        Stats.add_ts('updraft_lwp')
        return

    cpdef set_means(self, GridMeanVariables GMV, ReferenceState.ReferenceState Ref):

        cdef:
            Py_ssize_t i, k

        self.Area.bulkvalues = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0
        self.RH.bulkvalues[:] = 0.0
        self.rhoaQT.bulkvalues[:] = 0
        self.rhoaH.bulkvalues[:] = 0
        self.rhoaW.bulkvalues[:] = 0

        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.Area.bulkvalues[k] > 0.0:
                    for i in xrange(self.n_updrafts):
                        self.QT.bulkvalues[k] += self.Area.values[i,k] * self.QT.values[i,k]/self.Area.bulkvalues[k]
                        self.rhoaQT.bulkvalues[k] += self.rhoaQT.values[i,k]/self.Area.bulkvalues[k]/Ref.rho0_half[k]
                        self.QL.bulkvalues[k] += self.Area.values[i,k] * self.QL.values[i,k]/self.Area.bulkvalues[k]
                        self.H.bulkvalues[k] += self.Area.values[i,k] * self.H.values[i,k]/self.Area.bulkvalues[k]
                        self.rhoaH.bulkvalues[k] += self.rhoaH.values[i,k]/self.Area.bulkvalues[k]/Ref.rho0_half[k]
                        self.T.bulkvalues[k] += self.Area.values[i,k] * self.T.values[i,k]/self.Area.bulkvalues[k]
                        self.RH.bulkvalues[k] += self.Area.values[i,k] * self.RH.values[i,k]/self.Area.bulkvalues[k]
                        self.B.bulkvalues[k] += self.Area.values[i,k] * self.B.values[i,k]/self.Area.bulkvalues[k]
                        self.W.bulkvalues[k] += ((self.Area.values[i,k] + self.Area.values[i,k+1]) * self.W.values[i,k]
                                            /(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1]))
                        self.rhoaW.bulkvalues[k] += (self.rhoaW.values[i,k]/(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1])/Ref.rho0[k])

                else:
                    self.QT.bulkvalues[k] = GMV.QT.values[k]
                    self.rhoaQT.bulkvalues[k] = 0.0
                    self.rhoaH.bulkvalues[k] = 0.0
                    self.QL.bulkvalues[k] = 0.0
                    self.H.bulkvalues[k] = GMV.H.values[k]
                    self.RH.bulkvalues[k] = GMV.RH.values[k]
                    self.T.bulkvalues[k] = GMV.T.values[k]
                    self.B.bulkvalues[k] = 0.0
                    self.W.bulkvalues[k] = 0.0

                if self.QL.bulkvalues[k] > 1e-8 and self.Area.bulkvalues[k] > 0.0:
                    self.cloud_fraction[k] = 1.0
                else:
                    self.cloud_fraction[k] = 0.
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_new_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.new[i,k] = self.W.values[i,k]
                    self.rhoaW.new[i,k] = self.rhoaW.values[i,k]
                    self.Area.new[i,k] = self.Area.values[i,k]
                    self.QT.new[i,k] = self.QT.values[i,k]
                    self.rhoaQT.new[i,k] = self.rhoaQT.values[i,k]
                    self.QL.new[i,k] = self.QL.values[i,k]
                    self.H.new[i,k] = self.H.values[i,k]
                    self.rhoaH.new[i,k] = self.rhoaH.values[i,k]
                    self.THL.new[i,k] = self.THL.values[i,k]
                    self.T.new[i,k] = self.T.values[i,k]
                    self.B.new[i,k] = self.B.values[i,k]
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_old_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.old[i,k] = self.W.values[i,k]
                    self.rhoaW.old[i,k] = self.rhoaW.values[i,k]
                    self.Area.old[i,k] = self.Area.values[i,k]
                    self.QT.old[i,k] = self.QT.values[i,k]
                    self.rhoaQT.old[i,k] = self.rhoaQT.values[i,k]
                    self.QL.old[i,k] = self.QL.values[i,k]
                    self.H.old[i,k] = self.H.values[i,k]
                    self.rhoaH.old[i,k] = self.rhoaH.values[i,k]
                    self.THL.old[i,k] = self.THL.values[i,k]
                    self.T.old[i,k] = self.T.values[i,k]
                    self.B.old[i,k] = self.B.values[i,k]
        return

    # quick utility to set "tmp" arrays with values in the "new" arrays
    cpdef set_values_with_new(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.values[i,k] = self.W.new[i,k]
                    self.rhoaW.values[i,k] = self.rhoaW.new[i,k]
                    self.Area.values[i,k] = self.Area.new[i,k]
                    self.QT.values[i,k] = self.QT.new[i,k]
                    self.rhoaQT.values[i,k] = self.rhoaQT.new[i,k]
                    self.QL.values[i,k] = self.QL.new[i,k]
                    self.H.values[i,k] = self.H.new[i,k]
                    self.rhoaH.values[i,k] = self.rhoaH.new[i,k]
                    self.THL.values[i,k] = self.THL.new[i,k]
                    self.T.values[i,k] = self.T.new[i,k]
                    self.B.values[i,k] = self.B.new[i,k]
        return

    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState.ReferenceState Ref):

        Stats.write_profile('updraft_area', self.Area.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_w', self.W.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_rho_a_w', self.rhoaW.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_qt', self.QT.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_rho_a_qt', self.rhoaQT.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_ql', self.QL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_RH', self.RH.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        if self.H.name == 'thetal':
            Stats.write_profile('updraft_thetal', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('updraft_rho_a_thetal', self.rhoaH.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('updraft_s', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            Stats.write_profile('updraft_rho_a_s', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            #Stats.write_profile('updraft_thetal', self.THL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        Stats.write_profile('updraft_temperature', self.T.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_buoyancy', self.B.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])

        self.upd_cloud_diagnostics(Ref)
        Stats.write_profile('updraft_cloud_fraction', self.cloud_fraction[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        # Note definition of cloud cover : each updraft is associated with a cloud cover equal to the maximum
        # area fraction of the updraft where ql > 0. Each updraft is assumed to have maximum overlap with respect to
        # itself (i.e. no consideration of tilting due to shear) while the updraft classes are assumed to have no overlap
        # at all. Thus total updraft cover is the sum of each updraft's cover
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base',  np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top',   np.amax(self.cloud_top))
        Stats.write_ts('updraft_lwp',         self.lwp)
        return

    cpdef upd_cloud_diagnostics(self, ReferenceState.ReferenceState Ref):
        cdef Py_ssize_t i, k
        self.lwp = 0.

        for i in xrange(self.n_updrafts):
            #TODO check the setting of ghost point z_half

            self.cloud_base[i] = self.Gr.z_half[self.Gr.nzg-self.Gr.gw-1]
            self.cloud_top[i] = 0.0
            self.updraft_top[i] = 0.0
            self.cloud_cover[i] = 0.0

            for k in xrange(self.Gr.gw,self.Gr.nzg-self.Gr.gw):

                if self.Area.values[i,k] > 1e-3:
                    self.updraft_top[i] = fmax(self.updraft_top[i], self.Gr.z_half[k])
                    self.lwp += Ref.rho0_half[k] * self.QL.values[i,k] * self.Area.values[i,k] * self.Gr.dz

                    if self.QL.values[i,k] > 1e-8:
                        self.cloud_base[i]  = fmin(self.cloud_base[i],  self.Gr.z_half[k])
                        self.cloud_top[i]   = fmax(self.cloud_top[i],   self.Gr.z_half[k])
                        self.cloud_cover[i] = fmax(self.cloud_cover[i], self.Area.values[i,k])

        return


cdef class UpdraftThermodynamics:
    def __init__(self, n_updraft, Grid.Grid Gr,
                 ReferenceState.ReferenceState Ref, UpdraftVariables UpdVar,
                 RainVariables Rain):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft

        if UpdVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif UpdVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        # rain source from each updraft from all sub-timesteps
        self.prec_source_h  = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_qt = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')

        # rain source from all updrafts from all sub-timesteps
        self.prec_source_h_tot  = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.prec_source_qt_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')

        return

    cpdef clear_precip_sources(self):
        """
        clear precipitation source terms for QT and H from each updraft
        """
        self.prec_source_qt[:,:] = 0.
        self.prec_source_h[:,:]  = 0.
        return

    cpdef update_total_precip_sources(self):
        """
        sum precipitation source terms for QT and H from all sub-timesteps
        """
        self.prec_source_h_tot  = np.sum(self.prec_source_h,  axis=0)
        self.prec_source_qt_tot = np.sum(self.prec_source_qt, axis=0)
        return

    cpdef buoyancy(self, UpdraftVariables UpdVar, EnvironmentVariables EnvVar,
                   GridMeanVariables GMV, bint extrap):
        cdef:
            Py_ssize_t k, i
            double alpha, qv, qt, t, h
            Py_ssize_t gw = self.Gr.gw

        UpdVar.Area.bulkvalues = np.sum(UpdVar.Area.values,axis=0)

        if not extrap:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.nzg):
                        if UpdVar.Area.values[i,k] > 0.0:
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], UpdVar.T.values[i,k], UpdVar.QT.values[i,k], qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) #- GMV.B.values[k]
                        else:
                            UpdVar.B.values[i,k] = EnvVar.B.values[k]
                        UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], UpdVar.QT.values[i,k],
                                                    UpdVar.QL.values[i,k], 0.0, UpdVar.T.values[i,k])
        else:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        if UpdVar.Area.values[i,k] > 0.0:
                            qt = UpdVar.QT.values[i,k]
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            h = UpdVar.H.values[i,k]
                            t = UpdVar.T.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
                            UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, t)

                        elif UpdVar.Area.values[i,k-1] > 0.0 and k>self.Gr.gw:
                            sa = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k],
                                     qt, h)
                            qt -= sa.ql
                            qv = qt
                            t = sa.T
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
                            UpdVar.RH.values[i,k] = relative_humidity_c(self.Ref.p0_half[k], qt, qt-qv, 0.0, t)
                        else:
                            UpdVar.B.values[i,k] = EnvVar.B.values[k]
                            UpdVar.RH.values[i,k] = EnvVar.RH.values[k]


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[k]
                for i in xrange(self.n_updraft):
                    GMV.B.values[k] += UpdVar.Area.values[i,k] * UpdVar.B.values[i,k]
                for i in xrange(self.n_updraft):
                    UpdVar.B.values[i,k] -= GMV.B.values[k]
                EnvVar.B.values[k] -= GMV.B.values[k]

        return

    cpdef microphysics(self, UpdraftVariables UpdVar, RainVariables Rain, double dt):
        """
        compute precipitation source terms
        """
        cdef:
            Py_ssize_t k, i

            rain_struct rst
            mph_struct  mph
            eos_struct  sa

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):

                    # autoconversion and accretion
                    mph = microphysics_rain_src(
                        Rain.rain_model,
                        UpdVar.QT.new[i,k],
                        UpdVar.QL.new[i,k],
                        Rain.Upd_QR.values[k],
                        UpdVar.Area.new[i,k],
                        UpdVar.T.new[i,k],
                        self.Ref.p0_half[k],
                        self.Ref.rho0_half[k],
                        dt
                    )

                    # update Updraft.new
                    UpdVar.QT.new[i,k] = mph.qt
                    UpdVar.QL.new[i,k] = mph.ql
                    UpdVar.H.new[i,k]  = mph.thl

                    # update rain sources of state variables
                    self.prec_source_qt[i,k] -= mph.qr_src * UpdVar.Area.new[i,k]
                    self.prec_source_h[i,k]  += mph.thl_rain_src * UpdVar.Area.new[i,k]
        return
