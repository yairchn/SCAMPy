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
from Variables cimport GridMeanVariables, SubdomainVariable, SubdomainVariable_2m
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from libc.math cimport fmax, fmin
import pylab as plt


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

        if self.name == 'w':
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

cdef class UpdraftVariable_2m:
    def __init__(self, nz, loc, kind, name, units):
        self.values = np.zeros((0,nz),dtype=np.double, order='c')
        self.dissipation = np.zeros((0,nz),dtype=np.double, order='c')
        self.entr_gain = np.zeros((0,nz),dtype=np.double, order='c')
        self.detr_loss = np.zeros((0,nz),dtype=np.double, order='c')
        self.buoy = np.zeros((0,nz),dtype=np.double, order='c')
        self.press = np.zeros((0,nz),dtype=np.double, order='c')
        self.shear = np.zeros((0,nz),dtype=np.double, order='c')
        self.interdomain = np.zeros((0,nz),dtype=np.double, order='c')
        self.rain_src = np.zeros((0,nz),dtype=np.double, order='c')
        if loc != 'half':
            print('Invalid location setting for variable! Must be half')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

cdef class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Grid.Grid Gr):
        self.Gr = Gr
        self.n_updrafts = nu
        cdef:
            Py_ssize_t nzg = Gr.nzg
            Py_ssize_t i, k

        self.W = SubdomainVariable(nu, nzg, 'full', 'velocity', 'w','m/s' )
        self.Area = SubdomainVariable(nu, nzg, 'full', 'scalar', 'area_fraction','[-]' )
        self.QT = SubdomainVariable(nu, nzg, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = SubdomainVariable(nu, nzg, 'half', 'scalar', 'ql','kg/kg' )
        self.QR = SubdomainVariable(nu, nzg, 'half', 'scalar', 'qr','kg/kg' )
        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = SubdomainVariable(nu, nzg, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = SubdomainVariable(nu, nzg, 'half', 'scalar', 'thetal','K' )

        self.THL = SubdomainVariable(nu, nzg, 'half', 'scalar', 'thetal', 'K')
        self.T = SubdomainVariable(nu, nzg, 'half', 'scalar', 'temperature','K' )
        self.B = SubdomainVariable(nu, nzg, 'half', 'scalar', 'buoyancy','m^2/s^3' )


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

        self.TKE = SubdomainVariable_2m(nu, nzg, 'half', 'scalar', 'tke','m^2/s^2' )
        self.QTvar = SubdomainVariable_2m(nu, nzg, 'half', 'scalar', 'qt_var','kg^2/kg^2' )
        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.Hvar = SubdomainVariable_2m(nu, nzg, 'half', 'scalar', 's_var', '(J/kg/K)^2')
            self.HQTcov = SubdomainVariable_2m(nu, nzg, 'half', 'scalar', 's_qt_covar', '(J/kg/K)(kg/kg)' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.Hvar = SubdomainVariable_2m(nu, nzg, 'half', 'scalar', 'thetal_var', 'K^2')
            self.HQTcov = SubdomainVariable_2m(nu, nzg, 'half', 'scalar', 'thetal_qt_covar', 'K(kg/kg)' )

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

        self.cloud_base = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_top = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_cover = np.zeros((nu,), dtype=np.double, order='c')


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
                    # Simple treatment for now, revise when multiple updraft closures
                    # become more well defined
                    if self.prognostic:
                        self.Area.values[i,k] = 0.0 #self.updraft_fraction/self.n_updrafts
                    else:
                        self.Area.values[i,k] = self.updraft_fraction/self.n_updrafts
                    self.QT.values[i,k] = GMV.QT.values[k]
                    self.QL.values[i,k] = GMV.QL.values[k]
                    self.QR.values[i,k] = GMV.QR.values[k]
                    self.H.values[i,k] = GMV.H.values[k]
                    self.T.values[i,k] = GMV.T.values[k]
                    self.B.values[i,k] = 0.0
                self.Area.values[i,gw] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.Gr)
        self.QR.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_profile('updraft_bulk_area')
        Stats.add_profile('updraft_bulk_w')
        Stats.add_profile('updraft_bulk_qt')
        Stats.add_profile('updraft_bulk_ql')
        Stats.add_profile('updraft_bulk_qr')
        if self.H.name == 'thetal':
            Stats.add_profile('updraft_bulk_thetal')
        else:
            # Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_bulk_s')
        Stats.add_profile('updraft_bulk_temperature')
        Stats.add_profile('updraft_bulk_buoyancy')

        Stats.add_ts('updraft_bulk_cloud_cover')
        Stats.add_ts('updraft_bulk_cloud_base')
        Stats.add_ts('updraft_bulk_cloud_top')

        if self.calc_tke:
            Stats.add_profile('updraft_bulk_tke')
        if self.calc_scalar_var:
            Stats.add_profile('updraft_bulk_Hvar')
            Stats.add_profile('updraft_bulk_QTvar')
            Stats.add_profile('updraft_bulk_HQTcov')

        if self.n_updrafts>1:
            for i in self.n_updrafts:
                Stats.add_profile('updraft'+str(i)+'_area')
                Stats.add_profile('updraft'+str(i)+'_w')
                Stats.add_profile('updraft'+str(i)+'_qt')
                Stats.add_profile('updraft'+str(i)+'_ql')
                Stats.add_profile('updraft'+str(i)+'_qr')
                if self.H.name == 'thetal':
                    Stats.add_profile('updraft'+str(i)+'_thetal')
                else:
                    # Stats.add_profile('updraft_thetal')
                    Stats.add_profile('updraft'+str(i)+'_s')
                Stats.add_profile('updraft'+str(i)+'_temperature')
                Stats.add_profile('updraft'+str(i)+'_buoyancy')

                Stats.add_ts('updraft'+str(i)+'_cloud_cover')
                Stats.add_ts('updraft'+str(i)+'_cloud_base')
                Stats.add_ts('updraft'+str(i)+'_cloud_top')

                if self.calc_tke:
                    Stats.add_profile('updraft'+str(i)+'_tke')
                if self.calc_scalar_var:
                    Stats.add_profile('updraft'+str(i)+'_Hvar')
                    Stats.add_profile('updraft'+str(i)+'_QTvar')
                    Stats.add_profile('updraft'+str(i)+'_HQTcov')

        return

    cpdef set_means(self, GridMeanVariables GMV):

        cdef:
            Py_ssize_t i, k

        self.Area.bulkvalues = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.QR.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0


        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                if self.Area.bulkvalues[k] > 1.0e-20:
                    for i in xrange(self.n_updrafts):
                        self.QT.bulkvalues[k] += self.Area.values[i,k] * self.QT.values[i,k]/self.Area.bulkvalues[k]
                        self.QL.bulkvalues[k] += self.Area.values[i,k] * self.QL.values[i,k]/self.Area.bulkvalues[k]
                        self.QR.bulkvalues[k] += self.Area.values[i,k] * self.QR.values[i,k]/self.Area.bulkvalues[k]
                        self.H.bulkvalues[k] += self.Area.values[i,k] * self.H.values[i,k]/self.Area.bulkvalues[k]
                        self.T.bulkvalues[k] += self.Area.values[i,k] * self.T.values[i,k]/self.Area.bulkvalues[k]
                        self.B.bulkvalues[k] += self.Area.values[i,k] * self.B.values[i,k]/self.Area.bulkvalues[k]
                        self.W.bulkvalues[k] += ((self.Area.values[i,k] + self.Area.values[i,k+1]) * self.W.values[i,k]
                                            /(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1]))
                else:
                    self.QT.bulkvalues[k] = GMV.QT.values[k]
                    self.QR.bulkvalues[k] = GMV.QR.values[k]
                    self.QL.bulkvalues[k] = 0.0
                    self.H.bulkvalues[k] = GMV.H.values[k]
                    self.T.bulkvalues[k] = GMV.T.values[k]
                    self.B.bulkvalues[k] = 0.0
                    self.W.bulkvalues[k] = 0.0

        return
    # quick utility to set "new" arrays with values in the "values" arrays
    cpdef set_new_with_values(self):
        with nogil:
            for i in xrange(self.n_updrafts):
                for k in xrange(self.Gr.nzg):
                    self.W.new[i,k] = self.W.values[i,k]
                    self.Area.new[i,k] = self.Area.values[i,k]
                    self.QT.new[i,k] = self.QT.values[i,k]
                    self.QL.new[i,k] = self.QL.values[i,k]
                    self.QR.new[i,k] = self.QR.values[i,k]
                    self.H.new[i,k] = self.H.values[i,k]
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
                    self.Area.old[i,k] = self.Area.values[i,k]
                    self.QT.old[i,k] = self.QT.values[i,k]
                    self.QL.old[i,k] = self.QL.values[i,k]
                    self.QR.old[i,k] = self.QR.values[i,k]
                    self.H.old[i,k] = self.H.values[i,k]
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
                    self.Area.values[i,k] = self.Area.new[i,k]
                    self.QT.values[i,k] = self.QT.new[i,k]
                    self.QL.values[i,k] = self.QL.new[i,k]
                    self.QR.values[i,k] = self.QR.new[i,k]
                    self.H.values[i,k] = self.H.new[i,k]
                    self.THL.values[i,k] = self.THL.new[i,k]
                    self.T.values[i,k] = self.T.new[i,k]
                    self.B.values[i,k] = self.B.new[i,k]
        return


    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_profile('updraft_bulk_area', self.Area.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_bulk_w', self.W.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_bulk_qt', self.QT.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_bulk_ql', self.QL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_bulk_qr', self.QR.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        if self.H.name == 'thetal':
            Stats.write_profile('updraft_bulk_thetal', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        else:
            Stats.write_profile('updraft_bulk_s', self.H.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
            #Stats.write_profile('updraft_thetal', self.THL.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_bulk_temperature', self.T.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        Stats.write_profile('updraft_bulk_buoyancy', self.B.bulkvalues[self.Gr.gw:self.Gr.nzg-self.Gr.gw])
        self.get_cloud_base_top_cover()
        # Note definition of cloud cover : each updraft is associated with a cloud cover equal to the maximum
        # area fraction of the updraft where ql > 0. Each updraft is assumed to have maximum overlap with respect to
        # itself (i.e. no consideration of tilting due to shear) while the updraft classes are assumed to have no overlap
        # at all. Thus total updraft cover is the sum of each updraft's cover
        Stats.write_ts('updraft_bulk_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_bulk_cloud_base', np.amin(self.cloud_base))
        Stats.write_ts('updraft_bulk_cloud_top', np.amax(self.cloud_top))
        if self.n_updrafts>1:
            for i in self.n_updrafts:
                Stats.write_profile('updraft'+str(i)+'_area', self.Area.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                Stats.write_profile('updraft'+str(i)+'_w', self.W.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                Stats.write_profile('updraft'+str(i)+'_qt', self.QT.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                Stats.write_profile('updraft'+str(i)+'_ql', self.QL.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                Stats.write_profile('updraft'+str(i)+'_qr', self.QR.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                if self.H.name == 'thetal':
                    Stats.write_profile('updraft'+str(i)+'_thetal', self.H.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                else:
                    Stats.write_profile('updraft'+str(i)+'_s', self.H.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                    #Stats.write_profile('updraft_thetal', self.THL.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                Stats.write_profile('updraft'+str(i)+'_temperature', self.T.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                Stats.write_profile('updraft'+str(i)+'_buoyancy', self.B.values[i,self.Gr.gw:self.Gr.nzg-self.Gr.gw])
                self.get_cloud_base_top_cover()
                # Note definition of cloud cover : each updraft is associated with a cloud cover equal to the maximum
                # area fraction of the updraft where ql > 0. Each updraft is assumed to have maximum overlap with respect to
                # itself (i.e. no consideration of tilting due to shear) while the updraft classes are assumed to have no overlap
                # at all. Thus total updraft cover is the sum of each updraft's cover

                # Todo yair - check if the clould properties below are calcualted per upd
                #Stats.write_ts('updraft_cloud_cover_'+str(i), np.sum(self.cloud_cover))
                #Stats.write_ts('updraft_cloud_base_'+str(i), np.amin(self.cloud_base))
                #Stats.write_ts('updraft_cloud_top_'+str(i), np.amax(self.cloud_top))

        return

    cpdef get_cloud_base_top_cover(self):
        cdef Py_ssize_t i, k

        for i in xrange(self.n_updrafts):
            # Todo check the setting of ghost point z_half
            self.cloud_base[i] = self.Gr.z_half[self.Gr.nzg-self.Gr.gw-1]
            self.cloud_top[i] = 0.0
            self.cloud_cover[i] = 0.0
            for k in xrange(self.Gr.gw,self.Gr.nzg-self.Gr.gw):
                if self.QL.values[i,k] > 1e-8 and self.Area.values[i,k] > 1e-3:
                    self.cloud_base[i] = fmin(self.cloud_base[i], self.Gr.z_half[k])
                    self.cloud_top[i] = fmax(self.cloud_top[i], self.Gr.z_half[k])
                    self.cloud_cover[i] = fmax(self.cloud_cover[i], self.Area.values[i,k])


        return

cdef class UpdraftThermodynamics:
    def __init__(self, n_updraft, Grid.Grid Gr, ReferenceState.ReferenceState Ref, UpdraftVariables UpdVar):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft
        if UpdVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif UpdVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        return
    cpdef satadjust(self, UpdraftVariables UpdVar):
        #Update T, QL
        cdef:
            Py_ssize_t k, i
            eos_struct sa

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):
                    sa = eos(self.t_to_prog_fp,self.prog_to_t_fp, self.Ref.p0_half[k],
                             UpdVar.QT.values[i,k], UpdVar.H.values[i,k])
                    UpdVar.QL.values[i,k] = sa.ql
                    UpdVar.T.values[i,k] = sa.T
        return

    cpdef buoyancy(self,  UpdraftVariables UpdVar, EnvironmentVariables EnvVar,GridMeanVariables GMV, bint extrap):
        cdef:
            Py_ssize_t k, i
            double alpha, qv, qt, t, h
            Py_ssize_t gw = self.Gr.gw

        UpdVar.Area.bulkvalues = np.sum(UpdVar.Area.values,axis=0)


        if not extrap:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.nzg):
                        qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                        alpha = alpha_c(self.Ref.p0_half[k], UpdVar.T.values[i,k], UpdVar.QT.values[i,k], qv)
                        UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) #- GMV.B.values[k]
        else:
            with nogil:
                for i in xrange(self.n_updraft):
                    for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                        if UpdVar.Area.values[i,k] > 1e-3:
                            qt = UpdVar.QT.values[i,k]
                            qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                            h = UpdVar.H.values[i,k]
                            t = UpdVar.T.values[i,k]
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)

                        else:
                            sa = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k],
                                     qt, h)
                            qt -= sa.ql
                            qv = qt
                            t = sa.T
                            alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                            UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
        with nogil:
            for k in xrange(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[0,k]
                for i in xrange(self.n_updraft):
                    GMV.B.values[k] += UpdVar.Area.values[i,k] * UpdVar.B.values[i,k]
                for i in xrange(self.n_updraft):
                    UpdVar.B.values[i,k] -= GMV.B.values[k]
                EnvVar.B.values[0,k] -= GMV.B.values[k]

        return


#Implements a simple "microphysics" that clips excess humidity above a user-specified level
cdef class UpdraftMicrophysics:
    def __init__(self, paramlist, n_updraft, Grid.Grid Gr, ReferenceState.ReferenceState Ref):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft
        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        self.prec_source_h = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_qt = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_h_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        self.prec_source_qt_tot = np.zeros((Gr.nzg,), dtype=np.double, order='c')
        return

    cpdef compute_sources(self, UpdraftVariables UpdVar):
        """
        Compute precipitation source terms for QT, QR and H
        """
        cdef:
            Py_ssize_t k, i
            double tmp_qr

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):
                    tmp_qr = acnv_instant(UpdVar.QL.values[i,k], UpdVar.QT.values[i,k], self.max_supersaturation,\
                                          UpdVar.T.values[i,k], self.Ref.p0_half[k])
                    self.prec_source_qt[i,k] = -tmp_qr
                    self.prec_source_h[i,k]  = rain_source_to_thetal(self.Ref.p0_half[k], UpdVar.T.values[i,k],\
                                                 UpdVar.QT.values[i,k], UpdVar.QL.values[i,k], 0.0, tmp_qr)
                                                                                              #TODO assumes no ice
        self.prec_source_h_tot  = np.sum(np.multiply(self.prec_source_h,  UpdVar.Area.values), axis=0)
        self.prec_source_qt_tot = np.sum(np.multiply(self.prec_source_qt, UpdVar.Area.values), axis=0)

        return

    cpdef update_updraftvars(self, UpdraftVariables UpdVar):
        """
        Apply precipitation source terms to QL, QR and H
        """
        cdef:
            Py_ssize_t k, i

        with nogil:
            for i in xrange(self.n_updraft):
                for k in xrange(self.Gr.nzg):
                    UpdVar.QT.values[i,k] += self.prec_source_qt[i,k]
                    UpdVar.QL.values[i,k] += self.prec_source_qt[i,k]
                    UpdVar.QR.values[i,k] -= self.prec_source_qt[i,k]
                    UpdVar.H.values[i,k] += self.prec_source_h[i,k]
        return

    cdef void compute_update_combined_local_thetal(self, double p0, double T, double *qt, double *ql, double *qr, double *h,
                                               Py_ssize_t i, Py_ssize_t k) nogil :

        # Language note: array indexing must be used to dereference pointers in Cython. * notation (C-style dereferencing)
        # is reserved for packing tuples

        tmp_qr =  acnv_instant(ql[0], qt[0], self.max_supersaturation, T, p0)
        self.prec_source_qt[i,k] = -tmp_qr
        self.prec_source_h[i,k]  = rain_source_to_thetal(p0, T, qt[0], ql[0], 0.0, tmp_qr)
                                                                             #TODO - assumes no ice
        qt[0] += self.prec_source_qt[i,k]
        ql[0] += self.prec_source_qt[i,k]
        qr[0] -= self.prec_source_qt[i,k]
        h[0]  += self.prec_source_h[i,k]

        return
