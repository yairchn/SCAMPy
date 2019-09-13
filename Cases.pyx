import numpy as np
include "parameters.pxi"
import cython

from Grid cimport Grid
from Variables cimport GridMeanVariables
from ReferenceState cimport ReferenceState
from TimeStepping cimport  TimeStepping
cimport Surface
cimport Forcing
from NetCDFIO cimport NetCDFIO_Stats
from thermodynamic_functions cimport *
import math as mt
from libc.math cimport sqrt, log, fabs,atan, exp, fmax

def CasesFactory(namelist, paramlist):
    if namelist['meta']['casename'] == 'Soares':
        return Soares(paramlist)
    elif namelist['meta']['casename'] == 'Bomex':
        return Bomex(paramlist)
    elif namelist['meta']['casename'] == 'life_cycle_Tan2018':
        return life_cycle_Tan2018(paramlist)
    elif namelist['meta']['casename'] == 'Rico':
        return Rico_fix_flux(paramlist)
    elif namelist['meta']['casename'] == 'TRMM_LBA':
        return TRMM_LBA(paramlist)
    elif namelist['meta']['casename'] == 'ARM_SGP':
        return ARM_SGP(paramlist)
    elif namelist['meta']['casename'] == 'GATE_III':
        return GATE_III(paramlist)
    elif namelist['meta']['casename'] == 'DYCOMS_RF01':
        return DYCOMS_RF01(paramlist)
    elif namelist['meta']['casename'] == 'GABLS':
        return GABLS(paramlist)
    elif namelist['meta']['casename'] == 'SP':
        return SP(paramlist)

    else:
        print('case not recognized')
    return


cdef class CasesBase:
    def __init__(self, paramlist):
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref ):
        return
    cpdef initialize_forcing(self, Grid Gr,  ReferenceState Ref, GridMeanVariables GMV):
        return
    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        Stats.add_ts('Tsurface')
        Stats.add_ts('shf')
        Stats.add_ts('lhf')
        Stats.add_ts('ustar')
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        Stats.write_ts('Tsurface', self.Sur.Tsurface)
        Stats.write_ts('shf', self.Sur.shf)
        Stats.write_ts('lhf', self.Sur.lhf)
        Stats.write_ts('ustar', self.Sur.ustar)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        return
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        return


cdef class Soares(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'Soares2004'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingNone()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = False
        self.Fo.apply_subsidence = False
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1000.0 * 100.0
        Ref.qtg = 4.5e-3
        Ref.Tg = 300.0
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] theta = np.zeros((Gr.nzg,),dtype=np.double, order='c')
            double ql = 0.0, qi = 0.0
            Py_ssize_t k

        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            if Gr.z_half[k] <= 1350.0:
                GMV.QT.values[k] = 5.0e-3 - 3.7e-4* Gr.z_half[k]/1000.0
                theta[k] = 300.0

            else:
                GMV.QT.values[k] = 5.0e-3 - 3.7e-4 * 1.35 - 9.4e-4 * (Gr.z_half[k]-1350.0)/1000.0
                theta[k] = 300.0 + 2.0 * (Gr.z_half[k]-1350.0)/1000.0
            GMV.U.values[k] = 0.01

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = theta[k]
                GMV.T.values[k] =  theta[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = theta[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = theta[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref ):
        self.Sur.zrough = 1.0e-4
        self.Sur.Tsurface = 300.0
        self.Sur.qsurface = 5e-3
        self.Sur.lhf = 0.0 #2.5e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 6.0e-2 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = False
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux = g * ( 6.0e-2/self.Sur.Tsurface + (eps_vi -1.0)* 2.5e-5) # This will be overwritten
        self.Sur.initialize()

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV)
        return
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return

cdef class Bomex(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'Bomex'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 0.376e-4 # s^{-1}
        self.Fo.apply_subsidence = True
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.015e5  #Pressure at ground
        Ref.Tg = 300.4  #Temperature at ground
        Ref.qtg = 0.02245   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Bomex is cloud-free
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set Thetal profile
            if Gr.z_half[k] <= 520.:
                thetal[k] = 298.7
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                thetal[k] = 298.7 + (Gr.z_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000:
                thetal[k] = 302.4 + (Gr.z_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
            if Gr.z_half[k] > 2000.0:
                thetal[k] = 308.2 + (Gr.z_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

            #Set qt profile
            if Gr.z_half[k] <= 520:
                GMV.QT.values[k] = (17.0 + (Gr.z_half[k]) * (16.3-17.0)/520.0)/1000.0
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                GMV.QT.values[k] = (16.3 + (Gr.z_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0))/1000.0
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000.0:
                GMV.QT.values[k] = (10.7 + (Gr.z_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0))/1000.0
            if Gr.z_half[k] > 2000.0:
                GMV.QT.values[k] = (4.2 + (Gr.z_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0))/1000.0


            #Set u profile
            if Gr.z_half[k] <= 700.0:
                GMV.U.values[k] = -8.75
            if Gr.z_half[k] > 700.0:
                GMV.U.values[k] = -8.75 + (Gr.z_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = 299.1 * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # m/s
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize()
        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        cdef Py_ssize_t k
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Geostrophic velocity profiles. vg = 0
            self.Fo.ug[k] = -10.0 + (1.8e-3)*Gr.z_half[k]
            # Set large-scale cooling
            if Gr.z_half[k] <= 1500.0:
                self.Fo.dTdt[k] =  (-2.0/(3600 * 24.0))  * exner_c(Ref.p0_half[k])
            else:
                self.Fo.dTdt[k] = (-2.0/(3600 * 24.0) + (Gr.z_half[k] - 1500.0)
                                    * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner_c(Ref.p0_half[k])

            # Set large-scale drying
            if Gr.z_half[k] <= 300.0:
                self.Fo.dqtdt[k] = -1.2e-8   #kg/(kg * s)
            if Gr.z_half[k] > 300.0 and Gr.z_half[k] <= 500.0:
                self.Fo.dqtdt[k] = -1.2e-8 + (Gr.z_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

            #Set large scale subsidence
            if Gr.z_half[k] <= 1500.0:
                self.Fo.subsidence[k] = 0.0 + Gr.z_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
            if Gr.z_half[k] > 1500.0 and Gr.z_half[k] <= 2100.0:
                self.Fo.subsidence[k] = -0.65/100 + (Gr.z_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV)
        return
    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return

cdef class life_cycle_Tan2018(CasesBase):
    # Taken from: "An extended eddy- diffusivity mass-flux scheme for unified representation of subgrid-scale turbulence and convection"
    # Tan, Z., Kaul, C. M., Pressel, K. G., Cohen, Y., Schneider, T., & Teixeira, J. (2018).
    #  Journal of Advances in Modeling Earth Systems, 10. https://doi.org/10.1002/2017MS001162

    def __init__(self, paramlist):
        self.casename = 'life_cycle_Tan2018'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 0.376e-4 # s^{-1}
        self.Fo.apply_subsidence = True
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.015e5  #Pressure at ground
        Ref.Tg = 300.4  #Temperature at ground
        Ref.qtg = 0.02245   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Bomex is cloud-free
        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set Thetal profile
            if Gr.z_half[k] <= 520.:
                thetal[k] = 298.7
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                thetal[k] = 298.7 + (Gr.z_half[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000:
                thetal[k] = 302.4 + (Gr.z_half[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
            if Gr.z_half[k] > 2000.0:
                thetal[k] = 308.2 + (Gr.z_half[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)

            #Set qt profile
            if Gr.z_half[k] <= 520:
                GMV.QT.values[k] = (17.0 + (Gr.z_half[k]) * (16.3-17.0)/520.0)/1000.0
            if Gr.z_half[k] > 520.0 and Gr.z_half[k] <= 1480.0:
                GMV.QT.values[k] = (16.3 + (Gr.z_half[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0))/1000.0
            if Gr.z_half[k] > 1480.0 and Gr.z_half[k] <= 2000.0:
                GMV.QT.values[k] = (10.7 + (Gr.z_half[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0))/1000.0
            if Gr.z_half[k] > 2000.0:
                GMV.QT.values[k] = (4.2 + (Gr.z_half[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0))/1000.0


            #Set u profile
            if Gr.z_half[k] <= 700.0:
                GMV.U.values[k] = -8.75
            if Gr.z_half[k] > 700.0:
                GMV.U.values[k] = -8.75 + (Gr.z_half[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)


        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = 299.1 * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.lhf0 = self.Sur.lhf
        self.shf0 = self.Sur.shf
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # m/s
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.bflux = (g * ((8.0e-3 + (eps_vi-1.0)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (eps_vi-1) * 22.45e-3))))
        self.Sur.initialize()
        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # Geostrophic velocity profiles. vg = 0
            self.Fo.ug[k] = -10.0 + (1.8e-3)*Gr.z_half[k]
            # Set large-scale cooling
            if Gr.z_half[k] <= 1500.0:
                self.Fo.dTdt[k] =  (-2.0/(3600 * 24.0))  * exner_c(Ref.p0_half[k])
            else:
                self.Fo.dTdt[k] = (-2.0/(3600 * 24.0) + (Gr.z_half[k] - 1500.0)
                                    * (0.0 - -2.0/(3600 * 24.0)) / (3000.0 - 1500.0)) * exner_c(Ref.p0_half[k])
            # Set large-scale drying
            if Gr.z_half[k] <= 300.0:
                self.Fo.dqtdt[k] = -1.2e-8   #kg/(kg * s)
            if Gr.z_half[k] > 300.0 and Gr.z_half[k] <= 500.0:
                self.Fo.dqtdt[k] = -1.2e-8 + (Gr.z_half[k] - 300.0)*(0.0 - -1.2e-8)/(500.0 - 300.0) #kg/(kg * s)

            #Set large scale subsidence
            if Gr.z_half[k] <= 1500.0:
                self.Fo.subsidence[k] = 0.0 + Gr.z_half[k]*(-0.65/100.0 - 0.0)/(1500.0 - 0.0)
            if Gr.z_half[k] > 1500.0 and Gr.z_half[k] <= 2100.0:
                self.Fo.subsidence[k] = -0.65/100 + (Gr.z_half[k] - 1500.0)* (0.0 - -0.65/100.0)/(2100.0 - 1500.0)
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        weight = 1.0
        weight_factor = 0.01 + 0.99 *(np.cos(2.0*pi * TS.t /3600.0) + 1.0)/2.0
        weight = weight * weight_factor
        self.Sur.lhf = self.lhf0*weight
        self.Sur.shf = self.shf0*weight
        self.Sur.bflux = (g * ((8.0e-3*weight + (eps_vi-1.0)*(299.1 * 5.2e-5*weight  + 22.45e-3 * 8.0e-3*weight)) /(299.1 * (1.0 + (eps_vi-1) * 22.45e-3))))
        self.Sur.update(GMV)
        return
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        self.Fo.update(GMV)
        return

cdef class Rico(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'Rico'
        self.Sur = Surface.SurfaceFixedCoeffs(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        cdef double latitude = 18.0
        self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = True
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.0154e5  #Pressure at ground
        Ref.Tg = 299.8  #Temperature at ground
        cdef double pvg = pv_star(Ref.Tg)
        Ref.qtg = eps_v * pvg/(Ref.Pg - pvg)   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Rico is cloud-free
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] =  -9.9 + 2.0e-3 * Gr.z_half[k]
            GMV.V.values[k] = -3.8
            #Set Thetal profile
            if Gr.z_half[k] <= 740.0:
                thetal[k] = 297.9
            else:
                thetal[k] = 297.9 + (317.0-297.9)/(4000.0-740.0)*(Gr.z_half[k] - 740.0)

            #Set qt profile
            if Gr.z_half[k] <= 740.0:
                GMV.QT.values[k] =  (16.0 + (13.8 - 16.0)/740.0 * Gr.z_half[k])/1000.0
            elif Gr.z_half[k] > 740.0 and Gr.z_half[k] <= 3260.0:
                GMV.QT.values[k] = (13.8 + (2.4 - 13.8)/(3260.0-740.0) * (Gr.z_half[k] - 740.0))/1000.0
            else:
                GMV.QT.values[k] = (2.4 + (1.8-2.4)/(4000.0-3260.0)*(Gr.z_half[k] - 3260.0))/1000.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()


        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.zrough = 0.00015
        self.Sur.cm  = 0.001229
        self.Sur.ch = 0.001094
        self.Sur.cq = 0.001133
        # Adjust for non-IC grid spacing
        grid_adjust = (np.log(20.0/self.Sur.zrough)/np.log(Gr.z_half[Gr.gw]/self.Sur.zrough))**2
        self.Sur.cm = self.Sur.cm * grid_adjust
        self.Sur.ch = self.Sur.ch * grid_adjust
        self.Sur.cq = self.Sur.cq * grid_adjust
        self.Sur.Tsurface = 299.8
        self.Sur.initialize()
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        for k in xrange(Gr.nzg):
            # Geostrophic velocity profiles
            self.Fo.ug[k] = -9.9 + 2.0e-3 * Gr.z_half[k]
            self.Fo.vg[k] = -3.8
            # Set large-scale cooling
            self.Fo.dTdt[k] =  (-2.5/(3600.0 * 24.0))  * exner_c(Ref.p0_half[k])

            # Set large-scale moistening
            if Gr.z_half[k] <= 2980.0:
                self.Fo.dqtdt[k] =  (-1.0 + 1.3456/2980.0 * Gr.z_half[k])/86400.0/1000.0   #kg/(kg * s)
            else:
                self.Fo.dqtdt[k] = 0.3456/86400.0/1000.0

            #Set large scale subsidence
            if Gr.z_half[k] <= 2260.0:
                self.Fo.subsidence[k] = -(0.005/2260.0) * Gr.z_half[k]
            else:
                self.Fo.subsidence[k] = -0.005
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return



cdef class Rico_fix_flux(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'Rico'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        cdef double latitude = 18.0
        self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = True
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.0154e5  #Pressure at ground
        Ref.Tg = 299.8  #Temperature at ground
        cdef double pvg = pv_star(Ref.Tg)
        Ref.qtg = eps_v * pvg/(Ref.Pg - pvg)   #Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of Rico is cloud-free
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] =  -9.9 + 2.0e-3 * Gr.z_half[k]
            GMV.V.values[k] = -3.8
            #Set Thetal profile
            if Gr.z_half[k] <= 740.0:
                thetal[k] = 297.9
            else:
                thetal[k] = 297.9 + (317.0-297.9)/(4000.0-740.0)*(Gr.z_half[k] - 740.0)

            #Set qt profile
            if Gr.z_half[k] <= 740.0:
                GMV.QT.values[k] =  (16.0 + (13.8 - 16.0)/740.0 * Gr.z_half[k])/1000.0
            elif Gr.z_half[k] > 740.0 and Gr.z_half[k] <= 3260.0:
                GMV.QT.values[k] = (13.8 + (2.4 - 13.8)/(3260.0-740.0) * (Gr.z_half[k] - 740.0))/1000.0
            else:
                GMV.QT.values[k] = (2.4 + (1.8-2.4)/(4000.0-3260.0)*(Gr.z_half[k] - 3260.0))/1000.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()


        return
    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.zrough = 0.00015
        self.Sur.Tsurface = 299.8
        self.Sur.qsurface = 22.45e-3 # this is from Bomex
        self.Sur.lhf = 198.2732
        self.Sur.shf = 1.634431
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # m/s
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize()
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        for k in xrange(Gr.nzg):
            # Geostrophic velocity profiles
            self.Fo.ug[k] = -9.9 + 2.0e-3 * Gr.z_half[k]
            self.Fo.vg[k] = -3.8
            # Set large-scale cooling
            self.Fo.dTdt[k] =  (-2.5/(3600.0 * 24.0))  * exner_c(Ref.p0_half[k])

            # Set large-scale moistening
            if Gr.z_half[k] <= 2980.0:
                self.Fo.dqtdt[k] =  (-1.0 + 1.3456/2980.0 * Gr.z_half[k])/86400.0/1000.0   #kg/(kg * s)
            else:
                self.Fo.dqtdt[k] = 0.3456/86400.0/1000.0

            #Set large scale subsidence
            if Gr.z_half[k] <= 2260.0:
                self.Fo.subsidence[k] = -(0.005/2260.0) * Gr.z_half[k]
            else:
                self.Fo.subsidence[k] = -0.005
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        les_time = np.array([ 0.0, 0.02777778, 0.05555556, 0.08333334, 0.1111111, 0.1388889, 0.1666667,0.1944444, 0.2222222, 0.25, 0.2777778, 0.3055556, 0.3333333, 0.3611111, 
                              0.3888889, 0.4166667, 0.4444444, 0.4722222, 0.5, 0.5277778, 0.5555556, 0.5833333, 0.6111111, 0.6388889, 0.6666667, 0.6944444, 0.7222222, 0.75, 
                              0.7777778, 0.8055556, 0.8333333, 0.8611111, 0.8888889, 0.9166667, 0.9444444, 0.9722222, 1, 1.027778, 1.055556, 1.083333, 1.111111, 
                              1.138889, 1.166667, 1.194444, 1.222222, 1.25, 1.277778, 1.305556, 1.333333, 1.361111, 1.388889, 1.416667, 1.444444, 1.472222, 1.5, 
                              1.527778, 1.555556, 1.583333, 1.611111, 1.638889, 1.666667, 1.694444, 1.722222, 1.75, 1.777778, 1.805556, 1.833333, 1.861111, 1.888889, 
                              1.916667, 1.944444, 1.972222, 2, 2.027778, 2.055556, 2.083333, 2.111111, 2.138889, 2.166667, 2.194444, 2.222222, 2.25, 2.277778, 2.305556, 
                              2.333333, 2.361111, 2.388889, 2.416667, 2.444444, 2.472222, 2.5, 2.527778, 2.555556, 2.583333, 2.611111, 2.638889, 2.666667, 2.694444, 
                              2.722222, 2.75, 2.777778, 2.805556, 2.833333, 2.861111, 2.888889, 2.916667, 2.944444, 2.972222, 3, 3.027778, 3.055556, 3.083333, 3.111111, 
                              3.138889, 3.166667, 3.194444, 3.222222, 3.25, 3.277778, 3.305556, 3.333333, 3.361111, 3.388889, 3.416667, 3.444444, 3.472222, 3.5, 
                              3.527778, 3.555556, 3.583333, 3.611111, 3.638889, 3.666667, 3.694444, 3.722222, 3.75, 3.777778, 3.805556, 3.833333, 3.861111, 3.888889, 
                              3.916667, 3.944444, 3.972222, 4, 4.027778, 4.055555, 4.083333, 4.111111, 4.138889, 4.166667, 4.194445, 4.222222, 4.25, 4.277778, 4.305555, 
                              4.333333, 4.361111, 4.388889, 4.416667, 4.444445, 4.472222, 4.5, 4.527778, 4.555555, 4.583333, 4.611111, 4.638889, 4.666667, 4.694445, 
                              4.722222, 4.75, 4.777778, 4.805555, 4.833333, 4.861111, 4.888889, 4.916667, 4.944445, 4.972222, 5, 5.027778, 5.055555, 5.083333, 5.111111, 
                              5.138889, 5.166667, 5.194445, 5.222222, 5.25, 5.277778, 5.305555, 5.333333, 5.361111, 5.388889, 5.416667, 5.444445, 5.472222, 5.5, 
                              5.527778, 5.555555, 5.583333, 5.611111, 5.638889, 5.666667, 5.694445, 5.722222, 5.75, 5.777778, 5.805555, 5.833333, 5.861111, 5.888889, 
                              5.916667, 5.944445, 5.972222, 6, 6.027778, 6.055555, 6.083333, 6.111111, 6.138889, 6.166667, 6.194445, 6.222222, 6.25, 6.277778, 6.305555, 
                              6.333333, 6.361111, 6.388889, 6.416667, 6.444445, 6.472222, 6.5, 6.527778, 6.555555, 6.583333, 6.611111, 6.638889, 6.666667, 6.694445, 
                              6.722222, 6.75, 6.777778, 6.805555, 6.833333, 6.861111, 6.888889, 6.916667, 6.944445, 6.972222, 7, 7.027778, 7.055555, 7.083333, 7.111111, 
                              7.138889, 7.166667, 7.194445, 7.222222, 7.25, 7.277778, 7.305555, 7.333333, 7.361111, 7.388889, 7.416667, 7.444445, 7.472222, 7.5, 
                              7.527778, 7.555555, 7.583333, 7.611111, 7.638889, 7.666667, 7.694445, 7.722222, 7.75, 7.777778, 7.805555, 7.833333, 7.861111, 7.888889, 
                              7.916667, 7.944445, 7.972222, 8, 8.027778, 8.055555, 8.083333, 8.111111, 8.138889, 8.166667, 8.194445, 8.222222, 8.25, 8.277778, 8.305555, 
                              8.333333, 8.361111, 8.388889, 8.416667, 8.444445, 8.472222, 8.5, 8.527778, 8.555555, 8.583333, 8.611111, 8.638889, 8.666667, 8.694445, 
                              8.722222, 8.75, 8.777778, 8.805555, 8.833333, 8.861111, 8.888889, 8.916667, 8.944445, 8.972222, 9, 9.027778, 9.055555, 9.083333, 9.111111, 
                              9.138889, 9.166667, 9.194445, 9.222222, 9.25, 9.277778, 9.305555, 9.333333, 9.361111, 9.388889, 9.416667, 9.444445, 9.472222, 9.5, 
                              9.527778, 9.555555, 9.583333, 9.611111, 9.638889, 9.666667, 9.694445, 9.722222, 9.75, 9.777778, 9.805555, 9.833333, 9.861111, 9.888889, 
                              9.916667, 9.944445, 9.972222, 10, 10.02778, 10.05556, 10.08333, 10.11111, 10.13889, 10.16667, 10.19444, 10.22222, 10.25, 10.27778, 10.30556, 
                              10.33333, 10.36111, 10.38889, 10.41667, 10.44444, 10.47222, 10.5, 10.52778, 10.55556, 10.58333, 10.61111, 10.63889, 10.66667, 10.69444, 
                              10.72222, 10.75, 10.77778, 10.80556, 10.83333, 10.86111, 10.88889, 10.91667, 10.94444, 10.97222, 11, 11.02778, 11.05556, 11.08333, 11.11111, 
                              11.13889, 11.16667, 11.19444, 11.22222, 11.25, 11.27778, 11.30556, 11.33333, 11.36111, 11.38889, 11.41667, 11.44444, 11.47222, 11.5, 
                              11.52778, 11.55556, 11.58333, 11.61111, 11.63889, 11.66667, 11.69444, 11.72222, 11.75, 11.77778, 11.80556, 11.83333, 11.86111, 11.88889, 
                              11.91667, 11.94444, 11.97222, 12, 12.02778, 12.05556, 12.08333, 12.11111, 12.13889, 12.16667, 12.19444, 12.22222, 12.25, 12.27778, 12.30556, 
                              12.33333, 12.36111, 12.38889, 12.41667, 12.44444, 12.47222, 12.5, 12.52778, 12.55556, 12.58333, 12.61111, 12.63889, 12.66667, 12.69444, 
                              12.72222, 12.75, 12.77778, 12.80556, 12.83333, 12.86111, 12.88889, 12.91667, 12.94444, 12.97222, 13, 13.02778, 13.05556, 13.08333, 13.11111, 
                              13.13889, 13.16667, 13.19444, 13.22222, 13.25, 13.27778, 13.30556, 13.33333, 13.36111, 13.38889, 13.41667, 13.44444, 13.47222, 13.5, 
                              13.52778, 13.55556, 13.58333, 13.61111, 13.63889, 13.66667, 13.69444, 13.72222, 13.75, 13.77778, 13.80556, 13.83333, 13.86111, 13.88889, 
                              13.91667, 13.94444, 13.97222, 14, 14.02778, 14.05556, 14.08333, 14.11111, 14.13889, 14.16667, 14.19444, 14.22222, 14.25, 14.27778, 14.30556, 
                              14.33333, 14.36111, 14.38889, 14.41667, 14.44444, 14.47222, 14.5, 14.52778, 14.55556, 14.58333, 14.61111, 14.63889, 14.66667, 14.69444, 
                              14.72222, 14.75, 14.77778, 14.80556, 14.83333, 14.86111, 14.88889, 14.91667, 14.94444, 14.97222, 15, 15.02778, 15.05556, 15.08333, 15.11111, 
                              15.13889, 15.16667, 15.19444, 15.22222, 15.25, 15.27778, 15.30556, 15.33333, 15.36111, 15.38889, 15.41667, 15.44444, 15.47222, 15.5, 
                              15.52778, 15.55556, 15.58333, 15.61111, 15.63889, 15.66667, 15.69444,  15.72222, 15.75, 15.77778, 15.80556, 15.83333, 15.86111, 15.88889,
                              15.91667, 15.94444, 15.97222, 16, 16.02778, 16.05556, 16.08333, 16.11111,  16.13889, 16.16667, 16.19444, 16.22222, 16.25, 16.27778, 16.30556,
                              16.33333, 16.36111, 16.38889, 16.41667, 16.44444, 16.47222, 16.5,  16.52778, 16.55556, 16.58333, 16.61111, 16.63889, 16.66667, 16.69444,
                              16.72222, 16.75, 16.77778, 16.80556, 16.83333, 16.86111, 16.88889,  16.91667, 16.94444, 16.97222, 17, 17.02778, 17.05556, 17.08333, 17.11111,
                              17.13889, 17.16667, 17.19444, 17.22222, 17.25, 17.27778, 17.30556,  17.33333, 17.36111, 17.38889, 17.41667, 17.44444, 17.47222, 17.5,
                              17.52778, 17.55556, 17.58333, 17.61111, 17.63889, 17.66667, 17.69444,  17.72222, 17.75, 17.77778, 17.80556, 17.83333, 17.86111, 17.88889,
                              17.91667, 17.94444, 17.97222, 18, 18.02778, 18.05556, 18.08333, 18.11111,  18.13889, 18.16667, 18.19444, 18.22222, 18.25, 18.27778, 18.30556,
                              18.33333, 18.36111, 18.38889, 18.41667, 18.44444, 18.47222, 18.5,  18.52778, 18.55556, 18.58333, 18.61111, 18.63889, 18.66667, 18.69444,
                              18.72222, 18.75, 18.77778, 18.80556, 18.83333, 18.86111, 18.88889,  18.91667, 18.94444, 18.97222, 19, 19.02778, 19.05556, 19.08333, 19.11111,
                              19.13889, 19.16667, 19.19444, 19.22222, 19.25, 19.27778, 19.30556,  19.33333, 19.36111, 19.38889, 19.41667, 19.44444, 19.47222, 19.5,
                              19.52778, 19.55556, 19.58333, 19.61111, 19.63889, 19.66667, 19.69444,  19.72222, 19.75, 19.77778, 19.80556, 19.83333, 19.86111, 19.88889,
                              19.91667, 19.94444, 19.97222, 20, 20.02778, 20.05556, 20.08333, 20.11111,  20.13889, 20.16667, 20.19444, 20.22222, 20.25, 20.27778, 20.30556,
                              20.33333, 20.36111, 20.38889, 20.41667, 20.44444, 20.47222, 20.5,  20.52778, 20.55556, 20.58333, 20.61111, 20.63889, 20.66667, 20.69444,
                              20.72222, 20.75, 20.77778, 20.80556, 20.83333, 20.86111, 20.88889,  20.91667, 20.94444, 20.97222, 21, 21.02778, 21.05556, 21.08333, 21.11111,
                              21.13889, 21.16667, 21.19444, 21.22222, 21.25, 21.27778, 21.30556,  21.33333, 21.36111, 21.38889, 21.41667, 21.44444, 21.47222, 21.5,
                              21.52778, 21.55556, 21.58333, 21.61111, 21.63889, 21.66667, 21.69444,  21.72222, 21.75, 21.77778, 21.80556, 21.83333, 21.86111, 21.88889,
                              21.91667, 21.94444, 21.97222, 22, 22.02778, 22.05556, 22.08333, 22.11111,  22.13889, 22.16667, 22.19444, 22.22222, 22.25, 22.27778, 22.30556,
                              22.33333, 22.36111, 22.38889, 22.41667, 22.44444, 22.47222, 22.5,  22.52778, 22.55556, 22.58333, 22.61111, 22.63889, 22.66667, 22.69444,
                              22.72222, 22.75, 22.77778, 22.80556, 22.83333, 22.86111, 22.88889,  22.91667, 22.94444, 22.97222, 23, 23.02778, 23.05556, 23.08333, 23.11111,
                              23.13889, 23.16667, 23.19444, 23.22222, 23.25, 23.27778, 23.30556,  23.33333, 23.36111, 23.38889, 23.41667, 23.44444, 23.47222, 23.5,
                              23.52778, 23.55556, 23.58333, 23.61111, 23.63889, 23.66667, 23.69444,  23.72222, 23.75, 23.77778, 23.80556, 23.83333, 23.86111, 23.88889,
                              23.91667, 23.94444, 23.97222, 24 ])*3600.0
        les_shf = np.array([1.634431, 1.634431, 1.79452, 1.909038, 1.987758, 2.042218,
                            2.081862, 2.107408, 2.120582, 2.114897, 2.080142, 2.003892, 1.880542, 1.722328, 1.544722, 1.356961, 1.185304, 1.048697, 0.9521033, 0.8862211,
                            0.8449215, 0.8231402, 0.8216321, 0.8359701, 0.8623912, 0.9021153, 0.9522318, 1.009626, 1.069521, 1.129689, 1.187392, 1.242764, 1.295586,
                            1.345358, 1.391816, 1.436693, 1.480308, 1.522063, 1.56181, 1.599497, 1.634794, 1.668318, 1.700917, 1.733594, 1.765931, 1.798058, 1.829891,
                            1.860802, 1.890488, 1.918676, 1.945451, 1.97041, 1.994401, 2.018089,2.041842, 2.0654, 2.089087, 2.111267, 2.133322, 2.155273, 2.176622,
                            2.197964, 2.218805, 2.239033, 2.259177, 2.278551, 2.297817, 2.317205, 2.335516, 2.354051, 2.373094, 2.392959, 2.413882, 2.435321, 2.456663,
                            2.477272, 2.496835, 2.515525, 2.534206, 2.553178, 2.572371, 2.591444, 2.610397, 2.629738, 2.648828, 2.668193, 2.686878, 2.704977, 2.723071,
                            2.74062, 2.758249, 2.775855, 2.793381, 2.810968, 2.82853, 2.845749, 2.863355, 2.881758, 2.901238, 2.921374, 2.941951, 2.963069, 2.986204,
                            3.011251, 3.037213, 3.061997, 3.086073, 3.107474, 3.127269, 3.143203, 3.155811, 3.167178, 3.177665, 3.187873, 3.197936, 3.208085, 3.218326,
                            3.229058, 3.240333, 3.252007, 3.264657, 3.278037, 3.292397, 3.307385, 3.32224, 3.336803, 3.350777, 3.364536, 3.3794, 3.39604, 3.413329,
                            3.430218, 3.447373, 3.464916, 3.481149, 3.494637, 3.506901, 3.519053, 3.531336, 3.544381, 3.557632, 3.570166, 3.58236, 3.594043, 3.605765,
                            3.617468, 3.628335, 3.638835, 3.648117, 3.656043, 3.662935, 3.669211, 3.675105, 3.681086, 3.687702, 3.695244, 3.703983, 3.714075, 3.725419,
                            3.7367, 3.74733, 3.757429, 3.767954, 3.779252, 3.791089, 3.803895, 3.817291, 3.831483, 3.847074, 3.862719, 3.877701, 3.891222, 3.903064,
                            3.913483, 3.923044, 3.931121, 3.93856, 3.945792, 3.953758, 3.962597, 3.972297, 3.983008, 3.994182, 4.005928, 4.018024, 4.030252, 4.042444,
                            4.054478, 4.065949, 4.076443, 4.085856, 4.095154, 4.105059, 4.116054, 4.128097, 4.140843, 4.154712, 4.170597, 4.187395, 4.201637, 4.21317,
                            4.22296, 4.230227, 4.235349, 4.241577, 4.249332, 4.257581, 4.264964, 4.271679, 4.278381, 4.285117, 4.291739, 4.298341, 4.304749, 4.310658,
                            4.315855, 4.320012, 4.323587, 4.327088, 4.331383, 4.337102, 4.344682, 4.35385, 4.365065, 4.379044, 4.39495, 4.411206, 4.424986, 4.434377,
                            4.440598, 4.444909, 4.44837, 4.451554, 4.454647, 4.457855, 4.461133, 4.464486, 4.468227, 4.472092, 4.476064, 4.48033, 4.484643, 4.489188,
                            4.493729, 4.497972, 4.501735, 4.505696, 4.510359, 4.516641, 4.525282, 4.535996, 4.548131, 4.560594, 4.573325, 4.586694, 4.600972, 4.61566,
                            4.63028, 4.645144, 4.660759, 4.67639, 4.688473, 4.694128, 4.694682, 4.694312, 4.695217, 4.697851, 4.701424, 4.705014, 4.707944, 4.710546,
                            4.713297, 4.716175, 4.719152, 4.721634, 4.723565, 4.725155, 4.72665, 4.728536, 4.731249, 4.735344, 4.740477, 4.746492, 4.752718, 4.756315,
                            4.757022, 4.757632, 4.759938, 4.764208, 4.770625, 4.778108, 4.785649, 4.792824, 4.799726, 4.80578, 4.811276, 4.816344, 4.820308, 4.822897,
                            4.824687, 4.826346, 4.828174, 4.830565, 4.833576, 4.837051, 4.84118, 4.846497, 4.852745, 4.859675, 4.867013, 4.875052, 4.883379, 4.892065,
                            4.901244, 4.910941, 4.920598, 4.929197, 4.936514, 4.942663, 4.948359, 4.953713, 4.957819, 4.959922, 4.959975, 4.959262, 4.958376, 4.958676,
                            4.960652, 4.963802, 4.967896, 4.972949, 4.978578, 4.984569, 4.99032, 4.996207, 5.002171, 5.007968, 5.012641, 5.016554, 5.020069, 5.024059,
                            5.028557, 5.033698, 5.039052, 5.044362, 5.04946, 5.054722, 5.060261, 5.066553, 5.073889, 5.08216, 5.09047, 5.098004, 5.104412, 5.109458,
                            5.11325, 5.116282, 5.119227, 5.122694, 5.127189, 5.131632, 5.134293, 5.135016, 5.135044, 5.135504, 5.136415, 5.137763, 5.140181, 5.143372,
                            5.14718, 5.151512, 5.155872, 5.160148, 5.165023, 5.171045, 5.177503, 5.182269, 5.182644, 5.179191, 5.17591, 5.174685, 5.176522, 5.180526,
                            5.185416, 5.191361, 5.199033, 5.207997, 5.217803, 5.228569, 5.23986, 5.251144, 5.261673, 5.270413, 5.278263, 5.285948, 5.293543, 5.300558,
                            5.307538, 5.314994, 5.322587, 5.329451, 5.335275, 5.340576, 5.345736, 5.350441, 5.354101, 5.356206, 5.356946, 5.35722, 5.357856, 5.359034,
                            5.360602, 5.362178, 5.363319, 5.364271, 5.365369, 5.366634, 5.367927, 5.369032, 5.370096, 5.371151, 5.372629, 5.375333, 5.379229, 5.383252,
                            5.387007, 5.390365, 5.393364, 5.396031, 5.398673, 5.401186, 5.403506, 5.405801, 5.40839, 5.412143, 5.416846, 5.421615, 5.42633, 5.430865,
                            5.435502, 5.440095, 5.443929, 5.446815, 5.448782, 5.45028, 5.451114, 5.451939, 5.453733, 5.457012, 5.461453, 5.466185, 5.47083, 5.475437,
                            5.479887, 5.483892, 5.48724, 5.489774, 5.491846, 5.493781, 5.496045, 5.498553, 5.50073, 5.502481, 5.503853, 5.504962, 5.506061, 5.507179,
                            5.508607, 5.510784, 5.513527, 5.516364, 5.518565, 5.51858, 5.516057, 5.512893, 5.511178, 5.511426, 5.51037, 5.506244, 5.500928, 5.497112,
                            5.496271, 5.497959, 5.501408, 5.505744, 5.509729, 5.513041, 5.514598, 5.51402, 5.512156, 5.50984, 5.508637, 5.509088, 5.511622, 5.516383,
                            5.521658, 5.526398, 5.530559, 5.534876, 5.539259, 5.543005, 5.545922, 5.548956, 5.554169, 5.563324, 5.57585, 5.589678, 5.602908, 5.613461,
                            5.621253, 5.627852, 5.634678, 5.642333, 5.650508, 5.657955, 5.663526, 5.666877, 5.66806, 5.667863, 5.666154, 5.662537, 5.656762, 5.650253,
                            5.644118, 5.640198, 5.639241, 5.64108, 5.645429, 5.652122, 5.66151, 5.671721, 5.680774, 5.684882, 5.688179, 5.693891, 5.701779, 5.711828,
                            5.725646, 5.727399, 5.702496, 5.691324, 5.688145, 5.685778, 5.686693, 5.690804, 5.70736, 5.718785, 5.689218, 5.659789, 5.644348, 5.638592,
                            5.638037, 5.640782, 5.644202, 5.647502, 5.649586, 5.649906, 5.649776, 5.649739, 5.64965, 5.649291, 5.648512, 5.647834, 5.648341, 5.650343,
                            5.652225, 5.653803, 5.656147, 5.659467, 5.663828, 5.669117, 5.67584, 5.684757, 5.696943, 5.711311, 5.725071, 5.734146, 5.742591, 5.752276,
                            5.762705, 5.774479, 5.786627, 5.796857, 5.80556, 5.813916, 5.822899, 5.832396, 5.841979, 5.852942, 5.865793, 5.88127, 5.89888, 5.916802,
                            5.931684, 5.940957, 5.939208, 5.929018, 5.919448, 5.91337, 5.911106, 5.910784, 5.91156, 5.913368, 5.91581, 5.91863, 5.921325, 5.922926,
                            5.922999, 5.922737, 5.923411, 5.925576, 5.928063, 5.929002, 5.928725, 5.928051, 5.927358, 5.926435, 5.925511, 5.92631, 5.930225, 5.938534,
                            5.951222, 5.962683, 5.967409, 5.972044, 5.977255, 5.978479, 5.973461, 5.965891, 5.95805, 5.949652, 5.941849, 5.935129, 5.929466, 5.925043,
                            5.921345, 5.917893, 5.916094, 5.916503, 5.918301, 5.919901, 5.920828, 5.922056, 5.923494, 5.92522, 5.927212, 5.929893, 5.935143, 5.942149,
                            5.948834, 5.953499, 5.957002, 5.960819, 5.964654, 5.966921, 5.969021, 5.971401, 5.97562, 5.982995, 5.993459, 6.007733, 6.026596, 6.050559,
                            6.078705, 6.096413, 6.095231, 6.082922, 6.070377, 6.054824, 6.035952, 6.020619, 6.009626, 5.998487, 5.984032, 5.967141, 5.94909, 5.931829,
                            5.916249, 5.901451, 5.888156, 5.876653, 5.867305, 5.861281, 5.85989, 5.862917, 5.868137, 5.872825, 5.877087, 5.881156, 5.885043, 5.889254,
                            5.893681, 5.897377, 5.901439, 5.906103, 5.912044, 5.920357, 5.930196, 5.939063, 5.946259, 5.950891, 5.950498, 5.94681, 5.942304, 5.938737,
                            5.937853, 5.940165, 5.94515, 5.949762, 5.953689, 5.960344, 5.972229, 5.987456, 6.003677, 6.019004, 6.032909, 6.04563, 6.056892, 6.066408,
                            6.0744, 6.080966, 6.087443, 6.094278, 6.100101, 6.104059, 6.106279, 6.107409, 6.108473, 6.109149, 6.098575, 6.089839, 6.088602, 6.093524,
                            6.102939, 6.114849, 6.127265, 6.137537, 6.145335, 6.152123, 6.158554, 6.1633, 6.165883, 6.164651, 6.159179, 6.150803, 6.140064, 6.129086, 6.119238, 6.109319, 6.099386, 6.091144, 6.086633, 6.086, 6.086819,
                            6.075517, 6.070455, 6.091551, 6.155096, 6.201097, 6.244498, 6.265624, 6.271212, 6.264979, 6.250254, 6.22903, 6.205024, 6.182512, 6.162072,
                            6.139987, 6.115946, 6.089553, 6.063132, 6.038797, 6.015534, 5.99143, 5.966059, 5.942408, 5.922392, 5.906565, 5.897608, 5.8975, 5.904302,
                            5.914238, 5.926449, 5.935513, 5.922378, 5.912706, 5.921204, 5.948195, 5.986585, 6.034463, 6.032088, 6.032321, 6.039231, 6.03642, 6.046103,
                            6.08737, 6.095437, 6.066596, 6.018292, 5.970625, 5.93797, 5.944443, 5.97424, 6.032163, 6.256334, 6.39761, 6.478774, 6.511418, 6.543766,
                            6.593176, 6.650991, 6.716045, 6.721952, 6.686002, 6.634602, 6.571249, 6.504817, 6.447561, 6.396264, 6.351182, 6.313943, 6.286619, 6.262313,
                            6.229374, 6.189634, 6.159786, 6.149427, 6.160448, 6.18576, 6.209991, 6.243931, 6.33601, 6.406002, 6.448211, 6.43473, 6.381226, 6.33174,
                            6.347017, 6.454982, 6.587361, 6.743252, 6.908476, 7.074169, 7.227061, 7.338344, 7.416354, 7.459417, 7.466571, 7.434535, 7.376985, 7.343222,
                            7.341239, 7.353554, 7.32847, 7.260763, 7.203727, 7.174184, 7.165051, 7.150801, 7.114815, 7.072447, 7.033031, 6.996824, 6.962577])

        les_lhf = np.array([198.2732, 198.2732, 187.617, 178.7758, 171.6557, 166.0201, 161.4717, 158.2624, 156.214, 155.7761, 157.3714, 161.2656, 167.2728,
                            174.3094, 181.3316, 187.7534, 192.7522, 196.0927, 197.9232, 198.5954, 198.4118, 197.5412, 196.0658, 194.1777, 192.0608, 189.8055, 187.4846,
                            185.102, 182.7596, 180.4751, 178.3501, 176.4082, 174.6644, 173.1485, 171.8754, 170.79, 169.8685, 169.112, 168.5447, 168.1797, 168.0182,
                            168.0176, 168.1247, 168.2925, 168.4748, 168.6566, 168.8238, 168.9711, 169.0943, 169.1997, 169.304, 169.4048, 169.4862, 169.5373, 169.5424,
                            169.4644, 169.3031, 169.0852, 168.8172, 168.5113, 168.1717, 167.7709, 167.3362, 166.8933, 166.4291, 165.957, 165.4703, 164.9503, 164.4259,
                            163.886, 163.353, 162.8304, 162.3167, 161.8187, 161.3226, 160.843, 160.4107, 160.0301, 159.6883, 159.3697, 159.0699, 158.7979, 158.5556,
                            158.3249, 158.0992, 157.8698, 157.6438, 157.4144, 157.1611, 156.9048, 156.6235, 156.3101, 156.0104, 155.7413, 155.4894, 155.2519, 155.0273,
                            154.8119, 154.6105, 154.4345, 154.279, 154.1374, 154.0115, 153.9069, 153.8234, 153.7493, 153.6602, 153.535, 153.3572, 153.1241, 152.8242,
                            152.4689, 152.0916, 151.7206, 151.3895, 151.1007, 150.8386, 150.6043, 150.3836, 150.1664, 149.9359, 149.7068, 149.4892, 149.2886, 149.1093,
                            148.9424, 148.7923, 148.6661, 148.5673, 148.4981, 148.4424, 148.3871, 148.3206, 148.2338, 148.1346, 148.0303, 147.8997, 147.7356, 147.5493,
                            147.3397, 147.1232, 146.9218, 146.7476, 146.605, 146.4967, 146.44, 146.4233, 146.4076, 146.362, 146.2741, 146.1584, 146.0092, 145.831,
                            145.6448, 145.4661, 145.2886, 145.1256, 144.9754, 144.8271, 144.6891, 144.5723, 144.4781, 144.3804, 144.2507, 144.077, 143.8648, 143.6448,
                            143.4359, 143.2439, 143.0795, 142.9333, 142.8089, 142.7034, 142.6273, 142.5808, 142.5567, 142.5447, 142.533, 142.5096, 142.4701, 142.4088,
                            142.3193, 142.221, 142.1232, 142.0209, 141.8991, 141.7661, 141.628, 141.4744, 141.3034, 141.1051, 140.877, 140.6301, 140.3757, 140.1257,
                            139.8896, 139.6839, 139.5162, 139.3981, 139.3511, 139.3616, 139.4108, 139.4876, 139.5384, 139.5272, 139.4509, 139.326, 139.1714, 139.0087,
                            138.853, 138.7204, 138.6167, 138.5302, 138.4528, 138.3723, 138.2871, 138.2013, 138.1069, 138.0078, 137.9119, 137.8226, 137.731, 137.6364,
                            137.5542, 137.487, 137.4415, 137.4221, 137.408, 137.3788, 137.3234, 137.2467, 137.1496, 137.0398, 136.9442, 136.8666, 136.7918, 136.7137,
                            136.6328, 136.551, 136.4674, 136.3857, 136.3144, 136.2581, 136.2123, 136.1717, 136.1319, 136.0889, 136.0522, 136.0344, 136.0502, 136.0944,
                            136.1475, 136.1844, 136.1983, 136.2023, 136.2113, 136.2392, 136.2855, 136.3453, 136.4226, 136.5295, 136.6598, 136.7881, 136.8791, 136.9247,
                            136.9267, 136.888, 136.8077, 136.6902, 136.5442, 136.3651, 136.1604, 135.96, 135.7852, 135.6406, 135.522, 135.4322, 135.3857, 135.3897,
                            135.4174, 135.4311, 135.4049, 135.3457, 135.28, 135.2375, 135.2106, 135.1879, 135.1588, 135.122, 135.0784, 135.0396, 135.0164, 135.0114,
                            135.0311, 135.0701, 135.1172, 135.1615, 135.2095, 135.259, 135.2883, 135.2865, 135.2448, 135.1715, 135.0882, 135.0277, 135.0148, 135.0393,
                            135.078, 135.1182, 135.154, 135.1798, 135.1956, 135.198, 135.2003, 135.2164, 135.2447, 135.274, 135.2966, 135.3189, 135.3647, 135.4373,
                            135.5052, 135.5394, 135.5245, 135.4862, 135.4599, 135.4427, 135.4285,135.4226, 135.4315, 135.4537, 135.4814, 135.5093, 135.5305, 135.5428,
                            135.5375, 135.5193, 135.4897, 135.4548, 135.4161, 135.3642, 135.3045, 135.2511, 135.2084, 135.1809, 135.1811, 135.2186, 135.2871, 135.3683,
                            135.4534, 135.5353, 135.5877, 135.582, 135.5111, 135.3988, 135.2706, 135.1551, 135.0618, 134.992, 134.9655, 135.0151, 135.1447, 135.3213,
                            135.5032, 135.6691, 135.8072, 135.9203, 136.0238, 136.1207, 136.199, 136.2547, 136.2899, 136.3084, 136.3286, 136.3634, 136.4225, 136.5291,
                            136.6761, 136.8162, 136.9219, 136.9925, 137.0548, 137.1017, 137.1071, 137.0671, 136.9977, 136.9179, 136.8649, 136.8638, 136.9092, 136.9726,
                            137.0357, 137.0841, 137.1127, 137.122, 137.1322, 137.1657, 137.2325, 137.3238, 137.4304, 137.5363, 137.6149, 137.6659, 137.6961, 137.7129,
                            137.7129, 137.6895, 137.6544, 137.6254, 137.6087, 137.5893, 137.5553, 137.5097, 137.4574, 137.4011, 137.3543, 137.3253, 137.3099, 137.3159,
                            137.3543, 137.4117, 137.4669, 137.5194, 137.5704, 137.6189, 137.6677, 137.7218, 137.7844, 137.8646, 137.9689, 138.0835, 138.198, 138.3087,
                            138.407, 138.487, 138.5445, 138.5873, 138.6135, 138.6292, 138.6652, 138.7453, 138.8596, 138.9994, 139.1463, 139.287, 139.4014, 139.5,
                            139.597, 139.7069, 139.8358, 139.979, 140.1259, 140.2648, 140.3696, 140.4384, 140.4844, 140.5264, 140.5817, 140.6518, 140.7301, 140.8077,
                            140.8727, 140.9294, 140.9824, 141.0312, 141.0669, 141.0807, 141.0873, 141.11, 141.1649, 141.2618, 141.382, 141.4967, 141.5873, 141.6653,
                            141.7455, 141.8351, 141.9585, 142.1176, 142.2665, 142.3751, 142.4302, 142.43, 142.3777, 142.3015, 142.2227, 142.1533, 142.0966, 142.0529,
                            142.0287, 142.0224, 142.0194, 142.0102, 141.9923, 141.9775, 141.9806, 142.0217, 142.1228, 142.2766, 142.4484, 142.5979, 142.6908, 142.7279,
                            142.7255, 142.7198, 142.7429, 142.8048, 142.9061, 143.0282, 143.1577, 143.2847, 143.4103, 143.5434, 143.6784, 143.8127, 143.9496, 144.0983,
                            144.2634, 144.4456, 144.6652, 144.9104, 145.147, 145.3311, 145.4687, 145.5869, 145.7008, 145.7929, 145.8407, 145.8518, 145.8418, 145.8084,
                            145.7603, 145.7659, 145.7509, 145.6625, 145.5165, 145.3623, 145.3277, 145.9071, 146.6592, 147.1709, 147.4716, 147.624, 147.7153, 147.8239,
                            148.005, 148.556, 149.1108, 149.4541, 149.6095, 149.6221, 149.5076, 149.3065, 149.0674, 148.8078, 148.5438, 148.2983, 148.0919, 147.9006,
                            147.7105, 147.5231, 147.3383, 147.1679, 147.041, 146.9482, 146.8634, 146.7841, 146.7077, 146.6204, 146.516, 146.4125, 146.3402, 146.3197,
                            146.3417, 146.4095, 146.5569, 146.7678, 146.9594, 147.1068, 147.242, 147.3801, 147.5009, 147.5983, 147.6853, 147.7758, 147.8788, 148.0141,
                            148.1731, 148.3511, 148.5533, 148.791, 149.0558, 149.3268, 149.6006, 149.849, 150.0925, 150.2669, 150.3671, 150.4428, 150.5135, 150.5824,
                            150.6536, 150.7193, 150.7651, 150.7822, 150.7701, 150.7282, 150.6655, 150.6123, 150.5954, 150.6044, 150.6119, 150.6077, 150.6238, 150.6782,
                            150.7509, 150.8154, 150.8821, 150.9504, 151.0033, 151.0268, 151.0374, 151.097, 151.2753, 151.4798, 151.6887, 151.9253, 152.1928, 152.4481,
                            152.6525, 152.7903, 152.8642, 152.8884, 152.8782, 152.8503, 152.8092, 152.7446, 152.6633, 152.5776, 152.4996, 152.4444, 152.4228, 152.4577,
                            152.5704, 152.73, 152.9306, 153.1704, 153.4156, 153.6546, 153.9146, 154.1875, 154.4661, 154.7303, 154.9793, 155.2033, 155.3997, 155.5776,
                            155.767, 155.9701, 156.1698, 156.3417, 156.4498, 156.4986, 156.5029, 156.65, 157.06, 157.5635, 158.0533, 158.5487, 159.0369, 159.4742,
                            159.8432, 160.1118, 160.266, 160.3189, 160.3071, 160.2713, 160.2035, 160.085, 159.9198, 159.7107, 159.4349, 159.112, 158.7729, 158.4872,
                            158.3049, 158.2312, 158.2272, 158.2541, 158.2923, 158.3429, 158.4081, 158.4857, 158.553, 158.5903, 158.6024, 158.6273, 158.6895, 158.7608,
                            158.7997, 158.8257, 158.8448, 158.8031, 158.7126, 158.6109, 158.5236, 158.4557, 158.3835, 158.2827, 158.1496, 158.0014, 157.8703, 157.7762,
                            157.7395, 157.7884, 157.9228, 158.1111, 158.3146, 158.494, 158.6381, 158.7542, 158.8445, 158.9147, 158.9877, 159.0961, 159.2823, 159.5494,
                            159.8551, 160.2241, 160.7043, 161.0569, 161.3003, 161.499, 161.6627, 161.785, 161.8553, 161.8553, 161.8156, 161.796, 161.8292, 161.9117,
                            162.019, 162.1163, 162.1993, 162.2772, 162.3245, 162.3331, 162.3099, 162.2787, 162.2716, 162.2946, 162.3391, 162.3909, 162.5013, 162.747,
                            163.0345, 163.374, 163.9682, 164.6452, 165.2902, 165.8638, 166.3394, 166.7262, 167.1093, 167.5132, 167.8639, 168.0681, 168.0942, 167.9653,
                            167.7632, 167.5396, 167.2687, 166.952, 166.6481, 166.43, 166.3055, 166.2063, 166.0822, 165.9192, 165.7349, 165.5399, 165.3562, 165.2039,
                            165.0949, 165.1007, 165.3847, 165.6573, 165.877, 166.1264, 166.4306, 167.2922, 168.374, 169.2942, 170.1339, 171.0091, 171.818, 172.5518,
                            173.1673, 173.7262, 174.2814, 174.8056, 175.2886, 175.7793, 176.2045, 176.5777, 177.1588, 177.8094, 178.6233, 179.3018, 179.5905, 179.5568,
                            179.3611, 179.1021, 178.9272, 178.9231, 178.822, 178.4735, 177.9464, 177.2966, 176.5565, 175.7435, 174.933, 174.1767, 173.457, 173.0059,
                            172.7478, 172.4327, 172.0199, 171.6144, 171.3349, 171.1634, 171.0582, 171.139, 171.4761, 171.9454, 172.1351, 172.1058, 171.9492, 171.8677,
                            172.3948, 173.1041, 173.6964, 174.3968, 174.9307, 175.364, 175.5471, 175.4187, 175.1593, 174.8242, 174.4916, 174.032, 173.414, 172.711,
                            172.1147, 171.789, 171.5314, 171.1781, 170.774, 170.4526, 170.4128, 170.6267, 170.8727, 171.049, 171.116, 171.1088])
        self.Sur.lhf = np.interp(TS.t, les_time, les_lhf)
        self.Sur.shf = np.interp(TS.t, les_time, les_shf)
        print(self.Sur.lhf, self.Sur.shf, TS.t)
        self.Sur.update(GMV)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return


cdef class TRMM_LBA(CasesBase):
    # adopted from: "Daytime convective development over land- A model intercomparison based on LBA observations",
    # By Grabowski et al (2006)  Q. J. R. Meteorol. Soc. 132 317-344
    def __init__(self, paramlist):
        self.casename = 'TRMM_LBA'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard() # it was forcing standard
        self.inversion_option = 'thetal_maxgrad'
        self.Fo.apply_coriolis = False
        self.Fo.apply_subsidence = False
        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 991.3*100  #Pressure at ground
        Ref.Tg = 296.85   # surface values for reference state (RS) which outputs p0 rho0 alpha0
        pvg = pv_star(Ref.Tg)
        Ref.qtg = eps_v * pvg/(Ref.Pg - pvg)#Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:

            double [:] p1 = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # TRMM_LBA inputs from Grabowski et al. 2006
        z_in = np.array([0.130,  0.464,  0.573,  1.100,  1.653,  2.216,  2.760,
                         3.297,  3.824,  4.327,  4.787,  5.242,  5.686,  6.131,
                         6.578,  6.996,  7.431,  7.881,  8.300,  8.718,  9.149,
                         9.611, 10.084, 10.573, 11.008, 11.460, 11.966, 12.472,
                        12.971, 13.478, 13.971, 14.443, 14.956, 15.458, 16.019,
                        16.491, 16.961, 17.442, 17.934, 18.397, 18.851, 19.331,
                        19.809, 20.321, 20.813, 21.329, 30.000]) * 1000 - 130.0 #LES z is in meters

        p_in = np.array([991.3, 954.2, 942.0, 886.9, 831.5, 778.9, 729.8,
                         684.0, 641.7, 603.2, 570.1, 538.6, 509.1, 480.4,
                         454.0, 429.6, 405.7, 382.5, 361.1, 340.9, 321.2,
                         301.2, 281.8, 263.1, 246.1, 230.1, 213.2, 197.0,
                         182.3, 167.9, 154.9, 143.0, 131.1, 119.7, 108.9,
                         100.1,  92.1,  84.6,  77.5,  71.4,  65.9,  60.7,
                          55.9,  51.3,  47.2,  43.3,  10.3]) * 100 # LES pres is in pasc

        T_in = np.array([23.70,  23.30,  22.57,  19.90,  16.91,  14.09,  11.13,
                          8.29,   5.38,   2.29,  -0.66,  -3.02,  -5.28,  -7.42,
                        -10.34, -12.69, -15.70, -19.21, -21.81, -24.73, -27.76,
                        -30.93, -34.62, -38.58, -42.30, -46.07, -50.03, -54.67,
                        -59.16, -63.60, -67.68, -70.77, -74.41, -77.51, -80.64,
                        -80.69, -80.00, -81.38, -81.17, -78.32, -74.77, -74.52,
                        -72.62, -70.87, -69.19, -66.90, -66.90]) + 273.15 # LES T is in deg K

        RH_in = np.array([98.00,  86.00,  88.56,  87.44,  86.67,  83.67,  79.56,
                          84.78,  84.78,  89.33,  94.33,  92.00,  85.22,  77.33,
                          80.11,  66.11,  72.11,  72.67,  52.22,  54.67,  51.00,
                          43.78,  40.56,  43.11,  54.78,  46.11,  42.33,  43.22,
                          45.33,  39.78,  33.78,  28.78,  24.67,  20.67,  17.67,
                          17.11,  16.22,  14.22,  13.00,  13.00,  12.22,   9.56,
                           7.78,   5.89,   4.33,   3.00,   3.00])

        u_in = np.array([0.00,   0.81,   1.17,   3.44,   3.53,   3.88,   4.09,
                         3.97,   1.22,   0.16,  -1.22,  -1.72,  -2.77,  -2.65,
                        -0.64,  -0.07,  -1.90,  -2.70,  -2.99,  -3.66,  -5.05,
                        -6.64,  -4.74,  -5.30,  -6.07,  -4.26,  -7.52,  -8.88,
                        -9.00,  -7.77,  -5.37,  -3.88,  -1.15,  -2.36,  -9.20,
                        -8.01,  -5.68,  -8.83, -14.51, -15.55, -15.36, -17.67,
                       -17.82, -18.94, -15.92, -15.32, -15.32])

        v_in = np.array([-0.40,  -3.51,  -3.88,  -4.77,  -5.28,  -5.85,  -5.60,
                         -2.67,  -1.47,   0.57,   0.89,  -0.08,   1.11,   2.15,
                          3.12,   3.22,   3.34,   1.91,   1.15,   1.01,  -0.57,
                         -0.67,   0.31,   2.97,   2.32,   2.66,   4.79,   3.40,
                          3.14,   3.93,   7.57,   2.58,   2.50,   6.44,   6.84,
                          0.19,  -2.20,  -3.60,   0.56,   6.68,   9.41,   7.03,
                          5.32,   1.14,  -0.65,   5.27,   5.27])
        # interpolate to the model grid-points

        p1 = np.interp(Gr.z_half,z_in,p_in)
        GMV.U.values = np.interp(Gr.z_half,z_in,u_in)
        GMV.V.values = np.interp(Gr.z_half,z_in,v_in)

        # get the entropy from RH, p, T
        RH = np.zeros(Gr.nzg)
        RH[Gr.gw:Gr.nzg-Gr.gw] = np.interp(Gr.z_half[Gr.gw:Gr.nzg-Gr.gw],z_in,RH_in)
        RH[0] = RH[3]
        RH[1] = RH[2]
        RH[Gr.nzg-Gr.gw+1] = RH[Gr.nzg-Gr.gw-1]

        T = np.zeros(Gr.nzg)
        T[Gr.gw:Gr.nzg-Gr.gw] = np.interp(Gr.z_half[Gr.gw:Gr.nzg-Gr.gw],z_in,T_in)
        GMV.T.values = T
        theta_rho = RH*0.0
        epsi = 287.1/461.5
        cdef double PV_star # here pv_star is a function
        cdef double qv_star

        GMV.U.set_bcs(Gr)
        GMV.T.set_bcs(Gr)


        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            PV_star = pv_star(GMV.T.values[k])
            qv_star = PV_star*epsi/(p1[k]- PV_star + epsi*PV_star*RH[k]/100.0) # eq. 37 in pressel et al and the def of RH
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.QT.values[k] = qv_star*RH[k]/100.0
            if GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0)
            elif GMV.H.name == 'thetal':
                 GMV.H.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))

            GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))
            theta_rho[k] = theta_rho_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)

        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        #self.Sur.zrough = 1.0e-4 # not actually used, but initialized to reasonable value
        self.Sur.Tsurface = (273.15+23) * exner_c(Ref.Pg)
        self.Sur.qsurface = 22.45e-3 # kg/kg
        self.Sur.lhf = 5.2e-5 * Ref.rho0[Gr.gw -1] * latent_heat(self.Sur.Tsurface)
        self.Sur.shf = 8.0e-3 * cpm_c(self.Sur.qsurface) * Ref.rho0[Gr.gw-1]
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize()

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        self.Fo.dTdt = np.zeros(Gr.nzg, dtype=np.double)
        self.rad_time = np.linspace(10,360,36)*60
        z_in         = np.array([42.5, 200.92, 456.28, 743, 1061.08, 1410.52, 1791.32, 2203.48, 2647,3121.88, 3628.12,
                                 4165.72, 4734.68, 5335, 5966.68, 6629.72, 7324.12,
                                 8049.88, 8807, 9595.48, 10415.32, 11266.52, 12149.08, 13063, 14008.28,
                                 14984.92, 15992.92, 17032.28, 18103, 19205.08, 20338.52, 21503.32, 22699.48])
        rad_in   = np.array([[-1.386, -1.927, -2.089, -1.969, -1.805, -1.585, -1.406, -1.317, -1.188, -1.106, -1.103, -1.025,
                              -0.955, -1.045, -1.144, -1.119, -1.068, -1.092, -1.196, -1.253, -1.266, -1.306,  -0.95,  0.122,
                               0.255,  0.258,  0.322,  0.135,      0,      0,      0,      0,      0],
                             [ -1.23, -1.824, -2.011, -1.895, -1.729, -1.508, -1.331, -1.241, -1.109, -1.024, -1.018,  -0.94,
                              -0.867, -0.953, -1.046, -1.018, -0.972, -1.006, -1.119, -1.187, -1.209, -1.259, -0.919,  0.122,
                               0.264,  0.262,  0.326,  0.137,      0,      0,      0,      0,     0],
                             [-1.043, -1.692, -1.906, -1.796,  -1.63,  -1.41, -1.233, -1.142,  -1.01,  -0.92, -0.911, -0.829,
                              -0.754, -0.837, -0.923,  -0.89, -0.847, -0.895, -1.021, -1.101, -1.138, -1.201,  -0.88,  0.131,
                               0.286,  0.259,  0.332,   0.14,      0,      0,      0,      0,      0],
                             [-0.944, -1.613, -1.832,  -1.72, -1.555, -1.339, -1.163, -1.068, -0.935, -0.846, -0.835,  -0.75,
                              -0.673, -0.751, -0.833, -0.798,  -0.76, -0.817, -0.952, -1.042, -1.088, -1.159, -0.853,  0.138,
                               0.291,  0.265,  0.348,  0.136,      0,      0,      0,      0,      0],
                             [-0.833, -1.526, -1.757, -1.648, -1.485,  -1.27, -1.093, -0.998, -0.867, -0.778, -0.761, -0.672,
                              -0.594, -0.671, -0.748, -0.709, -0.676, -0.742, -0.887, -0.986, -1.041, -1.119, -0.825,  0.143,
                               0.296,  0.271,  0.351,  0.138,      0,      0,      0,      0,      0],
                             [-0.719, -1.425, -1.657,  -1.55, -1.392, -1.179, -1.003, -0.909, -0.778, -0.688, -0.667, -0.573,
                              -0.492, -0.566, -0.639, -0.596, -0.568, -0.647, -0.804, -0.914, -0.981,  -1.07, -0.793,  0.151,
                               0.303,  0.279,  0.355,  0.141,      0,      0,      0,      0,      0],
                             [-0.724, -1.374, -1.585, -1.482, -1.328, -1.116, -0.936, -0.842, -0.715, -0.624, -0.598, -0.503,
                              -0.421, -0.494, -0.561, -0.514,  -0.49,  -0.58, -0.745, -0.863, -0.938, -1.035, -0.764,  0.171,
                               0.291,  0.284,  0.358,  0.144,      0,      0,      0,      0,      0],
                             [-0.587,  -1.28, -1.513, -1.416, -1.264, -1.052, -0.874, -0.781, -0.655, -0.561, -0.532, -0.436,
                              -0.354, -0.424, -0.485, -0.435, -0.417, -0.517, -0.691, -0.817, -0.898,     -1,  -0.74,  0.176,
                               0.297,  0.289,   0.36,  0.146,      0,      0,      0,      0,      0],
                             [-0.506, -1.194, -1.426, -1.332, -1.182, -0.972, -0.795, -0.704, -0.578,  -0.48, -0.445, -0.347,
                              -0.267, -0.336, -0.391, -0.337, -0.325, -0.436,  -0.62, -0.756, -0.847,  -0.96, -0.714,   0.18,
                               0.305,  0.317,  0.348,  0.158,      0,      0,      0,      0,      0],
                             [-0.472,  -1.14, -1.364, -1.271, -1.123, -0.914, -0.738, -0.649, -0.522, -0.422, -0.386, -0.287,
                              -0.207, -0.273, -0.322, -0.267,  -0.26, -0.379, -0.569, -0.712, -0.811, -0.931, -0.696,  0.183,
                               0.311,   0.32,  0.351,   0.16,      0,      0,      0,      0,     0],
                             [-0.448, -1.091, -1.305, -1.214, -1.068, -0.858, -0.682, -0.594, -0.469, -0.368, -0.329, -0.229,
                              -0.149, -0.213, -0.257,   -0.2, -0.199, -0.327, -0.523, -0.668, -0.774, -0.903, -0.678,  0.186,
                               0.315,  0.323,  0.355,  0.162,      0,      0,      0,      0,      0],
                             [-0.405, -1.025, -1.228, -1.139, -0.996, -0.789, -0.615, -0.527, -0.402,   -0.3, -0.256, -0.156,
                              -0.077, -0.136, -0.173, -0.115, -0.121, -0.259, -0.463, -0.617, -0.732, -0.869, -0.656,   0.19,
                               0.322,  0.326,  0.359,  0.164,      0,      0,      0,      0,      0],
                             [-0.391, -0.983, -1.174, -1.085, -0.945, -0.739, -0.566, -0.478, -0.354, -0.251, -0.205, -0.105,
                              -0.027, -0.082, -0.114, -0.056, -0.069, -0.213,  -0.42, -0.579, -0.699,  -0.84, -0.642,  0.173,
                               0.327,  0.329,  0.362,  0.165,      0,      0,      0,      0,      0],
                             [-0.385, -0.946, -1.121, -1.032, -0.898, -0.695, -0.523, -0.434, -0.307, -0.203, -0.157, -0.057,
                               0.021, -0.031, -0.059, -0.001, -0.018, -0.168, -0.381, -0.546, -0.672, -0.819, -0.629,  0.176,
                               0.332,  0.332,  0.364,  0.166,      0,      0,      0,      0,      0],
                             [-0.383, -0.904, -1.063, -0.972, -0.834, -0.632, -0.464, -0.378, -0.251, -0.144, -0.096,  0.001,
                               0.079,  0.032,  0.011,  0.069,  0.044, -0.113, -0.332, -0.504, -0.637, -0.791, -0.611,  0.181,
                               0.338,  0.335,  0.367,  0.167,      0,      0,      0,      0,      0],
                             [-0.391, -0.873, -1.016, -0.929, -0.794, -0.591, -0.423, -0.337, -0.212, -0.104, -0.056,  0.043,
                               0.121,  0.077,  0.058,  0.117,  0.088, -0.075, -0.298, -0.475, -0.613, -0.772, -0.599,  0.183,
                               0.342,  0.337,   0.37,  0.168,      0,      0,      0,      0,      0],
                             [-0.359, -0.836, -0.976, -0.888, -0.755, -0.554, -0.386,   -0.3, -0.175, -0.067, -0.018,  0.081,
                                0.16,  0.119,  0.103,  0.161,  0.129, -0.039, -0.266, -0.448, -0.591, -0.755, -0.587,  0.187,
                               0.345,  0.339,  0.372,  0.169,      0,      0,      0,      0,     0],
                             [-0.328, -0.792, -0.928, -0.842, -0.709, -0.508, -0.341, -0.256, -0.131, -0.022,  0.029,  0.128,
                               0.208,   0.17,  0.158,  0.216,  0.179,  0.005, -0.228, -0.415, -0.564, -0.733, -0.573,   0.19,
                               0.384,  0.313,  0.375,   0.17,      0,      0,      0,      0,      0],
                             [-0.324, -0.767, -0.893, -0.807, -0.676, -0.476,  -0.31, -0.225, -0.101,  0.008,   0.06,  0.159,
                               0.239,  0.204,  0.195,  0.252,  0.212,  0.034, -0.203, -0.394, -0.546, -0.719, -0.564,  0.192,
                               0.386,  0.315,  0.377,  0.171,      0,      0,      0,      0,      0],
                             [ -0.31,  -0.74,  -0.86, -0.775, -0.647, -0.449, -0.283, -0.197, -0.073,  0.036,  0.089,  0.188,
                               0.269,  0.235,  0.229,  0.285,  0.242,  0.061, -0.179, -0.374,  -0.53, -0.706, -0.556,  0.194,
                               0.388,  0.317,  0.402,  0.158,      0,      0,      0,      0,      0],
                             [-0.244, -0.694, -0.818,  -0.73, -0.605, -0.415, -0.252, -0.163, -0.037,  0.072,  0.122,   0.22,
                               0.303,  0.273,  0.269,  0.324,  0.277,  0.093, -0.152,  -0.35,  -0.51, -0.691, -0.546,  0.196,
                               0.39,   0.32,  0.403,  0.159,      0,      0,      0,      0,      0],
                             [-0.284, -0.701, -0.803, -0.701, -0.568, -0.381, -0.225, -0.142, -0.017,  0.092,  0.143,  0.242,
                               0.325,  0.298,  0.295,   0.35,    0.3,  0.112, -0.134, -0.334, -0.497,  -0.68,  -0.54,  0.198,
                               0.392,  0.321,  0.404,   0.16,      0,      0,      0,      0,      0],
                             [-0.281, -0.686, -0.783,  -0.68, -0.547, -0.359, -0.202, -0.119,  0.005,  0.112,  0.163,  0.261,
                               0.345,  0.321,  0.319,  0.371,  0.319,   0.13, -0.118, -0.321, -0.486, -0.671, -0.534,  0.199,
                               0.393,  0.323,  0.405,  0.161,      0,      0,      0,      0,      0],
                             [-0.269, -0.667,  -0.76, -0.655, -0.522, -0.336, -0.181, -0.096,  0.029,  0.136,  0.188,  0.286,
                                0.37,  0.346,  0.345,  0.396,  0.342,   0.15, -0.102, -0.307, -0.473, -0.661, -0.528,    0.2,
                               0.393,  0.324,  0.405,  0.162,      0,      0,      0,      0,      0],
                             [-0.255, -0.653, -0.747, -0.643, -0.511, -0.325, -0.169, -0.082,  0.042,  0.149,  0.204,  0.304,
                               0.388,  0.363,  0.36 ,  0.409,  0.354,  0.164, -0.085, -0.289, -0.457, -0.649, -0.523,  0.193,
                               0.394,  0.326,  0.406,  0.162,      0,      0,      0,      0,      0],
                             [-0.265,  -0.65, -0.739, -0.634,   -0.5, -0.314, -0.159, -0.072,  0.052,  0.159,  0.215,  0.316,
                               0.398,  0.374,  0.374,  0.424,   0.37,  0.181, -0.065, -0.265, -0.429, -0.627, -0.519,   0.18,
                               0.394,  0.326,  0.406,  0.162,      0,      0,      0,      0,      0],
                             [-0.276, -0.647, -0.731, -0.626, -0.492, -0.307, -0.152, -0.064,  0.058,  0.166,  0.227,  0.329,
                               0.411,  0.389,   0.39,  0.441,  0.389,  0.207, -0.032, -0.228, -0.394, -0.596, -0.494,  0.194,
                               0.376,  0.326,  0.406,  0.162,      0,      0,      0,      0,      0],
                             [-0.271, -0.646,  -0.73, -0.625, -0.489, -0.303, -0.149, -0.061,  0.062,  0.169,  0.229,  0.332,
                               0.412,  0.388,  0.389,  0.439,  0.387,  0.206, -0.028, -0.209, -0.347, -0.524, -0.435,  0.195,
                               0.381,  0.313,  0.405,  0.162,      0,      0,      0,      0,      0],
                             [-0.267, -0.647, -0.734, -0.628,  -0.49, -0.304, -0.151, -0.062,  0.061,  0.168,  0.229,  0.329,
                               0.408,  0.385,  0.388,  0.438,  0.386,  0.206, -0.024, -0.194, -0.319,  -0.48,  -0.36,  0.318,
                               0.405,  0.335,  0.394,  0.162,      0,      0,      0,      0,      0],
                             [-0.274, -0.656, -0.745,  -0.64,   -0.5, -0.313, -0.158, -0.068,  0.054,  0.161,  0.223,  0.325,
                               0.402,  0.379,  0.384,  0.438,  0.392,  0.221,  0.001, -0.164, -0.278, -0.415, -0.264,  0.445,
                               0.402,  0.304,  0.389,  0.157,      0,      0,      0,      0,      0],
                             [-0.289, -0.666, -0.753, -0.648, -0.508,  -0.32, -0.164, -0.073,  0.049,  0.156,   0.22,  0.321,
                               0.397,  0.374,  0.377,   0.43,  0.387,  0.224,  0.014, -0.139, -0.236, -0.359, -0.211,  0.475,
                                 0.4,  0.308,  0.375,  0.155,      0,      0,      0,      0,      0],
                             [-0.302, -0.678, -0.765, -0.659, -0.517, -0.329, -0.176, -0.085,  0.038,  0.145,  0.208,   0.31,
                               0.386,  0.362,  0.366,  0.421,  0.381,  0.224,  0.022, -0.119, -0.201,   -0.3, -0.129,  0.572,
                               0.419,  0.265,  0.364,  0.154,      0,      0,      0,      0,      0],
                             [-0.314, -0.696, -0.786, -0.681, -0.539, -0.349, -0.196, -0.105,  0.019,  0.127,  0.189,  0.289,
                               0.364,   0.34,  0.346,  0.403,   0.37,  0.222,  0.036, -0.081, -0.133, -0.205, -0.021,  0.674,
                               0.383,  0.237,  0.359,  0.151,      0,      0,      0,      0,      0],
                             [-0.341, -0.719, -0.807, -0.702, -0.558, -0.367, -0.211,  -0.12,  0.003,  0.111,  0.175,  0.277,
                               0.351,  0.325,  0.331,   0.39,   0.36,  0.221,  0.048, -0.046, -0.074, -0.139,  0.038,  0.726,
                               0.429,  0.215,  0.347,  0.151,      0,      0,      0,      0,      0],
                             [ -0.35, -0.737, -0.829, -0.724, -0.577, -0.385, -0.229, -0.136, -0.011,  0.098,  0.163,  0.266,
                               0.338,   0.31,  0.316,  0.378,  0.354,  0.221,  0.062, -0.009, -0.012, -0.063,  0.119,  0.811,
                               0.319,  0.201,  0.343,  0.148,      0,      0,      0,      0,      0],
                             [-0.344,  -0.75, -0.856, -0.757, -0.607, -0.409,  -0.25, -0.156, -0.033,  0.076,  0.143,  0.246,
                               0.316,  0.287,  0.293,  0.361,  0.345,  0.225,  0.082,  0.035,  0.071,  0.046,  0.172,  0.708,
                               0.255,   0.21,  0.325,  0.146,      0,      0,      0,      0,      0]])/86400

        cdef:
            Py_ssize_t tt, k, ind1, ind2
        A = np.interp(Gr.z_half,z_in,rad_in[0,:])
        for tt in xrange(1,36):
            A = np.vstack((A, np.interp(Gr.z_half,z_in,rad_in[tt,:])))
        self.rad = np.multiply(A,1.0) # store matrix in self

        ind1 = int(mt.trunc(10.0/600.0))
        ind2 = int(mt.ceil(10.0/600.0))
        for k in xrange(Gr.nzg):
            if 10%600.0 == 0:
                self.Fo.dTdt[k] = self.rad[ind1,k]
            else:
                self.Fo.dTdt[k]    = (self.rad[ind2,k]-self.rad[ind1,k])/\
                                      (self.rad_time[ind2]-self.rad_time[ind1])*(10.0)+self.rad[ind1,k]


        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.lhf = 554.0 * mt.pow(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - TS.t)/5.25/3600.0))),1.3)
        self.Sur.shf = 270.0 * mt.pow(np.maximum(0, np.cos(np.pi/2*((5.25*3600.0 - TS.t)/5.25/3600.0))),1.5)
        self.Sur.update(GMV)
        # fix momentum fluxes to zero as they are not used in the paper
        self.Sur.rho_uflux = 0.0
        self.Sur.rho_vflux = 0.0
        return
    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        cdef:
            Py_ssize_t k, ind1, ind2

        ind2 = int(mt.ceil(TS.t/600.0))
        ind1 = int(mt.trunc(TS.t/600.0))
        if TS.t<600.0: # first 10 min use the radiative forcing of t=10min (as in the paper)
            for k in xrange(self.Fo.Gr.nzg):
                self.Fo.dTdt[k] = self.rad[0,k]
        elif TS.t>18900.0:
            for k in xrange(self.Fo.Gr.nzg):
                self.Fo.dTdt[k] = (self.rad[31,k]-self.rad[30,k])/(self.rad_time[31]-self.rad_time[30])\
                                      *(18900.0/60.0-self.rad_time[30])+self.rad[30,k]

        else:
            if TS.t%600.0 == 0:
                for k in xrange(self.Fo.Gr.nzg):
                    self.Fo.dTdt[k] = self.rad[ind1,k]
            else: # in all other cases - interpolate
                for k in xrange(self.Fo.Gr.nzg):
                    if self.Fo.Gr.z_half[k] < 22699.48:
                        self.Fo.dTdt[k]    = (self.rad[ind2,k]-self.rad[ind1,k])\
                                                 /(self.rad_time[ind2]-self.rad_time[ind1])\
                                                 *(TS.t/60.0-self.rad_time[ind1])+self.rad[ind1,k]
                    else:
                        self.Fo.dTdt[k] = 0.0

        self.Fo.update(GMV)

        return

cdef class ARM_SGP(CasesBase):
    # adopted from: "Large-eddy simulation of the diurnal cycle of shallow cumulus convection over land",
    # By Brown et al. (2002)  Q. J. R. Meteorol. Soc. 128, 1075-1093
    def __init__(self, paramlist):
        self.casename = 'ARM_SGP'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingStandard() # it was forcing standard
        self.inversion_option = 'thetal_maxgrad'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 8.5e-5
        self.Fo.apply_subsidence =False

        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 970.0*100 #Pressure at ground
        Ref.Tg = 299.0   # surface values for reference state (RS) which outputs p0 rho0 alpha0
        Ref.qtg = 15.2/1000#Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            Py_ssize_t k
            double [:] p1 = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # ARM_SGP inputs
        z_in = np.array([0.0, 50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0 ]) #LES z is in meters
        Theta_in = np.array([299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2]) # K
        r_in = np.array([15.2,15.17,14.98,14.8,14.7,13.5,3.0,3.0])/1000 # qt should be in kg/kg
        qt_in = np.divide(r_in,(1+r_in))

        # interpolate to the model grid-points
        Theta = np.interp(Gr.z_half,z_in,Theta_in)
        qt = np.interp(Gr.z_half,z_in,qt_in)


        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] = 10.0
            GMV.QT.values[k] = qt[k]
            GMV.T.values[k] = Theta[k]*exner_c(Ref.p0_half[k])
            if GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0)
            elif GMV.H.name == 'thetal':
                 GMV.H.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))

            GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))


        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.Tsurface = 299.0 * exner_c(Ref.Pg)
        self.Sur.qsurface = 15.2e-3 # kg/kg
        self.Sur.lhf = 5.0
        self.Sur.shf = -30.0
        self.Sur.ustar_fixed = True
        self.Sur.ustar = 0.28 # this is taken from Bomex -- better option is to approximate from LES tke above the surface
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize()

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        cdef:
            Py_ssize_t k
        for k in xrange(Gr.nzg):
            self.Fo.ug[k] = 10.0
            self.Fo.vg[k] = 0.0

        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        cdef:
            double [:] t_Sur_in = np.array([0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5]) * 3600 #LES time is in sec
            double [:] SH = np.array([-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10]) # W/m^2
            double [:] LH = np.array([5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0]) # W/m^2
        self.Sur.shf = np.interp(TS.t,t_Sur_in,SH)
        self.Sur.lhf = np.interp(TS.t,t_Sur_in,LH)
        # if fluxes vanish bflux vanish and wstar and obukov length are NaNs
        ## CK +++ I commented out the lines below as I don't think this is how we want to fix things!
        # if self.Sur.shf < 1.0:
        #     self.Sur.shf = 1.0
        # if self.Sur.lhf < 1.0:
        #     self.Sur.lhf = 1.0
        #+++++++++
        self.Sur.update(GMV)
        # fix momentum fluxes to zero as they are not used in the paper
        self.Sur.rho_uflux = 0.0
        self.Sur.rho_vflux = 0.0
        return

    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        cdef:
            double [:] t_in = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 14.5]) * 3600.0 #LES time is in sec
            double [:] AT_in = np.array([0.0, 0.0, 0.0, -0.08, -0.016, -0.016])/3600.0 # Advective forcing for theta [K/h] converted to [K/sec]
            double [:] RT_in = np.array([-0.125, 0.0, 0.0, 0.0, 0.0, -0.1])/3600.0  # Radiative forcing for theta [K/h] converted to [K/sec]
            double [:] Rqt_in = np.array([0.08, 0.02, 0.04, -0.1, -0.16, -0.3])/1000.0/3600.0 # Radiative forcing for qt converted to [kg/kg/sec]
            double dTdt = np.interp(TS.t,t_in,AT_in) + np.interp(TS.t,t_in,RT_in)
            double dqtdt =  np.interp(TS.t,t_in,Rqt_in)
        for k in xrange(self.Fo.Gr.nzg): # correct dims
                if self.Fo.Gr.z_half[k] <=1000.0:
                    self.Fo.dTdt[k] = dTdt
                    self.Fo.dqtdt[k]  = dqtdt * exner_c(self.Fo.Ref.p0_half[k])
                elif self.Fo.Gr.z_half[k] > 1000.0  and self.Fo.Gr.z_half[k] <= 2000.0:
                    self.Fo.dTdt[k] = dTdt*(1-(self.Fo.Gr.z_half[k]-1000.0)/1000.0)
                    self.Fo.dqtdt[k]  = dqtdt * exner_c(self.Fo.Ref.p0_half[k])\
                                        *(1-(self.Fo.Gr.z_half[k]-1000.0)/1000.0)
        self.Fo.update(GMV)

        return


cdef class GATE_III(CasesBase):
    # adopted from: "Large eddy simulation of Maritime Deep Tropical Convection",
    # By Khairoutdinov et al (2009)  JAMES, vol. 1, article #15
    def __init__(self, paramlist):
        self.casename = 'GATE_III'
        self.Sur = Surface.SurfaceFixedCoeffs(paramlist)
        self.Fo = Forcing.ForcingStandard() # it was forcing standard
        self.inversion_option = 'thetal_maxgrad'
        self.Fo.apply_subsidence = False
        self.Fo.apply_coriolis = False

        return
    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1013.0*100  #Pressure at ground
        Ref.Tg = 299.184   # surface values for reference state (RS) which outputs p0 rho0 alpha0
        Ref.qtg = 16.5/1000#Total water mixing ratio at surface
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double qv
            double [:] qt = np.zeros((Gr.nzg,),dtype=np.double,order='c')
            double [:] T = np.zeros((Gr.nzg,),dtype=np.double,order='c') # Gr.nzg = Gr.nz + 2*Gr.gw
            double [:] U = np.zeros((Gr.nzg,),dtype=np.double,order='c')
            double [:] theta_rho = np.zeros((Gr.nzg,),dtype=np.double,order='c')

        # GATE_III inputs - I extended them to z=22 km
        z_in  = np.array([ 0.0,   0.5,  1.0,  1.5,  2.0,   2.5,    3.0,   3.5,   4.0,   4.5,   5.0,  5.5,  6.0,  6.5,
                           7.0, 7.5, 8.0,  8.5,   9.0,   9.5, 10.0,   10.5,   11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
                           14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 27.0]) * 1000.0 #z is in meters
        r_in = np.array([16.5,  16.5, 13.5, 12.0, 10.0,   8.7,    7.1,   6.1,   5.2,   4.5,   3.6,  3.0,  2.3, 1.75, 1.3,
                         0.9, 0.5, 0.25, 0.125, 0.065, 0.003, 0.0015, 0.0007,  0.0003,  0.0001,  0.0001,  0.0001,  0.0001,
                         0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001, 0.0001])/1000 # mixing ratio should be in kg/kg
        U_in  = np.array([  -1, -1.75, -2.5, -3.6, -6.0, -8.75, -11.75, -13.0, -13.1, -12.1, -11.0, -8.5, -5.0, -2.6, 0.0,
                            0.5, 0.4,  0.3,   0.0,  -1.0, -2.5,   -3.5,   -4.5, -4.8, -5.0, -3.5, -2.0, -1.0, -1.0, -1.0,
                            -1.5, -2.0, -2.5, -2.6, -2.7, -3.0, -3.0, -3.0])# [m/s]
        qt_in = np.divide(r_in,(1+r_in)) # convert mixing ratio to specific humidity

        # temperature is taken from a different input plot at different z levels
        T_in = np.array([299.184, 294.836, 294.261, 288.773, 276.698, 265.004, 253.930, 243.662, 227.674, 214.266, 207.757, 201.973, 198.278, 197.414, 198.110, 198.110])
        z_T_in = np.array([0.0, 0.492, 0.700, 1.698, 3.928, 6.039, 7.795, 9.137, 11.055, 12.645, 13.521, 14.486, 15.448, 16.436, 17.293, 22.0])*1000.0 # for km

        # interpolate to the model grid-points
        T = np.interp(Gr.z_half,z_T_in,T_in) # interpolate to ref pressure level
        qt = np.interp(Gr.z_half,z_in,qt_in)
        U = np.interp(Gr.z_half,z_in,U_in)


        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.QT.values[k] = qt[k]
            GMV.T.values[k] = T[k]
            GMV.U.values[k] = U[k]

            if GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0)
            elif GMV.H.name == 'thetal':
                 GMV.H.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))

            GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                GMV.QT.values[k], 0.0, 0.0, latent_heat(GMV.T.values[k]))
        GMV.U.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.qsurface = 16.5/1000.0 # kg/kg
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.cm  = 0.0012
        self.Sur.ch = 0.0034337
        self.Sur.cq = 0.0034337
        self.Sur.Tsurface = 299.184
        self.Sur.initialize()

        return
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        #LES z is in meters
        z_in     = np.array([ 0.0,   0.5,  1.0,  1.5,   2.0,   2.5,    3.0,   3.5,   4.0,   4.5,   5.0,   5.5,   6.0,
                              6.5,  7.0,  7.5,   8.0,  8.5,   9.0,  9.5,  10.0,  10.5,  11.0,    11.5,   12.0, 12.5,
                              13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0]) * 1000.0
        u_in     = np.array([  -1, -1.75, -2.5, -3.6,  -6.0, -8.75, -11.75, -12.9, -13.1, -12.1, -11.0,  -8.5,  -5.0,
                               -2.6,  0.0,  0.5,   0.4,  0.3,   0.0, -1.0,  -3.0,  -3.5,  -4.5,    -4.6,   -5.0, -3.5,
                               -2.0, -1.0, -1.0, -1.0, -1.5, -2.0, -2.5, -2.6, -2.7, -3.0, -3.0])
        # Radiative forcing for T [K/d] converted to [K/sec]
        RAD_in   = np.array([-2.9,  -1.1, -0.8, -1.1, -1.25, -1.35,   -1.4,  -1.4, -1.44, -1.52,  -1.6, -1.54, -1.49,
                             -1.43, -1.36, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05,  -1.0,  -0.95,   -0.9,  -0.85, -0.8,
                             -0.75, -0.7, -0.6, -0.3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])/(24.0*3600.0)
        # Advective qt forcing  for theta [g/kg/d] converted to [kg/kg/sec]
        r_tend_in = np.array([ 0.0,   1.2,  2.0,  2.3,   2.2,   2.1,    1.9,   1.7,   1.5,  1.35,  1.22,  1.08,  0.95,
                               0.82,  0.7,  0.6,   0.5,  0.4,   0.3,  0.2,   0.1,  0.05, 0.0025, 0.0012, 0.0006,  0.0,
                               0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])/(24.0*3600.0)/1000.0
        # Radiative T forcing [K/d] converted to [K/sec]
        Ttend_in = np.array([ 0.0,  -1.0, -2.2, -3.0,  -3.5,  -3.8,   -4.0,  -4.1,  -4.2,  -4.2,  -4.1,  -4.0, -3.85,
                              -3.7, -3.5, -3.25, -3.0, -2.8,  -2.5, -2.1,  -1.7,  -1.3,   -1.0,   -0.7,   -0.5, -0.4,
                              -0.3, -0.2, -0.1,-0.05,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0])/(24.0*3600.0)

        Qtend_in = np.divide(r_tend_in,(1+r_tend_in)) # convert mixing ratio to specific humidity

        self.Fo.dqtdt = np.interp(Gr.z_half,z_in,Qtend_in)
        self.Fo.dTdt = np.interp(Gr.z_half,z_in,Ttend_in) + np.interp(Gr.z_half,z_in,RAD_in)
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return
    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV) # here lhf and shf are needed for calcualtion of bflux in surface and thus u_star
        return

    cpdef update_forcing(self, GridMeanVariables GMV,  TimeStepping TS):
        self.Fo.update(GMV)
        return


cdef class DYCOMS_RF01(CasesBase):
    """
    see Stevens et al 2005:
    Evaluation of Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus.
    Mon. Wea. Rev., 133, 1443â€“1462.
    doi: http://dx.doi.org/10.1175/MWR2930.1
    """
    def __init__(self, paramlist):
        self.casename = 'DYCOMS_RF01'
        self.Sur = Surface.SurfaceFixedFlux(paramlist)
        self.Fo = Forcing.ForcingDYCOMS_RF01() # radiation is included in Forcing
        self.inversion_option = 'thetal_maxgrad'
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg   = 1017.8 * 100.0
        Ref.qtg  = 9.0 / 1000.0
        # Use an exner function with values for Rd, and cp given in Stevens 2005 to compute temperature
        Ref.Tg   = 289.0 * exner_c(Ref.Pg, kappa = dycoms_Rd / dycoms_cp)
        Ref.initialize(Gr, Stats)
        return

    # helper function
    def dycoms_compute_thetal(self, p_, T_, ql_):
        """
        Compute thetal using constants from Stevens et al 2005 DYCOMS case.
        :param p: pressure [Pa]
        :param T: temperature [K]
        :param ql: liquid water specific humidity
        :return: theta l
        """
        theta_ = T_ / exner_c(p_, kappa = dycoms_Rd / dycoms_cp)
        return theta_ * mt.exp(-1. * dycoms_L * ql_ / (dycoms_cp * T_))

    # helper function
    def dycoms_sat_adjst(self, p_, thetal_, qt_):
        '''
        Use saturation adjustment scheme to compute temperature and ql given thetal and qt.
        We can't use the default scampy function because of different values of cp, Rd and L
        :param p: pressure [Pa]
        :param thetal: liquid water potential temperature  [K]
        :param qt:  total water specific humidity
        :return: T, ql
        '''
        #Compute temperature
        t_1 = thetal_ * exner_c(p_, kappa = dycoms_Rd / dycoms_cp)
        #Compute saturation vapor pressure
        pv_star_1 = pv_star(t_1)
        #Compute saturation specific humidity
        qs_1 = qv_star_c(p_, qt_, pv_star_1)

        if qt_ <= qs_1:
            #If not saturated return temperature and ql = 0.0
            return t_1, 0.0
        else:
            ql_1 = qt_ - qs_1
            f_1 = thetal_ - self.dycoms_compute_thetal(p_, t_1, ql_1)
            t_2 = t_1 + dycoms_L * ql_1 / dycoms_cp
            pv_star_2 = pv_star(t_2)
            qs_2 = qv_star_c(p_, qt_, pv_star_2)
            ql_2 = qt_ - qs_2

            while np.fabs(t_2 - t_1) >= 1e-9:
                pv_star_2 = pv_star(t_2)
                qs_2 = qv_star_c(p_, qt_, pv_star_2)
                ql_2 = qt_ - qs_2
                f_2 = thetal_ - self.dycoms_compute_thetal(p_, t_2, ql_2)
                t_n = t_2 - f_2 * (t_2 - t_1)/(f_2 - f_1)
                t_1 = t_2
                t_2 = t_n
                f_1 = f_2

            return t_2, ql_2

    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c') # helper variable to recalculate temperature
        ql     = np.zeros((Gr.nzg,), dtype=np.double, order='c') # DYCOMS case is saturated
        qi     = 0.0                                             # no ice

        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            # thetal profile as defined in DYCOMS
            if Gr.z_half[k] <= 840.0:
               thetal[k] = 289.0
            if Gr.z_half[k] > 840.0:
               thetal[k] = (297.5 + (Gr.z_half[k] - 840.0)**(1.0/3.0))

            # qt profile as defined in DYCOMS
            if Gr.z_half[k] <= 840.0:
               GMV.QT.values[k] = 9. / 1000.0
            if Gr.z_half[k] > 840.0:
               GMV.QT.values[k] = 1.5 / 1000.0

            # ql and T profile
            # (calculated by saturation adjustment using thetal and qt values provided in DYCOMS
            # and using Rd, cp and L constants as defined in DYCOMS)
            GMV.T.values[k], GMV.QL.values[k] = self.dycoms_sat_adjst(Ref.p0_half[k], thetal[k], GMV.QT.values[k])

            # thermodynamic variable profile (either entropy or thetal)
            # (calculated based on T and ql profiles.
            # Here we use Rd, cp and L constants as defined in scampy)
            GMV.THL.values[k] = t_to_thetali_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], qi)
            if GMV.H.name == 'thetal':
                GMV.H.values[k] = t_to_thetali_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], qi)
            elif GMV.H.name == 's':
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], qi)

            # buoyancy profile
            qv = GMV.QT.values[k] - qi - GMV.QL.values[k]
            alpha = alpha_c(Ref.p0_half[k], GMV.T.values[k], GMV.QT.values[k], qv)
            GMV.B.values[k] = buoyancy_c(Ref.alpha0_half[k], alpha)

            # velocity profile (geostrophic)
            GMV.U.values[k] = 7.0
            GMV.V.values[k] = -5.5

        # fill out boundary conditions
        GMV.U.set_bcs(Gr)
        GMV.V.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.QL.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.THL.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.B.set_bcs(Gr)

        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref ):
        self.Sur.zrough      = 1.0e-4
        self.Sur.ustar_fixed = False
        self.Sur.cm          = 0.0011

        # sensible heat flux
        self.Sur.shf = 15.0
        # latent heat flux
        self.Sur.lhf = 115.0

        self.Sur.Tsurface = 292.5    # K      # i.e. the SST from DYCOMS setup
        self.Sur.qsurface = 13.84e-3 # kg/kg  # TODO - taken from Pycles, maybe it would be better to calculate the q_star(sst) for scampy?
        #density_surface  = 1.22     # kg/m^3

        # buoyancy flux
        theta_flux       = self.Sur.shf / cpm_c(self.Sur.qsurface)        / Ref.rho0[Gr.gw-1]
        qt_flux          = self.Sur.lhf / latent_heat(self.Sur.Tsurface)  / Ref.rho0[Gr.gw-1]
        theta_surface    = self.Sur.Tsurface / exner_c(Ref.Pg)
        self.Sur.bflux   =  g * ((theta_flux + (eps_vi - 1.0) * (theta_surface * qt_flux + self.Sur.qsurface * theta_flux))
                                 / (theta_surface * (1.0 + (eps_vi-1) * self.Sur.qsurface)))
        self.Sur.Gr  = Gr
        self.Sur.Ref = Ref
        self.Sur.initialize()

        return
 
    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)

        # geostrophic velocity profiles
        self.Fo.ug[:] = 7.0
        self.Fo.vg[:] = -5.5

        # large scale subsidence
        divergence = 3.75e-6    # divergence is defined twice: here and in __init__ of ForcingDYCOMS_RF01 class
                                # To be able to have self.Fo.divergence available here,
                                # we would have to change the signature of ForcingBase class
        for k in xrange(Gr.gw, Gr.nzg-Gr.gw):
            self.Fo.subsidence[k] = - Gr.z_half[k] * divergence

        # no large-scale drying
        self.Fo.dqtdt[:] = 0. #kg/(kg * s)

        # radiation is treated as a forcing term (see eq. 3 in Stevens et. al. 2005)
        # cloud-top cooling + cloud-base warming + cooling in free troposphere
        self.Fo.calculate_radiation(GMV)

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        self.Fo.initialize_io(Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self, Stats)
        self.Fo.io(Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return

cdef class GABLS(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'GABLS'
        self.Sur = Surface.SurfaceMoninObukhov(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        cdef double latitude = 73.0
        self.Fo.coriolis_param = 1.39e-4 # s^{-1}
        # self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = False
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.0e5  #Pressure at ground
        Ref.Tg = 265.0  #Temperature at ground
        Ref.qtg = 1.0e-4 #Total water mixing ratio at surface. if set to 0, alpha0, rho0, p0 are NaN (TBD)
        Ref.initialize(Gr, Stats)
        return
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of GABLS cloud-free
            double [:] theta_pert = np.random.random_sample(Gr.nzg)
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            #Set wind velocity profile
            GMV.U.values[k] =  8.0
            GMV.V.values[k] =  0.0

            #Set Thetal profile
            if Gr.z_half[k] <= 100.0:
                thetal[k] = 265.0
            else:
                thetal[k] = 265.0 + (Gr.z_half[k] - 100.0) * 0.01

            #Set qt profile
            GMV.QT.values[k] = 0.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k]) # No water content
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.V.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.zrough = 0.1
        self.Sur.Tsurface = 265.0
        self.Sur.initialize()
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        cdef Py_ssize_t k
        for k in xrange(Gr.gw, Gr.nzg - Gr.gw):
            # Geostrophic velocity profiles.
            self.Fo.ug[k] = 8.0
            self.Fo.vg[k] = 0.0
        return

    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return

    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.Tsurface = 265.0 - (0.25/3600.0)*TS.t
        self.Sur.update(GMV)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return

# Not fully implemented yet - Ignacio
cdef class SP(CasesBase):
    def __init__(self, paramlist):
        self.casename = 'SP'
        self.Sur = Surface.SurfaceSullivanPatton(paramlist)
        self.Fo = Forcing.ForcingStandard()
        self.inversion_option = 'critical_Ri'
        self.Fo.apply_coriolis = True
        self.Fo.coriolis_param = 1.0e-4 # s^{-1}
        # self.Fo.coriolis_param = 2.0 * omega * np.sin(latitude * pi / 180.0 ) # s^{-1}
        self.Fo.apply_subsidence = False
        return

    cpdef initialize_reference(self, Grid Gr, ReferenceState Ref, NetCDFIO_Stats Stats):
        Ref.Pg = 1.0e5  #Pressure at ground
        Ref.Tg = 300.0  #Temperature at ground
        Ref.qtg = 1.0e-4   #Total water mixing ratio at surface. if set to 0, alpha0, rho0, p0 are NaN. 
        Ref.initialize(Gr, Stats)
        return
        
    cpdef initialize_profiles(self, Grid Gr, GridMeanVariables GMV, ReferenceState Ref):
        cdef:
            double [:] thetal = np.zeros((Gr.nzg,), dtype=np.double, order='c')
            double ql=0.0, qi =0.0 # IC of SP cloud-free
            Py_ssize_t k

        for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
            GMV.U.values[k] =  1.0
            GMV.V.values[k] =  0.0
            #Set Thetal profile
            if Gr.z_half[k] <= 974.0:
                thetal[k] = 300.0
            elif Gr.z_half[k] < 1074.0:
                thetal[k] = 300.0 + (Gr.z_half[k] - 974.0) * 0.08
            else:
                thetal[k] = 308.0 + (Gr.z_half[k] - 1074.0) * 0.003

            #Set qt profile
            GMV.QT.values[k] = 0.0

        if GMV.H.name == 'thetal':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.H.values[k] = thetal[k]
                GMV.T.values[k] =  thetal[k] * exner_c(Ref.p0_half[k])
                GMV.THL.values[k] = thetal[k]
        elif GMV.H.name == 's':
            for k in xrange(Gr.gw,Gr.nzg-Gr.gw):
                GMV.T.values[k] = thetal[k] * exner_c(Ref.p0_half[k])
                GMV.H.values[k] = t_to_entropy_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi)
                GMV.THL.values[k] = thetali_c(Ref.p0_half[k],GMV.T.values[k],
                                                 GMV.QT.values[k], ql, qi, latent_heat(GMV.T.values[k]))

        GMV.U.set_bcs(Gr)
        GMV.V.set_bcs(Gr)
        GMV.QT.set_bcs(Gr)
        GMV.H.set_bcs(Gr)
        GMV.T.set_bcs(Gr)
        GMV.satadjust()
        return

    cpdef initialize_surface(self, Grid Gr, ReferenceState Ref):
        self.Sur.Gr = Gr
        self.Sur.Ref = Ref
        self.Sur.zrough = 0.1
        self.Sur.Tsurface = 300.0
        theta_surface    = self.Sur.Tsurface / exner_c(Ref.Pg)
        theta_flux = 0.24
        self.Sur.bflux   =  g * theta_flux / theta_surface
        # self.Sur.bflux = 0.24 * exner_c(Ref.p0_half[Gr.gw]) * g / (Ref.p0_half[Gr.gw]*Ref.alpha0_half[Gr.gw]/Rd)
        self.Sur.initialize()
        return

    cpdef initialize_forcing(self, Grid Gr, ReferenceState Ref, GridMeanVariables GMV):
        self.Fo.Gr = Gr
        self.Fo.Ref = Ref
        self.Fo.initialize(GMV)
        cdef Py_ssize_t k
        for k in xrange(Gr.gw, Gr.nzg - Gr.gw):
            # Geostrophic velocity profiles. vg = 0
            self.Fo.ug[k] = 1.0
            self.Fo.vg[k] = 0.0
        return


    cpdef initialize_io(self, NetCDFIO_Stats Stats):
        CasesBase.initialize_io(self, Stats)
        return

    cpdef io(self, NetCDFIO_Stats Stats):
        CasesBase.io(self,Stats)
        return
        
    cpdef update_surface(self, GridMeanVariables GMV, TimeStepping TS):
        self.Sur.update(GMV)
        return

    cpdef update_forcing(self, GridMeanVariables GMV, TimeStepping TS):
        self.Fo.update(GMV)
        return
