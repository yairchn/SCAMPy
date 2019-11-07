from NetCDFIO cimport NetCDFIO_Stats
from Grid cimport  Grid
from ReferenceState cimport ReferenceState
from EDMF_Rain cimport RainVariables

cdef class EnvironmentVariable:
    cdef:
        double [:] values
        double [:] flux
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self,Grid Gr)

cdef class EnvironmentVariable_2m:
    cdef:
        double [:] values
        double [:] dissipation
        double [:] shear
        double [:] entr_gain
        double [:] detr_loss
        double [:] press
        double [:] buoy
        double [:] interdomain
        double [:] rain_src
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self,Grid Gr)

cdef class EnvironmentVariables:
    cdef:
        Grid Gr

        EnvironmentVariable W
        EnvironmentVariable Area
        EnvironmentVariable QT
        EnvironmentVariable QL
        EnvironmentVariable H
        EnvironmentVariable THL
        EnvironmentVariable RH
        EnvironmentVariable T
        EnvironmentVariable B
        EnvironmentVariable cloud_fraction

        EnvironmentVariable_2m TKE
        EnvironmentVariable_2m Hvar
        EnvironmentVariable_2m QTvar
        EnvironmentVariable_2m HQTcov

        bint calc_tke
        bint calc_scalar_var

        double cloud_base
        double cloud_top
        double cloud_cover
        double lwp

        str EnvThermo_scheme

    cpdef initialize_io(self, NetCDFIO_Stats Stats )
    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState Ref)
    cpdef env_cloud_diagnostics(self, ReferenceState Ref)

cdef class EnvironmentThermodynamics:
    cdef:
        Grid Gr
        ReferenceState Ref
        Py_ssize_t quadrature_order

        double (*t_to_prog_fp)(double p0, double T, double qt, double ql, double qi) nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil

        double [:] qt_dry
        double [:] th_dry
        double [:] t_cloudy
        double [:] qv_cloudy
        double [:] qt_cloudy
        double [:] th_cloudy

        double [:] Hvar_rain_dt
        double [:] QTvar_rain_dt
        double [:] HQTcov_rain_dt

        double [:] prec_source_qt
        double [:] prec_source_h

        void update_EnvVar(self, Py_ssize_t k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double alpha) nogil
        void update_EnvRain_sources(self, Py_ssize_t k, EnvironmentVariables EnvVar, double qr, double thl_rain_src) nogil
        void update_cloud_dry(self, Py_ssize_t k, EnvironmentVariables EnvVar, double T, double H, double qt, double ql, double qv) nogil

        void saturation_adjustment(self, EnvironmentVariables EnvVar)

        void sgs_mean(self, EnvironmentVariables EnvVar, RainVariables Rain, double dt)
        void sgs_quadrature(self, EnvironmentVariables EnvVar, RainVariables Rain, double dt)

    cpdef microphysics(self, EnvironmentVariables EnvVar, RainVariables Rain, double dt)
