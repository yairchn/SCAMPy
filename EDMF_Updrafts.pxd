cimport Grid
cimport ReferenceState
from Variables cimport GridMeanVariables
from NetCDFIO cimport NetCDFIO_Stats
from EDMF_Environment cimport EnvironmentVariables
from EDMF_Rain cimport RainVariables

cdef class UpdraftVariable:
    cdef:
        double [:,:] values
        double [:,:] new
        double [:,:] old
        double [:,:] tendencies
        double [:,:] flux
        double [:] bulkvalues
        str loc
        str kind
        str name
        str units
    cpdef set_bcs(self, Grid.Grid Gr)

cdef class UpdraftVariables:
    cdef:
        Grid.Grid Gr

        UpdraftVariable W
        UpdraftVariable Area
        UpdraftVariable QT
        UpdraftVariable QL
        UpdraftVariable H
        UpdraftVariable RH
        UpdraftVariable THL
        UpdraftVariable T
        UpdraftVariable B

        Py_ssize_t n_updrafts
        bint prognostic

        double [:] cloud_fraction
        double [:] cloud_base
        double [:] cloud_top
        double [:] updraft_top
        double [:] cloud_cover

        double updraft_fraction
        double lwp

    cpdef initialize(self, GridMeanVariables GMV)
    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats Stats, ReferenceState.ReferenceState Ref)
    cpdef set_means(self, GridMeanVariables GMV)
    cpdef set_new_with_values(self)
    cpdef set_old_with_values(self)
    cpdef set_values_with_new(self)
    cpdef upd_cloud_diagnostics(self, ReferenceState.ReferenceState Ref)

cdef class UpdraftThermodynamics:
    cdef:
        double (*t_to_prog_fp)(double p0, double T, double qt, double ql, double qi) nogil
        double (*prog_to_t_fp)(double H, double pd, double pv, double qt ) nogil

        Grid.Grid Gr
        ReferenceState.ReferenceState Ref
        Py_ssize_t n_updraft

        double [:,:] prec_source_h
        double [:,:] prec_source_qt
        double [:] prec_source_h_tot
        double [:] prec_source_qt_tot

    cpdef buoyancy(
        self, UpdraftVariables UpdVar, EnvironmentVariables EnvVar,
        GridMeanVariables GMV, bint extrap
    )

    # helper functions to calculate autoconversion source terms to THL and QT
    cpdef clear_precip_sources(self)
    cpdef update_total_precip_sources(self)

    cpdef microphysics(self, UpdraftVariables UpdVar, RainVariables Rain, double dt)
