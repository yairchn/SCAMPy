cimport Grid
cimport ReferenceState
from Variables cimport GridMeanVariables
from EDMF_Environment cimport EnvironmentThermodynamics
from EDMF_Updrafts cimport UpdraftThermodynamics
from NetCDFIO cimport NetCDFIO_Stats
from TimeStepping cimport TimeStepping

cdef class RainVariable:
    cdef:
        str loc
        str kind
        str name
        str units

        double [:] values
        double [:] new
        double [:] flux

    cpdef set_bcs(self, Grid.Grid Gr)

cdef class RainVariables:
    cdef:
        str rain_model

        double mean_rwp
        double env_rwp
        double upd_rwp
        double rain_area_value
        double max_supersaturation

        Grid.Grid Gr

        RainVariable QR
        RainVariable RainArea
        RainVariable Upd_QR
        RainVariable Upd_RainArea
        RainVariable Env_QR
        RainVariable Env_RainArea

    cpdef initialize_io(self, NetCDFIO_Stats Stats)
    cpdef io(self, NetCDFIO_Stats, ReferenceState.ReferenceState Ref)
    cpdef sum_subdomains_rain(self, UpdraftThermodynamics UpdThermo, EnvironmentThermodynamics EnvThermo)
    cpdef rain_diagnostics(self, ReferenceState.ReferenceState Ref)

cdef class RainPhysics:
    cdef :
        Grid.Grid Gr
        ReferenceState.ReferenceState Ref

        double [:] rain_evap_source_h
        double [:] rain_evap_source_qt

    cpdef solve_rain_fall(
        self, GridMeanVariables GMV, TimeStepping TS, RainVariable QR,
        RainVariable RainArea
    )

    cpdef solve_rain_evap(
        self, GridMeanVariables GMV, TimeStepping TS, RainVariable QR,
        RainVariable RainArea
    )
