cdef struct mph_struct:
    double T
    double thl
    double th
    double rho
    double qt
    double qv
    double ql
    double thl_rain_src
    double qr_src

cdef struct rain_struct:
    double qr
    double ar

cdef double rain_source_to_thetal(double p0, double T, double qr) nogil
cdef double rain_source_to_thetal_detailed(double p0, double T, double qt, double ql, double qr) nogil

cdef double acnv_instant(double ql, double qt, double T, double p0) nogil

cdef double terminal_velocity_single_drop_coeff(double rho) nogil
cdef double terminal_velocity(double q_rai, double rho) nogil
cdef double conv_q_vap_to_q_liq(double q_sat_liq, double q_liq) nogil
cdef double conv_q_liq_to_q_rai_acnv(double q_liq) nogil
cdef double conv_q_liq_to_q_rai_accr(double q_liq, double q_rai, double rho) nogil
cdef double conv_q_rai_to_q_vap(double q_rai, double q_tot, double q_liq, double T, double p, double rho) nogil

cdef mph_struct microphysics_rain_src(str rain_model, double qt, double ql, double qr, double area, double T, double p0, double rho, double dt) nogil

cdef rain_struct rain_area(double source_area, double source_qr, double current_area, double current_qr) nogil
