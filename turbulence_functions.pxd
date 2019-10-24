cdef struct entr_struct:
    double entr_sc
    double detr_sc
    double sorting_function
    double b_mix

cdef struct buoyant_stract:
    double b_mix
    double sorting_function

cdef struct chi_struct:
    double T_mix
    double ql_mix
    double qt_mix
    double qv_
    double alpha_mix
    double y1
    double x1

cdef struct entr_in_struct:
    double zi
    double wstar
    double z
    double sort_pow
    double sort_fact
    double dz
    double w
    double dw
    double b
    double rd
    double c_eps
    double dt
    double b_mean
    double b_env
    double af
    double tke
    double RH_upd
    double RH_env
    double ml
    double T_mean
    double p0
    double alpha0
    double T_up
    double qt_up
    double ql_up
    double T_env
    double qt_env
    double ql_env
    double H_up
    double H_env
    double w_env
    double env_Hvar
    double env_QTvar
    double env_HQTcov
    double dw_env
    double nh_pressure
    double dw2dz
    double L
    double zbl
    double poisson
    long quadrature_order

cdef struct pressure_in_struct:
    double updraft_top
    char *asp_label
    double a_med
    double a_kfull
    double a_khalf
    double a_kphalf
    double b_kfull
    double rho0_kfull
    double bcoeff_tan18
    double alpha1
    double alpha2
    double beta1
    double beta2
    double rd
    double w_kfull
    double w_khalf
    double w_kphalf
    double w_kenv
    double dzi
    double z_full
    double drag_sign
    double asp_ratio

cdef struct pressure_buoy_struct:
    double b_coeff
    double nh_pressure_b

cdef struct pressure_drag_struct:
    double nh_pressure_adv
    double nh_pressure_drag

cdef entr_struct entr_detr_dry(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_inverse_z(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_inverse_w(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_env_moisture_deficit(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_suselj(entr_in_struct entr_in) nogil
cdef entr_struct entr_detr_none(entr_in_struct entr_in) nogil
cdef double buoyancy_sorting(entr_in_struct entr_in) nogil
cdef buoyant_stract buoyancy_sorting_mean(entr_in_struct entr_in) nogil

cdef pressure_buoy_struct pressure_tan18_buoy(pressure_in_struct press_in) nogil
cdef pressure_drag_struct pressure_tan18_drag(pressure_in_struct press_in) nogil
cdef pressure_buoy_struct pressure_normalmode_buoy(pressure_in_struct press_in) nogil
cdef pressure_buoy_struct pressure_normalmode_buoysin(pressure_in_struct press_in) nogil
cdef pressure_drag_struct pressure_normalmode_drag(pressure_in_struct press_in) nogil

cdef double get_wstar(double bflux, double zi )
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit)
cdef double get_mixing_tau(double zi, double wstar) nogil

cdef double get_surface_tke(double ustar, double wstar, double zLL, double oblength) nogil
cdef double get_surface_variance(double flux1, double flux2, double ustar, double zLL, double oblength) nogil


cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil

cdef void construct_tridiag_diffusion(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a,
                                 double *b, double *c)
cdef void construct_tridiag_diffusion_implicitMF(Py_ssize_t nzg, Py_ssize_t gw,
                                            double dzi, double dt, double *rho_ae_K_m, double *massflux,
                                            double *rho, double *alpha, double *ae, double *a, double *b,
                                            double *c)
cdef void construct_tridiag_diffusion_dirichlet(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                           double *rho_ae_K_m, double *rho, double *ae, double *a,
                                           double *b, double *c)

cdef void tridiag_solve(Py_ssize_t nz, double *x, double *a, double *b, double *c)
