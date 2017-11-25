import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
include "parameters.pxi"
from thermodynamic_functions cimport qv_star_t, latent_heat

# Entrainment Rates

cdef entr_struct entr_detr_cloudy(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret

    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.entr_sc = 2.0e-3
        _ret.detr_sc= 3.0e-3
    else:
        _ret.entr_sc = 2.0e-3 * (1.0 - log(entr_in.z/entr_in.zi))
        _ret.detr_sc = 0.0

    return  _ret


cdef entr_struct entr_detr_dry(entr_in_struct entr_in)nogil:
    cdef entr_struct _ret
    cdef double eps = 1.0 # to avoid division by zero when z = 0 or z_i
    # Following Soares 2004
    _ret.entr_sc = 0.5*(1.0/entr_in.z + 1.0/fmax(entr_in.zi - entr_in.z, 10.0)) #vkb/(z + 1.0e-3)
    _ret.detr_sc = 0.0

    return  _ret

cdef entr_struct entr_detr_inverse_z(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret

    _ret.entr_sc = vkb/entr_in.z
    _ret.detr_sc= 0.0

    return _ret

cdef entr_struct entr_detr_inverse_w(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double tau = get_mixing_tau(entr_in.zi, entr_in.wstar)
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 3.0e-3
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = 1.0/(tau * fmax(entr_in.w,0.1)) #sets baseline to avoid errors
    return  _ret

cdef entr_struct entr_detr_inverse_w_linear(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
    if entr_in.z >= entr_in.zi :
        _ret.entr_sc = fmax(entr_in.b/fabs(entr_in.b+0.0000001),0.0)/(100.0 * fmax(entr_in.w,0.1))
        _ret.detr_sc = fmin(entr_in.b/fabs(entr_in.b+0.0000001),0.0)/(100.0 * fmax(entr_in.w,0.1))
    else:
        _ret.entr_sc = fmax(entr_in.b/fabs(entr_in.b+0.0000001),0.0)/(100.0 * fmax(entr_in.w,0.1))
        _ret.detr_sc = fmin(entr_in.b/fabs(entr_in.b+0.0000001),0.0)/(100.0 * fmax(entr_in.w,0.1))
    return  _ret

cdef entr_struct entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
    qv_up = entr_in.qt_up - entr_in.ql_up # if ice exists add here
    qv_mix = (qv_up + entr_in.qt_env)/2 # qv_env = qt_env
    dql_mix = qv_mix - qv_star_t(entr_in.p0, entr_in.T_mean)
    bmix = entr_in.b/2 + latent_heat(entr_in.T_mean) * dql_mix
    #flag = np.sign(bmix) # =1 if sign(bmix) = 1 and 0 if
    # eps0 = -K(dphi/dx); K = -l^2(dw/dx)
    #eps_sc = abs(entr_in.ml**2*entr_in.w*(entr_in.H_up-entr_in.H_env)/(entr_in.af*entr_in.L**2))
    #eps_w = np.abs(entr_in.ml**2*entr_in.w**2/(entr_in.af*entr_in.L**2))
    if bmix >= 0.0 :
        _ret.entr_sc = fabs(entr_in.ml**2*entr_in.w**2/(entr_in.af*entr_in.L**2))
        _ret.detr_sc = 0.0
    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = fabs(entr_in.ml**2*entr_in.w**2/(entr_in.af*entr_in.L**2))
    return  _ret

cdef entr_struct entr_detr_tke2(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 3.0e-3
    else:
        _ret.detr_sc = 0.0

    # _ret.entr_sc = (0.002 * sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) /
    #                 fmax(entr_in.af, 0.001) / fmax(entr_in.ml, 1.0))
    _ret.entr_sc = (0.05 * sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) / fmax(entr_in.af, 0.001) / fmax(entr_in.z, 1.0))
    return  _ret

# yair - this is a new entr-detr function that takes entr as proportional to TKE/w and detr ~ b/w2
cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    _ret.detr_sc = fabs(entr_in.b)/ fmax(entr_in.w * entr_in.w, 1e-3)
    _ret.entr_sc = sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) / fmax(sqrt(entr_in.af), 0.001) / 50000.0
    return  _ret
#
# cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil:
#     cdef entr_struct _ret
#     # in cloud portion from Soares 2004
#     if entr_in.z >= entr_in.zi :
#         _ret.detr_sc= 3.0e-3 +  0.2 * fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-4)
#     else:
#         _ret.detr_sc = 0.0
#
#     _ret.entr_sc = 0.2 * fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-4)
#     # or add to detrainment when buoyancy is negative
#     return  _ret


cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        # _ret.detr_sc= 4.0e-3 +  0.12 * fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-4)
        _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = 0.12 * fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)
    # or add to detrainment when buoyancy is negative
    return  _ret




# convective velocity scale
cdef double get_wstar(double bflux, double zi ):
    return cbrt(fmax(bflux * zi, 0.0))

# BL height
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit):
    cdef:
        double theta_rho_b = theta_rho[kmin]
        double h, Ri_bulk=0.0, Ri_bulk_low
        Py_ssize_t k = kmin


    # test if we need to look at the free convective limit
    if (u[kmin] * u[kmin] + v[kmin] * v[kmin]) <= 0.01:
        with nogil:
            for k in xrange(kmin,kmax):
                if theta_rho[k] > theta_rho_b:
                    break
        h = (z_half[k] - z_half[k-1])/(theta_rho[k] - theta_rho[k-1]) * (theta_rho_b - theta_rho[k-1]) + z_half[k-1]
    else:
        with nogil:
            for k in xrange(kmin,kmax):
                Ri_bulk_low = Ri_bulk
                Ri_bulk = g * (theta_rho[k] - theta_rho_b) * z_half[k]/theta_rho_b / (u[k] * u[k] + v[k] * v[k])
                if Ri_bulk > Ri_bulk_crit:
                    break
        h = (z_half[k] - z_half[k-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z_half[k-1]

    return h


cdef double get_mixing_tau(double zi, double wstar) nogil:
    # return 0.5 * zi / wstar
    return zi / (wstar + 0.001)




# MO scaling of near surface tke and scalar variance

cdef double get_surface_tke(double ustar, double wstar, double zLL, double oblength) nogil:
    if oblength < 0.0:
        return ((3.75 + cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar + 0.2 * wstar * wstar)
    else:
        return (3.75 * ustar * ustar)

cdef double get_surface_variance(double flux1, double flux2, double ustar, double zLL, double oblength) nogil:
    cdef:
        double c_star1 = -flux1/ustar
        double c_star2 = -flux2/ustar
    if oblength < 0.0:
        return 4.0 * c_star1 * c_star2 * pow(1.0 - 8.3 * zLL/oblength, -2.0/3.0)
    else:
        return 4.0 * c_star1 * c_star2



# Math-y stuff
cdef void construct_tridiag_diffusion(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return


cdef void construct_tridiag_diffusion_implicitMF(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *massflux, double *rho, double *alpha, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X + 0.5 * massflux[k-1] * dt * dzi/rho[k]
            b[k-gw] = 1.0 + Y/X + Z/X + 0.5 * dt * dzi * (massflux[k-1]-massflux[k])/rho[k]
            c[k-gw] = -Y/X - 0.5 * dt * dzi * massflux[k]/rho[k]

    return




cdef void construct_tridiag_diffusion_dirichlet(Py_ssize_t nzg, Py_ssize_t gw, double dzi, double dt,
                                 double *rho_ae_K_m, double *rho, double *ae, double *a, double *b, double *c):
    cdef:
        Py_ssize_t k
        double X, Y, Z #
        Py_ssize_t nz = nzg - 2* gw
    with nogil:
        for k in xrange(gw,nzg-gw):
            X = rho[k] * ae[k]/dt
            Y = rho_ae_K_m[k] * dzi * dzi
            Z = rho_ae_K_m[k-1] * dzi * dzi
            if k == gw:
                Z = 0.0
                Y = 0.0
            elif k == nzg-gw-1:
                Y = 0.0
            a[k-gw] = - Z/X
            b[k-gw] = 1.0 + Y/X + Z/X
            c[k-gw] = -Y/X

    return



cdef void tridiag_solve(Py_ssize_t nz, double *x, double *a, double *b, double *c):
    cdef:
        double * scratch = <double*> PyMem_Malloc(nz * sizeof(double))
        Py_ssize_t i
        double m

    scratch[0] = c[0]/b[0]
    x[0] = x[0]/b[0]

    with nogil:
        for i in xrange(1,nz):
            m = 1.0/(b[i] - a[i] * scratch[i-1])
            scratch[i] = c[i] * m
            x[i] = (x[i] - a[i] * x[i-1])*m


        for i in xrange(nz-2,-1,-1):
            x[i] = x[i] - scratch[i] * x[i+1]


    PyMem_Free(scratch)
    return








# Dustbin

cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag


