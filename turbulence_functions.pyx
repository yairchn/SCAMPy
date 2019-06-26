import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin, tanh
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
include "parameters.pxi"
from thermodynamic_functions cimport *

# Entrainment Rates
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

    eps_w = 1.0/(fmax(fabs(entr_in.w),1.0)* 1000)
    if entr_in.af>0.0:
        buoyant_frac  = buoyancy_sorting(entr_in)
        _ret.entr_sc = buoyant_frac*eps_w/2.0
        _ret.detr_sc = (1.0-buoyant_frac/2.0)*eps_w
    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
    return _ret


cdef entr_struct entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret

    #detr_alim = 0.12*del_bw2/(1+exp(-20.0*(entr_in.af-entr_in.au_lim)))
    #entr_alim = 0.12*eps_bw2/(1+exp( 20.0*(entr_in.af-0.0001)))
    #c_eps = sqrt(entr_in.af)
    c_eps = 0.12
    eps_bw2 = c_eps*fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)
    del_bw2 = c_eps*fabs(fmin(entr_in.b ,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
    del_bulk = 4.0e-3
    eps = c_eps*fabs(entr_in.b) / fmax(entr_in.w * entr_in.w, 1e-2)
    #eps =  1.0/(fmax(fabs(entr_in.w),0.001)*1000.0)
    #eps_bw2 = 1.0/(fmax(fabs(entr_in.w),1.0)*700.0)
    #esp = 0.0/entr_in.z

    if entr_in.af>0.0:
        c_eps = sqrt(entr_in.af)
        #temp = inter_critical_env_frac(entr_in)
        #_ret.chi_c = fmax(fmin(temp.x1,1.0),0.0)
        #buoyant_frac = stochastic_buoyancy_sorting(entr_in)
        buoyant_frac = buoyancy_sorting(entr_in)
        #_ret.entr_sc = _ret.chi_c**2.0*eps #+ entr_alim
        #_ret.detr_sc = (1.0 - _ret.chi_c)**2.0*eps #+ detr_alim
        _ret.entr_sc = buoyant_frac*eps #+ entr_alim
        _ret.detr_sc = (1.0-buoyant_frac)*eps #+ detr_alim
        #_ret.entr_sc = eps_bw2
        #_ret.detr_sc = del_bw2 #+ detr_alim
        #if entr_in.z >= entr_in.zi:
        #    _ret.detr_sc = eps #+ detr_alim
        _ret.buoyant_frac = buoyant_frac
    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
        _ret.buoyant_frac = 0.0

    return _ret


cdef double buoyancy_sorting(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t m_q, m_h
            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var
            double T_mix, ql_mix, qt_mix, qv_mix,bmix, alpha_mix
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim, qv_,
            double a, b_up, b_env, b_mean0, T_up
            double buoyant_frac = 0.0
            double inner_buoyant_frac = 0.0
            eos_struct sa
            double [:] weights
            double [:] abscissas

        with gil:
            abscissas, weights = np.polynomial.hermite.hermgauss(entr_in.quadrature_order)

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_env, entr_in.H_env)
        qv_ = entr_in.qt_env - sa.ql
        T_env = sa.T
        ql_env = sa.ql
        alpha_env = alpha_c(entr_in.p0, sa.T, entr_in.qt_env, qv_)
        b_env = buoyancy_c(entr_in.alpha0, alpha_env)

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_up, entr_in.H_up)
        qv_ = entr_in.qt_up - sa.ql
        T_up = sa.T
        ql_up = sa.ql
        alpha_up = alpha_c(entr_in.p0, sa.T, entr_in.qt_up, qv_)
        b_up = buoyancy_c(entr_in.alpha0, alpha_up)

        b_mean = entr_in.af*b_up +  (1.0-entr_in.af)*b_env

        a = 0.5

        if entr_in.env_QTvar != 0.0 and entr_in.env_Hvar != 0.0:
            sd_q = sqrt(entr_in.env_QTvar)
            sd_h = sqrt(entr_in.env_Hvar)
            #corr =  fmax(fmin(entr_in.env_HQTcov/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)
            corr =  entr_in.env_HQTcov/fmax(sd_h*sd_q, 1e-13)

            # limit sd_q to prevent negative qt_hat
            sd_q_lim = (1e-10 - entr_in.qt_env)/(sqrt2 * abscissas[0])
            sd_q = fmin(sd_q, sd_q_lim)
            qt_var = sd_q * sd_q
            sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h

            for m_q in xrange(entr_in.quadrature_order):
                qt_hat    = (entr_in.qt_env + sqrt2 * sd_q * abscissas[m_q])*(1-a) + a*entr_in.qt_up
                mu_h_star = entr_in.H_env + sqrt2 * corr * sd_h * abscissas[m_q]
                inner_buoyant_frac = 0.0
                for m_h in xrange(entr_in.quadrature_order):
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star)*(1-a) + a*entr_in.H_up
                    # condensation and calcualte buoyancy
                    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
                    qv_ = qt_hat - sa.ql
                    alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.alpha0, alpha_mix)  - b_mean #- entr_in.dw2dz/2.0

                    if bmix >0.0:
                        inner_buoyant_frac  += weights[m_h] * sqpi_inv
                buoyant_frac  += inner_buoyant_frac * weights[m_q] * sqpi_inv

        else:

            if b_up - b_mean  > 0.0: #- entr_in.dw2dz/2.0
                 buoyant_frac = 1.0

        return buoyant_frac

cdef double stochastic_buoyancy_sorting(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t i
            double Hmix, QTmix, corr, sigma_H, sigma_QT, bmix, alpha_mix,qv_, rand_H, rand_QT
            double a, b_up, b_env, b_mean0, T_up, buoyant_frac
            int n = 3
            eos_struct sa

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_env, entr_in.H_env)
        qv_ = entr_in.qt_env - sa.ql
        T_env = sa.T
        ql_env = sa.ql
        alpha_env = alpha_c(entr_in.p0, sa.T, entr_in.qt_env, qv_)
        b_env = buoyancy_c(entr_in.alpha0, alpha_env)

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_up, entr_in.H_up)
        qv_ = entr_in.qt_up - sa.ql
        T_up = sa.T
        ql_up = sa.ql
        alpha_up = alpha_c(entr_in.p0, sa.T, entr_in.qt_up, qv_)
        b_up = buoyancy_c(entr_in.alpha0, alpha_up)

        b_mean = entr_in.af*b_up +  (1.0-entr_in.af)*b_env
        sigma_QT = sqrt(entr_in.env_QTvar)
        corr    = entr_in.env_HQTcov/fmax(sqrt(entr_in.env_QTvar)*sqrt(entr_in.env_Hvar), 1e-13)
        sigma_H = sqrt(fmax(1.0-corr*corr,0.0)) * sqrt(entr_in.env_Hvar)
        #sigma_H = sqrt(entr_in.env_Hvar)
        buoyant_frac_s = 0.0
        for i in range(n):
            with gil:
                rand_QT = np.random.normal(entr_in.qt_env, sigma_QT ,1)
                rand_H  = np.random.normal(entr_in.H_env, sigma_H , 1)
            Hmix = (entr_in.H_up+rand_H)/2.0
            QTmix = (entr_in.qt_up+rand_QT)/2.0

            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0,  QTmix, Hmix)
            qv_ =  QTmix - sa.ql
            alpha_mix = alpha_c(entr_in.p0, sa.T, QTmix, qv_)
            bmix = buoyancy_c(entr_in.alpha0, alpha_mix)  - b_mean - entr_in.dw2dz/2.0

            if bmix>0:
                buoyant_frac +=1.0/float(n)

        return buoyant_frac

cdef chi_struct inter_critical_env_frac(entr_in_struct entr_in) nogil:
    cdef:
        chi_struct _ret
        double chi_c
        double ql_1, T_2, ql_2, f_1, f_2, qv_mix, T_1
        double b_up, b_mean, b_env
        double y0, y1, x0, x1, dx, dy,T_env, ql_env, T_up, ql_up ,T_mix, ql_mix, qt_mix, alpha_mix, b_mix
        double xatol=1e-4
        #int maxiters=10

    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_env, entr_in.H_env)
    qv_ = entr_in.qt_env - sa.ql
    T_env = sa.T
    ql_env = sa.ql
    alpha_env = alpha_c(entr_in.p0, sa.T, entr_in.qt_env, qv_)
    b_env = buoyancy_c(entr_in.alpha0, alpha_env)

    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_up, entr_in.H_up)
    qv_ = entr_in.qt_up - sa.ql
    T_up = sa.T
    ql_up = sa.ql
    alpha_up = alpha_c(entr_in.p0, sa.T, entr_in.qt_up, qv_)
    b_up = buoyancy_c(entr_in.alpha0, alpha_up)
    b_mean = entr_in.af*b_up +  (1.0-entr_in.af)*b_env
    b_up = b_up-b_mean
    b_env = b_env-b_mean

    x0 = 1.0
    y0 = b_env
    x1 = 0.0
    y1 = b_up

    for i in xrange(0, 10):
        dx = x1 - x0
        dy = y1 - y0
        x0 = x1
        y0 = y1
        if dy != 0.0:
            x1 -= y1 * dx / dy
            # f(x1) - calculate mixture buoyancy
            H_mix = (1.0-x1)*entr_in.H_up + x1*entr_in.H_env
            qt_mix = (1.0-x1)*entr_in.qt_up + x1*entr_in.qt_env
            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_mix, H_mix)
            ql_mix = sa.ql
            T_mix = sa.T
            qv_ = qt_mix - ql_mix
            alpha_mix = alpha_c(entr_in.p0, T_mix, qt_mix, qv_)
            b_mix = buoyancy_c(entr_in.alpha0, alpha_mix)-b_mean
            y1 = b_mix

            _ret.T_mix = T_mix
            _ret.ql_mix = ql_mix
            _ret.qt_mix = qt_mix
            _ret.qv_ = qv_
            _ret.alpha_mix = alpha_mix
            _ret.y1 = y1
            _ret.x1 = x1

            if fabs(x0-x1) < xatol:
                return _ret
        else:
            # with gil:
            #     print(418, dy, x0, y0, x1, y1)
            _ret.T_mix = entr_in.T_up
            _ret.ql_mix = entr_in.ql_up
            _ret.qt_mix = entr_in.qt_up
            _ret.qv_ = 0.0
            _ret.alpha_mix = entr_in.alpha0
            _ret.y1 = entr_in.b
            _ret.x1 = 0.5
            return _ret
    return _ret

cdef entr_struct entr_detr_tke2(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 3.0e-3
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = (0.05 * sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) / fmax(entr_in.af, 0.001) / fmax(entr_in.z, 1.0))
    return  _ret

# yair - this is a new entr-detr function that takes entr as proportional to TKE/w and detr ~ b/w2
cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    _ret.detr_sc = fabs(entr_in.b)/ fmax(entr_in.w * entr_in.w, 1e-3)
    _ret.entr_sc = sqrt(entr_in.tke) / fmax(entr_in.w, 0.01) / fmax(sqrt(entr_in.af), 0.001) / 50000.0
    return  _ret


cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil:
    cdef :
        entr_struct _ret
        double effective_buoyancy
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 4.0e-3 + 0.12 *fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = 0.12 * fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)

    return  _ret

cdef entr_struct entr_detr_suselj(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double entr_dry = 2.5e-3
        double l0

    l0 = (entr_in.zbl - entr_in.zi)/10.0
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
        _ret.entr_sc = 0.002 # 0.1 / entr_in.dz * entr_in.poisson

    else:
        _ret.detr_sc = 0.0
        _ret.entr_sc = 0.0 #entr_dry # Very low entrainment rate needed for Dycoms to work

    return  _ret

cdef entr_struct entr_detr_none(entr_in_struct entr_in)nogil:
    cdef entr_struct _ret
    _ret.entr_sc = 0.0
    _ret.detr_sc = 0.0

    return  _ret

# cdef evap_struct evap_sat_adjust(double p0, double thetal_, double qt_mix) nogil:
#     cdef:
#         evap_struct evap
#         double ql_1, T_2, ql_2, f_1, f_2, qv_mix, T_1

#     qv_mix = qt_mix
#     ql = 0.0

#     pv_1 = pv_c(p0,qt_mix,qt_mix)
#     pd_1 = p0 - pv_1

#     # evaporate and cool
#     T_1 = eos_first_guess_thetal(thetal_, pd_1, pv_1, qt_mix)
#     pv_star_1 = pv_star(T_1)
#     qv_star_1 = qv_star_c(p0,qt_mix,pv_star_1)

#     if(qt_mix <= qv_star_1):
#         evap.T = T_1
#         evap.ql = 0.0

#     else:
#         ql_1 = qt_mix - qv_star_1
#         prog_1 = t_to_thetali_c(p0, T_1, qt_mix, ql_1, 0.0)
#         f_1 = thetal_ - prog_1
#         T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt_mix)*cpd + qv_star_1 * cpv)
#         delta_T  = fabs(T_2 - T_1)

#         while delta_T > 1.0e-3 or ql_2 < 0.0:
#             pv_star_2 = pv_star(T_2)
#             qv_star_2 = qv_star_c(p0,qt_mix,pv_star_2)
#             pv_2 = pv_c(p0, qt_mix, qv_star_2)
#             pd_2 = p0 - pv_2
#             ql_2 = qt_mix - qv_star_2
#             prog_2 =  t_to_thetali_c(p0,T_2,qt_mix, ql_2, 0.0)
#             f_2 = thetal_ - prog_2
#             T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
#             T_1 = T_2
#             T_2 = T_n
#             f_1 = f_2
#             delta_T  = fabs(T_2 - T_1)

#         evap.T  = T_2
#         qv = qv_star_2
#         evap.ql = ql_2

#     return evap

# convective velocity scale
cdef double get_wstar(double bflux, double zi ):
    return cbrt(fmax(bflux * zi, 0.0))

# BL height
cdef double get_inversion(double *theta_rho, double *u, double *v, double *z_half,
                          Py_ssize_t kmin, Py_ssize_t kmax, double Ri_bulk_crit):
    cdef:
        double theta_rho_b = theta_rho[kmin]
        double h, Ri_bulk=0.0, Ri_bulk_low = 0.0
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

# Teixiera convective tau
cdef double get_mixing_tau(double zi, double wstar) nogil:
    # return 0.5 * zi / wstar
    #return zi / (fmax(wstar, 1e-5))
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


