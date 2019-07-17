import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin, tanh, erf
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
    buoyant_frac = buoyancy_sorting(entr_in)
    _ret.entr_sc = buoyant_frac*eps_w/2.0
    _ret.detr_sc = (1.0-buoyant_frac/2.0)*eps_w
    return _ret

cdef entr_struct entr_detr_env_moisture_deficit(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double chi_c, RH_env, RH_upd

    c_eps = sqrt(entr_in.af*(1.0-entr_in.af)) # Bomex
    RH_upd = entr_in.RH_upd
    RH_env = entr_in.RH_env
    # chi_struct = inter_critical_env_frac(entr_in)
    # _ret.buoyant_frac = buoyancy_sorting(entr_in)

    eps_bw2 = entr_in.c_eps*fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)
    del_bw2 = entr_in.c_eps*fabs(entr_in.b) / fmax(entr_in.w * entr_in.w, 1e-2)

    _ret.entr_sc = eps_bw2
    if entr_in.ql_up>0.0:
    # if entr_in.z>entr_in.zi:
        #_ret.detr_sc = del_bw2*(1.0+(RH_upd/fmax((RH_upd - RH_env),1.0))**2.0)
        _ret.detr_sc = del_bw2*(1.0+fmax((RH_upd - RH_env),0.0)/RH_upd)**6.0
        #_ret.detr_sc = del_bw2*(1.0+(entr_in.ql_up-1e-5)/(qv_s*(RH_upd/100.0 - RH_env/100.0)+1e-5))
    else:
        _ret.detr_sc =  del_bw2

    return _ret

cdef entr_struct entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil:

    cdef:
        entr_struct _ret
        double chi_c,eps_bw2, del_bw2, c_eps, beta, d, D_
        #c_eps = 0.03, beta = 0.05, d = 3.0
    c_eps = 0.03
    beta = 0.05
    d = 3.0
    ret_b = buoyancy_sorting_mean(entr_in)
    b_mix = ret_b.b_mix
    buoyant_frac = ret_b.buoyant_frac

    eps_bw2 = c_eps*fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)
    del_bw2 = c_eps*fabs(entr_in.b) / fmax(entr_in.w * entr_in.w, 1e-2)
    _ret.buoyant_frac = buoyant_frac
    _ret.b_mix = b_mix
    # _ret.entr_sc = eps_bw2*(fmax(buoyant_frac,0.0)) # yair
    _ret.entr_sc = eps_bw2 # Tapio
    if entr_in.ql_up>0.0:
        # _ret.detr_sc = del_bw2*(fmax(1.0-buoyant_frac,0.0))# yair
        D_ = 0.5*(1.0+erf(beta*(buoyant_frac)))# Tapio
        _ret.detr_sc = del_bw2*(1.0+d*D_)# Tapio
        # _ret.detr_sc = del_bw2*(1.0+d*erf(beta*(buoyant_frac)))# Tapio2
        # with gil:
        #     print(del_bw2*(1.0+d*D_),del_bw2*(fmax(1.0-buoyant_frac,0.0)))
    else:
        _ret.detr_sc = 0.0

    return _ret


cdef buoyant_stract buoyancy_sorting_mean(entr_in_struct entr_in) nogil:

        cdef:
            double qv_ ,T_env ,ql_env ,alpha_env ,b_env, T_up ,ql_up ,alpha_up ,b_up, b_mean, b_avg, b_mix, qt_mix , H_mix
            double buoyant_frac = 0.0
            eos_struct sa
            buoyant_stract ret_b

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
        b_avg = 0.5*b_up + 0.5*b_env - b_mean

        qt_mix = (entr_in.qt_up+entr_in.qt_env)/2.0
        H_mix = (entr_in.H_up+entr_in.H_env)/2.0
        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_mix, H_mix)
        qv_ = (entr_in.qt_up+entr_in.qt_env)/2.0 - sa.ql
        alpha_mix = alpha_c(entr_in.p0, sa.T, qt_mix, qv_)
        b_mix = buoyancy_c(entr_in.alpha0, alpha_mix)
        buoyant_frac = -(b_mix-b_mean)/fmax(fabs(b_up-b_env),0.0000001)
        ret_b.b_mix = b_mix - b_mean
        ret_b.buoyant_frac = buoyant_frac

        return ret_b

cdef double buoyancy_sorting(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t m_q, m_h
            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var
            double T_mix, ql_mix, qt_mix, qv_mix,bmix, alpha_mix
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim, qv_,
            double a, b_up, b_env, b_mean0, T_up
            #double[:] inner
            int i_b

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
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star + entr_in.H_up)/2.0
                    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
                    qv_ = qt_hat - sa.ql
                    L_ = latent_heat(sa.T)
                    dT = L_*((entr_in.ql_up+entr_in.ql_env)/2.0- sa.ql)/1004.0
                    alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.alpha0, alpha_mix)  - b_mean #- entr_in.dw2dz/2.0

                    if bmix >0.0:
                        inner_buoyant_frac  += weights[m_h] * sqpi_inv
                buoyant_frac  += inner_buoyant_frac * weights[m_q] * sqpi_inv
                   
        else:
            if b_up - b_mean  > 0.0:
                 buoyant_frac = 1.0
            else:
                buoyant_frac  = 0.0

        return buoyant_frac

cdef double stochastic_buoyancy_sorting(entr_in_struct entr_in) nogil:
        cdef:
            Py_ssize_t i
            double Hmix, QTmix, corr, sigma_H, sigma_QT, bmix, alpha_mix,qv_, rand_H, rand_QT, T_hat, qt_hat
            double a, b_up, b_env, b_mean0, T_up, buoyant_frac
            # double [:] mean
            # double [:,:] cov
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

        # cov[1,1] = entr_in.env_QTvar
        # cov[1,2] = entr_in.env_HQTcov
        # cov[2,1] = entr_in.env_HQTcov
        # cov[2,2] = entr_in.env_Hvar
        # with gil:
        #     print('-----------> buoyant_frac_s')
        for i in range(n):
            with gil:
                rand_QT,rand_H = np.random.multivariate_normal([entr_in.qt_env,entr_in.H_env],[[entr_in.env_QTvar,entr_in.env_HQTcov],[entr_in.env_HQTcov,entr_in.env_Hvar]], 1).T

            # compute h_hat from mixing T and qt
            QTmix = (entr_in.qt_up+rand_QT)/2.0
            h_hat = rand_H
            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, QTmix, h_hat)
            T_hat = (entr_in.T_up+sa.T)/2.0
            h_hat = thetali_mix(entr_in.p0, qt_hat, T_hat)
            with gil:
                if np.isnan(h_hat):
                    print(sa.T, h_hat)
            # with gil:
            #     print(sa.T, h_hat)

            Hmix = (entr_in.H_up+h_hat)/2.0

            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0,  QTmix, Hmix)
            qv_ =  QTmix - sa.ql
            alpha_mix = alpha_c(entr_in.p0, sa.T, QTmix, qv_)
            bmix = buoyancy_c(entr_in.alpha0, alpha_mix)  - b_mean #- entr_in.dw2dz/2.0

            if bmix>0:
                buoyant_frac_s +=1.0/float(n)
            else:
                with gil:
                    print('bmix = ',bmix, Hmix, QTmix, h_hat, T_hat, rand_H, rand_QT)
            # with gil:
            #     print(i, Hmix,entr_in.H_up, rand_H, QTmix, entr_in.qt_up, rand_QT)
            #     print(entr_in.env_QTvar,entr_in.env_HQTcov,entr_in.env_Hvar)
        # with gil:
        #         print(buoyant_frac_s)
        return buoyant_frac_s

cdef double critical_env_frac_RH(entr_in_struct entr_in) nogil:
    cdef:
        double qv_s, qv, Xc, chi_c, pv

    RH_upd = entr_in.RH_upd
    RH_env = entr_in.RH_env

    # if entr_in.ql_up>0.0:
    #     qv     = entr_in.qt_env-entr_in.ql_env
    #     pv     = pv_c(entr_in.p0, entr_in.qt_env, qv)
    #     qv_s   = qv_star_c(entr_in.p0, entr_in.qt_env, pv)
    #     Xc     = (entr_in.ql_up - 1e-5)/((qv_s-qv)+1e-5)
    #     chi_c  = Xc/(1.0+Xc)
    #     t_a = 1.0
    #     a = k/t_a*(ql0/1e-5) + (qv_s-qv)/1e-5
    #     b = k/t_a + (qv_s-qv)/1e-5
    #     t_new = t_a*
    #     t_mix = k*(Xc+Xc*Xc/2.0)
    #     with gil:
    #         print(qv ,qv_s ,Xc ,(entr_in.ql_up - 1e-5),((qv_s-qv)+1e-5),chi_c)
    # else:
    #     chi_c  = 1.0-entr_in.af
    chi_c  = 1.0-entr_in.af

    return chi_c

cdef chi_struct inter_critical_env_frac(entr_in_struct entr_in) nogil:
    cdef:
        chi_struct _ret
        double chi_c
        double ql_1, T_2, ql_2, f_1, f_2, qv_mix, T_1
        double b_up, b_mean, b_env
        double y0, y1, x0, x1, dx, dy,T_env, ql_env, T_up, ql_up ,T_mix, ql_mix, qt_mix, alpha_mix, b_mix
        double xatol=1e-3
        double lastside = 0.0
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


    for i in xrange(0, 20):
        # dx = x1 - x0
        # dy = y1 - y0
        # x0 = x1
        # y0 = y1
        # # if dy != 0.0:
        # #     while fabs(y1)>xatol:
        # x1 -= y1 * dx / dy
        x = (x0 * y1 - x1 * y0)/ (y1 - y0)
        # f(x1) - calculate mixture buoyancy
        H_mix = (1.0-x)*entr_in.H_up + x*entr_in.H_env
        qt_mix = (1.0-x)*entr_in.qt_up + x*entr_in.qt_env
        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_mix, H_mix)
        ql_mix = sa.ql
        T_mix = sa.T
        qv_ = qt_mix - ql_mix
        alpha_mix = alpha_c(entr_in.p0, T_mix, qt_mix, qv_)
        b_mix = buoyancy_c(entr_in.alpha0, alpha_mix)-b_mean
        y = b_mix
        if y*y0<0:
            if fabs(x-x1)<=xatol:
                _ret.T_mix = T_mix
                _ret.ql_mix = ql_mix
                _ret.qt_mix = qt_mix
                _ret.qv_ = qv_
                _ret.alpha_mix = alpha_mix
                _ret.y1 = y1
                _ret.x1 = x1
                return _ret
            x1 = x
            y1 = y
            if lastside == 1.0:
                y0= y0/2.0
            lastside += 1.0
        else:
            if fabs(x0-x) <= xatol:
                return _ret
            x0 = x
            y0 = y
            if lastside == 1.0:
                y1 = y1/2.0
            lastside = -1.0



        # if fabs(y1)<xatol:
            
        #     with gil:
        #         print('secent worked')

        #     return _ret

        #         # with gil:
        #         #     if x1>1.0 or x1<0.0:
        #         #         print('x1',x1,'y1', y1)
        #         #         print('T_up', T_up,'T_env', T_env, 'T_mix', T_mix)
        #         #         print('ql_up', ql_up, 'ql_env', ql_env, 'ql_mix', ql_mix)
        #         #         print('alpha_up', alpha_up, 'alpha_env', alpha_env, 'alpha_mix', alpha_mix)

        #     # if fabs(x0-x1) < xatol:
        #         # return _ret
        # else:
        #     with gil:
        #         print('secent fails')
        #     _ret.T_mix = entr_in.T_up
        #     _ret.ql_mix = entr_in.ql_up
        #     _ret.qt_mix = entr_in.qt_up
        #     _ret.qv_ = 0.0
        #     _ret.alpha_mix = entr_in.alpha0
        #     _ret.y1 = entr_in.b
        #     _ret.x1 = 0.5
        #     return _ret
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

cdef double thetali_mix(double p0, double qt, double T) nogil:
    cdef:
        double qv, pv, pv_star_, qv_star_, thetali, pi, L
        double qi = 0.0

    pv = pv_c(p0,qt,qt)
    pv_star_ = pv_star(T)
    qv_star_ = qv_star_c(p0,qt,pv_star_)
    ql = qt-qv_star_
    pi = exner_c(p0, kappa)
    L = latent_heat(T)
    thetali = thetali_c(p0, T, qt, ql, qi, L)
    with gil:
        if np.isnan(thetali):
            print(p0, qt, T,pv ,pv_star_ ,qv_star_ ,ql ,pi ,L ,thetali)
    return thetali

# Dustbin

cdef bint set_cloudbase_flag(double ql, bint current_flag) nogil:
    cdef bint new_flag
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag


