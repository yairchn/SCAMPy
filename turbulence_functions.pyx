import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin, tanh, erf
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
include "parameters.pxi"
from thermodynamic_functions cimport *
from utility_functions import *
import pylab as plt

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


cdef entr_struct entr_detr_upd_specific(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret

    for i in range(entr_in.n_updrafts):
        if entr_in.upd_number == 0:
            eps_w = 1.0/(fmax(entr_in.w,1.0)*1000)
            if entr_in.af>0.0:
                partiation_func  = entr_detr_buoyancy_sorting_mean(entr_in)
                _ret.entr_sc = partiation_func*eps_w
                _ret.detr_sc = (1.0-partiation_func)*eps_w

            else:
                _ret.entr_sc = 0.0
                _ret.detr_sc = 0.0

        elif entr_in.upd_number == 1:
            if entr_in.z >= entr_in.zi :
            #if entr_in.ql_up >= 0.0:
                _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-9)
            else:
                _ret.detr_sc = 0.0

            _ret.entr_sc = 0.12 * fmax(entr_in.b ,0.0) / fmax(entr_in.w * entr_in.w, 1e-9)# + entr_in.press

        else:
            if entr_in.z >= entr_in.zi :
            #if entr_in.ql_up >= 0.0:
                _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-9)
            else:
                _ret.detr_sc = 0.0

            _ret.entr_sc = 0.12 * fmax(entr_in.b ,0.0) / fmax(entr_in.w * entr_in.w, 1e-9)# + entr_in.press


    return _ret



cdef entr_struct entr_detr_Poisson_entr(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double entr_dry =  10000.0
        double bmix
    bmix = entr_in.beta*entr_in.b+(1.0-entr_in.beta)*entr_in.b_env
    #entr_dry = 0.12*entr_in.b/fmax(entr_in.w**2,0.0001)
    _ret.entr_sc= fmax(bmix,0.0)*entr_dry*entr_in.entr_poisson/entr_in.z
    _ret.detr_sc = fmin(bmix,0.0)*entr_dry*entr_in.entr_poisson/entr_in.z #
    if _ret.entr_sc>0.00:
        with gil:
            print _ret.detr_sc, _ret.entr_sc, entr_in.z, entr_in.entr_poisson

    return  _ret

cdef entr_struct entr_detr_linear_sum(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double esp_w, eps_b_w2, esp_suselj
        double del_w, del_b_w2, del_suselj
        double entr_dry = 2.5e-3
        double l0

    eps_ = 1.0/(fmax(fabs(entr_in.w),1.0)* 500)
    if entr_in.af>0.0:
        partiation_func  = entr_detr_buoyancy_sorting(entr_in)
        eps_w = partiation_func*eps_/2.0
        del_w = (1.0-partiation_func/2.0)*eps_
    else:
        eps_w = 0.0
        del_w = 0.0


    if entr_in.z >= entr_in.zi :
        del_b_w2= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
    else:
        del_b_w2 = 0.0

    eps_b_w2 = 0.12 * fmax(entr_in.b,0.0) / fmax(entr_in.w * entr_in.w, 1e-2)



    l0 = (entr_in.zbl - entr_in.zi)/10.0
    if entr_in.z >= entr_in.zi :
        eps_suselj= 4.0e-3 +  0.12* fabs(fmin(entr_in.b,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
        del_suselj = 0.1 / entr_in.dz * entr_in.poisson

    else:
        eps_suselj = 0.0
        del_suselj = 0.0


    _ret.entr_sc = 0.5*sqrt(eps_w*eps_w + eps_b_w2*eps_b_w2 + eps_suselj*eps_suselj )
    _ret.detr_sc = 0.5*sqrt(del_w*del_w + del_b_w2*del_b_w2 + del_suselj*del_suselj )

    return _ret


cdef entr_struct entr_detr_inverse_w(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double alpha_up, b_up

    #eps_w = sqrt(entr_in.tke)/(fmax(entr_in.w,1.0)*entr_in.rd*sqrt(fmax(entr_in.af,0.0001)))
    eps_w = 1.0/(fmax(entr_in.w,0.1)*500.0)
    #eps_w = 0.1*entr_in.dbdz/fmax(entr_in.b,0.0001)
    #eps_w = 0.15*(entr_in.b-entr_in.b_env)/fmax((entr_in.w-entr_in.w_env)**2.0,0.0001)
    #eps_w = 0.0012
    # somewhere between 0.01 and 0.05 might be it
    # alpha_up = alpha_c(entr_in.p0, entr_in.T_up, entr_in.qt_up, entr_in.qt_up-entr_in.ql_up)
    # b_up = buoyancy_c(entr_in.alpha0, alpha_up) - entr_in.b_mean# - entr_in.b_mean #
    # with gil:
    #     print b_up, entr_in.b, alpha_up
    #     plt.figure()
    #     plt.show()


    if entr_in.af>0.0:
        partiation_func  = entr_detr_buoyancy_sorting_mean(entr_in)
        #partiation_func = entr_in.normalized_skew
        #with gil:
        #    print partiation_func
        _ret.entr_sc = (partiation_func)*eps_w # - entr_in.normalized_skew/1000.0
        _ret.detr_sc = (1.0-partiation_func)*eps_w


        #if entr_in.z>entr_in.zi and entr_in.z <2000.0:
        #    _ret.detr_sc += 1.0e-3

    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
    return _ret



cdef entr_struct entr_detr_functional_form(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double a1, a2, a3, b1, b2, b3, epsilon1, epsilon2, epsilon3, epsilon, delta, epsilon4, delta4
        double partiation_func, pi1, pi2, pi3, pi4, pi5, pi6, logb, eps_w

    if entr_in.af>0.0:

        pi1 = entr_in.tke/fmax(entr_in.w**2,1e-2)
        pi2 = fmax(entr_in.b,0.0)*entr_in.rd/fmax(fabs(entr_in.tke),1e-2)
        pi3 = fmax(entr_in.b,0.0)*entr_in.rd/fmax(entr_in.w**2,1e-2)
        pi4 = entr_in.tke/fmax(entr_in.w**2,1e-2)
        pi5 = fabs(fmin(entr_in.b,0.0))*entr_in.rd/fmax(fabs(entr_in.tke),1e-2)
        pi6 = fabs(fmin(entr_in.b,0.0))*entr_in.rd/fmax(entr_in.w **2.0, 1e-2)

        # pi1 = entr_in.tke/fmax(entr_in.w**2,1e-2)
        # pi2 = entr_in.b*entr_in.rd/fmax(entr_in.tke,1e-2)
        # pi3 = entr_in.b*entr_in.rd/fmax(entr_in.w**2,1e-2)

        a1 = pow(pi1,entr_in.alpha1e)*pow(pi2,entr_in.alpha2e)*pow(pi3,entr_in.alpha3e)
        a2 = pow(pi1,entr_in.alpha3e)*pow(pi2,entr_in.alpha1e)*pow(pi3,entr_in.alpha2e)
        a3 = pow(pi1,entr_in.alpha2e)*pow(pi2,entr_in.alpha3e)*pow(pi3,entr_in.alpha1e)

        b1 = pow(pi4,entr_in.alpha1d)*pow(pi5,entr_in.alpha2d)*pow(pi6,entr_in.alpha3d)
        b2 = pow(pi4,entr_in.alpha3d)*pow(pi5,entr_in.alpha1d)*pow(pi6,entr_in.alpha2d)
        b3 = pow(pi4,entr_in.alpha2d)*pow(pi5,entr_in.alpha3d)*pow(pi6,entr_in.alpha1d)

        epsilon1 = 1.0/entr_in.rd/sqrt(fmax(entr_in.af,0.001))
        epsilon2 = (fmax(entr_in.b,0.0)+entr_in.press)/fmax(entr_in.w*entr_in.w,0.01)
        epsilon3 = (fmax(entr_in.b,0.0)+entr_in.press)/fmax(entr_in.tke,0.01)
        epsilon4 = -entr_in.dbdz/fmax(entr_in.b,0.0001)

        delta1 = 1.0/entr_in.rd/sqrt(fmax(entr_in.af,0.001))
        delta2 = (fmax(-entr_in.b,0.0)+entr_in.press)/fmax(entr_in.w*entr_in.w,0.01)
        delta3 = (fmax(-entr_in.b,0.0)+entr_in.press)/fmax(entr_in.tke,0.01)
        delta4 = -entr_in.dbdz/fmax(entr_in.b,0.0001)


        #epsilon = (a1*epsilon1 + a2*epsilon2 + a3*epsilon3)/3.0
        #delta = (b1*delta1 + b2*delta2 + b3*delta3)/3.0

        logb  = entr_detr_buoyancy_sorting_mean(entr_in)

        eps_w = pi1*sqrt(epsilon1*epsilon1+epsilon4*epsilon4)
        del_w = sqrt(delta2*delta2 + delta1*delta1 + delta4*delta4)*pi1
        _ret.entr_sc = eps_w*logb/2.0
        _ret.detr_sc = del_w*(1.0-logb)/2.0

    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
    return _ret

cdef double entr_detr_buoyancy_sorting_mean(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double wdw_mix, wdw_env, T_mean, ql_mix, qv_mix, qt_mix, w_mix, dw_mix, b_env, alpha_up, alpha_env

    w_mix =  entr_in.beta*entr_in.w + (1.0-entr_in.beta)*entr_in.w_env
    dw_mix = entr_in.beta*entr_in.dw + (1.0-entr_in.beta)*entr_in.dw_env
    #entr_in.beta = 0.5
    qt_mix = entr_in.beta*entr_in.qt_up + (1.0-entr_in.beta)*entr_in.qt_env
    H_mix = entr_in.beta*entr_in.H_up + (1.0-entr_in.beta)*entr_in.H_env
    evap = evap_sat_adjust(entr_in.p0, H_mix, qt_mix)
    qv_mix = qt_mix-evap.ql
    Tmix = evap.T
    alpha_mix = alpha_c(entr_in.p0, Tmix, qt_mix, qv_mix)
    wdw_mix = w_mix*dw_mix
    bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean
    #bmix = bmix + w_mix*w_mix/2.0

    if  bmix > 0.0:
        partiation_func = 1.0
    else:
        partiation_func = 0.0

    # b_env = entr_in.b_env
    # b_up = entr_in.b
    # wdw_mix = w_mix*dw_mix
    # #b_rel = b_up-b_env+(entr_in.w-entr_in.w_env)*(entr_in.w-entr_in.w_env)/40.0
    #
    # brel_mix = bmix# + wdw_mix
    # brel_env = b_env# + wdw_env
    # brel_up = b_up# + wdw_up
    # x0 = brel_mix/fmax(fabs(brel_env), 1e-8)
    # sigma = entr_in.Poisson_rand*(brel_up-brel_env)/fmax(fabs(brel_env), 1e-8)
    # if sigma == 0.0:
    #     partition_func = 0.5
    # else:
    #     partition_func = (1-erf((brel_env/fmax(fabs(brel_env), 1e-8)-x0)/(1.4142135623*sigma)))/2
    #
    #

    #
    #
    # wdw_env = entr_in.w_env*entr_in.dw_env
    # wdw_up = entr_in.w*entr_in.dw
    #
    # partiation_func = 1.0*(1.0+tanh(buoyancy_ratio))/2.0

    return partiation_func

cdef double entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t m_q, m_h
            #double[:] inner
            int i_b

            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim, bmix, qv_
            double partiation_func = 0.0
            double inner_partiation_func = 0.0
            eos_struct sa
            double [:] weights
            double [:] abscissas
        with gil:
            abscissas, weights = np.polynomial.hermite.hermgauss(entr_in.quadrature_order)
            #print np.multiply(weights[0],1.0), np.multiply(weights[1],1.0), np.multiply(weights[2],1.0)

        if entr_in.env_QTvar > 0.0 and entr_in.env_Hvar > 0.0:
            sd_q = sqrt(entr_in.env_QTvar)
            sd_h = sqrt(entr_in.env_Hvar)
            corr = fmax(fmin(entr_in.env_HQTcov/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

            # limit sd_q to prevent negative qt_hat
            sd_q_lim = (1e-10 - entr_in.qt_env)/(sqrt2 * abscissas[0])
            sd_q = fmin(sd_q, sd_q_lim)
            qt_var = sd_q * sd_q
            sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h

            for m_q in xrange(entr_in.quadrature_order):
                qt_hat    = (entr_in.qt_env + sqrt2 * sd_q * abscissas[m_q] + entr_in.qt_up)/2.0
                mu_h_star = entr_in.H_env + sqrt2 * corr * sd_h * abscissas[m_q]
                inner_partiation_func = 0.0
                for m_h in xrange(entr_in.quadrature_order):
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star + entr_in.H_up)/2.0
                    # condensation
                    #sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
                    evap = evap_sat_adjust(entr_in.p0,h_hat, qt_hat)
                    # calcualte buoyancy
                    qv_ = qt_hat - evap.ql
                    alpha_mix = alpha_c(entr_in.p0, evap.T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean

                    # sum only the points with positive buoyancy to get the buoyant fraction
                    if bmix >0.0:
                        inner_partiation_func  += weights[m_h] * sqpi_inv
                partiation_func  += inner_partiation_func * weights[m_q] * sqpi_inv

        else:
            h_hat = ( entr_in.H_env + entr_in.H_up)/2.0
            qt_hat = ( entr_in.qt_env + entr_in.qt_up)/2.0

            # condensation
            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
            # calcualte buoyancy
            alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qt_hat - sa.ql)
            bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean

        return partiation_func


cdef double entr_detr_buoyancy_sorting2(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t m_q, m_h
            #double[:] inner
            int i_b

            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var, b_up,alpha_up,b_env, alpha_env
            double upd_h_hat, upd_qt_hat, upd_sd_h, upd_sd_q, upd_corr, upd_mu_h_star, upd_sigma_h_star, upd_qt_var
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim,upd_sd_q_lim, bmix, qv_
            double partiation_func = 0.0
            double inner_partiation_func = 0.0
            eos_struct sa
            double [:] weights
            double [:] abscissas
        with gil:
            abscissas, weights = np.polynomial.hermite.hermgauss(entr_in.quadrature_order)

        if entr_in.env_QTvar*entr_in.env_Hvar*entr_in.upd_QTvar*entr_in.upd_Hvar > 0.0:
            sd_q = sqrt(entr_in.env_QTvar)
            sd_h = sqrt(entr_in.env_Hvar)
            corr = fmax(fmin(entr_in.env_HQTcov/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

            upd_sd_q = sqrt(entr_in.upd_QTvar)
            upd_sd_h = sqrt(entr_in.upd_Hvar)
            upd_corr = fmax(fmin(entr_in.upd_HQTcov/fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

            # limit sd_q to prevent negative qt_hat
            sd_q_lim = (1e-10 - entr_in.qt_env)/(sqrt2 * abscissas[0])
            sd_q = fmin(sd_q, sd_q_lim)
            qt_var = sd_q * sd_q
            sigma_h_star = sqrt(fmax(1.0-corr*corr,0.0)) * sd_h

            upd_sd_q_lim = (1e-10 - entr_in.qt_up)/(sqrt2 * abscissas[0])
            upd_sd_q = fmin(upd_sd_q, upd_sd_q_lim)
            upd_qt_var = upd_sd_q * upd_sd_q
            upd_sigma_h_star = sqrt(fmax(1.0-upd_corr*upd_corr,0.0)) * upd_sd_h
            alpha_env = alpha_c(entr_in.p0, entr_in.T_env, entr_in.qt_env, entr_in.qt_env-entr_in.ql_env)
            b_env = buoyancy_c(entr_in.alpha0, alpha_env) # - entr_in.b_mean #


            for m_q in xrange(entr_in.quadrature_order):
                qt_hat    = (entr_in.qt_env + sqrt2 * sd_q * abscissas[m_q] + entr_in.qt_up +sqrt2 * upd_sd_q * abscissas[m_q])/2.0
                mu_h_star = entr_in.H_env + sqrt2 * corr * sd_h * abscissas[m_q]
                upd_mu_h_star = entr_in.H_up + sqrt2 * upd_corr * upd_sd_h * abscissas[m_q]
                inner_partiation_func = 0.0
                for m_h in xrange(entr_in.quadrature_order):
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star +
                            sqrt2 * upd_sigma_h_star * abscissas[m_h] + upd_mu_h_star )/2.0
                    # condensation
                    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
                    # calcualte buoyancy
                    qv_ = qt_hat - sa.ql
                    alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.alpha0, alpha_mix)# + (entr_in.w)**2/2.0 #


                    # sum only the points with positive buoyancy to get the buoyant fraction
                    if bmix > b_env:
                        inner_partiation_func  += weights[m_h] * sqpi_inv

                partiation_func  += inner_partiation_func * weights[m_q] * sqpi_inv

        else:


            h_hat = ( entr_in.H_env + entr_in.H_up)/2.0
            qt_hat = ( entr_in.qt_env + entr_in.qt_up)/2.0

            # condensation
            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
            # calcualte buoyancy
            alpha_mix = alpha_c(entr_in.p0, sa.T, qt_hat, qt_hat - sa.ql)
            bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean
            if bmix >0.0:
                partiation_func  = 1.0
            else:
                partiation_func  = 0.0

        return partiation_func




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
    cdef:
        entr_struct _ret
        double press

    # in cloud portion from Soares 2004
    if entr_in.af > 0.0:
        press = entr_in.alpha0*entr_in.press/entr_in.af
        if entr_in.z >= entr_in.zi :
        #if entr_in.ql_up >= 0.0:
            _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b ,0.0)) / fmax(entr_in.w * entr_in.w, 1e-2)
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
        _ret.entr_sc = 0.1 / entr_in.dz * entr_in.poisson

    else:
        _ret.detr_sc = 0.0
        _ret.entr_sc = 0.0 #entr_dry # Very low entrainment rate needed for Dycoms to work

    return  _ret

cdef entr_struct entr_detr_none(entr_in_struct entr_in)nogil:
    cdef entr_struct _ret
    _ret.entr_sc = 0.0001
    _ret.detr_sc = 0.0001
    #if entr_in.z >= entr_in.zi :
    #    _ret.detr_sc = 0.0000000001

    return  _ret

cdef evap_struct evap_sat_adjust(double p0, double thetal_, double qt_mix) nogil:
    cdef:
        evap_struct evap
        double ql_1, T_2, ql_2, f_1, f_2, qv_mix, T_1

    qv_mix = qt_mix
    ql = 0.0

    pv_1 = pv_c(p0,qt_mix,qt_mix)
    pd_1 = p0 - pv_1

    # evaporate and cool
    T_1 = eos_first_guess_thetal(thetal_, pd_1, pv_1, qt_mix)
    pv_star_1 = pv_star(T_1)
    qv_star_1 = qv_star_c(p0,qt_mix,pv_star_1)

    if(qt_mix <= qv_star_1):
        evap.T = T_1
        evap.ql = 0.0

    else:
        ql_1 = qt_mix - qv_star_1
        prog_1 = t_to_thetali_c(p0, T_1, qt_mix, ql_1, 0.0)
        f_1 = thetal_ - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt_mix)*cpd + qv_star_1 * cpv)
        delta_T  = fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p0,qt_mix,pv_star_2)
            pv_2 = pv_c(p0, qt_mix, qv_star_2)
            pd_2 = p0 - pv_2
            ql_2 = qt_mix - qv_star_2
            prog_2 =  t_to_thetali_c(p0,T_2,qt_mix, ql_2, 0.0)
            f_2 = thetal_ - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = fabs(T_2 - T_1)

        evap.T  = T_2
        qv = qv_star_2
        evap.ql = ql_2

    return evap

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
    return zi / (fmax(wstar, 1e-5))




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


