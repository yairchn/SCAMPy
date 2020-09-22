import numpy as np
cimport numpy as np
from libc.math cimport cbrt, sqrt, log, fabs,atan, exp, fmax, pow, fmin, tanh, erf, sin
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
include "parameters.pxi"
from thermodynamic_functions cimport *
from utility_functions cimport *

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

    eps_w = 1.0/(fmax(fabs(entr_in.w_upd),1.0)* 1000)
    if entr_in.a_upd>0.0:
        sorting_function  = buoyancy_sorting(entr_in)
        _ret.entr_sc = sorting_function*eps_w/2.0
        _ret.detr_sc = (1.0-sorting_function/2.0)*eps_w
    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
    return _ret

cdef entr_struct entr_detr_env_moisture_deficit_b_ED_MF(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double moisture_deficit_e, moisture_deficit_d, c_det, mu, db, dw, logistic_e, logistic_d, ed_mf_ratio, bmix
        double l[2]

    moisture_deficit_d = (fmax((entr_in.RH_upd/100.0)**entr_in.sort_pow-(entr_in.RH_env/100.0)**entr_in.sort_pow,0.0))**(1.0/entr_in.sort_pow)
    moisture_deficit_e = (fmax((entr_in.RH_env/100.0)**entr_in.sort_pow-(entr_in.RH_upd/100.0)**entr_in.sort_pow,0.0))**(1.0/entr_in.sort_pow)
    _ret.sorting_function = moisture_deficit_e
    c_det = entr_in.c_det
    if (entr_in.ql_up+entr_in.ql_env)==0.0:
        c_det = 0.0

    dw   = entr_in.w_upd - entr_in.w_env
    if dw < 0.0:
        dw -= 0.001
    else:
        dw += 0.001

    db = (entr_in.b_upd - entr_in.b_env)
    mu = entr_in.c_mu/entr_in.c_mu0

    inv_timescale = fabs(db/dw)
    logistic_e = 1.0/(1.0+exp(-mu*db/dw*(entr_in.chi_upd - entr_in.a_upd/(entr_in.a_upd+entr_in.a_env))))
    logistic_d = 1.0/(1.0+exp( mu*db/dw*(entr_in.chi_upd - entr_in.a_upd/(entr_in.a_upd+entr_in.a_env))))

    #Logistic of buoyancy fluxes
    inv_timescale = fabs(db/dw)
    ed_mf_ratio = fabs(entr_in.buoy_ed_flux)/(fabs(entr_in.a_upd*entr_in.a_env*(entr_in.w_upd-entr_in.w_env)*(entr_in.b_upd - entr_in.b_env))+1e-8)
    logistic_e *= (1.0/(1.0+exp(entr_in.c_ed_mf*(ed_mf_ratio-1.0))))
    _ret.entr_sc = inv_timescale/dw*(entr_in.c_ent*logistic_e + c_det*moisture_deficit_e)
    _ret.detr_sc = inv_timescale/dw*(entr_in.c_ent*logistic_d + c_det*moisture_deficit_d)

    return _ret

cdef entr_struct entr_detr_env_moisture_deficit(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double moisture_deficit_e, moisture_deficit_d, c_det, mu, db, dw, logistic_e, logistic_d, ed_mf_ratio, bmix
        double l[2]

    moisture_deficit_d = (fmax((entr_in.RH_upd/100.0)**entr_in.sort_pow-(entr_in.RH_env/100.0)**entr_in.sort_pow,0.0))**(1.0/entr_in.sort_pow)
    moisture_deficit_e = (fmax((entr_in.RH_env/100.0)**entr_in.sort_pow-(entr_in.RH_upd/100.0)**entr_in.sort_pow,0.0))**(1.0/entr_in.sort_pow)
    _ret.sorting_function = moisture_deficit_e
    c_det = entr_in.c_det
    if (entr_in.ql_up+entr_in.ql_env)==0.0:
        c_det = 0.0

    dw   = entr_in.w_upd - entr_in.w_env
    if dw < 0.0:
        dw = fmin(-0.1,dw)
    else:
        dw = fmax(0.1,dw)

    db = (entr_in.b_upd - entr_in.b_env)
    # with gil:
    #     print('b in entr ',entr_in.b_upd, entr_in.b_env)
    mu = entr_in.c_mu/entr_in.c_mu0

    inv_timescale = fabs(db/dw)
    logistic_e = 1.0/(1.0+exp(-mu*db/dw*(entr_in.chi_upd - entr_in.a_upd/(entr_in.a_upd+entr_in.a_env))))
    logistic_d = 1.0/(1.0+exp( mu*db/dw*(entr_in.chi_upd - entr_in.a_upd/(entr_in.a_upd+entr_in.a_env))))

    #smooth min
    # with gil:
    #     l[0] = entr_in.tke_coef*fabs(db/sqrt(entr_in.tke+1e-8))
    #     l[1] = fabs(db/dw)
    #     inv_timescale = lamb_smooth_minimum(l, 0.1, 0.0005)
    _ret.entr_sc = inv_timescale/dw*(entr_in.c_ent*logistic_e + c_det*moisture_deficit_e)
    _ret.detr_sc = inv_timescale/dw#*(entr_in.c_ent*logistic_d + c_det*moisture_deficit_d)

    amin2 = 0.00001 #entr_in.amin*entr_in.amin
    # entr_lim = (1.0 + 10.0*exp(-entr_in.a_upd**2.0/(2*amin2)) - exp(-(1.0-entr_in.a_upd)**2/(2*amin2)))
    # entr_lim = (1.0 + 0.004*exp(-entr_in.a_upd**2.0/(2*amin2)) - 0.004*exp(-(1.0-entr_in.a_upd)**2/(2*amin2)))
    entr_lim1 =  0.4 * exp(     -entr_in.a_upd **2.0/(2*amin2))
    entr_lim2 =  1.0 - exp(-(1.0-entr_in.a_upd)**2.0/(2*amin2))
    detr_lim1 =  0.4 * exp(-(1.0-entr_in.a_upd)**2.0/(2*amin2))
    detr_lim2 =  1.0 - exp(     -entr_in.a_upd **2.0/(2*amin2))

    # detr_lim = 1.0 - 0.004*exp(-entr_in.a_upd**2/(2*amin2)) + 0.004*exp(-(1.0-entr_in.a_upd)**2/(2*amin2))
    # _ret.entr_sc += entr_lim1*fabs(dw)/dw
    # _ret.entr_sc *= entr_lim2
    # _ret.detr_sc += detr_lim1*fabs(dw)/dw
    # _ret.detr_sc *= detr_lim2

    return _ret

cdef entr_struct entr_detr_env_moisture_deficit_div(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double moisture_deficit_e, moisture_deficit_d, c_det, mu, db, dw, logistic_e, logistic_d, ed_mf_ratio, bmix
        double l[2]

    moisture_deficit_d = (fmax((entr_in.RH_upd/100.0)**entr_in.sort_pow-(entr_in.RH_env/100.0)**entr_in.sort_pow,0.0))**(1.0/entr_in.sort_pow)
    moisture_deficit_e = (fmax((entr_in.RH_env/100.0)**entr_in.sort_pow-(entr_in.RH_upd/100.0)**entr_in.sort_pow,0.0))**(1.0/entr_in.sort_pow)
    _ret.sorting_function = moisture_deficit_e
    c_det = entr_in.c_det
    if (entr_in.ql_up+entr_in.ql_env)==0.0:
        c_det = 0.0

    dw   = entr_in.w_upd - entr_in.w_env
    if dw < 0.0:
        dw -= 0.001
    else:
        dw += 0.001

    db = (entr_in.b_upd - entr_in.b_env)
    mu = entr_in.c_mu/entr_in.c_mu0

    inv_timescale = fabs(db/dw)
    logistic_e = 1.0/(1.0+exp(-mu*db/dw*(entr_in.chi_upd - entr_in.a_upd/(entr_in.a_upd+entr_in.a_env))))
    logistic_d = 1.0/(1.0+exp( mu*db/dw*(entr_in.chi_upd - entr_in.a_upd/(entr_in.a_upd+entr_in.a_env))))

    entr_MdMdz = fmax( entr_in.dMdz/fmax(entr_in.M,1e-12),0.0)
    detr_MdMdz = fmax(-entr_in.dMdz/fmax(entr_in.M,1e-12),0.0)


    #smooth min
    with gil:
        l[0] = entr_in.tke_coef*fabs(db/sqrt(entr_in.tke+1e-8))
        l[1] = fabs(db/dw)
        inv_timescale = lamb_smooth_minimum(l, 0.1, 0.0005)

    _ret.entr_sc = inv_timescale/dw*(entr_in.c_ent*logistic_e + c_det*moisture_deficit_e) + entr_MdMdz * entr_in.c_div
    _ret.detr_sc = inv_timescale/dw*(entr_in.c_ent*logistic_d + c_det*moisture_deficit_d) + detr_MdMdz * entr_in.c_div


    return _ret


cdef entr_struct entr_detr_buoyancy_sorting(entr_in_struct entr_in) nogil:

    cdef:
        entr_struct _ret
        double eps_bw2, del_bw2, D_, sorting_function, eta, pressure,a1 ,a2 ,c ,d

    ret_b = buoyancy_sorting_mean(entr_in)
    b_mix = ret_b.b_mix
    eps_bw2 = entr_in.c_ent*fmax(entr_in.b_upd,0.0) / fmax(entr_in.w_upd * entr_in.w_upd, 1e-2)
    del_bw2 = entr_in.c_ent*fabs(entr_in.b_upd) / fmax(entr_in.w_upd * entr_in.w_upd, 1e-2)
    _ret.b_mix = b_mix
    _ret.sorting_function = ret_b.sorting_function
    _ret.entr_sc = eps_bw2
    if entr_in.ql_up>0.0:
        D_ = 0.5*(1.0+entr_in.sort_pow*(ret_b.sorting_function))
        _ret.detr_sc = del_bw2*(1.0+entr_in.c_det*D_)
    else:
        _ret.detr_sc = 0.0

    return _ret

cdef buoyant_stract buoyancy_sorting_mean(entr_in_struct entr_in) nogil:

        cdef:
            double qv_ ,T_env ,ql_env ,rho_env ,b_env, T_up ,ql_up ,rho_up ,b_up, b_mean, b_mix, qt_mix , H_mix
            double sorting_function = 0.0
            eos_struct sa
            buoyant_stract ret_b

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_env, entr_in.H_env)
        qv_ = entr_in.qt_env - sa.ql
        T_env = sa.T
        ql_env = sa.ql
        rho_env = rho_c(entr_in.p0, sa.T, entr_in.qt_env, qv_)
        b_env = buoyancy_c(entr_in.rho0, rho_env)

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_up, entr_in.H_up)
        qv_ = entr_in.qt_up - sa.ql
        T_up = sa.T
        ql_up = sa.ql
        rho_up = rho_c(entr_in.p0, sa.T, entr_in.qt_up, qv_)
        b_up = buoyancy_c(entr_in.rho0, rho_up)

        b_mean = entr_in.a_upd*b_up +  (1.0-entr_in.a_upd)*b_env

        # qt_mix = (0.25*entr_in.qt_up + 0.75*entr_in.qt_env)
        # H_mix =  (0.25*entr_in.H_up  + 0.75*entr_in.H_env)
        qt_mix = (0.5*entr_in.qt_up + 0.5*entr_in.qt_env)
        H_mix =  (0.5*entr_in.H_up  + 0.5*entr_in.H_env)
        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_mix, H_mix)
        qv_ = (entr_in.qt_up+entr_in.qt_env)/2.0 - sa.ql
        rho_mix = rho_c(entr_in.p0, sa.T, qt_mix, qv_)
        b_mix = buoyancy_c(entr_in.rho0, rho_mix)-b_mean
        sorting_function = -(b_mix)/fmax(fabs(b_up-b_env),0.0000001)
        ret_b.b_mix = b_mix
        ret_b.sorting_function = sorting_function

        return ret_b

cdef double buoyancy_sorting(entr_in_struct entr_in) nogil:

        cdef:
            Py_ssize_t m_q, m_h
            int i_b

            double h_hat, qt_hat, sd_h, sd_q, corr, mu_h_star, sigma_h_star, qt_var, T_hat
            double sqpi_inv = 1.0/sqrt(pi)
            double sqrt2 = sqrt(2.0)
            double sd_q_lim, bmix, qv_
            double L_, dT, Tmix
            double sorting_function = 0.0
            double inner_sorting_function = 0.0
            eos_struct sa
            double [:] weights
            double [:] abscissas
        with gil:
            abscissas, weights = np.polynomial.hermite.hermgauss(entr_in.quadrature_order)

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_env, entr_in.H_env)
        qv_ = entr_in.qt_env - sa.ql
        T_env = sa.T
        ql_env = sa.ql
        rho_env = rho_c(entr_in.p0, sa.T, entr_in.qt_env, qv_)
        b_env = buoyancy_c(entr_in.rho0, rho_env)

        sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, entr_in.qt_up, entr_in.H_up)
        qv_ = entr_in.qt_up - sa.ql
        T_up = sa.T
        ql_up = sa.ql
        rho_up = rho_c(entr_in.p0, sa.T, entr_in.qt_up, qv_)
        b_up = buoyancy_c(entr_in.rho0, rho_up)

        b_mean = entr_in.a_upd*b_up +  (1.0-entr_in.a_upd)*b_env

        if entr_in.env_QTvar != 0.0 and entr_in.env_Hvar != 0.0:
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
                inner_sorting_function = 0.0
                for m_h in xrange(entr_in.quadrature_order):
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star + entr_in.H_up)/2.0
                    # condensation - evaporation
                    sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
                    # calcualte buoyancy
                    qv_ = qt_hat - sa.ql
                    L_ = latent_heat(sa.T)
                    dT = L_*((entr_in.ql_up+entr_in.ql_env)/2.0- sa.ql)/1004.0
                    rho_mix = rho_c(entr_in.p0, sa.T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.rho0, rho_mix) - b_mean #- entr_in.dw2dz

                    if bmix >0.0:
                        inner_sorting_function  += weights[m_h] * sqpi_inv

                sorting_function  += inner_sorting_function * weights[m_q] * sqpi_inv
        else:
            h_hat = ( entr_in.H_env + entr_in.H_up)/2.0
            qt_hat = ( entr_in.qt_env + entr_in.qt_up)/2.0

            # condensation
            sa  = eos(t_to_thetali_c, eos_first_guess_thetal, entr_in.p0, qt_hat, h_hat)
            # calcualte buoyancy
            rho_mix = rho_c(entr_in.p0, sa.T, qt_hat, qt_hat - sa.ql)
            bmix = buoyancy_c(entr_in.rho0, rho_mix) - entr_in.b_mean
            if bmix  - entr_in.dw2dz >0.0:
                sorting_function  = 1.0
            else:
                sorting_function  = 0.0

        return sorting_function

cdef entr_struct entr_detr_tke(entr_in_struct entr_in) nogil:
    cdef entr_struct _ret
    _ret.detr_sc = fabs(entr_in.b_upd)/ fmax(entr_in.w_upd * entr_in.w_upd, 1e-3)
    _ret.entr_sc = sqrt(entr_in.tke) / fmax(entr_in.w_upd, 0.01) / fmax(sqrt(entr_in.a_upd), 0.001) / 50000.0
    return  _ret


cdef entr_struct entr_detr_b_w2(entr_in_struct entr_in) nogil:
    cdef :
        entr_struct _ret
        double effective_buoyancy
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 4.0e-3 + 0.12 *fabs(fmin(entr_in.b_upd,0.0)) / fmax(entr_in.w_upd * entr_in.w_upd, 1e-2)
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = 0.12 * fmax(entr_in.b_upd,0.0) / fmax(entr_in.w_upd * entr_in.w_upd, 1e-2)

    return  _ret

cdef entr_struct entr_detr_suselj(entr_in_struct entr_in) nogil:
    cdef:
        entr_struct _ret
        double entr_dry = 2.5e-3
        double l0

    l0 = (entr_in.zbl - entr_in.zi)/10.0
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 4.0e-3 +  0.12* fabs(fmin(entr_in.b_upd,0.0)) / fmax(entr_in.w_upd * entr_in.w_upd, 1e-2)
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

cdef pressure_buoy_struct pressure_tan18_buoy(pressure_in_struct press_in) nogil:
    cdef:
        pressure_buoy_struct _ret

    _ret.b_coeff = press_in.bcoeff_tan18
    _ret.nh_pressure_b = -1.0 * press_in.rho0_kfull * press_in.a_kfull * press_in.b_kfull * _ret.b_coeff

    return _ret

cdef pressure_drag_struct pressure_tan18_drag(pressure_in_struct press_in) nogil:
    cdef:
        pressure_drag_struct _ret

    _ret.nh_pressure_adv = 0.0
    _ret.nh_pressure_drag = -1.0 * press_in.rho0_kfull * sqrt(press_in.a_kfull)* sqrt(press_in.a_kfull) * (1.0/press_in.rd
                          * (press_in.w_kfull - press_in.w_kenv)*fabs(press_in.w_kfull - press_in.w_kenv))

    return _ret

cdef pressure_buoy_struct pressure_normalmode_buoy(pressure_in_struct press_in) nogil:
    cdef:
        pressure_buoy_struct _ret

    _ret.b_coeff = press_in.alpha1 / ( 1+press_in.alpha2*press_in.asp_ratio**2 )
    _ret.nh_pressure_b = -1.0 * press_in.rho0_kfull * press_in.a_kfull * press_in.b_kfull * _ret.b_coeff

    return _ret

cdef pressure_drag_struct pressure_normalmode_drag(pressure_in_struct press_in) nogil:
    cdef:
        pressure_drag_struct _ret

    _ret.nh_pressure_adv = press_in.rho0_kfull * press_in.a_kfull * press_in.beta1*press_in.w_kfull*(press_in.w_kfull
                          -press_in.w_kmfull)*press_in.dzi

    # drag as w_dif and account for downdrafts
    _ret.nh_pressure_drag = -1.0 * press_in.rho0_kfull * press_in.a_kfull * press_in.beta2 * (press_in.w_kfull -
                            press_in.w_kenv)*fabs(press_in.w_kfull - press_in.w_kenv)/fmax(press_in.updraft_top, 500.0)

    return _ret

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
    if ((3.75 + cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar)==0.0:
        with gil:
            print(zLL,oblength ,ustar)
    if oblength < 0.0:
        return ((3.75 + cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar)
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
