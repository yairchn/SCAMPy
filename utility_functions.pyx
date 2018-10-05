import numpy as np
import scipy.special as sp
from libc.math cimport exp, fmax, log, fabs
from scipy.stats import norm
cimport cython

# compute the mean of the values above a given percentile (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficient for a single updraft or nth updraft of n updrafts
cpdef double percentile_mean_norm(double percentile, Py_ssize_t nsamples):
    cdef:
        double [:] x = norm.rvs(size=nsamples)
        double xp = norm.ppf(percentile)
    return np.ma.mean(np.ma.masked_less(x,xp))

# compute the mean of the values between two percentiles (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficients for 1 to n-1 updrafts when using n updrafts
cpdef double percentile_bounds_mean_norm(double low_percentile, double high_percentile, Py_ssize_t nsamples):
    cdef:
        double [:] x = norm.rvs(size=nsamples)
        double xp_low = norm.ppf(low_percentile)
        double xp_high = norm.ppf(high_percentile)
    return np.ma.mean(np.ma.masked_greater(np.ma.masked_less(x,xp_low),xp_high))


cdef double interp2pt(double val1, double val2) nogil:
    return 0.5*(val1 + val2)

cdef double logistic(double x, double slope, double mid) nogil:
    return 1.0/(1.0 + exp( -slope * (x-mid)))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double smooth_minimum(double [:] x, double a) nogil:
    cdef:
      unsigned int i = 0
      double num, den, leng
    with gil:
        leng = len(x)
    num = 0; den = 0
    while(i<leng):
      if (x[i]>1.0e-5):
        num += x[i]*exp(-a*(x[i]))
        den += exp(-a*(x[i]))
      i += 1
    smin = num/den
    return smin


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double smooth_maximum(double [:] x, double a) nogil:
    cdef:
      unsigned int i = 0
      double num, den, leng
    with gil:
        leng = len(x)
    num = 0; den = 0
    while(i<leng):
      if (x[i]>1.0e-5):
        num += x[i]*exp(a*(x[i]))
        den += exp(a*(x[i]))
      i += 1
    smin = num/den
    return smin

cdef double roe_velocity(double fp, double fm, double varp, double varm) nogil:
    cdef:
        double roe_vel
    if fabs(varp-varm)>0.0:
        roe_vel = (fp-fm)/(varp-varm)
    else:
        roe_vel = 0.0
    return roe_vel
