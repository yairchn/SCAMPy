import numpy as np
import scipy.special as sp
from libc.math cimport exp, log
from scipy.stats import norm
cimport cython
cimport numpy as np

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
      double num, den
      double leng

    num = 0; den = 0
    leng = x.shape[0]
    while(i<leng):
      if (x[i]>1.0e-5):
        num += x[i]*exp(-a*(x[i]))
        den += exp(-a*(x[i]))
      i += 1
    smin = num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double auto_smooth_minimum( const double [:] x, double f):
    cdef:
      unsigned int i = 0
      double num, den
      double leng, lmin, lmin2
      double scale
      double a = 1.0
      np.ndarray[double, ndim=1] x_ = np.empty(len(x))

    lmin = 1.0e5; lmin2 = 1.0e5
    leng = x.shape[0]
    while (i<leng):
      x_[i] = x[i]
      i += 1
    # Get min and second min values
    i = 0
    lmin = min(x_)
    with nogil:
      while(i<leng):
        if (x_[i]<lmin2 and x_[i]>lmin+1.0e-5):
          lmin2 = x_[i]
        x_[i] -= lmin
        i += 1

      # Set relative maximum importance of second min term
      scale = (lmin2-lmin)/lmin*(1.0/(1.0+exp(lmin2-lmin)))
      if (scale>f):
        a = log((lmin2-lmin)/lmin/f-1.0)/(lmin2-lmin)

      i = 0
      num = 0.0; den = 0.0;
      while(i<leng):
        num += x_[i]*exp(-a*(x_[i]))
        den += exp(-a*(x_[i]))
        i += 1
      smin = lmin + num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double smooth_minimum2(double [:] x, double l0) nogil:
    cdef:
      unsigned int i = 0, numLengths = 0
      double smin = 0.0
      double leng

    leng = x.shape[0]
    while(i<leng):
      if (x[i]>1.0e-5):
        smin += exp(-x[i]/l0)
        numLengths += 1
      i += 1
    smin /=  float(numLengths)
    smin =- l0*log(smin)
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double softmin(double [:] x, double k):
    cdef:
      unsigned int i = 1, j = 1
      double smin = 0.0, lmin, num, den, eps = 0.1, lam = 1.0
      double leng

    leng = x.shape[0]
    lmin = min(x)
    num = 1.0
    den = 1.0
    while(j<leng):
      if (x[j]-lmin>eps*lmin):
        lam = log( ( (1.0+lmin/(k*x[j]))**(1.0/(len(x)-1.0)) - 1.0 )**(-1) )
        lam /= ((x[j]-lmin)/lmin)
        break;
      j += 1
    while(i<leng):
      x[i] /= lmin
      num += x[i]*exp(-lam*(x[i]-1.0))
      den += exp(-lam*(x[i]-1.0))
      i += 1
    smin = lmin*num/den
    return smin

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double hardmin(double [:] x):
    cdef:
      unsigned i = 0
      double lmin = 1.0e6
      double leng

    leng = x.shape[0]
    while(i<leng):
      if (x[i]>1.0e-5 and x[i]<lmin):
        lmin = x[i]
      i += 1

    return min(x)
