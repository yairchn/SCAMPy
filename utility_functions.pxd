cdef double interp2pt(double val1, double val2) nogil
cdef double logistic(double x, double slope, double mid) nogil
cpdef double percentile_mean_norm(double percentile, Py_ssize_t nsamples)
cpdef double percentile_bounds_mean_norm(double low_percentile, double high_percentile, Py_ssize_t nsamples)
cdef double smooth_minimum(double [:] x, double a) nogil
cdef double auto_smooth_minimum(const double [:] x, double f)
cdef double smooth_minimum2(double [:] x, double l0) nogil
cdef double softmin(double [:] x, double k)
cdef double hardmin(double [:] x)
