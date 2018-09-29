cdef double interp2pt(double val1, double val2) nogil
cdef double logistic(double x, double slope, double mid) nogil
cdef double smooth_minimum(double x1, double x2 , double a) nogil
cdef double smooth_maximum(double x1, double x2 , double a) nogil # double x3 ,
cpdef double percentile_mean_norm(double percentile, Py_ssize_t nsamples)
cpdef double percentile_bounds_mean_norm(double low_percentile, double high_percentile, Py_ssize_t nsamples)
cdef double smooth_minimum2(double [:] x, double a) nogil