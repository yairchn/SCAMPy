#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True


cdef class Grid:
    cdef:
        Py_ssize_t gw
        Py_ssize_t nz
        Py_ssize_t nzg
        double beta
        double [:] z
        double [:] z_half
        double [:] dz
        double [:] dz_half
        double [:] dzi
        double [:] dzi_half
