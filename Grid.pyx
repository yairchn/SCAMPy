#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

#Adapated from PyCLES: https://github.com/pressel/pycles

cimport numpy as np
import numpy as np
import time
cdef class Grid:
    '''
    A class for storing information about the LES grid.
    '''
    def __init__(self,namelist):
        '''

        :param namelist: Namelist dictionary
        :param Parallel: ParallelMPI class
        :return:
        '''

        #Get the grid spacing
        dz = namelist['grid']['dz']

        #Set the inverse grid spacing

        # self.dzi = 1.0/self.dz

        #Get the grid dimensions and ghost points
        self.beta = namelist['grid']['stretch_power']
        self.gw = namelist['grid']['gw']
        self.nz = namelist['grid']['nz']
        self.nzg = self.nz + 2 * self.gw

        self.z_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.z = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')

        self.dz_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.dzi_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.dz = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.dzi = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')

        cdef int i, count = 0
        if namelist['grid']['stretched'] == True:
            for i in xrange(self.gw-1,self.nz+2*self.gw):
                self.z[i] = dz*(i-1)**self.beta
                if i>self.gw-1:
                    self.z_half[i] = dz*((i-1)**self.beta + (i-2)**self.beta)/2.0

            for k in xrange(1,self.gw):
                self.z[self.gw - 1 - k] = -self.z[self.gw -1 + k]
            for k in xrange(self.gw):
                self.z_half[self.gw - 1 - k] = -self.z_half[self.gw + k]

            self.dz[0] = self.z[1]-self.z[0]
            self.dzi[0] = 1.0/self.z[i]
            self.dz_half[0] = self.z_half[1]-self.z_half[0]
            self.dzi_half[0] = 1.0/self.z_half[0]
            for i in xrange(1,self.nz+2*self.gw):
                self.dz[i] = self.z[i]-self.z[i-1]
                self.dzi[i] = 1.0/self.z[i]
                self.dz_half[i] = self.z_half[i]-self.z_half[i-1]
                self.dzi_half[i] = 1.0/self.z_half[i]


        else:
            for i in xrange(-self.gw,self.nz+self.gw,1):
                self.z[count] = (i + 1) * dz
                self.z_half[count] = (i+0.5)* dz
                if count==0:
                    self.dz[count] = dz
                    self.dzi[count] = 1.0/dz
                    self.dz_half[count] = dz
                    self.dzi_half[count] = 1.0/dz
                else:
                    self.dz[count] = self.z[count]-self.z[count-1]
                    self.dzi[count] = 1.0/self.dz[count]
                    self.dz_half[count] = self.z_half[count]-self.z_half[count-1]
                    self.dzi_half[count] = 1.0/self.dz_half[count]
                count += 1



        return




