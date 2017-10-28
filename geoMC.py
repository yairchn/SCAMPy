#!/usr/bin/env python
"""
Geometric MCMC samplers
Shiwei Lan @ CalTech, 2017
-------------------------------
After the class is instantiated with arguments, call sample to collect MCMC samples which will be stored in 'result' folder.
-----------------------------------
Created October 10, 2017
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@gmail.com"

import numpy as np
import timeit, time


class geoMC(object):
    def __init__(self, parameter_init, geometry_fun, alg_name, step_size, step_num=1, low_bd=[-np.inf], upp_bd=[np.inf],
                 bdy_hdl='reject'):
        # parameters
        self.q = np.array(parameter_init)
        print(self.q)
        try:
            self.dim = len(self.q)
        except:
            self.dim = len([self.q])

        self.geom = geometry_fun  # returns geometric quantities of objective function
        self.geom_opt = (0)  # getting function value
        self.u = self.geom(self.q, self.geom_opt)

        # sampling setting
        self.h = step_size
        self.L = step_num
        if 'HMC' not in alg_name: self.L = 1
        self.alg_name = alg_name

        # domain of parameter space
        if len(low_bd) == 0: low_bd = [-np.inf]
        if len(upp_bd) == 0: upp_bd = [np.inf]
        if len(low_bd) == 1: low_bd = low_bd * self.dim
        if len(upp_bd) == 1: upp_bd = upp_bd * self.dim
        self.lb = low_bd
        self.ub = upp_bd
        self.bdy_hdl = bdy_hdl

    # handle boundary constraint
    def hdl_const(self, q, v=None):
        acpt_ind = True
        violt_ind = np.vstack([q < self.lb, q > self.ub])  # indicator of violating boundary constraint
        if 'reject' in self.bdy_hdl:
            if violt_ind.any():  # reject if any
                acpt_ind = False
                return q, v, acpt_ind
            else:
                return q, v, acpt_ind
        elif 'bounce' in self.bdy_hdl:
            while 1:  # bounce off the boundary until all components satisfy the constraint
                if violt_ind[0, :].any():
                    idx_l = violt_ind[0, :]
                    q[idx_l] = 2 * self.lb[idx_l] - q[idx_l]
                    v[idx_l] = -v[idx_l]
                elif violt_ind[1, :].any():
                    idx_u = violt_ind[1, :]
                    q[idx_u] = 2 * self.ub[idx_u] - q[idx_u]
                    v[idx_u] = -v[idx_u]
                else:
                    break
                violt_ind = np.vstack([q < self.lb, q > self.ub])
            return q, v, acpt_ind
        else:
            error('Option for handling boundary constraint not available!')

    # random walk Metropolis
    def RWM(self):
        # sample velocity
        v = np.random.randn(self.dim)

        # generate proposal according to random walk
        q = self.q + self.h * v

        # handle boundary constraint
        q, v, acpt = self.hdl_const(q, v)

        if acpt:
            # update geometry
            # q is the parameter , u is the new costfun
            u = self.geom(q, self.geom_opt)

            # Metropolis test
            logr = -u + self.u
            # detemine if the last move is better
            if np.isfinite(logr) and np.log(np.random.uniform()) < min(0, logr):
                # accept
                self.q = q;
                self.u = u;
                acpt = True
            else:
                acpt = False

        # return accept indicator
        return acpt


    # sample with given method
    def sample(self, num_samp, num_burnin):
        name_sampler = str(self.alg_name)
        try:
            sampler = getattr(self, name_sampler)
        except AtributeError:
            print(self.alg_name, 'not found!')
        else:
            print('Running ' + self.alg_name + ' now...\n')

        # allocate space to store results
        self.samp = np.zeros((num_samp, self.dim))
        self.engy = np.zeros(num_samp + num_burnin)
        accp = 0.0  # online acceptance
        self.acpt = 0.0  # final acceptance rate

        for s in range(num_samp + num_burnin):

            if s == num_burnin:
                # start the timer
                tic = timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # generate MCMC sample with given sampler
            try:
                acpt_idx = sampler()
                num_cons_bad = 0  # number of consecutive bad proposals
            except RuntimeError:
                # acpt_idx=False
                # print('Bad proposal encountered!')
                # pass # reject bad proposal: bias introduced
                num_cons_bad += 1
                if num_cons_bad < 10:
                    s -= 1
                    print('Bad proposal encountered! Retrying...')
                    continue  # retry until a valid proposal is made
                else:
                    print('10 consecutive bad proposals encountered! Passing...')
                    pass  # reject it and keep going

            accp += acpt_idx

            # display acceptance at intervals
            if (s + 1) % 100 == 0:
                print('Acceptance at %d iterations: %0.2f' % (s + 1, accp / 100))
                accp = 0.0

            # save results
            self.engy[s] = -self.u
            if s >= num_burnin:
                self.samp[s - num_burnin,] = self.q.T
                self.acpt += acpt_idx

        # stop timer
        toc = timeit.default_timer()
        self.time = toc - tic
        self.acpt /= num_samp
        print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
              % (self.time, num_samp, self.acpt))

        # save to file
        self.save()

    # save samples
    def save(self):
        import os, errno
        import pickle
        # create folder
        cwd = os.getcwd()
        self.savepath = os.path.join(cwd, 'result')
        try:
            os.makedirs(self.savepath)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
        # name file
        ctime = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.filename = self.alg_name + '_dim' + str(self.dim) + '_' + ctime
        # dump data
        f = open(os.path.join(self.savepath, self.filename + '.pckl'), 'wb')
        pickle.dump([self.alg_name, self.h, self.L, self.samp, self.engy, self.acpt, self.time], f)
        f.close()
