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
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@gmail.com"

import numpy as np
import timeit, time


class geoMC(object):
    def __init__(self, parameter_init, geometry_fun, alg_name, step_size=1.0, step_num=1, low_bd=[-np.inf],
                 upp_bd=[np.inf], bdy_hdl='reject', adpt=True):

        # parameters
        self.q = np.array(parameter_init)
        try:
            self.dim = len(self.q)
        except:
            self.dim = len([self.q])


        # geometry needed
        self.geom = geometry_fun  # returns geometric quantities of objective function

        self.geom_opt = [0]  # getting function value
        if any(s in alg_name for s in ['MALA', 'HMC']): self.geom_opt.append(1)  # getting gradient
        if any(s in alg_name for s in ['mMALA', 'mHMC']): self.geom_opt.append(2)  # getting 2nd order derivatives
        self.u = self.geom(self.q,self.geom_opt)  # function value (u), gradient (du), Fisher information (FI) and Cholesky factor of metric (cholG)
        # I deltee these , self.du, self.FI, self.cholG
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

        # adaptation of step size
        self.adpt = adpt
        h_adpt = {}
        if self.adpt:
            h_adpt['h'] = self.init_h()
            #             h_adpt['h']=self.h;
            h_adpt['mu'] = np.log(10 * h_adpt['h'])
            h_adpt['loghn'] = 0
            h_adpt['An'] = 0
            # constants' setting
            h_adpt['gamma'] = 0.05
            h_adpt['n0'] = 10
            h_adpt['kappa'] = 0.75
            h_adpt['a0'] = 0.65
        self.h_adpt = h_adpt

    # resample v
    def resample_aux(self):
        v = np.random.randn(self.dim)

        return v

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

    # one step integration to make a proposal
    def onestep(self, q, v, h=1): # I deleted du=0,
        #if any(s > 0 for s in self.geom_opt):
            # update velocity for half step
            # v = v - h / 2 * du

        # generate proposal according to Langevin dynamics for a whole step
        q = q + h * v

        # handle boundary constraint
        q, v, acpt = self.hdl_const(q, v)

        # update geometry
        u = self.geom(q, self.geom_opt) # I deleteted these , du, _, _

        #if any(s > 0 for s in self.geom_opt):
            # update velocity for another half step
         #   v = v - h / 2 * du

        return q, v, u, acpt # I deleted these  du,

    # (adaptive) random walk Metropolis
    def RWM(self, s=1, Nadpt=0):
        # initialization
        q = self.q.copy()

        # sample velocity
        v = self.resample_aux()

        # current energy
        E_cur = self.u

        # one step move to make proposal
        q, v, u, acpt = self.onestep(q, v,  self.h) # ,0  and ,_

        if acpt:
            # new energy
            E_prp = u

            # Metropolis test
            logr = -E_prp + E_cur

            if np.isfinite(logr) and np.log(np.random.uniform()) < min(0, logr):
                # accept
                self.q = q
                self.u = u
                acpt = True
            else:
                acpt = False

            # adapt step size h if needed
            if self.adpt:
                if s <= Nadpt:
                    self.h_adpt = self.dual_avg(s, np.exp(np.min([0, logr])))
                    print('New step size: %0.2f' % self.h_adpt['h'])
                    print('\tNew averaged step size: %0.6f\n' % np.exp(self.h_adpt['loghn']))
                # stop adaptation and freeze the step size
                if s == Nadpt:
                    self.h_adpt['h'] = np.exp(self.h_adpt['loghn'])
                    print('Adaptation completed; step size freezed at: %0.6f\n' % self.h_adpt['h'])
                self.h = self.h_adpt['h']

        # return accept indicator
        return acpt

    # (adaptive) Metropolis Adjusted Langevin Algorithm
    # def MALA(self, s=1, Nadpt=0):
    #     # initialization
    #     q = self.q.copy();
    #     du = self.du.copy()
    #
    #     # sample velocity
    #     v = self.resample_aux()
    #
    #     # current energy
    #     E_cur = self.u + v.dot(v) / 2
    #
    #     # one step move to make proposal
    #     #q, v, u, du, acpt = self.onestep(q, v, du, self.h)
    #
    #     if acpt:
    #         # new energy
    #         E_prp = u + v.dot(v) / 2
    #
    #         # Metropolis test
    #         logr = -E_prp + E_cur
    #
    #         if np.isfinite(logr) and np.log(np.random.uniform()) < min(0, logr):
    #             # accept
    #             self.q = q;
    #             self.u = u;
    #             self.du = du;
    #             acpt = True
    #         else:
    #             acpt = False
    #
    #         # adapt step size h if needed
    #         if self.adpt:
    #             if s <= Nadpt:
    #                 self.h_adpt = self.dual_avg(s, np.exp(np.min([0, logr])))
    #                 print('New step size: %0.2f' % self.h_adpt['h'])
    #                 print('\tNew averaged step size: %0.6f\n' % np.exp(self.h_adpt['loghn']))
    #             # stop adaptation and freeze the step size
    #             if s == Nadpt:
    #                 self.h_adpt['h'] = np.exp(self.h_adpt['loghn'])
    #                 print('Adaptation completed; step size freezed at: %0.6f\n' % self.h_adpt['h'])
    #             self.h = self.h_adpt['h']
    #
    #     # return accept indicator
    #     return acpt

    # (adaptive) Hamiltonian Monte Carlo
    # def HMC(self, s=1, Nadpt=0):
    #     # initialization
    #     q = self.q.copy();
    #     du = self.du.copy()
    #     rth = np.sqrt(self.h)  # make the scale comparable to MALA
    #
    #     # sample velocity
    #     v = self.resample_aux()
    #
    #     # current energy
    #     E_cur = self.u + v.dot(v) / 2
    #
    #     randL = np.int(np.ceil(np.random.uniform(0, self.L)))
    #
    #     for l in range(randL):
    #         # one step move to make proposal
    #         q, v, u, du, acpt = self.onestep(q, v, du, self.h)
    #         if not acpt:
    #             break
    #
    #     if acpt:
    #         # new energy
    #         E_prp = u + v.dot(v) / 2
    #
    #         # Metropolis test
    #         logr = -E_prp + E_cur
    #
    #         if np.isfinite(logr) and np.log(np.random.uniform()) < min(0, logr):
    #             # accept
    #             self.q = q;
    #             self.u = u;
    #             self.du = du;
    #             acpt = True
    #         else:
    #             acpt = False
    #
    #         # adapt step size h if needed
    #         if self.adpt:
    #             if s <= Nadpt:
    #                 self.h_adpt = self.dual_avg(s, np.exp(np.min([0, logr])))
    #                 print('New step size: %0.2f' % self.h_adpt['h'])
    #                 print('\tNew averaged step size: %0.6f\n' % np.exp(self.h_adpt['loghn']))
    #             # stop adaptation and freeze the step size
    #             if s == Nadpt:
    #                 self.h_adpt['h'] = np.exp(self.h_adpt['loghn'])
    #                 print('Adaptation completed; step size freezed at: %0.6f\n' % self.h_adpt['h'])
    #             self.h = self.h_adpt['h']
    #
    #     # return accept indicator
    #     return acpt

    # find reasonable initial step size
    def init_h(self):
        v0 = self.resample_aux()
        E_cur = self.u + (v0.dot(v0)) / 2
        h = 1.0
        q, v, u,  _ = self.onestep(self.q, v0, h) # self.du, du,
        E_prp = u + (v.dot(v)) / 2
        logr = -E_prp + E_cur
        a = 2 * (np.exp(logr) > 0.5) - 1.0
        while a * logr > -a * np.log(2):
            h = h * pow(2, a)
            q, v, u,  _ = self.onestep(self.q, v0,  h) # self.du, du,
            E_prp = u + (v.dot(v)) / 2
            logr = -E_prp + E_cur

        return h

    # dual-averaging to adapt step size
    def dual_avg(self, s, an):
        hn_adpt = self.h_adpt
        hn_adpt['An'] = (1 - 1.0 / (s + hn_adpt['n0'])) * hn_adpt['An'] + (hn_adpt['a0'] - an) / (s + hn_adpt['n0'])
        logh = hn_adpt['mu'] - np.sqrt(s) / hn_adpt['gamma'] * hn_adpt['An']
        hn_adpt['loghn'] = pow(s, -hn_adpt['kappa']) * logh + (1 - pow(s, -hn_adpt['kappa'])) * hn_adpt['loghn']
        hn_adpt['h'] = np.exp(logh)

        return hn_adpt

    # sample with given method
    def sample(self, num_samp1, num_burnin1):
        num_burnin = int(num_burnin1)
        num_samp = int(num_samp1)
        name_sampler = str(self.alg_name)
        try:
            sampler = getattr(self, name_sampler)
        except AtributeError:
            print(self.alg_name, 'not found!')
        else:
            print('Running ' + self.alg_name + ' now...\n')
        # allocate space to store results
        print('yair-yair')
        print(type(num_samp))
        print(type(self.dim))
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
                if self.adpt:
                    acpt_idx = sampler(s + 1, num_burnin)
                else:
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
