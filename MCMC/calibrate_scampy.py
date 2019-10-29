import numpy as np
import pandas as pd
import os

import sys
localpath = os.getcwd()
sys.path.append('/Users/yaircohen/Documents/codes/scampy/MCMC/')
sys.path.append(localpath)
# this is the path for the tools needed _ yair

from ces.calibrate import *
import run_scampy

model = run_scampy.forward_scampy('Bomex')
# Yair this is the true data
model.ustar = np.array([1.0]) # for testing
y_obs = model(model.ustar) # ustar is the true parameter
model.n_obs = 3
#  Problem setup
gamma = 0.005 # covariance matrix
Gamma = gamma**2 * np.identity(model.n_obs) # Gamma is the noise
model.p = np.size(model.ustar) # dimension of theta and of ustar

# y_obs = y_obs + 1.0 * gamma * np.random.normal(0,1,model.n_obs)
# I can replace y_obs by a true data vector. need to write a code for generateing true data vector that is like G

tqdm.write('Remember, the number of params is: %s'%(np.size(model.ustar)))
Jnoise = np.linalg.cholesky(Gamma)

def run_neki(J):
    p = model.p
    enki          = flow(p = p, n_obs = model.n_obs, J = J) # initialize the ensamble object
    enki.ustar    = model.ustar#.reshape(p,-1) # pass the true parameters. True data is generate by running the model with true parmaeters
    enki.T        = 200 # number of itr
    enki.mu       = 0.0 * np.ones((p,)) #.reshape(p,-1) # prior mean
    enki.sigma    = 100. * np.identity(p) # prior covar
    enki.parallel = False
    enki.mute_bar = True # display progress bar
    enki.nexp     = ''
    model.l_window = ''

    np.random.seed(1)
    U0 = 10 * np.random.normal(0, 1, [enki.p, J])
    enki.run(y_obs, U0, model, Gamma, Jnoise, save_online = True)
    return enki


np.random.seed(1)
#Js = 2**np.arange(3,11)
# Js = 2**np.arange(3, 12, 2)
# Js = [8, 32, 128, 512]
Js = [8] # Js i sthe number of ensamble memebrs

enkis = []
for J in tqdm(Js, desc = 'Ensembles: ', position = 0):
    enkis.append(run_neki(J))
