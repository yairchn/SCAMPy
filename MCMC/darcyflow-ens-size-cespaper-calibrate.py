
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.stats import tmean, tstd
from sklearn.utils import resample

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import os
from scipy.spatial import distance_matrix

plt.style.use('seaborn-white')
sns.set_style("ticks")
sns.set_context("talk")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sys
localpath = os.getcwd()
sys.path.append('/Users/agarbuno/postdoc/python/modules/ipuq/')
sys.path.append(localpath)
# this is the path for the tools needed _ yair 

from ces.calibrate import *
from run_scampy import *


# <a id='contents'></a> 
# # Table of contents
# 
# 0. **[Setup](#init)**
# 1. **[Calibrate](#calibrate)**  
# 2. **[Emulate](#emulate)**  
#     **2.1. [Linear prior](#linear-emulate-all)**  
#     **2.2. [Mixed prior](#mixed-emulate-all)**  
#     **2.3. [Time variability](#time-var)**  
#     **2.4. [Parameter variability](#par-var)**
# 3. **[Sample](#sample)**

# <a id='init'></a> 
# #  Problem setup 

# In[4]:

model = run_scampy.foreard_scampy()
# model = darcy.model()

xs, ys = np.meshgrid(np.linspace(0, 1, int(model.Nmesh)), np.linspace(0, 1, int(model.Nmesh)))
np.random.seed(1)
grid_pts    = np.vstack((xs.flatten(), ys.flatten()))           # Flat the (x,y) coordinates
model.obs_index = np.random.choice(int(model.p), model.n_obs, replace=False, 
                             p = U /U.sum())                    # Sample without replacement 
model.obs_locs = grid_pts[:, model.obs_index]                                     # Recover the random (x,y) coordinates

# Yair this is the true data
y_obs = model(model.ustar)

#  Problem setup
gamma = 0.005 # covariance matrix
Gamma = gamma**2 * np.identity(model.n_obs)
y_obs = y_obs + 1.0 * gamma * np.random.normal(0,1,model.n_obs)


# In[14]:


tqdm.write('Remember, the number of params is: %s'%(len(model.ustar)))


# In[15]:


y_obs


# <a id='calibrate'></a> 
# # 1. Calibrate
# [go back](#contents)

# In[15]:


Jnoise = np.linalg.cholesky(Gamma)

def run_neki(J): 
    p = model.p

    enki          = flow(p = p, n_obs = model.n_obs, J = J)
    enki.ustar    = model.ustar.reshape(p,-1)
    enki.T        = 200
    enki.mu       = 0.0 * np.ones((p,)).reshape(p,-1)
    enki.sigma    = 100. * np.identity(p)
    enki.parallel = False
    enki.mute_bar = True
    enki.nexp     = ''
    model.l_window = ''
    
    np.random.seed(1)
    U0 = 10 * np.random.normal(0, 1, [enki.p, J])
    enki.run(y_obs, U0, model, Gamma, Jnoise, save_online = True)
    return enki


# In[17]:


np.random.seed(1)
#Js = 2**np.arange(3,11)
Js = 2**np.arange(3, 12, 2)
Js = [8, 32, 128, 512]

enkis = []
for J in tqdm(Js, desc = 'Ensembles: ', position = 0):
    enkis.append(run_neki(J))


# In[20]:


k = np.concatenate((np.arange(6), 100 * np.arange(6, int(model.Nmesh))))
K1, K2 = np.meshgrid(k, k)

eigs = (model.tau**(model.alpha-1))*(np.pi**2 * (K1**2 + K2**2) + model.tau**2)**(-model.alpha/2)
eigs[0,0] = 1e-10
rank = (-eigs).flatten().argsort()


# In[21]:


sns.set_context("talk")
sns.set_style("ticks")

colors = [u'#1a601a',u'#238023',u'#2ca02c',u'#56b356',u'#80c680']


# In[22]:


fig, axes = plt.subplots(1,2,figsize = (18, 5))

for ii, enki in enumerate(enkis):
    axes[0].plot(enki.metrics['v'], label = Js[ii], color = colors[ii]) ;
    axes[1].plot(enki.metrics['r'], label = Js[ii], color = colors[ii]) ;

plt.legend()
axes[0].set_title(r'Ensemble location');
axes[1].set_title(r'Collapse evolution');
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);


# In[24]:


fig, axes = plt.subplots(1,2,figsize = (18, 5))
Js = 2**np.arange(3, 12, 2)

for ii, enki in enumerate(enkis):
    axes[0].plot(enki.metrics['t'], enki.metrics['v'], label = Js[ii], color = colors[ii]) ;
    axes[1].plot(enki.metrics['t'], enki.metrics['r'], label = Js[ii], color = colors[ii]) ;

plt.legend()
axes[0].set_title(r'Ensemble location');
axes[1].set_title(r'Collapse evolution');
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);

#directory = '/Users/agarbuno/github-repos/eks-revision/img/darcy'
#file = 'evolve'
#plt.savefig(directory+file+'.eps', format='eps', dpi=300)


# In[25]:


for ii, enki in tqdm(enumerate(enkis)):                 # Cycling through each dimension
    enki.metrics['sob_v'] = []
    enki.metrics['sob_r'] = []
    for t in np.arange(enki.Uall.shape[0]):       # Cycling through each time step
        enki.metrics['sob_v'].append(np.sqrt((((enki.Uall[t] - enki.Uall[t].mean(axis = 1).reshape(-1,1)) * eigs.flatten().reshape(-1,1))**2).sum(axis = 0).mean()))
        enki.metrics['sob_r'].append(np.sqrt((((enki.Uall[t] - enki.ustar) * eigs.flatten().reshape(-1,1))**2).sum(axis = 0).mean()))


# In[26]:


for ii, enki in tqdm(enumerate(enkis)):                 # Cycling through each dimension
    enki.metrics['L_v'] = []
    enki.metrics['L_r'] = []
    for t in np.arange(enki.Uall.shape[0]):       # Cycling through each time step
        enki.metrics['L_v'].append(np.sqrt(((( enki.Uall[t] - enki.Uall[t].mean(axis = 1).reshape(-1,1) ))**2).sum(axis = 0).mean()))
        enki.metrics['L_r'].append(np.sqrt(((( enki.Uall[t] - enki.ustar ))**2).sum(axis = 0).mean()))


# In[28]:


fig, axes = plt.subplots(1,2,figsize = (18, 5))
Js = 2**np.arange(3, 12, 2)

for ii, enki in enumerate(enkis):
    axes[0].plot(enki.metrics['t'], enki.metrics['L_v'][:-1], label = Js[ii], color = colors[ii]) ;
    axes[1].plot(enki.metrics['t'], enki.metrics['L_r'][:-1], label = Js[ii], color = colors[ii]) ;

plt.legend()
axes[0].set_title(r'Ensemble location');
axes[1].set_title(r'Collapse evolution');
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);

#directory = '/Users/agarbuno/github-repos/eks-revision/img/darcy'
#file = 'evolve'
#plt.savefig(directory+file+'.eps', format='eps', dpi=300)


# In[29]:


fig, axes = plt.subplots(1,2,figsize = (18, 5))
Js = 2**np.arange(3, 12, 2)

for ii, enki in enumerate(enkis):
    axes[0].plot(enki.metrics['t'], enki.metrics['sob_v'][:-1], label = Js[ii], color = colors[ii]) ;
    axes[1].plot(enki.metrics['t'], enki.metrics['sob_r'][:-1], label = Js[ii], color = colors[ii]) ;

plt.legend()
axes[0].set_title(r'Ensemble location');
axes[1].set_title(r'Collapse evolution');
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);

#directory = '/Users/agarbuno/github-repos/eks-revision/img/darcy'
#file = 'evolve_sob'
#plt.savefig(directory+file+'.eps', format='eps', dpi=300)


# In[30]:


fig, axes = plt.subplots(1,1,figsize = (18, 5))

for ii, enki in enumerate(enkis):
    red = (np.var(enki.Uall[-1], axis = 1)/np.var(enki.Uall[0], axis = 1))
    plt.plot(1-red[rank], label = Js[ii], color = colors[ii])
    #print(256. - red.sum())

plt.xscale('log')
plt.title('Posterior variance reduction')
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);
plt.xlabel('Rank by Eigenvalues');


# In[32]:


256. - red.sum()


# In[35]:


fig, axes = plt.subplots(1,1,figsize = (18, 5)) 

for ii, enki in enumerate(enkis):
    resid = enki.Ustar - model.ustar.reshape(-1,1)

    plt.plot(resid.mean(axis = 1)[rank],  label = Js[ii], color = colors[ii])
    plt.fill_between(np.arange(model.Nmesh**2), 
                     resid.mean(axis = 1)[rank] - 2.0 * resid.std(axis = 1)[rank],
                     resid.mean(axis = 1)[rank] + 2.0 * resid.std(axis = 1)[rank], 
                     alpha = .2, color = colors[ii])
plt.xscale('log')
plt.title('Residual');
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);
plt.xlabel('Rank by Eigenvalues');


# In[37]:


fig, axes = plt.subplots(1,1,figsize = (18, 5)) 

for ii, enki in enumerate(enkis):
    resid = enki.Ustar - model.ustar.reshape(-1,1)
    sns.kdeplot(np.sqrt((resid**2).sum(axis = 0)), label = Js[ii], 
                color = colors[ii], shade = True)
    
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);


# In[39]:


fig, axes = plt.subplots(1,1,figsize = (18, 5)) 

for ii, enki in enumerate(enkis):
    resid = enki.Ustar - model.ustar.reshape(-1,1)
    norms = np.sqrt(((resid**2)[rank,:] * eigs.flatten()[rank,np.newaxis]).sum(axis = 0))
    sns.kdeplot(norms, label = Js[ii], color = colors[ii], shade = True)
    
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);


# In[43]:


prior = multivariate_normal(mean = enki.mu.flatten(), cov = enki.sigma)


# In[47]:


prior.logpdf(current)


# In[48]:


n_mcmc = 50000
beta   = 0.02
scales = np.sqrt(np.diag(np.cov(enki.Ustar)))
samples = []
current = enkis[-1].Ustar.mean(axis = 1)
flag_noise = False

g = model(current)
phi_current = ((g - y_obs) * np.linalg.solve(2 * Gamma, g - y_obs)).sum()
phi_current -= prior.logpdf(current)

samples.append(current)
accept = 0.

for k in tqdm(range(n_mcmc), desc = 'MCMC: '):
    proposal = np.sqrt(1 - beta**2) * current + np.sqrt(beta) * scales * np.random.normal(0, 1, enki.p)
    
    g_proposal = model(proposal)
    phi_proposal = ((g_proposal - y_obs) * np.linalg.solve(2 * Gamma, g_proposal - y_obs)).sum()
    phi_proposal -= prior.logpdf(proposal)

    if np.random.uniform() < np.exp(phi_current - phi_proposal):
        current = proposal
        phi_current = phi_proposal
        accept += 1.
        
    samples.append(current)
    
    
print('Acceptance rate: %s'%(accept/n_mcmc))

samples_truth = np.array(samples)


# In[49]:


fig, axes = plt.subplots(1,1,figsize = (18, 5))

for ii, enki in enumerate(enkis):
    red = (np.var(enki.Uall[-1], axis = 1)/np.var(enki.Uall[0], axis = 1))
    plt.plot(1-red[rank], label = Js[ii], color = colors[ii])
    print(256. - red.sum())

mcmc_red = np.var(samples_truth, axis = 0)/np.var(enki.Uall[0], axis = 1)
plt.plot(1-mcmc_red[rank], label = 'MCMC', color = u'#ff7f0e')
print(256. - mcmc_red.sum())
    
plt.xscale('log')
plt.title('Posterior variance reduction')
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);
plt.xlabel('Rank by Eigenvalue');

#directory = '/Users/agarbuno/github-repos/eks-revision/img/darcy_'
#file = 'variance_reduction'
#plt.savefig(directory+file+'.eps', format='eps', dpi=300, bbox_inches='tight')


# In[54]:


fig, axes = plt.subplots(1,1,figsize = (18, 5)) 

for ii, enki in enumerate(enkis):
    resid = enki.Ustar - model.ustar.reshape(-1,1)

    plt.plot(resid.mean(axis = 1)[rank],  label = Js[ii], color = colors[ii])
    plt.fill_between(np.arange(model.Nmesh**2), 
                     resid.mean(axis = 1)[rank] - 2.0 * resid.std(axis = 1)[rank],
                     resid.mean(axis = 1)[rank] + 2.0 * resid.std(axis = 1)[rank], 
                     alpha = .2, color = colors[ii])

resid_mcmc = (samples_truth - model.ustar)    

plt.plot(resid_mcmc.mean(axis = 0)[rank],  label = 'MCMC', color = u'#ff7f0e')

plt.fill_between(np.arange(model.Nmesh**2), 
                 resid_mcmc.mean(axis = 0)[rank] - 2.0 * resid_mcmc.std(axis = 0)[rank],
                 resid_mcmc.mean(axis = 0)[rank] + 2.0 * resid_mcmc.std(axis = 0)[rank], 
                 alpha = .2, color = u'#ff7f0e')    

plt.xscale('log')
plt.title('Residuals');
plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.);
plt.xlabel('Rank by Eigenvalues');

#directory = '/Users/agarbuno/github-repos/eks-revision/img/darcy_'
#file = 'component_marginals'
#plt.savefig(directory+file+'.pdf', dpi=300, bbox_inches='tight')

