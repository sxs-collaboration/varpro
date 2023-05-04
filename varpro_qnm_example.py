#!/usr/bin/env python

import numpy as np

### varpro example
t = np.arange(0,90,0.1)

## two mode example
vp_maxN = 1
a0 = 0.3 + 1.j*0.8
a1 = 0.7 + 1.j*0.5
om0 = 0.55 - 0.08*1.j
om1 = 0.35 - 0.2*1.j
wf = a0*np.exp(-1.j*om0*t) + a1*np.exp(-1.j*om1*t)

## modes might be out of order in varpro results
print('expected linear parameters: ',np.real(a0),np.imag(a0),
      np.real(a1),np.imag(a1))
print('expected nonlinear parameters: ',-np.imag(om0),np.real(om0),
      -np.imag(om1),np.real(om1))

# ## four mode example
# vp_maxN = 3
# a0 = 0.3 + 1.j*0.4
# a1 = 0.7 + 1.j*0.6
# a2 = 0.2 + 1.j*0.1
# a3 = 0.8 + 1.j*0.9
# om0 = 0.6 - 0.08*1.j
# om1 = 0.5 - 0.16*1.j
# om2 = 0.4 - 0.32*1.j
# om3 = 0.3 - 0.64*1.j
# wf = a0*np.exp(-1.j*om0*t) + a1*np.exp(-1.j*om1*t) \
#    + a2*np.exp(-1.j*om2*t) + a3*np.exp(-1.j*om3*t)

# ## modes might be out of order in varpro results
# print('expected linear parameters: ',np.real(a0),np.imag(a0),
#       np.real(a1),np.imag(a1),
#       np.real(a2),np.imag(a2),
#       np.real(a3),np.imag(a3))
# print('expected nonlinear parameters: ',-np.imag(om0),np.real(om0),
#       -np.imag(om1),np.real(om1),
#       -np.imag(om2),np.real(om2),
#       -np.imag(om3),np.real(om3))



def examplePhiFunction(alpha,t):

   phiN = 2*(vp_maxN + 1)
   Phi = np.empty([t.shape[0],phiN])
   dPhi = np.empty([t.shape[0],2*phiN]);

   ind1 = []
   ind2 = []
   for n in np.arange(0,phiN,2):
      n0 = n
      n1 = n + 1
      Phi[:,n0] = np.multiply(np.exp(- alpha[n0] * t),np.cos(alpha[n1] * t))
      Phi[:,n1] = np.multiply(np.exp(- alpha[n0] * t),np.sin(alpha[n1] * t))

      ind1.extend([n0,n0,n1,n1])
      ind2.extend([n0,n1,n0,n1])

      dPhi[:,2*n] = np.multiply(- np.ndarray.flatten(t),Phi[:,n0])
      dPhi[:,2*n+1] = np.multiply(np.multiply(- t,np.exp(- alpha[n0] * t)),
                                  np.sin(alpha[n1] * t))
      dPhi[:,2*n+2] = np.multiply(- np.ndarray.flatten(t),Phi[:,n1])
      dPhi[:,2*n+3] = np.multiply(np.multiply(t,np.exp(- alpha[n0] * t)),
                                  np.cos(alpha[n1] * t))

   Ind = np.array([ind1,ind2])

   return Phi,dPhi,Ind



from varpro import *

## set the real and imag parts for varpro
y = np.real(wf)
yi = np.imag(wf)

## set tolerance
vp_tol = 1.0e-8
## set the weight to 1
weight = np.ones(len(y))

## set guess (setting all to 1 for now)
this_guess = np.ones(2*(vp_maxN+1))

#### call varpro (with or without bounds)

## no bounds
alpha,c,wresid,resid_norm,y_est,CorMx,sigmas = varpro(t,y,yi,weight,this_guess,2*(vp_maxN+1),lambda alpha: examplePhiFunction(alpha,t),None,ftol=vp_tol,gtol=vp_tol,xtol=vp_tol,max_nfev=1000)

# ## using bounds
# ## (bounds example: sets the lower bound to zero for each mode and upper bound to inf)
# bounds = (list(this_guess*0),list(this_guess*0+np.inf))
# alpha,c,wresid,resid_norm,y_est,CorMx,sigmas = varpro(t,y,yi,weight,this_guess,2*(vp_maxN+1),lambda alpha: examplePhiFunction(alpha,t),bounds,ftol=vp_tol,gtol=vp_tol,xtol=vp_tol,max_nfev=1000)
