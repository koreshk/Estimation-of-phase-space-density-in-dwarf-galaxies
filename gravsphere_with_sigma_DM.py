
###########################################################
#GravSphere
###########################################################

#Python programme to Jeans model discrete data assuming
#a "coreNFWtides" spherical dark matter halo and some 
#fixed radial profile for the "baryons" with varying 
#mass to light ratio. The code and its various improvements 
#is described in the following papers:
#https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4541R/abstract
#https://ui.adsabs.harvard.edu/abs/2018MNRAS.481..860R/abstract
#https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.1401R/abstract
#https://ui.adsabs.harvard.edu/abs/2020MNRAS.498..144G/abstract
#https://ui.adsabs.harvard.edu/abs/2020JCAP...09..004A/abstract
#https://ui.adsabs.harvard.edu/abs/2020arXiv201003572A/abstract

#To run the code, you should first "prepare" your data in
#the format GravSphere needs using the "binulator.py" code. 
#You will find examples for how to do this on real and mock 
#data there.

###########################################################
#Functions for emcee Jeans fitting:
def lnprior_set_single(theta,n_betpars,bet0min,bet0max,\
                       betinfmin,betinfmax,\
                       betr0min,betr0max,betnmin,betnmax,\
                       nu_components,nupars_min,nupars_max,\
                       n_mpars,logM200low,logM200high,\
                       clow,chigh,logrclow,logrchigh,\
                       nlow,nhigh,logrtlow,logrthigh,\
                       dellow,delhigh,\
                       logMcenlow,logMcenhigh,\
                       acenlow,acenhigh,\
                       Arotlow,Arothigh,\
                       drangelow,drangehigh,\
                       Mstar_min,Mstar_max):

    ndims = len(theta)
    minarr = np.zeros(ndims)
    maxarr = np.zeros(ndims)
    minarr[0] = bet0min
    maxarr[0] = bet0max
    minarr[1] = betinfmin
    maxarr[1] = betinfmax
    minarr[2] = betr0min
    maxarr[2] = betr0max
    minarr[3] = betnmin
    maxarr[3] = betnmax

    minarr[n_betpars:n_betpars+nu_components*2] = nupars_min
    maxarr[n_betpars:n_betpars+nu_components*2] = nupars_max
    minarr[n_betpars+nu_components*2] = logM200low
    maxarr[n_betpars+nu_components*2] = logM200high
    minarr[n_betpars+nu_components*2+1] = clow
    maxarr[n_betpars+nu_components*2+1] = chigh
    minarr[n_betpars+nu_components*2+2] = logrclow
    maxarr[n_betpars+nu_components*2+2] = logrchigh
    minarr[n_betpars+nu_components*2+3] = nlow
    maxarr[n_betpars+nu_components*2+3] = nhigh
    minarr[n_betpars+nu_components*2+4] = logrtlow
    maxarr[n_betpars+nu_components*2+4] = logrthigh
    minarr[n_betpars+nu_components*2+5] = dellow
    maxarr[n_betpars+nu_components*2+5] = delhigh
    minarr[n_betpars+nu_components*2+6] = logMcenlow
    maxarr[n_betpars+nu_components*2+6] = logMcenhigh
    minarr[n_betpars+nu_components*2+7] = acenlow
    maxarr[n_betpars+nu_components*2+7] = acenhigh
    minarr[n_betpars+nu_components*2+8] = Arotlow
    maxarr[n_betpars+nu_components*2+8] = Arothigh
    minarr[n_betpars+nu_components*2+9] = drangelow
    maxarr[n_betpars+nu_components*2+9] = drangehigh
    minarr[ndims-1] = Mstar_min
    maxarr[ndims-1] = Mstar_max

    if all(minarr < theta < maxarr for minarr,theta,maxarr in \
               zip(minarr,theta,maxarr)):
        return 0.0
    return -np.inf

def lnprob_single(theta,x1,x2,y1,y1err,y2,y2err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x1,x2,y1,y1err,y2,y2err)

def lnprob_single_vs(theta,x1,x2,y1,y1err,\
                     y2,y2err,\
                     vsp1val,vsp1pdf,vsp2val,vsp2pdf):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x1,x2,y1,y1err,\
                       y2,y2err,\
                       vsp1val,vsp1pdf,vsp2val,vsp2pdf)

def lnprob_single_prop(theta,x1,x2,x3,y1,y1err,\
                       y2,y2err,y3,y3err,y4,y4err):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x1,x2,x3,y1,y1err,\
                       y2,y2err,y3,y3err,y4,y4err)

def lnprob_single_prop_vs(theta,x1,x2,x3,y1,y1err,\
                          y2,y2err,y3,y3err,y4,y4err,\
                          vsp1val,vsp1pdf,vsp2val,vsp2pdf):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x1,x2,x3,y1,y1err,\
                       y2,y2err,y3,y3err,y4,y4err,\
                       vsp1val,vsp1pdf,vsp2val,vsp2pdf)

def lnlike_single(theta,x1,x2,y1,y1err,y2,y2err):
    betpars = theta[0:n_betpars]
    nupars = theta[n_betpars:n_betpars+nu_components*2]
    Mpars = theta[n_betpars+nu_components*2:\
                  n_betpars+nu_components*2+n_mpars]
    Arot = theta[n_betpars+nu_components*2+n_mpars]
    drange = theta[n_betpars+nu_components*2+n_mpars+1]
    Mstar = theta[ndim-1]
    nuparsu = np.array(nupars)
    Mparsu = np.array(Mpars)
    Mparsu[0] = 10.**Mpars[0]
    Mparsu[2] = 10.**Mpars[2]
    Mparsu[4] = 10.**Mpars[4]
    Mparsu[6] = 10.**Mpars[6]

    #Add dummy data points for low and high x1
    #to ensure +ve definite surface density
    #if using negative Plummer components:
    if (nuparsu[0] < 0 or nuparsu[1] < 0 or nuparsu[2] < 0):
        x1, y1, y1err = Sig_addpnts(x1,y1,y1err)
    
    sigr2, Sig, sigLOS2 = \
        sigp_fit(x1*drange,x2*drange,\
                 nuparsu,Mparsu,betpars,Mstar,Arot)

    #And now, shrink the error wherever the
    #total density is negative to disfavour
    #those models (see binulator_surffuncs.py
    #for details):
    if (theta[0] < 0 or theta[1] < 0 or theta[2] < 0):
        if (np.min(Sig) < 0):
            y1err[np.where(Sig < 0)] = \
                np.min(y1err)/np.float(len(x1))/1.0e3
        
    #Handle distance uncertainty. Convert distances to
    #angle units before doing model-data comparison:
    duse = dgal_kpc * drange
    x1u = x1 / dgal_kpc
    x2u = x2 / dgal_kpc
            
    #And convert model to angel units, but using
    #new rather than default distance:
    model1 = Sig
    model2 = np.sqrt(sigLOS2)/1000.

    inv_sigma2_1 = 1.0/y1err**2
    inv_sigma2_2 = 1.0/y2err**2

    lnlike_out = -0.5*(np.sum((y1-model1)**2*inv_sigma2_1)+\
                       np.sum((y2-model2)**2*inv_sigma2_2))

    if (cosmo_cprior == 'yes'):
        #Add the conc. to the likelihood function
        #as a Gaussian in logspace:
        M200 = Mparsu[0]
        log_conc = np.log10(Mparsu[1])
        log_cmean = np.log10(cosmo_cfunc(M200,h))
        lnlike_out = lnlike_out - \
                     (log_conc-log_cmean)**2.0/(2.0*sig_c200**2.0)
    
    if (lnlike_out != lnlike_out):
        lnlike_out = -np.inf

    return lnlike_out

def lnlike_single_vs(theta,x1,x2,y1,y1err,y2,y2err,\
                     vsp1val,vsp1pdf,vsp2val,vsp2pdf):
    betpars = theta[0:n_betpars]
    nupars = theta[n_betpars:n_betpars+nu_components*2]
    Mpars = theta[n_betpars+nu_components*2:\
                  n_betpars+nu_components*2+n_mpars]
    Arot = theta[n_betpars+nu_components*2+n_mpars]
    drange = theta[n_betpars+nu_components*2+n_mpars+1]
    Mstar = theta[ndim-1]
    nuparsu = np.array(nupars)
    Mparsu = np.array(Mpars)
    Mparsu[0] = 10.**Mpars[0]
    Mparsu[2] = 10.**Mpars[2]
    Mparsu[4] = 10.**Mpars[4]
    Mparsu[6] = 10.**Mpars[6]

    #Add dummy data points for low and high x1
    #to ensure +ve definite surface density
    #if using negative Plummer components:
    if (nuparsu[0] < 0 or nuparsu[1] < 0 or nuparsu[2] < 0):
        x1, y1, y1err = Sig_addpnts(x1,y1,y1err)
    
    sigr2, Sig, sigLOS2, vs1, vs2 = \
        sigp_fit_vs(x1*drange,x2*drange,\
                    nuparsu,Mparsu,betpars,Mstar,Arot)
    
    #And now, shrink the error wherever the
    #total density is negative to disfavour
    #those models (see binulator_surffuncs.py
    #for details):
    if (theta[0] < 0 or theta[1] < 0 or theta[2] < 0):
        if (np.min(Sig) < 0):
            y1err[np.where(Sig < 0)] = \
                  np.min(y1err)/np.float(len(x1))/1.0e3

    #Handle distance uncertainty. Convert distances to
    #angle units before doing model-data comparison:
    duse = dgal_kpc * drange
    x1u = x1 / dgal_kpc
    x2u = x2 / dgal_kpc
    
    #And convert model to angle units, but using
    #new rather than default distance:
    model1 = Sig
    model2 = np.sqrt(sigLOS2)/1000.
    model3 = vs1/1.0e12
    model4 = vs2/1.0e12

    inv_sigma2_1 = 1.0/y1err**2
    inv_sigma2_2 = 1.0/y2err**2

    lnlike_out = -0.5*(np.sum((y1-model1)**2*inv_sigma2_1)+\
                 np.sum((y2-model2)**2*inv_sigma2_2))+\
                 np.log(vsp_pdf(model3,vsp1val,vsp1pdf))+\
                 np.log(vsp_pdf(model4,vsp2val,vsp2pdf))

    if (cosmo_cprior == 'yes'):
        #Add the conc. to the likelihood function
        #as a Gaussian in logspace:
        M200 = Mparsu[0]
        log_conc = np.log10(Mparsu[1])
        log_cmean = np.log10(cosmo_cfunc(M200,h))
        lnlike_out = lnlike_out - \
                     (log_conc-log_cmean)**2.0/(2.0*sig_c200**2.0)
    
    if (lnlike_out != lnlike_out):
        lnlike_out = -np.inf

    return lnlike_out

def lnlike_single_prop(theta,x1,x2,x3,y1,y1err,y2,y2err,\
                       y3,y3err,y4,y4err):
    betpars = theta[0:n_betpars]
    nupars = theta[n_betpars:n_betpars+nu_components*2]
    Mpars = theta[n_betpars+nu_components*2:\
                  n_betpars+nu_components*2+n_mpars]
    Arot = theta[n_betpars+nu_components*2+n_mpars]
    drange = theta[n_betpars+nu_components*2+n_mpars+1]
    Mstar = theta[ndim-1]
    nuparsu = np.array(nupars)
    Mparsu = np.array(Mpars)
    Mparsu[0] = 10.**Mpars[0]
    Mparsu[2] = 10.**Mpars[2]
    Mparsu[4] = 10.**Mpars[4]
    Mparsu[6] = 10.**Mpars[6]

    #Add dummy data points for low and high x1
    #to ensure +ve definite surface density
    #if using negative Plummer components:
    if (nuparsu[0] < 0 or nuparsu[1] < 0 or nuparsu[2] < 0):
        x1, y1, y1err = Sig_addpnts(x1,y1,y1err)

    sigr2, Sig, sigLOS2, sigpmr2, sigpmt2 = \
        sigp_fit_prop(x1*drange,x2*drange,x3*drange,\
                      nuparsu,Mparsu,betpars,Mstar,Arot)

    #And now, shrink the error wherever the
    #total density is negative to disfavour
    #those models (see binulator_surffuncs.py
    #for details):
    if (theta[0] < 0 or theta[1] < 0 or theta[2] < 0):
        if (np.min(Sig) < 0):
            y1err[np.where(Sig < 0)] = \
                  np.min(y1err)/np.float(len(x1))/1.0e3

    #Handle distance uncertainty. Convert distances to
    #angle units before doing model-data comparison:
    duse = dgal_kpc * drange
    x1u = x1 / dgal_kpc
    x2u = x2 / dgal_kpc
    x3u = x3 / dgal_kpc
    y3u = y3 / dgal_kpc
    y3erru = y3err / dgal_kpc
    y4u = y4 / dgal_kpc
    y4erru = y4err / dgal_kpc

    #And convert model to angle units, but using
    #new rather than default distance:
    model1 = Sig
    model2 = np.sqrt(sigLOS2)/1000.
    model3 = np.sqrt(sigpmr2)/1000. / duse
    model4 = np.sqrt(sigpmt2)/1000. / duse

    inv_sigma2_1 = 1.0/y1err**2
    inv_sigma2_2 = 1.0/y2err**2
    inv_sigma2_3 = 1.0/y3erru**2
    inv_sigma2_4 = 1.0/y4erru**2

    lnlike_out = -0.5*(np.sum((y1-model1)**2*inv_sigma2_1)+\
                       np.sum((y2-model2)**2*inv_sigma2_2)+\
                       np.sum((y3u-model3)**2*inv_sigma2_3)+\
                       np.sum((y4u-model4)**2*inv_sigma2_4))

    if (cosmo_cprior == 'yes'):
        #Add the conc. to the likelihood function
        #as a Gaussian in logspace:
        M200 = Mparsu[0]
        log_conc = np.log10(Mparsu[1])
        log_cmean = np.log10(cosmo_cfunc(M200,h))
        lnlike_out = lnlike_out - \
            (log_conc-log_cmean)**2.0/(2.0*sig_c200**2.0)
        
    if (lnlike_out != lnlike_out):
        lnlike_out = -np.inf            
    
    return lnlike_out

def lnlike_single_prop_vs(theta,x1,x2,x3,y1,y1err,y2,y2err,\
                          y3,y3err,y4,y4err,\
                          vsp1val,vsp1pdf,vsp2val,vsp2pdf):
    betpars = theta[0:n_betpars]
    nupars = theta[n_betpars:n_betpars+nu_components*2]
    Mpars = theta[n_betpars+nu_components*2:\
                  n_betpars+nu_components*2+n_mpars]
    Arot = theta[n_betpars+nu_components*2+n_mpars]
    drange = theta[n_betpars+nu_components*2+n_mpars+1]
    Mstar = theta[ndim-1]
    nuparsu = np.array(nupars)
    Mparsu = np.array(Mpars)
    Mparsu[0] = 10.**Mpars[0]
    Mparsu[2] = 10.**Mpars[2]
    Mparsu[4] = 10.**Mpars[4]
    Mparsu[6] = 10.**Mpars[6]

    #Add dummy data points for low and high x1
    #to ensure +ve definite surface density
    #if using negative Plummer components:
    if (nuparsu[0] < 0 or nuparsu[1] < 0 or nuparsu[2] < 0):
        x1, y1, y1err = Sig_addpnts(x1,y1,y1err)

    sigr2, Sig, sigLOS2, sigpmr2, sigpmt2, vs1, vs2 = \
        sigp_fit_prop_vs(x1*drange,x2*drange,x3*drange,\
                         nuparsu,Mparsu,betpars,Mstar,Arot)

    #And now, shrink the error wherever the
    #total density is negative to disfavour
    #those models (see binulator_surffuncs.py
    #for details):
    if (theta[0] < 0 or theta[1] < 0 or theta[2] < 0):
        if (np.min(Sig) < 0):
            y1err[np.where(Sig < 0)] = \
                  np.min(y1err)/np.float(len(x1))/1.0e3
            
    #Handle distance uncertainty. Convert distances to
    #angle units before doing model-data comparison:
    duse = dgal_kpc * drange
    x1u = x1 / dgal_kpc
    x2u = x2 / dgal_kpc
    x3u = x3 / dgal_kpc
    y3u = y3 / dgal_kpc
    y3erru = y3err / dgal_kpc
    y4u = y4 / dgal_kpc
    y4erru = y4err / dgal_kpc
    
    #And convert model to angle units, but using
    #new rather than default distance:                                                        
    model1 = Sig
    model2 = np.sqrt(sigLOS2)/1000.
    model3 = np.sqrt(sigpmr2)/1000. / duse
    model4 = np.sqrt(sigpmt2)/1000. / duse
    model5 = vs1/1.0e12
    model6 = vs2/1.0e12
    
    inv_sigma2_1 = 1.0/y1err**2
    inv_sigma2_2 = 1.0/y2err**2
    inv_sigma2_3 = 1.0/y3erru**2
    inv_sigma2_4 = 1.0/y4erru**2
    
    lnlike_out = -0.5*(np.sum((y1-model1)**2*inv_sigma2_1)+\
                       np.sum((y2-model2)**2*inv_sigma2_2)+\
                       np.sum((y3u-model3)**2*inv_sigma2_3)+\
                       np.sum((y4u-model4)**2*inv_sigma2_4))+\
                       np.log(vsp_pdf(model5,vsp1val,vsp1pdf))+\
                       np.log(vsp_pdf(model6,vsp2val,vsp2pdf))

    if (cosmo_cprior == 'yes'):
        #Add the conc. to the likelihood function
        #as a Gaussian in logspace:
        M200 = Mparsu[0]
        log_conc = np.log10(Mparsu[1])
        log_cmean = np.log10(cosmo_cfunc(M200,h))
        lnlike_out = lnlike_out - \
            (log_conc-log_cmean)**2.0/(2.0*sig_c200**2.0)
        
    if (lnlike_out != lnlike_out):
        lnlike_out = -np.inf
        
    return lnlike_out


###########################################################
#Main code:

#Suppress warning output:
import warnings
warnings.simplefilter("ignore")

#Forbid plots to screen so GravSphere can run
#remotely:
import matplotlib as mpl
mpl.use('Agg')

#Imports & dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import emcee
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.integrate import simps as integrator
from functions import *
from constants import *
from binulator_surffuncs import * 
from binulator_velfuncs import * 
from figures import * 
import sys

#Welcome blurb: 
print('###### GRAVSPHERE VERSION 1.5 ######\n')

#Default to run on a single CPU:
nprocs = 1

###########################################################
#Code parameters:
datadir = './Data/'
nwalkers = 250
nmodels = 50000

#Codemode [run or plot]:
codemode = 'plot'

###########################################################
#Input data selection here.

#MW satellites:
from gravsphere_initialise_Fornax import *
#from gravsphere_initialise_Draco import *
#from gravsphere_initialise_UMi import *
#from gravsphere_initialise_Carina import *
#from gravsphere_initialise_LeoI import *
#from gravsphere_initialise_LeoII import *
#from gravsphere_initialise_Sextans import *
#from gravsphere_initialise_Sculptor import *
#from gravsphere_initialise_Fornax import *
#from gravsphere_initialise_CVnI import *
#from gravsphere_initialise_SegI import *
#from gravsphere_initialise_SMC import *
#from gravsphere_initialise_Ocen import *

#Mocks:
#from gravsphere_initialise_PlumCoreOm import *
#from gravsphere_initialise_PlumCuspOm import *
#from gravsphere_initialise_SMCmock import *
#from gravsphere_initialise_Ocenmock import *

#M31 satellites:
#from gravsphere_initialise_And21 import *

np.int = int
nprocs = 64

#Output some key choices:
print('Running on %d cores' % (nprocs))
print('Doing galaxy:',whichgal)
print('Model parameters:')
print('M200low, M200high [1e9 Msun]:', \
    10.0**logM200low/1.0e9, 10.0**logM200high/1.0e9)
print('clow, chigh:', clow, chigh)
if (cosmo_cprior == 'yes'):
    if (mWDM > 0):
        print('Warm dark matter cosmology with mWDM(keV):',mWDM)
    else:
        print('Cold dark matter cosmology')
if (logMcenlow > 0.0):
    print('Including central dark mass in the range [1e6 Msun]:', \
          10.0**logMcenlow/1.0e6, 10.0**logMcenhigh/1.0e6)
if (Arothigh > 1.0e-3):
    print('Including rotation')
if (drangelow < 0.999):
    print('Allowing distance to vary in the fit over the range [kpc]:', \
          dgal_kpc*drangelow, dgal_kpc*drangehigh)

#Set up output data folder structure:
outdir = outdirbase
if (propermotion == 'yes'):
    outdir = outdir + 'Propermotion/'
if (virialshape == 'yes'):
    outdir = outdir + 'VirialShape/'
if (cosmo_cprior == 'yes'):
    outdir = outdir + 'CosmoC/'

#Set tracer model and vel. ani. functions:
n_betpars = 4
nu = multiplumden
nu_components = 3
Sigfunc = multiplumsurf
n_nupars = nu_components * 2

if (propermotion == 'no'):
    if (virialshape == 'no'):
        lnlike = lnlike_single
        lnprob = lnprob_single
        lnprior_set = lnprior_set_single
    else:
        lnlike = lnlike_single_vs
        lnprob = lnprob_single_vs
        lnprior_set = lnprior_set_single
elif (propermotion == 'yes'):
    if (virialshape == 'no'):
        lnlike = lnlike_single_prop
        lnprob = lnprob_single_prop
        lnprior_set = lnprior_set_single
    else:
        lnlike = lnlike_single_prop_vs
        lnprob = lnprob_single_prop_vs
        lnprior_set = lnprior_set_single

###########################################################
#Read in the the data. We assume here that the errors
#on the velocity dispersion are Gaussian symmetric.
#If this is not a good approximation (c.f. output from the 
#binulator), then this can be improved. The errors on the 
#VSPs are rarely Gaussian symmetric and so we use the 
#correct likelihood function from the binulator in this case.
data = np.genfromtxt(infile+'_p0best.txt',dtype='f8')
pfits = data
data = np.genfromtxt(infile+'_Rhalf.txt',dtype='f8')
Rhalf = data[0]
data = np.genfromtxt(infile+'_surfden.txt',dtype='f8')
rbin_phot = data[:,0]
surfden = data[:,1]
surfdenerr = data[:,2]
data = np.genfromtxt(infile+'_vel.txt',dtype='f8')
rbin_kin = data[:,0]
sigpmean = data[:,4]
sigperr = (data[:,6]-data[:,5])/2.0
data = np.genfromtxt(infile+'_vsps.txt',dtype='f8')
vs1bin = data[0,0]
vs1err = (data[0,2]-data[0,1])/2.0
vs1lo = data[0,1]
vs1hi = data[0,2]
vs2bin = data[1,0]
vs2err = (data[1,2]-data[1,1])/2.0
vs2lo = data[1,1]
vs2hi = data[1,2]
data = np.genfromtxt(infile+'_vsp1full.txt',dtype='f8')
vsp1val, vsp1pdf = vsppdf_calc(data)
data = np.genfromtxt(infile+'_vsp2full.txt',dtype='f8')
vsp2val, vsp2pdf = vsppdf_calc(data)

if (propermotion == 'yes'):
    data = np.genfromtxt(infile+'_velproptan.txt',dtype='f8')
    rbin_kinp = data[:,0]
    sigpmt = data[:,4]
    sigpmterr = (data[:,6]-data[:,5])/2.0

    data = np.genfromtxt(infile+'_velpropR.txt',dtype='f8')
    rbin_kinp2 = data[:,0]
    sigpmr = data[:,4]
    sigpmrerr = (data[:,6]-data[:,5])/2.0

    #Check data:
    if (np.sum(rbin_kinp-rbin_kinp2) != 0):
        print('Need same radial binning for tangential and radial propermotions. Oops! Bye bye.')
        sys.exit(0)
    
print('Inner/outer radial bin (phot):', \
    np.min(rbin_phot),np.max(rbin_phot))
print('Inner/outer radial bin (kin):', \
    np.min(rbin_kin),np.max(rbin_kin))

#Set up the baryonic mass profile. If this is
#not assumed to have the same radial profile
#as the tracer stars, then it must be set, above,
#in the galaxy initialisation script. This
#should be normalised to peak at 1.0 so that 
#when multiplied by Mstar, it yields the total
#stellar mass.
if (baryonmass_follows_tracer == 'yes'):
    if (barrad_min == 0):
        barrad_min = 1.0e-3
    Mstar_rad = np.logspace(np.log10(barrad_min),\
        np.log10(barrad_max),np.int(bar_pnts))
    norm = pfits[0] + pfits[1] + pfits[2]
    Mstar_prof = \
        threeplummass(Mstar_rad,pfits[0]/norm,\
                      pfits[1]/norm,pfits[2]/norm,\
                      pfits[3],pfits[4],pfits[5])

#Set beta scale radius based on Rhalf:
betr0min = np.log10(0.5*Rhalf)
betr0max = np.log10(2.0*Rhalf)
        
#Set Jeans radial-grid based also on Rhalf:
if (rmax < 0):
    rmin = Rhalf / 100.0
    rmax = Rhalf * 50.0
print('Inner/outer radial Jeans grid:', rmin, rmax)

#Set up the mass model functions:
M = lambda r, Mpars: \
    corenfw_tides_mass(r,Mpars[0],Mpars[1],Mpars[2],\
                         Mpars[3],Mpars[4],Mpars[5])
rho = lambda r, Mpars: \
    corenfw_tides_den(r,Mpars[0],Mpars[1],Mpars[2],\
                        Mpars[3],Mpars[4],Mpars[5])
dlnrhodlnr = lambda r, Mpars: \
    corenfw_tides_dlnrhodlnr(r,Mpars[0],Mpars[1],Mpars[2],\
                               Mpars[3],Mpars[4],Mpars[5])
#Central dark mass:
Mcentral = lambda r, Mpars: \
           plummass(r,Mpars[6:8])
n_mpars = 8

#Set min/max priors on the stellar mass:                        
Mstar_min = Mstar - Mstar_err
Mstar_max = Mstar + Mstar_err

#Set up the Jeans functions to use for the fit:
#N.B. +6 on the ndims here corresponds to:
#varying stellar mass to light ratio (+1)
#varying distance (+1)
#varying rotation parameter (+1)
ndim = n_betpars + n_nupars + n_mpars + 3
if (propermotion == 'no'):
    if (virialshape == 'no'):
        sigp_fit = lambda r1,r2,nupars,Mpars,betpars,Mstar,Arot: \
            sigp(r1,r2,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,betpars,\
                 Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,Guse,rmin,rmax)
    else:
        sigp_fit_vs = lambda r1,r2,nupars,Mpars,betpars,Mstar,Arot: \
            sigp_vs(r1,r2,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,betpars,\
                    Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,Guse,rmin,rmax)
#############################################################
        sigma_DM_fit = lambda r2,nupars,Mpars,betpars,Mstar,Arot: \
            sigma_DM_vs(r2,rho,M,Mcentral,beta_DM,betaf_DM,nupars,Mpars,betpars,\
                    Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,Guse,rmin,rmax)
#############################################################
elif (propermotion == 'yes'):
    if (virialshape == 'no'):
        sigp_fit_prop = lambda r1,r2,r3,nupars,Mpars,betpars,Mstar,Arot: \
            sigp_prop(r1,r2,r3,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,betpars,\
                      Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,Guse,rmin,rmax)
    else:
        sigp_fit_prop_vs = lambda r1,r2,r3,nupars,Mpars,betpars,Mstar,Arot: \
            sigp_prop_vs(r1,r2,r3,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,betpars,\
                         Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,Guse,rmin,rmax)

#Set the priors and starting blob for the tracer density profile.
#Code is a bit more involved here just to cope with potentially
#negative Plummer masses, used to fit some steeply falling tracer
#density profiles:
nupars_min = np.zeros(len(pfits))
nupars_max = np.zeros(len(pfits))
nupars_minstart = np.zeros(len(pfits))
nupars_maxstart = np.zeros(len(pfits))
for i in range(len(pfits)):
    if (pfits[i] > 0):
        nupars_min[i] = pfits[i]*(1.0-tracertol)
        nupars_max[i] = pfits[i]*(1.0+tracertol)
    else:
        nupars_min[i] = pfits[i]*(1.0+tracertol)
        nupars_max[i] = pfits[i]*(1.0-tracertol)
    if (tracertol < 0.01):
        nupars_minstart[i] = nupars_min[i]
        nupars_maxstart[i] = nupars_max[i]
    else:
        if (pfits[i] > 0):
            nupars_minstart[i] = pfits[i]*0.99
            nupars_maxstart[i] = pfits[i]*1.01
        else:
            nupars_minstart[i] = pfits[i]*1.01
            nupars_maxstart[i] = pfits[i]*0.99

###########################################################
#Emcee fitting code:

#Set up walkers:
if (codemode == 'run'):
    print('Running in fitting mode ... ')
    print('Will write output to:', outdir)
    
    #Initialise the walkers:
    pos = np.zeros((nwalkers, ndim), dtype='float')
    pos[:,0] = np.random.uniform(bet0min,bet0max,nwalkers)
    pos[:,1] = np.random.uniform(betinfmin,betinfmax,nwalkers)
    pos[:,2] = np.random.uniform(betr0min,betr0max,nwalkers)
    pos[:,3] = np.random.uniform(betnmin,betnmax,nwalkers)
    for i in range(len(pfits)):
        pos[:,n_betpars+i] = \
            np.random.uniform(nupars_minstart[i],\
                nupars_maxstart[i],nwalkers)
    pos[:,n_betpars+nu_components*2] = \
        np.random.uniform(logM200low,logM200high,nwalkers)
    pos[:,n_betpars+nu_components*2+1] = \
        np.random.uniform(clow,chigh,nwalkers)
    pos[:,n_betpars+nu_components*2+2] = \
        np.random.uniform(logrclow,logrchigh,nwalkers)
    pos[:,n_betpars+nu_components*2+3] = \
        np.random.uniform(nlow,nhigh,nwalkers)
    pos[:,n_betpars+nu_components*2+4] = \
        np.random.uniform(logrtlow,logrthigh,nwalkers)
    pos[:,n_betpars+nu_components*2+5] = \
        np.random.uniform(dellow,delhigh,nwalkers)
    pos[:,n_betpars+nu_components*2+6] = \
        np.random.uniform(logMcenlow,logMcenhigh,nwalkers)
    pos[:,n_betpars+nu_components*2+7] = \
        np.random.uniform(acenlow,acenhigh,nwalkers)
    pos[:,n_betpars+nu_components*2+8] = \
        np.random.uniform(Arotlow,Arothigh,nwalkers)
    pos[:,n_betpars+nu_components*2+9] = \
        np.random.uniform(drangelow,drangehigh,nwalkers)
    pos[:,ndim-1] = \
        np.random.uniform(Mstar_min,Mstar_max,nwalkers)

    #Set up fitting function and priors: 
    if (propermotion == 'no'):
        x1 = rbin_phot
        x2 = rbin_kin
        y1 = surfden
        y1err = surfdenerr
        y2 = sigpmean
        y2err = sigperr
    elif (propermotion == 'yes'):
        x1 = rbin_phot
        x2 = rbin_kin
        x3 = rbin_kinp
        y1 = surfden
        y1err = surfdenerr
        y2 = sigpmean
        y2err = sigperr
        y3 = sigpmr
        y3err = sigpmrerr
        y4 = sigpmt
        y4err = sigpmterr
            
    lnprior = lambda theta: \
        lnprior_set(theta,n_betpars,bet0min,bet0max,\
                    betinfmin,betinfmax,\
                    betr0min,betr0max,betnmin,betnmax,\
                    nu_components,nupars_min,nupars_max,\
                    n_mpars,logM200low,logM200high,\
                    clow,chigh,logrclow,logrchigh,\
                    nlow,nhigh,logrtlow,logrthigh,\
                    dellow,delhigh,\
                    logMcenlow,logMcenhigh,\
                    acenlow,acenhigh,\
                    Arotlow,Arothigh,\
                    drangelow,drangehigh,\
                    Mstar_min,Mstar_max)            

    print('Running chains ... ')
    if __name__ == '__main__':
        with Pool(processes = nprocs) as pool:
            if (propermotion == 'no'):
                if (virialshape == 'no'):
                    sampler = \
                        emcee.EnsembleSampler(nwalkers,ndim,lnprob,\
                            args=(x1,x2,y1,y1err,y2,y2err),pool=pool)
                else:
                    sampler = \
                        emcee.EnsembleSampler(nwalkers,ndim,lnprob,\
                            args=(x1,x2,y1,y1err,y2,y2err,\
                                  vsp1val,vsp1pdf,vsp2val,vsp2pdf),pool=pool)
            elif (propermotion == 'yes'):
                if (virialshape == 'no'):
                    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,\
                            args=(x1,x2,x3,y1,y1err,y2,y2err,\
                                  y3,y3err,y4,y4err),pool=pool)
                else:
                    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,\
                            args=(x1,x2,x3,y1,y1err,y2,y2err,\
                                  y3,y3err,y4,y4err,\
                                  vsp1val,vsp1pdf,vsp2val,vsp2pdf),pool=pool)
            sampler.run_mcmc(pos, nmodels, progress = True)
    
        #Store the output (including the data):
        print('Writing data to file ... ')
        f = open(outdir+'output_sigp.txt','w')
        for i in range(len(rbin_kin)):
            f.write('%f %f %f\n' % \
                    (rbin_kin[i], sigpmean[i], sigperr[i]))
        f.close()
        f = open(outdir+'output_surfden.txt','w')
        for i in range(len(rbin_phot)):
            f.write('%f %f %f\n' % \
                    (rbin_phot[i], surfden[i], surfdenerr[i]))
        f.close()
        if (propermotion == 'yes'):
            f = open(outdir+'output_prop.txt','w')
            for i in range(len(rbin_kinp)):
                f.write('%f %f %f %f %f\n' % \
                       (rbin_kinp[i],\
                        sigpmr[i],sigpmrerr[i],sigpmt[i],\
                        sigpmterr[i]))
            f.close()
    
        burn = np.int(0.75*nmodels)
        chisq = -2.0 * \
                sampler.get_log_prob(discard=burn, flat=True)
        par_test = sampler.get_chain(discard=burn, flat=True)
    
        f = open(outdir+'Boutput_chain.txt','w')
        for i in range(len(chisq)):
            outstr = str(chisq[i]) + ' '
            for j in range(ndim):
                outstr = outstr + str(par_test[i,j]) + ' '
            outstr = outstr + '\n'
            f.write(outstr)
        f.close()

codemode = 'plot'
###########################################################
#Plotting code:
if (codemode == 'plot'):
    print('Running in plotting mode ... ')
    print('Loading data from:', outdir)

    #Read in the data:
    data_in = \
        np.genfromtxt(outdir+'output_surfden.txt',dtype='f8')
    rbin_phot = data_in[:,0]
    surfden = data_in[:,1]
    surfdenerr = data_in[:,2]
    data_in = \
        np.genfromtxt(outdir+'output_sigp.txt',dtype='f8')
    rbin_kin = data_in[:,0]
    sigpmean = data_in[:,1]
    sigperr = data_in[:,2]
    
    if (propermotion == 'yes'):
        data_in = \
            np.genfromtxt(outdir+'output_prop.txt',dtype='f8')
        rbin_kinp = data_in[:,0]
        sigpmr = data_in[:,1]
        sigpmrerr = data_in[:,2]
        sigpmt = data_in[:,3]
        sigpmterr = data_in[:,4]

    #Set radius array to use for plotting mass profiles
    #etc:
    if (rplot_inner > 0):
        rleft = rplot_inner
    else:   
        rleft = np.min(rbin_phot)
    if (rplot_outer > 0):
        rright = rplot_outer
    else:   
        rright = np.max(rbin_phot)
    if (rplot_pnts < 0):
        rplot_pnts = len(rbin_phot)
    print('Setting plot range:', rleft, rright)
    rbin = np.logspace(np.log10(rleft),np.log10(rright),\
                       np.int(rplot_pnts))

    #Load in the emcee data:
    data_in = \
        np.genfromtxt(outdir+'Boutput_chain.txt',dtype='f8')
    chisq = data_in[:,0]
    par_test = np.zeros((len(chisq),ndim), dtype='float')
    for i in range(1,ndim+1):
        par_test[:,i-1] = data_in[:,i]

    #Make sure no *really* bad models remain in the chains. 
    #In practice, this cut makes no difference to the end result.
    if (np.min(chisq) == np.inf):
        print('No viable models. Uh oh... bye bye. Minimum chisq:', np.min(chisq))
        sys.exit(0)
    index = np.where(chisq < np.min(chisq)*500.0)[0]
    print('Min/max chisq:', np.min(chisq[index]), np.max(chisq[index]))

    #Cut the confidence intervals from the chains:
    nsamples = 1000
    sample_choose = index[np.random.randint(len(index), \
                                            size=nsamples)]

    #Set up arrays to store confidence intervals:
    M_int = np.zeros((7,len(rbin)))
    rho_int = np.zeros((7,len(rbin)))
    dlnrhodlnr_int = np.zeros((7,len(rbin)))
    Mstar_int = np.zeros((7,len(Mstar_rad)))
    Mdynrat_int = np.zeros((7,len(Mstar_rad)))
    nu_int = np.zeros((7,len(Mstar_rad)))
    Mcen_int = np.zeros((7,len(rbin)))
    if (calc_Jfac == 'yes'):
        J_int = np.zeros(7)
    if (calc_Dfac == 'yes'):
        D_int = np.zeros(7)                
    Mstore = np.zeros((len(rbin),nsamples))
    rhostore = np.zeros((len(rbin),nsamples))
    dlnrhodlnrstore = np.zeros((len(rbin),nsamples))
    Mstarstore = np.zeros((len(Mstar_rad),nsamples))
    Mdynratstore = np.zeros((len(Mstar_rad),nsamples))
    nustore = np.zeros((len(Mstar_rad),nsamples))
    M200store = np.zeros(nsamples)
    vmaxstore = np.zeros(nsamples)
    cstore = np.zeros(nsamples)
    rcstore = np.zeros(nsamples)
    nstore = np.zeros(nsamples)
    rtstore = np.zeros(nsamples)
    delstore = np.zeros(nsamples)
    Mcenstore = np.zeros((len(rbin),nsamples))
    Arotstore = np.zeros(nsamples)
    dstore = np.zeros(nsamples)
    McenMstore = np.zeros(nsamples)
    Mcenastore = np.zeros(nsamples)
    if (calc_Jfac == 'yes'):
        Jstore = np.zeros(nsamples)    
    if (calc_Dfac == 'yes'):
        Dstore = np.zeros(nsamples)
    bet_int = np.zeros((7,len(rbin)))
    #########################################
    bet_DM_int = np.zeros((7,len(rbin)))
    #########################################
    betstar_int = np.zeros((7,len(rbin)))
    Sig_int = np.zeros((7,len(rbin)))
    sigp_int = np.zeros((7,len(rbin)))
    #########################################
    sigma_DM_int = np.zeros((7,len(rbin)))
    #########################################
    vphirot_int = np.zeros((7,len(rbin)))
    if (virialshape == 'yes'):
        vs1_int = np.zeros((7,1))
        vs2_int = np.zeros((7,1))
    if (propermotion == 'yes'):
        sigpmr_int = np.zeros((7,len(rbin)))
        sigpmt_int = np.zeros((7,len(rbin)))
    betstore = np.zeros((len(rbin),nsamples))
    betstarstore = np.zeros((len(rbin),nsamples))
    ############################################
    bet_DM_store = np.zeros((len(rbin),nsamples))
    ############################################
    Sigstore = np.zeros((len(rbin),nsamples))
    sigpstore = np.zeros((len(rbin),nsamples))
    ############################################
    sigma_DM_store = np.zeros((len(rbin),nsamples))
    ############################################
    vphirotstore = np.zeros((len(rbin),nsamples))
    if (virialshape == 'yes'):
        vs1store = np.zeros(nsamples)
        vs2store = np.zeros(nsamples)
    if (propermotion == 'yes'):
        sigpmrstore = np.zeros((len(rbin),nsamples))
        sigpmtstore = np.zeros((len(rbin),nsamples))

    for i in range(nsamples):
        print(i,'of',nsamples)
        theta = par_test[sample_choose[i],:]
        betpars = theta[0:n_betpars]
        #######################################
        betpars_DM = [0, np.random.uniform(0, 0.5617640216382228), np.random.uniform(0, 3.09814472257849), np.random.uniform(0.7280155282387563, 1.360105456581073)]
        #######################################
        nupars = theta[n_betpars:n_betpars+nu_components*2]
        Mpars = theta[n_betpars+nu_components*2:\
                      n_betpars+nu_components*2+n_mpars]
        Arot = theta[n_betpars+nu_components*2+n_mpars]
        drange = theta[n_betpars+nu_components*2+n_mpars+1]
        Mstar = theta[ndim-1]
        nuparsu = np.array(nupars)
        Mparsu = np.array(Mpars)
        Mparsu[0] = 10.**Mpars[0]
        Mparsu[2] = 10.**Mpars[2]
        Mparsu[4] = 10.**Mpars[4]
        Mparsu[6] = 10.**Mpars[6]

        #Calculate all profiles we want to plot:
        if (propermotion == 'no'):
            if (virialshape == 'no'):
                sigr2,Sig,sigLOS2 = \
                   sigp_fit(rbin,rbin,nuparsu,\
                             Mparsu,betpars,Mstar,Arot)
            else:
                sigr2,Sig,sigLOS2,vs1,vs2 = \
                    sigp_fit_vs(rbin,rbin,nuparsu,\
                                Mparsu,betpars,Mstar,Arot)
##########################################################
                sigr2_DM = sigma_DM_fit(rbin,Mparsu,\
                                Mparsu,betpars_DM,Mstar,Arot)
##########################################################
        elif (propermotion == 'yes'):
            if (virialshape == 'no'):
                sigr2,Sig,sigLOS2,sigpmr2,sigpmt2 = \
                    sigp_fit_prop(rbin,rbin,rbin,nuparsu,Mparsu,betpars,\
                                  Mstar,Arot)
            else:
                sigr2,Sig,sigLOS2,sigpmr2,sigpmt2,vs1,vs2 = \
                    sigp_fit_prop_vs(rbin,rbin,rbin,nuparsu,Mparsu,betpars,\
                                     Mstar,Arot)

        Mr = M(rbin,Mparsu)
        betar = beta(rbin,betpars)
        ##########################
        betar_DM = beta_DM(rbin, betpars_DM)
        ##########################
        rhor = rho(rbin,Mparsu)
        dlnrhodlnrr = dlnrhodlnr(rbin,Mparsu)
        Mstarr = Mstar_prof*Mstar
        nu_mass_r = multiplummass(Mstar_rad,nuparsu)
        Mcenr = Mcentral(rbin,Mparsu)
        
        Mstore[:,i] = Mr
        betstore[:,i] = betar
        #############################
        bet_DM_store[:,i] = betar_DM
        #############################
        betstarstore[:,i] = betar/(2.0-betar)
        sigpstore[:,i] = np.sqrt(sigLOS2)/1000.
        #############################
        sigma_DM_store[:,i] = np.sqrt(sigr2_DM)/1000.
        ############################# 
        Sigstore[:,i] = Sig
        vphirotstore[:,i] = np.sqrt(2.0*sigLOS2*Arot*rbin/Rhalf)/1000.
        rhostore[:,i] = rhor
        dlnrhodlnrstore[:,i] = dlnrhodlnrr
        Mstarstore[:,i] = Mstarr
        Mdynratstore[:,i] = M(Mstar_rad,Mparsu)/Mstarr
        nustore[:,i] = nu_mass_r
        Mcenstore[:,i] = Mcenr
        
        vmaxstore[i] = vmax_func(Mparsu[0],Mparsu[1],h)
        M200store[i] = Mparsu[0]
        cstore[i] = Mparsu[1]
        rcstore[i] = Mparsu[2]
        nstore[i] = Mparsu[3]
        rtstore[i] = Mparsu[4]
        delstore[i] = Mparsu[5]
        Arotstore[i] = Arot
        dstore[i] = drange*dgal_kpc
        McenMstore[i] = Mparsu[6]
        Mcenastore[i] = Mparsu[7]
        if (calc_Jfac == 'yes'):
            alpha_rmax = dgal_kpc*alpha_Jfac_deg/deg
            Jstore[i] = get_J(rho,Mparsu,dgal_kpc,alpha_rmax)
        if (calc_Dfac == 'yes'):
            alpha_rmax = dgal_kpc*alpha_Dfac_deg/deg
            Dstore[i] = get_D(rho,Mparsu,dgal_kpc,alpha_rmax)                                    
        if (virialshape == 'yes'):
            vs1store[i] = vs1/1.0e12
            vs2store[i] = vs2/1.0e12
        if (propermotion == 'yes'):
            sigpmrstore[:,i] = np.sqrt(sigpmr2)/1000.
            sigpmtstore[:,i] = np.sqrt(sigpmt2)/1000.

        #Solve for confidence intervals for each of these:
        for j in range(len(rbin)):
            M_int[0,j], M_int[1,j], M_int[2,j], M_int[3,j], \
                M_int[4,j], M_int[5,j], M_int[6,j] = \
                calcmedquartnine(Mstore[j,:])
            rho_int[0,j], rho_int[1,j], rho_int[2,j], rho_int[3,j], \
                rho_int[4,j], rho_int[5,j], rho_int[6,j] = \
                calcmedquartnine(rhostore[j,:])
            dlnrhodlnr_int[0,j], dlnrhodlnr_int[1,j],\
                dlnrhodlnr_int[2,j], \
                dlnrhodlnr_int[3,j], \
                dlnrhodlnr_int[4,j], \
                dlnrhodlnr_int[5,j], \
                dlnrhodlnr_int[6,j] = \
                calcmedquartnine(dlnrhodlnrstore[j,:])
        for j in range(len(Mstar_rad)):
            Mstar_int[0,j], Mstar_int[1,j], Mstar_int[2,j], \
                Mstar_int[3,j], \
                Mstar_int[4,j], \
                Mstar_int[5,j], \
                Mstar_int[6,j] = \
                calcmedquartnine(Mstarstore[j,:])
        for j in range(len(Mstar_rad)):
            Mdynrat_int[0,j], Mdynrat_int[1,j], \
                Mdynrat_int[2,j], \
                Mdynrat_int[3,j], \
                Mdynrat_int[4,j], \
                Mdynrat_int[5,j], \
                Mdynrat_int[6,j] = \
                calcmedquartnine(Mdynratstore[j,:])
        for j in range(len(Mstar_rad)):
            nu_int[0,j], nu_int[1,j], nu_int[2,j], \
                nu_int[3,j], \
                nu_int[4,j], \
                nu_int[5,j], \
                nu_int[6,j] = \
                calcmedquartnine(nustore[j,:])
        for j in range(len(rbin)):
            Mcen_int[0,j], Mcen_int[1,j], Mcen_int[2,j], \
                Mcen_int[3,j], \
                Mcen_int[4,j], \
                Mcen_int[5,j], \
                Mcen_int[6,j] = \
                calcmedquartnine(Mcenstore[j,:])
        if (calc_Jfac == 'yes'):
            J_int = \
                calcmedquartnine(Jstore[:])
        if (calc_Dfac == 'yes'):
            D_int = \
                calcmedquartnine(Dstore[:])
        for j in range(len(rbin)):
            bet_int[0,j], bet_int[1,j], bet_int[2,j], \
                bet_int[3,j], \
                bet_int[4,j], \
                bet_int[5,j], \
                bet_int[6,j] = \
                calcmedquartnine(betstore[j,:])
####################################################
        for j in range(len(rbin)):
            bet_DM_int[0,j], bet_DM_int[1,j], bet_DM_int[2,j], \
                bet_DM_int[3,j], \
                bet_DM_int[4,j], \
                bet_DM_int[5,j], \
                bet_DM_int[6,j] = \
                calcmedquartnine(bet_DM_store[j,:])
####################################################
            betstar_int[0,j], betstar_int[1,j], \
                betstar_int[2,j], betstar_int[3,j], \
                betstar_int[4,j], \
                betstar_int[5,j], \
                betstar_int[6,j] = \
                calcmedquartnine(betstarstore[j,:])
            sigp_int[0,j], sigp_int[1,j], sigp_int[2,j], \
                sigp_int[3,j], \
                sigp_int[4,j], \
                sigp_int[5,j], \
                sigp_int[6,j] = \
                calcmedquartnine(sigpstore[j,:])
######################################################
            sigma_DM_int[0,j], sigma_DM_int[1,j], sigma_DM_int[2,j], \
                sigma_DM_int[3,j], \
                sigma_DM_int[4,j], \
                sigma_DM_int[5,j], \
                sigma_DM_int[6,j] = \
                calcmedquartnine(sigma_DM_store[j,:])
######################################################
            Sig_int[0,j], Sig_int[1,j], Sig_int[2,j], \
                Sig_int[3,j], \
                Sig_int[4,j], \
                Sig_int[5,j], \
                Sig_int[6,j] = \
                calcmedquartnine(Sigstore[j,:])
            vphirot_int[0,j], vphirot_int[1,j], vphirot_int[2,j], \
                vphirot_int[3,j], \
                vphirot_int[4,j], \
                vphirot_int[5,j], \
                vphirot_int[6,j] = \
                calcmedquartnine(vphirotstore[j,:])
            if (propermotion == 'yes'):
                sigpmr_int[0,j], sigpmr_int[1,j], \
                    sigpmr_int[2,j], sigpmr_int[3,j], \
                    sigpmr_int[4,j], \
                    sigpmr_int[5,j], \
                    sigpmr_int[6,j] = \
                    calcmedquartnine(sigpmrstore[j,:])
                sigpmt_int[0,j], sigpmt_int[1,j], \
                    sigpmt_int[2,j], sigpmt_int[3,j], \
                    sigpmt_int[4,j], \
                    sigpmt_int[5,j], \
                    sigpmt_int[6,j] = \
                    calcmedquartnine(sigpmtstore[j,:])
        if (virialshape == 'yes'):
            vs1_int[0], vs1_int[1], \
                vs1_int[2], vs1_int[3], \
                vs1_int[4], \
                vs1_int[5], \
                vs1_int[6] = \
                calcmedquartnine(vs1store[:])
            vs2_int[0], vs2_int[1], \
                vs2_int[2], vs2_int[3], \
                vs2_int[4], \
                vs2_int[5], \
                vs2_int[6] = \
                calcmedquartnine(vs2store[:])

    #######################################################
    #And now make the plots:

    #First calculate median distance; plot all data at
    #median:
    dmed, dsixlow, dsixhi,\
        dninelow, dninehi, \
        dnineninelow, dnineninehi = calcmedquartnine(dstore)
    dcorr = dmed/dgal_kpc
    
    ##### Stellar surface density ##### 
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.loglog()

    plt.errorbar(rbin_phot*dcorr,surfden,surfdenerr,\
                 color='b',ecolor='b',linewidth=2,alpha=0.75,\
                 fmt='o')
    plt.fill_between(rbin,Sig_int[5,:],Sig_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(rbin,Sig_int[3,:],Sig_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(rbin,Sig_int[1,:],Sig_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    plt.plot(rbin,Sig_int[0,:],'k',linewidth=mylinewidth,\
             label=r'Fit')

    plt.axvline(x=Rhalf,color='blue',alpha=0.5,\
                linewidth=mylinewidth)

    plt.xlabel(r'$R\,[{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$\Sigma_*\,[N\,{\rm kpc}^{-2}]$',\
                   fontsize=myfontsize)
    plt.xlim([np.min(rbin),np.max(rbin)])
    plt.ylim([ymin_Sigstar,ymax_Sigstar])
    plt.savefig(outdir+'output_Sigstar.pdf',bbox_inches='tight')'''

    f = open(outdir+'graph_surfden_Data.txt', 'w')
    for i in range(len(rbin_phot)):
        outstr = str(rbin_phot[i]) + ' ' + str(surfden[i]) + ' ' + str(surfdenerr[i]) + '\n'
        f.write(outstr)
    f.close()

    f = open(outdir+'dcorr.txt','w')
    f.write(str(dcorr))
    f.close()

    f = open(outdir+'graph_surfden_fit.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],Sig_int[0,i],Sig_int[1,i],\
                 Sig_int[2,i],Sig_int[3,i],\
                 Sig_int[4,i],Sig_int[5,i],\
                 Sig_int[6,i]))
    f.close()

    f = open(outdir+'Rhalf.txt','w')
    f.write(str(Rhalf))
    f.close()
 

    ##### Stellar projected velocity dispersion #####  
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    
    plt.errorbar(np.log10(rbin_kin*dcorr),sigpmean,sigperr,\
                 linewidth=2,color='b',alpha=0.75,\
                 fmt='o')

    sel = sigp_int[0,:] > 0
    plt.fill_between(np.log10(rbin[sel]),sigp_int[5,:][sel],\
                     sigp_int[6,:][sel],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin[sel]),sigp_int[3,:][sel],\
                     sigp_int[4,:][sel],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin[sel]),sigp_int[1,:][sel],\
                     sigp_int[2,:][sel],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    plt.plot(np.log10(rbin[sel]),sigp_int[0,:][sel],'k',linewidth=mylinewidth,\
             label=r'Fit')
    plt.axvline(x=np.log10(Rhalf),color='blue',alpha=0.5,\
                linewidth=mylinewidth)
                
    plt.xlabel(r'${\rm Log}_{10}[R/{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$\sigma_{\rm LOS}[{\rm km\,s}^{-1}]$',\
                   fontsize=myfontsize)

    plt.ylim([0,y_sigLOSmax])
    plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])
    
    plt.savefig(outdir+'output_sigLOS.pdf',bbox_inches='tight')'''
    
    sel = sigp_int[0,:]>0
    f = open(outdir+'sel.txt','w')
    for i in range(len(sel)):
        outstr = str(sel[i]) + '\n'
        f.write(outstr)
    f.close()

    f = open(outdir+'graph_vLOS_Data.txt','w')
    for i in range(len(rbin_kin)):
        outstr = str(rbin_kin[i]) + ' ' + str(sigpmean[i]) + ' ' + str(sigperr[i]) + '\n'
        f.write(outstr)
    f.close()

    f = open(outdir+'graph_vLOS_fit.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],sigp_int[0,i],sigp_int[1,i],\
                 sigp_int[2,i],sigp_int[3,i],\
                 sigp_int[4,i],sigp_int[5,i],\
                 sigp_int[6,i]))
    f.close()
##############################################################################
    ##### DM radial velocity dispersion #####
    f = open(outdir + 'output_sigma_DM.txt', 'w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],sigma_DM_int[0,i],sigma_DM_int[1,i],\
                 sigma_DM_int[2,i],sigma_DM_int[3,i],\
                 sigma_DM_int[4,i],sigma_DM_int[5,i],\
                 sigma_DM_int[6,i]))
    f.close()
##############################################################################

    ##### Stellar proper motion dispersions ##### 
    if (propermotion == 'yes'):
        #First in the radial direction (on the sky):
        fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(mylinewidth)
        plt.xticks(fontsize=myfontsize)
        plt.yticks(fontsize=myfontsize)

        psel = sigpmr > 0
        plt.errorbar(np.log10(rbin_kinp[psel]*dcorr),sigpmr[psel]*dcorr,sigpmrerr[psel]*dcorr,\
                     linewidth=2,color='b',alpha=0.75,\
                     fmt='o')
            
        sel = sigpmr_int[0,:] > 0
        plt.fill_between(np.log10(rbin[sel]),sigpmr_int[5,:][sel],\
                         sigpmr_int[6,:][sel],\
                         facecolor='black',alpha=alp3sig,\
                         edgecolor='none')
        plt.fill_between(np.log10(rbin[sel]),sigpmr_int[3,:][sel],\
                         sigpmr_int[4,:][sel],\
                         facecolor='black',alpha=0.33,\
                         edgecolor='none')
        plt.fill_between(np.log10(rbin[sel]),sigpmr_int[1,:][sel],\
                         sigpmr_int[2,:][sel],\
                         facecolor='black',alpha=0.66,\
                         edgecolor='none')
        plt.plot(np.log10(rbin[sel]),sigpmr_int[0,:][sel],'k',linewidth=mylinewidth,\
                 label=r'Fit')
        plt.axvline(x=np.log10(Rhalf),color='blue',alpha=0.5,\
                    linewidth=mylinewidth)

        plt.xlabel(r'${\rm Log}_{10}[R/{\rm kpc}]$',\
                       fontsize=myfontsize)
        plt.ylabel(r'$\sigma_{\rm pmr}[{\rm km\,s}^{-1}]$',\
                       fontsize=myfontsize)

        plt.ylim([0,y_sigLOSmax])
        plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])
        
        plt.savefig(outdir+'output_sigpmr.pdf',bbox_inches='tight')

        #Then in the tangential direction (on the sky): 
        fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(mylinewidth)
        plt.xticks(fontsize=myfontsize)
        plt.yticks(fontsize=myfontsize)
 
        psel = sigpmt > 0
        plt.errorbar(np.log10(rbin_kinp[psel]*dcorr),sigpmt[psel]*dcorr,sigpmterr[psel]*dcorr,\
                     linewidth=2,color='b',alpha=0.75,\
                     fmt='o')
            
        sel = sigpmt_int[0,:] > 0
        plt.fill_between(np.log10(rbin[sel]),sigpmt_int[5,:][sel],\
                         sigpmt_int[6,:][sel],\
                         facecolor='black',alpha=alp3sig,\
                         edgecolor='none')
        plt.fill_between(np.log10(rbin[sel]),sigpmt_int[3,:][sel],\
                         sigpmt_int[4,:][sel],\
                         facecolor='black',alpha=0.33,\
                         edgecolor='none')
        plt.fill_between(np.log10(rbin[sel]),sigpmt_int[1,:][sel],\
                         sigpmt_int[2,:][sel],\
                         facecolor='black',alpha=0.66,\
                         edgecolor='none')
        plt.plot(np.log10(rbin[sel]),sigpmt_int[0,:][sel],'k',linewidth=mylinewidth,\
                 label=r'Fit')
        plt.axvline(x=np.log10(Rhalf),color='blue',alpha=0.5,\
                    linewidth=mylinewidth)

        plt.xlabel(r'${\rm Log}_{10}[R/{\rm kpc}]$',\
                       fontsize=myfontsize)
        plt.ylabel(r'$\sigma_{\rm pmt}[{\rm km\,s}^{-1}]$',\
                       fontsize=myfontsize)
                       
        plt.ylim([0,y_sigLOSmax])
        plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])

        plt.savefig(outdir+'output_sigpmt.pdf',bbox_inches='tight')

    ##### Stellar beta(r) anisotropy profile ##### 
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.fill_between(np.log10(rbin),bet_int[5,:],bet_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),bet_int[3,:],bet_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),bet_int[1,:],bet_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    plt.plot(np.log10(rbin),bet_int[0,:],'k',linewidth=mylinewidth,\
             label=r'Fit')'''

    #And true answer (for mock data):
    '''if (overtrue == 'yes'):
        plt.plot(np.log10(ranal),betatrue,'b--',linewidth=mylinewidth,\
                 label=r'True')

    plt.xlabel(r'${\rm Log}_{10}[r/{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$\beta$',\
                   fontsize=myfontsize)
    plt.ylim([np.min([bet0min,betinfmin]),np.max([bet0max,betinfmax])])
    plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])

    plt.savefig(outdir+'output_beta.pdf',bbox_inches='tight')'''

    #Write the above data to files for comparitive plotting later:
    f = open(outdir+'output_bet.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],bet_int[0,i],bet_int[1,i],bet_int[2,i],bet_int[3,i],\
                 bet_int[4,i],bet_int[5,i],bet_int[6,i]))
    f.close()

###################################################################################
    ##### DM beta(r) prior profile #####
    f = open(outdir + 'output_beta_DM.txt', 'w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],bet_DM_int[0,i],bet_DM_int[1,i],bet_DM_int[2,i],bet_DM_int[3,i],\
                 bet_DM_int[4,i],bet_DM_int[5,i],bet_DM_int[6,i]))
    f.close()
###################################################################################

    ###### Stellar symmetrised betastar(r) profile ##### 
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.fill_between(np.log10(rbin),betstar_int[5,:],betstar_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),betstar_int[3,:],betstar_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),betstar_int[1,:],betstar_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    plt.plot(np.log10(rbin),betstar_int[0,:],'k',linewidth=mylinewidth,\
             label=r'Fit')'''
 
    #And true answer (for mock data):
    '''if (overtrue == 'yes'):
         plt.plot(np.log10(ranal),betatruestar,'b--',linewidth=mylinewidth,\
                 label=r'True')
 
    plt.xlabel(r'${\rm Log}_{10}[r/{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$\tilde{\beta}$',\
                   fontsize=myfontsize)
    plt.ylim([-1,1])
    plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])
 
    plt.savefig(outdir+'output_betastar.pdf',bbox_inches='tight')'''

    #Write the above data to files for comparitive plotting later:
    f = open(outdir+'output_betstar.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],betstar_int[0,i],betstar_int[1,i],\
                 betstar_int[2,i],betstar_int[3,i],\
                 betstar_int[4,i],betstar_int[5,i],\
                 betstar_int[6,i]))
    f.close()

    ##### Cumulative mass profiles ##### 
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    plt.loglog()

    plt.fill_between(rbin,M_int[5,:],M_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(rbin,M_int[3,:],M_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(rbin,M_int[1,:],M_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    if (Mstar > 1.0):
        plt.plot(rbin,M_int[0,:],'k',linewidth=mylinewidth,\
                     label=r'Fit Dark Matter')
    else:
        plt.plot(rbin,M_int[0,:],'k',linewidth=mylinewidth,\
                     label=r'Fit')

    plt.fill_between(Mstar_rad,Mstar_int[5,:],Mstar_int[6,:],\
                     facecolor=colorpop2,alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(Mstar_rad,Mstar_int[3,:],Mstar_int[4,:],\
                     facecolor=colorpop2,alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(Mstar_rad,Mstar_int[1,:],Mstar_int[2,:],\
                     facecolor=colorpop2,alpha=0.66,\
                     edgecolor='none')

    plt.fill_between(rbin,Mcen_int[5,:],Mcen_int[6,:],\
                     facecolor=colorpop3,alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(rbin,Mcen_int[3,:],Mcen_int[4,:],\
                     facecolor=colorpop3,alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(rbin,Mcen_int[1,:],Mcen_int[2,:],\
                     facecolor=colorpop3,alpha=0.66,\
                     edgecolor='none')            
    
    if (Mstar > 1.0):
        plt.plot(Mstar_rad,Mstar_int[0,:],color=colorpop2,\
                 linewidth=mylinewidth,\
                 label=r'Fit Stars')
    if (np.max(Mcen_int) > 0.0):
        plt.plot(rbin,Mcen_int[0,:],color=colorpop3,\
                 linewidth=mylinewidth,\
                 label=r'Fit Central Dark Mass')'''

        
    #Overplot true answer (for mock data):
    '''if (overtrue == 'yes'):
        plt.plot(ranal,truemass,'b--',linewidth=mylinewidth,\
                 label=r'True')        

    plt.axvline(x=Rhalf,color='blue',alpha=0.5,\
                linewidth=mylinewidth)

    plt.xlabel(r'$r\,[{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$M(<r)\,[{\rm M}_\odot]$',\
                   fontsize=myfontsize)

    plt.ylim([yMlow,yMhigh])
    plt.xlim([np.min(rbin),np.max(rbin)])

    plt.legend(loc='upper left',fontsize=mylegendfontsize)
    plt.savefig(outdir+'output_Mass.pdf',bbox_inches='tight')'''

    #Write the above data to files for comparitive plotting later:
    f = open(outdir+'output_M.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
         (rbin[i],M_int[0,i],M_int[1,i],M_int[2,i],M_int[3,i],\
              M_int[4,i], M_int[5,i], M_int[6,i]))
    f.close()

    #And the Mdyn/Mstar ratio:
    f = open(outdir+'output_MdynMstar.txt','w')
    for i in range(len(Mstar_rad)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
         (Mstar_rad[i],Mdynrat_int[0,i],Mdynrat_int[1,i],Mdynrat_int[2,i],\
              Mdynrat_int[3,i],\
              Mdynrat_int[4,i], Mdynrat_int[5,i], Mdynrat_int[6,i]))
    f.close()

    #And nu_mass_r:
    f = open(outdir+'output_nu_mass_r.txt','w')
    for i in range(len(Mstar_rad)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
         (Mstar_rad[i],nu_int[0,i],nu_int[1,i],nu_int[2,i],\
              nu_int[3,i],\
              nu_int[4,i], nu_int[5,i], nu_int[6,i]))
    f.close()

    #And Mstar:
    f = open(outdir+'output_Mass_Mstar.txt','w')
    for i in range(len(Mstar_rad)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (Mstar_rad[i],Mstar_int[0,i],Mstar_int[1,i],Mstar_int[2,i],\
                 Mstar_int[3,i],\
                 Mstar_int[4,i], Mstar_int[5,i], Mstar_int[6,i]))
    f.close()

    #And central dark mass:
    f = open(outdir+'output_Mass_Mcen.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
                (rbin[i],Mcen_int[0,i],Mcen_int[1,i],Mcen_int[2,i],\
                 Mcen_int[3,i],\
                 Mcen_int[4,i], Mcen_int[5,i], Mcen_int[6,i]))
    f.close()
    
    ##### Dark matter density profile ##### 
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    plt.loglog()

    plt.fill_between(rbin,rho_int[5,:],rho_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(rbin,rho_int[3,:],rho_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(rbin,rho_int[1,:],rho_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    plt.plot(rbin,rho_int[0,:],'k',linewidth=mylinewidth,\
             label=r'Fit')'''

    #Overplot true solution (for mock data): 
    '''if (overtrue == 'yes'):
        plt.plot(ranal,trueden,'b--',linewidth=mylinewidth,\
                 label=r'True')

    plt.axvline(x=Rhalf,color='blue',alpha=0.5,\
                linewidth=mylinewidth)

    plt.xlabel(r'$r\,[{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$\rho\,[{\rm M}_\odot\,{\rm kpc}^{-3}]$',\
                   fontsize=myfontsize)


    plt.xlim([np.min(rbin),np.max(rbin)])
    plt.ylim([yrholow,yrhohigh])

    plt.savefig(outdir+'output_rho.pdf',bbox_inches='tight')'''

    #Write the above data to files for comparitive plotting later:
    f = open(outdir+'output_rho.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
         (rbin[i],rho_int[0,i],rho_int[1,i],rho_int[2,i],rho_int[3,i],\
              rho_int[4,i],rho_int[5,i],rho_int[6,i]))
    f.close()

    #And the coreNFW parameters:
    f = open(outdir+'output_M200c200_chain.txt','w')
    for i in range(len(M200store)):
        f.write('%f %f %f %f %f %f %f\n' % \
                (M200store[i],cstore[i],nstore[i],rcstore[i],rtstore[i],\
                 delstore[i],vmaxstore[i]))
    f.close()

    ##### Dark matter log density exponent #####
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.fill_between(np.log10(rbin),dlnrhodlnr_int[5,:],dlnrhodlnr_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),dlnrhodlnr_int[3,:],dlnrhodlnr_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),dlnrhodlnr_int[1,:],dlnrhodlnr_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
    plt.plot(np.log10(rbin),dlnrhodlnr_int[0,:],'k',linewidth=mylinewidth,\
             label=r'Fit')'''

    #And overplot true model (for mock data):
    '''if (overtrue == 'yes'):
        plt.plot(np.log10(ranal),truedlnrhodlnr,'b--',linewidth=mylinewidth,\
                 label=r'True')
 
    plt.axvline(x=np.log10(Rhalf),color='blue',alpha=0.5,\
                linewidth=mylinewidth)
    
    plt.xlabel(r'${\rm Log}_{10}[r/{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'${\rm dln}\rho/{\rm dln}r$',\
                   fontsize=myfontsize)


    plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])
    plt.ylim([-4,0])

    plt.savefig(outdir+'output_dlnrhodlnr.pdf',bbox_inches='tight')'''

    #Write the above data to files for comparitive plotting later:
    f = open(outdir+'output_dlnrhodlnr.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f\n' % \
         (rbin[i],dlnrhodlnr_int[0,i],dlnrhodlnr_int[1,i],\
              dlnrhodlnr_int[2,i],dlnrhodlnr_int[3,i],\
              dlnrhodlnr_int[4,i],dlnrhodlnr_int[5,i],\
              dlnrhodlnr_int[6,i]))
    f.close()

    ##### Rotation velocity profile #####
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.fill_between(np.log10(rbin),vphirot_int[5,:],vphirot_int[6,:],\
                     facecolor='black',alpha=alp3sig,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),vphirot_int[3,:],vphirot_int[4,:],\
                     facecolor='black',alpha=0.33,\
                     edgecolor='none')
    plt.fill_between(np.log10(rbin),vphirot_int[1,:],vphirot_int[2,:],\
                     facecolor='black',alpha=0.66,\
                     edgecolor='none')
            
    plt.axvline(x=np.log10(Rhalf),color='blue',alpha=0.5,\
                                linewidth=mylinewidth)

    plt.xlabel(r'${\rm Log}_{10}[R/{\rm kpc}]$',\
               fontsize=myfontsize)
    plt.ylabel(r'$v_\phi[{\rm km}\,{\rm s}^{-1}]$',\
               fontsize=myfontsize)
    

    plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])
    plt.ylim([0,y_sigLOSmax])
    
    plt.savefig(outdir+'output_vphirot.pdf',bbox_inches='tight')'''
    
    #######################################################
    #Additional write output
    #And write the D+J-factor data:
    if (calc_Jfac == 'yes'):
        f = open(outdir+'output_Jfac.txt','w')
        for i in range(len(Jstore)):
            f.write('%f\n' % Jstore[i])
        f.close()
    if (calc_Dfac == 'yes'):
        f = open(outdir+'output_Dfac.txt','w')
        for i in range(len(Dstore)):
            f.write('%f\n' % Dstore[i])
        f.close()                                   

    #And write the distance data:
    f = open(outdir+'output_dstore.txt','w')
    for i in range(len(dstore)):
        f.write('%f\n' % dstore[i])
    f.close()

    #######################################################
    ##### Histograms #####
    
    ##### Distance #####
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    nbin = 25

    n, bins, patches = plt.hist(dstore,bins=nbin,\
                                range=(np.min(dstore),\
                                       np.max(dstore)),\
                                facecolor='b', \
                                histtype='bar',alpha=0.5, \
                                label='distance')
    
    plt.xlabel(r'$d\,[{\rm kpc}]$',\
               fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    plt.ylim([0,np.max(n)])
    plt.savefig(outdir+'output_d.pdf',bbox_inches='tight')'''
    
    ##### J-factor #####
    if (calc_Jfac == 'yes'):
        fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(mylinewidth)
        plt.xticks(fontsize=myfontsize)
        plt.yticks(fontsize=myfontsize)
        nbin = 25

        n, bins, patches = plt.hist(Jstore,bins=nbin,\
                                    range=(np.min(Jstore),\
                                           np.max(Jstore)),\
                                    facecolor='b', \
                                    histtype='bar',alpha=0.5, \
                                    label='J')
        
        plt.xlabel(r'$J\,[{\rm GeV}^2\,{\rm c}^{-4}\,{\rm cm}^{-5}]$',\
                   fontsize=myfontsize)
        plt.ylabel(r'$N$',fontsize=myfontsize)
        plt.ylim([0,np.max(n)])
        plt.savefig(outdir+'output_Jfac.pdf',bbox_inches='tight')

    ##### D-factor #####
    if (calc_Dfac == 'yes'):
        fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(mylinewidth)
        plt.xticks(fontsize=myfontsize)
        plt.yticks(fontsize=myfontsize)
        nbin = 25

        n, bins, patches = plt.hist(Dstore,bins=nbin,\
                                    range=(np.min(Dstore),\
                                           np.max(Dstore)),\
                                    facecolor='b', \
                                    histtype='bar',alpha=0.5, \
                                    label='D')
        
        plt.xlabel(r'$D\,[{\rm GeV}\,{\rm c}^{-2}\,{\rm cm}^{-2}]$',\
                   fontsize=myfontsize)
        plt.ylabel(r'$N$',fontsize=myfontsize)
        plt.ylim([0,np.max(n)])
        plt.savefig(outdir+'output_Dfac.pdf',bbox_inches='tight')

    ##### Virial shape parameters #####
    if (virialshape == 'yes'):
        #And make a plot of the Virial shape parameters, if 
        #activated:
        '''fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(mylinewidth)
        plt.xticks(fontsize=myfontsize)
        plt.yticks(fontsize=myfontsize)
        nbin = 25

        n, bins, patches = plt.hist(vs1store,bins=nbin,\
                range=(np.min(vs1store),\
                       np.max(vs1store)),\
                facecolor='b', \
                histtype='bar',alpha=0.5, \
                label='vs_1')
        plt.errorbar([vs1bin],[0.5*np.max(n)],\
                     xerr=[[vs1bin-vs1lo],[vs1hi-vs1bin]],fmt='ob')
 
        plt.xlabel(r'$v_{s1}\,[{\rm km}^4\,{\rm s}^{-4}]$',\
                   fontsize=myfontsize)
        plt.ylabel(r'$N$',fontsize=myfontsize)
        plt.ylim([0,np.max(n)])
        plt.savefig(outdir+'output_vs1.pdf',bbox_inches='tight')'''

        '''fig = plt.figure(figsize=(figx,figy))
        ax = fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(mylinewidth)
        plt.xticks(fontsize=myfontsize)
        plt.yticks(fontsize=myfontsize)
        nbin = 25

        n, bins, patches = plt.hist(vs2store,bins=nbin,\
                range=(np.min(vs2store),\
                       np.max(vs2store)),\
                facecolor='r', \
                       histtype='bar',alpha=0.5)
        plt.errorbar(vs2bin,[0.5*np.max(n)],\
                     xerr=[[vs2bin-vs2lo],[vs2hi-vs2bin]],fmt='or')

        plt.xlabel(r'$v_{s2}\,[{\rm km}^4\,{\rm s}^{-4}\,{\rm kpc}^2]$',\
                   fontsize=myfontsize)
        plt.ylabel(r'$N$',fontsize=myfontsize)
        plt.ylim([0,np.max(n)])
        plt.savefig(outdir+'output_vs2.pdf',bbox_inches='tight')'''

    ##### coreNFWtides model parameters #####
    nbin = 15
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    
    n, bins, patches = plt.hist(np.log10(M200store),bins=nbin,\
                                range=(logM200low,logM200high),\
                                facecolor='b', \
                                histtype='bar',alpha=0.5)

    plt.xlabel(r'${\rm Log}_{10}[M_{200}/{\rm M}_\odot]$',\
               fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    
    plt.xlim([logM200low,logM200high])
    plt.ylim([0,np.max(n)])
    
    plt.savefig(outdir+'output_M200.pdf',bbox_inches='tight')'''

    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    vmaxlow = 5.0
    vmaxhigh = 50.0
    n, bins, patches = plt.hist(vmaxstore,bins=nbin,\
                                range=(vmaxlow,vmaxhigh),\
                                facecolor='b', \
                                histtype='bar',alpha=0.5)

    plt.xlabel(r'$v_{\rm max}\,[{\rm km/s}]$',\
               fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)

    plt.xlim([vmaxlow,vmaxhigh])
    plt.ylim([0,np.max(n)])
    
    plt.savefig(outdir+'output_vmax.pdf',bbox_inches='tight')'''
    
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    
    n, bins, patches = plt.hist(cstore,bins=nbin,\
                                range=(clow,chigh),\
                                facecolor='b', \
                                histtype='bar',alpha=0.5)

    plt.xlabel(r'$c$',fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)

    plt.xlim([clow,chigh])
    plt.ylim([0,np.max(n)])

    plt.savefig(outdir+'output_c.pdf',bbox_inches='tight')'''
  
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    ax.xaxis.set_ticks(np.arange(logrclow,logrchigh+1.0,1.0))
    
    n, bins, patches = plt.hist(np.log10(rcstore),bins=nbin,\
                                range=(logrclow,logrchigh),\
                                facecolor='k', \
                                histtype='bar',alpha=0.5)
        
    plt.xlabel(r'${\rm Log}_{10}[r_c/{\rm kpc}]$',fontsize=myfontsize)
    plt.xlim([logrclow,logrchigh])

    plt.ylabel(r'$N$',fontsize=myfontsize) 
    plt.ylim([0,np.max(n)])
    plt.savefig(outdir+'output_rc.pdf',bbox_inches='tight')'''

    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    n, bins, patches = plt.hist(np.log10(rtstore),bins=nbin,\
                                range=(logrtlow,\
                                       logrthigh),\
                                facecolor='k', \
                                histtype='bar',alpha=0.5)

    plt.xlabel(r'${\rm Log}_{10}[r_t/{\rm kpc}]$',fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    plt.savefig(outdir+'output_rt.pdf',bbox_inches='tight')'''
        
    sigmstore = np.zeros(len(rcstore))
    for i in range(len(rcstore)):
        sigmstore[i] = sidm_novel(rcstore[i],M200store[i],cstore[i],\
                                  oden,rhocrit)
                                  
    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    n, bins, patches = plt.hist(sigmstore,bins=nbin,\
                  range=(sigmlow,sigmhigh),\
                  facecolor='k', \
                  histtype='bar',alpha=0.5)

    plt.ylim([0.0,np.max(n)])
    plt.xlabel(r'$\sigma/m\,({\rm cm}^2/{\rm g})$',\
        fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    plt.savefig(outdir+'output_sigm.pdf',bbox_inches='tight')'''

    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)
    
    n, bins, patches = plt.hist(nstore,bins=nbin,\
                                range=(nlow,nhigh),\
                                facecolor='b', \
                                histtype='bar',alpha=0.5)

    plt.xlabel(r'$n$',\
               fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    
    plt.xlim([nlow,nhigh])
    plt.ylim([0,np.max(n)])
    plt.savefig(outdir+'output_n.pdf',bbox_inches='tight')'''

    '''fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    tick_spacing = 0.01
    ax.minorticks_on()
    ax.tick_params('both', length=20, width=2, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    n, bins, patches = plt.hist(delstore,bins=nbin,\
                                range=(dellow,delhigh),\
                                facecolor='b', \
                                histtype='bar',alpha=0.5)

    plt.xlabel(r'$\delta$',\
               fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)

    plt.xlim([dellow,delhigh])
    plt.ylim([0,np.max(n)])
    plt.savefig(outdir+'output_del.pdf',bbox_inches='tight')'''

    #Calculate M200 +/- 68%:
    M200med, M200sixlow, M200sixhi,\
        M200ninelow, M200ninehi, \
        M200nineninelow, M200nineninehi = calcmedquartnine(M200store)
    print('*******************************')
    print('M200 -/+ 68% :: ', M200med, M200sixlow, M200sixhi)
    f = open(outdir+'output_M200vals.txt','w')
    f.write('%f %f %f %f %f %f %f\n' % \
            (M200med, M200sixlow, M200sixhi,\
             M200ninelow, M200ninehi, \
             M200nineninelow, M200nineninehi))
    f.close()                                        
    
    #And same for vmax:
    vmaxmed, vmaxsixlow, vmaxsixhi,\
        vmaxninelow, vmaxninehi, \
        vmaxnineninelow, vmaxnineninehi = calcmedquartnine(vmaxstore)
    print('*******************************')
    print('vmax -/+ 68% :: ', vmaxmed, vmaxsixlow, vmaxsixhi)           
    f = open(outdir+'output_vmaxvals.txt','w')
    f.write('%f %f %f %f %f %f %f\n' % \
            (vmaxmed, vmaxsixlow, vmaxsixhi,\
             vmaxninelow, vmaxninehi, \
             vmaxnineninelow, vmaxnineninehi))
    f.close()
    
    #And the same for rt: 
    rtmed, rtsixlow, rtsixhi,\
        rtninelow, rtninehi, \
        rtnineninelow, rtnineninehi = calcmedquartnine(rtstore)
    print('*******************************')
    print('rt -/+ 68% :: ', rtmed, rtsixlow, rtsixhi)

    #And the same for d (already calculated, above):
    print('*******************************')
    print('d -/+ 68% :: ', dmed, dsixlow, dsixhi)

    #And the same for Mcen:
    Mcenmed, Mcensixlow, Mcensixhi,\
        Mcenninelow, Mcenninehi, \
        Mcennineninelow, Mcennineninehi = calcmedquartnine(McenMstore)
    print('*******************************')
    print('Mcen -/+ 68% :: ', Mcenmed, Mcensixlow, Mcensixhi)

    #And the same for acen:
    acenmed, acensixlow, acensixhi,\
        acenninelow, acenninehi, \
        acennineninelow, acennineninehi = calcmedquartnine(Mcenastore)
    print('*******************************')
    print('acen -/+ 68% :: ', acenmed, acensixlow, acensixhi)

    #And the same for J-factor:
    if (calc_Jfac == 'yes'):
        Jmed, Jsixlow, Jsixhi,\
            Jninelow, Jninehi, \
            Jnineninelow, Jnineninehi = calcmedquartnine(Jstore)
        print('*******************************')
        print('J -/+ 68% :: ', Jmed, Jsixlow, Jsixhi)


###########################################################
#Exit:
print('\nThank you for using GravSphere! Have a nice day.\n')
