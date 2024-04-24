import warnings
warnings.simplefilter("ignore")

alp3sig = 0.1

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

from scipy.integrate import quad 

from functions import *
from constants import *
from binulator_surffuncs import * 
from binulator_velfuncs import * 
from figures import * 
import sys

#MW satellites:
#from gravsphere_initialise_Draco import *
#from gravsphere_initialise_UMi import *
#from gravsphere_initialise_Carina import *
#from gravsphere_initialise_LeoI import *
#from gravsphere_initialise_LeoII import *
#from gravsphere_initialise_Sextans import *
#from gravsphere_initialise_Sculptor import *
#from gravsphere_initialise_Fornax import *
from gravsphere_initialise_CVnI import *
#from gravsphere_initialise_SegI import *
#from gravsphere_initialise_SMC import *
#from gravsphere_initialise_Ocen import *

whichgal = 'Sculptor_Dwarf_Galaxy'

outdir = './results from cluster/'+whichgal+'/CosmoC/'

infile = 'output_bet.txt'
data = np.genfromtxt(outdir + infile, dtype = 'f8')
rbin = data[:,0]
bet_int = np.transpose(data[:,1:8])

infile = 'output_M.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
rbin = data[:,0]
M_int = np.transpose(data[:,1:8])

infile = 'output_rho.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
rbin = data[:,0]
rho_int = np.transpose(data[:,1:8])

from scipy.optimize import curve_fit as fit

M_tot = M_int[0,:]
M_tot_lo = M_int[1,:]
M_tot_hi = M_int[2,:]
M_tot_lolo = M_int[3,:]
M_tot_hihi = M_int[4,:]

rho = rho_int[0,:]
rho_lo = rho_int[1,:]
rho_hi = rho_int[2,:]
rho_lolo = rho_int[3,:]
rho_hihi = rho_int[4,:]

approximation_beta = lambda x, x_0, beta_0, beta_inf, n: beta_0 + ( beta_inf - beta_0 ) / ( 1 + (x_0/x)**n )
g = lambda x: x**(2*beta_0) * ( (x_0/x)**(n) + 1 )**(2 / n * (beta_inf - beta_0))

beta = bet_int[0,:]

(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , rbin, beta, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr = g(rbin)
beta_lo = bet_int[1,:]
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , rbin, beta_lo, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_lo = g(rbin)


beta_hi = bet_int[2,:]
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , rbin, beta_hi, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_hi = g(rbin)

beta_lolo = bet_int[3,:]
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , rbin, beta_lolo, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_lolo = g(rbin)

beta_hihi = bet_int[4,:]
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , rbin, beta_hihi, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_hihi = g(rbin)

C = 4.3e-6 * np.array([integrator( M_tot[:i]/rbin[:i]**2 * g_arr[:i] * rho[:i] , rbin[:i]) for i in range(1,len(rbin)+1)])
C_lo = 4.3e-6 * np.array([integrator( M_tot_lo[:i]/rbin[:i]**2 * rho[:i] * g_arr[:i] , rbin[:i]) for i in range(1,len(rbin)+1)])
C_hi = 4.3e-6 * np.array([integrator( M_tot_hi[:i]/rbin[:i]**2 * rho[:i] * g_arr[:i] , rbin[:i]) for i in range(1,len(rbin)+1)])
C_lolo = 4.3e-6 * np.array([integrator( M_tot_lolo[:i]/rbin[:i]**2 * rho[:i] * g_arr[:i] , rbin[:i]) for i in range(1,len(rbin)+1)])
C_hihi = 4.3e-6 * np.array([integrator( M_tot_hihi[:i]/rbin[:i]**2 * rho[:i] * g_arr[:i] , rbin[:i]) for i in range(1,len(rbin)+1)])

sigma_DM = (C[-1]-C)/ g_arr / rho
sigma_DM_lo = (C_lo[-1]-C_lo)/ g_arr_hi / rho_hi
sigma_DM_hi = (C_hi[-1]-C_hi)/ g_arr_lo / rho_lo
sigma_DM_lolo = (C_lolo[-1]-C_lolo)/ g_arr_hihi / rho_hihi
sigma_DM_hihi = (C_hihi[-1]-C_hihi)/ g_arr_lolo / rho_lolo

fig = plt.figure(figsize=(figx,figy))
ax = plt.gca()

ax.set_xscale('log')
plt.plot(rbin, np.sqrt(sigma_DM) , color = 'black')
plt.fill_between(rbin, np.sqrt(sigma_DM_lo), np.sqrt(sigma_DM_hi),facecolor = 'black', alpha = 0.66)
plt.fill_between(rbin, np.sqrt(sigma_DM_lolo), np.sqrt(sigma_DM_hihi),facecolor = 'black', alpha = 0.33)


plt.savefig(outdir + 'sigma_DM.pdf')

f = open(outdir+'sigma_DM.txt','w')
for i in range(len(rbin)):
    f.write('%f %f %f %f %f %f\n' % (rbin[i],np.sqrt(sigma_DM)[i],np.sqrt(sigma_DM_lo)[i],np.sqrt(sigma_DM_hi)[i],np.sqrt(sigma_DM_lolo)[i],np.sqrt(sigma_DM_hihi)[i]))
f.close()