import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate
from scipy.integrate import simps as integrator
from numpy import pi
from scipy.optimize import root, bisect
import pandas as pd

whichgal = 'Sculptor_Dwarf_Galaxy'
outdir = './gravsphere-master/results from cluster/' + whichgal + '/CosmoC/'

T_0 = 2.7255 #температура реликтовых фотонов (K)
k = 8.62e-5 # eV/K
T_eff =(4/11)**(1/3) * T_0 *k*1e-9 #GeV
M_pl = 1.2e19 #GeV
G = 1/M_pl**2
g = 2
rho_DM = 1e-45 #GeV^4
rho_crit = 4e-47

infile = 'output_M200vals.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
M_DM = data[0] * 1e57

infile = 'graph_vLOS_fit.txt'
#infile = 'sigma_DM.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
rbin = data[:,0]
sigp_int = np.transpose(data[:,1:8])

sel = sigp_int[0,:] > 0

rbin = rbin[sel]
vel = sigp_int[0,:][sel]
vello = sigp_int[1,:][sel]
velhi = sigp_int[2,:][sel]
vellolo = sigp_int[3,:][sel]
velhihi = sigp_int[4,:][sel]

infile = 'output_rho.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
#rbin = data[:,0][sel]
rho = data[:,1][sel]
rholo = data[:,2][sel]
rhohi = data[:,3][sel]
rhololo = data[:,4][sel]
rhohihi = data[:,5][sel]

infile = 'output_bet.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
data = data[:,1:8]
beta = data[:,0][sel]
betalo = data[:,1][sel]
betahi = data[:,2][sel]
betalolo = data[:,3][sel]
betahihi = data[:,4][sel]

'''
infile = 'reconstructed_beta_DM.txt'
data = np.genfromtxt(outdir+infile, dtype = 'f8')
X = data[:,0]
bet_int = np.transpose(data[:,1:8])
beta = np.interp(rbin,X,bet_int[0,:], len(rbin))
betalo = np.interp(rbin,X,bet_int[1,:], len(rbin))
betahi = np.interp(rbin,X,bet_int[2,:], len(rbin))
betalolo = np.interp(rbin,X,bet_int[3,:], len(rbin))
betahihi = np.interp(rbin,X,bet_int[4,:], len(rbin))
'''

Q = rho / vel**3 * 8e-33
Qlo = rholo / velhi**3 * 8e-33
Qhi = rhohi / vello**3 * 8e-33
Qlolo = rhololo / velhihi**3 * 8e-33
Qhihi = rhohihi / vellolo**3 * 8e-33

#___Параметры без погрешности___
C = (1/(2*math.pi))**(3/2)
A = g * 11.3 / ( 2*math.pi )**3 * 1e-9

#Ограничение из фазовой плотности#
m_phase = (2*C/A * Q/(1-beta))**(1/3) * 1e6
m_phasehi = (2*C/A * Qhi/(1-betahi))**(1/3) * 1e6
m_phaselo = (2*C/A * Qlo/(1-betalo))**(1/3) * 1e6
m_phaselolo = (2*C/A * Qlolo/(1-betalolo))**(1/3) * 1e6
m_phasehihi = (2*C/A * Qhihi/(1-betahihi))**(1/3) * 1e6

minerr_ind = np.where((m_phasehi - m_phaselo)/m_phase == min((m_phasehi - m_phaselo)/m_phase))[0][0]

plt.plot(rbin, m_phase, color= 'black')
plt.fill_between(rbin, m_phaselo, m_phasehi, facecolor = 'black', alpha = 0.66)
plt.fill_between(rbin, m_phaselolo, m_phasehihi, facecolor = 'black', alpha = 0.33)

plt.axvline(rbin[minerr_ind], label = 'min error')

plt.loglog()
plt.legend()

outfile = 'masses.txt'
f = open(outdir + outfile, 'w')
f.write('%f %f %f %f %f\n' % (m_phase[minerr_ind], m_phaselo[minerr_ind], m_phasehi[minerr_ind], m_phaselolo[minerr_ind], m_phasehihi[minerr_ind]))
f.write('%f %f %f %f %f\n' % (m_phase[0], m_phaselo[0], m_phasehi[0], m_phaselolo[0], m_phasehihi[0]))
f.close()
#print('min error mass:', m_phase[minerr_ind]*1e6 , '+/- 68%', m_phasehi[minerr_ind]*1e6, '/', m_phaselo[minerr_ind]*1e6, '+/- 95%', m_phasehihi[minerr_ind]*1e6, '/', m_phaselolo[minerr_ind]*1e6)

plt.savefig(outdir+'mass_phi.pdf',bbox_inches='tight')
print('Q mass estimated.')

rbin = rbin * 1.6e35
X = rbin #np.linspace(rbin[0],rbin[-1],100)

def estimate_mass_EMF(Q, m_initial):
    Q_int = Q #np.interp(X,rbin,Q)
    ind_int = minerr_ind #max(np.where(Q_int >= Q[minerr_ind])[0])
    #Q_int[np.where(Q_int[:ind_int] < Q_int[ind_int])[0]] = Q_int[ind_int]
    Q = Q_int[:ind_int+1]/(1-beta[:ind_int+1])
    
    F_maxwell_r = lambda m: Q[:ind_int] * C / m**4
    F_maxwell_p_r = lambda p_r, m: np.exp(-1/2 / m**2 * p_r**2 / vel[:ind_int]**2)
    F_maxwell_p_t = lambda p_t, m: np.exp(-1/2 / m**2 * p_t**2/ (1-beta[:ind_int]) /vel[:ind_int]**2 )
    
    p_max_r2 = lambda m : -2 * m**2 * vel[:ind_int]**2 * np.log(Q[ind_int]/ Q[:ind_int])
    p_max_t2 = lambda m, p_r: (1-beta[:ind_int])*(p_max_r2(m) - p_r**2)
    
    Int_maxwell_p_t = lambda m, p_r: np.array([integrate.quad(lambda p_t: p_t*F_maxwell_p_t(p_t, m)[i], 0, np.sqrt(p_max_t2(m, p_r)[i]))[0] for i in range(ind_int)])
    Int_maxwell_p_t_p_r = lambda m: np.array([  integrate.quad(lambda p_r: F_maxwell_p_r(p_r, m)[i] * Int_maxwell_p_t(m,p_r)[i], 0, np.sqrt(p_max_r2(m)[i]) )[0] for i in range(ind_int)])
    Int_maxwell_rp =  lambda m: integrator(X[:ind_int]**2 * F_maxwell_r(m) * Int_maxwell_p_t_p_r(m), X[:ind_int])
    
    V_p_t = lambda m, p_r: np.array([integrate.quad(lambda p_t: p_t, 0, np.sqrt(p_max_t2(m, p_r)[i]))[0] for i in range(ind_int)])
    V_p_t_p_r = lambda m: np.array([  integrate.quad(lambda p_r: V_p_t(m,p_r)[i], 0, np.sqrt(p_max_r2(m)[i]) )[0] for i in range(ind_int)])
    V_rp = lambda m: integrator(X[:ind_int]**2 * V_p_t_p_r(m), X[:ind_int])
    
    D_maxwell = lambda m: (2*np.pi)*(4*np.pi) * (Int_maxwell_rp(m) - (m/m_initial)**3 * V_rp(m))
    D_fermi = lambda m: 4*np.pi * M_DM / rho_DM * T_eff**3 * (A / m * integrate.quad(lambda p: p**2 * 1/(np.exp(p)+1), 0, np.log(2*(m/m_initial)**3 - 1))[0] - 1/3 * (m/m_initial)**3 * np.log(2*(m/m_initial)**3 - 1)**3)
    mass_EMF = root(lambda m: D_fermi(m) - D_maxwell(m), m_initial).x[0]
    #mass_EMF = root(lambda m: D_fermi(m) - D_maxwell(m), m_initial).x[0]
    return mass_EMF*1e6

vel = np.interp(X,rbin,vel) /3e5 
beta = np.interp(X,rbin,beta)
mass_EMF = estimate_mass_EMF(Q, m_phase[minerr_ind]*1e-6)
print('...')

vel = np.interp(X,rbin,velhi) /3e5 
beta = np.interp(X,rbin,betalo)
mass_EMFlo = estimate_mass_EMF(Qlo, m_phaselo[minerr_ind]*1e-6)
print('...')

vel = np.interp(X,rbin,vello) /3e5 
beta = np.interp(X,rbin,betahi)
mass_EMFhi = estimate_mass_EMF(Qhi, m_phasehi[minerr_ind]*1e-6)
print('...')

vel = np.interp(X,rbin,velhihi) /3e5 
beta = np.interp(X,rbin,betalolo)
mass_EMFlolo = estimate_mass_EMF(Qlolo, m_phaselolo[minerr_ind]*1e-6)
print('...')

vel = np.interp(X,rbin,vellolo) /3e5 
beta = np.interp(X,rbin,betahihi)
mass_EMFhihi = estimate_mass_EMF(Qhihi, m_phasehihi[minerr_ind]*1e-6)
print('...')

outfile = 'masses.txt'
f = open(outdir + outfile, 'w')
#print('EMF bound', mass_EMF*1e6, '+/- 68%', mass_EMFhi*1e6,'/', mass_EMFlo*1e6, '+/- 95%', mass_EMFhihi, '/', mass_EMFlolo)
f.write('%f %f %f %f %f\n' % (mass_EMF, mass_EMFlo, mass_EMFhi, mass_EMFlolo, mass_EMFhihi))
f.close()

print('EMF mass estimated.')