import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.integrate import simps as integrator
from scipy.optimize import root, bisect
from scipy.special import erf as Erf, erfi as Erfi

from gravsphere_initialise_CVnI import *
from figures import *

whichgal = 'Z_126-111'
bound_radius = 0.065513

outdir = './gravsphere-master/results from cluster/' + whichgal + '/' # + '/CosmoC/'

T_0 = 2.7255 #K
k = 8.62e-5 # eV/K
T_eff =(4/11)**(1/3) * T_0 *k*1e-9 #GeV
M_pl = 1.2e19 #GeV
G = 1/M_pl**2
g = 2
rho_DM = 1e-45 #GeV^4
rho_crit = 4e-47

infile = 'output_sigma_DM.txt'
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

infile = 'output_beta_DM.txt'
data = np.genfromtxt(outdir + infile, dtype = 'f8')
X = data[:,0]
beta = np.interp(rbin, X, data[:,1])
betalo = np.interp(rbin, X, data[:,2])
betahi = np.interp(rbin, X, data[:,3])
betalolo = np.interp(rbin, X, data[:,4])
betahihi = np.interp(rbin, X, data[:,5])

Q = rho / vel**3 * 8e-33
Qlo = rholo / velhi**3 * 8e-33
Qhi = rhohi / vello**3 * 8e-33
Qlolo = rhololo / velhihi**3 * 8e-33
Qhihi = rhohihi / vellolo**3 * 8e-33

#___Параметры без погрешности___
C = (1/(2*np.pi))**(3/2)
A = 11.16 / ( 2*np.pi )**3 * 1e-9

#Ограничение из фазовой плотности#
m_phase = (2*C/A * Q/(1-beta))**(1/3) * 1e6
m_phasehi = (2*C/A * Qhi/(1-betahi))**(1/3) * 1e6
m_phaselo = (2*C/A * Qlo/(1-betalo))**(1/3) * 1e6
m_phaselolo = (2*C/A * Qlolo/(1-betalolo))**(1/3) * 1e6
m_phasehihi = (2*C/A * Qhihi/(1-betahihi))**(1/3) * 1e6

minerr_ind = np.where(rbin >= bound_radius)[0][0] #np.where((m_phasehi - m_phaselo)/m_phase == min((m_phasehi - m_phaselo)/m_phase))[0][0]
print('bound radius:', rbin[minerr_ind])

fig = plt.figure(figsize=(figx*1.2,figy))
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(mylinewidth)
plt.xticks(fontsize=myfontsize)
plt.yticks(fontsize=myfontsize)

plt.plot(rbin, m_phase, color= 'black')
plt.fill_between(rbin, m_phaselo, m_phasehi, facecolor = 'black', alpha = 0.66)
plt.fill_between(rbin, m_phaselolo, m_phasehihi, facecolor = 'black', alpha = 0.33)

plt.axvline(rbin[minerr_ind], label = 'bound radius')
#plt.xlim(1e-2,1)
#plt.ylim(1e-1, 1e2)
plt.loglog()
plt.legend()
plt.xlabel(r'$r_{{\rm bound}}\,[{\rm kpc}]$',\
               fontsize=myfontsize)
plt.ylabel(r'$m^{(Q)}\,[{\rm keV}]$',\
               fontsize=myfontsize)

plt.savefig(outdir + 'mass_Q.pdf')

outfile = 'masses_Q.txt'
f = open(outdir + outfile, 'w')
f.write('%f %f %f %f %f\n' % (m_phase[minerr_ind], m_phaselo[minerr_ind], m_phasehi[minerr_ind], m_phaselolo[minerr_ind], m_phasehihi[minerr_ind]))
#f.write('%f %f %f %f %f\n' % (m_phase[0], m_phaselo[0], m_phasehi[0], m_phaselolo[0], m_phasehihi[0]))
f.close()
#print('min error mass:', m_phase[minerr_ind]*1e6 , '+/- 68%', m_phasehi[minerr_ind]*1e6, '/', m_phaselo[minerr_ind]*1e6, '+/- 95%', m_phasehihi[minerr_ind]*1e6, '/', m_phaselolo[minerr_ind]*1e6)

print('Q mass estimated.')

ind_int = minerr_ind
rbin = rbin * 1.6e35
X = rbin[ind_int:]
plt.cla()

def estimate_mass_EMF(Q, m_initial):
    m_max = lambda m: (Q[0]/Q[-1])**(1/4)*m
    p_max = lambda m: np.log( Q/Q[0] * (m_max(m)/m)**4 )
    e_p_max = lambda m: Q/Q[0] * (m_max(m)/m)**4
    
    def D_maxwell(m):
        x_max = np.where(Q/Q[0] * (m_max(m)/m)**4 >= 1)
        Int = Q[x_max]*C/m**4 * 4*np.pi*X[x_max]**2 * np.pi**(3/2) * np.sqrt(2) * m**3*(1-beta[x_max]) * vel[x_max]**3 * ( Erf(np.sqrt(p_max(m)[x_max])) - np.sqrt(1/beta[x_max]-1)*Erfi(np.sqrt(beta[x_max])/np.sqrt(1-beta[x_max])*np.sqrt(p_max(m)[x_max])) * e_p_max(m)[x_max]**(-1/(1-beta[x_max])) )
        V = C*Q[0]/m_max(m)**4 * 4*np.pi*X[x_max]**2 * np.pi/3 * 2**(3/2) * m**3 * (1-beta[x_max]) * vel[x_max]**3 * p_max(m)[x_max]**(3/2)
        return integrator(Int-V) + (Int[0]-V[0])*X[0]
        
    N = lambda m: ( integrator(Q*C/m**4 * 4*np.pi*X**2 * np.pi**(3/2) * np.sqrt(2) * m**3*(1-beta) * vel**3) + Q[0]*C/m**4 * 4*np.pi*X[0]**3 * np.pi**(3/2) * np.sqrt(2) * m**3*(1-beta[0]) * vel[0]**3)/(4*np.pi * T_eff**3 * (A / m * 1.80309))
    D_fermi = lambda m: 4*np.pi*N(m) * T_eff**3 * (A / m * integrate.quad(lambda p: p**2 * 1/(np.exp(p)+1), 0, np.log( 2 * m_max(m)**4/m/m_initial**3 - 1 ))[0] - 1/3 * C*Q[0]/m_max(m)**4 * np.log( 2 * m_max(m)**4/m/m_initial**3 - 1 )**3)
    mass_EMF = bisect(lambda m: D_fermi(m) - D_maxwell(m), m_initial, 10*m_initial)
    
    return mass_EMF



vel = vel[ind_int:] /3e5 
beta = beta[ind_int:]
Q_a = Q[ind_int:]/(1-beta)

mass_EMF = estimate_mass_EMF(Q_a, m_phase[minerr_ind]*1e-6)

vel = velhi[ind_int:] /3e5 
beta = betalo[ind_int:]
Q_a = Qlo[ind_int:]/(1-beta)

mass_EMFlo = estimate_mass_EMF(Q_a, m_phaselo[minerr_ind]*1e-6)

vel = vello[ind_int:] /3e5 
beta = betahi[ind_int:]
Q_a = Qhi[ind_int:]/(1-beta)

mass_EMFhi = estimate_mass_EMF(Q_a, m_phasehi[minerr_ind]*1e-6)

vel = velhihi[ind_int:] /3e5 
beta = betalolo[ind_int:]
Q_a = Qlolo[ind_int:]/(1-beta)

mass_EMFlolo = estimate_mass_EMF(Q_a, m_phaselolo[minerr_ind]*1e-6)

vel = vellolo[ind_int:] /3e5
beta = betahihi[ind_int:]
Q_a = Qhihi[ind_int:]/(1-beta)

mass_EMFhihi = estimate_mass_EMF(Q_a, m_phasehihi[minerr_ind]*1e-6)

outfile = 'masses_EMF.txt'
f = open(outdir + outfile, 'w')
#print('EMF bound', mass_EMF*1e6, '+/- 68%', mass_EMFhi*1e6,'/', mass_EMFlo*1e6, '+/- 95%', mass_EMFhihi, '/', mass_EMFlolo)
f.write('%f %f %f %f %f\n' % (mass_EMF*1e6, mass_EMFlo*1e6, mass_EMFhi*1e6, mass_EMFlolo*1e6, mass_EMFhihi*1e6))
f.close()

print('EMF mass estimated.')
