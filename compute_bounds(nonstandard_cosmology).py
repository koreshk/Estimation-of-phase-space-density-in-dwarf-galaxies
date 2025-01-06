import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.integrate import simps as integrator
from scipy.optimize import root, bisect
from scipy.special import erf as Erf, erfi as Erfi

from gravsphere_initialise_CVnI import *
from figures import *

from tqdm import tqdm

whichgal = 'Z_126-111'
bound_radius = 1e-3 * 95
cosmology = 'LRT'

outdir = './gravsphere-master/results from cluster/' + whichgal + '/' # + '/CosmoC/'

T_0 = 2.7255 #K
k = 8.62e-5 # eV/K
T_eff =(4/11)**(1/3) * T_0 *k*1e-9 #GeV
M_pl = 1.2e19 #GeV
G = 1/M_pl**2
g = 2
rho_DM = 1e-45 #GeV^4
rho_crit = 4e-47

if (cosmology == 'K'):
    beta = 1
    N = 3.74e-6
    cosm_const = 2.55505
    
if (cosmology == 'LRT'):
    beta = 3
    N = 3.6e-7
    cosm_const = 5.6822 

gamma = beta/3
eps_max = gamma + W(gamma*np.exp(-gamma))
eps_max =  eps_max.real

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

infile = 'mass-r.txt'
m_phase_arr = np.genfromtxt(outdir + infile, dtype = 'f8')

bound_index = np.where(rbin>=bound_radius)[0][0]
print(rbin[bound_index])
nsamples = 1000

###################################################
#___Параметры без погрешности___
C = (1/(2*np.pi))**(3/2)
A = 11.16 / ( 2*np.pi )**3 * 1e-9
np.int = int
m_arr = np.zeros((len(rbin),int(nsamples)))
for i in range(nsamples):
    m_Q = m_phase_arr[i,:]
    m_arr[:,i] = (m_Q**3 * 11.16e-3 / 2 / (2*np.pi)**3 / N / (eps_max - gamma) / eps_max**(gamma-1) )**(1/(5-gamma))

def calcmedquartnine(array):
    index = np.argsort(array,axis=0)
    median = array[index[np.int(len(array)/2.)+1]]
    sixlowi = np.int(16./100. * len(array))
    sixhighi = np.int(84./100. * len(array))
    ninelowi = np.int(2.5/100. * len(array))
    ninehighi = np.int(97.5/100. * len(array))
    nineninelowi = np.int(0.15/100. * len(array))
    nineninehighi = np.int(99.85/100. * len(array))

    sixhigh = array[index[sixhighi]]
    sixlow = array[index[sixlowi]]
    ninehigh = array[index[ninehighi]]
    ninelow = array[index[ninelowi]]
    nineninehigh = array[index[nineninehighi]]
    nineninelow = array[index[nineninelowi]]

    return median, sixlow, sixhigh, ninelow, ninehigh,\
        nineninelow, nineninehigh
mass_int = np.zeros((7,len(rbin)))
for j in range(len(rbin)):
    mass_int[0,j], mass_int[1,j], mass_int[2,j], \
                mass_int[3,j], \
                mass_int[4,j], \
                mass_int[5,j], \
                mass_int[6,j] = \
                calcmedquartnine(m_arr[j,:])
print(mass_int[0, bound_index])

'''
outfile = 'mass-r-cosmolog.txt'
f = open(outdir + outfile, 'w')
for i in range(nsamples):
    f.write(''.join(map(str, m_arr[:,i]))+'\n')
f.close()

    
fig = plt.figure(figsize=(figx,figy))
ax = fig.add_subplot(111)
plt.plot(rbin, mass_int[0,:], color= 'black')
plt.fill_between(rbin, mass_int[1,:], mass_int[2,:], facecolor = 'black', alpha = 0.66)
plt.fill_between(rbin, mass_int[3,:], mass_int[4,:], facecolor = 'black', alpha = 0.33)
minerr_ind = np.where(rbin >= bound_radius)[0][0]
plt.axvline(rbin[minerr_ind])
#plt.xlim(1e-2,1)
#plt.ylim(1e-1, 1e2)
plt.loglog()
plt.xlabel(r'$r\,[{\rm kpc}]$',\
               fontsize=myfontsize)
plt.ylabel(r'$m\,[{\rm keV}]$',\
               fontsize=myfontsize)

plt.savefig(outdir + 'mass-r.pdf')
'''
def calconesidebound(array):
    index = np.argsort(array,axis=0)
    CL = np.int(5./100. * len(array))

    bound = array[index[CL]]

    return bound
print('one-side-bound: ', calconesidebound(m_arr[bound_index,:]))

'''
m_new_lo = (m_Q_lo**3 * 11.16e-3 / 2 / (2*np.pi)**3 / N / (eps_max - gamma) / eps_max**(gamma-1) )**(1/(5-gamma))
m_new_hi = (m_Q_hi**3 * 11.16e-3 / 2 / (2*np.pi)**3 / N / (eps_max - gamma) / eps_max**(gamma-1) )**(1/(5-gamma))
m_new_lolo = (m_Q_lolo**3 * 11.16e-3 / 2 / (2*np.pi)**3 / N / (eps_max - gamma) / eps_max**(gamma-1) )**(1/(5-gamma))
m_new_hihi = (m_Q_hihi**3 * 11.16e-3 / 2 / (2*np.pi)**3 / N / (eps_max - gamma) / eps_max**(gamma-1) )**(1/(5-gamma))


outfile = 'masses_Q(LRT_cosmology).txt'
f = open(outdir + outfile, 'w')
f.write('%f %f %f %f %f\n' % (m_new, m_new_lo, m_new_hi, m_new_lolo, m_new_hihi))
f.close()
'''



Q = rho / vel**3 * 8e-33
Qlo = rholo / velhi**3 * 8e-33
Qhi = rhohi / vello**3 * 8e-33
Qlolo = rhololo / velhihi**3 * 8e-33
Qhihi = rhohihi / vellolo**3 * 8e-33

#___Параметры без погрешности___
C = (1/(2*np.pi))**(3/2)
A = 11.16 / ( 2*np.pi )**3 * 1e-9

minerr_ind = np.where(rbin >= bound_radius)[0][0]

ind_int = minerr_ind
rbin = rbin * 1.6e35
X = rbin[ind_int:]

Norm_const = N / (1e-6)**(1-gamma)


def estimate_mass_EMF(Q, m_initial):
    #m_max = lambda m: (Q[0]/Q[-1])**(1/4)*m
    #print(m_max(m_new))
    p_max = np.log( Q/Q[-1])
    e_p_max = Q/Q[-1]
    
    x_max = np.where(Q/Q[-1] >= 1)
    Integral_1 = 4*np.pi*X[x_max]**2 * np.pi**(3/2)* 2 * np.sqrt(2) *(1-beta[x_max]) * vel[x_max]**3 * ( Erf(np.sqrt(p_max[x_max])) - np.sqrt(1/beta[x_max]-1)*Erfi(np.sqrt(beta[x_max])/np.sqrt(1-beta[x_max])*np.sqrt(p_max[x_max])) * e_p_max[x_max]**(-1/(1-beta[x_max])) )
    Integral_2 = 4*np.pi*X[x_max]**2 * 4*np.pi/3 * 2**(3/2) * (1-beta[x_max]) * vel[x_max]**3 * p_max[x_max]**(3/2)
    
    f = lambda m: C*Q[-1]/m**4
    if (gamma > 0): eps_lim_1 = lambda m: bisect(lambda x: Norm_const * x**gamma / (np.exp(x) + 1) * m**(1-gamma) - f(m), 0, eps_max)
    else: eps_lim_1 = 0  
    eps_lim_2 = lambda m: bisect(lambda x: Norm_const * x**gamma / (np.exp(x) + 1) * m**(1-gamma) - f(m), eps_max, 100*eps_max)
    
    def D_maxwell(m):
        
        Int = Q[x_max]*C/m * Integral_1
        V = C*Q[-1]/m * Integral_2
        return integrator(Int-V) + (Int[0]-V[0])*X[0]
        
    N = lambda m: ( integrator(Q*C/m * 4*np.pi*X**2 * np.pi**(3/2) * 2 * np.sqrt(2) *(1-beta) * vel**3) + Q[0]*C/m * 4*np.pi*X[0]**3 * np.pi**(3/2) * 2 * np.sqrt(2) *(1-beta[0]) * vel[0]**3)/(4*np.pi * T_eff**3 * (Norm_const * m**(1-gamma) * cosm_const))
    
    
    
    #print('eps_lim:', eps_lim_1(m_new), eps_lim_2(m_new))
    D_fermi = lambda m: 4*np.pi*N(m) * T_eff**3 * (Norm_const * m**(1-gamma) * integrate.quad(lambda p: p**(2+gamma) * 1/(np.exp(p)+1), eps_lim_1(m), eps_lim_2(m) )[0] - 1/3 * C*Q[-1]/m**4 * (eps_lim_2(m)**3 - eps_lim_1(m)**3))
    #print(D_fermi(m_new), D_maxwell(m_new))
    mass_EMF = bisect(lambda m: D_fermi(m) - D_maxwell(m), m_initial, 10*m_initial)
    #print(mass_EMF)
    return mass_EMF


infile = 'beta_dist.txt'
beta_dist = np.genfromtxt(outdir + infile, dtype = 'f8')

infile = 'rho_dist.txt'
rho_dist = np.genfromtxt(outdir + infile, dtype = 'f8')

infile = 'sigma_dist.txt'
sigma_dist = np.genfromtxt(outdir + infile, dtype = 'f8')


mass_EMF = np.zeros(int(nsamples))

for i in tqdm(range(nsamples)):
    vel = (sigma_dist[i,:])[sel]
    rho = (rho_dist[i,:])[sel]
    beta = (beta_dist[i,:])[sel]
    
    Q = rho/(1-beta)/vel**3 * 8e-33    
    
    vel = vel[ind_int:] /3e5 
    beta = beta[ind_int:]
    Q_a = Q[ind_int:]
    
    mass_EMF[i] = estimate_mass_EMF(Q_a, m_arr[bound_index,i]*1e-6)*1e6

print('mean: ', calcmedquartnine(mass_EMF)[0])
print('one-side-bound: ', calconesidebound(mass_EMF))

'''
vel = velhi[ind_int:] /3e5 
beta = betalo[ind_int:]
Q_a = Qlo[ind_int:]/(1-beta)

mass_EMFlo = estimate_mass_EMF(Q_a, m_new_lo*1e-6)

vel = vello[ind_int:] /3e5 
beta = betahi[ind_int:]
Q_a = Qhi[ind_int:]/(1-beta)

mass_EMFhi = estimate_mass_EMF(Q_a, m_new_hi*1e-6)

vel = velhihi[ind_int:] /3e5 
beta = betalolo[ind_int:]
Q_a = Qlolo[ind_int:]/(1-beta)

mass_EMFlolo = estimate_mass_EMF(Q_a, m_new_lolo*1e-6)

vel = vellolo[ind_int:] /3e5
beta = betahihi[ind_int:]
Q_a = Qhihi[ind_int:]/(1-beta)

mass_EMFhihi = estimate_mass_EMF(Q_a, m_new_hihi*1e-6)

outfile = 'masses_EMF(LRT_cosmology).txt'
f = open(outdir + outfile, 'w')
#print('EMF bound', mass_EMF*1e6, '+/- 68%', mass_EMFhi*1e6,'/', mass_EMFlo*1e6, '+/- 95%', mass_EMFhihi, '/', mass_EMFlolo)
f.write('%f %f %f %f %f\n' % (mass_EMF*1e6, mass_EMFlo*1e6, mass_EMFhi*1e6, mass_EMFlolo*1e6, mass_EMFhihi*1e6))
f.close()

print('EMF mass estimated.')
'''
