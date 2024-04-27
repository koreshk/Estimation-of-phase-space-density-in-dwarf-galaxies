##### Dark velocity dispersion #####
from scipy.optimize import curve_fit as fit
import numpy as np
from scipy.integrate import simps as integrator
import matplotlib.pyplot as plt

infile = './My_Data/read_prior.txt'
data = np.genfromtxt(infile, dtype = 'f8')
X = data[:,0]
bet_int = data[:,1]
#print(len(X), len(bet_int), len(rbin))
#bet_int = np.interp(rbin, X, beta)

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

beta = bet_int/2

(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , X, beta, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr = g(rbin)

beta_lo = bet_int/4
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , X, beta_lo, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_lo = g(rbin)


beta_hi = bet_int*3/4
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , X, beta_hi, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_hi = g(rbin)

beta_lolo = 0
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , X, beta_lolo, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_lolo = 1

beta_hihi = bet_int
(x_0, beta_0, beta_inf, n), _ = fit( approximation_beta , X, beta_hihi, p0 = [(rbin[-1] + rbin[0]) / 2, 0, 1, 2] )
g_arr_hihi = g(rbin)

C = 4.3e-6 * np.array([integrator( M_tot[i:]/rbin[i:]**2 * g_arr[i:] * rho[i:] , rbin[i:]) for i in range(len(rbin))])
C_lo = 4.3e-6 * np.array([integrator( M_tot_lo[i:]/rbin[i:]**2 * rho[i:] * g_arr[i:] , rbin[i:]) for i in range(len(rbin))])
C_hi = 4.3e-6 * np.array([integrator( M_tot_hi[i:]/rbin[i:]**2 * rho[i:] * g_arr[i:] , rbin[i:]) for i in range(len(rbin))])
C_lolo = 4.3e-6 * np.array([integrator( M_tot_lolo[i:]/rbin[i:]**2 * rho[i:] * g_arr[i:] , rbin[i:]) for i in range(len(rbin))])
C_hihi = 4.3e-6 * np.array([integrator( M_tot_hihi[i:]/rbin[i:]**2 * rho[i:] * g_arr[i:] , rbin[i:]) for i in range(len(rbin))])

sigma_DM = C / g_arr / rho + M_tot[0] / rbin[0] * 4.3e-6 * rho[0]/rho * g_arr[0]/g_arr
sigma_DM_lo = C_lo / g_arr_hi / rho_hi + M_tot_lo[0] / rbin[0] * 4.3e-6 * rho[0]/rho_hi * g_arr[0]/g_arr_hi
sigma_DM_hi = C_hi / g_arr_lo / rho_lo + M_tot_hi[0] / rbin[0] * 4.3e-6 * rho[0]/rho_lo * g_arr[0]/g_arr_lo
sigma_DM_lolo = C_lolo / g_arr_hihi / rho_hihi + M_tot_lolo[0] / rbin[0] * 4.3e-6 * rho[0]/rho_hihi * g_arr[0]/g_arr_hihi
sigma_DM_hihi = C_hihi / g_arr_lolo / rho_lolo + M_tot_hihi[0] / rbin[0] * 4.3e-6 * rho[0]/rho_lolo * g_arr[0]/g_arr_lolo

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
