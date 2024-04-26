#####################################################################
Download_Data = 'no'
Data_to_binulator = 'yes'
Bin_param = 'no'
gravsphere_param = 'yes'






gal_name = 'Carina_dSph'










outdir = './gravsphere-master/My_Data/'
########################################################################

import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
import pyvo

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord

from tqdm import tqdm
from scipy.optimize import curve_fit as fit
#from scipy.stats.norm import pdf as gaussian 

if(Download_Data == 'yes'):
    
    Simbad.add_votable_fields('distance')
    Simbad.add_votable_fields('flux(V)')
    Simbad.add_votable_fields('flux_bibcode(V)')
    
    gal_query = Simbad.query_object(gal_name)
    gal_query.to_pandas().to_excel(outdir + gal_name +'_galaxy_properties.xlsx')
    
    if (gal_query['Distance_unit'] == 'kpc'): koeff = 1
    if (gal_query['Distance_unit'] == 'Mpc'): koeff = 1e3
    Distance = gal_query['Distance_distance'] * koeff
    
    coord = SkyCoord(ra=gal_query['RA'][0], dec=gal_query['DEC'][0], unit=(u.hourangle, u.deg), frame='icrs')
    gal_coord = (u.Quantity(coord.ra, unit = u.hourangle).to(u.deg).value, u.Quantity(coord.dec, unit = u.deg).value)

    tap_simbad = pyvo.dal.TAPService("https://simbad.u-strasbg.fr/simbad/sim-tap")
    query = """SELECT main_id AS "child id", link_bibcode, coo_bibcode, DISTANCE(POINT('ICRS',""" +  str(gal_coord[0]) + """,""" +  str(gal_coord[1]) + """), POINT('ICRS', ra, dec)) as dist,
            otype, ra, dec, membership, rvz_radvel, rvz_err, rvz_bibcode,
            cluster_table.id AS "parent galaxy"
            FROM (SELECT oidref, id FROM ident WHERE id = ' """ + gal_name + """ ' ) AS cluster_table,
            basic JOIN h_link ON basic.oid = h_link.child
            WHERE h_link.parent = cluster_table.oidref AND membership = 100; 
            """
    result_table =  tap_simbad.search(query)
    
    Data = pd.DataFrame({'id': result_table['child id'], 'bib': result_table['link_bibcode'] ,'R': result_table['dist'] * np.pi / 180 * Distance, 'Dens': np.array([1 for i in range(len(result_table['child id']))]),\
                         'vel': result_table['rvz_radvel'], 'vel_err': result_table['rvz_err'], 'vel_bib': result_table['rvz_bibcode'], 'memb_prob': result_table['membership'], 'otype': result_table['otype'], 'coo_bib': result_table['coo_bibcode']})
    Data.sort_values(by = 'R', inplace=True)
    Data.reset_index(drop=True, inplace=True)
    
    types = np.unique(Data['otype'])
    print('types = ', types)
    
    qso_ind = np.where(Data['otype'] == 'QSO')[0]
    Data.drop(qso_ind, inplace = True)
    Data.reset_index(drop=True, inplace=True)
    
    ids = np.unique(Data['id'])
    
    pd.options.mode.chained_assignment = None
    for i in tqdm(range(len(ids))):
        ind = np.where(Data['id'] == ids[i])[0]
        bibcodes = list(Data['bib'][ind])
        Data.drop(ind[:-1], inplace=True)
        Data.reset_index(drop=True, inplace=True)
        Data['bib'][ind[-1]] = ';'.join(bibcodes)
        
    Data.to_excel(outdir + gal_name +'.xlsx')
    print('Complete')

if (Data_to_binulator == 'yes'):
    Data = pd.read_excel(outdir + gal_name + '.xlsx')
    
    '''    
    My_Data = pd.DataFrame({'R': Data['R'], 'vel': Data['vel'], 'velerr': Data['vel_err']})
    My_Data['R'] = My_Data['R']
    My_Data.drop(np.where(My_Data['vel'].isnull() == True)[0], inplace=True)
    My_Data.reset_index(drop=True, inplace=True) 
    
    plt.hist(Data['vel'], color = 'blue', edgecolor = 'black', bins = int(np.sqrt(len(Data['vel']))))
    plt.xlabel('v, km/s')
    plt.ylabel('quantity')    
    plt.savefig(outdir + gal_name + '_hist.pdf') 
    plt.cla()

    dgal_kpc = 86.0
    arcsec = 360./2./np.pi * 60. * 60.
    arcmin = arcsec / 60.    
    data_kin_vs = np.genfromtxt('./gravsphere-master/Data/Walker_dwarfs/scl_justin1_spec.dat')
    gotvz = np.where(data_kin_vs[:,6])[0]
    Rkin = data_kin_vs[gotvz,4]
    vz = data_kin_vs[gotvz,6]
    vzerr = data_kin_vs[gotvz,7]
    
    GravData = pd.DataFrame({'R': Rkin*60 ,'vel': vz, 'velerr': vzerr})
    
    plt.scatter(My_Data['R'], My_Data['vel'], edgecolors = 'blue', alpha = 0.33 , label = 'MyData')
    plt.scatter(GravData['R'], GravData['vel'], edgecolors= 'red',s = 0.6, label = 'gravsphere example')
    plt.legend()
    plt.xlabel('r, asec')
    plt.ylabel('v, km/s')
    plt.savefig(outdir + gal_name + '.pdf')
    plt.show()
    '''
    
    ##################################################################3
    
    Data_phot = pd.DataFrame({ 'R': Data['R'], 'Dens': Data['Dens'] })
    plt.hist(Data_phot['R'], bins = int(np.sqrt(len(Data_phot['R']))), edgecolor = 'black')
    plt.savefig(outdir + 'radius_hist.pdf')
    Data_phot.to_csv(outdir + gal_name + '_phot.csv')
    
    Data_spec = pd.DataFrame({'R': Data['R'], 'vel': Data['vel'], 'velerr': Data['vel_err']})
    Data_spec.drop(np.where(Data_spec['vel'].isnull() == True)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)
    Data_spec.drop(np.where(Data_spec['velerr'].isnull() == True)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)
    Data_spec.to_csv(outdir + gal_name + '_spec.csv')
    
    plt.scatter(Data_spec['R'], Data_spec['vel'])
    plt.show()
    
    Data = pd.read_excel(outdir + gal_name + '_galaxy_properties.xlsx')
    Data['N_memb(phot)'] = len(Data_phot['R'])
    Data['N_memb(spec)'] = len(Data_spec['R'])
    
    Data['Nibn'] = int(np.sqrt(Data['N_memb(phot)']))
    Data['Nbin_kin'] = int(np.sqrt(Data['N_memb(spec)']))
    Data.to_excel(outdir + gal_name + '_galaxy_properties.xlsx')
    
    
    
    '''
    Data_spec.drop(np.where(Data_spec['vel'] >= high)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)
    Data_spec.drop(np.where(Data_spec['vel'] <= low)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)
    '''
    
    '''
    vel_sys = np.mean(Data_spec['vel'])
    Data_spec['vel'] = Data_spec['vel'] - vel_sys
    Data_spec.drop(np.where(Data_spec['vel'] - 3 * Data_spec['velerr'] <= 0)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)   
    Data_spec.to_csv(outdir + gal_name + '_spec.csv')
    '''
    
if(Bin_param == 'yes'):
    Data_phot = pd.read_csv(outdir + gal_name + '_phot.csv')
    print('Nbin = ', int(np.sqrt( len( Data_phot['R'] ) )))
    
    Data_spec = pd.read_csv(outdir + gal_name + '_spec.csv')
    print('Nbinkin = ', int(np.sqrt( len( Data_spec['R'] ) )))
    
if(gravsphere_param == 'yes'):
    Mass_vega = 2.135
    Distance_vega = 7.86e-3
    
    gal = pd.read_excel(outdir + gal_name + '_galaxy_properties.xlsx')
    Vmag = gal['FLUX_V'][0]
    Distance = gal['Distance_distance'][0]
    unit = gal['Distance_unit'][0]
    if (unit == 'kpc'): koeff = 1
    if (unit == 'Mpc'): koeff = 1e3
    
    Distance = Distance * koeff
    
    M = Mass_vega * 10**(-Vmag/2.5) * (Distance/Distance_vega)**2
    
    print('dgal_kpc = ', Distance)
    print('Mass = ', M*1e-6, 'e6')
