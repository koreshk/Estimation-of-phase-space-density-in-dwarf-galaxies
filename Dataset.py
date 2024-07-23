#####################################################################
Download_Data = 'yes'
Data_to_binulator = 'yes'
Bin_param = 'yes'
gravsphere_param = 'yes'





gal_name = 'Z_126-111'









outdir = './gravsphere-master/My_Data/'
########################################################################

from gravsphere_initialise_CVnI import *
from figures import *

import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
import pyvo

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord

from tqdm import tqdm

if(Download_Data == 'yes'):
    
    Simbad.add_votable_fields('distance')
    Simbad.add_votable_fields('flux(V)')
    Simbad.add_votable_fields('flux_bibcode(V)')
    
    gal_query = Simbad.query_object(gal_name)
    
    gal_properties = pd.DataFrame({'Name': (gal_query['MAIN_ID'][0])[5:],\
                                   'RA(ICRS) [h:m:s]': gal_query['RA'][0],\
                                   'DEC(ICRS) [d:m:s]': gal_query['DEC'][0],\
                                   'Coordinate reference': gal_query['COO_BIBCODE'][0],\
                                   'Distance': gal_query['Distance_distance'][0],\
                                   '[Distance unit]': gal_query['Distance_unit'][0],\
                                   'Distance reference': gal_query['Distance_bibcode'][0],\
                                   'Flux(V) [Vega mag]': gal_query['FLUX_V'][0],\
                                   'Flux reference': gal_query['FLUX_BIBCODE_V'][0],\
                                   }, index=[0])
    gal_properties.to_excel(outdir + whichgal.replace(' ','_') +'_galaxy_properties.xlsx', index=False)
    
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
    Data.sort_values(by = 'bib', inplace=True)
    Data.reset_index(drop=True, inplace=True)
    
    types = np.unique(Data['otype'])
    
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
    
    Data.sort_values(by = 'R', inplace=True)
    Data.reset_index(drop=True, inplace=True)
    
    result_Data = pd.DataFrame({'id': Data['id'],\
                                'object reference': Data['bib'],\
                                'coordinate reference': Data['coo_bib'],\
                                'r [kpc]': Data['R'],\
                                'Dens (Membership probability)': Data['Dens'],\
                                'velocity [km/s]': Data['vel'],\
                                'velocity uncertainty [km/s]': Data['vel_err'],\
                                'velocity reference': Data['vel_bib'],\
                                'object type': Data['otype']})
    
    result_Data.to_excel(outdir + whichgal.replace(' ','_') +'.xlsx', index=False)
    print('Complete')

if (Data_to_binulator == 'yes'):
    whichgal = whichgal.replace(' ','_')
    Data = pd.read_excel(outdir + gal_name + '.xlsx')
    
    '''
    Data.drop(np.where(Data['r [kpc]'] >= 1.5)[0], inplace=True)
    Data.reset_index(drop=True, inplace=True) 
    Data.drop(np.where(Data['object type'] == 'QSO')[0], inplace=True)
    Data.reset_index(drop=True, inplace=True) 
    '''
    
    Data_phot = pd.DataFrame({ 'R': Data['r [kpc]'], 'Dens': Data['Dens (Membership probability)'] })
    
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)    

    Y, X = np.histogram(Data_phot['R'], bins = int(np.sqrt(len(Data_phot['R']))))
    plt.hist(Data_phot['R'], bins = int(np.sqrt(len(Data_phot['R']))), edgecolor = 'black')
    plt.axhline(max(Y)/2, color = 'black')
    ind = np.where(Y >= max(Y)/2)[0][0]
    plt.axvline(X[ind], color = 'red', label = 'r =' + str(X[ind]))
    plt.axvline(X[ind+1], color = 'green', label = 'r =' + str(X[ind+1]))
    plt.legend()
    
    plt.xlabel(r'$r\,[{\rm kpc}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$N_{{\rm stars}}$',\
                   fontsize=myfontsize)    
    plt.savefig(outdir + gal_name + '_radius_hist.pdf')
    
    Data_phot.to_csv(outdir + gal_name + '_phot.csv')
    
    Data_spec = pd.DataFrame({'R': Data['r [kpc]'], 'vel': Data['velocity [km/s]'], 'velerr': Data['velocity uncertainty [km/s]']})
    
    Data_spec.drop(np.where(Data_spec['vel'].isnull() == True)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)
    Data_spec.drop(np.where(Data_spec['velerr'].isnull() == True)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)
    
    '''
    Data_spec.drop(np.where(Data_spec['vel'] <= -100)[0], inplace=True)
    Data_spec.reset_index(drop=True, inplace=True)  
    '''
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize) 
    
    plt.hist(Data_spec['vel'], bins = int(np.sqrt(len(Data_spec['vel']))), edgecolor = 'black')
    
    plt.xlabel(r'$v\,[{\rm km/s}]$',\
                   fontsize=myfontsize)
    plt.ylabel(r'$N_{{\rm stars}}$',\
                   fontsize=myfontsize)    
    plt.savefig(outdir + gal_name + '_vel_hist.pdf')
    
    Data_spec.to_csv(outdir + gal_name + '_spec.csv')
    
if(Bin_param == 'yes'):
    whichgal = whichgal.replace(' ','_')
    
    Data = pd.read_excel(outdir + gal_name + '_galaxy_properties.xlsx')
    Data['Number of members (photometric)'] = len(Data_phot['R'])
    Data['Number of members (kinematic)'] = len(Data_spec['R'])
    
    Data['Nbin'] = int(np.sqrt(len(Data_phot['R'])))
    Data['Nbinkin'] = int(np.sqrt(len(Data_phot['R'])))
    
    Data.to_excel(outdir + gal_name + '_galaxy_properties.xlsx', index=False)    
    
    print('Nbin = ', Data['Nbin'][0])
    print('Nbinkin = ', Data['Nbinkin'][0])
    
if(gravsphere_param == 'yes'):
    whichgal = whichgal.replace(' ','_')
    Mass_vega = 2.135
    Distance_vega = 7.86e-3
    
    gal = pd.read_excel(outdir + gal_name + '_galaxy_properties.xlsx')
    Vmag = gal['Flux(V) [Vega mag]'][0]
    Distance = gal['Distance'][0]
    unit = gal['[Distance unit]'][0]
    if (unit == 'kpc'): koeff = 1
    if (unit == 'Mpc'): koeff = 1e3
    
    Distance = Distance * koeff
    
    M = Mass_vega * 10**(-Vmag/2.5) * (Distance/Distance_vega)**2
    
    gal['Mass [Msun]'] = str(M*1e-6) + 'e6'
    gal.to_excel(outdir + gal_name + '_galaxy_properties.xlsx', index=False)
    
    print('dgal_kpc = ', Distance)
    print('Mass = ', M*1e-6, 'e6')