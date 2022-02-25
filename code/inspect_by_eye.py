"""
Author: Trevor Dorn-Wallenstein
Date: 5.19.21

Purpose: Going through the new TESS lightcurves of YSGs by eye and determine 
which ones are pulsators, which ones have a bad SLFV fit, and whether there's
anything else interesting about them.

NOTE: This wound up being a bad idea


"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm import tqdm

from TESStools import *

cool_sgs = pd.read_csv('merged.csv',index_col=0)

try:
    contaminated = pd.read_csv('contaminated.csv')
except FileNotFoundError:
    contaminated = pd.DataFrame(columns=['TIC','contaminant'])

#typical TESS mags and amplitudes in magnitudes of some likely contaminants
RR = 15, 1.0 # https://ui.adsabs.harvard.edu/abs/2020svos.conf..465P/abstract
betaSep_lmc = 18 , 2.5*np.log10(1+0.01) # https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.5871B/abstract
betaSep_smc = 19 , 2.5*np.log10(1+0.01) # https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.5871B/abstract
    
if __name__ == '__main__':
    
    
    for tic,star in tqdm(cool_sgs.iterrows(), total=len(cool_sgs)):
        
        if tic not in contaminated['TIC'].values:
            
            if star['highest_amp'] > 0:
            
                fig, ax = plt.subplots(2,2,figsize=(10,6),dpi=180)

                fig.suptitle(f"TIC{tic}, T={star['teff']}, L={star['lum']}")

                print(star[['teff','lum']])

                lc, lc_smooth = lc_extract(get_lc_from_id(tic), smooth=128)
                time, flux, err = lc['Time'], lc['Flux'], lc['Err']

                # plot year 1 and year 3 LC

                cyc1 = time.values < 1800
                cyc3 = time.values >= 1800

                if np.any(cyc1):
                    ax[0,0].plot(time[cyc1],flux[cyc1],'k.',ms=1,rasterized=True)
                    ax[0,0].plot(lc_smooth['Time'][cyc1],lc_smooth['Flux'][cyc1],'C3-',rasterized=True)
                    ax[0,0].set(title='Cycle 1 LC',xlabel='Time [d]',ylabel='Flux')
                else:
                    ax[0,0].text(0.5,0.5,'No Cycle 1 Data',ha='center',va='center')
                    ax[0,0].set(title='Cycle 1 Light Curve')
                    ax[0,0].set_xticks([])
                    ax[0,0].set_yticks([])

                if np.any(cyc3):
                    ax[1,0].plot(time[cyc3],flux[cyc3],'k.',ms=1,rasterized=True)
                    ax[1,0].plot(lc_smooth['Time'][cyc3],lc_smooth['Flux'][cyc3],'C3-',rasterized=True)
                    ax[1,0].set(title='Cycle 3 Light Curve',xlabel='Time [d]',ylabel='Flux')
                else:
                    ax[1,0].text(0.5,0.5,'No Cycle 3 Data',ha='center',va='center')
                    ax[1,0].set(title='Cycle 3 Light Curve')
                    ax[1,0].set_xticks([])
                    ax[1,0].set_yticks([])

                ls = LombScargle(time,flux,dy=err)
                freq,power=ls.autopower(normalization='psd')
                power /= len(time) #correct units
                try:
                    popt, pcov, resid = fit_red_noise(freq, power)
                except RuntimeError:
                    popt, pcov, resid = fit_red_noise(freq[freq < 50], power[freq < 50])
                    freq = freq[freq < 50]

                ax[0,1].loglog(freq, power,rasterized=True,label='Data')
                ax[0,1].loglog(freq, np.power(noise_func(freq, *popt),2.0),rasterized=True,label='Fit')
                ax[0,1].set(xlabel=r'Frequency [d$^{-1}$]',ylabel='Power',title='Lomb-Scargle Periodogram')
                ax[0,1].legend()

                T0 = star['Tmag']
                observed = star['highest_amp']

                T1s = np.linspace(T0-1,20,300)
                dTs = T1s-T0

                R = np.power(10.0,-dTs/2.5)
                real = 2.5*np.log10(1.0 + observed/R)

                ax[1,1].plot(T1s,real)
                ax[1,1].plot([RR[0],RR[0]],[RR[1],real[np.argmin(np.abs(T1s-RR[0]))]],c='C0',zorder=0)
                ax[1,1].scatter(*RR,c='C0',label='Halo RR Lyra')
                ax[1,1].text(RR[0]-0.6,0.5*(real[np.argmin(np.abs(T1s-RR[0]))]+RR[1]),f'{np.abs(real[np.argmin(np.abs(T1s-RR[0]))]-RR[1]):.2f} mag',rotation=90,va='center')

                ax[1,1].plot([betaSep_lmc[0],betaSep_lmc[0]],[betaSep_lmc[1],real[np.argmin(np.abs(T1s-betaSep_lmc[0]))]],c='C2',zorder=0)
                ax[1,1].scatter(*betaSep_lmc,c='C2',label=r'LMC $\beta$ Cep')
                ax[1,1].text(betaSep_lmc[0]+0.05,0.5*(real[np.argmin(np.abs(T1s-betaSep_lmc[0]))]+betaSep_lmc[1]),f'{np.abs(real[np.argmin(np.abs(T1s-betaSep_lmc[0]))]-betaSep_lmc[1]):.2f} mag',rotation=270,va='center')

                ax[1,1].plot([betaSep_smc[0],betaSep_smc[0]],[betaSep_smc[1],real[np.argmin(np.abs(T1s-betaSep_smc[0]))]],c='C3',zorder=0)
                ax[1,1].scatter(*betaSep_smc,c='C3',label=r'SMC $\beta$ Cep')
                ax[1,1].text(betaSep_smc[0]+0.05,0.5*(real[np.argmin(np.abs(T1s-betaSep_smc[0]))]+betaSep_smc[1]),f'{np.abs(real[np.argmin(np.abs(T1s-betaSep_smc[0]))]-betaSep_smc[1]):.2f} mag',rotation=270,va='center')

                ax[1,1].legend()

                ax[1,1].set_xlabel('TESS Mag. of Contaminant/Companion')
                ax[1,1].set_ylabel('Amplitude of Variable [mag]')
                    
                plt.show()
                
                contaminant = input('Is it a likely contaminant? y/n: ')

                plt.clf()
            
            else:
                
                contaminant = 'n'
            
            new_row = {'TIC':tic,'contaminant':contaminant}
            
            contaminated = contaminated.append(new_row,ignore_index=True)
            
            contaminated.to_csv('contaminated.csv',index=False)

