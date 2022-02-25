"""
Author: Trevor Dorn-Wallenstein
Date: 5.19.21

Purpose: Going through the new TESS lightcurves of YSGs automatically and determine 
which ones are pulsators, which ones have a bad SLFV fit, and whether there's
anything else interesting about them.


"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import h5py as h5

from tqdm import tqdm

from TESStools import *

cool_sgs = pd.read_csv('sample.csv',index_col=0)
slfv_emcee = pd.read_csv('slfv_params.csv', index_col=0)

inspected = pd.DataFrame(columns=['TIC','n_peaks','highest_amp','highest_amp_error','f0','f0_error','lc_length'])
    
if __name__ == '__main__':
    
    prewhitened_frequencies = h5.File('prewhitening.hdf5',mode='w')
    for tic,star in tqdm(cool_sgs.iterrows(), total=len(cool_sgs)):
    
        lc, lc_smooth = lc_extract(get_lc_from_id(tic), smooth=128)
        time, flux, err = lc['Time'], lc['Flux'], lc['Err']
        lc_length = time.max()-time.min()

        ls = LombScargle(time,flux,dy=err)
        freq,power=ls.autopower(normalization='psd')
        power /= len(time) #correct units
        try:
            popt, pcov, resid = fit_red_noise(freq, power)
        except RuntimeError:
            popt, pcov, resid = fit_red_noise(freq[freq < 50], power[freq < 50])
            freq = freq[freq < 50]

        good_fs, good_amps, good_phases, good_snrs, good_peaks = prewhiten_harmonic(time, flux, err, 
                                                                                    max_freq=50.0, max_nfreq=30,
                                                                                    verbose=False)
        prewhitened_frequencies.create_dataset(f'{tic}/good_fs',data=good_fs)
        prewhitened_frequencies.create_dataset(f'{tic}/good_amps',data=good_amps)
        prewhitened_frequencies.create_dataset(f'{tic}/good_phases',data=good_phases)
        prewhitened_frequencies.create_dataset(f'{tic}/good_snrs',data=good_snrs)
        prewhitened_frequencies.create_dataset(f'{tic}/good_peaks',data=good_peaks)

        n_peaks = len(good_fs)

        if len(good_fs) > 0:
            highest_amp = good_amps[0,0]
            highest_amp_error = good_amps[0,1]
            f0 = good_fs[0,0]
            f0_error = good_fs[0,1]
        else:
            highest_amp = np.nan
            highest_amp_error = np.nan
            f0 = np.nan
            f0_error = np.nan

        new_row = {'TIC':tic,'n_peaks':n_peaks,'highest_amp':highest_amp,'highest_amp_error':highest_amp_error,
                   'f0':f0,'f0_error':f0_error,'lc_length':lc_length}

        inspected = inspected.append(new_row,ignore_index=True)

        inspected.to_csv('auto_inspected.csv',index=False)
    prewhitened_frequencies.close()

