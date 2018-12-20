import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import pandas as pd
from TESStools import *
import corner

#Call this file like python pipeline.py ticid.txt where ticid.txt is a file that contains 
#the TESS ID numbers for the sources you want to run the pipeline on

if __name__ == '__main__':
    
    from sys import argv
    
    tid_file = argv[1]
    tids = np.genfromtxt(tid_file, dtype=int)
    
    massive = pd.read_csv(data)

    gp_columns = ['tid','log_S0_0', 'log_omega0_0', 'log_S0_1', 'log_Q_1', 'log_omega0_1', 'log_sigma']
    latex_names = ['$\log{S_0}_1$','$\log{\omega_0}_1$','$\log{S_0}_2$','$\log{Q}_2$','$\log{\omega_0}_2$','$\log{\sigma}$']
    gp_tab = Table([[] for item in gp_columns], names=gp_columns)
    
    for tid in tids:
        
        fname = np.unique(massive['CommonName'][massive['ticid']==tid])[0].replace('*','').replace(' ','')
        print(tid, fname)
        
        #load in the lightcurve
        lc, lc_smooth = lc_extract(get_lc_from_id(tid), smooth=128)
        print('LC loaded...')
        
        #plot the lightcurve and the 128 point rolling median
        fig = plt.figure(figsize=(15,5))
        plt.scatter(lc['Time'],lc['Flux'],s=1,c='k',label='Data')
        plt.plot(lc_smooth['Time'],lc_smooth['Flux'],c='C2',label='Rolling Median')
        plt.gca().set(xlabel='BJD - 2457000 [$d$]', ylabel='Normalized PDCSAP\_FLUX [$e^-/s$]')
        h, l = plt.gca().get_legend_handles_labels()
        plt.legend(handles = h[::-1], labels = l[::-1], prop={'size': 14})
        plt.savefig('{}_lc.pdf'.format(fname))
        fig.close('all')
        print('LC plotted...')
        
        #Normalize by a polynomial
        lc, p = polynorm(lc, deg=7)
        print('LC normalized')
        
        #Now calculate LS periodogram below 10 days
        ls = LombScargle(lc['Time'], lc['NormFlux'])
        freq, power = ls.autopower(minimum_frequency=1.0/10.0,
                            maximum_frequency=1.0/0.1)

        fs, ps = peak_finder(freq, power, 10, width=10)

        dom = fs[np.argmax(ps)]
        dp = 1.0/dom

        FAL = ls.false_alarm_level(0.01)

        fig = plt.figure(figsize=(15,5))

        plt.plot(1.0/freq,power)
        plt.axhline(y=FAL, ls='--', c='k', label='$FAP=0.01$')
        for i in np.arange(1,6):
            if i == 1:
                plt.axvline(x=1.0/dom, ls='--', c='C4', label='{0:.2f}-day Period and Harmonics'.format(dp))
            else:
                plt.axvline(x=1.0/(i*dom), ls='--', c='C4')
        plt.legend(loc=2, framealpha=0.9, prop={'size': 14})
        plt.gca().set(xlabel='Period [$d$]', ylabel='Lomb-Scargle Power', xlim=(0.1,10))
        plt.savefig('{}_periodogram.pdf'.format(fname))
        fig.close('all')
        
        freq, power = ls.autopower(minimum_frequency=1.0/28.0,
                        maximum_frequency=1.0/0.1)

        fig = plt.figure(figsize=(15,5))
        plt.loglog(1.0/freq,power)
        plt.gca().set(xlabel='Period [$d$]', ylabel='Lomb-Scargle Power', xlim=(0.1,28))
        plt.savefig('{}_logperiodogram.pdf'.format(fname))
        fig.close('all')
        
        print('Periodograms plotted...')
        
        # Now let's do some of the GP stuff on this with twin SHOs
        gp = dSHO_maxlikelihood(lc)
        
        print('GP likelihood maximized...')
        
        x = np.linspace(np.min(lc['Time']), np.max(lc['Time']), 5000)
        pred_mean, pred_var = gp.predict(lc['NormFlux'], x, return_var=True)
        pred_std = np.sqrt(pred_var)
        
        fig = plt.figure(figsize=(15,5))
        plt.scatter(lc['Time'], lc['NormFlux'], label='Data', s=1, c='k')
        plt.plot(x, pred_mean, color='C3', label='Gaussian Process')
        plt.fill_between(x, pred_mean+pred_std, pred_mean-pred_std, color='C3', alpha=0.3,
                 edgecolor="none")
        plt.gca().set(xlabel='BJD - 2457000 [$d$]', ylabel='Normalized PDCSAP\_FLUX [$e^-/s$]')
        plt.savefig('{}_GPlc.pdf'.format(fname))
        fig.close('all')
        
        omega = np.exp(np.linspace(np.log(2.0*np.pi/28.0), np.log(2.0*np.pi/0.1), 5000))
        psd = gp.kernel.get_psd(omega)

        fig = plt.figure(figsize=(15,5))
        
        plt.plot(1.0/freq,power/np.max(power),c='C0')
        plt.plot(2.0*np.pi/omega, psd/np.max(psd), color='C3')
        for k in gp.kernel.terms:
            plt.plot(2.0*np.pi/omega, k.get_psd(omega)/np.max(psd), "--", color='C3')
            plt.axhline(y=np.exp(gp.kernel.terms[2].get_parameter_dict()['log_sigma']),ls='--', color='C3')

        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(2.0*np.pi/omega[-1], 2.0*np.pi/omega[0])
        plt.xlabel("Timescale [d]")
        plt.ylabel("Normalized Power")
        
        plt.savefig('{}_GPperiodogram.pdf'.format(fname))
        fig.close('all')
        
        print('GP max likelihood plots made...')
        print('Starting sampler...')
        
        sampler, gp = dSHO_emcee(lc, gp)
        
        print('Samples complete!')
        
        fig = plt.figure(figsize=(15,5))
        
        plt.scatter(lc['Time'], lc['NormFlux'], label='Data', s=1, c='k')

        # Plot 24 posterior samples.
        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=24)]:
            gp.set_parameter_vector(s)
            mu = gp.predict(lc['NormFlux'], lc['Time'], return_cov=False)
            plt.plot(lc['Time'], mu, color='C3', alpha=0.3)

        plt.gca().set(xlabel='BJD - 2457000 [$d$]', ylabel='Normalized PDCSAP\_FLUX [$e^-/s$]')
        plt.savefig('{}_GPdraws.pdf'.format(fname))
        fig.close('all')
        
        fig = plt.figure(figsize=(15,15))
        true_params = [np.mean(sampler.flatchain[:,i]) for i in range(sampler.flatchain.shape[1])]
        gp_tab.add_row([tid]+true_params)
        corner.corner(sampler.flatchain, truths=true_params,
              labels=latex_names)
        plt.savefig('{}_GPpost.pdf'.format(fname))
        fig.close('all')
        
        print('emcee plots made! On to the next')
        
        
ascii.write(gp_tab,'gp_tab.tex',format='aastex')
        
        