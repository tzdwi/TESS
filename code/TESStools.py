import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from matplotlib import pyplot as plt
from glob import glob
from astropy.stats import LombScargle
from scipy import stats
import warnings
import celerite
from celerite import terms
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import emcee

data_dir = '../data/massive_lcs/'

def get_lc_from_id(ticid, normalize=True, clean=True):
    
    """
    Step 1: load in the light curve from the TIC number
    
    Parameters
    ----------
    
    ticid : int
        TIC number
    
    normalize : bool, default True
        If True, median divides each Sector's data, creates a column 'NormPDCSAP_FLUX'
        
    clean : bool, default True
        If True, selects only data with QUALITY = 0
        
    Returns
    -------
    
    outlc : `pandas.DataFrame`
        Pandas DataFrame containing the lightcurve, with all sector's lightcurves appended
    
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        files = glob(data_dir+'*{0:016d}*'.format(ticid))
        if len(files) == 0:
            raise ValueError('TIC number not recognized')
        else:
            lcs = []
            for f in files:
                lc = Table.read(f, format='fits')
                if clean:
                    lc = lc[lc['QUALITY'] == 0]
                if normalize:
                    lc['NormPDCSAP_FLUX'] = lc['PDCSAP_FLUX']/np.nanmedian(lc['PDCSAP_FLUX'])
                    lc['NormPDCSAP_FLUX_ERR'] = lc['PDCSAP_FLUX_ERR']/np.nanmedian(lc['PDCSAP_FLUX'])
                lcs.append(lc)
            outlc = vstack(lcs)
            return outlc.to_pandas()
        
def peak_finder(f,p,n,FAL=0,width=20):
    """Given frequency/power periodogram, find the first n peaks with power greater than FAL"""
    
    assert (len(f) == len(p))&(len(f) > width),"Length of frequency array and power array much be equal, and >20"
    
    peaks = []
    strengths = []
    
    for i in range(width,len(f) - width):
        
        if np.all(p[i] > p[i -width:i])&np.all(p[i] > p[i + 1:i+width+1])&(p[i]>FAL):
            
            peaks.append(f[i])
            strengths.append(p[i])
    
    isort = np.argsort(strengths)
    peaks = np.array(peaks)[isort]
    strengths = np.array(strengths)[isort]
    return peaks[::-1][:n],strengths[::-1][:n]

def lc_extract(lc, normalize=True, smooth=None):
    """Extract and clean flux from lightcurve df. Optionally also 
    smooth the lightcurve on a rolling median, specified by smooth. Returns both"""
    
    rtime = lc['TIME'].values
    if normalize and ('NormPDCSAP_FLUX' in lc.columns):
        rflux = lc['NormPDCSAP_FLUX'].values
        rerr = lc['NormPDCSAP_FLUX_ERR'].values
    else:
        rflux = lc['PDCSAP_FLUX'].values
        rerr = lc['PDCSAP_FLUX_ERR'].values

    time = rtime[~np.isnan(rflux)]
    flux = rflux[~np.isnan(rflux)]
    err = rerr[~np.isnan(rflux)]

    lc_df = pd.DataFrame(data={'Time':time,'Flux':flux,'Err':err}).sort_values('Time')
    
    if smooth is not None:
        lc_df_smooth = lc_df.rolling(smooth, center=True).median()
        return lc_df, lc_df_smooth
    
    return lc_df

def polynorm(lc, deg=5):
    """Fits lightcurve with polynomial. Inserts 'NormFlux' column with normalized values, returns
    lightcurve in polynomial coefficients"""
    
    p = np.polyfit(lc['Time'].values,lc['Flux'].values,deg=deg)
    vals = np.polyval(p, lc['Time'])
    lc['NormFlux'] = lc['Flux']/vals
    lc['NormErr'] = lc['Err']/vals
    return lc, p

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def dSHO_maxlikelihood(lc, npeaks = 1):
    # Now let's do some of the GP stuff on this
    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(lc['NormFlux']) / (w0 * Q)
    bounds = dict(log_S0=(-16, 16), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                           bounds=bounds)
    kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

    # A periodic component
    for i in range(npeaks):
        Q = 1.0
        w0 = 3.0
        S0 = np.var(lc['NormFlux']) / (w0 * Q)
        kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                                bounds=bounds)
    
    sigma = np.median(lc['NormErr'])
    
    kernel += terms.JitterTerm(log_sigma=np.log(sigma))

    gp = celerite.GP(kernel, mean=np.mean(lc['NormFlux']))
    gp.compute(lc['Time'], lc['NormErr'])  # You always need to call compute once.
    print("Initial log likelihood: {0}".format(gp.log_likelihood(lc['NormFlux'])))
    
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()

    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(lc['NormFlux'], gp))
    gp.set_parameter_vector(r.x)
    
    print("Final log likelihood: {0}".format(gp.log_likelihood(lc['NormFlux'])))
    
    print("Maximum Likelihood Soln: {}".format(gp.get_parameter_dict()))
    
    return gp

def dSHO_emcee(lc, gp, nwalkers = 32, burnin = 500, production = 3000):
    """The full emcee sampling"""
    def log_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(lc['NormFlux']) + lp
    
    initial = gp.get_parameter_vector()
    ndim, nwalkers = len(initial), nwalkers
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

    print("Running {0} steps of burn-in...".format(burnin))
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, burnin)

    print("Running {0} steps of production...".format(production))
    sampler.reset()
    sampler.run_mcmc(p0, production)
    
    return sampler, gp

def parametrized_sin(t, freq, amp, phase):
    "A sin function, used for prewhitening"
    return amp * np.sin(2.0*np.pi*freq*t + phase)

def prewhiten(time, flux, err, verbose = True):
    """
    Runs through a prewhitening procedure to reproduce the variability as sin functions
    
    
    Parameters
    ----------
    time : array-like
        times
    flux : array-like
        fluxes
    err : array-like
        corresponding errors.
    verbose : bool
        If set, will print out every 10th stage of prewhitening, as well as some other diagnostics
        
    Returns
    -------
    good_fs : `~numpy.ndarray`
        Nx2 array with first dimension frequencies, and the second errors
    good_amps :`~numpy.ndarray`
        Nx2 array with first dimension amplitudes, and the second errors
        
    good_amps :`~numpy.ndarray`
        Nx2 array with first dimension amplitudes, and the second errors
    
    """
    
    pseudo_NF = 0.5 / (np.mean(np.diff(time)))
    rayleigh = 1.0 / (np.max(time)-np.min(time))
    if verbose:
        print('f_Ny = {0}, f_R = {1}'.format(pseudo_NF,rayleigh))
    
    #Step 1: subtract off the mean, save original arrays for later
    flux -= np.mean(flux)
    
    original_flux = flux.copy()
    original_err = err.copy()
    
    found_fs = []
    err_fs = []
    found_amps = []
    err_amps = []
    found_phases = []
    err_phases = []
    found_snrs = []
    
    #Step 2: Calculate the Lomb Scargle periodogram
    ls = LombScargle(time, flux, normalization='psd')
    frequency, power = ls.autopower(minimum_frequency=1.0/30.0,
                    maximum_frequency=pseudo_NF)

    #Step 3: Find frequency of max power
    f_0 = frequency[np.argmax(power)]

    #Step 4: Fit the sin. Initial guess is that frequency, the max flux point, and no phase
    # Then save the fit params
    p0 = [f_0, np.max(flux), 0]
    bounds = ([f_0-rayleigh,0,-np.inf],[f_0+rayleigh,np.inf,np.inf])

    popt, pcov = curve_fit(parametrized_sin, time, flux, bounds=bounds, p0=p0)

    found_fs.append(popt[0])
    found_amps.append(popt[1])
    phase = popt[2]
    while phase >= np.pi:
        phase -= 2.0*np.pi
    while phase <= -np.pi:
        phase += 2.0*np.pi
    found_phases.append(phase)
    
    #Calculate the errors
    err_fs.append(np.sqrt(6.0/len(time)) * rayleigh * np.std(flux) / (np.pi * popt[1]))
    err_amps.append(np.sqrt(2.0/len(time)) * np.std(flux))
    err_phases.append(np.sqrt(2.0/len(time)) * np.std(flux) / popt[1])
    
    #Calculate the BIC up to a constant: -2 log L + m log (N)
    log_like_ish = np.sum(np.power(((original_flux - np.sum([parametrized_sin(time, f, amp, phase)
                                    for f, amp, phase in zip(found_fs, found_amps, found_phases)],
                                    axis=0)) / original_err),2.0))
    
    bic = log_like_ish + 3.0*len(found_fs)*np.log(len(time))
    #bic with no fit is:
    old_bic = np.sum(np.power((original_flux/ original_err),2.0))
    bic_dif = bic - old_bic
    
    #subtract off the fit
    flux -= parametrized_sin(time, *popt)
    
    #now loop until BIC hits a minimum
    j = 0
    while bic_dif <= 0:
        #Reset old_bic
        old_bic = bic
        #Lomb Scargle
        ls = LombScargle(time, flux, normalization='psd')
        frequency, power = ls.autopower(minimum_frequency=1.0/30.0,
                        maximum_frequency=pseudo_NF)
        #Highest peak
        f_0 = frequency[np.argmax(power)]
        
        #Fit
        p0 = [f_0, np.max(flux), 0]
        bounds = ([f_0-rayleigh,0,-np.inf],[f_0+rayleigh,np.inf,np.inf])
        popt, pcov = curve_fit(parametrized_sin, time, flux, bounds=bounds, p0=p0)
        
        found_fs.append(popt[0])
        found_amps.append(popt[1])
        phase = popt[2]
        while phase >= np.pi:
            phase -= 2.0*np.pi
        while phase <= -np.pi:
            phase += 2.0*np.pi
        found_phases.append(phase)
        
        #Calculate the errors
        err_fs.append(np.sqrt(6.0/len(time)) * rayleigh * np.std(flux) / (np.pi * popt[1]))
        err_amps.append(np.sqrt(2.0/len(time)) * np.std(flux))      
        err_phases.append(np.sqrt(2.0/len(time)) * np.std(flux) / popt[1])
        
        #Calculate BIC 
        log_like_ish = np.sum(np.power(((original_flux - np.sum([parametrized_sin(time, f, amp, 
                                        phase) for f, amp, phase in zip(found_fs, found_amps, 
                                        found_phases)], axis=0)) / original_err),2.0))
        bic = log_like_ish + 3.0*len(found_fs)*np.log(len(time))
        bic_dif = bic - old_bic
        #subtract off the fit
        flux -= parametrized_sin(time, *popt)
        j+=1
        if (j % 10 == 0) and verbose:
            print(j)
    if verbose:
        print('Found {} frequencies'.format(len(found_fs)-1))
    #pop the last from each array, as it made the fit worse, then turn into numpy arrays
    found_fs = np.array(found_fs[:-1])
    found_amps = np.array(found_amps[:-1])
    found_phases = np.array(found_phases[:-1])
    err_fs = np.array(err_fs[:-1])
    err_amps = np.array(err_amps[:-1])
    err_phases = np.array(err_phases[:-1])
    
    #now add back the last sin function to arrive at "just the noise", and calculate SNR
    flux += parametrized_sin(time, *popt)
    found_snrs = found_amps/np.std(flux)
    
    #Now loop through frequencies. If any of the less-strong peaks are within 1.5/T,
    #get rid of it.
    good_fs = np.array([[found_fs[0],err_fs[0]]])
    good_amps = np.array([[found_amps[0],err_amps[0]]])
    good_phases = np.array([[found_phases[0],err_phases[0]]])
    good_snrs = np.array([found_snrs[0]])
    
    for f,ef,a,ea,p,ep,s in zip(found_fs[1:],err_fs[1:],found_amps[1:],err_amps[1:],found_phases[1:],err_phases[1:],found_snrs[1:]):
        if ~np.any(np.abs(good_fs[:,0] - f) <= 1.5*rayleigh):
            good_fs = np.append(good_fs,[[f,ef]],axis=0)
            good_amps = np.append(good_amps,[[a,ea]],axis=0)
            good_phases = np.append(good_phases,[[p,ep]],axis=0)
            good_snrs = np.append(good_snrs,[s],axis=0)
    if verbose:
        print('{} unique frequencies'.format(len(good_fs)))
    
    return good_fs, good_amps, good_phases #, good_snrs

def harmonic_search(fs, max_n = 10):
    """
    Search for harmonics from a list of frequencies and associated errors
    
    Parameters
    ----------
    fs : Nx2 array like
        fs[:,0] should contain the frequencies, with associated errors in fs[:,1]
        
    max_n : int
        maximum harmonic number to search for.
        
    Returns
    -------
    fundamentals : array like
        value of fundamental frequencies
    harmonics : array like
        value of harmonic frequencies found
    ns : array like
        which order harmonic we've found
    
    """
    fundamentals = []
    harmonics = []
    ns = []
    
    for f,e in fs:
        for i in range(2,max_n+1):
            expected_freq = i*f
            expected_error = i*e
            for fo,eo in fs:
                total_error = np.sqrt(eo**2.0 + expected_error**2.0)
                if np.abs(expected_freq - fo) <= total_error:
                    fundamentals.append(f)
                    harmonics.append(fo)
                    ns.append(i)
    return np.array([fundamentals, harmonics, ns])

def combo_search(fs):
    """
    Search for combinations from a list of frequencies and associated errors
    
    Parameters
    ----------
    fs : Nx2 array like
        fs[:,0] should contain the frequencies, with associated errors in fs[:,1]
        
    Returns
    -------
    f0s : array like
        value of first frequencies
    f1s : array like
        value of second frequencies
    f2s : array like
        value of frequency closest to sum
    
    """
    
    f0s = []
    f1s = []
    f2s = []
    
    for i in range(len(fs)-1):
        
        f, fe = fs[i]
        
        for k in range(i+1,len(fs)):
            
            f1, f1e = fs[k]
            fsum = f+f1
            
            absdifs = np.abs(fsum - fs[:,0])
            diferrs = np.sqrt(fe**2.0 + f1e**2.0 + fs[:,1]**2.0)
            if np.any(absdifs <= diferrs):
                
                f0s.append(f)
                f1s.append(f1)
                f2s.append(fs[:,0][absdifs <= diferrs][0])
                
    return np.array([f0s,f1s,f2s])