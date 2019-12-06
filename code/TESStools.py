import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from glob import glob
from astropy.timeseries import LombScargle
from scipy import stats
import warnings
import celerite
from celerite import terms
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import emcee
import multiprocessing

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


"""

WAVELET TRANSFORMS, BASED ON FOSTER 1996

"""

#Defining functions

def phi_1(t,omega,tau):
    """
    Returns 1 for all times; sets the coarse detail in the wavelet transform. Note that 
    the parameters omega and tau have no use here; we only include them to loop over
    the basis functions.
    
    Parameters
    ----------
    t : array-like
        times
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
        
    Returns
    -------
    out : array-like
        array of 1s of length t
    
    """
    return np.ones(len(t))

def phi_2(t,omega,tau):
    """
    Second basis function, cos(omega(t-tau))
    
    Parameters
    ----------
    t : array-like
        times
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
        
    Returns
    -------
    out : array-like
        value of phi_2
    
    """
    return np.cos(omega*(t-tau))

def phi_3(t,omega,tau):
    """
    Third basis function, sin(omega(t-tau))
    
    Parameters
    ----------
    t : array-like
        times of observations
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
        
    Returns
    -------
    out : array-like
        value of phi_3
    
    """
    return np.sin(omega*(t-tau))

def w_alpha(t,omega,tau,c): 
    """
    Weighting function for each point at a given omega and tau; (5-3) in Foster (1996)
    
    Parameters
    ----------
    t : array-like
        times of observations
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
    c : float
        Decay constant of the Gaussian envelope for the wavelet
        
    Returns
    -------
    weights : array-like
        Statistical weights of data points
    
    """
    return np.exp(-c*np.power(omega*(t - tau),2.0))

def N_eff(ws):
    """
    Effective number of points contributing to the transform; (5-4) in Foster (1996)
    
    Parameters
    ----------
    ws : array-like
        weights of observations, already calculated
        
    Returns
    -------
    Neff : float
        Effective number of data points
    
    """
    
    return np.power(np.sum(ws),2.0)/np.sum(np.power(ws,2.0))

def function_inner_product(func1,func2,ws):
    """
    Define the inner product of two functions; (4-2) in Foster (1996)
    
    Parameters
    ----------
    func1 : array-like
        Values of f at times corresponding to the weights
    func2 : array-like
        Values of g at times corresponding to the weights
    ws : array-like
        weights of observations, already calculated
        
    Returns
    -------
    inner_product : float
        Inner product of func1 and func2
    
    """
    num = np.sum(ws*func1*func2)
    den = np.sum(ws)
    return num/den

def S_matrix(func_vals,ws):
    """
    Define the S-matrix; (4-2) in Foster (1996)
    
    Takes the values of the functions already evaluated at the times of observations
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        weights of observations, already calculated
        
    Returns
    -------
    S : `numpy.matrix`
        S-matrix; size len(func_vals)xlen(func_vals)
    
    """
    S = np.array([[function_inner_product(f1,f2,ws) for f1 in func_vals] for f2 in func_vals])
    return np.matrix(S)

def inner_product_vector(func_vals,ws,y):
    """
    Generates a column vector consisting of the inner products between the basis
    functions and the observed data
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    phi_y : `numpy.array`
        Column vector where phi_y_i = phi_i * y
    
    """
    return np.array([[function_inner_product(func,y,ws) for func in func_vals]]).T

def coeffs(func_vals,ws,y):
    """
    Calculate the coefficients of each $\phi$. Adapted from (4-4) in Foster (1996)
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    coeffs : `numpy.array`
        Contains coefficients for each basis function
    
    """
    S_m = S_matrix(func_vals,ws)
    phi_y = inner_product_vector(func_vals,ws,y)
    return np.linalg.solve(S_m,phi_y).T

def V_x(f1_vals,ws,y):
    """
    Calculate the weighted variation of the data. Adapted from (5-9) in Foster (1996)
    
    Parameters
    ----------
    f1_vals : array-like
        Array of values of the first basis function; should be equivalent to
        `numpy.ones(len(y))`
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    vx : float
        Weighted variation of the data
    
    """
    return function_inner_product(y,y,ws) - np.power(function_inner_product(f1_vals,y,ws),2.0)

def y_fit(func_vals,ws,y):
    """
    Calculate the value of the model. 
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    y_f : array-like
        Values of the fit model
    y_a : `numpy.array`
        The coefficients returned by `coeffs`
    
    """
    y_a = coeffs(func_vals,ws,y)
    return y_a.dot(func_vals),y_a

def V_y(func_vals,f1_vals,ws,y):
    """
    Calculate the weighted variation of the model. Adapted from (5-10) in Foster (1996) 
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    f1_vals : array-like
        Array of values of the first basis function; should be equivalent to
        `numpy.ones(len(y))`
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    vy : float
        Weighted variation of the model
    y_a :float
        Coefficients from `coeffs`
    
    """
    y_f,y_a = y_fit(func_vals,ws,y)
    return function_inner_product(y_f,y_f,ws) - np.power(function_inner_product(f1_vals,y_f,ws),2.0),y_a

def WWZ(func_list,f1,y,t,omega,tau,c=0.0125,exclude=True):
    """
    Calculate the Weighted Wavelet Transform of the data `y`, measured at times `t`,
    evaluated at a wavelet scale $\omega$ and shift $\tau$, for a decay factor of the
    Gaussian envelope `c`. Adapted from (5-11) in Foster (1996)
    
    Parameters
    ----------
    func_list : array-like
        Array or list containing the basis functions, not yet evaluated
    f1 : array-like
        First basis function. Should be equivalent to `lambda x: numpy.ones(len(x))`
    y : array-like
        Observed data
    t : array-like
        Times of observations
    omega : float
        Scale of wavelet; corresponds to an angular frequency
    tau : float
        Shift of wavelet; corresponds to a time
    c : float
        Decay rate of Gaussian envelope of wavelet. Default 0.0125
    exclude : bool
        If exclude is True, returns 0 if the nearest data point is more than one cycle away.
        Default True.
        
    Returns
    -------
    WWZ : float
        WWZ of the data at the given frequency/time.
    WWA : float
        Corresponding amplitude of the signal at the given frequency/time
    
    """
    
    if exclude and (np.min(np.abs(t-tau)) > 2.0*np.pi/omega):
        return 0.0, 0.0
    
    ws = w_alpha(t,omega,tau,c)
    Neff = N_eff(ws)
    
    func_vals = np.array([f(t,omega,tau) for f in func_list])
    
    f1_vals = f1(t,omega,tau)
    
    Vx = V_x(f1_vals,ws,y)
    Vy,y_a = V_y(func_vals,f1_vals,ws,y)
    
    y_a_rows = y_a[0]
    
    return ((Neff - 3.0) * Vy)/(2.0 * (Vx - Vy)),np.sqrt(np.power(y_a_rows[1],2.0)+np.power(y_a_rows[2],2.0))


def MP_WWZ(func_list,f1,y,t,omegas,taus,
           c=0.0125,exclude=True,mp=True,n_processes=4):
    """
    Calculate the Weighted Wavelet Transform of the data `y`, measured at times `t`,
    evaluated on a grid of wavelet scales `omegas` and shifts `taus`, for a decay factor of 
    the Gaussian envelope `c`. Adapted from (5-11) in Foster (1996).
    
    Note that this can be incredibly slow for a large enough light curve and a dense enough 
    grid of omegas and taus, so we include multiprocessing to speed it up.
    
    Parameters
    ----------
    func_list : array-like
        Array or list containing the basis functions, not yet evaluated
    f1 : array-like
        First basis function. Should be equivalent to `lambda x: numpy.ones(len(x))`
    y : array-like
        Observed data
    t : array-like
        Times of observations
    omega : array-like
        Scale of wavelets; corresponds to an angular frequency
    tau : array-like
        Shift of wavelets; corresponds to a time
    c : float
        Decay rate of Gaussian envelope of wavelet. Default 0.0125
    exclude : bool
        If exclude is True, returns 0 if the nearest data point is more than one cycle away.
        Default True.
    mp : bool
        If `mp` is True, uses the `multiprocessing.Pool` object to calculate the WWZ
        at each point. Default True
    n_processes : int
        If `mp` is True, sets the `processes` parameter of `multiprocessing.Pool`
        
    Returns
    -------
    wwz : array-like
        WWZ of the data evaluated on the frequency/time grid. Shape is 
        `(len(omegas),len(taus))`
    wwa : array-like
        WWA of the data evaluated on the frequency/time grid. Shape is 
        `(len(omegas),len(taus))`
    
    """
    
    if mp:
        args = np.array([[func_list,f1,y,t,omega,tau,c,exclude] for omega in omegas for tau in taus])
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.starmap(WWZ, args)
            transform = np.array(results).reshape(len(omegas),len(taus),2)
            wwz = transform[:,:,0]
            wwa = transform[:,:,1]
        
    else:
        transform = np.array([[WWZ(func_list,f1,y,t,omega,tau,c,exclude) for tau in taus] for omega in omegas])
        wwz = transform[:,:,0].T
        wwa = transform[:,:,1].T
    
    return wwz,wwa


def make_WWZ_plot(wwz,wwa,omegas,taus,t,y,lombscargle=True,**kwargs):
    """
    Makes a pretty plot with the WWZ info
    
    Parameters
    ----------
    wwz : array-like
        `(len(omegas),len(taus))` Weighted Wavelet Z-transform array
    wwa : array-like
        `(len(omegas),len(taus))` Weighted Wavelet Amplitude array
    omegas : array-like
        1-D array of angular frequencies
    taus : array-like
        1-D array of time-shifts
    t : array-like
        1-D array of time observations
    y : array-like
        1-D array of flux or magnitude observations
    lombscargle : bool
        Whether or not to plot the Lomb-Scargle periodogram to compare with the WWZ spectrum. 
        Default True.
    **kwargs
        Passed to `matplotlib.pyplot.figure()`
        
    Returns
    -------
    fig : `matplotlib.pyplot.Figure`
        Figure object
    ax : list
        List of `matplotlib.pyplot.Axes` objects
    
    """
    if lombscargle:
        ls = LombScargle(t,y)
        freq,power = ls.autopower()
    
    fig = plt.figure(constrained_layout=True,**kwargs)

    gs = GridSpec(5, 4, figure=fig)
    lcax = fig.add_subplot(gs[0, :3])
    wwzax = fig.add_subplot(gs[1:3,:3])
    wwaax = fig.add_subplot(gs[3:,:3])
    zsumax = fig.add_subplot(gs[1:3,3])
    asumax = fig.add_subplot(gs[3:,3])

    lcax.scatter(time,flux,s=1,c='k')
    lcax.set(ylabel='Normalized Flux',xlim=(np.min(time),np.max(time)))

    wwzax.contourf(taus,omegas/2.0/np.pi,wwz,levels=100,cmap='cividis')
    wwzax.fill_between(2*np.pi/omegas+np.min(t),0,omegas/2/np.pi,alpha=0.5,facecolor='white')
    wwzax.fill_between(np.max(t)-2*np.pi/omegas,0,omegas/2/np.pi,alpha=0.5,facecolor='white')
    wwzax.set(ylabel=r'Frequency [d$^{-1}$]',ylim=(np.min(omegas)/2/np.pi,np.max(omegas/2/np.pi)))

    wwaax.contourf(taus,omegas/2.0/np.pi,wwa,levels=100,cmap='cividis')
    wwaax.fill_between(2*np.pi/omegas+np.min(t),0,omegas/2/np.pi,alpha=0.5,facecolor='white')
    wwaax.fill_between(np.max(t)-2*np.pi/omegas,0,omegas/2/np.pi,alpha=0.5,facecolor='white')
    wwaax.set(xlabel='Time [d]',ylabel=r'Frequency [d$^{-1}$]',
              ylim=(np.min(omegas)/2/np.pi,np.max(omegas/2/np.pi)))

    zsumax.plot(np.mean(wwz,axis=1),omegas/2.0/np.pi)
    if lombscargle:
        scale = np.max(np.mean(wwz,axis=1))/np.max(power)
        zsumax.plot(power*scale,freq,c='k')
    zsumax.set(yticks=[],xlabel=r'$\langle WWZ \rangle$',
               ylim=(np.min(omegas)/2/np.pi,np.max(omegas/2/np.pi)))

    asumax.plot(np.mean(wwa,axis=1),omegas/2.0/np.pi)
    asumax.set(yticks=[],xlabel=r'$\langle WWA \rangle$',
               ylim=(np.min(omegas)/2/np.pi,np.max(omegas/2/np.pi)))
    if lombscargle:
        scale = np.max(np.mean(wwa,axis=1))/np.max(power)
        asumax.plot(power*scale,freq,c='k')
    
    return fig, [lxax,wwzax,wwaax,zsumax,asumax]
    
    