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
import emcee

data = 'massive_2min.csv'
data_dir = '../data/massive_lcs/'
massive = pd.read_csv(data)

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

def lc_extract(lc, smooth=None):
    """Extract and clean flux from lightcurve df. Optionally also 
    smooth the lightcurve on a rolling median, specified by smooth. Returns both"""
    
    rtime = lc['TIME'].values
    if 'NormPDCSAP_FLUX' in lc.columns:
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