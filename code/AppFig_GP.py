# OK. So we've gathered our sample. We've run the MCMC to measure SLFV. 
# We've also done simple prewhitening to give us a prior on frequency.

# Now let's jointly model the SLFV and pulsations with a GP

# This is an edit to FindFYPS_GP to just record the final SLFV params + number of frequencies


import numpy as np
import pandas as pd
from TESStools import *
import os
import warnings
from multiprocessing import Pool, cpu_count
from scipy.stats import multivariate_normal
from tqdm.auto import tqdm
import h5py as h5

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from pymc3_ext.utils import eval_in_model
import arviz as az

import exoplanet

cool_sgs = pd.read_csv('sample.csv',index_col=0)

slfv_emcee = pd.read_csv('slfv_params.csv')

prewhitening_summary = pd.read_csv('prewhitening.csv')


# Here's a function that maximizes the likelihood of a GP + arbitrary number of sinusoids
def pm_fit_gp_sin(tic, fs=None, amps=None, phases=None, model=None, return_var=False, thin=50):
    """
    Use PyMC3 to do a maximum likelihood fit for a GP + multiple periodic signals
    
    Inputs
    ------
    time : array-like
        Times of observations
    flux : array-like
        Observed fluxes
    err : array-like
        Observational uncertainties    
    fs : array-like, elements are PyMC3 distributions
        Array with frequencies to fit, default None (i.e., only the GP is fit)
    amps : array-like, elements are PyMC3 distributions
        Array with amplitudes to fit, default None (i.e., only the GP is fit)
    phases : array-like, elements are PyMC3 distributions
        Array with phases to fit, default None (i.e., only the GP is fit)
    model : `pymc3.model.Model`
        PyMC3 Model object, will fail unless given
    return_var : bool, default True
        If True, returns the variance of the GP
    thin : integer, default 50
        Calculate the variance of the GP every `thin` points.
        
    Returns
    -------
    map_soln : dict
        Contains best-fit parameters and the gp predictions
    logp : float
        The log-likelihood of the model
    bic : float
        The Bayesian Information Criterion, -2 ln P + m ln N
    var : float
        If `return_var` is True, returns the variance of the GP
    """
    
    assert model is not None, "Must provide a PyMC3 model object"
    
    #Extract LC
    lc, lc_smooth = lc_extract(get_lc_from_id(tic), smooth='10T')
    time, flux, err = lc['Time'].values, lc['Flux'].values, lc['Err'].values
    
    #Initial Values for SLFV
    slfv_pars = slfv_emcee[slfv_emcee['tic'] == tic]
    
    #Mean model
    mean_flux = pm.Normal("mean_flux", mu = 1.0, sigma=np.std(flux))
    if fs is not None:
        #Making a callable for celerite
        mean_model = tt.sum([a * tt.sin(2.0*np.pi*f*time + phi) for a,f,phi in zip(amps,fs,phases)],axis=0) + mean_flux
        #And add it to the model
        pm.Deterministic("mean", mean_model)

    else:
        mean_model = mean_flux
        mean = pm.Deterministic("mean", mean_flux)
        

    # A jitter term describing excess white noise (analogous to C_w)
    aw = slfv_pars['alphaw'].item() 
        
    log_jitter = pm.Uniform("log_jitter", lower=np.log(aw)-15, upper=np.log(aw)+15, testval=np.log(np.median(np.abs(np.diff(flux)))))

    # A term to describe the SLF variability
    # sigma is the standard deviation of the GP, rho roughly corresponds to the 
    #breakoff in the power spectrum. rho and tau are related by a factor of 
    #pi/Q (the quality factor)

    #guesses for our parameters
    a0 = slfv_pars['alpha'].item()
    tau_char = slfv_pars['tau'].item()
    nu_char = 1.0/(2.0*np.pi*tau_char)
    omega_0_guess = 2*np.pi*nu_char
    Q_guess = 1/np.sqrt(2)
    sigma_guess = a0 * np.sqrt(omega_0_guess*Q_guess) * np.power(np.pi/2.0, 0.25)
    
    #sigma
    logsigma = pm.Uniform("log_sigma", lower=np.log(sigma_guess)-10, upper=np.log(sigma_guess)+10)
    sigma = pm.Deterministic("sigma",tt.exp(logsigma))
    
    #rho (characteristic timescale)
    logrho = pm.Uniform("log_rho", lower=np.log(0.01/nu_char), upper=np.log(100.0/nu_char))
    rho = pm.Deterministic("rho", tt.exp(logrho))
    
    nuchar = pm.Deterministic("nu_char", 1.0 / rho)
    
    #tau (damping timescale)
    logtau = pm.Uniform("log_tau", lower=np.log(0.01*2.0*Q_guess/omega_0_guess),upper=np.log(100.0*2.0*Q_guess/omega_0_guess))
    tau = pm.Deterministic("tau", tt.exp(logtau))
    
    nudamp = pm.Deterministic("nu_damp", 1.0 / tau)
    
    #We also want to track Q, as it's a good estimate of how stochastic the 
    #process is.
    Q = pm.Deterministic("Q", np.pi*tau/rho)

    kernel = terms.SHOTerm(sigma=sigma, rho=rho, tau=tau)

    gp = GaussianProcess(
        kernel,
        t=time,
        diag=err ** 2.0 + tt.exp(2 * log_jitter),
        quiet=True,
    )


    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    gp.marginal("gp", observed=flux-mean_model)

    # Compute the mean model prediction for plotting purposes
    pm.Deterministic("pred", gp.predict(flux-mean_model))

    # Optimize to find the maximum a posteriori parameters
    map_soln = pmx.optimize()
    logp = model.logp(map_soln)
    # parameters are tau, sigma, rho, mean, jitter, plus 3 per frequency 
    if fs is not None:
        n_par = 5.0 + (3.0 * len(fs))
    else:
        n_par = 5.0
    bic = -2.0*logp + n_par * np.log(len(time))
    
    #compute variance as well...
    if return_var:
        eval_in_model(gp.compute(time[::thin],yerr=err[::thin]), map_soln)
        mu, var = eval_in_model(gp.predict(flux[::thin], t=time[::thin], return_var=True), map_soln)
        return map_soln, logp, bic, var
        
    return map_soln, logp, bic


if __name__ == '__main__':
    
    thin = 50 #What to thin by if we're computing the variance of the gp
    tics = []
    rhos = []
    taus = [] 
    sigmas = []
    jitters = [] # log
    
    for tic, row in tqdm(cool_sgs.iterrows(), total=len(cool_sgs)):
        lc, lc_smooth = lc_extract(get_lc_from_id(tic), smooth='10T')
        time, flux, err = lc['Time'].values, lc['Flux'].values, lc['Err'].values
        
        this_summ = prewhitening_summary[prewhitening_summary['TIC']==tic]
        nf = this_summ['n_peaks'].item()
        print(tic)

        if nf == 0.0:
            print('No frequencies found by prewhitening, fitting GP...')
            # Fit just the GP
            with pm.Model() as model_np:
                map_soln, logp, bic_np = pm_fit_gp_sin(tic, model=model_np, thin=thin)

            rhos.append(map_soln['rho'].item())
            taus.append(map_soln['tau'].item())
            sigmas.append(map_soln['sigma'].item())
            jitters.append(map_soln['log_jitter'].item())
            tics.append(tic)

                
        else: #If we DID find frequencies...
            print('Prewhitening frequencies found, fitting everything...')
            with h5.File('prewhitening.hdf5','r') as pw: #load them in from the HDF5 file
                good_fs = pw[f'{tic}/good_fs'][()]
                good_amps = pw[f'{tic}/good_amps'][()]
                good_phases = pw[f'{tic}/good_phases'][()]
            nf_found = np.min([good_fs.shape[0],10]) #limiting to the first ten frequencies
            with pm.Model() as model:
                fs = [pm.Uniform(f"f{i}", lower = good_fs[i, 0] - 3*good_fs[i,1], upper=good_fs[i, 0] + 3*good_fs[i,1]) for i in range(nf_found)]
                amps = [pm.Uniform(f"a{i}", lower = np.max([good_amps[i, 0] - 3*good_amps[i,1],0.0]), upper=good_amps[i, 0] + 3*good_amps[i,1]) for i in range(nf_found)]
                phis = [pmx.Angle(f"phi{i}", testval = good_phases[i,0]) for i in range(nf_found)]
                try: #very occasionally something will break
                    map_soln, logp, bic = pm_fit_gp_sin(tic, fs=fs, amps=amps, phases=phis, model=model)
                    rhos.append(map_soln['rho'].item())
                    taus.append(map_soln['tau'].item())
                    sigmas.append(map_soln['sigma'].item())
                    jitters.append(map_soln['log_jitter'].item())
                    tics.append(tic)
                    
                except:
                    print(f'Broke on TIC {tic}, N_f={nf_found}')
                    continue
            
 
    out_df = pd.DataFrame({'Tau':taus,'Rho':rhos,'Sigma':sigmas,'LogJitter':jitters},index=tics)
    out_df.to_csv('AppFig_GP_results.csv')
