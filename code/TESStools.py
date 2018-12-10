import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from matplotlib import pyplot as plt
from glob import glob
from astropy.stats import LombScargle
from scipy import stats
import warnings

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
        
        files = glob(data_dir+'*{}*'.format(ticid))
        if len(files) == 0:
            raise ValueError('TIC number not recognized')
        elif len(files) > 2:
            raise ValueError('TIC number ambiguous')
        else:
            lcs = []
            for f in files:
                lc = Table.read(f, format='fits')
                if clean:
                    lc = lc[lc['QUALITY'] == 0]
                if normalize:
                    lc['NormPDCSAP_FLUX'] = lc['PDCSAP_FLUX']/np.nanmedian(lc['PDCSAP_FLUX'])
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
    else:
        rflux = lc['PDCSAP_FLUX'].values

    time = rtime[~np.isnan(rflux)]
    flux = rflux[~np.isnan(rflux)]

    lc_df = pd.DataFrame(data={'Time':time,'Flux':flux}).sort_values('Time')
    
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
    return lc, p