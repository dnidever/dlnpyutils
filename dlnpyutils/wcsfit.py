#!/usr/bin/env python
#
# WCSFIT.PY - Fit a WCS using stars with known RA/DEC coordinates
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220930'  # yyyymmdd

import time
import numpy as np
import warnings
from astropy.wcs import WCS,SkyCoord
from functools import partial
from astropy.utils.exceptions import AstropyWarning
from scipy.optimize import curve_fit
import copy
from . import utils as dln
from . import ladfit

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def wcsfit(wcs,tab):
    """
    wcs  WCS object
    tab  catalog with x, y, ra, and dec of matched sources
    """

    coo = Skycoord(ra=tab['ra'],dec=tab['dec'],unit='deg')
    
    #func = partial(wcs.pixel_to_world((x,y),0))

    def newwcs(x,*pars):
        # pars = [CRVAL1,CRVAL2,CDELT1,CDELT2,PC1_1,PC1_2,PC2_1,PC2_2]
        twcs = wcs.copy()
        twcs.wcs.crval = pars[0:2]
        twcs.wcs.cdelt = pars[2:4]
        twcs.wcs.pc[0,:] = pars[4:6]
        twcs.wcs.pc[1,:] = pars[6:8]
        return twcs
    def diffcoords(x,*pars):
        twcs = newwcs(x,*pars)
        vcoo = twcs.pixel_to_world(tab['x'],tab['dec'])
        vra = vcoo.ra.deg
        vdec = vcoo.dec.deg
        diff = coo.separation(vcoo)
        
    estimates = [wcs.wcs.crval[0],wcs.wcs.crval[1],wcs.wcs.cdelt[0],wcs.wcs.cdelt[1],
                 wcs.wcs.pc[0,0],wcs.wcs.pc[0,1],wcs.wcs.pc[1,0],wcs.wcs,pc[1,1]]
    bounds = (np.zeros(len(estimates),float)-np.inf,np.zeros(len(estimates),float)+np.inf)
    x = np.zeros(len(tab),float)
    y = np.zeros(len(tab),float)    
    popt,pcov = curve_fit(diffcoords,x,y,x0=estimates,bounds=bounds)

    wcs2 = newwcs(1,*popt)
    
    return wcs2
