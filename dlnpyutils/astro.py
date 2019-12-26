#!/usr/bin/env python
#
# ASTRO.PY - astronomy utility functions
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20191226'  # yyyymmdd

#import re
#import logging
#import os
#import sys
import numpy as np
import warnings
#from astropy.io import fits
#from astropy.table import Table, Column
#from astropy import modeling
#from astropy.convolution import Gaussian1DKernel, convolve
#from glob import glob
#from scipy.signal import medfilt
#from scipy.ndimage.filters import median_filter,gaussian_filter1d
#from scipy.optimize import curve_fit, least_squares
#from scipy.special import erf
#from scipy.interpolate import interp1d
#from scipy.linalg import svd
from astropy.utils.exceptions import AstropyWarning
#import socket
#from scipy.signal import convolve2d
#from scipy.ndimage.filters import convolve
#import astropy.stats
from . import utils


# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Convert wavelengths in air to vacuum
def airtovac(wave):
    """
    Convert air wavelengths to vacuum wavelengths 

    Wavelengths are corrected for the index of refraction of air under 
    standard conditions.  Wavelength values below 2000 A will not be 
    altered.  Uses relation of Ciddor (1996).

    INPUT/OUTPUT:
      WAVE_AIR - Wavelength in Angstroms, scalar or vector
              If this is the only parameter supplied, it will be updated on
              output to contain double precision vacuum wavelength(s). 

    EXAMPLE:
      If the air wavelength is  W = 6056.125 (a Krypton line), then 
      AIRTOVAC, W yields an vacuum wavelength of W = 6057.8019

    METHOD:
	Formula from Ciddor 1996, Applied Optics 62, 958

    NOTES: 
      Take care within 1 A of 2000 A.   Wavelengths below 2000 A *in air* are
      not altered.       
    REVISION HISTORY
      Written W. Landsman                November 1991
      Use Ciddor (1996) formula for better accuracy in the infrared 
          Added optional output vector, W Landsman Mar 2011
      Iterate for better precision W.L./D. Schlegel  Mar 2011
    """

    wave_air = np.atleast_1d(wave).copy()  # makes sure it's an array
    wave_vac = np.atleast_1d(wave).copy()  # initialize
    
    g,ng = utils.where(wave_vac >= 2000)     #Only modify above 2000 A
    
    if ng>0:
        for iter in range(2):
            sigma2 = (1e4/wave_vac[g] )**2     # Convert to wavenumber squared
            
            # Compute conversion factor
            fact = 1.0 +  5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/( 57.362e0 - sigma2)
            
            wave_vac[g] = wave_air[g]*fact              # Convert Wavelength

    return wave_vac


def vactoair(wave_vac):
    """
    Convert vacuum wavelengths to air wavelengths

    Corrects for the index of refraction of air under standard conditions.  
    Wavelength values below 2000 A will not be altered.  Accurate to 
    about 10 m/s.


    INPUT/OUTPUT:
	WAVE_VAC - Vacuum Wavelength in Angstroms, scalar or vector
		If the second parameter is not supplied, then this will be
               updated on output to contain double precision air wavelengths.

    EXAMPLE:
	If the vacuum wavelength is  W = 2000, then 

	IDL> VACTOAIR, W 

	yields an air wavelength of W = 1999.353 Angstroms

    METHOD:
	Formula from Ciddor 1996  Applied Optics , 35, 1566

    REVISION HISTORY
      Written, D. Lindler 1982 
      Documentation W. Landsman  Feb. 1989
      Use Ciddor (1996) formula for better accuracy in the infrared 
           Added optional output vector, W Landsman Mar 2011
    """

  
    wave_vac = np.atleast_1d(wave_vac).copy()  # makes sure it's an array
    wave_air = np.atleast_1d(wave_vac).copy()  # initialize
    g,ng = utils.where(wave_air >= 2000)     # Only modify above 2000 A
    
    if ng>0:
        sigma2 = (1e4/wave_vac[g] )**2   # Convert to wavenumber squared

        # Compute conversion factor
        fact = 1.0 +  5.792105e-2/(238.0185e0 - sigma2) + 1.67917e-3/( 57.362e0 - sigma2)
    
        # Convert wavelengths
        wave_air[g] = wave_vac[g]/fact

    return wave_air

