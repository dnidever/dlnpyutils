#!/usr/bin/env python

#import os
import sys
import numpy as np
#import warnings
#from astropy.io import fits
#from astropy.utils.exceptions import AstropyWarning
#import time
#import shutil
#import re
#import subprocess
#import glob
#import logging
#import socket
#from scipy.signal import convolve2d
#from scipy.ndimage.filters import convolve
import astropy.stats

# Median Absolute Deviation
def mad(data, axis=None, func=None, ignore_nan=False):
    return 1.4826 * astropy.stats.median_absolute_deviation(data,axis=axis,func=func,ignore_nan=ignore_nan)

def minmax(a):
    return np.array([np.min(a),np.max(a)])

def stat(a,silent=False):
    #  This is what stat returns:
    #  info[0]: Number of Elements
    #  info[1]: Minimum
    #  info[2]: Maximum
    #  info[3]: Range
    #  info[4]: Mean
    #  info[5]: Median
    #  info[6]: Standard Deviation
    #  info[7]: Standard Error
    #  info[8]: Root Mean Square (R.M.S.)
    #  info[9]: MAD estimate of St.Dev.
    info = np.zeros(10)
    info[0] = len(a)
    info[1] = np.min(a)
    info[2] = np.max(a)
    info[3] = info[2]-info[1]
    info[4] = np.mean(a)
    if info[0] > 1:
        info[5] = np.median(a)
        info[6] = np.std(a,ddof=1)               # std. dev.
        info[7] = info[6]/np.sqrt(info[0])       # std. err.
        info[8] = np.sqrt(np.sum(a**2)/info[0])  # RMS
        info[9] = mad(a)                         # MAD

    if silent is not True:
        print('----------------------')
        print('elements = %d' % info[0])
        print('minimum  = %f' % info[1])
        print('maximum  = %f' % info[2])
        print('range    = %f' % info[3])
        print('mean     = %f' % info[4])
        print('median   = %f' % info[5])
        print('st. dev. = %f' % info[6])
        print('st. err. = %f' % info[7])
        print('r.m.s.   = %f' % info[8])
        print('mad s.d. = %f' % info[9])
        print('----------------------')
    return
