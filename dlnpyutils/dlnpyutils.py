#!/usr/bin/env python
#
# DLNPYUTILS.PY - Utility functions.
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20180823'  # yyyymmdd

import re
import logging
import os
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
    """ Calculate the median absolute deviation."""
    if type(data) is not np.ndarray: raise ValueError("data must be a numpy array")    
    return 1.4826 * astropy.stats.median_absolute_deviation(data,axis=axis,func=func,ignore_nan=ignore_nan)

def minmax(a):
    """ Return a 2-element array of minimum and maximum."""
    if type(a) is not np.ndarray: raise ValueError("a must be a numpy array")
    return np.array([np.min(a),np.max(a)])

def stat(a=None,silent=False):
    """ Returns basic statistics on an array."""
    if a is None: raise ValueError("a must be input")
    if type(a) is not np.ndarray: raise ValueError("a must be a numpy array")
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


def strlen(a):
    """ Calculate the string lengths of a string array."""
    if a is None: raise ValueError("a must be input")
    n = len(np.array(a))
    out = np.zeros(n,int)
    for i in range(n):
        out[i] = len(a[i])
    return out
      

def strjoin(a=None,b=None,sep=None):
    """ Join two lists/arrays of strings"""
    if (a is None) | (b is None): raise ValueError("a and b must be input")
    if sep is None: sep=''
    n = len(np.array(a))
    len1 = strlen(a)
    len2 = strlen(b)
    nlen = np.max([len1+len2])+len(sep)
    out = np.zeros(n,(np.str,nlen))
    for i in range(n):
        out[i] = sep.join((a[i],b[i]))
    if type(a) is list: return list(out)
    return out


def strsplit(lst=None,delim=None,asarray=False):
    """ Split a string array."""
    if (lst is None): raise ValueError("lst must be input")
    out = [l.split(delim) for l in lst]
    if asarray is True:
        nlst = len(np.array(lst))
        nel = [len(o) for o in out]
        nlen = np.max(strlen(lst))
        outarr = np.zeros((nlst,np.max(nel)),(np.str,nlen))
        for i in range(nlst):
            temp = np.array(out[i])
            ntemp = len(temp)
            outarr[i,0:ntemp] = temp
        return outarr
    else:
        return out


# Standard grep function that works on string list
def grep(lines=None,expr=None,index=False):
    """
    Similar to the standard unit "grep" but run on a list of strings.
    Returns a list of the matching lines unless index=True is set,
    then it returns the indices.

    Parameters
    ----------
    lines : list
          The list of string lines to check.
    expr  : str
          Scalar string expression to search for.
    index : bool, optional
          If this is ``True`` then the indices of matching lines will be
          returned instead of the actual lines.  index is ``False`` by default.

    Returns
    -------
    out : list
        The list of matching lines or indices.

    Example
    -------

    Search for a string and return the matching lines:

    .. code-block:: python

        mlines = grep(lines,"hello")

    Search for a string and return the indices of the matching lines:

    .. code-block:: python

        index = grep(lines,"hello",index=True)

    """
    if lines is None: raise ValueError("lines must be input")
    if expr is None: raise ValueError("expr must be input")
    out = []
    cnt = 0L
    for l in lines:
        m = re.search(expr,l)
        if m != None:
            if index is False:
                out.append(l)
            else:
                out.append(cnt)
        cnt = cnt+1
    return out


# Read in all lines of files
def readlines(fil=None):
    """
    Read in all lines of a file.
    
    Parameters
    ----------
    file : str
         The name of the file to load.
   
    Returns
    -------
    lines : list
          The list of lines from the file

    Example
    -------

    .. code-block:: python

       lines = readlines("file.txt")

    """
    if fil is None: raise ValueError("File not input")
    f = open(fil,'r')
    lines = f.readlines()
    f.close()
    return lines


# Write all lines to file
def writelines(filename=None,lines=None,overwrite=True):
    """
    Write a list of lines to a file.
    
    Parameters
    ----------
    filename : str
        The filename to write the lines to.
    lines : list
         The list of lines to write to a file.
    overwrite : bool, optional, default is True
        If the output file already exists, then overwrite it.

    Returns
    -------
    Nothing is returned.  The lines are written to `fil`.

    Example
    -------

    .. code-block:: python

       writelines("file.txt",lines)

    """
    # Not enough inputs
    if lines is None: raise ValueError("No lines input")
    if filename is None: raise ValueError("No file name input")
    # Check if the file exists already
    if os.path.exists(filename):
        if overwrite is True:
            os.remove(filename)
        else:
            print(filename+" already exists and overwrite=False")
            return
    # Write the file
    f = open(filename,'w')
    f.writelines(lines)
    f.close()


# Remove indices from a list
def remove_indices(lst=None,index=None):
    """
    This will remove elements from a list given their indices.

    Parameters
    ----------
    lst : list
          The list from which to remove elements.
    index : list or array
          The list or array of indices to remove.

    Returns
    -------
    newlst : list
           The new list with indices removed.

    Example
    -------

    Remove indices 1 and 5 from array `arr`.

    .. code-block:: python

        index = [1,5]
        arr  = range(10)
        arr2 = remove_indices(arr,index)
        print(arr)
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    """
    if lst is None: raise ValueError("list must be input")
    if index is None: raise ValueError("index must be input")
    newlst = []
    for i in range(len(lst)):
       if i not in index: newlst.append(lst[i])
    return newlst


# Little function used by numlines
def blocks(files, size=65536):
    """
    This is a small utility function used by numlines()
    """
    while True:
        b = files.read(size)
        if not b: break
        yield b


# Read number of lines in a file
def numlines(fil=None):
    """
    This function quickly counts the number of lines in a file.

    Parameters
    ----------
    fil : str
          The filename to check the number of lines.

    Returns
    -------
    nlines : int
           The number of lines in `fil`.

    Example
    -------

    .. code-block:: python

        n = numlines("file.txt")

    """
    if fil is None: raise ValueError("file must be input")
    with open(fil, "r") as f:
        return (sum(bl.count("\n") for bl in blocks(f)))

    # Could also use this
    #count=0
    #for line in open(fil): count += 1


# Set up basic logging to screen
def basiclogger(name=None):
    """
    This sets up a basic logger that writes just to the screen.
    """
    if name is None: name = "log"
    logger = logging.getLogger(name)
    # Only add a handler if none exists
    #  the logger might already have been created
    if len(logger.handlers)==0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)-2s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger






