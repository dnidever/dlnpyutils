#!/usr/bin/env python
#
# DLNPYUTILS.PY - Utility functions.
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20180823'  # yyyymmdd

import re
import logging
import os
import sys
import gzip
import time
import numpy as np
import shutil
import warnings
from astropy.io import fits
from astropy.table import Table, Column
from astropy import modeling
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from glob import glob as glb
from scipy.signal import medfilt
from scipy.stats import mstats
from scipy.ndimage import median_filter,gaussian_filter1d,generic_filter
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d,splrep,BSpline
from scipy.linalg import svd
from scipy.signal import savgol_filter
import astropy.stats
from matplotlib.backend_bases import MouseButton
import pandas as pd
#import matplotlib.pyplot as plt
import dill as pickl
import traceback
import inspect
import hashlib

# Ignore these warnings, it's a bug
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# NUMPY_LT_1_18
def _expand_dims(data, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the
    expanded array shape.

    This function allows for tuple axis arguments.
    ``numpy.expand_dims`` currently does not allow that, but it will in
    numpy v1.18 (https://github.com/numpy/numpy/pull/14051).
    ``_expand_dims`` can be replaced with ``numpy.expand_dims`` when the
    minimum support numpy version is v1.18.

    Parameters
    ----------
    data : array-like
        Input array.
    axis : int or tuple of int
        Position in the expanded axes where the new axis (or axes) is
        placed.  A tuple of axes is now supported.  Out of range axes as
        described above are now forbidden and raise an `AxisError`.

    Returns
    -------
    result : ndarray
        View of ``data`` with the number of dimensions increased.
    """

    if isinstance(data, np.matrix):
        data = np.asarray(data)
    else:
        data = np.asanyarray(data)

    if not isinstance(axis, (tuple, list)):
        axis = (axis,)

    out_ndim = len(axis) + data.ndim
    axis = np.core.numeric.normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(data.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return data.reshape(shape)


# Size, number of elements
def size(a=None):
    """Returns the number of elements"""
    if a is None: return 0
    if type(a) is str:
        return 1
    try:
        return len(a)
    except:
        return np.array(a,ndmin=1).size


# Median Absolute Deviation
def mad(data, axis=None, func=None, ignore_nan=True, zero=False):

    """
    Calculate a robust standard deviation using the `median absolute
    deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation>`_.

    The MAD is defined as ``median(abs(a - median(a)))``.

    This is a straight-up copy of the astropy median_absolute_deviation
    and mad_std() functions but with the addition of the zero keyword.

    Parameters
    ----------
    data : array_like
        Input array or object that can be converted to an array.
    axis : `None`, int, or tuple of ints, optional
        The axis or axes along which the MADs are computed.  The default
        (`None`) is to compute the MAD of the flattened array.
    func : callable, optional
        The function used to compute the median. Defaults to `numpy.ma.median`
        for masked arrays, otherwise to `numpy.median`.
    ignore_nan : bool
        Ignore NaN values (treat them as if they are not in the array) when
        computing the median.  This will use `numpy.ma.median` if ``axis`` is
        specified, or `numpy.nanmedian` if ``axis==None`` and numpy's version
        is >1.10 because nanmedian is slightly faster in this case.
    zero : bool
        Do not subtract the median.  Want the scatter around zero.

    Returns
    -------
    mad : float or `~numpy.ndarray`
        The median absolute deviation of the input array.  If ``axis``
        is `None` then a scalar will be returned, otherwise a
        `~numpy.ndarray` will be returned.

    Examples
    --------
    Generate random variates from a Gaussian distribution and return the
    median absolute deviation for that distribution::

        >>> import numpy as np
        >>> from astropy.stats import median_absolute_deviation
        >>> rand = np.random.RandomState(12345)
        >>> from numpy.random import randn
        >>> mad = median_absolute_deviation(rand.randn(1000))
        >>> print(mad)    # doctest: +FLOAT_CMP
        0.65244241428454486


    """

    #if type(data) is not np.ndarray: raise ValueError("data must be a numpy array")  
    if type(data) is not np.ndarray:
        data = np.array(data)
    
    if func is None:
        # Check if the array has a mask and if so use np.ma.median
        # See https://github.com/numpy/numpy/issues/7330 why using np.ma.median
        # for normal arrays should not be done (summary: np.ma.median always
        # returns an masked array even if the result should be scalar). (#4658)
        if isinstance(data, np.ma.MaskedArray):
            is_masked = True
            func = np.ma.median
            if ignore_nan:
                data = np.ma.masked_where(np.isnan(data), data, copy=True)
        elif ignore_nan:
            is_masked = False
            func = np.nanmedian
        else:
            is_masked = False
            func = np.median
    else:
        is_masked = None

    data = np.asanyarray(data)
    # np.nanmedian has `keepdims`, which is a good option if we're not allowing
    # user-passed functions here
    data_median = func(data, axis=axis)

    # broadcast the median array before subtraction
    if axis is not None:
        data_median = _expand_dims(data_median, axis=axis)  # NUMPY_LT_1_18

    if zero == False:
        result = func(np.abs(data - data_median), axis=axis, overwrite_input=True)
    # Don't subtract the median, want scatter around zero
    else:
        result = func(np.abs(data), axis=axis, overwrite_input=True)    
        
    if axis is None and np.ma.isMaskedArray(result):
        # return scalar version
        result = result.item()
    elif np.ma.isMaskedArray(result) and not is_masked:
        # if the input array was not a masked array, we don't want to return a
        # masked array
        result = result.filled(fill_value=np.nan)

    # Return MAD
    # NOTE: 1. / scipy.stats.norm.ppf(0.75) = 1.482602218505602

    return result * 1.482602218505602

def median(data,axis=None,even=False,high=True,nan=False):
    """
    Return the median of the data.
    This is similar to the numpy version, but it
    does NOT average the central two values if there are
    an even number of elements.

    Parameters
    ----------
    data : numpy array
       The data array to take the median of.
    axis : int, optional
       Take the median along this axis.
    even : bool, optional
       Return the average of the two central values if there
         are an even number of elements.  Default is False.
    high : bool, optional
       If not averaging the two central values, then take
         the higher value.  Default high is True.
    nan : bool, optional
       Ignore NaNs.  Default is False.

    Returns
    -------
    med : float or numpy array
       The median of the data.

    Example
    -------

    med = median(data,axis=0)
    
    By D. Nidever  Nov 2023
    """

    # No axis
    if axis is None:
        iseven = data.size % 2 == 0
    # Along axis
    else:
        iseven = data.shape[axis] % 2 == 0
        
    # Even selected or odd number of elements
    #  use normal numpy median()
    if even or iseven==False:
        if nan:
            return np.nanmedian(data,axis=axis)
        else:
            return np.median(data,axis=axis)

    # Calculate median with no averaging of central
    # two elements.  Use argsort() to do this

    # Ignore the NaNs
    #  np.argsort() puts NaNs at the end of the list
    #  Use np.sum(np.isfinite()) to get the number of
    #  finite points and adjust the indexing accordingly
    
    # No axis
    if axis is None:
        si = np.argsort(data.ravel())        
        npts = len(si)
        if nan:
            npts = np.sum(np.isfinite(data))
        half = npts // 2
        # Pick low or high point of the two middle values        
        if high:
            midind = half
        else:
            midind = half-1
        index = si[midind]
        med = data.ravel()[index]

    # Along axis
    else:
        si = np.argsort(data,axis=axis)
        if nan:
            npts = np.sum(np.isfinite(data),axis=axis)
        else:
            npts = data.shape[axis]
        half = npts // 2
        # Pick low or high point of the two middle values
        if high:
            midind = half
        else:
            midind = half-1
        # Use slice object
        slc = [slice(None)]*data.ndim   # one slice object per dimension
        slc[axis] = midind
        slc = tuple(slc)
        index = si[slc]
        # Add dimension
        newshape = list(data.shape)
        newshape[axis] = 1
        index = index.reshape(newshape)
        med = np.take_along_axis(data,index,axis=axis)
        # Remove extra axis
        newshape = list(data.shape)
        del newshape[axis]
        med = med.reshape(newshape)

    return med

def minmax(a):
    """ Return a 2-element array of minimum and maximum."""
    if type(a) is not np.ndarray:
        a = np.array(a)
    return np.array([np.min(a),np.max(a)])

def stat(a=None,silent=False):
    """ Returns basic statistics on an array."""
    if a is None: raise ValueError("a must be input")
    if type(a) is not np.ndarray:
        a = np.array(a)
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
    info = np.zeros(10,float)
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

def where(statement,comp=False):
    """ Wrapper around numpy.where() to be more like IDL"""
    # If comp=True then the complement will be returned
    gd, = np.where(statement)
    ngd = len(gd)
    if comp:
        bd, = np.where(~statement)
        nbd = len(bd)
        return gd,ngd,bd,nbd
    else:
        return gd,ngd

def glob(inp):
    """ Similar to glob.glob() but allows input of lists."""
    if type(inp) is str:
        return glb(inp)
    elif type(inp) is list:
        out = []
        for l in inp:
            out += glb(l)
        return out
    else:
        raise ValueError(inp,' not understood')

def file_copy(src,dest,overwrite=False):
    """ Copy files."""
    nsrc = size(src)
    if nsrc==1 and type(src) is str:
        src = [src]
    ndest = size(dest)
    if ndest==1 and type(dest) is str:
        dest = [dest]
    # Loop over files
    for i in range(nsrc):
        src1 = src[i]
        dest1 = dest[i]
        if os.path.isdir(src1):
            raise ValueError(src1+' is a directory')
        # Destination is directory, add base name
        if os.path.isdir(dest1):
            dest1 += os.path.basename(src1)
        # Check if the destination file exists already
        exists1 = os.path.exists(dest1)
        if os.path.abspath(src1)==os.path.abspath(dest1):
            print(src1+' and '+dest1+' are the same file')
            continue
        if exists1 and overwrite==False:
            print(dest1+' exists and overwrite=False')
            continue
        if exists1 and overwrite:
            os.remove(exists1)
        # Copy the file
        shutil.copyfile(src1,dest1)

def file_move(src,dest,overwrite=False):
    """ Copy files.  dest can be a filename or directory."""
    nsrc = size(src)
    if nsrc==1 and type(src) is str:
        src = [src]
    ndest = size(dest)
    if ndest==1 and type(dest) is str:
        dest = [dest]
    # Loop over files
    for i in range(nsrc):
        src1 = src[i]
        dest1 = dest[i]
        if os.path.isdir(src1):
            raise ValueError(src1+' is a directory')
        # Destination is directory, add base name
        if os.path.isdir(dest1):
            dest1 += os.path.basename(src1)
        # Check if the destination file exists already
        exists1 = os.path.exists(dest1)
        if os.path.abspath(src1)==os.path.abspath(dest1):
            print(src1+' and '+dest1+' are the same file')
            continue
        if exists1 and overwrite==False:
            print(dest1+' exists and overwrite=False')
            continue
        if exists1 and overwrite:
            os.remove(exists1)
        # Move the file
        shutil.move(src1,dest1)
        
    
def strlen(lst=None):
    """ Calculate the string lengths of a string array."""
    if lst is None: raise ValueError("lst must be input")
    n = size(lst)
    out = np.zeros(n,int)
    for i,a in enumerate(np.array(lst,ndmin=1)):
        out[i] = len(a)
    if n==1: out=int(out)
    return out


def strip(lst=None,chars=None):
    """ Strip on a scalar or list."""
    if lst is None: raise ValueError("lst must be input")
    if type(lst) is str: return lst.strip(chars)
    return [o.strip(chars) for o in np.array(lst,ndmin=1)]


def strjoin(a=None,b=None,c=None,sep=None):
    """ Join two string lists/arrays or scalars"""
    if (a is None) | (b is None): raise ValueError("a and b must be input")
    na = size(a)
    nb = size(b)
    nc = size(c)
    if sep is None: sep=''
    n = np.max([na,nb,nc])
    len1 = strlen(a)
    t1 = type(a)
    len2 = strlen(b)
    t2 = type(b)
    if nc>0:
        len3 = strlen(c)
        t3 = type(c)
    else:
        len3 = 0
        t3 = t2
    nlen = np.max(len1)+np.max(len2)+np.max(len3)+len(sep)
    out = np.zeros(n,(str,nlen))
    for i in range(n):
        if na>1:
            a1 = a[i]
        else:
            a1 = np.array(a,ndmin=1)[0]
        arr = tuple(a1)
        if nb>1:
            b1 = b[i]
        else:
            b1 = np.array(b,ndmin=1)[0]
        arr += tuple(b1)
        if nc>0:
            if nc>1:
                c1 = c[i]
            else:
                c1 = np.array(c,ndmin=1)[0]
            arr += tuple(c1)
        out[i] = sep.join(arr)
    if (n==1) & (t1 is str) & (t2 is str) & (t3 is str): return out[0]  # scalar
    if (t1 is list) | (t2 is list) | (t3 is list): return list(out)
    return out


def strsplit(lst=None,delim=None,asarray=False):
    """ Split a string array."""
    if (lst is None): raise ValueError("lst must be input")
    if size(lst)==1:
        out = lst.split(delim)
    else:
        out = [l.split(delim) for l in lst]
    if asarray is True:
        nlst = np.array(lst).size
        nel = [len(o) for o in out]
        nlen = np.max(strlen(lst))
        outarr = np.zeros((nlst,np.max(nel)),(str,nlen))
        for i in range(nlst):
            temp = np.array(out[i])
            ntemp = len(temp)
            outarr[i,0:ntemp] = temp
        return outarr
    else:
        return out

def pathjoin(indir=None,name=None):
    """ Join two or more pathname components, inserting '/' as needed
    Same as os.path.join but also works on arrays/lists."""
    if indir is None: raise ValueError("must input indir")
    if name is None: raise ValueError("must input name")
    nindir = size(indir)
    nname = size(name)
    n = np.max([nindir,nname])
    len1 = strlen(indir)
    len2 = strlen(name)
    nlen = np.max(len1)+np.max(len2)+1
    out = np.zeros(n,(str,nlen))
    for i in range(n):
        if nindir>1:
            indir1 = indir[i]
        else:
            indir1 = np.array(indir,ndmin=1)[0]
        if indir1[-1] != '/': indir1+='/'
        if nname>1:
            name1 = nname[i]
        else:
            name1 = np.array(name,ndmin=1)[0]
        out[i] = indir1+name1
    if (n==1) & (type(indir) is str) & (type(name) is str): return out[0]  # scalar
    if (type(indir) is list) | (type(name) is list): return list(out)
    return out

def first_el(lst):
    """ Return the first element"""
    if lst is None: return None
    if size(lst)>1: return lst[0]
    if (size(lst)==1) & (type(lst) is list): return lst[0]
    if (size(lst)==1) & (type(lst) is np.ndarray): return lst.item(0)
    return lst
        

# Standard grep function that works on string list
def grep(lines=None,expr=None,index=False):
    """
    Similar to the standard unix "grep" but run on a list of strings.
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
    cnt = 0
    for l in np.array(lines,ndmin=1):
        m = re.search(expr,l)
        if m != None:
            if index is False:
                out.append(l)
            else:
                out.append(cnt)
        cnt = cnt+1
    return out

# find() function for arrays
def find(arr,expr):
    """
    The standard python string find() but for lists or np.char.array arrays.
    Note that this uses re so it understands regular expressions.

    Parameters
    ----------
    arr : list or np.char.array
        The list of string lines to check.
    expr  : str
        Scalar string expression to search for.  Can be a regular expression.

    Returns
    -------
    out : numpy array
        The array of indices.

    Example
    -------

    Search for a string and return the matching lines:

    .. code-block:: python

        mlines = find(arr,"hello")

    """
    out = []
    for l in np.array(arr,ndmin=1):
        m = re.search(expr,l)
        if m != None:
            out.append(m.start())
        else:
            out.append(-1)
    return np.array(out).astype(int)

# Create an empty file
def touch(fname):
    open(fname, 'a').close()


# Read in all lines of files
def readlines(fil=None,comment=None,raw=False,nreadline=None,noblank=False):
    """
    Read in all lines of a file.
    
    Parameters
    ----------
    file : str
         The name of the file to load.  This can be a gzipped file.
    comment : str
         Comment line character to ignore (e.g., "#").
    raw : bool, optional, default is false
         Do not trim \n off the ends of the lines.
    nreadline : int, optional
         Read only this number of lines.  Default is to read all lines.
    noblank : boolean, optional
         Remove blank lines or lines with only whitespace.  Default is False.

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
    # Read gzipped file
    if fil.endswith('.gz'):
        fp = gzip.open(fil)
        contents = fp.read() # contents now has the uncompressed bytes of foo.gz
        fp.close()
        lines = contents.decode('utf-8') # u_str is now a unicode string
        lines = lines.split('\n')
    # Read normal ASCII file
    else:
        if nreadline is None:
            with open(fil,'r') as f:
                lines = f.readlines()
        else:
            with open(fil,'r') as f:
                lines = []
                for i in range(nreadline):
                    lines.append( f.readline() )
    # Remove blank lines
    if noblank:
        lines = [l for l in lines if l.strip()!='']
    # Strip newline off
    if raw is False: lines = [l.rstrip('\n') for l in lines]
    # Check for comment string:
    if comment is not None:
        lines = [l for l in lines if l.startswith(comment)==False]
    return lines


# Write all lines to file
def writelines(filename=None,lines=None,overwrite=True,raw=False):
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
    raw : bool, optional, default is False
        Do not modify the lines. Write out as is.

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
    # Modify the input as needed
    if raw is False:
        # List, make sure it ends with \n
        if type(lines) is list:
            for i,l in enumerate(lines):
                if l.endswith('\n') is False:
                    lines[i] += '\n'
            # Make sure final element does not end in \n
            #n = size(lines)
            #if n>1:
            #    if lines[-1].endswith('\n'):
            #        lines[-1] = lines[-1][0:-1]
            #else:
            #    if lines[0].endswith('\n'):
            #        lines = lines[0][0:-1]
    # Convert string to list
    if (type(lines) is str): lines=list(lines)
    # Convert numpy array and numbers to list of strings
    if type(lines) is not list:
        if hasattr(lines,'__iter__'):
            lines = [str(l)+'\n' for l in lines]
            # Make sure final element does not end in \n        
            #if lines[-1].endswith('\n'): lines[-1] = lines[-1][0:-1]        
        else:
            lines = str(lines)
    # Write the file
    f = open(filename,'w')
    f.writelines(lines)
    f.close()

def loadinput(inp,comment='#',exist=False):
    """
    PURPOSE:
    This program can be used to load command-line inputs.
    The input can be:
    (1) an array list of files, i.e. ['one.txt','two.txt']
    (2) a globbed list, i.e. "*.txt"
    (3) a comma separated list, i.e.  'one.txt,two.txt'
    (4) the name of a file that contains a list, i.e. "list.txt"
       This can NOT be used in combination with any of the other three options.

    INPUTS:
      input    The input list.  There are three possibilities
               (1) an array list of files
               (2) a globbed list, i.e. "*.txt"
               (3) a comma separated list, i.e.  'one.txt,two.txt'
               (4) the name of a file that contains a list, i.e. "@list.txt"
                 This can NOT be used in combination with any of the other three options.
      =comment Comment string to use when loading a list file. By default
                comment='#'
      /exist   Files must exist

    OUTPUTS:
      list     The list of files
      =count   The number of elements in list

    USAGE:
    list = loadinput('*.fits')

    By D.Nidever   April 2007

    """

    if type(inp) is not list:
        inp = [inp]

    # Check if this is a file I should read in
    if inp[0][0]=='@':
        fil = inp[0][1:]
        out = readlines(fil,comment=comment)
    else:
        # Break up comma-delimited list        
        lst = []
        for l in inp:
            if l.find(','):
                l = l.split(',')
                lst += l
            else:
                lst.append(l)
        # Glob
        out = []
        for l in lst:
            if l.find('*')>-1 or l.find('?')>-1:
                gl = glob(l)
                if len(gl)>0:
                    out += gl
            else:
                out.append(l)

    # Files must exist
    if exist:
        out = out[exists(out)]

    return out
    

# Remove indices from a list
def remove_indices(lst=None,index=None):
    """
    This will remove elements from a list given their indices.
    Use numpy.delete() for numpy arrays instead.

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
    if type(lst) is np.ndarray: newlst = np.array(newlst)
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
    try:
        with open(fil, "r") as f:
            return (sum(bl.count("\n") for bl in blocks(f)))
    except UnicodeDecodeError:
        with open(fil,"rb") as f:
            return f.read().count(b'\n')
    except:
        traceback.print_exc()
        
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


# Delete files
def remove(files=None,allow=True):
    """ Delete a list of files."""
    if files is None: raise ValueError("No files input")
    nfiles = np.array(files).size
    for f in np.array(files,ndmin=1):
        if os.path.exists(f):
            os.remove(f)
        else:
            if allow is False: raise Exception(f+" does not exist")

# Do files exist
def exists(files=None):
    """ Check if a list of files exist."""
    if files is None: raise ValueError("No files input")
    nfiles = np.array(files).size
    out = np.zeros(nfiles,bool)+False
    for i,f in enumerate(np.array(files,ndmin=1)):
        if os.path.exists(f): out[i] = True
    return out

def lt(x,limit):
    """Takes the lesser of x or limit"""
    # np.minimum() also does this
    if np.array(x).size>1:
        out = [i if (i<limit) else limit for i in x]
    else:
        out = x if (x<limit) else limit
    if type(x) is np.ndarray: return np.array(out)
    return out
    
def gt(x,limit):
    """Takes the greater of x or limit"""
    # np.maximum() also does this
    if np.array(x).size>1:
        out = [i if (i>limit) else limit for i in x]
    else:
        out = x if (x>limit) else limit
    if type(x) is np.ndarray: return np.array(out)
    return out        

def limit(x,llimit,ulimit):
    """Require x to be within upper and lower limits"""
    # np.clip() does this
    return lt(gt(x,llimit),ulimit)

def valrange(array):
    if size(array)==1:
        return 0.0
    else:
        return np.max(array)-np.min(array)

def signs(inp):
    """ Return the sign of input.  Return +1.0 for 0.0"""
    s = np.sign(inp)
    bad,nbad = where(s== 0)
    if nbad>0:
        if size(s)>1:
            s[bad] = 1
        else:
            s = 1.0
    return s

def scale(arr,oldrange,newrange):
    """
    This function maps an array or image onto a new
    scale given two points on the old scale and
    the corresponding points on the new scale.
    The array is converted to double type.
    It's similar to BYTSCL.PRO except that you
    can set the bottom value as well.
    The ranges can be increasing or decreasing.

    INPUTS:
    arr      The array of values to be scaled
    oldrange Two-element array specifiying The original range which
               will be scaled to newrange.
    newrange Two-element array specifiying The new range which
               the oldrange will be scaled to.

    OUTPUTS:
    narr     The new scaled array

    USAGE:
    arr2 = scale(arr,[0,1],[150,2000])

    By D.Nidever   March 2007
    """

    if len(newrange)!=2:
        raise ValueError("newrange must be a 2-element array or list")
    if len(oldrange)!=2:
        raise ValueError("oldrange must be a 2-element array or list")
    
    # Does it flip around
    signchange = 1.0
    if signs(oldrange[1]-oldrange[0]) != signs(newrange[1]-newrange[0]):
        signchange = -1.0 
    # Scale
    narr = valrange(newrange) * signchange*(np.float64(arr)-oldrange[0])/valrange(oldrange) + newrange[0]
    return narr
    
def scale_vector(vector, minrange, maxrange):
    """ Scale a vector to minrange and maxrange. """

    # Make sure we are working with floating point numbers.
    minRange = np.float64( minrange )
    maxRange = np.float64( maxrange )

    # Make sure we have a valid range.
    if (maxRange == minrange):
        raise ValueError("Range max and min are coincidental")
        return vector*0+minrange

    vectormin = np.float64(np.min(vector))
    vectormax = np.float64(np.max(vector))
    
    # Calculate the scaling factors.
    scaleFactor = [((minrange * vectormax)-(maxrange * vectormin)) /
                   (vectormax - vectormin), (maxrange - minrange) / (vectormax - vectormin)]

    # Return the scaled vector.
    return vector * scaleFactor[1] + scaleFactor[0]
    

def quadratic_bisector(x,y):
    """ Calculate the axis of symmetric or bisector of parabola"""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    n = len(x)
    if n<3:
        return None
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
    Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
    Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
    Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
    #a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    #b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    denom = Sxx*Sx2x2 - Sxx2**2
    if denom==0:
        return np.nan
    a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
    b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
    if a==0:
        return np.nan
    return -b/(2*a)

def quadratic_coefficients(x,y,axis=0):
    """ Calculate the quadratic coefficients from the three points."""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    # y = ax**2 + b*x + c
    n = len(x)
    if np.array(x).ndim > 1:
        n = np.array(x).shape[axis]
            
    if n<3:
        return None
    Sxx = np.sum(x**2,axis=axis) - np.sum(x,axis=axis)**2/n
    Sxy = np.sum(x*y,axis=axis) - np.sum(x,axis=axis)*np.sum(y,axis=axis)/n
    Sxx2 = np.sum(x**3,axis=axis) - np.sum(x,axis=axis)*np.sum(x**2,axis=axis)/n
    Sx2y = np.sum(x**2 * y,axis=axis) - np.sum(x**2,axis=axis)*np.sum(y,axis=axis)/n
    Sx2x2 = np.sum(x**4,axis=axis) - np.sum(x**2,axis=axis)**2/n
    #a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    #b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    denom = Sxx*Sx2x2 - Sxx2**2
    if np.array(x).ndim==1:
        if denom==0:
            coef = np.zeros(3,float)+np.nan
            return coef
    else:
        bad = (denom==0)
        denom[bad] = 1
    a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
    b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
    if np.array(x).ndim==1:
        c = np.median(y - (a*x**2+b*x),axis=axis)
    else:
        newshape = [-1,-1]
        newshape[axis] = 1
        c = np.median(y - (a.reshape(newshape)*x**2+b.reshape(newshape)*x),axis=axis)        
    if np.array(x).ndim==1:
        coef = np.zeros(3,float)
        coef[:] = [a,b,c]
    else:
        xshape = np.array(x).shape
        if axis==0:
            nx = xshape[1]
        else:
            nx = xshape[0]
        coef = np.zeros((3,nx),float)
        coef[0,:] = a
        coef[1,:] = b
        coef[2,:] = c        
        if np.sum(bad)>0:
            coef[0,bad] = np.nan
            coef[1,bad] = np.nan
            coef[2,bad] = np.nan            
        
    return coef

def linear_coefficients(x,y,silent=True):
    """ Calculate the lienar coefficients from two points."""

    # y = mx + b

    m = (y[1]-y[0])/(x[1]-x[0])
    b = y[0] - m*x[0]

    if silent==False:
        if b<0:
            print('y = '+str(m)+'*x - '+str(np.abs(b)))
        else:
            print('y = '+str(m)+'*x + '+str(b))            
    return m,b
    
def wtmean(x,sigma,error=False,reweight=False,magnitude=False):
    """ Calculate weighted mean and error"""
    # np.average() can take weights as well.
    n = len(x)
    wt = 1/sigma**2
    # Magnitudes
    if magnitude:
        fmn = np.sum( 2.5118864**x * wt) / np.sum(wt)
        xmn = 2.50*np.log10(fmn)
    else:
        xmn = np.sum(wt*x)/np.sum(wt)
    # Reweight the points based on the residuals
    #  using formula similar to the one given by
    #  Stetson (1996) pg.4
    if reweight:
        if magnitude:
            resid = x-xmn
            wt2 = wt/(1+np.abs(resid)**2/np.mean(sigma))            
            fmn2 = np.sum( 2.5118864**x * wt2) / np.sum(wt2)
            xmn = 2.50*np.log10(fmn2)
        else:
            resid = x-xmn
            wt2 = wt/(1+np.abs(resid)**2/np.mean(sigma))
            xmn = np.sum(wt2*x)/np.sum(wt2)
    # Include uncertainty
    if error:
        if magnitude:
            xerr = np.sqrt(1.0/np.sum(wt))
        else:
            xerr = np.sqrt( np.sum( ((x-xmn)**2)*wt)*n / ((n-1)*np.sum(wt))) / np.sqrt(n)
        return xmn,xerr
    else:
        return xmn

    
def mediqrslope(x,y):
    """ Calculate robust slope.  Calculate the slopes of all points in the 3+4th quartile using
        the median X/Y values of the 1st quartile points, and then the same with 1+2nd quartile
        with the median X/Y values of th 4th quartile points.  Then median is found of all the
        slopes."""
    xx = np.array(x).ravel()
    yy = np.array(y).ravel()
    n = len(xx)
    si = np.argsort(xx)
    nh = n//2
    nq = n//4
    # First quartile median X/Y values
    x1 = np.median(xx[si[0:nq]])
    y1 = np.median(yy[si[0:nq]])
    slp34 = (yy[si[nh:]]-y1)/(xx[si[nh:]]-x1)
    # Fourth quartile median X/Y values    
    x4 = np.median(xx[si[-nq:]])
    y4 = np.median(yy[si[-nq:]])
    slp12 = (yy[si[0:nh]]-y4)/(xx[si[0:nh]]-x4)
    allslp = np.hstack((slp12,slp34))
    slp = np.median(allslp)
    return slp    

    
def iqrslope(x,y):
    """ Calculate robust slope using median X/Y values of first quartile
         and last quartile points."""
    xx = np.array(x).ravel()
    yy = np.array(y).ravel()
    n = len(xx)
    si = np.argsort(xx)
    nq = n//4
    x1 = np.median(xx[si[0:nq]])
    y1 = np.median(yy[si[0:nq]])
    x2 = np.median(xx[si[-nq:]])
    y2 = np.median(yy[si[-nq:]])
    slp = (y2-y1)/(x2-x1)
    return slp    

def medslope(x,y):
    """ Calculate robust slope using median X/Y values of first half
         and second half of sorted points."""
    xx = np.array(x).ravel()
    yy = np.array(y).ravel()
    n = len(xx)
    si = np.argsort(xx)
    nh = n//2
    x1 = np.median(xx[si[0:nh]])
    y1 = np.median(yy[si[0:nh]])
    x2 = np.median(xx[si[nh:]])
    y2 = np.median(yy[si[nh:]])
    slp = (y2-y1)/(x2-x1)
    return slp
    
def wtslope(x,y,sigma,error=False,reweight=False):
    """ Calculate weighted slope and error"""
    n = len(x)
    wt = 1/sigma**2
    totwt = np.sum(wt)
    mnx = np.sum(wt*x)/totwt
    mny = np.sum(wt*y)/totwt
    wtx =  (np.sum(wt*x*y)/totwt-mnx*mny)/(np.sum(wt*x**2)/totwt-mnx**2)
    # Reweight the points based on the residuals
    #  using formula similar to the one given by
    #  Stetson (1996) pg.4
    if reweight:
        resid = y-wtx*x
        resid -= np.mean(resid)
        wt2 = wt/(1+np.abs(resid)**2/np.mean(sigma))
        totwt2 = np.sum(wt2)
        mnx2 = np.sum(wt2*x)/totwt2
        mny2 = np.sum(wt2*y)/totwt2
        wtx =  (np.sum(wt2*x*y)/totwt2-mnx2*mny2)/(np.sum(wt2*x**2)/totwt2-mnx2**2)
    if error:
        wtxerr = 1.0/np.sqrt( np.sum(wt*x**2)-mnx**2 * np.sum(wt))
        return wtx, wtxerr
    else:
        return wtx

def robust_slope_old(x,y,sigma,limits=None,npt=15,reweight=False):
    """ Calculate robust weighted slope"""
    # Maybe add sigma outlier rejection in the future
    n = len(x)
    if n==2:
        return wtslope(x,y,sigma,error=True,reweight=reweight)
    # Calculate weighted pmx/pmxerr
    wt_slp,wt_slperr = wtslope(x,y,sigma,error=True,reweight=reweight)
    wt_y, wt_yerr = wtmean(y,sigma,error=True,reweight=reweight)
    # Unweighted slope
    uwt_slp = wtslope(x,y,sigma*0+1,reweight=reweight)
    # Calculate robust loss metric for range of slope values
    #   chisq = Sum( abs(y-(x*slp-mean(x*slp)))/sigma )
    if limits is None:
        limits = np.array([np.min([0.5*wt_slp,0.5*uwt_slp]), np.max([1.5*wt_slp,1.5*uwt_slp])])
    slp_step = (np.max(limits)-np.min(limits))/(npt-1)
    slp_arr = np.arange(npt)*slp_step + np.min(limits)
    # Vectorize it
    resid = np.outer(y,np.ones(npt))-np.outer(x,np.ones(npt))*np.outer(np.ones(n),slp_arr)
    mnresid = np.mean(resid,axis=0)
    resid -= np.outer(np.ones(n),mnresid)    # remove the mean
    chisq = np.sum( np.abs(resid) / np.outer(sigma,np.ones(npt)) ,axis=0)
    bestind = np.argmin(chisq)
    best_slp = slp_arr[bestind]
    # Get parabola bisector
    lo = np.maximum(0,bestind-2)
    hi = np.maximum(bestind+2,n)
    quad_slp = quadratic_bisector(slp_arr[lo:hi],chisq[lo:hi])
    # Problem with parabola bisector, use best point instead
    if np.isnan(quad_slp) | (np.abs(quad_slp-best_slp)> slp_step):
        best_slp = best_slp
    else:
        best_slp = quad_slp
    return best_slp, wt_slperr
    
def robust_slope(x,y,sigma,limits=None,npt=15,reweight=False):
    """ Calculate robust weighted slope"""
    # Maybe add sigma outlier rejection in the future
    n = len(x)
    if n==2:
        return wtslope(x,y,sigma,error=True,reweight=reweight)
    # Calculate weighted pmx/pmxerr
    wt_slp,wt_slperr = wtslope(x,y,sigma,error=True,reweight=reweight)
    wt_y, wt_yerr = wtmean(y,sigma,error=True,reweight=reweight)
    # Unweighted slope
    uwt_slp = wtslope(x,y,sigma*0+1,reweight=reweight)
    # Calculate robust loss metric for range of slope values
    #   chisq = Sum( abs(y-(x*slp-mean(x*slp)))/sigma )
    if limits is None:
        limits = np.array([np.min([0.5*wt_slp,0.5*uwt_slp]), np.max([1.5*wt_slp,1.5*uwt_slp])])
    slp_step = (np.max(limits)-np.min(limits))/(npt-1)
    slp_arr = np.arange(npt)*slp_step + np.min(limits)
    # Vectorize it
    resid = np.outer(y,np.ones(npt))-np.outer(x,np.ones(npt))*np.outer(np.ones(n),slp_arr)
    mnresid = np.mean(resid,axis=0)
    resid -= np.outer(np.ones(n),mnresid)    # remove the mean
    chisq = np.sum( np.abs(resid) / np.outer(sigma,np.ones(npt)) ,axis=0)
    bestind = np.argmin(chisq)
    best_slp = slp_arr[bestind]
    # Get parabola bisector
    # fixed two bugs on 08/01/20 that biased the results
    #  np.minimum was a np.maximum and lo:hi+1 was missing the +1
    lo = np.maximum(0,bestind-2)
    hi = np.minimum(bestind+2,npt-1)
    quad_slp = quadratic_bisector(slp_arr[lo:hi+1],chisq[lo:hi+1])
    # Problem with parabola bisector, use best point instead
    if np.isnan(quad_slp) | (np.abs(quad_slp-best_slp)> slp_step):
        best_slp = best_slp
    else:
        best_slp = quad_slp
    return best_slp, wt_slperr

def wtmedian(val,wt):
    """Weighted median can be computed by sorting the set of numbers and finding the
    smallest numbers which sums to half the weight of total weight."""
    # https://en.wikipedia.org/wiki/Weighted_median
    si = np.argsort(val.flatten())
    totwt = np.cumsum(np.abs(wt).flatten()[si])
    ind = totwt.searchsorted(totwt.max()*0.5)
    return val.flatten()[si[ind-1]]

def iqrstdev(data):
    """ Use the interquartile range to estimate the standard deviation robustly."""
    val = np.percentile(data,[25,75])
    iqr = val[1]-val[0]
    # for a normally distributed dataset we should have
    # Q1 = -0.67*sigma + mean
    # Q3 = 0.67*sigma + mean
    # Sigma = (Q3-Q1)/1.34
    sigma = iqr/1.34
    return sigma

def sigclipmean(data,nsig=2.5):
    """ Sigma-clipped mean."""
    fnt = np.isfinite(data)
    med = np.median(data[fnt])
    sig = mad(data[fnt])
    good, = np.where(np.abs(data[fnt]-med) < nsig*sig)
    mn = np.mean(data[fnt][good])
    return mn

def gausswtmean(data,sig=None):
    """ Compute weighted mean using a Gaussian with center of the median and sigma of the MAD (or input)."""
    # try sqrt() of Gaussian
    fnt = np.isfinite(data)
    med = np.median(data[fnt])
    if sig is None:
        sig = mad(data[fnt])
    # sqrt(gaussian) to not downweight outliers so much
    wt = np.exp(-0.25*(data[fnt]-med)**2/sig**2)
    totwt = np.sum(wt)
    mn = np.sum(wt*data[fnt])/totwt
    return mn

def gmean(data,weights=None):
    """ Compute geometric mean."""
    # can add an array of weights as well
    return mstats.gmean(data,weights=weights)

def running_mean(x, N):
    """ Computing running mean."""
    #https://stackoverflow.com/questions/13728392/moving-average-or-running-mean    
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def skewquartile(data):
    """ Measure the skewness robustly based on quartiles."""

    # Get quartiles, from Beaumont's quartile.pro function
    quarts = np.percentile(data,[25,50,75])
    q1 = quarts[0]
    q2 = quarts[1]
    q3 = quarts[2]
    skew = ( q1 - 2*q2 + q3 ) / ( q3 - q1 )
    return skew

def skewgauss(x,par):
    """ Return a skewed Gaussian."""
    # This is a skewed Gaussian
    #  See http://en.wikipedia.org/wiki/Skew_normal_distribution
    # par = [height, center, sigma, alpha]
    # alpha is the skewness of the Gaussian

    ht = par[0]
    cen = par[1]
    sig = par[2]
    alpha = par[3]

    x2 = (x-cen)/sig  # rescale 
    y = ht * np.exp(-0.5 * x2**2 ) * 0.5*(1 + erf(alpha*x2/np.sqrt(2)))

    return y

def gaussian(x, amp, cen, sig, const=0.0, slp=0.0):
    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    #return (amp / (np.sqrt(2*np.pi) * sig)) * np.exp(-(x-cen)**2 / (2*sig**2)) + const
    return amp * np.exp(-(x-cen)**2 / (2*sig**2)) + const + slp*(x-cen)

def gaussbin(x, amp, cen, sig, const=0, slp=0.0, dx=1.0):
    """1-D gaussian with pixel binning
    
    This function returns a binned Gaussian
    par = [height, center, sigma]
    
    Parameters
    ----------
    x : array
       The array of X-values.
    amp : float
       The Gaussian height/amplitude.
    cen : float
       The central position of the Gaussian.
    sig : float
       The Gaussian sigma.
    const : float, optional, default=0.0
       A constant offset.
    slp : float, optional, default=0.0
       A linear slope around cen.
    dx : float, optional, default=1.0
      The width of each "pixel" (scalar).
    
    Returns
    -------
    geval : array
          The binned Gaussian in the pixel

    """

    xcen = np.array(x)-cen             # relative to the center
    x1cen = xcen - 0.5*dx  # left side of bin
    x2cen = xcen + 0.5*dx  # right side of bin

    t1cen = x1cen/(np.sqrt(2.0)*sig)  # scale to a unitless Gaussian
    t2cen = x2cen/(np.sqrt(2.0)*sig)

    # For each value we need to calculate two integrals
    #  one on the left side and one on the right side

    # Evaluate each point
    #   ERF = 2/sqrt(pi) * Integral(t=0-z) exp(-t^2) dt
    #   negative for negative z
    geval_lower = erf(t1cen)
    geval_upper = erf(t2cen)

    geval = amp*np.sqrt(2.0)*sig * np.sqrt(np.pi)/2.0 * ( geval_upper - geval_lower )
    geval += const + slp*(x-cen)   # add constant offset and slope

    return geval

def gaussfit(x,y,initpar=None,sigma=None, bounds=None, binned=False):
    """Fit 1-D Gaussian to X/Y data"""
    #gmodel = Model(gaussian)
    #result = gmodel.fit(y, x=x, amp=initpar[0], cen=initpar[1], sig=initpar[2], const=initpar[3])
    #return result
    if initpar is None:
        initpar = [np.max(y),x[np.argmax(y)],1.0,np.median(y)]
    if bounds is None:
        bounds = (-np.inf,np.inf)
    func = gaussian
    if binned is True: func=gaussbin
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)


def voigt(x, height, cen, sigma, gamma, const=0.0, slp=0.0):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian sigma.

    """

    maxy = np.real(wofz((1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
    return (height/maxy) * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi) + const + slp*(x-cen)

def voigtfit(x,y,initpar=None,sigma=None,bounds=(-np.inf,np.inf)):
    """Fit a Voigt profile to data."""
    if initpar is None:
        initpar = [np.max(y),x[np.argmax(y)],1.0,1.0,np.median(y),0.0]
    func = voigt
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)

def voigtarea(pars):
    """ Compute area of Voigt profile"""
    sig = np.maximum(pars[2],pars[3])
    x = np.linspace(-20*sig,20*sig,1000)+pars[1]
    dx = x[1]-x[0]
    v = voigt(x,np.abs(pars[0]),pars[1],pars[2],pars[3])
    varea = np.sum(v*dx)
    return varea

def poly(x,coef,*args):
    """ Evaluate a polynomial function of a variable."""
    # p(x) = p[0] * x**deg + ... + p[deg]
    y = np.array(x).copy()*0.0
    # concatenate coefficients
    if len(args)>0:
        coef = np.hstack((coef,np.array(args)))
    n = len(coef)
    for i in range(n):
        y += coef[i]*x**(n-1-i)
    return y

def poly_resid(coef,x,y,sigma=1.0):
    sig = sigma
    if sigma is None: sig=1.0
    return (poly(x,coef)-y)/sig

def poly_fit(x,y,nord,robust=False,sigma=None,initpar=None,bounds=(-np.inf,np.inf),error=False,max_nfev=None):
    if initpar is None: initpar = np.zeros(nord+1,float)
    # Normal polynomial fitting
    #if sigma is None: sigma=np.zeros(len(x))+1
    #coef, cov = curve_fit(poly, x, y, p0=initpar, sigma=sigma, bounds=bounds)
    #perr = np.sqrt(np.diag(cov))
    #return coef, perr

    #weights = None
    #if sigma is not None: weights=1/sigma**2
    #if error:
    #    if len(x)>nord+3:
    #        coef, cov = np.polyfit(x,y,nord,w=weights,cov='unscaled')
    #        perr = np.sqrt(np.diag(cov))
    #    else:
    #        coef = np.polyfit(x,y,nord,w=weights)
    #        perr = coef.copy()*0.0
    #
    #    return coef, perr
    #else:
    #    coef = np.polyfit(x,y,nord,w=weights)
    #    return coef

    if robust==True:
        loss = 'soft_l1'
        f_scale = 0.1
    else:
        loss = 'linear'
        f_scale = 1.0
    if sigma is None: sigma=np.zeros(len(x),float)+1
    # using jac='3-point' seems to improve the results a lot!
    res = least_squares(poly_resid, initpar, loss=loss, f_scale=f_scale, args=(x,y,sigma), max_nfev=max_nfev, jac='3-point')
    if res.success is False:
        print("Problem with least squares polynomial fitting. Status="+str(res.status)+" Trying np.polyfit instead.")
        # Try np.polyfit
        if error:
            coef,cov = np.polyfit(x,y,nord,w=1/sigma**2,cov=True)
            perr = np.sqrt(np.diag(cov))
            return coef,perr
        else:
            coef = np.polyfit(x,y,nord,w=1/sigma**2)
            return coef
    coef = res.x
    
    # Calculate the covariance matrix
    #  this is how scipy.optimize.curve_fit computes the covariance matrix
    #  https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739
    if error:
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        # Compute errors on the parameters
        perr = np.sqrt(np.diag(pcov))
        return coef, perr
    else:
        return coef

# Derivative or slope of an array
def slope(array):
    """Derivative or slope of an array: slp = slope(array)"""
    n = len(array)
    return array[1:n]-array[0:n-1]


def smooth(y, width, fillvalue=np.nan):
    """ Smooth a curve"""
    if y.ndim==1:
        cumsum_vec = np.cumsum(np.insert(y, 0, 0)) 
        ma_vec = (cumsum_vec[width:] - cumsum_vec[:-width]) / width
        #box = np.ones(box_pts)/box_pts
        #y_smooth = np.convolve(y, box, mode='same')
    else:
        # Do each axis separately
        # https://gist.github.com/kwinkunks/769e39e8314b5479842a77b18e4e3eda
        kernel = np.ones(width) / width
        def convfunc(arr1d):
            return np.convolve(arr1d, kernel, mode='same')
        first_pass = np.apply_along_axis(convfunc, axis=0, arr=y)
        ma_vec = np.apply_along_axis(convfunc, axis=1, arr=first_pass)
    return ma_vec

def boxcar(y, box_pts,boundary='wrap'):
    """ Boxcar smooth a 1-D or 2-D array."""
    if y.ndim==1:
        kernel = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, kernel, mode='same')
    else:
        if np.array(box_pts).size == 0:
            kernel = np.ones([box_pts,box_pts])/box_pts**2
        elif np.array(box_pts).size == 1:
            kernel = np.ones([box_pts[0],box_pts[0]])/box_pts[0]**2	   
        else:
            kernel = np.ones(box_pts)/(box_pts[0]*box_pts[1])

	# scipy.signal.convolve2d() does nothing if one of the dimensions                                                     
        #  has size=1                                                                                                         
        if kernel.shape[0]>1 and kernel.shape[1]>1:
            from scipy.signal import convolve2d
            y_smooth = convolve2d(y,kernel,mode='same',boundary=boundary)
        else:
            width = np.max(np.array(box_pts))
            kernel1 = np.ones(width) / width
            def convfunc(arr1d):
                return np.convolve(arr1d, kernel1, mode='same')
            if kernel.shape[0]==1:
                y_smooth = np.apply_along_axis(convfunc, axis=1, arr=y)
            else:
                y_smooth = np.apply_along_axis(convfunc, axis=0, arr=y)

    return y_smooth

# Gaussian filter
def gsmooth(data,fwhm,mask=None,boundary='extend',fill=0.0,truncate=4.0,squared=False):
    # astropy.convolve automatically ignores NaNs
    # Create kernel
    xsize = np.ceil(fwhm/2.35*truncate*2)
    if data.ndim==1:
        if xsize % 2 == 0: xsize+=1   # must be odd        
        g = Gaussian1DKernel(stddev=fwhm/2.35,x_size=xsize)
    else:
        if size(fwhm)==1:
            if xsize % 2 == 0: xsize+=1   # must be odd            
            g = Gaussian2DKernel(fwhm/2.35,x_size=xsize)
        else:
            if xsize[0] % 2 == 0: xsize[0]+=1   # must be odd
            if xsize[1] % 2 == 0: xsize[1]+=1   # must be odd              
            g = Gaussian2DKernel(fwhm[0]/2.35,fwhm[1]/2.35,x_size=xsize)            
    if squared is False:
        return convolve(data, g.array, mask=mask, boundary=boundary, fill_value=fill)
        #return gaussian_filter1d(data,fwhm/2.35,axis=axis,mode=mode,cval=cval,truncate=truncate)
    else:
        return convolve(data, g.array**2, mask=mask, boundary=boundary, fill_value=fill, normalize_kernel=False)
        #return gaussian_filter1d(data,fwhm/2.35,axis=axis,mode=mode,cval=cval,truncate=truncate)**2

def savgol(y,nbin,order=3):
    """ Smooth data with Savitzky-Golay filter."""
    if nbin % 2 == 0:  # width must be odd
        nbin += 1
    yhat = savgol_filter(y, nbin, order) # window size 51, polynomial order 3
    return yhat
        
# Rebin data
def rebin(arr,new_shape=None,binsize=None,tot=False,med=False,maximum=False,
          bitwiseor=False):
    """
    Rebin data in 1D or 2D

    Parameters
    ----------
    arr : numpy array
       The data array to rebin.
    new_shape : tuple or list, optional
       Tuple or list of new output shape.  Either new_shape or binsize must be input.
    binsize : tuple or list, optional
       Tuple or list of binsize.  Either new_shape or binsize must be input. 
    tot : boolean, optional
       Return the sum instead of the mean (the default).
    med : boolean, optional
       Return the median instead of the mean.
    maximum : boolean, optional
       Return the maximum instead of the mean.
    bitwiseor : boolean, optional
       Return the bitwise OR of the binned data.

    Returns
    -------
    out : numpy array
       The rebinned data.

    Example
    -------

    out = rebin(arr,(100,300))

    """
    if arr.ndim>2:
        raise Exception("Maximum 2D arrays")
    if arr.ndim==0:
        raise Exception("Must be an array")
    if new_shape is None and binsize is None:
        raise Exception("new_shape or binsize must be input")
    
    if arr.ndim==2:
        if binsize is None:
            binsize = (arr.shape[0] // new_shape[0], arr.shape[1] // new_shape[1])
        else:
            new_shape = (arr.shape[0]//binsize[0], arr.shape[1]//binsize[1])
        shape = (new_shape[0], binsize[0],
                 new_shape[1], binsize[1])
        slc = (slice(0,new_shape[0]*binsize[0],None),slice(0,new_shape[1]*binsize[1],None))
        # Median
        if med:
            out = np.median(arr[slc].reshape(shape),axis=(1,3))
        # Sum
        elif tot:
            out = arr[slc].reshape(shape).sum(-1).sum(1)
        # Maximum
        elif maximum:
            out = arr[slc].reshape(shape).max(-1).max(1)
        # Bitwise OR
        elif bitwiseor:
            out = np.bitwise_or.reduce(arr[slc].reshape(shape),axis=1)
            out = np.bitwise_or.reduce(out,axis=-1)
        # Mean
        else:
            out = arr[slc].reshape(shape).mean(-1).mean(1)
    
    elif arr.ndim==1:
        if binsize is None:
            binsize = arr.shape[0] // np.array(new_shape,ndmin=1)[0]
        else:
            new_shape = arr.shape[0] // binsize
        shape = (np.array(new_shape,ndmin=1)[0], binsize)
        slc = slice(0,np.array(new_shape,ndmin=1)[0]*binsize,None)
        # Median
        if med:
            out = np.median(arr[slc].reshape(shape),axis=1)
        # Sum
        elif tot:
            out = arr[slc].reshape(shape).sum(-1)
        # Maximum
        elif maximum:
            out = arr[slc].reshape(shape).max(-1)
        # Bitwise OR
        elif bitwiseor:
            out = arr[slc].reshape(shape)
            out = np.bitwise_or.reduce(out,axis=-1)
        # Mean
        else:
            out = arr[slc].reshape(shape).mean(-1)

    else:
        raise NotImplementedError('Only 1-D and 2-D arrays supported so far')

    return out
    
def roi_cut(xcut,ycut,x,y):
    """
    Use cuts in a 2D plane to select points from arrays.

    Parameters
    ----------
    xcut : numpy array
         Array of x-values for the cut
    ycut : numpy array
         Array of y-values for the cut
    x : numpy array or list
         Array of x-values that should be cut
    y : numpy array or list
         Array of y-values that should be cut

    Returns
    -------
    ind : numpy array
       The indices of values OUTSIDE the cut
    cutind : 
       The indices of values INSIDE the cut

    Example
    -------

    .. code-block:: python

        ind, cutind = roi_cut(xcut,ycut,x,y)

    """

    from matplotlib.path import Path

    tupVerts = list(zip(xcut,ycut))

    points = np.vstack((x,y)).T
    
    p = Path(tupVerts) # make a polygon
    inside = p.contains_points(points)

    ind, = np.where(~inside)
    cutind, = np.where(inside)

    return ind, cutind


def create_index(arr):
    """
    Create an index of array values like reverse indices.

    arr[index['index'][index['lo'][2]:index['hi'][2]+1]]
    """
    
    narr = size(arr)
    if narr==0:
        raise ValueError('arr has no elements')
    si = np.argsort(arr)
    sarr = np.array(arr)[si]
    brklo, = np.where(sarr != np.roll(sarr,1))
    nbrk = len(brklo)
    if nbrk>0:
        brkhi = np.hstack((brklo[1:nbrk]-1,narr-1))
        num = brkhi-brklo+1
        index = {'index':np.atleast_1d(si),'value':np.atleast_1d(sarr[brklo]),
                 'num':np.atleast_1d(num),'lo':np.atleast_1d(brklo),'hi':np.atleast_1d(brkhi)}
    else:
        index = {'index':np.atleast_1d(si),'value':np.atleast_1d(arr[0]),
                 'num':np.atleast_1d(narr),'lo':np.atleast_1d(0),'hi':np.atleast_1d(narr-1)}

    return index

def duplicates(mylist):
    """
    Find duplicates in a list
    """
    mylist = list(mylist)
    dup = {x for x in mylist if mylist.count(x) > 1}
    dup = list(dup)
    return dup
    

def match(a,b,epsilon=0):
    """
    Routine to match values in two vectors.
    
    CALLING SEQUENCE:
        match, a, b, suba, subb, [ COUNT =, /SORT, EPSILON =  ]
  
    INPUTS:
        a,b - two vectors to match elements, numeric or string data types
    
    OUTPUTS:
      suba - subscripts of elements in vector a with a match
                  in vector b
      subb - subscripts of the positions of the elements in
                  vector b with matchs in vector a.
  
          suba and subb are ordered such that a[suba] equals b[subb]
          suba and subb are set to !NULL if there are no matches (or set to -1
                if prior to IDL Version 8.0)
  
    OPTIONAL INPUT KEYWORD:
          /SORT - By default, MATCH uses two different algorithm: (1) the
                  /REVERSE_INDICES keyword to HISTOGRAM is used for integer data,
                  while (2) a sorting algorithm is used for non-integer data.  The
                  histogram algorithm is usually faster, except when the input
                  vectors are sparse and contain very large numbers, possibly
                  causing memory problems.   Use the /SORT keyword to always use
                  the sort algorithm.
          epsilon - if values are within epsilon, they are considered equal. Used only
                  only for non-integer matching.  Note that input vectors should
                  be unique to within epsilon to provide one-to-one mapping.
                  Default=0.
   
    OPTIONAL KEYWORD OUTPUT:
          COUNT - set to the number of matches, integer scalar
   
    SIDE EFFECTS:
          The obsolete system variable !ERR is set to the number of matches;
          however, the use !ERR is deprecated in favor of the COUNT keyword
   
    RESTRICTIONS:
          The vectors a and b should not have duplicate values within them.
          You can use rem_dup function to remove duplicate values
          in a vector
   
    EXAMPLE:
          If a = [3,5,7,9,11]   & b = [5,6,7,8,9,10]
          then
                  IDL> match, a, b, suba, subb, COUNT = count
   
          will give suba = [1,2,3], subb = [0,2,4],  COUNT = 3
          and       a[suba] = b[subb] = [5,7,9]
   
   
    METHOD:
          For non-integer data types, the two input vectors are combined and
          sorted and the consecutive equal elements are identified.   For integer
          data types, the /REVERSE_INDICES keyword to HISTOGRAM of each array
          is used to identify where the two arrays have elements in common.

    HISTORY:
         D. Lindler  Mar. 1986.
         Fixed "indgen" call for very large arrays   W. Landsman  Sep 1991
         Added COUNT keyword    W. Landsman   Sep. 1992
         Fixed case where single element array supplied   W. Landsman Aug 95
         Use a HISTOGRAM algorithm for integer vector inputs for improved
               performance                W. Landsman         March 2000
         Work again for strings           W. Landsman         April 2000
         Use size(/type)                  W. Landsman         December 2002
         Work for scalar integer input    W. Landsman         June 2003
         Assume since V5.4, use COMPLEMENT to WHERE() W. Landsman Apr 2006
         Added epsilon keyword            Kim Tolbert         March 14, 2008
         Fix bug with Histogram method with all negative values W. Landsman/
         R. Gutermuth, return !NULL for no matches  November 2017
         Added epsilon test in na=1||nb=1 section (missed that when added
               epsilon in 2008)           Kim Tolbert         July 10, 2018
  
    """

    #da = size(a,/type) & db =size(b,/type)
    #if keyword_set(sort) then hist = 0b else $
    #  hist = (( da LE 3 ) || (da GE 12)) &&  ((db LE 3) || (db GE 12 ))

    na = size(a)             # number of elements in a
    nb = size(b)             # number of elements in b
    
    # Check for a single element array
    if (na==1) | (nb==1):
        if (nb>1):
            if epsilon==0.0:
                subb, = np.where(b==a)
                nw = len(subb)
            else:
                subb, = np.where(np.abs(b-a) < epsilon)
                nw = len(subb)
            if (nw>0):
                suba = np.zeros(nw,int)
            else:
                suba = np.array([])
        else:
            if epsilon==0.0:
                suba, = np.where(a==b)
                nw = len(suba)
            else:
                suba, = np.where(np.abs(a-b) < epsilon)
                nw = len(suba)
            if (nw>0):
                subb = np.zeros(nw,int)
            else:
                subb = np.array([])
        count = nw
        return suba,subb

    # Conver to numpy.chararray if either of them are strings
    a1 = first_el(a)
    b1 = first_el(b)
    if isinstance(a,np.chararray) | isinstance(a1,str) | \
       isinstance(b,np.chararray) | isinstance(b1,str):
        atemp = np.char.array(a)
        btemp = np.char.array(b)
        # Use the dtype with the largest number of characters
        if atemp.dtype > btemp.dtype:
            dtype = atemp.dtype
        else:
            dtype = btemp.dtype
        c = np.zeros(na+nb,dtype=dtype)
        c[0:na] = atemp
        c[na:] = btemp
        c = np.char.array(c)  # convert to np.chararray, removes trailing strings
        del atemp, btemp
        #c = np.hstack((np.char.array(a),np.char.array(b)))           # combined list of a and b
    else:
        c = np.hstack((np.array(a),np.array(b)))                     # combined list of a and b
    ind = np.hstack((np.arange(na),np.arange(nb)))               # combined list of indices
    vec = np.hstack((np.zeros(na,bool),np.zeros(nb,bool)+True))  # flag of which vector in  combined
    #list   False - a   True - b

    # sort combined list
    sub = np.argsort(c)
    c = c[sub]
    ind = ind[sub]
    vec = vec[sub]

    # find duplicates in sorted combined list
    n = na + nb                            #t otal elements in c
    if epsilon == 0.0:
      firstdup, = np.where( (c == np.roll(c,-1)) & (vec != np.roll(vec,-1)) )
      count = len(firstdup)
    else:
      firstdup, = np.where( (np.abs(c - np.roll(c,-1)) < epsilon) & (vec != np.roll(vec,-1)) )
      count = len(firstdup)

    if count==0:               # any found?
      suba = np.array([])
      subb = np.array([])
      return suba,subb

    dup = np.zeros( count*2, int )        # both duplicate values
    even = np.arange( len(firstdup))*2     # Changed to LINDGEN 6-Sep-1991
    dup[even] = firstdup
    dup[even+1] = firstdup+1
    ind = ind[dup]                         # indices of duplicates
    vec = vec[dup]                         # vector id of duplicates
    vone, = np.where(vec)
    vzero, = np.where(~vec)
    subb = ind[vone]                       # b subscripts
    suba = ind[vzero]


    # # Integer calculation using histogram.
    # else:
    #
    #     minab = min(a, MAX=maxa) > min(b, MAX=maxb) #Only need intersection of ranges
    #     maxab = maxa < maxb
    #
    #     #If either set is empty, or their ranges don't intersect:
    #     #  result = NULL (which is denoted by integer = -1)
    #     !ERR = -1
    #     if !VERSION.RELEASE GE '8.0' then begin
    #        suba = !NULL
    #        subb = !NULL
    #     endif else begin
    #        suba = -1
    #        subb = -1
    #     endelse
    #     COUNT = 0L
    #     if maxab lt minab then return       #No overlap
    #
    #     ha = histogram([a], MIN=minab, MAX=maxab, reverse_indices=reva)
    #     hb = histogram([b], MIN=minab, MAX=maxab, reverse_indices=revb)
    #
    #     r = where((ha ne 0) and (hb ne 0), count)
    #
    #     if count gt 0 then begin
    #        suba = reva[reva[r]]
    #        subb = revb[revb[r]]
    
    return suba, subb

# Interpolation with extrapolation
def interp(x,y,xout,kind='cubic',bounds_error=False,assume_sorted=True,
           extrapolate=True,exporder=2,fill_value=np.nan):
    """
    Interpolate data using scipy.interpolate.interp1d.  This function allows for
    extrapolation on the ends as well.

    Parameters
    ----------
    x : numpy array
       Array of x-values to fit.
    y : numpy array
       Array of y-values to fit.
    xout : numpy array
       Array or scalar of desired output x-values.
    kind : str, optional
       The order of the interpolation: 'linear', 'quadratic', or 'cubic'.
         Default is 'cubic'.
    bounds_error : boolean, optional
       Throw an exception if the requested values are outside the data range.
         Default is False.
    assume_sorted : boolean, optional
       Assume that the x-values are sorted.  Default is True.
    extrapolate : bool, optional
       Extrapolate requested values outside of the data range.  Default is True.
    exporder : int, optional
       The extrapolation polynomial order.  Default is 2.
    fill_value : float, optional
       The fill value to use for requested points outside of the data range.

    Returns
    -------
    yout : numpy array
       Interpolated y-values.

    Example
    -------

    yout = interp(x,y,xout)

    """

    xo = np.atleast_1d(xout)
    
    # X must be unique and sorted
    if assume_sorted==False:
        u,ui = np.unique(x,return_index=True)
        si = ui[np.argsort(x[ui])]
    else:
        si = np.arange(len(x))
    # Run scipy.interpolate.interp1d
    yout = interp1d(x[si],y[si],kind=kind,bounds_error=bounds_error,
                    fill_value=(fill_value,fill_value),assume_sorted=assume_sorted)(xo)
    # Need to extrapolate
    if ((np.min(xo)<np.min(x)) | (np.max(xo)>np.max(x))) & (extrapolate is True):
        u,ui = np.unique(x,return_index=True)
        si = ui[np.argsort(x[ui])]
        npix = len(x)
        nfit = np.min([np.maximum(exporder+1,1),npix])
        # At the beginning
        if (np.min(xo)<np.min(x)):
            coef1 = poly_fit(x[0:nfit], y[0:nfit], exporder)
            bd1, nbd1 = where(xo < np.min(x))
            yout[bd1] = poly(xo[bd1],coef1)
        # At the end
        if (np.max(xo)>np.max(x)):
            coef2 = poly_fit(x[npix-nfit:], y[npix-nfit:], exporder)
            bd2, nbd2 = where(xo > np.max(x))
            yout[bd2] = poly(xo[bd2],coef2)     
    return yout

def concatenate(a,b=None):
    # Concatenate two or more numpy structured arrays
    # Can input two numpy structured arrays or a list of them
    if (b is None and type(a) is not list) | (b is not None and type(a) is list):
        raise Exception('Must input two numpy structured arrays or a list of them')
        return
    if type(a) is not list: a=list(a)
    if b is not None: a.append(b)

    # Get dtypes for all of the numpy structured arrays
    ncat = size(a)
    dtypearr = []
    ncols = []
    nrows = []
    for a1 in a:
        dtype1 = a1.dtype
        dtypearr.append(dtype1)
        ncols.append(len(dtype1.names))
        nrows.append(len(a1))
    ncols = np.array(ncols)
    nrows = np.array(nrows)
    # Ncols not the same
    if np.min(ncols) != np.max(ncols):
        raise Exception('Number of columns are not the same: min='+str(np.min(ncols))+' max='+str(np.max(ncols)))
    # Checking column names
    colnames = np.zeros((ncat,ncols[0]),dtype=(str,100))
    for i in range(ncat):
        colnames[i,:] = a[i].dtype.names
        if i>0:
            if list(colnames[0,:]) != list(colnames[i,:]):
                raise Exception('Column names are not the same')

    # Make the final dtype
    #   make sure string columns are same length
    dtype_list = []
    for f in a[0].dtype.names:
        if a[0].dtype[f].char == 'S':
            isize = []
            for d1 in dtypearr:
                isize.append(d1[f].itemsize)
            maxsize = np.max(isize)
            dtype_list.append((f,'S'+str(maxsize)))
        else:
            dtype_list.append((f,a[0].dtype[f].str))
    dtype = np.dtype(dtype_list)

    # Create the final structure and load the data
    nlstr = np.sum(nrows)
    lstr = np.zeros(nlstr,dtype=dtype)
    count = 0
    for i in range(ncat):
        a1 = a[i]
        n = len(a1)
        lstr[count:count+n] = a[i]
        count += n
    return lstr

def addcatcols(cat,dt):
    """ Add new columns to an existing numpty structured array catalog."""
    ncat = len(cat)
    odt = cat.dtype

    # Concatenate the dtypes
    dtype_list = []
    for f in cat.dtype.names:
        dtype_list.append((f,cat.dtype[f]))
    #    dtype_list.append((f,cat.dtype[f].str))
    for f in dt.names:
        dtype_list.append((f,dt[f]))
    #    dtype_list.append((f,dt[f].str))
    newdtype = np.dtype(dtype_list)    
    
    # Create the final structure and load the data
    new = np.zeros(ncat,dtype=newdtype)
    for n in cat.dtype.names: new[n] = cat[n]

    return new    
    

def onclick(event):
    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #      ('double' if event.dblclick else 'single', event.button,
    #       event.x, event.y, event.xdata, event.ydata))

    #global ix, iy
    if event.xdata is None:
        global cid
        fig.canvas.mpl_disconnect(cid)
        print('Done.  Coordinates are in global "coords" list')
        return
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy))

    global coords
    try:
        dum = len(coords)
    except:
        coords = []
    coords.append((ix, iy))

    return

def add_elements(cat,num=10000):
    """ Add more elements to a catalog"""
    ncat = len(cat)
    old = cat.copy()
    num = gt(num,ncat)
    cat = np.zeros(ncat+num,dtype=old.dtype)
    cat[0:ncat] = old
    del old
    return cat 


def ellipsecoords(pars,npoints=100):
    """ Create coordinates of an ellipse."""
    # [x,y,asemi,bsemi,theta]
    # copied from ellipsecoords.pro
    xc = pars[0]
    yc = pars[1]
    asemi = pars[2]
    bsemi = pars[3]
    pos_ang = pars[4]
    phi = 2*np.pi*(np.arange(npoints)/(npoints-1))   # Divide circle into Npoints
    ang = np.deg2rad(pos_ang)                             # Position angle in radians
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    x =  asemi*np.cos(phi)                              # Parameterized equation of ellipse
    y =  bsemi*np.sin(phi)
    xprime = xc + x*cosang - y*sinang               # Rotate to desired position angle
    yprime = yc + x*sinang + y*cosang
    return xprime, yprime

def closest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx



def sexig2ten(inp):
    """ Convert sexigesimal to decimal."""

    inp = np.atleast_1d(inp)
    ninp = size(inp)
    tenarr = np.zeros(ninp,np.float64)
    
    # array input
    for i in range(ninp):
        rgt = inp[i].strip()
        arr = rgt.split(':')

        # Still only one element.  Maybe try spaces
        if len(arr)==1:
            arr = rgt.split()

        # Multiple elements
        if len(arr)>1:
            ten = np.abs(np.float64(arr[0]))
            ten += np.float64(arr[1])/60.0
            if len(arr)>2:
                ten += np.float64(arr[2])/3600.0 
            if np.float64(arr[0])<0:  # negative
                ten *= -1
            tenarr[i] = ten
                
        # Only one element, assume it's a number
        else:
            tenarr[i] = np.float64(arr)


    if ninp==1:
        tenarr = tenarr[0]

    return tenarr


def ladmdfunc(b, x, y, eps=1e-7):
    a = np.median(y - b*x)
    d = y - (b * x + a)
    absdev = np.sum(np.abs(d))
    nz, = np.where(y != 0.0)
    nzcount = len(nz)
    if nzcount != 0:
        d[nz] = d[nz] / np.abs(y[nz]) #Normalize
    nz, = np.where(np.abs(d) > eps)
    nzcount = len(nz)
    if nzcount != 0:        #Sign fcn, +1 for d > 0, -1 for d < 0, else 0.
        return np.sum(x[nz] * ((d[nz] > 0)*1 - (d[nz] < 0)*1)), a, absdev
    else:
        return 0.0, a, absdev

def ladfit(xin, yin):
    """
    LADFIT

    This function fits the paired data {X(i), Y(i)} to the linear model,
    y = A + Bx, using a "robust" least absolute deviation method. The
    result is a two-element vector containing the model parameters, A
    and B.


    Result = LADFIT(X, Y)

    Parameters
    ----------
       X:    An n-element vector of type integer, float or double.

       Y:    An n-element vector of type integer, float or double.

    EXAMPLE:
       Define two n-element vectors of paired data.
         x = [-3.20, 4.49, -1.66, 0.64, -2.43, -0.89, -0.12, 1.41, $
               2.95, 2.18,  3.72, 5.26]
         y = [-7.14, -1.30, -4.26, -1.90, -6.19, -3.98, -2.87, -1.66, $
              -0.78, -2.61,  0.31,  1.74]
       Compute the model parameters, A and B.
         result = ladfit(x, y, absdev = absdev)
       The result should be the two-element vector:
         [-3.15301, 0.930440]
       The keyword parameter should be returned as:
         absdev = 0.636851

    REFERENCE:
       Numerical Recipes, The Art of Scientific Computing (Second Edition)
       Cambridge University Press, 2nd Edition.
       ISBN 0-521-43108-5
    This is adapted from the routine MEDFIT described in:
    Fitting a Line by Minimizing Absolute Deviation, Page 703.

    MODIFICATION HISTORY:
      Written by:  GGS, RSI, September 1994
      Modified:    GGS, RSI, July 1995
                    Corrected an infinite loop condition that occured when
                    the X input parameter contained mostly negative data.
      Modified:    GGS, RSI, October 1996
                    If least-absolute-deviation convergence condition is not
                    satisfied, the algorithm switches to a chi-squared model.
                    Modified keyword checking and use of double precision.
      Modified:    GGS, RSI, November 1996
                    Fixed an error in the computation of the median with
                    even-length input data. See EVEN keyword to MEDIAN.
      Modified:    DMS, RSI, June 1997
         Simplified logic, remove SIGN and MDfunc2 functions.
      Modified:    RJF, RSI, Jan 1999
         Fixed the variance computation by adding some double
         conversions.  This prevents the function from generating
         NaNs on some specific datasets (bug 11680).
      Modified: CT, RSI, July 2002: Convert inputs to float or double.
            Change constants to double precision if necessary.
      CT, March 2004: Check for quick return if we found solution.
      March 2021, DLN translated from IDL to python 
    """
    
    nx = len(xin)
  
    if nx != len(yin):
        raise ValuError("X and Y must be vectors of equal length.")

    x = np.float64(xin)
    y = np.float64(yin)

    sx = np.sum(x)
    sy = np.sum(y)

    #  the variance computation is sensitive to roundoff, so we do this
    #  math in DP
    sxy = np.sum(x*y)
    sxx = np.sum(x*x)
    delx = nx * sxx - sx**2

    if (delx == 0.0):         #All X's are the same
        result = [np.median(y), 0.0] #Bisect the range w/ a flat line
        absdev = np.sum(np.abs(y-np.median(y)))/nx
        return np.array(result), absdev

    aa = (sxx * sy - sx * sxy) / delx #Least squares solution y = x * aa + bb
    bb = (nx * sxy - sx * sy) / delx
    chisqr = np.sum((y - (aa + bb*x))**2)
    sigb = np.sqrt(chisqr / delx)     #Standard deviation
    
    b1 = bb
    eps = 1e-7
    f1,aa,absdev = ladmdfunc(b1, x, y, eps=eps)

    # Quick return. The initial least squares gradient is the LAD solution.
    if (f1 == 0.):
        bb = b1
        absdev = absdev / nx
        return np.array([aa, bb],float), absdev

    #delb = ((f1 >= 0) ? 3.0 : -3.0) * sigb
    delb = 3.0*sigb if (f1 >= 0) else -3.0*sigb
    
    b2 = b1 + delb
    f2,aa,absdev = ladmdfunc(b2, x, y, eps=eps)

    while (f1*f2 > 0):     #Bracket the zero of the function
        b1 = b2
        f1 = f2
        b2 = b1 + delb
        f2,aa,absdev = ladmdfunc(b2, x, y, eps=eps)


    # In case we finish early.
    bb = b2
    f = f2

    #Narrow tolerance to refine 0 of fcn.
    sigb = 0.01 * sigb

    while ((np.abs(b2-b1) > sigb) and (f != 0)): #bisection of interval b1,b2.
        bb = 0.5 * (b1 + b2)
        if (bb == b1 or bb == b2):
            break
        f,aa,absdev = ladmdfunc(bb, x, y, eps=eps)
        if (f*f1 >= 0):
            f1 = f
            b1 = bb
        else:
            f2 = f
            b2 = bb

    absdev = absdev / nx

    return np.array([aa, bb],float), absdev


def fread(line,fmt):
    """
    Read the values in a string into variables using a format string.
    (1X, A8, 4I5, F9.3, F15.3, 2F9.1)
    """
    # Transform the format string into an array
    if fmt.startswith('('):
        fmt = fmt[1:]
    if fmt.endswith(')'):
        fmt = fmt[:-1]
    fmtarr = fmt.split(',')
    # Expand repeat values, e.g. 2I5 -> I5,I5
    fmtlist = []
    for i in range(len(fmtarr)):
        fmt1 = fmtarr[i].strip()
        # X format starts with a number
        if fmt1.find('X')>-1:
            fmtlist.append(fmt1)
            continue
        # Repeats
        if fmt1[0].isnumeric():
            ind, = np.where(np.char.array(list(fmt1)).isalpha()==True)
            ind = ind[0]
            num = int(fmt1[0:ind])
            
            fmtlist += list(np.repeat(fmt1[ind:],num))
        else:
            fmtlist.append(fmt1)
    # Start the output tuple
    out = ()
    count = 0
    for i in range(len(fmtlist)):
        fmt1 = fmtlist[i]
        # Ignore X formats
        if fmt1.find('X')==-1:
            if fmt1[0]=='A':
                num = int(fmt1[1:])
                out1 = line[count:count+num]
            elif fmt1[0]=='I':
                num = int(fmt1[1:])
                out1 = int(line[count:count+num])
            else:
                ind = fmt1.find('.')
                num = int(fmt1[1:ind])
                out1 = float(line[count:count+num])
            out = out + (out1,)
            count += num
            
        # X formats, increment the counter
        else:
            ind = fmt1.find('X')
            num = int(fmt1[0:ind])
            count += num

    return out

def randf(low,high,num):
    """ Pick random floats between low and high (inclusive)."""
    return np.random.rand(num)*(high-low)+low
    
def isnumber(s):
    """ Returns True if string is a number or float. """
    # Handles negatives and scientific notation as well
    # use isdigit() for integers.
    try:
        float(s)
        return True
    except:
        return False

def bootstrap(data,statistic,niter=100,indexargs=False,args=None,kwargs=None):
    """
    Perform bootstrap statistics.
    
    Parameters
    ----------
    data : list or numpy array
       The data on which to perform the statistic.
    statistic : str or callable
       Name of statistical function ('sum','mean','median', etc.) or callable
         function to compute statistic.
    niter : int, optional
       Number of bootstrap iterationas.  Default is 100.
    indexargs : boolean, optional
       Apply the random ordering to the args/kwargs that are arrays and have
          the same size as data.  Default is False.
    args : tuple, optional
       Extra positional arguments to pass to `statistic`.
    kwargs : dictionary, optional
       Extra keyword arguments to pass to `statistic`.

    Returns
    -------
    sigma : numpy array
      The 1-sigma bootstrap values for each output that
        statistic returns.

    Example
    -------

    sigma = bootstrap(xdata,ydata,statistic)

    """

    # String name
    if type(statistic) is str:
        if statistic=='sum' or statistic=='np.sum' or statistic=='numpy.sum':
            statistic = np.sum
        if statistic=='nansum' or statistic=='np.nansum' or statistic=='numpy.nansum':
            statistic = np.nansum
        elif statistic=='max' or statistic=='np.max' or statistic=='numpy.max':
            statistic = np.max
        elif statistic=='nanmax' or statistic=='np.nanmax' or statistic=='numpy.nanmax':
            statistic = np.nanmax
        elif statistic=='min' or statistic=='np.min' or statistic=='numpy.min':
            statistic = np.min
        elif statistic=='nanmin' or statistic=='np.nanmin' or statistic=='numpy.nanmin':
            statistic = np.nanmin                        
        elif statistic=='mean' or statistic=='np.mean' or statistic=='numpy.mean':
            statistic = np.mean
        elif statistic=='nanmean' or statistic=='np.nanmean' or statistic=='numpy.nanmean':
            statistic = np.nanmean            
        elif statistic=='median' or statistic=='np.median' or statistic=='numpy.median':
            statistic = np.median
        elif statistic=='nanmedian' or statistic=='np.nanmedian' or statistic=='numpy.nanmedian':
            statistic = np.nanmedian            
        elif statistic=='std' or statistic=='np.std' or statistic=='numpy.std':
            statistic = np.std
        elif statistic=='nanstd' or statistic=='np.nanstd' or statistic=='numpy.nanstd':
            statistic = np.nanstd            
        elif statistic=='mad':
            statistic = np.mad
        else:
            raise ValueError('Statistic '+statistic+' not supported')
    ndata = np.array(data).size
    # Get number of outputs    
    vals = (data,)
    if args is not None: vals = (vals,args)
    if kwargs is not None: vals = (vals,kwargs)
    out0 = statistic(*vals)
    nout = np.array(out0).size

    # Initialize seed using data so reproducible
    if type(data[0]) is not int:
        seed = int(np.abs(data[0])*1e5)
    else:
        seed = data[0]
    rng = np.random.default_rng(seed)
    # Bootstrap loop
    ind = np.arange(ndata).astype(int)
    btout = np.zeros((niter,nout),float)
    for i in range(niter):
        # Randomly pick elements with replacement
        rndind = rng.choice(ind,ndata,replace=True)
        data1 = data[rndind]   # apply to data array
        vals = (data1,)
        # Apply the random ordering to the input arguments if
        #  they are arrays or lists
        if indexargs:
            # Apply random ordering to args
            if args is not None:
                if type(args) is not list and type(args) is not tuple:
                    args = (args,)
                nargs = len(args)
                for j in range(nargs):
                    # Array and same size as data, apply the random ordering
                    if np.array(args[j]).size > 1 and np.array(args[j]).size==ndata:
                        vals = (vals,args[j][rndind])
                    else:
                        vals = (vals,args[j])
            # Apply random ordering to kwargs
            if kwargs is not None:
                nkwargs = len(kwargs)
                kwargs1 = kwargs.copy()  # start new dictionary
                for k in kwargs1.keys():
                    # Array and same size as data, apply the random ordering                    
                    if np.array(kwargs[k]).size > 1 and np.array(kwargs[k]).size==ndata:
                        kwargs1[k] = kwargs[k][rndind]
                vals = (vals,kwargs1)
        # Do not apply random ordering to args/kwargs
        else:
            if args is not None: vals = (vals,args)
            if kwargs is not None: vals = (vals,kwargs)
        out1 = statistic(*vals)
        btout[i,:] = out1
    # Calculate robust standard deviation for each coefficient
    sigma = np.zeros(nout,float)
    for i in range(nout):
        med = np.median(btout[:,i])
        sigma[i] = mad(btout[:,i]-med,zero=True)
    if nout==1:
        sigma = sigma[0]
        
    return sigma

def bspline(x,y,w=None,nquantiles=10,nord=3,knots=None,extrapolate=False):
    """
    Fit a B-Spline to data.  This is a thin wrapper for scipy.interpolate.splrep
    and scipy.interpolate.BSpline.

    Parameters
    ----------
    x : numpy array
       Array of x-values to fit.
    y : numpy array
       Array of y-values to fit.
    w : numpy array, optional
       Array of weights for x/y.
    nquantiles : int, optional
       Number of quantiles to use to create the knots. Default is 10.
    nord : int, optional
       The degree of the spline fit.  Default is 3.
    knots : numpy array, optional
       The knot points for the BSpline.
    extrapolate : bool, optional
       Should the B-Spline be allowed to extrapolate.  Default is False.

    Returns
    -------

    spline : BSpline object
       BSpline object that can be used to interpolate


    Example
    -------

    spline = bspline(x,y,knots=knots,nord=3)

    """

    # X must be unique and sorted
    u,ui = np.unique(x,return_index=True)
    si = ui[np.argsort(x[ui])]
    # Knots
    if knots is None and nquantiles is None:
        raise ValueError('nquantiles or knots must be input')
    if knots is None:
        qs = np.linspace(0, 1, nquantiles+2)[1:-1]
        knots = np.quantile(x[si], qs)
    # Perform the fitting
    if w is not None:
        tck = splrep(x[si],y[si],w[si], t=knots, k=nord)
    else:
        tck = splrep(x[si],y[si], t=knots, k=nord)
    # Generate the BSpline object
    spline = BSpline(*tck, extrapolate=extrapolate)

    return spline


def help(*args,verbose=False):
    """ Like IDL help"""
    
    if len(args)==0:
        local = locals().keys()
        globl = globals().keys()
        # remove _ and __ ones
        globl = [g for g in globl if g[0]!='_']
        local = [l for l in local if l[0]!='_']
        allvars = globl+local
        for i in range(len(allvars)):
            print(allvars[i],exec('type('+allvars[i]+')'))
        
    # Get variables names input
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    varnames = code[code.find('(')+1:-1].split(',')
    varnames = [v.strip() for v in varnames]
    
    # Loop over arguments
    for i in range(len(args)):
        a = args[i]
        atype = type(a)
        try:
            ndim = a.ndim
        except:
            ndim = 0
        try:
            shape = a.shape
        except:
            shape = None
        try:
            na = len(a)
        except:
            na = 0
            
        # Is this a table?
        tab = False
        if atype == Table:
            colnames = a.colnames
            dtype = a.dtype
            tab = True
        elif atype == np.ndarray and a.dtype.names is not None:
            colnames = a.columns
            dtype = a.dtype
            tab = True
        elif atype == pd.DataFrame:
            colnames = a.columns
            dtype = a.dtype
            tab = True

        if tab==False:
            fmt = '%-16s %-22s %-10s'
            if atype == np.ndarray and a.dtype.names is None:
                atype = '<np.ndarray '+str(a.dtype)+'>'
            if ndim==0:
                data = (varnames[i],atype,'scalar')
            elif shape is not None:
                if len(shape)==1:
                    sshape = '['+str(shape[0])+']'
                else:
                    sshape = '['+','.join(np.char.array(shape).astype(str))+']'
                data = (varnames[i],atype,sshape)
            else:
                data = (varnames[i],atype,'['+str(na)+']')            
        else:
            fmt = '%-16s %-22s %-10s'
            data = (varnames[i],type(a),'['+str(na)+']')
                
        # Short output
        if verbose==False:
            fmt = '%-16s %-22s %-10s'
            if len(varnames[i])>16:
                print(varnames[i])
                print(fmt % ('',data[1],data[2]))
            else:
                print(fmt % data)

        # Verbose output of table
        else:
            if tab==False:
                fmt = '%-16s %-22s %-10s'
                if len(varnames[i])>16:
                    print(varnames[i])
                    print(fmt % ('',data[1],data[2]))
                    print(str(a))
                else:
                    print(fmt % data)
                    print(str(a))
                    
            else:
                print(str(type(a))+', '+str(len(colnames))+' columns, ['+str(na)+']')
                fmt = '%-20s %-10s %-20s'                    
                for j in range(len(colnames)):
                    col = a[colnames[j]][0]
                    if size(col)>1 and type(col) != str:
                        data = (colnames[j],dtype[j].base,'Array['+str(size(col))+']')           
                    else:
                        data = (colnames[j],dtype[j].base,str(col))
                    if len(colnames[j])>16:
                        print(colnames[j])
                        print(fmt % ('',data[1],data[2]))
                    else:
                        print(fmt % data)


def modfits(filename,data=None,header=None,wcs=None,update=False,
            extnum=None,extname=None):
    """
    Update data or header in an existing FITS file

    Parameters
    ----------
    filename : str
      Filename of the FITS file to modify.
    data : numpy array, optional
      New data to stick in the FITS file.
    header : Header object, optional
      New header to add to FITS.  Default is to replace the existing header.
        If update=True, then the existing header is updated.
    wcs : WCS object, optional
      WCS object.  The header is updated with this WCS.
    update : boolean, optional
      Update header rather than replace.  Default is False.
    extnum : int, optional
      Extension number.  Default is 0.
    extname : str, optional
      Extension name.

    Returns
    -------
    FITS file is updated.

    Example
    -------

    modfits('filename.fits',header=newhead)

    """

    if os.path.exists(filename)==False:
        raise ValueError(filename+" NOT FOUND")
    if data is None and header is None:
        raise ValueError('data or header must not be None')

    hdu = fits.open(filename)

    if extnum is not None:
        if int(extnum) > len(hdu)-1:
            raise ValueError('EXTNUM '+str(extnum)+' error. Only '+str(len(hdu))+' HDUs')
    
    # EXTNAME
    if extname is not None:
        extnames = [h.header.get('EXTNAME') for h in hdu]
        if extname not in extnames and extname.upper() not in extnames:
            raise ValueError('EXTNAME '+extname+' not found in '+filename)
        extnum, = np.where(np.char.array(extnames).astype(str).lower() == extname.lower())
        extnum = extnum[0]
    # Use first extension by default
    if extnum is None: extnum = 0

    # Update data
    if data is not None:
        hdu[extnum].data = data
    # Update header
    if header is not None:
        if update:
            hdu[extnum].header.extend(header,update=True)
        else:
            hdu[extnum].header = header
    # Add WCS
    if wcs is not None:
        hdu[extnum].header.extend(wcs.to_header(),update=True)
    
    hdu.close()
    

def pickle(filename,data):
    """
    Pickle data.
    """
    with open(filename,'wb') as f:
        pickl.dump(data,f)

def unpickle(filename):
    """
    Load pickled data.
    """
    with open(filename,'rb') as f:
        data = pickl.load(f)
    return data

def pwd():
    """ Return the current working directory."""

    return os.path.abspath(os.curdir)

def poly2d_wrap(x,*args):
    """ thin wrapper for curve_fit"""
    xx = x[0]
    yy = x[1]
    return poly2d(xx,yy,*args)

def poly2d(x,y,*args):
    """ 2D polynomial surface"""

    p = args
    np = len(p)
    if np==0:
        a = p[0]
    elif np==3:
        a = p[0] + p[1]*x + p[2]*y
    elif np==4:
        a = p[0] + p[1]*x + p[2]*x*y + p[3]*y
    elif np==6:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x*y + p[4]*y + p[5]*y**2
    elif np==8:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x*y + p[4]*(x**2)*y + p[5]*x*y**2 + p[6]*y + p[7]*y**2
    elif np==11:
        a = p[0] + p[1]*x + p[2]*x**2.0 + p[3]*x**3.0 + p[4]*x*y + p[5]*(x**2.0)*y + \
            p[6]*x*y**2.0 + p[7]*(x**2.0)*(y**2.0) + p[8]*y + p[9]*y**2.0 + p[10]*y**3.0
    elif np==15:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*y + p[6]*x*y + \
            p[7]*(x**2)*y + p[8]*(x**3)*y + p[9]*y**2 + p[10]*x*y**2 + p[11]*(x**2)*y**2 + \
            p[12]*y**3 + p[13]*x*y**3 + p[14]*y**4
    elif np==21:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*x**5 + p[6]*y + p[7]*x*y + \
            p[8]*(x**2)*y + p[9]*(x**3)*y + p[10]*(x**4)*y + p[11]*y**2 + p[12]*x*y**2 + \
            p[13]*(x**2)*y**2 + p[14]*(x**3)*y**2 + p[15]*y**3 + p[16]*x*y**3 + p[17]*(x**2)*y**3 + \
            p[18]*y**4 + p[19]*x*y**4 + p[20]*y**5
    else:
        raise Exception('Only 3, 4, 6, 8, 11 amd 15 parameters supported')

    return a

def gradient_n(arr, n, d=1, axis=0):
    """Differentiate np.ndarray n times.

    Similar to np.diff, but additional support of pixel distance d
    and padding of the result to the same shape as arr.

    If n is even: np.diff is applied and the result is zero-padded
    If n is odd: 
        np.diff is applied n-1 times and zero-padded.
        Then gradient is applied. This ensures the right output shape.
    https://stackoverflow.com/questions/23419193/second-order-gradient-in-numpy
    """
    n2 = int((n // 2) * 2)
    diff = arr

    if n2 > 0:
        a0 = max(0, axis)
        a1 = max(0, arr.ndim-axis-1)
        diff = np.diff(arr, n2, axis=axis) / d**n2
        diff = np.pad(diff, tuple([(0,0)]*a0 + [(1,1)] +[(0,0)]*a1),
                      'constant', constant_values=0)

    if n > n2:
        assert n-n2 == 1, 'n={:f}, n2={:f}'.format(n, n2)
        diff = np.gradient(diff, d, axis=axis)

    return diff

def splice(a,b,axis=0):
    """ Splice/interleave two arrays."""
    # Get new shape
    newshape = list(a.shape)
    newshape[axis] = a.shape[axis] + b.shape[axis]
    # Create new array
    new = np.zeros(newshape,a.dtype)
    # Create slice list
    slc = []
    for i in range(a.ndim):
        slc.append(slice(0,a.shape[i]))
    # Add first array
    slc[axis] = slice(0,newshape[axis],2)
    new[tuple(slc)] = a
    slc[axis] = slice(1,newshape[axis],2)    
    new[tuple(slc)] = b    
    return new

def roll(a, shift, axis=0, wrapvalue=np.nan):
    """
    This is like numpy.roll() but masks the wrapping elements.
    https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.  Default is axis=0.
    wrapvalue : int or float, optional
        The value to insert for wrapped elements.  Default is numpy.nan
        for floats and -999999 for integers.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    """

    # Cannot use NaN for integers
    if np.issubdtype(np.array(a).dtype,np.integer) and np.isnan(wrapvalue):
        wrapvalue = -999999
    
    # Use the regular np.roll(), then mask the values
    res = np.roll(a,shift,axis=axis)
    
    # Shifts along ultiple axes
    shft = np.atleast_1d(shift)
    axs = np.atleast_1d(axis)
    nshift = np.array(shift).size
    naxis = np.array(axis).size
    nroll = np.max([nshift,naxis])
    if nshift>1 or naxis>1:
        if nshift>1 and naxis==1:
            raise ValueError('If shift is an array/tuple, then axis must be an array/tuple of the same size')
        elif nshift==1 and naxis>1:
            shft = np.zeros(naxis,int)+shift

    # By default np.roll() uses axis=None which shifts all of the
    # values in across axes.
            
    # Loop over each shifting axis and mask
    for i in range(nroll):
        shft1 = shft[i]
        ax1 = axs[i]
        # Create slice list
        slc = []
        for j in range(a.ndim):
            slc.append(slice(0,a.shape[j]))
        if shft1 >= 0:
            slc[ax1] = slice(0,shft1)
            res[tuple(slc)] = wrapvalue
        else: 
            slc[ax1] = slice(shft1,a.shape[ax1])           
            res[tuple(slc)] = wrapvalue
            
    return res

def tail(filename,nlines=10,verbose=False):
    """ https://gist.github.com/amitsaha/5990310 """

    bufsize = 8192
    fsize = os.stat(filename).st_size

    niter = 0
    with open(filename) as f:
        if bufsize > fsize:
            bufsize = fsize-1
        data = []
        while True:
            niter +=1
            f.seek(fsize-bufsize*niter)
            data.extend(f.readlines())
            if len(data) >= nlines or f.tell() == 0:
                if verbose:
                    print(''.join(data[-nlines:]))
                break
    out = data[-nlines:]
    return out

def isdefined(myvariable):
    """
    Check if a variable has been defined yet or not.
    """
    frame = inspect.currentframe()
    res = False
    try:
        if myvariable in frame.f_back.f_locals or myvariable in frame.f_back.f_globals:
            res = True
    finally:
        del frame
    return res

def nanmedfilt(x,size,mode='reflect'):
    """
    Median filter than handles NaNs.
    """
    return generic_filter(x, np.nanmedian, size=size)

def roots(x,y=None):
    """ Find roots, i.e. where array should be zero."""
    if y is None:
        yy = x
        xx = np.arange(len(x))
    else:
        si = np.argsort(x)
        xx = x[si]
        yy = y[si]        
    rootind = np.array([],float)
    gddwn, = np.where((yy[:-1]>=0) & (yy[1:]<0))
    for i in range(len(gddwn)):
        coef = (yy[gddwn[i]+1]-yy[gddwn[i]]) / (xx[gddwn[i]+1]-xx[gddwn[i]])
        rt = -yy[gddwn[i]]/coef + xx[gddwn[i]]
        rootind = np.append(rootind,rt)
    gdup, = np.where((yy[:-1]<0) & (yy[1:]>=0))
    for i in range(len(gdup)):
        coef = (yy[gdup[i]+1]-yy[gdup[i]]) / (xx[gdup[i]+1]-xx[gdup[i]])
        rt = -yy[gdup[i]]/coef + xx[gdup[i]]
        rootind = np.append(rootind,rt)
    # Ones that are exactly zero
    gdzero, = np.where(yy==0)
    if len(gdzero)>0:
        rootind = np.append(rootind,xx[gdzero])
    return np.unique(rootind)

def weighted_median(values, weights):
    """
    Compute the weighted median of an array of values.
    
    This implementation sorts values and computes the cumulative
    sum of the weights. The weighted median is the smallest value for
    which the cumulative sum is greater than or equal to half of the
    total sum of weights.

    Parameters
    ----------
    values : array-like
        List or array of values on which to calculate the weighted median.
    weights : array-like
        List or array of weights corresponding to the values.

    Returns
    -------
    float
        The weighted median of the input values.

    https://gist.github.com/robbibt/c7ec5f0cb3e4e0cee5ed3156bcb666de
    """
    # Convert input values and weights to numpy arrays
    values = np.array(values)
    weights = np.array(weights)
    
    # Get the indices that would sort the array
    sort_indices = np.argsort(values)
    
    # Sort values and weights according to the sorted indices
    values_sorted = values[sort_indices]
    weights_sorted = weights[sort_indices]  

    # Compute the cumulative sum of the sorted weights
    cumsum = weights_sorted.cumsum()
    
    # Calculate the cutoff as half of the total weight sum
    cutoff = weights_sorted.sum() / 2.
    
    # Return the smallest value for which the cumulative sum is greater
    # than or equal to the cutoff
    return values_sorted[cumsum >= cutoff][0]

def md5sum(fname):
    """ Compute md5sum of a file """
    if os.path.exists(fname)==False:
        raise FileNotFoundError(fname)
    md5 = hashlib.md5()
    # handle content in binary form
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b''):
        #while chunk := f.read(4096):
            md5.update(chunk)
    return md5.hexdigest()

def random_distribution(array,n,integer=False):
    """ Generate a random set of points drawn from the input distribution."""
    # Array can be 1-D or higher dimensions
    pdf = np.atleast_1d(array)
    shape = pdf.shape
    ndim = pdf.ndim
    # Unravel any higher dimensions to 1D
    if ndim > 1:
        pdf = pdf.flatten()
    # Make sure all values are non-negative
    pdf = np.maximum(pdf,1e-20)
    #pdf2d = array.copy()
    #pdf1d = np.maximum(result.ravel(),1e-5)
    cdf = np.cumsum(pdf)
    cdf /= np.max(cdf)
    index = np.arange(len(cdf)).astype(float)/(len(cdf)-1)
    rnd = np.random.rand(n)
    newindex = interp1d(cdf,index)(rnd)
    out = newindex * (len(cdf)-1)
    # 1-D
    if ndim == 1:
        if integer:
            out = np.round(out).astype(int)
    # Multi-D
    else:
        outint = np.round(out).astype(int)
        dout = out-outint
        unravel_out = np.unravel_index(outint,shape)
        out = np.zeros((n,ndim),float)
        for i in range(ndim):
            out[:,i] = unravel_out[i]
        # out is currently integers, but want it to be a "real"
        # take the leftover "dout" and distribution it to each dimension
        if integer==False:
            rndvec = np.random.rand(n,ndim)*2-1        # -1 to +1
            totrndvec = np.sqrt(np.sum(rndvec**2,axis=1))
            rndvec /= totrndvec.reshape(-1,1)
            err = np.abs(dout).reshape(-1,1) * rndvec
            out += err
            
    return out
        
